#!/usr/bin/env python3
"""Diff two PPU conformance transcripts (RPCS3 oracle vs ps3recomp) field by field.

usage: compare.py <manifest.json> <oracle.txt> <ours.txt> [--show N] [--json out]

A transcript is the TTY output of ppu_conform.elf: "PPUCONF BEGIN n", one hex
line per case (the saved state slot), plus one more hex line (the scratch
buffer) after each memory case, then "PPUCONF END".
"""
import argparse
import json
import struct
import sys
from collections import OrderedDict, defaultdict

S_GPR, S_FPR, S_VR = 0x000, 0x100, 0x200
FIELDS = ([("r%d" % i, S_GPR + 8 * i, 8) for i in range(32)] +
          [("f%d" % i, S_FPR + 8 * i, 8) for i in range(32)] +
          [("v%d" % i, S_VR + 16 * i, 16) for i in range(32)] +
          [("cr", 0x400, 4), ("xer", 0x408, 8), ("lr", 0x410, 8), ("ctr", 0x418, 8),
           ("fpscr", 0x420, 8), ("vscr", 0x430, 16), ("vrsave", 0x440, 4), ("flag", 0x444, 4)])


def read_transcript(path):
    lines = open(path, "rb").read().decode("latin1").split("\n")
    out, on = [], False
    for l in lines:
        l = l.strip()
        if l.startswith("PPUCONF BEGIN"):
            on, out = True, []
            continue
        if l.startswith("PPUCONF END"):
            return out, True
        if on and l and all(c in "0123456789abcdef" for c in l):
            out.append(l)
    return out, False


def split_cases(man, lines):
    res, i = [], 0
    for c in man["cases"]:
        st = bytes.fromhex(lines[i]) if i < len(lines) else None
        i += 1
        sc = None
        if c["mem"]:
            sc = bytes.fromhex(lines[i]) if i < len(lines) else None
            i += 1
        res.append((st, sc))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("oracle"); ap.add_argument("ours")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--json")
    ap.add_argument("--ignore", default="r1", help="comma-separated fields to skip (r1: the harness stack)")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    ignore = set(x for x in a.ignore.split(",") if x)
    lo, end_o = read_transcript(a.oracle)
    lu, end_u = read_transcript(a.ours)
    co, cu = split_cases(man, lo), split_cases(man, lu)
    per_op = OrderedDict()
    examples = defaultdict(list)
    missing = 0
    for c, (so, sco), (su, scu) in zip(man["cases"], co, cu):
        e = per_op.setdefault(c["op"], {"cases": 0, "bad": 0, "fields": defaultdict(int)})
        e["cases"] += 1
        if so is None or su is None:
            missing += 1
            e["bad"] += 1
            e["fields"]["<missing>"] += 1
            continue
        diffs = []
        mask = c.get("mask", {})
        for name, off, n in FIELDS:
            if name in ignore:
                continue
            vo, vu, vi = so[off:off + n], su[off:off + n], bytes.fromhex(c["input"])[off:off + n]
            if name == "vscr":
                # RPCS3 keeps VSCR in BE word 0; the ISA (and ps3recomp) in word 3
                vo, vu, vi = vo[0:4], vu[12:16], vi[12:16]
            if c.get("vscr_dest") is not None and name == "v%d" % c["vscr_dest"]:
                vo = bytes(12) + vo[0:4]          # RPCS3 mfvscr result sits in word 0
            if name in mask:
                m = bytes.fromhex(mask[name])
                vo = bytes(x & y for x, y in zip(vo, m)); vu = bytes(x & y for x, y in zip(vu, m))
            if vo != vu:
                diffs.append((name, vo.hex(), vu.hex(), vi.hex()))
        # A store-conditional may fail spuriously (PowerISA); RPCS3's reservation
        # model sometimes does. Oracle-failed / ours-succeeded is allowed.
        if c.get("resv") and not (so[0x400] & 0x20) and (su[0x400] & 0x20):
            diffs = [x for x in diffs if x[0] != "cr"]
            sco = scu
        if c["mem"] and sco != scu:
            k = next(i for i in range(len(sco)) if sco[i] != scu[i])
            diffs.append(("mem+0x%x" % k, sco[k:k + 16].hex(), scu[k:k + 16].hex(), ""))
        if diffs:
            e["bad"] += 1
            for d in diffs:
                e["fields"][d[0]] += 1
            if len(examples[c["op"]]) < a.show:
                examples[c["op"]].append((c, diffs))
    nbad = sum(1 for e in per_op.values() if e["bad"])
    print("oracle: %d lines%s   ours: %d lines%s" % (len(lo), "" if end_o else " (NO END)",
                                                    len(lu), "" if end_u else " (NO END)"))
    for opn, e in per_op.items():
        tag = "ok  " if not e["bad"] else "FAIL"
        f = "" if not e["bad"] else "  fields: " + ", ".join("%s(%d)" % kv for kv in
                                                              sorted(e["fields"].items(), key=lambda kv: -kv[1])[:6])
        print("%s %-12s %4d cases %4d mismatched%s" % (tag, opn, e["cases"], e["bad"], f))
    for opn, exs in examples.items():
        for c, diffs in exs:
            print("\n-- %s  words=%s  pc=0x%x" % (c["name"], " ".join("%08x" % w for w in c["words"]), c["pc"]))
            for name, o, u, i in diffs[:8]:
                print("   %-8s oracle=%s ours=%s  (in=%s)" % (name, o, u, i))
    print("\nops: %d  failing ops: %d  cases: %d  missing: %d" % (len(per_op), nbad, len(man["cases"]), missing))
    if a.json:
        json.dump({k: {"cases": v["cases"], "bad": v["bad"], "fields": dict(v["fields"])}
                   for k, v in per_op.items()}, open(a.json, "w"), indent=1)
    sys.exit(1 if nbad or missing or not (end_o and end_u) else 0)


if __name__ == "__main__":
    main()
