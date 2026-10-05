#!/usr/bin/env python3
"""Per-instruction diff of the SPU conformance transcripts.

usage: compare_spu.py MANIFEST ORACLE OURS [--show N] [--json OUT]

Each case line holds the registers its operand fields name (rt, ra, rb, rc
order, 16 bytes each). For every op: cases, mismatching cases, and for the
first N mismatches the instruction word, the inputs, and both results.
"""
import argparse
import json
import re
import sys
from collections import OrderedDict


def case_lines(path):
    return [l.strip() for l in open(path, "rb").read().decode("latin-1").splitlines()
            if re.fullmatch(r"[0-9a-f]{128}", l.strip())]


def qwords(h):
    return [h[i:i + 32] for i in range(0, 128, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("oracle"); ap.add_argument("ours")
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    cases = man["cases"]
    want, got = case_lines(a.oracle), case_lines(a.ours)
    if len(want) != len(cases) or len(got) != len(cases):
        print("line counts: cases %d oracle %d ours %d" % (len(cases), len(want), len(got)))
    per = OrderedDict()
    for i, c in enumerate(cases):
        p = per.setdefault(c["op"], dict(n=0, bad=[]))
        p["n"] += 1
        if i < len(want) and i < len(got) and want[i] != got[i]:
            p["bad"].append(i)
    nbad = 0
    for op, p in per.items():
        if not p["bad"]:
            continue
        nbad += 1
        print("FAIL %-8s %3d/%d cases" % (op, len(p["bad"]), p["n"]))
        for i in p["bad"][:a.show]:
            c = cases[i]
            names = ["rt", "ra", "rb", "rc"][:len(c["regs"])]
            print("     word %08X regs %s" % (c["word"], " ".join("%s=r%d" % (n, r) for n, r in zip(names, c["regs"]))))
            print("       in : %s" % " ".join(q for q in c["inputs"][:len(c["regs"])]))
            w, g = qwords(want[i]), qwords(got[i])
            for k, n in enumerate(names):
                if w[k] != g[k]:
                    print("       %s oracle %s\n       %s ours   %s" % (n, w[k], " " * len(n), g[k]))
    print("ops: %d  failing: %d  cases: %d" % (len(per), nbad, len(cases)))
    if a.json:
        json.dump({op: dict(n=p["n"], bad=len(p["bad"])) for op, p in per.items()}, open(a.json, "w"), indent=1)
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
