#!/usr/bin/env python3
"""SPURS conformance: ps3recomp's HLE libs/spurs against the firmware's real
libsre, run by RPCS3 (LLE is RPCS3's default for libsre).

Each test module t_*.py builds one PPU program (spurs_lib.build) that records
labelled values; RPCS3 runs it headless (the oracle), ps3recomp lifts and runs
it, and the transcripts are compared record by record.

usage: run_spurs_conform.py --work DIR [--only t_core,t_workload] [--skip-oracle]
       [--repeat N] [--rpcs3 PATH]
"""
import argparse
import glob
import importlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ppu"))
from run_ppu_conform import DEFAULT_RPCS3, run_oracle, sh  # noqa: E402
import spurs_lib  # noqa: E402

SPU_ENV = {"RD_SPU_INTERP": "1", "RD_SPU_INTERP_ASYNC": "1", "SPU_CH_BLOCK": "1", "PS3_VERBOSE": "0"}
TAG = spurs_lib.TAG.decode()


def transcript(path):
    t = open(path, "rb").read().decode("latin-1")
    m = re.search(TAG + r" BEGIN.*?" + TAG + r" END", t, re.S)
    if not m:
        return None
    return re.findall(r"@([0-9a-f]+)\s*$", m.group(0), re.M)


def compare(lines, want, got):
    bad = []
    for i, meta in enumerate(lines):
        w = want[i] if i < len(want) else None
        g = got[i] if i < len(got) else None
        if w is None or g is None:
            bad.append((meta["label"], w, g)); continue
        if len(w) != len(g):
            bad.append((meta["label"], w, g)); continue
        wb, gb = bytearray.fromhex(w), bytearray.fromhex(g)
        for off, n in meta["ignore"]:
            for k in range(off, min(off + n, len(wb))):
                wb[k] = gb[k] = 0
        if wb != gb:
            bad.append((meta["label"], w, g))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--rpcs3", default=DEFAULT_RPCS3)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    os.makedirs(work, exist_ok=True)
    tests = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(HERE, "t_*.py")))
    if a.only:
        tests = [t for t in tests if t in a.only.split(",")]
    py = sys.executable
    rec, bld = os.path.join(work, "recompiled"), os.path.join(work, "build")
    summary = []
    for name in tests:
        tdir = os.path.join(work, name)
        os.makedirs(tdir, exist_ok=True)
        elf = os.path.join(tdir, name + ".elf")
        mod = importlib.import_module(name)
        lines = spurs_lib.build(mod, elf)
        json.dump(lines, open(os.path.join(tdir, "manifest.json"), "w"), indent=1)
        print("== %s: %d records" % (name, len(lines)), flush=True)

        if not a.skip_oracle:
            run_oracle(a.rpcs3, elf, tdir, a.timeout, marker=(TAG + " END").encode())
        want = transcript(os.path.join(tdir, "oracle.txt"))
        if want is None:
            print("   oracle produced no transcript (see ~/Library/Caches/rpcs3/RPCS3.log)")
            summary.append((name, "NO ORACLE")); continue

        # lift into the shared recompiled dir and rebuild (runtime objects are reused)
        load = os.path.join(tdir, "load")
        if sh([py, os.path.join(ROOT, "tools", "ppu_loader.py"), elf, "-o", load],
              stdout=subprocess.DEVNULL).returncode:
            summary.append((name, "LOADER FAILED")); continue
        shutil.rmtree(rec, ignore_errors=True)
        if sh([py, os.path.join(ROOT, "tools", "ppu_lifter.py"), elf,
               "--functions", os.path.join(load, name + ".functions.json"),
               "--hle-stubs", os.path.join(load, name + ".imports.json"),
               "-o", rec], stdout=subprocess.DEVNULL).returncode:
            summary.append((name, "LIFT FAILED")); continue
        if not os.path.exists(os.path.join(bld, "build.ninja")):
            sh(["cmake", "-S", os.path.join(ROOT, "templates", "project"), "-B", bld, "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", "-DRECOMP_DIR=" + rec], stdout=subprocess.DEVNULL)
        r = sh(["cmake", "--build", bld, "-j", str(a.jobs)], capture_output=True, text=True)
        if r.returncode:
            print("\n".join(l for l in (r.stdout + r.stderr).split("\n") if " error" in l)[:3000])
            summary.append((name, "BUILD FAILED")); continue

        results = []
        for k in range(a.repeat):
            ours = os.path.join(tdir, "ours%d.txt" % k)
            with open(ours, "wb") as fo, open(os.path.join(tdir, "ours%d.stderr.txt" % k), "wb") as fe:
                try:
                    subprocess.run([os.path.join(bld, "MyGameRecomp"), elf], stdout=fo, stderr=fe,
                                   timeout=a.timeout, env=dict(os.environ, **SPU_ENV), cwd=tdir)
                except subprocess.TimeoutExpired:
                    pass
            got = transcript(ours)
            if got is None:
                results.append(None); continue
            results.append(compare(lines, want, got))

        first_bad = next((r for r in results if r), None)
        if any(r is None for r in results):
            print("   ours: no complete transcript in %d/%d runs (hang or crash; see ours*.stderr.txt)"
                  % (sum(r is None for r in results), len(results)))
        if first_bad:
            for label, w, g in first_bad:
                print("   DIFF %s\n        libsre: %s\n        ours:   %s" % (label, w, g))
        clean = sum(1 for r in results if r == [])
        status = "IDENTICAL" if clean == len(results) else "%d/%d identical" % (clean, len(results))
        print("   %s" % status, flush=True)
        summary.append((name, status if clean == len(results) else
                        "%s, %d differing records" % (status, len(first_bad or []))))

    print("\nsummary:")
    for name, st in summary:
        print("  %-14s %s" % (name, st))
    sys.exit(0 if all(st == "IDENTICAL" for _, st in summary) else 1)


if __name__ == "__main__":
    main()
