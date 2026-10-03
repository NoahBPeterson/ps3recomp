#!/usr/bin/env python3
"""Multicore conformance: RPCS3 (oracle) vs ps3recomp on SPU thread groups,
MFC DMA, atomics and PPU<->SPU mailboxes/signals.

  1. gen_mc_conform.py writes mc_conform.elf (PPU harness + embedded SPU images)
  2. RPCS3 runs it headless (same oracle config as the PPU suite)
  3. ps3recomp lifts the PPU side; the SPU images run on its SPU interpreter
     as resident, concurrent threads with blocking channels
  4. the MCCONF BEGIN..END transcripts must be identical

usage: run_mc_conform.py --work DIR [--rpcs3 PATH] [--skip-oracle] [--skip-build]
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ppu"))
from run_ppu_conform import DEFAULT_RPCS3, run_oracle, sh  # noqa: E402

# The faithful SPU thread model: interpret images that have no lifted
# registration, on their own host threads, with channel reads that block.
SPU_ENV = {"RD_SPU_INTERP": "1", "RD_SPU_INTERP_ASYNC": "1", "SPU_CH_BLOCK": "1",
           "PS3_VERBOSE": "0"}


def cut(path):
    m = re.search(rb"MCCONF BEGIN.*?MCCONF END", open(path, "rb").read(), re.S)
    return m.group(0).decode("latin-1").splitlines() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rpcs3", default=DEFAULT_RPCS3)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    os.makedirs(work, exist_ok=True)
    elf = os.path.join(work, "mc_conform.elf")
    py = sys.executable

    if sh([py, os.path.join(HERE, "gen_mc_conform.py"), "-o", elf]).returncode:
        sys.exit(2)
    if not a.skip_oracle:
        run_oracle(a.rpcs3, elf, work, a.timeout, marker=b"MCCONF END")

    rec, bld = os.path.join(work, "recompiled"), os.path.join(work, "build")
    if not a.skip_build:
        load = os.path.join(work, "load")
        if sh([py, os.path.join(ROOT, "tools", "ppu_loader.py"), elf, "-o", load],
              stdout=subprocess.DEVNULL).returncode:
            sys.exit(2)
        if sh([py, os.path.join(ROOT, "tools", "ppu_lifter.py"), elf,
               "--functions", os.path.join(load, "mc_conform.functions.json"),
               "--hle-stubs", os.path.join(load, "mc_conform.imports.json"),
               "-o", rec], stdout=subprocess.DEVNULL).returncode:
            sys.exit(2)
        if not os.path.exists(os.path.join(bld, "build.ninja")):
            if sh(["cmake", "-S", os.path.join(ROOT, "templates", "project"), "-B", bld, "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=Release", "-DRECOMP_DIR=" + rec],
                  stdout=subprocess.DEVNULL).returncode:
                sys.exit(2)
        r = sh(["cmake", "--build", bld, "-j", str(a.jobs)], capture_output=True, text=True)
        if r.returncode:
            print("\n".join(l for l in (r.stdout + r.stderr).split("\n") if " error" in l or "Error" in l)[:4000])
            sys.exit(2)

    ours = os.path.join(work, "ours.txt")
    with open(ours, "wb") as fo, open(os.path.join(work, "ours.stderr.txt"), "wb") as fe:
        try:
            subprocess.run([os.path.join(bld, "MyGameRecomp"), elf], stdout=fo, stderr=fe,
                           timeout=a.timeout, env=dict(os.environ, **SPU_ENV), cwd=work)
        except subprocess.TimeoutExpired:
            print("lifted run: TIMEOUT")

    want, got = cut(os.path.join(work, "oracle.txt")), cut(ours)
    if want is None:
        print("oracle transcript has no MCCONF block"); sys.exit(2)
    if got is None:
        print("our transcript has no MCCONF block (see ours.stderr.txt)"); sys.exit(1)
    bad = [(i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
    if len(want) != len(got):
        bad.append((min(len(want), len(got)), "<%d lines>" % len(want), "<%d lines>" % len(got)))
    for i, w, g in bad:
        print("line %d\n  oracle: %s\n  ours:   %s" % (i, w, g))
    print("IDENTICAL (%d lines)" % len(want) if not bad else "%d line(s) differ" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
