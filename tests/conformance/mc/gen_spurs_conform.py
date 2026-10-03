#!/usr/bin/env python3
"""Generate the SPURS conformance ELF: a PPU program that drives libsre
(cellSpurs) through real firmware imports, checked by its printed transcript.

On RPCS3 the imports bind to the firmware's own libsre.sprx (LLE is RPCS3's
default for libsre), so the oracle is Sony's SPURS kernel running on emulated
SPUs; ps3recomp answers the same imports with its HLE libs/spurs.

Tests:
  W  custom workload: SPURS on 2 SPUs runs a hand-written policy module (the
     path WWS-style job managers take). Each dispatch increments a shared
     counter with GETLLAR/PUTLLC, capped at N_PM, and returns to the kernel
     through r0; the PPU waits for the counter, then shuts the workload down,
     waits for the shutdown, removes it and finalizes SPURS. Every call's return
     code and the counter line are printed, with the r4 (workload data, u64)
     and r5 (poll status) the kernel entered the module with.
"""
import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ppu"))
import ppc_asm as P          # noqa: E402
import spu_asm as S          # noqa: E402
from gen_ppu_conform import Asm, write_elf, emit_routines, TEXT_BASE  # noqa: E402
from gen_mc_conform import Data  # noqa: E402
from lv2_imports import Imports, SDK_VERSION  # noqa: E402

N_PM = 64               # policy-module increments
PM_BASE = 0xA00         # SPURS loads a policy module at LS 0xA00
SYS_TIMER_USLEEP, SYS_PROCESS_EXIT = 141, 3
FUNCS = ["_start", "hexdump", "puts", "reset_scratch", "load_state", "save_state", "dump_scratch"]

IMPORTS = Imports({
    "cellSysmodule": ["cellSysmoduleLoadModule"],
    "cellSpurs": ["_cellSpursAttributeInitialize", "cellSpursInitializeWithAttribute",
                  "_cellSpursWorkloadAttributeInitialize", "cellSpursAddWorkloadWithAttribute",
                  "cellSpursReadyCountStore", "cellSpursShutdownWorkload",
                  "cellSpursWaitForWorkloadShutdown", "cellSpursRemoveWorkload", "cellSpursFinalize"],
})


def policy_module():
    """Entered at LS 0xA00 by the SPURS kernel: r0 = kernel exit address,
    r3 = kernel context (LS 0x100), r4 = workload data (u64: the counter line
    EA). One capped atomic increment per dispatch, then back to the kernel."""
    a = S.SpuAsm(PM_BASE)
    a.il(9, 0)
    # record the entry registers the kernel handed over: r4 doubleword 0 (the
    # u64 workload data) and r5's preferred word (poll status)
    a.fsmbi(8, 0xFF00); a.and_(6, 4, 8)
    a.fsmbi(8, 0xF000); a.and_(7, 5, 8)
    a.rotqbyi(4, 4, 4)                          # data is a u64: low word -> preferred slot
    a.ila(10, 0x30000)                          # LS line buffer
    a.label("again")
    a.wrch(S.MFC_LSA, 10); a.wrch(S.MFC_EAH, 9); a.wrch(S.MFC_EAL, 4)
    a.il(12, S.GETLLAR); a.wrch(S.MFC_Cmd, 12); a.rdch(12, S.MFC_RdAtomicStat)
    a.lqd(21, 10, 0)
    a.il(22, N_PM); a.clgt(23, 22, 21)          # counter < N_PM ?
    a.brz(23, "out")
    a.il(22, 1); a.a(21, 21, 22); a.stqd(21, 10, 0)
    a.stqd(6, 10, 16); a.stqd(7, 10, 32)
    a.wrch(S.MFC_LSA, 10); a.wrch(S.MFC_EAH, 9); a.wrch(S.MFC_EAL, 4)
    a.il(12, S.PUTLLC); a.wrch(S.MFC_Cmd, 12); a.rdch(13, S.MFC_RdAtomicStat)
    a.brnz(13, "again")
    a.label("out")
    a.bi(0)
    code = a.bytes()
    return code + b"\0" * (-len(code) % 16)


def build(out_path):
    pm = policy_module()
    D = Data()
    IMPORTS.take(D)                             # first: the tables must be in the low 32 KB
    D.take("opd", 8 * len(FUNCS))
    for k in ("hex", "written", "saved_r1", "group_lr", "hdr", "end"):
        D.take(k, 64)
    D.take("line", 2 * 0x400 + 16)
    D.take("pristine", 16); D.take("scratch", 16)
    D.take("spurs", 4096, 128)
    D.take("sattr", 512, 8); D.take("wattr", 512, 8)
    D.take("prio", 16, 16, bytes([8] * 8))
    D.take("pm", len(pm), 128, pm)
    D.take("counter", 128, 128)
    D.take("wid", 16)
    D.take("rcs", 64)
    HDR, END = b"SPURSCONF BEGIN\n", b"SPURSCONF END\n"
    calls = []

    def emit(d):
        t = Asm(TEXT_BASE)

        def puts(addr, n):
            t.emit(P.li32(3, addr), P.addi(4, 0, n)); t.bl("puts")

        def dump(addr, n):
            t.emit(P.li32(3, addr), P.addi(4, 0, n)); t.bl("hexdump")

        def call(fn, *args, stack=()):
            """args -> r3.. (int, or ("mem", addr) for a guest word); stack ->
            the parameter save area (args 9+ at 112(r1)). Stores rc."""
            for i, v in enumerate(args):
                if isinstance(v, tuple):
                    t.emit(P.li32(3 + i, v[1]), P.lwz(3 + i, 3 + i, 0))
                else:
                    t.emit(P.li32(3 + i, v & 0xFFFFFFFF) if v >= 0 else [P.addi(3 + i, 0, v)])
            for i, v in enumerate(stack):
                t.emit(P.li32(11, v), P.std(11, 1, 112 + 8 * i))
            t.bl("imp_" + fn)
            t.emit(P.ld(2, 1, 40))
            t.emit(P.li32(11, d["rcs"] + 4 * len(calls)), P.stw(3, 11, 0))
            calls.append(fn)

        calls.clear()
        t.label("_start")
        t.emit(P.li32(30, d["saved_r1"]), P.std(1, 30, 0), P.stdu(1, 1, -512))
        puts(d["hdr"], len(HDR))

        call("cellSysmoduleLoadModule", 0x0A)
        call("_cellSpursAttributeInitialize", d["sattr"], 2, SDK_VERSION, 2, 100, 1000, 0)
        call("cellSpursInitializeWithAttribute", d["spurs"], d["sattr"])
        # (attr, revision, sdk, pm, size, data u64 = counter EA, priority, minCnt) + maxCnt on the stack
        call("_cellSpursWorkloadAttributeInitialize", d["wattr"], 1, SDK_VERSION, d["pm"], len(pm),
             d["counter"], d["prio"], 1, stack=(2,))
        call("cellSpursAddWorkloadWithAttribute", d["spurs"], d["wid"], d["wattr"])
        call("cellSpursReadyCountStore", d["spurs"], ("mem", d["wid"]), 1)
        # wait for the policy module to reach N_PM (bounded: 5000 x 1 ms)
        t.emit(P.li32(20, 5000))
        t.label("poll")
        t.emit(P.li32(9, d["counter"]), P.lwz(9, 9, 0), P.D(11, 0, 9, N_PM))
        t.bc("polled", 12, 2)
        t.emit(P.addi(3, 0, 1000), P.addi(11, 0, SYS_TIMER_USLEEP), P.sc())
        t.emit(P.addi(20, 20, -1), P.D(11, 0, 20, 0))
        t.bc("poll", 4, 2)
        t.label("polled")
        call("cellSpursShutdownWorkload", d["spurs"], ("mem", d["wid"]))
        call("cellSpursWaitForWorkloadShutdown", d["spurs"], ("mem", d["wid"]))
        call("cellSpursRemoveWorkload", d["spurs"], ("mem", d["wid"]))
        call("cellSpursFinalize", d["spurs"])
        dump(d["rcs"], 4 * len(calls))
        dump(d["counter"], 48)

        puts(d["end"], len(END))
        t.emit(P.addi(3, 0, 0), P.addi(11, 0, SYS_PROCESS_EXIT), P.sc(), P.b(0))
        emit_routines(t, d)
        IMPORTS.emit(t, d)
        return t

    d = D.layout(0x10000000)
    t = emit(d)
    text = t.bytes()
    data_base = (TEXT_BASE + len(text) + 0xFFFF) & ~0xFFFF
    d = D.layout(data_base)
    t = emit(d)
    text = t.bytes()

    data = bytearray(D.off)
    for n, o, size, init in D.items:
        data[o:o + len(init)] = init

    def put(addr, b):
        data[addr - data_base:addr - data_base + len(b)] = b
    toc = d["opd"] + 0x8000
    put(d["opd"], b"".join(struct.pack(">II", t.labels[f], toc) for f in FUNCS))
    put(d["hex"], b"0123456789abcdef")
    put(d["hdr"], HDR); put(d["end"], END)
    IMPORTS.fill(put, d, t)
    elf = write_elf(text, data_base, bytes(data), d["opd"], 8 * len(FUNCS), IMPORTS.extra_ph(d))
    open(out_path, "wb").write(elf)
    print("wrote %s (%d SPURS calls, policy module %d bytes)" % (out_path, len(calls), len(pm)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    build(ap.parse_args().out)
