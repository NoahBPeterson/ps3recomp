#!/usr/bin/env python3
"""Generate the multicore conformance ELF: PPU <-> SPU interaction over raw lv2
syscalls (no imports), checked by its printed transcript.

Tests (each prints hex lines on tty 0, deterministic by construction):
  A  DMA: one SPU GETs then PUTs a pattern buffer for every (size, alignment)
     in a table (1..16 KB, naturally aligned small transfers), then a GETL
     gather; the PPU prints the destination buffers.
  B  atomics: 6 SPUs each increment all four words of a 128-byte line N times
     with GETLLAR/PUTLLC retry loops while the PPU increments word 0 N times
     with lwarx/stwcx. -- expected word0 = 7N, words 1..3 = 6N.
  C  mailbox + signals: the PPU writes four values into the SPU's inbound
     mailbox and two SNR signals; the SPU sums them and exits with the sum.
  Every group join prints (cause, status) and each thread's exit status.

The same bytes run on RPCS3 and through ps3recomp; run_mc_conform.py diffs.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ppu"))
import ppc_asm as P          # noqa: E402
import spu_asm as S          # noqa: E402
from gen_ppu_conform import Asm, write_elf, emit_routines, TEXT_BASE  # noqa: E402

SC = dict(process_exit=3, tty_write=403, spu_initialize=169, image_import=157,
          group_create=170, thread_initialize=172, group_start=173, group_join=178,
          get_exit_status=165, write_snr=184, write_spu_mb=190)

N_ATOMIC = 400
SPU_BASE = 0x0          # SPU programs load at LS 0
LS_BUF = 0x10000        # data area in LS
DMA_CASES = ([(1, o) for o in (0, 1, 7, 15)] + [(2, o) for o in (0, 2, 14)] + [(4, o) for o in (0, 4, 12)] +
             [(8, o) for o in (0, 8)] + [(16 * k, 0) for k in (1, 2, 3, 7, 8, 64, 255, 1024)])


# ------------------------------------------------------------------ SPU programs

def spu_dma_prog():
    """arg1 (r3 pref word) = EA of a param block:
         +0 src EA, +4 dst EA, +8 count, then count x {size, src_off, dst_off, ls_off}
       For each entry: GET size bytes src+src_off -> LS ls_off, then PUT LS ls_off
       -> dst+dst_off. Then a GETL gather of the list at +0x400 into LS 0x30000 and
       a PUT of the gathered bytes to dst+0x8000. Exits with 0x600D."""
    a = S.SpuAsm(SPU_BASE)
    a.rotqbyi(3, 3, 4)                          # arg1 is a u64: low word -> preferred slot
    # GET the param block (2 KB: entries + the GETL list at +0x400) to LS 0x20000
    a.ila(10, 0x20000); a.il(11, 0x800)
    a.mfc(S.GET, 10, 3, 11, 1, 12); a.wait_tag(1, 12)
    a.lqa(20, 0x20000)                         # w0 src, w1 dst, w2 count
    a.ori(21, 20, 0)                            # r21 pref = src
    a.rotqbyi(22, 20, 4)                        # r22 pref = dst
    a.rotqbyi(23, 20, 8)                        # r23 pref = count
    a.ila(24, 0x20010)                          # r24 = LS ptr to entries
    a.label("loop")
    a.lqd(25, 24, 0)                            # size, src_off, dst_off, ls_off
    a.rotqbyi(26, 25, 4); a.rotqbyi(27, 25, 8); a.rotqbyi(28, 25, 12)
    a.a(29, 21, 26)                             # src EA
    a.mfc(S.GET, 28, 29, 25, 2, 12); a.wait_tag(2, 12)
    a.a(29, 22, 27)                             # dst EA
    a.mfc(S.PUT, 28, 29, 25, 2, 12); a.wait_tag(2, 12)
    a.ai(24, 24, 16); a.ai(23, 23, -1); a.brnz(23, "loop")
    # GETL: list at LS 0x20400 (copied in with the param block), 8 elements
    a.ila(10, 0x30000); a.ila(29, 0x20400); a.il(11, 8 * 8)
    a.wrch(S.MFC_LSA, 10)
    a.il(12, 0); a.wrch(S.MFC_EAH, 12)          # list elements' EA high word
    a.wrch(S.MFC_EAL, 29); a.wrch(S.MFC_Size, 11)
    a.il(12, 3); a.wrch(S.MFC_TagID, 12); a.il(12, S.GETL); a.wrch(S.MFC_Cmd, 12)
    a.wait_tag(3, 12)
    a.ila(10, 0x30000); a.ilhu(29, 0); a.iohl(29, 0x8000); a.a(29, 22, 29); a.il(11, 8 * 0x40)
    a.mfc(S.PUT, 10, 29, 11, 3, 12); a.wait_tag(3, 12)
    a.il(30, 0x600D); a.wrch(S.SPU_WrOutMbox, 30); a.stop(0x102)
    return a.bytes()


def spu_atomic_prog():
    """arg1 = EA of the 128-byte line, arg2 = thread index. N GETLLAR/PUTLLC
    increments of all four words of the line's first quadword. Exits with
    0xA000 | index."""
    a = S.SpuAsm(SPU_BASE)
    a.rotqbyi(3, 3, 4); a.rotqbyi(4, 4, 4)      # u64 args: low words -> preferred slots
    a.ila(10, 0x10000)                          # LS line buffer (128-aligned)
    a.ori(5, 4, 0)                              # keep index (r4 pref)
    a.il(20, N_ATOMIC); a.il(22, 1)
    a.label("again")
    a.wrch(S.MFC_LSA, 10); a.il(12, 0); a.wrch(S.MFC_EAH, 12); a.wrch(S.MFC_EAL, 3)
    a.il(12, S.GETLLAR); a.wrch(S.MFC_Cmd, 12); a.rdch(12, S.MFC_RdAtomicStat)
    a.lqd(21, 10, 0); a.a(21, 21, 22); a.stqd(21, 10, 0)
    a.wrch(S.MFC_LSA, 10); a.il(12, 0); a.wrch(S.MFC_EAH, 12); a.wrch(S.MFC_EAL, 3)
    a.il(12, S.PUTLLC); a.wrch(S.MFC_Cmd, 12); a.rdch(13, S.MFC_RdAtomicStat)
    a.brnz(13, "again")                         # bit 0 set: reservation lost, retry
    a.ai(20, 20, -1); a.brnz(20, "again")
    a.ilhu(30, 0); a.iohl(30, 0xA000); a.or_(30, 30, 5)
    a.wrch(S.SPU_WrOutMbox, 30); a.stop(0x102)
    return a.bytes()


def spu_mbox_prog():
    """Sum four inbound-mailbox words and both signal-notification registers;
    exit with the sum."""
    a = S.SpuAsm(SPU_BASE)
    a.il(30, 0)
    for _ in range(4):
        a.rdch(31, S.SPU_RdInMbox); a.a(30, 30, 31)
    a.rdch(31, S.SPU_RdSigNotify1); a.a(30, 30, 31)
    a.rdch(31, S.SPU_RdSigNotify2); a.a(30, 30, 31)
    a.wrch(S.SPU_WrOutMbox, 30); a.stop(0x102)
    return a.bytes()


def wrap_spu(code, tmpdir, name):
    raw = os.path.join(tmpdir, name + ".bin")
    elf = os.path.join(tmpdir, name + ".elf")
    open(raw, "wb").write(code)
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "wrap_spu_elf.py"), raw,
                    "--entry", hex(SPU_BASE), "--base", hex(SPU_BASE), "--out", elf],
                   check=True, capture_output=True)
    return open(elf, "rb").read()


# ------------------------------------------------------------------ PPU program

class Data:
    def __init__(self):
        self.items, self.off = [], 0

    def take(self, name, size, align=16, init=b""):
        self.off = (self.off + align - 1) & ~(align - 1)
        self.items.append((name, self.off, size, init))
        self.off += size
        return name

    def layout(self, base):
        return {n: base + o for n, o, _, _ in self.items}


def build(out_path):
    tmp = tempfile.mkdtemp(prefix="mcconf")
    imgs = {"dma": wrap_spu(spu_dma_prog(), tmp, "dma"),
            "atomic": wrap_spu(spu_atomic_prog(), tmp, "atomic"),
            "mbox": wrap_spu(spu_mbox_prog(), tmp, "mbox")}

    D = Data()
    for k in ("opd", "hex", "written", "saved_r1", "group_lr", "hdr", "end"):
        D.take(k, 64)
    D.take("line", 2 * 0x400 + 16)
    D.take("pristine", 16); D.take("scratch", 16)
    for k, b in imgs.items():
        D.take("img_" + k, len(b), 128, b)
        D.take("imgs_" + k, 32)                            # sys_spu_image
    D.take("gname", 16, 16, b"conform\0")
    D.take("gattr", 16); D.take("tattr", 16); D.take("targ", 32)
    D.take("ids", 64)                                      # group id, thread ids
    D.take("join", 16)                                     # cause, status
    D.take("exst", 64)                                     # exit statuses
    # DMA test buffers
    src = bytes((i * 7 + (i >> 8) * 13 + 1) & 0xFF for i in range(0x10000))
    D.take("dma_src", 0x10000, 128, src)
    D.take("dma_dst", 0x10000, 128)
    D.take("dma_param", 0x800, 128)
    # atomic line
    D.take("line128", 128, 128)
    HDR = b"MCCONF BEGIN\n"
    END = b"MCCONF END\n"

    def emit(d):
        t = Asm(TEXT_BASE)

        def sc(num, *args):
            for i, v in enumerate(args):
                t.emit(P.li32(3 + i, v & 0x7FFFFFFF) if v >= 0 else [P.addi(3 + i, 0, v)])
            t.emit(P.addi(11, 0, num), P.sc())

        def lwz_arg(reg, addr):
            t.emit(P.li32(reg, addr), P.lwz(reg, reg, 0))

        def puts(addr, n):
            t.emit(P.li32(3, addr), P.addi(4, 0, n)); t.bl("puts")

        def dump(addr, n):
            for o in range(0, n, 0x400):
                t.emit(P.li32(3, addr + o), P.addi(4, 0, min(0x400, n - o))); t.bl("hexdump")

        def group(img, nthreads, args_for):
            """create group, init threads (args_for(i) -> (arg1, arg2)), start."""
            sc(SC["group_create"], d["ids"], nthreads, 100, d["gattr"])
            for i in range(nthreads):
                a1, a2 = args_for(i)
                # sys_spu_thread_argument: arg1/arg2 as u64 (high word 0)
                t.emit(P.li32(5, d["targ"]), P.li32(6, a1), P.stw(6, 5, 4), P.li32(6, a2), P.stw(6, 5, 12))
                lwz_arg(4, d["ids"])
                t.emit(P.li32(3, d["ids"] + 4 + 4 * i), P.addi(5, 0, i), P.li32(6, d["imgs_" + img]),
                       P.li32(7, d["tattr"]), P.li32(8, d["targ"]), P.addi(11, 0, SC["thread_initialize"]), P.sc())
            lwz_arg(3, d["ids"])
            t.emit(P.addi(11, 0, SC["group_start"]), P.sc())

        def join_and_report(nthreads):
            lwz_arg(3, d["ids"])
            t.emit(P.li32(4, d["join"]), P.li32(5, d["join"] + 4), P.addi(11, 0, SC["group_join"]), P.sc())
            for i in range(nthreads):
                lwz_arg(3, d["ids"] + 4 + 4 * i)
                t.emit(P.li32(4, d["exst"] + 4 * i), P.addi(11, 0, SC["get_exit_status"]), P.sc())
            dump(d["join"], 8)
            dump(d["exst"], 4 * nthreads)

        t.label("_start")
        t.emit(P.li32(30, d["saved_r1"]), P.std(1, 30, 0))
        puts(d["hdr"], len(HDR))
        sc(SC["spu_initialize"], 6, 0)
        for k in imgs:
            sc(SC["image_import"], d["imgs_" + k], d["img_" + k], len(imgs[k]), 0)

        # A: DMA
        group("dma", 1, lambda i: (d["dma_param"], 0))
        join_and_report(1)
        dump(d["dma_dst"], 0x8000 + 8 * 0x40)

        # B: atomics (6 SPUs + this PPU thread)
        group("atomic", 6, lambda i: (d["line128"], i))
        t.emit(P.li32(29, d["line128"]), P.li32(28, N_ATOMIC))
        t.label("ppu_atomic")
        t.emit(P.X(31, 27, 0, 29, 20),                          # lwarx r27,0,r29
               P.addi(27, 27, 1),
               P.X(31, 27, 0, 29, 150, 1))                      # stwcx. r27,0,r29
        t.emit((16 << 26) | (4 << 21) | (2 << 16) | ((-12) & 0xFFFC))  # bne- lwarx (retry)
        t.emit(P.addi(28, 28, -1), P.D(11, 0, 28, 0))           # cmpwi r28,0
        here = t.pc
        t.emit((16 << 26) | (4 << 21) | (2 << 16) | ((t.labels["ppu_atomic"] - here) & 0xFFFC))  # bne loop
        join_and_report(6)
        dump(d["line128"], 16)

        # C: mailbox + signals
        group("mbox", 1, lambda i: (0, 0))
        for v in (0x11, 0x222, 0x3333, 0x44444):
            lwz_arg(3, d["ids"] + 4)
            t.emit(P.li32(4, v), P.addi(11, 0, SC["write_spu_mb"]), P.sc())
        for n, v in ((0, 0x500000), (1, 0x6000000)):
            lwz_arg(3, d["ids"] + 4)
            t.emit(P.addi(4, 0, n), P.li32(5, v), P.addi(11, 0, SC["write_snr"]), P.sc())
        join_and_report(1)

        puts(d["end"], len(END))
        t.emit(P.addi(3, 0, 0), P.addi(11, 0, SC["process_exit"]), P.sc(), P.b(0))
        emit_routines(t, d)
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
    FUNCS = ["_start", "hexdump", "puts", "reset_scratch", "load_state", "save_state", "dump_scratch"]
    toc = d["opd"] + 0x8000
    put(d["opd"], b"".join(struct.pack(">II", t.labels[f], toc) for f in FUNCS))
    put(d["hex"], b"0123456789abcdef")
    put(d["hdr"], HDR); put(d["end"], END)
    put(d["gattr"], struct.pack(">IIiI", 8, d["gname"], 0, 0))
    put(d["tattr"], struct.pack(">III", d["gname"], 8, 0))
    # DMA param block: src, dst, count, entries; GETL list at +0x400
    ents, ls, dst_off, src_off = [], LS_BUF, 0, 0x100
    for size, al in DMA_CASES:
        so = (src_off + 15) & ~15 | al
        do = (dst_off + 15) & ~15 | al
        lo = (ls + 15) & ~15 | al
        ents.append((size, so, do, lo))
        src_off, dst_off, ls = so + size + 3, do + size + 3, lo + size
    assert dst_off < 0x8000 and ls < 0x20000
    blk = struct.pack(">III", d["dma_src"], d["dma_dst"], len(ents)).ljust(16, b"\0")
    blk += b"".join(struct.pack(">IIII", *e) for e in ents)
    blk = blk.ljust(0x400, b"\0")
    blk += b"".join(struct.pack(">II", 0x40, d["dma_src"] + 0x9000 + 0x300 * i) for i in range(8))
    put(d["dma_param"], blk)
    elf = write_elf(text, data_base, bytes(data), d["opd"], 8 * len(FUNCS))
    open(out_path, "wb").write(elf)
    print("wrote %s (%d DMA cases, %d atomic increments per agent)" % (out_path, len(ents), N_ATOMIC))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    build(ap.parse_args().out)
