/* spu_interp_diff_test.c -- differential test of the SPU interpreter.
 *
 * For every opcode in the SPU ISA (table taken from RPCS3 SPUOpcodes.h, an
 * independent source from tools/spu_disasm.py) this generates random
 * instruction words (random register / immediate fields) and random register
 * and local-store contents, executes ONE instruction through the interpreter's
 * spu_step(), and compares the result (destination register, all other
 * registers, stored quadword, pc, link, halt/stop state) against an
 * independent reference written from the SPU ISA on a plain big-endian
 * byte-array register model (so host-lane-order bugs in spu_helpers.h show up).
 * It also cross-checks the interpreter's opcode DECODE for all 2048 11-bit
 * prefixes plus random words.
 *
 * Build / run (from this directory):
 *   gcc -std=gnu11 -O1 -I.. -o spu_interp_diff_test spu_interp_diff_test.c -lm && ./spu_interp_diff_test [trials-per-op] [seed]
 * Exit status 0 = no mismatch.
 *
 * Float ops are compared on "tame" operands (normal numbers, mid exponents)
 * with a 1-ulp tolerance: the SPU rounds single precision toward zero and the
 * helpers use host round-to-nearest. frest/frsqest/fi/dftsv are not covered
 * (estimate tables; see report). Channel ops check only channel number and
 * operand routing via stubs.
 */
#include "../spu_interp.c"   /* spu_step / spu_decode1 are static */
#undef A
#undef B
#undef T
#undef I
#undef DST
#undef DSTC
#undef PREF
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---- stubs for the runtime the interpreter links against ---- */
static uint32_t g_ch_n, g_ch_last; static u128 g_ch_val; static uint32_t g_ch_cnt;
u128 spu_rdch(spu_context* c, uint32_t ch) { (void)c; g_ch_last = ch; g_ch_n++; return g_ch_val; }
void spu_wrch(spu_context* c, uint32_t ch, u128 v) { (void)c; g_ch_last = ch; g_ch_n++; g_ch_val = v; }
uint32_t spu_rchcnt(spu_context* c, uint32_t ch) { (void)c; g_ch_last = ch; g_ch_n++; return g_ch_cnt; }
int g_spu_ls_watch_n, g_spu_ls_probe, g_wws_read_probe, g_wws_code_probe, g_spu_smc_watch;
spu_lifted_fn spu_lifted_lookup(const spu_context* c, uint32_t a) { (void)c;(void)a; return 0; }
void spu_spurs_taskset_syscall(spu_context* c) { (void)c; }
void spu_ls_watch_slow(uint32_t l, int w, const uint8_t* p, uint32_t pc, uint32_t lr) { (void)l;(void)w;(void)p;(void)pc;(void)lr; }

/* ---- ISA opcode table: X(magn, value, NAME). insn>>21 == value<<magn | low(magn) (RPCS3 SPUOpcodes.h) ---- */
#define SPU_OPTAB(X) \
    X(0, 0x0, STOP) \
    X(0, 0x1, LNOP) \
    X(0, 0x2, SYNC) \
    X(0, 0x3, DSYNC) \
    X(0, 0xc, MFSPR) \
    X(0, 0xd, RDCH) \
    X(0, 0xf, RCHCNT) \
    X(0, 0x40, SF) \
    X(0, 0x41, OR) \
    X(0, 0x42, BG) \
    X(0, 0x48, SFH) \
    X(0, 0x49, NOR) \
    X(0, 0x53, ABSDB) \
    X(0, 0x58, ROT) \
    X(0, 0x59, ROTM) \
    X(0, 0x5a, ROTMA) \
    X(0, 0x5b, SHL) \
    X(0, 0x5c, ROTH) \
    X(0, 0x5d, ROTHM) \
    X(0, 0x5e, ROTMAH) \
    X(0, 0x5f, SHLH) \
    X(0, 0x78, ROTI) \
    X(0, 0x79, ROTMI) \
    X(0, 0x7a, ROTMAI) \
    X(0, 0x7b, SHLI) \
    X(0, 0x7c, ROTHI) \
    X(0, 0x7d, ROTHMI) \
    X(0, 0x7e, ROTMAHI) \
    X(0, 0x7f, SHLHI) \
    X(0, 0xc0, A) \
    X(0, 0xc1, AND) \
    X(0, 0xc2, CG) \
    X(0, 0xc8, AH) \
    X(0, 0xc9, NAND) \
    X(0, 0xd3, AVGB) \
    X(0, 0x10c, MTSPR) \
    X(0, 0x10d, WRCH) \
    X(0, 0x128, BIZ) \
    X(0, 0x129, BINZ) \
    X(0, 0x12a, BIHZ) \
    X(0, 0x12b, BIHNZ) \
    X(0, 0x140, STOPD) \
    X(0, 0x144, STQX) \
    X(0, 0x1a8, BI) \
    X(0, 0x1a9, BISL) \
    X(0, 0x1aa, IRET) \
    X(0, 0x1ab, BISLED) \
    X(0, 0x1ac, HBR) \
    X(0, 0x1b0, GB) \
    X(0, 0x1b1, GBH) \
    X(0, 0x1b2, GBB) \
    X(0, 0x1b4, FSM) \
    X(0, 0x1b5, FSMH) \
    X(0, 0x1b6, FSMB) \
    X(0, 0x1b8, FREST) \
    X(0, 0x1b9, FRSQEST) \
    X(0, 0x1c4, LQX) \
    X(0, 0x1cc, ROTQBYBI) \
    X(0, 0x1cd, ROTQMBYBI) \
    X(0, 0x1cf, SHLQBYBI) \
    X(0, 0x1d4, CBX) \
    X(0, 0x1d5, CHX) \
    X(0, 0x1d6, CWX) \
    X(0, 0x1d7, CDX) \
    X(0, 0x1d8, ROTQBI) \
    X(0, 0x1d9, ROTQMBI) \
    X(0, 0x1db, SHLQBI) \
    X(0, 0x1dc, ROTQBY) \
    X(0, 0x1dd, ROTQMBY) \
    X(0, 0x1df, SHLQBY) \
    X(0, 0x1f0, ORX) \
    X(0, 0x1f4, CBD) \
    X(0, 0x1f5, CHD) \
    X(0, 0x1f6, CWD) \
    X(0, 0x1f7, CDD) \
    X(0, 0x1f8, ROTQBII) \
    X(0, 0x1f9, ROTQMBII) \
    X(0, 0x1fb, SHLQBII) \
    X(0, 0x1fc, ROTQBYI) \
    X(0, 0x1fd, ROTQMBYI) \
    X(0, 0x1ff, SHLQBYI) \
    X(0, 0x201, NOP) \
    X(0, 0x240, CGT) \
    X(0, 0x241, XOR) \
    X(0, 0x248, CGTH) \
    X(0, 0x249, EQV) \
    X(0, 0x250, CGTB) \
    X(0, 0x253, SUMB) \
    X(0, 0x258, HGT) \
    X(0, 0x2a5, CLZ) \
    X(0, 0x2a6, XSWD) \
    X(0, 0x2ae, XSHW) \
    X(0, 0x2b4, CNTB) \
    X(0, 0x2b6, XSBH) \
    X(0, 0x2c0, CLGT) \
    X(0, 0x2c1, ANDC) \
    X(0, 0x2c2, FCGT) \
    X(0, 0x2c3, DFCGT) \
    X(0, 0x2c4, FA) \
    X(0, 0x2c5, FS) \
    X(0, 0x2c6, FM) \
    X(0, 0x2c8, CLGTH) \
    X(0, 0x2c9, ORC) \
    X(0, 0x2ca, FCMGT) \
    X(0, 0x2cb, DFCMGT) \
    X(0, 0x2cc, DFA) \
    X(0, 0x2cd, DFS) \
    X(0, 0x2ce, DFM) \
    X(0, 0x2d0, CLGTB) \
    X(0, 0x2d8, HLGT) \
    X(0, 0x35c, DFMA) \
    X(0, 0x35d, DFMS) \
    X(0, 0x35e, DFNMS) \
    X(0, 0x35f, DFNMA) \
    X(0, 0x3c0, CEQ) \
    X(0, 0x3ce, MPYHHU) \
    X(0, 0x340, ADDX) \
    X(0, 0x341, SFX) \
    X(0, 0x342, CGX) \
    X(0, 0x343, BGX) \
    X(0, 0x346, MPYHHA) \
    X(0, 0x34e, MPYHHAU) \
    X(0, 0x398, FSCRRD) \
    X(0, 0x3b8, FESD) \
    X(0, 0x3b9, FRDS) \
    X(0, 0x3ba, FSCRWR) \
    X(0, 0x3bf, DFTSV) \
    X(0, 0x3c2, FCEQ) \
    X(0, 0x3c3, DFCEQ) \
    X(0, 0x3c4, MPY) \
    X(0, 0x3c5, MPYH) \
    X(0, 0x3c6, MPYHH) \
    X(0, 0x3c7, MPYS) \
    X(0, 0x3c8, CEQH) \
    X(0, 0x3ca, FCMEQ) \
    X(0, 0x3cb, DFCMEQ) \
    X(0, 0x3cc, MPYU) \
    X(0, 0x3d0, CEQB) \
    X(0, 0x3d4, FI) \
    X(0, 0x3d8, HEQ) \
    X(1, 0x1d8, CFLTS) \
    X(1, 0x1d9, CFLTU) \
    X(1, 0x1da, CSFLT) \
    X(1, 0x1db, CUFLT) \
    X(2, 0x40, BRZ) \
    X(2, 0x41, STQA) \
    X(2, 0x42, BRNZ) \
    X(2, 0x44, BRHZ) \
    X(2, 0x46, BRHNZ) \
    X(2, 0x47, STQR) \
    X(2, 0x60, BRA) \
    X(2, 0x61, LQA) \
    X(2, 0x62, BRASL) \
    X(2, 0x64, BR) \
    X(2, 0x65, FSMBI) \
    X(2, 0x66, BRSL) \
    X(2, 0x67, LQR) \
    X(2, 0x81, IL) \
    X(2, 0x82, ILHU) \
    X(2, 0x83, ILH) \
    X(2, 0xc1, IOHL) \
    X(3, 0x4, ORI) \
    X(3, 0x5, ORHI) \
    X(3, 0x6, ORBI) \
    X(3, 0xc, SFI) \
    X(3, 0xd, SFHI) \
    X(3, 0x14, ANDI) \
    X(3, 0x15, ANDHI) \
    X(3, 0x16, ANDBI) \
    X(3, 0x1c, AI) \
    X(3, 0x1d, AHI) \
    X(3, 0x24, STQD) \
    X(3, 0x34, LQD) \
    X(3, 0x44, XORI) \
    X(3, 0x45, XORHI) \
    X(3, 0x46, XORBI) \
    X(3, 0x4c, CGTI) \
    X(3, 0x4d, CGTHI) \
    X(3, 0x4e, CGTBI) \
    X(3, 0x4f, HGTI) \
    X(3, 0x5c, CLGTI) \
    X(3, 0x5d, CLGTHI) \
    X(3, 0x5e, CLGTBI) \
    X(3, 0x5f, HLGTI) \
    X(3, 0x74, MPYI) \
    X(3, 0x75, MPYUI) \
    X(3, 0x7c, CEQI) \
    X(3, 0x7d, CEQHI) \
    X(3, 0x7e, CEQBI) \
    X(3, 0x7f, HEQI) \
    X(4, 0x8, HBRA) \
    X(4, 0x9, HBRR) \
    X(4, 0x21, ILA) \
    X(7, 0x8, SELB) \
    X(7, 0xb, SHUFB) \
    X(7, 0xc, MPYA) \
    X(7, 0xd, FNMS) \
    X(7, 0xe, FMA) \
    X(7, 0xf, FMS)

enum { R_UNK = 0,
#define X(m, v, n) R_##n,
SPU_OPTAB(X)
#undef X
  R_COUNT };
static const char* const r_name[R_COUNT] = { "?",
#define X(m, v, n) #n,
SPU_OPTAB(X)
#undef X
};
static const struct { int magn; uint32_t val; int op; } r_tab[] = {
#define X(m, v, n) { m, v, R_##n },
SPU_OPTAB(X)
#undef X
};
#define R_NTAB ((int)(sizeof r_tab / sizeof r_tab[0]))

static int ref_decode(uint32_t insn) {
    uint32_t p = insn >> 21; int op = R_UNK;
    for (int i = 0; i < R_NTAB; i++) if ((p >> r_tab[i].magn) == r_tab[i].val) op = r_tab[i].op;
    return op;
}

/* ---- big-endian quadword model ---- */
typedef struct { uint8_t b[16]; } Q;
typedef unsigned __int128 u128b;
static uint32_t qw(const Q* q, int i) { return (uint32_t)q->b[4*i]<<24 | q->b[4*i+1]<<16 | q->b[4*i+2]<<8 | q->b[4*i+3]; }
static void sw(Q* q, int i, uint32_t v) { q->b[4*i]=v>>24; q->b[4*i+1]=v>>16; q->b[4*i+2]=v>>8; q->b[4*i+3]=v; }
static uint16_t qh(const Q* q, int i) { return (uint16_t)(q->b[2*i]<<8 | q->b[2*i+1]); }
static void sh_(Q* q, int i, uint16_t v) { q->b[2*i]=v>>8; q->b[2*i+1]=v; }
static uint64_t qd(const Q* q, int i) { return (uint64_t)qw(q,2*i)<<32 | qw(q,2*i+1); }
static void sd(Q* q, int i, uint64_t v) { sw(q,2*i,(uint32_t)(v>>32)); sw(q,2*i+1,(uint32_t)v); }
static Q q_of(u128 v) { Q q; for (int i = 0; i < 4; i++) sw(&q, i, v._u32[i]); return q; }
static u128b q128(const Q* q) { u128b r = 0; for (int i = 0; i < 16; i++) r = (r << 8) | q->b[i]; return r; }
static Q of128(u128b v) { Q q; for (int i = 15; i >= 0; i--) { q.b[i] = (uint8_t)v; v >>= 8; } return q; }
static Q splat32(uint32_t v) { Q q; for (int i = 0; i < 4; i++) sw(&q, i, v); return q; }
static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t u_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* ---- reference result ---- */
typedef struct {
    int dest; Q val;                 /* destination reg (-1 none) */
    uint32_t next;                   /* next pc */
    int halt, stop; uint32_t stop_code;
    int st; uint32_t st_addr; Q st_val;
    int ch_op; uint32_t ch_no;       /* 1 rdch, 2 wrch, 3 rchcnt */
    int fuzzy;                       /* float result: allow 1 ulp per word */
    int fuzzd;                       /* double result */
} Ref;

static int32_t sx_(uint32_t v, int bits) { uint32_t m = 1u << (bits - 1); return (int32_t)((v ^ m) - m); }
#define LSM 0x3FFFFu

/* ref_exec: regs = original register file (Q computed on demand), ls = pristine LS */
static Ref ref_exec(int op, uint32_t insn, uint32_t pc, const u128* regs, const uint8_t* ls, uint32_t srr0, uint32_t evt) {
    Ref R; memset(&R, 0, sizeof R); R.dest = -1; R.next = (pc + 4) & LSM;
    int rt = insn & 0x7F, ra = (insn >> 7) & 0x7F, rb = (insn >> 14) & 0x7F;
    int rc = insn & 0x7F, rt4 = (insn >> 21) & 0x7F;
    uint32_t i7 = (insn >> 14) & 0x7F, i8 = (insn >> 14) & 0xFF;
    int32_t si10 = sx_((insn >> 14) & 0x3FF, 10), si16 = sx_((insn >> 7) & 0xFFFF, 16);
    uint32_t i16 = (insn >> 7) & 0xFFFF, i18 = (insn >> 7) & 0x3FFFF;
    Q A = q_of(regs[ra]), B = q_of(regs[rb]), T = q_of(regs[rt]);   /* T: rt as a source (also rc in RRR) */
    Q C = T; (void)rc;
    Q r; memset(&r, 0, sizeof r);
    int dest = rt;
    uint32_t aw0 = qw(&A, 0), bw0 = qw(&B, 0);
    int i;
#define W(q,i) qw(&(q),i)
#define LANE4(expr) for (i = 0; i < 4; i++) { uint32_t a = W(A,i), b = W(B,i), t = W(T,i); (void)a;(void)b;(void)t; sw(&r, i, (uint32_t)(expr)); }
#define LANE8(expr) for (i = 0; i < 8; i++) { uint16_t a = qh(&A,i), b = qh(&B,i); (void)a;(void)b; sh_(&r, i, (uint16_t)(expr)); }
#define LANE16(expr) for (i = 0; i < 16; i++) { uint8_t a = A.b[i], b = B.b[i]; (void)a;(void)b; r.b[i] = (uint8_t)(expr); }
#define LANE16I(expr) for (i = 0; i < 16; i++) { uint8_t a = A.b[i]; (void)a; r.b[i] = (uint8_t)(expr); }
    switch (op) {
    case R_STOP: R.stop = 1; R.stop_code = insn & 0x3FFF; dest = -1; break;
    case R_STOPD: R.stop = 1; R.stop_code = 0; dest = -1; break;
    case R_LNOP: case R_NOP: case R_SYNC: case R_DSYNC: case R_MTSPR: case R_HBR: case R_HBRA: case R_HBRR:
    case R_FSCRWR: dest = -1; break;
    case R_MFSPR: case R_FSCRRD: break;                         /* reads as zero */
    case R_RDCH: R.ch_op = 1; R.ch_no = ra; break;
    case R_WRCH: R.ch_op = 2; R.ch_no = ra; dest = -1; break;
    case R_RCHCNT: R.ch_op = 3; R.ch_no = ra; break;
    /* integer */
    case R_A: LANE4(a + b); break;
    case R_SF: LANE4(b - a); break;
    case R_AH: LANE8(a + b); break;
    case R_SFH: LANE8(b - a); break;
    case R_AI: LANE4(a + (uint32_t)si10); break;
    case R_AHI: LANE8(a + (uint16_t)si10); break;
    case R_SFI: LANE4((uint32_t)si10 - a); break;
    case R_SFHI: LANE8((uint16_t)si10 - a); break;
    case R_ADDX: LANE4(a + b + (t & 1)); break;
    case R_SFX: LANE4(b - a - 1 + (t & 1)); break;
    case R_CG: LANE4(((uint64_t)a + b) >> 32); break;
    case R_CGX: LANE4(((uint64_t)a + b + (t & 1)) >> 32); break;
    case R_BG: LANE4(b >= a ? 1 : 0); break;
    case R_BGX: LANE4(((uint64_t)b + (uint32_t)~a + (t & 1)) >> 32); break;
    case R_MPY: LANE4((int32_t)(int16_t)a * (int32_t)(int16_t)b); break;
    case R_MPYU: LANE4((uint32_t)(a & 0xFFFF) * (b & 0xFFFF)); break;
    case R_MPYH: LANE4(((int32_t)(int16_t)(a >> 16) * (int32_t)(int16_t)b) * 65536); break;
    case R_MPYHH: LANE4((int32_t)(int16_t)(a >> 16) * (int32_t)(int16_t)(b >> 16)); break;
    case R_MPYHHU: LANE4((a >> 16) * (b >> 16)); break;
    case R_MPYS: LANE4((int32_t)(int16_t)(((int32_t)(int16_t)a * (int32_t)(int16_t)b) >> 16)); break;
    case R_MPYHHA: LANE4(t + (int32_t)(int16_t)(a >> 16) * (int32_t)(int16_t)(b >> 16)); break;
    case R_MPYHHAU: LANE4(t + (a >> 16) * (b >> 16)); break;
    case R_MPYI: LANE4((int32_t)(int16_t)a * si10); break;
    case R_MPYUI: LANE4((a & 0xFFFF) * (uint16_t)si10); break;
    case R_MPYA: { dest = rt4; Q c = q_of(regs[rc]); for (i = 0; i < 4; i++) { uint32_t a = W(A,i), b = W(B,i);
            sw(&r, i, (uint32_t)((int32_t)(int16_t)a * (int32_t)(int16_t)b) + W(c,i)); } break; }
    case R_CLZ: LANE4(a ? __builtin_clz(a) : 32); break;
    case R_CNTB: LANE16I(__builtin_popcount(a)); break;
    case R_XSBH: LANE8((int16_t)(int8_t)(qh(&A,i) & 0xFF)); break;
    case R_XSHW: LANE4((int32_t)(int16_t)(a & 0xFFFF)); break;
    case R_XSWD: for (i = 0; i < 2; i++) sd(&r, i, (uint64_t)(int64_t)(int32_t)qw(&A, 2*i+1)); break;
    case R_ORX: sw(&r, 0, W(A,0)|W(A,1)|W(A,2)|W(A,3)); break;
    case R_ABSDB: LANE16(a > b ? a - b : b - a); break;
    case R_AVGB: LANE16(((unsigned)a + b + 1) >> 1); break;
    case R_SUMB: for (i = 0; i < 4; i++) { uint32_t sa = 0, sb = 0; for (int k = 0; k < 4; k++) { sa += A.b[4*i+k]; sb += B.b[4*i+k]; }
            sh_(&r, 2*i, (uint16_t)sb); sh_(&r, 2*i+1, (uint16_t)sa); } break;
    /* logical */
    case R_AND: LANE16(a & b); break;
    case R_OR: LANE16(a | b); break;
    case R_XOR: LANE16(a ^ b); break;
    case R_NAND: LANE16(~(a & b)); break;
    case R_NOR: LANE16(~(a | b)); break;
    case R_ANDC: LANE16(a & ~b); break;
    case R_ORC: LANE16(a | ~b); break;
    case R_EQV: LANE16(~(a ^ b)); break;
    case R_ANDI: LANE4(a & (uint32_t)si10); break;
    case R_ORI: LANE4(a | (uint32_t)si10); break;
    case R_XORI: LANE4(a ^ (uint32_t)si10); break;
    case R_ANDHI: LANE8(a & (uint16_t)si10); break;
    case R_ORHI: LANE8(a | (uint16_t)si10); break;
    case R_XORHI: LANE8(a ^ (uint16_t)si10); break;
    case R_ANDBI: LANE16I(a & (uint8_t)si10); break;
    case R_ORBI: LANE16I(a | (uint8_t)si10); break;
    case R_XORBI: LANE16I(a ^ (uint8_t)si10); break;
    case R_SELB: { dest = rt4; Q c = q_of(regs[rc]); for (i = 0; i < 16; i++) r.b[i] = (c.b[i] & B.b[i]) | (~c.b[i] & A.b[i]); break; }
    case R_SHUFB: { dest = rt4; Q c = q_of(regs[rc]);
        for (i = 0; i < 16; i++) { uint8_t s = c.b[i], v;
            if ((s & 0xC0) == 0x80) v = 0; else if ((s & 0xE0) == 0xC0) v = 0xFF; else if ((s & 0xE0) == 0xE0) v = 0x80;
            else { int k = s & 0x1F; v = k < 16 ? A.b[k] : B.b[k - 16]; }
            r.b[i] = v; } break; }
    /* compares */
    case R_CEQ: LANE4(a == b ? ~0u : 0); break;
    case R_CEQH: LANE8(a == b ? 0xFFFF : 0); break;
    case R_CEQB: LANE16(a == b ? 0xFF : 0); break;
    case R_CEQI: LANE4(a == (uint32_t)si10 ? ~0u : 0); break;
    case R_CEQHI: LANE8(a == (uint16_t)si10 ? 0xFFFF : 0); break;
    case R_CEQBI: LANE16I(a == (uint8_t)si10 ? 0xFF : 0); break;
    case R_CGT: LANE4((int32_t)a > (int32_t)b ? ~0u : 0); break;
    case R_CGTH: LANE8((int16_t)a > (int16_t)b ? 0xFFFF : 0); break;
    case R_CGTB: LANE16((int8_t)a > (int8_t)b ? 0xFF : 0); break;
    case R_CGTI: LANE4((int32_t)a > si10 ? ~0u : 0); break;
    case R_CGTHI: LANE8((int16_t)a > (int16_t)si10 ? 0xFFFF : 0); break;
    case R_CGTBI: LANE16I((int8_t)a > (int8_t)si10 ? 0xFF : 0); break;
    case R_CLGT: LANE4(a > b ? ~0u : 0); break;
    case R_CLGTH: LANE8(a > b ? 0xFFFF : 0); break;
    case R_CLGTB: LANE16(a > b ? 0xFF : 0); break;
    case R_CLGTI: LANE4(a > (uint32_t)si10 ? ~0u : 0); break;
    case R_CLGTHI: LANE8(a > (uint16_t)si10 ? 0xFFFF : 0); break;
    case R_CLGTBI: LANE16I(a > (uint8_t)si10 ? 0xFF : 0); break;
    case R_HEQ: dest = -1; R.halt = (int32_t)aw0 == (int32_t)bw0; break;
    case R_HEQI: dest = -1; R.halt = (int32_t)aw0 == si10; break;
    case R_HGT: dest = -1; R.halt = (int32_t)aw0 > (int32_t)bw0; break;
    case R_HGTI: dest = -1; R.halt = (int32_t)aw0 > si10; break;
    case R_HLGT: dest = -1; R.halt = aw0 > bw0; break;
    case R_HLGTI: dest = -1; R.halt = aw0 > (uint32_t)si10; break;
    /* shifts / rotates */
    case R_SHL: LANE4((b & 0x3F) > 31 ? 0 : a << (b & 0x3F)); break;
    case R_SHLI: LANE4((i7 & 0x3F) > 31 ? 0 : a << (i7 & 0x3F)); break;
    case R_ROT: LANE4((a << (b & 31)) | (uint32_t)((uint64_t)a >> (32 - (b & 31)) & 0xFFFFFFFFu)); break;
    case R_ROTI: LANE4((a << (i7 & 31)) | (uint32_t)((uint64_t)a >> (32 - (i7 & 31)) & 0xFFFFFFFFu)); break;
    case R_ROTM: LANE4(((0u - b) & 0x3F) > 31 ? 0 : a >> ((0u - b) & 0x3F)); break;
    case R_ROTMI: LANE4(((0u - i7) & 0x3F) > 31 ? 0 : a >> ((0u - i7) & 0x3F)); break;
    case R_ROTMA: LANE4(((0u - b) & 0x3F) > 31 ? (uint32_t)((int32_t)a >> 31) : (uint32_t)((int32_t)a >> ((0u - b) & 0x3F))); break;
    case R_ROTMAI: LANE4(((0u - i7) & 0x3F) > 31 ? (uint32_t)((int32_t)a >> 31) : (uint32_t)((int32_t)a >> ((0u - i7) & 0x3F))); break;
    case R_SHLH: LANE8((b & 0x1F) > 15 ? 0 : a << (b & 0x1F)); break;
    case R_SHLHI: LANE8((i7 & 0x1F) > 15 ? 0 : a << (i7 & 0x1F)); break;
    case R_ROTH: LANE8(((uint32_t)a << (b & 15)) | ((uint32_t)a >> (16 - (b & 15)))); break;
    case R_ROTHI: LANE8(((uint32_t)a << (i7 & 15)) | ((uint32_t)a >> (16 - (i7 & 15)))); break;
    case R_ROTHM: LANE8(((0u - b) & 0x1F) > 15 ? 0 : a >> ((0u - b) & 0x1F)); break;
    case R_ROTHMI: LANE8(((0u - i7) & 0x1F) > 15 ? 0 : a >> ((0u - i7) & 0x1F)); break;
    case R_ROTMAH: LANE8(((0u - b) & 0x1F) > 15 ? (int16_t)a >> 15 : (int16_t)a >> ((0u - b) & 0x1F)); break;
    case R_ROTMAHI: LANE8(((0u - i7) & 0x1F) > 15 ? (int16_t)a >> 15 : (int16_t)a >> ((0u - i7) & 0x1F)); break;
    case R_SHLQBI: case R_SHLQBII: { unsigned n = (op == R_SHLQBI ? bw0 : i7) & 7; r = of128(q128(&A) << n); break; }
    case R_ROTQBI: case R_ROTQBII: { unsigned n = (op == R_ROTQBI ? bw0 : i7) & 7; u128b v = q128(&A); r = of128(n ? (v << n) | (v >> (128 - n)) : v); break; }
    case R_ROTQMBI: case R_ROTQMBII: { unsigned n = (0u - (op == R_ROTQMBI ? bw0 : i7)) & 7; r = of128(q128(&A) >> n); break; }
    case R_SHLQBY: case R_SHLQBYI: case R_SHLQBYBI: { unsigned n = op == R_SHLQBY ? (bw0 & 0x1F) : op == R_SHLQBYI ? (i7 & 0x1F) : ((bw0 >> 3) & 0x1F);
        for (i = 0; i < 16; i++) r.b[i] = (i + n < 16) ? A.b[i + n] : 0; break; }
    case R_ROTQBY: case R_ROTQBYI: case R_ROTQBYBI: { unsigned n = op == R_ROTQBY ? (bw0 & 0xF) : op == R_ROTQBYI ? (i7 & 0xF) : ((bw0 >> 3) & 0xF);
        for (i = 0; i < 16; i++) r.b[i] = A.b[(i + n) & 15]; break; }
    case R_ROTQMBY: case R_ROTQMBYI: case R_ROTQMBYBI: { unsigned n = op == R_ROTQMBY ? ((0u - bw0) & 0x1F) : op == R_ROTQMBYI ? ((0u - i7) & 0x1F) : ((0u - (bw0 >> 3)) & 0x1F);
        for (i = 0; i < 16; i++) r.b[i] = (i >= (int)n && n < 16) ? A.b[i - n] : 0; break; }
    /* masks / gathers */
    case R_FSM: { uint32_t v = aw0; for (i = 0; i < 4; i++) sw(&r, i, ((v >> (3 - i)) & 1) ? ~0u : 0); break; }
    case R_FSMH: { uint32_t v = aw0; for (i = 0; i < 8; i++) sh_(&r, i, ((v >> (7 - i)) & 1) ? 0xFFFF : 0); break; }
    case R_FSMB: { uint32_t v = aw0; for (i = 0; i < 16; i++) r.b[i] = ((v >> (15 - i)) & 1) ? 0xFF : 0; break; }
    case R_FSMBI: for (i = 0; i < 16; i++) r.b[i] = ((i16 >> (15 - i)) & 1) ? 0xFF : 0; break;
    case R_GB: { uint32_t v = 0; for (i = 0; i < 4; i++) v |= (W(A,i) & 1) << (3 - i); sw(&r, 0, v); break; }
    case R_GBH: { uint32_t v = 0; for (i = 0; i < 8; i++) v |= (qh(&A,i) & 1u) << (7 - i); sw(&r, 0, v); break; }
    case R_GBB: { uint32_t v = 0; for (i = 0; i < 16; i++) v |= (A.b[i] & 1u) << (15 - i); sw(&r, 0, v); break; }
    case R_CBD: case R_CBX: case R_CHD: case R_CHX: case R_CWD: case R_CWX: case R_CDD: case R_CDX: {
        uint32_t t = aw0 + ((op==R_CBD||op==R_CHD||op==R_CWD||op==R_CDD) ? (uint32_t)sx_(i7,7) : bw0);
        int sz = (op==R_CBD||op==R_CBX) ? 1 : (op==R_CHD||op==R_CHX) ? 2 : (op==R_CWD||op==R_CWX) ? 4 : 8;
        for (i = 0; i < 16; i++) r.b[i] = 0x10 + i;
        int p = t & 15 & ~(sz - 1);
        if (sz == 1) r.b[p] = 3;
        else if (sz == 2) { r.b[p] = 2; r.b[p+1] = 3; }
        else for (i = 0; i < sz; i++) r.b[p+i] = i;
        break; }
    /* immediates */
    case R_IL: r = splat32((uint32_t)si16); break;
    case R_ILH: for (i = 0; i < 8; i++) sh_(&r, i, (uint16_t)i16); break;
    case R_ILHU: r = splat32(i16 << 16); break;
    case R_ILA: r = splat32(i18); break;
    case R_IOHL: LANE4(t | i16); break;
    /* memory */
    case R_LQD: case R_LQX: case R_LQA: case R_LQR: {
        uint32_t ad = op == R_LQD ? (aw0 + (uint32_t)(si10 * 16)) : op == R_LQX ? aw0 + bw0
                    : op == R_LQA ? (uint32_t)(si16 * 4) : pc + (uint32_t)(si16 * 4);
        ad &= 0x3FFF0; memcpy(r.b, ls + ad, 16); break; }
    case R_STQD: case R_STQX: case R_STQA: case R_STQR: {
        uint32_t ad = op == R_STQD ? (aw0 + (uint32_t)(si10 * 16)) : op == R_STQX ? aw0 + bw0
                    : op == R_STQA ? (uint32_t)(si16 * 4) : pc + (uint32_t)(si16 * 4);
        R.st = 1; R.st_addr = ad & 0x3FFF0; R.st_val = T; dest = -1; break; }
    /* branches */
    case R_BR: R.next = (pc + (uint32_t)(si16 * 4)) & 0x3FFFC; dest = -1; break;
    case R_BRA: R.next = (uint32_t)(si16 * 4) & 0x3FFFC; dest = -1; break;
    case R_BRSL: R.next = (pc + (uint32_t)(si16 * 4)) & 0x3FFFC; sw(&r, 0, (pc + 4) & 0x3FFFC); break;
    case R_BRASL: R.next = (uint32_t)(si16 * 4) & 0x3FFFC; sw(&r, 0, (pc + 4) & 0x3FFFC); break;
    case R_BRZ: dest = -1; if (W(T,0) == 0) R.next = (pc + (uint32_t)(si16 * 4)) & 0x3FFFC; break;
    case R_BRNZ: dest = -1; if (W(T,0) != 0) R.next = (pc + (uint32_t)(si16 * 4)) & 0x3FFFC; break;
    case R_BRHZ: dest = -1; if ((W(T,0) & 0xFFFF) == 0) R.next = (pc + (uint32_t)(si16 * 4)) & 0x3FFFC; break;
    case R_BRHNZ: dest = -1; if ((W(T,0) & 0xFFFF) != 0) R.next = (pc + (uint32_t)(si16 * 4)) & 0x3FFFC; break;
    case R_BI: R.next = aw0 & 0x3FFFC; dest = -1; break;
    case R_BISL: R.next = aw0 & 0x3FFFC; sw(&r, 0, (pc + 4) & 0x3FFFC); break;
    case R_IRET: R.next = srr0 & 0x3FFFC; dest = -1; break;
    case R_BISLED: sw(&r, 0, (pc + 4) & 0x3FFFC); if (evt) R.next = aw0 & 0x3FFFC; break;
    case R_BIZ: dest = -1; if (W(T,0) == 0) R.next = aw0 & 0x3FFFC; break;
    case R_BINZ: dest = -1; if (W(T,0) != 0) R.next = aw0 & 0x3FFFC; break;
    case R_BIHZ: dest = -1; if ((W(T,0) & 0xFFFF) == 0) R.next = aw0 & 0x3FFFC; break;
    case R_BIHNZ: dest = -1; if ((W(T,0) & 0xFFFF) != 0) R.next = aw0 & 0x3FFFC; break;
    /* single precision (tame operands; 1-ulp tolerance, see header) */
    case R_FA: R.fuzzy = 1; LANE4(u_of(f_of(a) + f_of(b))); break;
    case R_FS: R.fuzzy = 1; LANE4(u_of(f_of(a) - f_of(b))); break;
    case R_FM: R.fuzzy = 1; LANE4(u_of(f_of(a) * f_of(b))); break;
    case R_FMA: case R_FMS: case R_FNMS: { R.fuzzy = 1; dest = rt4; Q c = q_of(regs[rc]);
        for (i = 0; i < 4; i++) { double x = (double)f_of(W(A,i)) * (double)f_of(W(B,i)), cc = f_of(W(c,i));
            double v = op == R_FMA ? x + cc : op == R_FMS ? x - cc : cc - x; sw(&r, i, u_of((float)v)); } break; }
    case R_FCEQ: LANE4(f_of(a) == f_of(b) ? ~0u : 0); break;
    case R_FCGT: LANE4(f_of(a) > f_of(b) ? ~0u : 0); break;
    case R_FCMEQ: LANE4(fabsf(f_of(a)) == fabsf(f_of(b)) ? ~0u : 0); break;
    case R_FCMGT: LANE4(fabsf(f_of(a)) > fabsf(f_of(b)) ? ~0u : 0); break;
    case R_CFLTS: for (i = 0; i < 4; i++) { double v = trunc((double)f_of(W(A,i)) * ldexp(1.0, 173 - (int)i8));
            sw(&r, i, v >= 2147483648.0 ? 0x7FFFFFFFu : v <= -2147483649.0 ? 0x80000000u : (uint32_t)(int32_t)v); } break;
    case R_CFLTU: for (i = 0; i < 4; i++) { double v = trunc((double)f_of(W(A,i)) * ldexp(1.0, 173 - (int)i8));
            sw(&r, i, v >= 4294967296.0 ? 0xFFFFFFFFu : v <= 0 ? 0 : (uint32_t)v); } break;
    case R_CSFLT: R.fuzzy = 1; for (i = 0; i < 4; i++) sw(&r, i, u_of((float)((double)(int32_t)W(A,i) * ldexp(1.0, (int)i8 - 155)))); break;
    case R_CUFLT: R.fuzzy = 1; for (i = 0; i < 4; i++) sw(&r, i, u_of((float)((double)W(A,i) * ldexp(1.0, (int)i8 - 155)))); break;
    /* double precision (IEEE nearest-even, fused FMA) */
#define DBL(x) ({ uint64_t _u = qd(&x, d_); double _d; memcpy(&_d, &_u, 8); _d; })
#define DSET(v) do { double _v = (v); uint64_t _u; memcpy(&_u, &_v, 8); sd(&r, d_, _u); } while (0)
    case R_DFA: for (int d_ = 0; d_ < 2; d_++) DSET(DBL(A) + DBL(B)); break;
    case R_DFS: for (int d_ = 0; d_ < 2; d_++) DSET(DBL(A) - DBL(B)); break;
    case R_DFM: for (int d_ = 0; d_ < 2; d_++) DSET(DBL(A) * DBL(B)); break;
    case R_DFMA: for (int d_ = 0; d_ < 2; d_++) DSET(fma(DBL(A), DBL(B), DBL(T))); break;
    case R_DFMS: for (int d_ = 0; d_ < 2; d_++) DSET(fma(DBL(A), DBL(B), -DBL(T))); break;
    case R_DFNMS: for (int d_ = 0; d_ < 2; d_++) DSET(fma(-DBL(A), DBL(B), DBL(T))); break;
    case R_DFNMA: for (int d_ = 0; d_ < 2; d_++) DSET(-fma(DBL(A), DBL(B), DBL(T))); break;
    case R_DFCEQ: for (int d_ = 0; d_ < 2; d_++) sd(&r, d_, DBL(A) == DBL(B) ? ~0ull : 0); break;
    case R_DFCGT: for (int d_ = 0; d_ < 2; d_++) sd(&r, d_, DBL(A) > DBL(B) ? ~0ull : 0); break;
    case R_DFCMEQ: for (int d_ = 0; d_ < 2; d_++) sd(&r, d_, fabs(DBL(A)) == fabs(DBL(B)) ? ~0ull : 0); break;
    case R_DFCMGT: for (int d_ = 0; d_ < 2; d_++) sd(&r, d_, fabs(DBL(A)) > fabs(DBL(B)) ? ~0ull : 0); break;
    case R_FESD: for (int d_ = 0; d_ < 2; d_++) DSET((double)f_of(qw(&A, 2*d_))); break;
    case R_FRDS: for (int d_ = 0; d_ < 2; d_++) { uint64_t u = qd(&A, d_); double dd; memcpy(&dd, &u, 8); sw(&r, 2*d_, u_of((float)dd)); } break;
    default: R.next = ~0u; return R;      /* not covered by the reference: caller skips */
    }
    R.dest = dest; R.val = r;
    (void)C;
    return R;
}

/* ---- RNG / operand generation ---- */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static uint32_t special32(void) {
    static const uint32_t sp[] = { 0, ~0u, 0x80000000u, 0x7FFFFFFFu, 1, 0xFFFFu, 0x10000u, 0x8000u, 0xFFFF0000u, 0x7Fu, 0x80u, 0xFFu, 0x01010101u, 0x80808080u, 0x7F7F7F7Fu };
    switch (rnd() & 3) {
    case 0: return sp[rnd() % (sizeof sp / sizeof sp[0])];
    case 1: return rnd() % 70;
    case 2: return (uint32_t)(0 - (rnd() % 70));
    default: return 1u << (rnd() & 31);
    }
}
static uint32_t tame_f(void) {   /* normal float, exponent 110..140 */
    return (rnd() & 0x80000000u) | ((110u + rnd() % 31) << 23) | (rnd() & 0x7FFFFFu);
}
static void gen_regs(u128* g, int mode) {
    for (int r = 0; r < 128; r++) {
        for (int i = 0; i < 4; i++) {
            uint32_t v;
            if (mode == 1) v = tame_f();
            else if (mode == 2) { /* tame doubles across a word pair */
                static uint32_t hi; if (i & 1) v = rnd(); else { hi = ((rnd() & 0x80000000u) | ((1000u + rnd() % 48) << 20) | (rnd() & 0xFFFFFu)); v = hi; }
            }
            else switch (rnd() & 3) { case 0: v = rnd(); break; case 1: v = special32(); break;
                case 2: { uint32_t b = rnd() & 3; static const uint8_t bs[] = {0,1,0x7F,0x80,0xFF}; v = 0; for (int k = 0; k < 4; k++) { v = v << 8 | (b ? bs[rnd() % 5] : (rnd() & 0xFF)); } break; }
                default: v = (rnd() & 0xFFFF) | (special32() << 16); }
            g[r]._u32[i] = v;
        }
    }
}
static int is_float_op(int op) { switch (op) { case R_FA: case R_FS: case R_FM: case R_FMA: case R_FMS: case R_FNMS: case R_FCEQ: case R_FCGT:
    case R_FCMEQ: case R_FCMGT: case R_CFLTS: case R_CFLTU: case R_CSFLT: case R_CUFLT: case R_FESD: return 1; } return 0; }
static int is_double_op(int op) { switch (op) { case R_DFA: case R_DFS: case R_DFM: case R_DFMA: case R_DFMS: case R_DFNMS: case R_DFNMA:
    case R_DFCEQ: case R_DFCGT: case R_DFCMEQ: case R_DFCMGT: case R_FRDS: return 1; } return 0; }

static spu_context g_ctx;
static uint8_t g_ls0[0x40000];   /* pristine LS */
static const char* lname(int op, char* buf) { int i = 0; for (const char* s = r_name[op]; *s; s++) buf[i++] = (*s >= 'A' && *s <= 'Z') ? *s + 32 : *s; buf[i] = 0; return buf; }

static int ulp_close(uint32_t x, uint32_t y) {
    if (x == y) return 1;
    if (((x >> 23) & 0xFF) == 0xFF || ((y >> 23) & 0xFF) == 0xFF) return 1;   /* overflow/inf/NaN: SPU extended range differs from host, not compared */
    if (((x ^ y) & 0x80000000u) && ((x | y) & 0x7FFFFFFFu)) return 0;
    int32_t d = (int32_t)(x & 0x7FFFFFFFu) - (int32_t)(y & 0x7FFFFFFFu); return d >= -1 && d <= 1;
}

static long g_total_fail; static long g_total_trials; static int g_ops_tested, g_ops_failing;

static int run_op_trials(int op, long trials) {
    char nm[32]; lname(op, nm);
    long fails = 0, ran = 0; static u128 orig[128];
    int kind = (op == R_RDCH || op == R_WRCH || op == R_RCHCNT);
    for (long t = 0; t < trials; t++) {
        /* build insn: prefix for this op, random remaining bits */
        int ei = -1; for (int i = 0; i < R_NTAB; i++) if (r_tab[i].op == op) ei = i;
        int pbits = 11 - r_tab[ei].magn, lbits = 32 - pbits;
        uint32_t insn = (r_tab[ei].val << lbits) | (rnd() & ((lbits == 32) ? ~0u : ((1u << lbits) - 1)));
        if (ref_decode(insn) != op) { fprintf(stderr, "internal: table overlap for %s\n", nm); return 0; }
        /* register-field aliasing */
        if (rnd() % 3 == 0) { int a = rnd() % 5; insn = (insn & ~(0x7Fu << 7)) | (a << 7); if (rnd() & 1) insn = (insn & ~(0x7Fu << 14)) | (a << 14); if (rnd() & 1) insn = (insn & ~0x7Fu) | a; }
        int fmt_rrr = (op == R_SELB || op == R_SHUFB || op == R_MPYA || op == R_FMA || op == R_FMS || op == R_FNMS);
        if (fmt_rrr && rnd() % 3 == 0) { int a = rnd() % 5; insn = (insn & ~(0x7Fu << 21)) | (a << 21); if (rnd()&1) insn = (insn & ~0x7Fu) | (a); }
        uint32_t pc = (rnd() % 8 == 0) ? ((rnd() & 1) ? 0x3FFFC : 0) : ((rnd() & 0x3FFFC));
        gen_regs(orig, is_float_op(op) ? 1 : is_double_op(op) ? 2 : 0);
        if (rnd() % 4 == 0) { int ra = (insn >> 7) & 0x7F, rb = (insn >> 14) & 0x7F;
            orig[rb] = orig[ra]; if (rnd() & 1) { orig[rb]._u8[rnd() & 15] ^= (uint8_t)(rnd() | 1); } }
        uint32_t srr0 = rnd();
        uint32_t evt = rnd() & 1;
        g_ctx.event_mask = evt ? 0xFF : 0; g_ctx.event_status = 0xFF;
        g_ctx.srr0 = srr0;
        memcpy(g_ctx.gpr, orig, sizeof orig);
        /* put insn at pc */
        uint32_t m = pc & 0x3FFFC;
        g_ctx.ls[m] = insn >> 24; g_ctx.ls[m+1] = insn >> 16; g_ctx.ls[m+2] = insn >> 8; g_ctx.ls[m+3] = insn;
        { uint8_t save[4]; memcpy(save, g_ls0 + m, 4); memcpy(g_ls0 + m, g_ctx.ls + m, 4);   /* ref LS sees the insn too */
          g_ctx.pc = pc; g_ctx.status = SPU_STATUS_RUNNING; g_ctx.stop_code = 0;
          Ref R = ref_exec(op, insn, pc, orig, g_ls0, srr0, evt);
          memcpy(g_ls0 + m, save, 4);
          if (R.next == ~0u) { memcpy(g_ctx.ls + m, g_ls0 + m, 4); return -1; }  /* uncovered */
          /* channel stubs */
          g_ch_n = 0; g_ch_cnt = rnd(); for (int i = 0; i < 4; i++) g_ch_val._u32[i] = rnd();
          u128 chv = g_ch_val; uint32_t chc = g_ch_cnt;
          int halted = spu_step(&g_ctx);
          ran++;
          char why[200]; why[0] = 0; int bad = 0;
          /* destination */
          Q got; if (R.dest >= 0) {
              got = q_of(g_ctx.gpr[R.dest]);
              if (R.ch_op == 1) R.val = q_of(chv);
              if (R.ch_op == 3) { memset(&R.val, 0, 16); sw(&R.val, 0, chc); }
              if (R.fuzzy) { int ok = 1; for (int i = 0; i < 4; i++) { uint32_t g = qw(&got,i), e = qw(&R.val,i); if (!ulp_close(g, e)) ok = 0; }
                  if (!ok) { bad = 1; snprintf(why, sizeof why, "dest r%d got %08X_%08X_%08X_%08X exp %08X_%08X_%08X_%08X", R.dest,
                      qw(&got,0),qw(&got,1),qw(&got,2),qw(&got,3), qw(&R.val,0),qw(&R.val,1),qw(&R.val,2),qw(&R.val,3)); } }
              else if (memcmp(&got, &R.val, 16)) { bad = 1; snprintf(why, sizeof why, "dest r%d got %08X_%08X_%08X_%08X exp %08X_%08X_%08X_%08X", R.dest,
                      qw(&got,0),qw(&got,1),qw(&got,2),qw(&got,3), qw(&R.val,0),qw(&R.val,1),qw(&R.val,2),qw(&R.val,3)); }
          }
          /* untouched registers */
          if (!bad) { u128 tmp[128]; memcpy(tmp, g_ctx.gpr, sizeof tmp); if (R.dest >= 0) tmp[R.dest] = orig[R.dest];
              if (memcmp(tmp, orig, sizeof tmp)) { bad = 1; for (int r = 0; r < 128; r++) if (memcmp(&tmp[r], &orig[r], 16)) { snprintf(why, sizeof why, "clobbered r%d (dest r%d)", r, R.dest); break; } } }
          /* store / pc / halt / stop / channel */
          if (!bad && R.st) { if (memcmp(g_ctx.ls + R.st_addr, R.st_val.b, 16)) { bad = 1; snprintf(why, sizeof why, "store @%05X mismatch", R.st_addr); } }
          if (!bad && !R.st) { /* LS untouched outside the insn word: sample the quad at the insn's neighbours */ }
          if (!bad) {
              if (R.stop) { if (!halted || g_ctx.status != SPU_STATUS_STOPPED_BY_STOP || g_ctx.stop_code != R.stop_code) { bad = 1; snprintf(why, sizeof why, "stop: halted=%d status=%X code=%X exp %X", halted, g_ctx.status, g_ctx.stop_code, R.stop_code); } }
              else if (R.halt) { if (!halted) { bad = 1; snprintf(why, sizeof why, "expected halt"); } }
              else if (halted) { bad = 1; snprintf(why, sizeof why, "unexpected halt (status %X)", g_ctx.status); }
              else if (((uint32_t)g_ctx.pc & LSM) != R.next) { bad = 1; snprintf(why, sizeof why, "pc %05X exp %05X", (uint32_t)g_ctx.pc, R.next); }
          }
          if (!bad && R.ch_op) {
              if (g_ch_n != 1 || g_ch_last != R.ch_no) { bad = 1; snprintf(why, sizeof why, "channel %u calls=%u exp ch %u", g_ch_last, g_ch_n, R.ch_no); }
              else if (R.ch_op == 2) { Q w = q_of(g_ch_val); Q e = q_of(orig[insn & 0x7F]); if (memcmp(&w, &e, 16)) { bad = 1; snprintf(why, sizeof why, "wrch value"); } }
          } else if (!bad && R.ch_op == 0 && g_ch_n) { bad = 1; snprintf(why, sizeof why, "unexpected channel access"); }
          if (bad) { fails++; if (fails <= 3) printf("  MISMATCH %s insn=%08X pc=%05X: %s\n", nm, insn, pc, why); }
          /* restore LS from pristine (instruction word + any stored quad) */
          memcpy(g_ctx.ls + m, g_ls0 + m, 4);
          if (R.st) memcpy(g_ctx.ls + R.st_addr, g_ls0 + R.st_addr, 16);
          if (m < 0 || (t & 255) == 0) { if (memcmp(g_ctx.ls, g_ls0, sizeof g_ls0)) { printf("  LS corrupted outside the store target by %s insn=%08X\n", nm, insn); memcpy(g_ctx.ls, g_ls0, sizeof g_ls0); fails++; } }
        }
    }
    (void)kind;
    g_total_trials += ran; g_total_fail += fails; g_ops_tested++;
    if (fails) { g_ops_failing++; printf("FAIL %-10s %ld / %ld mismatches\n", nm, fails, ran); }
    return fails != 0;
}

/* ---- decode cross-check ---- */
static long decode_check(void) {
    long bad = 0, n = 0; spu_ins d;
    for (uint32_t p = 0; p < 2048; p++) for (int k = 0; k < 64; k++) {
        uint32_t insn = (p << 21) | (rnd() & 0x1FFFFF);
        if (k == 0) insn = p << 21; if (k == 1) insn = (p << 21) | 0x1FFFFF;
        spu_decode1(insn, 0, &d); n++;
        int rop = ref_decode(insn);
        const char* have = d.op == SPU_word ? "?" : spu_op_name[d.op]; char nm[32]; lname(rop, nm);
        int match = (rop == R_UNK) ? (d.op == SPU_word) : (d.op != SPU_word && !strcmp(have, nm));
        if (!match) { if (bad < 12) printf("  DECODE %08X: interp '%s' ref '%s'\n", insn, have, rop == R_UNK ? "?" : nm); bad++; }
    }
    for (long i = 0; i < 2000000; i++) { uint32_t insn = rnd() ^ (rnd() << 16); spu_decode1(insn, 0, &d); n++;
        int rop = ref_decode(insn); const char* have = d.op == SPU_word ? "?" : spu_op_name[d.op]; char nm[32]; lname(rop, nm);
        int match = (rop == R_UNK) ? (d.op == SPU_word) : (d.op != SPU_word && !strcmp(have, nm));
        if (!match) { if (bad < 12) printf("  DECODE %08X: interp '%s' ref '%s'\n", insn, have, rop == R_UNK ? "?" : nm); bad++; } }
    printf("decode cross-check: %ld words, %ld mismatches\n", n, bad);
    return bad;
}

int main(int argc, char** argv) {
    long trials = argc > 1 ? atol(argv[1]) : 20000;
    if (argc > 2) g_rng ^= (uint64_t)atoll(argv[2]) * 0x2545F4914F6CDD1Dull;
    spu_context_init(&g_ctx, 0);
    for (int i = 0; i < 0x40000; i++) g_ctx.ls[i] = (uint8_t)rnd();
    memcpy(g_ls0, g_ctx.ls, sizeof g_ls0);
    long dbad = decode_check();
    int skipped = 0; char skipnames[2048] = "";
    for (int op = 1; op < R_COUNT; op++) {
        int r = run_op_trials(op, trials);
        if (r < 0) { skipped++; char nm[32]; lname(op, nm); strcat(skipnames, nm); strcat(skipnames, " "); g_ops_tested += 0; }
    }
    printf("ops executed: %d (+%d not covered by reference: %s)\n", g_ops_tested, skipped, skipnames);
    printf("trials: %ld, execution mismatches: %ld in %d ops; decode mismatches: %ld\n", g_total_trials, g_total_fail, g_ops_failing, dbad);
    int ok = !g_total_fail && !dbad;
    printf("spu_interp_diff_test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
