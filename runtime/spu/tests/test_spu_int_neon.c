/* The NEON integer/permute helpers must be bit-identical to their scalar
 * references (spu_*_ref) for random operands and every shift amount. Build on
 * arm64:  cc -O2 -I.. -I../.. -I../../.. test_spu_int_neon.c -o t && ./t */
#include "../spu_helpers.h"
#include <stdio.h>

static uint64_t s_rng = 0x2545F4914F6CDD1Dull;
static uint32_t rnd32(void)
{
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return (uint32_t)(s_rng >> 16);
}
static u128 rnd128(void) { u128 r; for (int i = 0; i < 4; i++) r._u32[i] = rnd32(); return r; }

static long s_bad;
static void check(const char* op, u128 got, u128 ref, u128 a, u128 b, int imm)
{
    if (!memcmp(&got, &ref, 16)) return;
    if (s_bad++ < 12)
        printf("MISMATCH %s a=%08X%08X%08X%08X b=%08X%08X%08X%08X imm=%d got=%08X%08X%08X%08X ref=%08X%08X%08X%08X\n",
               op, a._u32[0], a._u32[1], a._u32[2], a._u32[3], b._u32[0], b._u32[1], b._u32[2], b._u32[3], imm,
               got._u32[0], got._u32[1], got._u32[2], got._u32[3], ref._u32[0], ref._u32[1], ref._u32[2], ref._u32[3]);
}

int main(void)
{
#if !defined(__ARM_NEON)
    puts("test_spu_int_neon: no NEON, nothing to compare");
    return 0;
#else
    const long N = 200000;
    for (long it = 0; it < N; it++) {
        u128 a = rnd128(), b = rnd128(), c = rnd128();
        if (it & 1) b._u32[0] = rnd32() & 0xFF;          /* small shift counts */
        if (it % 5 == 0) b = a;                           /* equal operands for compares */
        int imm = (int)(rnd32() & 0x3FF) - 0x200;         /* 10-bit signed immediate */
        if (it % 3 == 0) imm = (int32_t)a._u32[rnd32() & 3];
        int sh = (int)(it % 64) - 16;
#define C2(op) check(#op, spu_##op(a, b), spu_##op##_ref(a, b), a, b, 0)
#define CI(op, v) check(#op, spu_##op(a, v), spu_##op##_ref(a, v), a, b, v)
        C2(a); C2(sf); C2(and); C2(or); C2(xor); C2(ceq); C2(cgt); C2(clgt);
        CI(ai, imm); CI(sfi, imm); CI(andi, imm); CI(ori, imm); CI(xori, imm);
        CI(ceqi, imm); CI(cgti, imm); CI(clgti, imm);
        check("selb", spu_selb(a, b, c), spu_selb_ref(a, b, c), a, b, 0);
        CI(shlqbyi, sh); CI(rotqbyi, sh); CI(rotqmbyi, sh); CI(cwd, sh);
        C2(shlqby); C2(rotqby); C2(shlqbybi); C2(rotqbybi); C2(rotqmby); C2(rotqmbybi);
    }
    if (s_bad) { printf("test_spu_int_neon: %ld mismatches\n", s_bad); return 1; }
    printf("test_spu_int_neon: ok (%ld random cases x 27 ops)\n", N);
    return 0;
#endif
}
