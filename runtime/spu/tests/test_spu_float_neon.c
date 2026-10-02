/* The NEON spu_fa/fs/fm/fma/fms/fnms must be bit-identical to the scalar
 * references (spu_*_ref) for every input: random words, ordinary floats, the
 * normal/extended-range boundaries, zeros and denormals, mixed per lane.
 * Prints a rough speed comparison too. Build on arm64:
 *   cc -O2 -I.. -I../.. -I../../.. test_spu_float_neon.c -o t && ./t */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t s_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void)
{
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return (uint32_t)(s_rng >> 16);
}

static uint32_t rnd_float(void)
{
    uint32_t sign = (rnd32() & 1u) << 31, man = rnd32() & 0x7FFFFFu, e;
    switch (rnd32() % 10) {
    case 0: e = 0; break;                              /* zero / denormal */
    case 1: e = 1 + rnd32() % 4; break;                /* bottom of normal range */
    case 2: e = 248 + rnd32() % 8; break;              /* top, into extended range */
    case 3: return rnd32();                            /* anything */
    case 4: return sign;                               /* +-0 */
    default: e = 110 + rnd32() % 36; break;            /* ordinary */
    }
    return sign | (e << 23) | man;
}

int main(void)
{
#if !defined(__ARM_NEON)
    puts("test_spu_float_neon: no NEON, nothing to compare");
    return 0;
#else
    long bad = 0;
    const long N = 4000000;
    for (long it = 0; it < N; it++) {
        u128 a, b, c, got, ref;
        for (int i = 0; i < 4; i++) { a._u32[i] = rnd_float(); b._u32[i] = rnd_float(); c._u32[i] = rnd_float(); }
        /* close magnitudes stress cancellation in fa/fs and fma */
        if (it % 7 == 0) for (int i = 0; i < 4; i++) b._u32[i] = (a._u32[i] ^ 0x80000000u) + (rnd32() & 0xFF);
        const int op = (int)(it % 6);
        switch (op) {
        case 0: got = spu_fa(a, b);      ref = spu_fa_ref(a, b); break;
        case 1: got = spu_fs(a, b);      ref = spu_fs_ref(a, b); break;
        case 2: got = spu_fm(a, b);      ref = spu_fm_ref(a, b); break;
        case 3: got = spu_fma(a, b, c);  ref = spu_fma_ref(a, b, c); break;
        case 4: got = spu_fms(a, b, c);  ref = spu_fms_ref(a, b, c); break;
        default: got = spu_fnms(a, b, c); ref = spu_fnms_ref(a, b, c); break;
        }
        if (memcmp(&got, &ref, 16)) {
            if (bad++ < 10)
                printf("MISMATCH op=%d a=%08X %08X %08X %08X b=%08X %08X %08X %08X c=%08X %08X %08X %08X\n"
                       "  got=%08X %08X %08X %08X ref=%08X %08X %08X %08X\n", op,
                       a._u32[0], a._u32[1], a._u32[2], a._u32[3], b._u32[0], b._u32[1], b._u32[2], b._u32[3],
                       c._u32[0], c._u32[1], c._u32[2], c._u32[3],
                       got._u32[0], got._u32[1], got._u32[2], got._u32[3],
                       ref._u32[0], ref._u32[1], ref._u32[2], ref._u32[3]);
        }
    }
    if (bad) { printf("test_spu_float_neon: %ld mismatches in %ld cases\n", bad, N); return 1; }
    printf("test_spu_float_neon: ok (%ld random cases)\n", N);

    /* speed on ordinary operands that stay ordinary */
    static u128 in[256][3];
    for (int k = 0; k < 256; k++)
        for (int i = 0; i < 4; i++) {
            in[k][0]._u32[i] = (rnd32() & 0x807FFFFFu) | ((120u + rnd32() % 16) << 23);
            in[k][1]._u32[i] = (rnd32() & 0x807FFFFFu) | ((120u + rnd32() % 16) << 23);
            in[k][2]._u32[i] = (rnd32() & 0x807FFFFFu) | ((120u + rnd32() % 16) << 23);
        }
    const int R = 20000000;
    uint32_t acc = 0;
    clock_t t0 = clock();
    for (int i = 0; i < R; i++) { const u128* q = in[i & 255];
        u128 r1 = spu_fma(q[0], q[1], q[2]), r2 = spu_fm(q[0], q[1]); acc += r1._u32[i & 3] ^ r2._u32[(i + 1) & 3]; }
    clock_t t1 = clock();
    for (int i = 0; i < R; i++) { const u128* q = in[i & 255];
        u128 r1 = spu_fma_ref(q[0], q[1], q[2]), r2 = spu_fm_ref(q[0], q[1]); acc += r1._u32[i & 3] ^ r2._u32[(i + 1) & 3]; }
    clock_t t2 = clock();
    printf("fma+fm pair: neon %.1f ns, scalar %.1f ns (acc %u)\n",
           (t1 - t0) * 1e9 / CLOCKS_PER_SEC / R, (t2 - t1) * 1e9 / CLOCKS_PER_SEC / R, acc);
    return 0;
#endif
}
