/* e2_gemm.c - int8 quantized GEMM with runtime CPU dispatch.
 *
 * Weights: symmetric int8, one scale per output row.  Packed layout is
 *   [N/16][K/4][16 rows][4 k]
 * so the 512 bytes for one 16-output tile are contiguous over K.
 *
 * Activations: asymmetric u8, zero point 128, one scale per 128-element
 * block.  Block scales keep the relative error low when a row contains an
 * outlier (the embedding models have many), at the cost of one f32 FMA per
 * output per block.  The weight sum used for the zero-point correction is
 * precomputed per block.
 *
 * Primary kernels: AVX-512 VNNI and AVX-VNNI (vpdpbusd).  One instruction
 * computes 16 (AVX-512) or 8 (AVX-VNNI) output channels from 4 k values.
 */
#include "e2_internal.h"
#include "e2_gpu.h"

#include <math.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define E2_X86 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define E2_ARM 1
#endif

#define BLK 128

typedef void (*gemm_kernel)(float *y, int nstride, const uint8_t *xq,
                            const float *xs, int xs_stride, int m, int k,
                            const e2_qmat *w, int nb0, int nb1);

/* ------------------------------------------------------------------ */
/* scalar                                                             */
/* ------------------------------------------------------------------ */

static void gemm_scalar(float *y, int ns, const uint8_t *xq, const float *xs,
                        int xss, int m, int k, const e2_qmat *w, int nb0, int nb1) {
    int k4 = k / 4, nblk = k / BLK;
    for (int nb = nb0; nb < nb1; nb++) {
        int n0 = nb * 16;
        int cnt = w->n - n0; if (cnt > 16) cnt = 16;
        const int8_t *wp = w->q + (size_t)nb * k4 * 64;
        for (int r = 0; r < m; r++) {
            const uint8_t *ap = xq + (size_t)r * k;
            float *yp = y + (size_t)r * ns + n0;
            for (int b = 0; b < nblk; b++) {
                int32_t acc[16];
                for (int t = 0; t < cnt; t++) acc[t] = -128 * w->wsum[(size_t)(n0 + t) * nblk + b];
                for (int kb = b * (BLK / 4); kb < (b + 1) * (BLK / 4); kb++) {
                    const int8_t *wv = wp + (size_t)kb * 64;
                    uint32_t a4;
                    memcpy(&a4, ap + kb * 4, 4);
                    unsigned char a0 = a4, a1 = a4 >> 8, a2 = a4 >> 16, a3 = a4 >> 24;
                    for (int t = 0; t < cnt; t++) {
                        const int8_t *q = wv + t * 4;
                        acc[t] += a0 * q[0] + a1 * q[1] + a2 * q[2] + a3 * q[3];
                    }
                }
                float s = xs[(size_t)r * xss + b];
                if (b == 0)
                    for (int t = 0; t < cnt; t++) yp[t] = (float)acc[t] * s * w->scale[n0 + t];
                else
                    for (int t = 0; t < cnt; t++) yp[t] += (float)acc[t] * s * w->scale[n0 + t];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* AVX-512 VNNI                                                       */
/* ------------------------------------------------------------------ */

#ifdef E2_X86
/* Generic 16-output-tile kernel: 4 rows x 1 tile.  Used for the leftover
 * tiles when n is not a multiple of 64. */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
static void gemm_t1(float *y, int ns, const uint8_t *xq, const float *xs,
                    int xss, int m, int k, const e2_qmat *w, int nb0, int nb1) {
    int k4 = k / 4, blk = w->blk ? w->blk : BLK, nblk = k / blk;
    const __m512i k128 = _mm512_set1_epi32(128);
    size_t ts = w->tstride ? w->tstride : (size_t)k4 * 64;
    int wss = w->wsmstride ? w->wsmstride : nblk;
    for (int nb = nb0; nb < nb1; nb++) {
        int n0 = nb * 16;
        const int8_t *wp = w->q + (size_t)nb * ts;
        __m512 wscale = _mm512_loadu_ps(w->scale + n0);
        for (int b = 0; b < nblk; b++) {
            int32_t wsbuf[16];
            for (int t = 0; t < 16; t++) wsbuf[t] = w->wsum[(size_t)(n0 + t) * wss + b];
            __m512i base = _mm512_loadu_si512((const void *)wsbuf);
            base = _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_mullo_epi32(base, k128));
            const int8_t *wq = wp + (size_t)b * (blk / 4) * 64;
            for (int r0 = 0; r0 < m; r0 += 4) {
                int mm = m - r0; if (mm > 4) mm = 4;
                const uint8_t *ap = xq + (size_t)r0 * k + (size_t)b * blk;
                __m512i a0 = base, a1 = base, a2 = base, a3 = base;
                for (int kb = 0; kb < blk / 4; kb++) {
                    __m512i wv = _mm512_loadu_si512((const void *)(wq + (size_t)kb * 64));
                    a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(*(const int *)(ap + kb * 4)), wv);
                    a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32(*(const int *)(ap + k + kb * 4)), wv);
                    a2 = _mm512_dpbusd_epi32(a2, _mm512_set1_epi32(*(const int *)(ap + 2 * k + kb * 4)), wv);
                    a3 = _mm512_dpbusd_epi32(a3, _mm512_set1_epi32(*(const int *)(ap + 3 * k + kb * 4)), wv);
                }
                __m512i R[4] = {a0, a1, a2, a3};
                for (int i = 0; i < mm; i++) {
                    float *yp = y + (size_t)(r0 + i) * ns + n0;
                    __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(R[i]),
                                             _mm512_mul_ps(wscale, _mm512_set1_ps(xs[(size_t)(r0 + i) * xss + b])));
                    if (b) f = _mm512_add_ps(f, _mm512_loadu_ps(yp));
                    _mm512_storeu_ps(yp, f);
                }
            }
        }
    }
}

/* Weight prefetch distance in packed 64-byte rows.  When m is small the GEMM
 * is DRAM-latency bound on the weight stream, so the distance is the knob that
 * matters; it costs nothing when the weights are cached. */
static int g_pf = 48;

/* Row block of the AVX-VNNI kernel: the f32 accumulator tile for RB rows of
 * 16 outputs stays in L1 (RB=32 -> 2 KB) and so does the activation slab
 * (RB x k bytes).  32 measured better than 64/128/256 once the accumulator
 * moved out of the output buffer. */
static int g_rb = 32;      /* AVX-VNNI kernel default */
static int g_rb512 = 64;   /* AVX-512 kernel default: it has no f32 tile to keep
                            * in L1, and 32 measured 10-17% worse there */

/* Fast path: 4 rows x 4 output tiles (64 columns).  24 uops per 16 vpdpbusd,
 * the minimum for a 16-accumulator budget. */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
static void gemm_avx512(float *y, int ns, const uint8_t *xq, const float *xs,
                        int xss, int m, int k, const e2_qmat *w, int nb0, int nb1) {
    int k4 = k / 4, blk = w->blk ? w->blk : BLK, nblk = k / blk;
    const __m512i k128 = _mm512_set1_epi32(128);
    size_t ts = w->tstride ? w->tstride : (size_t)k4 * 64;
    int wss = w->wsmstride ? w->wsmstride : nblk;
    int nb = nb0;
    /* Row block outermost: a 64-row activation slab stays in L1 while every
     * output-tile group is swept.  The old order re-read the whole activation
     * matrix once per 64-column tile group, which dominated at large m. */
    int RB = g_rb512;
    int nb4 = nb0 + ((nb1 - nb0) / 4) * 4;
    for (int rb = 0; rb < m; rb += RB) {
        int rend = rb + RB; if (rend > m) rend = m;
        for (nb = nb0; nb < nb4; nb += 4) {
        int n0 = nb * 16;
        const int8_t *wp = w->q + (size_t)nb * ts;
        __m512 WSC4[4];
        for (int t2 = 0; t2 < 4; t2++) WSC4[t2] = _mm512_loadu_ps(w->scale + n0 + 16 * t2);
        for (int b = 0; b < nblk; b++) {
            __m512i base[4];
            for (int t2 = 0; t2 < 4; t2++) {
                int32_t wsbuf[16];
                for (int t = 0; t < 16; t++)
                    wsbuf[t] = w->wsum[(size_t)(n0 + 16 * t2 + t) * wss + b];
                __m512i v = _mm512_loadu_si512((const void *)wsbuf);
                base[t2] = _mm512_sub_epi32(_mm512_setzero_si512(),
                                            _mm512_mullo_epi32(v, k128));
            }
            size_t kboff = (size_t)b * (blk / 4) * 64;
            const int8_t *wq0 = wp + kboff;
            const int8_t *wq1 = wq0 + ts;
            const int8_t *wq2 = wq1 + ts;
            const int8_t *wq3 = wq2 + ts;
            for (int r0 = rb; r0 < rend; r0 += 4) {
                int mm = rend - r0; if (mm > 4) mm = 4;
                const uint8_t *ap = xq + (size_t)r0 * k + (size_t)b * blk;
                __m512i p00 = base[0], p01 = base[0], p02 = base[0], p03 = base[0];
                __m512i p10 = base[1], p11 = base[1], p12 = base[1], p13 = base[1];
                __m512i p20 = base[2], p21 = base[2], p22 = base[2], p23 = base[2];
                __m512i p30 = base[3], p31 = base[3], p32 = base[3], p33 = base[3];
                for (int kb = 0; kb < blk / 4; kb++) {
                    const uint8_t *a = ap + kb * 4;
                    __m512i a0 = _mm512_set1_epi32(*(const int *)(a));
                    __m512i a1 = _mm512_set1_epi32(*(const int *)(a + k));
                    __m512i a2 = _mm512_set1_epi32(*(const int *)(a + 2 * k));
                    __m512i a3 = _mm512_set1_epi32(*(const int *)(a + 3 * k));
                    __m512i w0 = _mm512_loadu_si512((const void *)(wq0 + (size_t)kb * 64));
                    __m512i w1 = _mm512_loadu_si512((const void *)(wq1 + (size_t)kb * 64));
                    __m512i w2 = _mm512_loadu_si512((const void *)(wq2 + (size_t)kb * 64));
                    __m512i w3 = _mm512_loadu_si512((const void *)(wq3 + (size_t)kb * 64));
                    p00 = _mm512_dpbusd_epi32(p00, a0, w0); p01 = _mm512_dpbusd_epi32(p01, a1, w0);
                    p02 = _mm512_dpbusd_epi32(p02, a2, w0); p03 = _mm512_dpbusd_epi32(p03, a3, w0);
                    p10 = _mm512_dpbusd_epi32(p10, a0, w1); p11 = _mm512_dpbusd_epi32(p11, a1, w1);
                    p12 = _mm512_dpbusd_epi32(p12, a2, w1); p13 = _mm512_dpbusd_epi32(p13, a3, w1);
                    p20 = _mm512_dpbusd_epi32(p20, a0, w2); p21 = _mm512_dpbusd_epi32(p21, a1, w2);
                    p22 = _mm512_dpbusd_epi32(p22, a2, w2); p23 = _mm512_dpbusd_epi32(p23, a3, w2);
                    p30 = _mm512_dpbusd_epi32(p30, a0, w3); p31 = _mm512_dpbusd_epi32(p31, a1, w3);
                    p32 = _mm512_dpbusd_epi32(p32, a2, w3); p33 = _mm512_dpbusd_epi32(p33, a3, w3);
                }
                __m512i R[4][4] = {{p00, p01, p02, p03}, {p10, p11, p12, p13},
                                   {p20, p21, p22, p23}, {p30, p31, p32, p33}};
                for (int i = 0; i < mm; i++) {
                    float sc = xs[(size_t)(r0 + i) * xss + b];
                    float *yp = y + (size_t)(r0 + i) * ns + n0;
                    __m512 vs = _mm512_set1_ps(sc);
                    for (int t2 = 0; t2 < 4; t2++) {
                        __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(R[t2][i]),
                                                 _mm512_mul_ps(WSC4[t2], vs));
                        if (b) f = _mm512_add_ps(f, _mm512_loadu_ps(yp + 16 * t2));
                        _mm512_storeu_ps(yp + 16 * t2, f);
                    }
                }
            }
        }
        }
    }
    if (nb4 < nb1) gemm_t1(y, ns, xq, xs, xss, m, k, w, nb4, nb1);
}
#endif

/* ------------------------------------------------------------------ */
/* AVX2: 16 rows x 4 k widened to i16 (vpmaddubsw would saturate: a u8 x i8
 * pair sum reaches 64k > i16).  The two `vpmaddwd` partials a row produces
 * stay interleaved in one ymm and are summed once per K block, so the inner
 * loop is 4 loads+4 madd+4 add for 64 MACs. */
#ifdef E2_X86
__attribute__((target("avx2")))
static void gemm_avx2(float *y, int ns, const uint8_t *xq, const float *xs,
                      int xss, int m, int k, const e2_qmat *w, int nb0, int nb1) {
    int blk = w->blk ? w->blk : BLK, nblk = k / blk;
    size_t ts = w->tstride ? w->tstride : (size_t)(k / 4) * 64;
    int wss = w->wsmstride ? w->wsmstride : nblk;
    for (int nb = nb0; nb < nb1; nb++) {
        int n0 = nb * 16;
        int cnt = w->n - n0; if (cnt > 16) cnt = 16;
        const int8_t *wp = w->q + (size_t)nb * ts;
        __m256 ws0 = _mm256_loadu_ps(w->scale + n0);
        __m256 ws1 = _mm256_loadu_ps(w->scale + n0 + 8);
        for (int b = 0; b < nblk; b++) {
            int32_t wsbuf[16];
            for (int t = 0; t < 16; t++) wsbuf[t] = -128 * w->wsum[(size_t)(n0 + t) * wss + b];
            const int8_t *wq = wp + (size_t)b * (blk / 4) * 64;
            for (int r = 0; r < m; r++) {
                const uint8_t *ap = xq + (size_t)r * k + (size_t)b * blk;
                __m256i acc[4];
                memset(acc, 0, sizeof(acc));
                for (int kb = 0; kb < blk / 4; kb++) {
                    int a4;
                    memcpy(&a4, ap + (size_t)kb * 4, 4);
                    __m256i a16 = _mm256_cvtepu8_epi16(
                        _mm256_castsi256_si128(_mm256_set1_epi32(a4)));
                    const int8_t *wv = wq + (size_t)kb * 64;
                    for (int g = 0; g < 4; g++) {
                        __m256i w16 = _mm256_cvtepi8_epi16(
                            _mm_loadu_si128((const __m128i *)(wv + g * 16)));
                        acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(w16, a16));
                    }
                }
                int32_t part[32];
                for (int g = 0; g < 4; g++)
                    _mm256_storeu_si256((__m256i *)(part + g * 8), acc[g]);
                int32_t sum[16];
                for (int t = 0; t < 16; t++) sum[t] = part[2 * t] + part[2 * t + 1] + wsbuf[t];
                __m256 s0 = _mm256_set1_ps(xs[(size_t)r * xss + b]);
                __m256 f0 = _mm256_mul_ps(_mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)sum)), s0), ws0);
                __m256 f1 = _mm256_mul_ps(_mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(sum + 8))), s0), ws1);
                float out[16];
                _mm256_storeu_ps(out, f0);
                _mm256_storeu_ps(out + 8, f1);
                float *yp = y + (size_t)r * ns + n0;
                if (b == 0)
                    for (int t = 0; t < cnt; t++) yp[t] = out[t];
                else
                    for (int t = 0; t < cnt; t++) yp[t] += out[t];
            }
        }
    }
}
#endif

/* AVX-VNNI (Alder Lake / Arrow Lake / Zen 5 have the 256-bit VEX form):
 * 4 rows x 1 tile, 8 accumulators, ~18 MAC/uop. */
#ifdef E2_X86
/* AVX-VNNI (Alder Lake / Arrow Lake / Zen 4/5 have the 256-bit VEX form):
 * 4 rows x 1 output tile (16 columns), 8 integer accumulators.
 *
 * The f32 accumulators for a whole row block live in a small stack buffer that
 * stays in L1, so each output element is written once instead of
 * `nblk` times (read-modify-write per 128-wide k block).  For k=512/n=2048/m=2048
 * that is 112 MB of output traffic per GEMM turned into 16 MB. */
#define VNNI_ACC_ROWS 128
__attribute__((target("avx2,avxvnni")))
static void gemm_avxvnni(float *y, int ns, const uint8_t *xq, const float *xs,
                         int xss, int m, int k, const e2_qmat *w, int nb0, int nb1) {
    int blk = w->blk ? w->blk : BLK, nblk = k / blk;
    size_t ts = w->tstride ? w->tstride : (size_t)(k / 4) * 64;
    int wss = w->wsmstride ? w->wsmstride : nblk;
    const __m256i k128 = _mm256_set1_epi32(128);
    int RB = g_rb;
    if (RB > VNNI_ACC_ROWS) RB = VNNI_ACC_ROWS;
    for (int rb = 0; rb < m; rb += RB) {
        int rend = rb + RB; if (rend > m) rend = m;
        for (int nb = nb0; nb < nb1; nb++) {
            int n0 = nb * 16;
            int cnt = w->n - n0; if (cnt > 16) cnt = 16;
            const int8_t *wp = w->q + (size_t)nb * ts;
            __m256 wscale0 = _mm256_loadu_ps(w->scale + n0);
            __m256 wscale1 = _mm256_loadu_ps(w->scale + n0 + 8);
            float acc[RB][16];
            memset(acc, 0, (size_t)(rend - rb) * sizeof(acc[0]));
            for (int b = 0; b < nblk; b++) {
                int32_t wsbuf[16];
                for (int t = 0; t < 16; t++) wsbuf[t] = w->wsum[(size_t)(n0 + t) * wss + b];
                __m256i base0 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_mullo_epi32(
                    _mm256_loadu_si256((const __m256i *)wsbuf), k128));
                __m256i base1 = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_mullo_epi32(
                    _mm256_loadu_si256((const __m256i *)(wsbuf + 8)), k128));
                const int8_t *wq = wp + (size_t)b * (blk / 4) * 64;
                for (int r0 = rb; r0 < rend; r0 += 4) {
                    int mm = rend - r0; if (mm > 4) mm = 4;
                    const uint8_t *ap = xq + (size_t)r0 * k + (size_t)b * blk;
                    __m256i a0l = base0, a0h = base1, a1l = base0, a1h = base1;
                    __m256i a2l = base0, a2h = base1, a3l = base0, a3h = base1;
                    for (int kb = 0; kb < blk / 4; kb++) {
                        /* letting dpbusd read the weight from memory (gcc still
                         * CSEs the two uses) measures ~3% faster than explicit
                         * locals.  The 3 KB run-ahead covers DRAM latency when
                         * m is small enough that the weight stream is the limit
                         * (m=8 cold: 0.105 -> 0.062 ms) and costs nothing when
                         * the weights are cached. */
                        const __m256i *wv = (const __m256i *)(wq + (size_t)kb * 64);
                        __builtin_prefetch((const char *)wv + g_pf * 64, 0, 3);
                        int x0, x1, x2, x3;
                        memcpy(&x0, ap + (size_t)kb * 4, 4);
                        memcpy(&x1, ap + k + (size_t)kb * 4, 4);
                        memcpy(&x2, ap + 2 * k + (size_t)kb * 4, 4);
                        memcpy(&x3, ap + 3 * k + (size_t)kb * 4, 4);
                        __m256i v;
                        v = _mm256_set1_epi32(x0);
                        a0l = _mm256_dpbusd_epi32(a0l, v, _mm256_loadu_si256(wv));
                        a0h = _mm256_dpbusd_epi32(a0h, v, _mm256_loadu_si256(wv + 1));
                        v = _mm256_set1_epi32(x1);
                        a1l = _mm256_dpbusd_epi32(a1l, v, _mm256_loadu_si256(wv));
                        a1h = _mm256_dpbusd_epi32(a1h, v, _mm256_loadu_si256(wv + 1));
                        v = _mm256_set1_epi32(x2);
                        a2l = _mm256_dpbusd_epi32(a2l, v, _mm256_loadu_si256(wv));
                        a2h = _mm256_dpbusd_epi32(a2h, v, _mm256_loadu_si256(wv + 1));
                        v = _mm256_set1_epi32(x3);
                        a3l = _mm256_dpbusd_epi32(a3l, v, _mm256_loadu_si256(wv));
                        a3h = _mm256_dpbusd_epi32(a3h, v, _mm256_loadu_si256(wv + 1));
                    }
                    __m256i R[4][2] = {{a0l, a0h}, {a1l, a1h}, {a2l, a2h}, {a3l, a3h}};
                    for (int i = 0; i < mm; i++) {
                        __m256 s = _mm256_set1_ps(xs[(size_t)(r0 + i) * xss + b]);
                        __m256 f0 = _mm256_mul_ps(_mm256_mul_ps(
                            _mm256_cvtepi32_ps(R[i][0]), s), wscale0);
                        __m256 f1 = _mm256_mul_ps(_mm256_mul_ps(
                            _mm256_cvtepi32_ps(R[i][1]), s), wscale1);
                        float *ap2 = acc[r0 + i - rb];
                        _mm256_storeu_ps(ap2, _mm256_add_ps(_mm256_loadu_ps(ap2), f0));
                        _mm256_storeu_ps(ap2 + 8, _mm256_add_ps(_mm256_loadu_ps(ap2 + 8), f1));
                    }
                }
            }
            for (int r = rb; r < rend; r++) {
                float *yp = y + (size_t)r * ns + n0;
                if (cnt == 16) {
                    _mm256_storeu_ps(yp, _mm256_loadu_ps(acc[r - rb]));
                    _mm256_storeu_ps(yp + 8, _mm256_loadu_ps(acc[r - rb] + 8));
                } else {
                    for (int t = 0; t < cnt; t++) yp[t] = acc[r - rb][t];
                }
            }
        }
    }
}
#endif

/* ------------------------------------------------------------------ */
/* ARM NEON dotprod (build with -march=armv8.2-a+dotprod on aarch64)  */
/* ------------------------------------------------------------------ */

#if defined(E2_ARM) && defined(__ARM_FEATURE_DOTPROD)
static void gemm_neon(float *y, int ns, const uint8_t *xq, const float *xs,
                      int xss, int m, int k, const e2_qmat *w, int nb0, int nb1) {
    int k4 = k / 4, nblk = k / BLK;
    for (int nb = nb0; nb < nb1; nb++) {
        int n0 = nb * 16;
        int cnt = w->n - n0; if (cnt > 16) cnt = 16;
        const int8_t *wp = w->q + (size_t)nb * k4 * 64;
        for (int r = 0; r < m; r++) {
            const uint8_t *ap = xq + (size_t)r * k;
            float *yp = y + (size_t)r * ns + n0;
            for (int b = 0; b < nblk; b++) {
                int32x4_t z = vdupq_n_s32(0);
                int32x4_t a0 = z, a1 = z, a2 = z, a3 = z;
                int32_t ws[16];
                for (int t = 0; t < 16; t++) ws[t] = -128 * w->wsum[(size_t)(n0 + t) * nblk + b];
                a0 = vld1q_s32(ws + 0); a1 = vld1q_s32(ws + 4);
                a2 = vld1q_s32(ws + 8); a3 = vld1q_s32(ws + 12);
                for (int kb = b * (BLK / 4); kb < (b + 1) * (BLK / 4); kb++) {
                    uint32_t av32;
                    memcpy(&av32, ap + (size_t)kb * 4, 4);
                    uint8x16_t av = vreinterpretq_u8_u32(vdupq_n_u32(av32));
                    const int8_t *wv = wp + (size_t)kb * 64;
                    a0 = vdotq_s32(a0, av, vld1q_s8(wv));
                    a1 = vdotq_s32(a1, av, vld1q_s8(wv + 16));
                    a2 = vdotq_s32(a2, av, vld1q_s8(wv + 32));
                    a3 = vdotq_s32(a3, av, vld1q_s8(wv + 48));
                }
                int32_t acc[16];
                vst1q_s32(acc + 0, a0); vst1q_s32(acc + 4, a1);
                vst1q_s32(acc + 8, a2); vst1q_s32(acc + 12, a3);
                float s = xs[(size_t)r * xss + b];
                if (b == 0)
                    for (int t = 0; t < cnt; t++) yp[t] = (float)acc[t] * s * w->scale[n0 + t];
                else
                    for (int t = 0; t < cnt; t++) yp[t] += (float)acc[t] * s * w->scale[n0 + t];
            }
        }
    }
}
#endif

/* ------------------------------------------------------------------ */
/* dispatch                                                           */
/* ------------------------------------------------------------------ */

static gemm_kernel g_kernel;

static gemm_kernel select_kernel(void) {
    const char *force = getenv("E2_KERNEL");
    if (force && !strcmp(force, "scalar")) return gemm_scalar;
#ifdef E2_X86
    __builtin_cpu_init();
    if (force && !strcmp(force, "avx2")) return gemm_avx2;
    if (force && !strcmp(force, "avx512")) return gemm_avx512;
    if (force && !strcmp(force, "avxvnni")) return gemm_avxvnni;
    if (__builtin_cpu_supports("avx512vnni")) return gemm_avx512;
    if (__builtin_cpu_supports("avxvnni")) return gemm_avxvnni;
    if (__builtin_cpu_supports("avx2")) return gemm_avx2;
#endif
#if defined(E2_ARM) && defined(__ARM_FEATURE_DOTPROD)
    return gemm_neon;
#else
    return gemm_scalar;
#endif
}

/* ------------------------------------------------------------------ */

/* Quantise rows to u8 with one scale per 128-element block. */
static void quant_rows_s(const float *x, int m, int k, uint8_t *q, float *scale, int blk) {
    int nblk = k / blk;
    for (int r = 0; r < m; r++) {
        const float *xp = x + (size_t)r * k;
        uint8_t *qp = q + (size_t)r * k;
        float *sp = scale + (size_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            const float *xb = xp + b * blk;
            uint8_t *qb = qp + b * blk;
            float mx = 0.0f;
            for (int i = 0; i < blk; i++) {
                float a = fabsf(xb[i]);
                if (a > mx) mx = a;
            }
            float s = mx / 127.0f;
            if (s < 1e-12f) s = 1e-12f;
            float inv = 1.0f / s;
            for (int i = 0; i < blk; i++) {
                long v = lrintf(xb[i] * inv) + 128;
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                qb[i] = (uint8_t)v;
            }
            sp[b] = s;
        }
    }
}

#ifdef E2_X86
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void quant_rows_avx512(const float *x, int m, int k, uint8_t *q, float *scale, int blk) {
    int nblk = k / blk;
    const __m512i sign = _mm512_set1_epi32((int)0x80000000u);
    const __m512 v128 = _mm512_set1_ps(128.0f), vmax = _mm512_set1_ps(1e38f);
    for (int r = 0; r < m; r++) {
        const float *xp = x + (size_t)r * k;
        uint8_t *qp = q + (size_t)r * k;
        float *sp = scale + (size_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            const float *xb = xp + b * blk;
            uint8_t *qb = qp + b * blk;
            __m512 mx = _mm512_setzero_ps();
            for (int i = 0; i < blk; i += 16) {
                __m512i v = _mm512_andnot_si512(sign, _mm512_loadu_si512((const void *)(xb + i)));
                mx = _mm512_max_ps(mx, _mm512_castsi512_ps(v));
            }
            float mxf = _mm512_reduce_max_ps(mx);
            float s = mxf / 127.0f;
            if (s < 1e-12f) s = 1e-12f;
            __m512 vinv = _mm512_set1_ps(1.0f / s);
            (void)vmax;
            for (int i = 0; i < blk; i += 16) {
                __m512 v = _mm512_loadu_ps(xb + i);
                __m512i iv = _mm512_cvtps_epi32(_mm512_fmadd_ps(v, vinv, v128));
                _mm_storeu_si128((__m128i *)(qb + i), _mm512_cvtusepi32_epi8(iv));
            }
            sp[b] = s;
        }
    }
}
#endif

/* AVX2 twin of quant_rows_s: same block max, same scale, same round
 * (vcvtps2dq = nearest-even) and the same 0..255 clamp via pack saturation.
 * The 128 offset is added in the integer domain so no float rounding edge is
 * introduced.  Used on hosts without AVX-512 (Arrow Lake / Zen 5). */
#ifdef E2_X86
__attribute__((target("avx2")))
static void quant_rows_avx2(const float *x, int m, int k, uint8_t *q, float *scale, int blk) {
    int nblk = k / blk;
    const __m256i sign = _mm256_set1_epi32((int)0x80000000u);
    const __m256i off = _mm256_set1_epi16(128);
    for (int r = 0; r < m; r++) {
        const float *xp = x + (size_t)r * k;
        uint8_t *qp = q + (size_t)r * k;
        float *sp = scale + (size_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            const float *xb = xp + b * blk;
            uint8_t *qb = qp + b * blk;
            __m256 mx0 = _mm256_setzero_ps(), mx1 = mx0;
            for (int i = 0; i < blk; i += 16) {
                /* max(v, mx) so a NaN lane keeps mx, like the scalar `a > mx` */
                mx0 = _mm256_max_ps(_mm256_castsi256_ps(_mm256_andnot_si256(
                    sign, _mm256_castps_si256(_mm256_loadu_ps(xb + i)))), mx0);
                mx1 = _mm256_max_ps(_mm256_castsi256_ps(_mm256_andnot_si256(
                    sign, _mm256_castps_si256(_mm256_loadu_ps(xb + i + 8)))), mx1);
            }
            __m256 mx = _mm256_max_ps(mx0, mx1);
            __m128 h = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
            h = _mm_max_ps(h, _mm_movehl_ps(h, h));
            h = _mm_max_ss(h, _mm_movehdup_ps(h));
            float s = _mm_cvtss_f32(h) / 127.0f;
            if (s < 1e-12f) s = 1e-12f;
            __m256 vinv = _mm256_set1_ps(1.0f / s);
            for (int i = 0; i < blk; i += 16) {
                __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + i), vinv));
                __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + i + 8), vinv));
                __m256i p = _mm256_add_epi16(_mm256_packs_epi32(i0, i1), off);
                p = _mm256_permute4x64_epi64(p, 0xd8);
                _mm_storeu_si128((__m128i *)(qb + i),
                                 _mm_packus_epi16(_mm256_castsi256_si128(p),
                                                  _mm256_extracti128_si256(p, 1)));
            }
            sp[b] = s;
        }
    }
}
#endif

typedef void (*quant_kernel)(const float *, int, int, uint8_t *, float *, int);

static quant_kernel quant_select(void) {
    const char *force = getenv("E2_QUANT");
    if (force && !strcmp(force, "scalar")) return quant_rows_s;
#ifdef E2_X86
    __builtin_cpu_init();
    if (force && !strcmp(force, "avx2")) return quant_rows_avx2;
    if (__builtin_cpu_supports("avx512bw")) return quant_rows_avx512;
    if (force && !strcmp(force, "avx512")) return quant_rows_avx512;
    if (__builtin_cpu_supports("avx2")) return quant_rows_avx2;
#endif
    return quant_rows_s;
}

void e2_quant_rows_blk(const float *x, int m, int k, uint8_t *q, float *scale, int blk) {
    static quant_kernel kern;
    if (!blk) blk = BLK;
    if (!kern) kern = quant_select();
    /* blk % 16 so the vector loops stay in bounds; k % blk so no tail */
    if ((blk % 16) == 0 && (k % blk) == 0) {
        kern(x, m, k, q, scale, blk);
        return;
    }
    quant_rows_s(x, m, k, q, scale, blk);
}

void e2_quant_rows(const float *x, int m, int k, uint8_t *q, float *scale) {
    e2_quant_rows_blk(x, m, k, q, scale, BLK);
}

typedef struct {
    const float *x; int m, k; uint8_t *q; float *scale; int blk;
} quant_arg;

static void quant_work(void *vp, int tid, int nt) {
    quant_arg *a = vp;
    int r0, r1; e2_split(a->m, tid, nt, &r0, &r1);
    if (r0 < r1)
        e2_quant_rows_blk(a->x + (size_t)r0 * a->k, r1 - r0, a->k,
                          a->q + (size_t)r0 * a->k,
                          a->scale + (size_t)r0 * (a->k / (a->blk ? a->blk : BLK)),
                          a->blk);
}

typedef struct {
    float *y;
    const uint8_t *xq;
    const float *xs;
    int xss, m, k, ns;
    const e2_qmat *w;
    gemm_kernel kern;
} gemm_arg;

static void gemm_work(void *vp, int tid, int nt) {
    gemm_arg *a = vp;
    int nblk = (a->w->n + 15) / 16;
    /* Split columns when there are enough 4-tile groups per thread; otherwise
     * split rows.  A n=256 attention PV GEMM has 16 tiles, so the old column
     * split gave each thread 2 tiles and fell into the 1-tile slow path. */
    if (nblk < nt * 4) {
        /* Disjoint 4-row-aligned row ranges.  Rounding only the start made
         * adjacent threads overlap, and the b==0 store raced with the other
         * thread's b>0 accumulate. */
        int r0, r1;
        e2_split(a->m, tid, nt, &r0, &r1);
        r0 &= ~3;
        if (tid != nt - 1) r1 &= ~3; else r1 = a->m;
        if (r0 < r1)
            a->kern(a->y + (size_t)r0 * a->ns, a->ns, a->xq + (size_t)r0 * a->k,
                    a->xs + (size_t)r0 * a->xss, a->xss, r1 - r0, a->k, a->w, 0, nblk);
        return;
    }
    int nb0, nb1;
    e2_split(nblk, tid, nt, &nb0, &nb1);
    if (nb0 < nb1) a->kern(a->y, a->ns, a->xq, a->xs, a->xss, a->m, a->k, a->w, nb0, nb1);
}

/* Single-threaded micro-GEMM used to rank the cores of a hybrid part when the
 * pool pins itself.  Clock alone is a bad proxy: an E-core runs vpdpbusd at a
 * different rate and has less load bandwidth, so calibrate on the real kernel.
 * Returns milliseconds for 20 passes over m=8 k=512 n=64. */
double e2_gemm_calib(void) {
    enum { N = 64, K = 512, M = 8 };
    static uint8_t xq[M * K];
    static float xs[M * (K / 128)];
    static int8_t wq[(N / 16) * (K / 4) * 64];
    static float wsc[N];
    static int32_t wsum[N * (K / 128)];
    static float y[M * N];
    static int ready;
    e2_gemm_init();
    if (!g_kernel) return 0.0;
    if (!ready) {
        memset(xq, 128, sizeof(xq));
        for (int i = 0; i < M * (K / 128); i++) xs[i] = 1e-3f;
        for (size_t i = 0; i < sizeof(wq); i++) wq[i] = (int8_t)(i * 37u);
        for (int i = 0; i < N; i++) wsc[i] = 1e-4f;
        ready = 1;
    }
    e2_qmat w;
    memset(&w, 0, sizeof(w));
    w.q = wq; w.scale = wsc; w.wsum = wsum;
    w.n = N; w.k = K; w.blk = BLK; w.packed = 1;
    double t0 = e2_now_ms();
    for (int r = 0; r < 20; r++)
        g_kernel(y, N, xq, xs, K / 128, M, K, &w, 0, N / 16);
    return e2_now_ms() - t0;
}

void e2_gemm_init(void) {
    if (g_kernel) return;
    g_kernel = select_kernel();
    const char *pf = getenv("E2_GEMM_PF");
    if (pf) g_pf = atoi(pf);
    const char *rb = getenv("E2_GEMM_RB");
    if (rb) { g_rb = g_rb512 = atoi(rb); if (g_rb <= 0) g_rb = g_rb512 = 1 << 28; }
}

/* One linear, including the input-clip path the shared-quantiser version
 * cannot express. */
static void linear_one(e2_ctx *ctx, float *y, const float *x, int m, const e2_linear *l);

/* Try the GPU GEMM; returns 0 if it produced y, nonzero to use the CPU. */
static int gpu_try(e2_ctx *ctx, float *y, int ns, const uint8_t *xq, const float *xs,
                   int xss, int m, int k, const e2_qmat *w) {
    if (!ctx || ctx->device != E2_DEVICE_GPU || !ctx->gpu || m < 32) return -1;
    return e2_gpu_gemm(ctx, y, ns, xq, xs, xss, m, k, w, 0, (w->n + 15) / 16);
}

void e2_linear_forward(e2_ctx *ctx, float *y, const float *x, int m, const e2_linear *l) {
    if (l->in_min || l->in_max) { linear_one(ctx, y, x, m, l); return; }
    e2_linear_forward_multi(ctx, x, m, &l, &y, 1);
}

/* Several linears reading the same activation row block (q/k/v, gate/up):
 * quantise once and run every GEMM on it.  The quantiser was a full extra
 * read+max pass over m*k floats for each of those. */
void e2_linear_forward_multi(e2_ctx *ctx, const float *x, int m,
                             const e2_linear *const *ls, float *const *ys, int n) {
    const e2_qmat *w = &ls[0]->w;
    e2_gemm_init();
    int k = w->k;
    int blk = w->blk ? w->blk : BLK;
    int nblk = k / blk;
    if (!g_kernel || nblk == 0) return;
    /* A linear that needs a clip, or a different k/block, cannot use the shared
     * activation quantisation; it takes the single-linear path instead.  (This
     * used to *truncate* the list, which silently skipped every linear after
     * the first clipped one -- e.g. the audio tower's v_proj.) */
    if (ls[0]->in_min || ls[0]->in_max) {
        for (int i = 0; i < n; i++) linear_one(ctx, ys[i], x, m, ls[i]);
        return;
    }
    for (int i = 1; i < n; i++) {
        const e2_qmat *q = &ls[i]->w;
        if (q->k != k || (q->blk ? q->blk : BLK) != blk || ls[i]->in_min || ls[i]->in_max)
            linear_one(ctx, ys[i], x, m, ls[i]);
    }

    int m4 = (m + 15) & ~15;
    size_t qneed = (size_t)m4 * k, sneed = (size_t)m4 * nblk;
    if (ctx->qcap < qneed) {
        free(ctx->qbuf);
        ctx->qbuf = malloc(qneed);
        ctx->qcap = qneed;
    }
    if (ctx->scap < sneed * sizeof(float)) {
        free(ctx->sbuf);
        ctx->sbuf = malloc(sneed * sizeof(float));
        ctx->scap = sneed * sizeof(float);
    }
    uint8_t *xq = ctx->qbuf;
    float *xs = ctx->sbuf;
    if (!xq || !xs) return;

    if (ctx->nthreads > 1 && (long)m * k >= 262144) {
        quant_arg qa = {x, m, k, xq, xs, blk};
        e2_parallel(ctx, quant_work, &qa);
    } else {
        e2_quant_rows_blk(x, m, k, xq, xs, blk);
    }
    if (m4 > m) {
        memset(xq + (size_t)m * k, 128, (size_t)(m4 - m) * k);
        for (int r = m; r < m4; r++)
            for (int b = 0; b < nblk; b++) xs[(size_t)r * nblk + b] = 1.0f;
    }

    gemm_arg a = {0, xq, xs, nblk, m, k, w->n, w, g_kernel};
    for (int i = 0; i < n; i++) {
        const e2_linear *l = ls[i];
        if (i && (l->w.k != k || (l->w.blk ? l->w.blk : BLK) != blk ||
                  l->in_min || l->in_max))
            continue;                    /* already done above */
        a.y = ys[i];
        a.w = &l->w;
        a.ns = l->w.n;
        long work = (long)m * l->w.n * k;
        int gpu = gpu_try(ctx, ys[i], l->w.n, xq, xs, nblk, m, k, &l->w) == 0;
        if (!gpu) {
            if (ctx->nthreads <= 1 || work < 2000000L)
                gemm_work(&a, 0, 1);
            else
                e2_parallel(ctx, gemm_work, &a);
        }
        if (l->w.bias || l->out_min || l->out_max) {
            float lo = l->out_min ? l->out_min[0] : -INFINITY;
            float hi = l->out_max ? l->out_max[0] : INFINITY;
            int nn = l->w.n;
            for (int r = 0; r < m; r++) {
                float *yp = ys[i] + (size_t)r * nn;
                if (l->w.bias)
                    for (int j = 0; j < nn; j++) yp[j] += l->w.bias[j];
                if (l->out_min || l->out_max)
                    for (int j = 0; j < nn; j++) {
                        if (yp[j] < lo) yp[j] = lo;
                        if (yp[j] > hi) yp[j] = hi;
                    }
            }
        }
    }
}

static void linear_one(e2_ctx *ctx, float *y, const float *x, int m, const e2_linear *l) {
    const e2_qmat *w = &l->w;
    e2_gemm_init();
    int k = w->k, blk = w->blk ? w->blk : BLK, nblk = k / blk;
    if (!g_kernel || nblk == 0) return;
    int m4 = (m + 15) & ~15;
    size_t qneed = (size_t)m4 * k, sneed = (size_t)m4 * nblk;
    if (ctx->qcap < qneed) {
        free(ctx->qbuf);
        ctx->qbuf = malloc(qneed);
        ctx->qcap = qneed;
    }
    if (ctx->scap < sneed * sizeof(float)) {
        free(ctx->sbuf);
        ctx->sbuf = malloc(sneed * sizeof(float));
        ctx->scap = sneed * sizeof(float);
    }
    uint8_t *xq = ctx->qbuf;
    float *xs = ctx->sbuf;
    if (!xq || !xs) return;
    if (l->in_min || l->in_max) {
        float lo = l->in_min ? l->in_min[0] : -INFINITY;
        float hi = l->in_max ? l->in_max[0] : INFINITY;
        size_t tneed = (size_t)m * k;
        if (ctx->tcap < tneed * sizeof(float)) {
            free(ctx->tbuf);
            ctx->tbuf = malloc(tneed * sizeof(float));
            ctx->tcap = ctx->tbuf ? tneed * sizeof(float) : 0;
        }
        float *tmp = ctx->tbuf;
        if (!tmp) return;
        for (int i = 0; i < m * k; i++) {
            float v = x[i];
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            tmp[i] = v;
        }
        x = tmp;
    }
    if (ctx->nthreads > 1 && (long)m * k >= 262144) {
        quant_arg qa = {x, m, k, xq, xs, blk};
        e2_parallel(ctx, quant_work, &qa);
    } else {
        e2_quant_rows_blk(x, m, k, xq, xs, blk);
    }
    if (m4 > m) {
        memset(xq + (size_t)m * k, 128, (size_t)(m4 - m) * k);
        for (int r = m; r < m4; r++)
            for (int b = 0; b < nblk; b++) xs[(size_t)r * nblk + b] = 1.0f;
    }
    gemm_arg a = {y, xq, xs, nblk, m, k, w->n, w, g_kernel};
    long work = (long)m * w->n * k;
    if (gpu_try(ctx, y, w->n, xq, xs, nblk, m, k, w) != 0) {
        if (ctx->nthreads <= 1 || work < 2000000L)
            gemm_work(&a, 0, 1);
        else
            e2_parallel(ctx, gemm_work, &a);
    }
    if (w->bias || l->out_min || l->out_max) {
        float lo = l->out_min ? l->out_min[0] : -INFINITY;
        float hi = l->out_max ? l->out_max[0] : INFINITY;
        int nn = w->n;
        for (int r = 0; r < m; r++) {
            float *yp = y + (size_t)r * nn;
            if (w->bias)
                for (int j = 0; j < nn; j++) yp[j] += w->bias[j];
            if (l->out_min || l->out_max)
                for (int j = 0; j < nn; j++) {
                    if (yp[j] < lo) yp[j] = lo;
                    if (yp[j] > hi) yp[j] = hi;
                }
        }
    }
}

/* Single-threaded linear with caller-owned scratch.  Used by the batched
 * short-sequence attention path, where the outer parallel loop already owns
 * the threads and all ctx scratch buffers would race. */
void e2_linear_forward_st(float *y, const float *x, int m, const e2_linear *l,
                          uint8_t *xq, float *xs, float *tmp) {
    const e2_qmat *w = &l->w;
    e2_gemm_init();
    int k = w->k, blk = w->blk ? w->blk : BLK, nblk = k / blk;
    if (l->in_min || l->in_max) {
        float lo = l->in_min ? l->in_min[0] : -INFINITY;
        float hi = l->in_max ? l->in_max[0] : INFINITY;
        for (int i = 0; i < m * k; i++) {
            float v = x[i];
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            tmp[i] = v;
        }
        x = tmp;
    }
    e2_quant_rows_blk(x, m, k, xq, xs, blk);
    gemm_arg a = {y, xq, xs, nblk, m, k, w->n, w, g_kernel};
    gemm_work(&a, 0, 1);
    if (w->bias || l->out_min || l->out_max) {
        float lo = l->out_min ? l->out_min[0] : -INFINITY;
        float hi = l->out_max ? l->out_max[0] : INFINITY;
        for (int r = 0; r < m; r++) {
            float *yp = y + (size_t)r * w->n;
            if (w->bias)
                for (int j = 0; j < w->n; j++) yp[j] += w->bias[j];
            if (l->out_min || l->out_max)
                for (int j = 0; j < w->n; j++) {
                    if (yp[j] < lo) yp[j] = lo;
                    if (yp[j] > hi) yp[j] = hi;
                }
        }
    }
}

/* Linear with already-quantised activations (no clip, no bias expected). */
void e2_linear_forward_pq(e2_ctx *ctx, float *y, const uint8_t *xq,
                          const float *xs, int m, const e2_linear *l) {
    const e2_qmat *w = &l->w;
    e2_gemm_init();
    int k = w->k, blk = w->blk ? w->blk : BLK, nblk = k / blk;
    gemm_arg a = {y, xq, xs, nblk, m, k, w->n, w, g_kernel};
    long work = (long)m * w->n * k;
    if (gpu_try(ctx, y, w->n, xq, xs, nblk, m, k, w) != 0) {
        if (!ctx || ctx->nthreads <= 1 || work < 2000000L)
            gemm_work(&a, 0, 1);
        else
            e2_parallel(ctx, gemm_work, &a);
    }
}

/* Caller-owned m4-padded u8 activation scratch (ctx->qbuf) plus its block
 * scales, for the fused producer+quantiser paths. */
uint8_t *e2_quant_scratch(e2_ctx *ctx, int m, int k, float **xs) {
    int blk = BLK, m4 = (m + 15) & ~15, nblk = k / blk;
    size_t qneed = (size_t)m4 * k, sneed = (size_t)m4 * nblk;
    if (ctx->qcap < qneed) {
        free(ctx->qbuf);
        ctx->qbuf = malloc(qneed);
        ctx->qcap = ctx->qbuf ? qneed : 0;
    }
    if (ctx->scap < sneed * sizeof(float)) {
        free(ctx->sbuf);
        ctx->sbuf = malloc(sneed * sizeof(float));
        ctx->scap = ctx->sbuf ? sneed * sizeof(float) : 0;
    }
    if (!ctx->qbuf || !ctx->sbuf) return NULL;
    if (m4 > m) {
        memset(ctx->qbuf + (size_t)m * k, 128, (size_t)(m4 - m) * k);
        for (int r = m; r < m4; r++)
            for (int b = 0; b < nblk; b++) ctx->sbuf[(size_t)r * nblk + b] = 1.0f;
    }
    *xs = ctx->sbuf;
    return ctx->qbuf;
}

/* Single-threaded GEMM over an output-tile range [nb0, nb1).  Attention uses
 * this to score only the key band a query block can see, and to contract only
 * that band in the P x V step.  Tile nb0 lands on column 0, so the caller's
 * buffer describes just the band. */
void e2_gemm_range(float *y, int ns, const uint8_t *xq, const float *xs, int xss,
                   int m, int k, const e2_qmat *w, int nb0, int nb1) {
    e2_gemm_init();
    if (nb0 < 0) nb0 = 0;
    if (nb1 > (w->n + 15) / 16) nb1 = (w->n + 15) / 16;
    if (nb0 >= nb1 || m <= 0) return;
    g_kernel(y - (size_t)nb0 * 16, ns, xq, xs, xss, m, k, w, nb0, nb1);
}

/* f32 helpers (debug) */
void e2_gemm_nt_f32(float *y, const float *x, const float *w, int m, int k, int n) {
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) {
            const float *xp = x + (size_t)r * k;
            const float *wp = w + (size_t)c * k;
            float s = 0.0f;
            for (int i = 0; i < k; i++) s += xp[i] * wp[i];
            y[(size_t)r * n + c] = s;
        }
}

void e2_gemm_f32(float *y, const float *x, const float *w, int m, int k, int n) {
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) y[(size_t)r * n + c] = 0.0f;
    for (int r = 0; r < m; r++)
        for (int i = 0; i < k; i++) {
            float a = x[(size_t)r * k + i];
            const float *wp = w + (size_t)i * n;
            float *yp = y + (size_t)r * n;
            for (int c = 0; c < n; c++) yp[c] += a * wp[c];
        }
}

/* ------------------------------------------------------------------ */
/* Packer for on-the-fly int8 matrices (attention weights, ...).       */
/* ------------------------------------------------------------------ */

typedef struct {
    const float *w;
    int n, k, rs, cs, blk;
    int8_t *qw;
    float *sc;
    int32_t *ws;
} pack_arg;

/* cs == 1, k % 16 == 0 fast path: quantise a contiguous row with AVX-512,
 * then scatter the 4-byte groups into the 16-row tile layout. */
#ifdef E2_X86
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void pack_rows_avx512(const pack_arg *a, int r0, int r1, int8_t *row) {
    const int k = a->k, k4 = k / 4, nblk = k / a->blk;
    const __m512i sign = _mm512_set1_epi32((int)0x80000000u);
    const __m512i ones8 = _mm512_set1_epi8(1);
    const __m512i ones16 = _mm512_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        if (r >= a->n) { a->sc[r] = 1.0f; continue; }
        const float *wr = a->w + (size_t)r * a->rs;
        __m512 mx = _mm512_setzero_ps();
        for (int i = 0; i < k; i += 16) {
            __m512i v = _mm512_andnot_si512(sign, _mm512_loadu_si512((const void *)(wr + i)));
            mx = _mm512_max_ps(mx, _mm512_castsi512_ps(v));
        }
        float s = _mm512_reduce_max_ps(mx) / 127.0f;
        if (s < 1e-12f) s = 1e-12f;
        a->sc[r] = s;
        __m512 vinv = _mm512_set1_ps(1.0f / s);
        for (int i = 0; i < k; i += 16) {
            __m512 v = _mm512_loadu_ps(wr + i);
            __m512i iv = _mm512_cvtps_epi32(_mm512_mul_ps(v, vinv));
            _mm_storeu_si128((__m128i *)(row + i), _mm512_cvtsepi32_epi8(iv));
        }
        int8_t *dst = a->qw + (size_t)(r / 16) * k4 * 64 + (size_t)(r % 16) * 4;
        for (int kb = 0; kb < k4; kb++)
            memcpy(dst + (size_t)kb * 64, row + (size_t)kb * 4, 4);
        int32_t *wrow = a->ws + (size_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            const int8_t *src = row + (size_t)b * a->blk;
            __m512i acc = _mm512_setzero_si512();
            for (int i = 0; i < a->blk; i += 64) {
                __m512i v = _mm512_loadu_si512((const void *)(src + i));
                __m512i p = _mm512_maddubs_epi16(ones8, v);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p, ones16));
            }
            wrow[b] = _mm512_reduce_add_epi32(acc);
        }
    }
}
#endif

static void pack_rows_s(const pack_arg *a, int r0, int r1) {
    const int k = a->k, k4 = k / 4, nblk = k / a->blk;
    for (int r = r0; r < r1; r++) {
        if (r >= a->n) { a->sc[r] = 1.0f; continue; }
        const float *wr = a->w + (size_t)r * a->rs;
        float mx = 0.0f;
        for (int i = 0; i < k; i++) {
            float v = fabsf(wr[(size_t)i * a->cs]);
            if (v > mx) mx = v;
        }
        float s = mx / 127.0f;
        if (s < 1e-12f) s = 1e-12f;
        a->sc[r] = s;
        float inv = 1.0f / s;
        int8_t *dst = a->qw + (size_t)(r / 16) * k4 * 64 + (size_t)(r % 16) * 4;
        int32_t *wrow = a->ws + (size_t)r * nblk;
        for (int b = 0; b < nblk; b++) {
            int32_t acc = 0;
            for (int kk = b * (a->blk / 4); kk < (b + 1) * (a->blk / 4); kk++) {
                int8_t *d = dst + (size_t)kk * 64;
                for (int j = 0; j < 4; j++) {
                    long v = lrintf(wr[(size_t)(kk * 4 + j) * a->cs] * inv);
                    if (v < -127) v = -127;
                    if (v > 127) v = 127;
                    d[j] = (int8_t)v;
                    acc += (int32_t)v;
                }
            }
            wrow[b] = acc;
        }
    }
}

typedef struct { const pack_arg *a; int fast, rows; int8_t *rowbuf; } pack_job;

static int pack_fast_ok(void) {
    static int v = -1;
    if (v < 0) {
        v = 0;
#ifdef E2_X86
        __builtin_cpu_init();
        v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
            && __builtin_cpu_supports("avx512vl");
#endif
    }
    return v;
}

static void pack_work(void *vp, int tid, int nt) {
    pack_job *j = vp;
    int r0, r1;
    e2_split(j->rows, tid, nt, &r0, &r1);
#ifdef E2_X86
    if (j->fast) { pack_rows_avx512(j->a, r0, r1, j->rowbuf + (size_t)tid * j->a->k); return; }
#endif
    pack_rows_s(j->a, r0, r1);
}

/* Pack into caller-owned buffers (qw zeroed for the tile padding by the
 * caller, or by a previous call -- the padding rows are multiplied by a
 * masked-zero activation, so their contents never reach the output). */
int e2_pack_mat_into(e2_ctx *ctx, const float *w, int n, int k, int rs, int cs,
                     int8_t *qw, float *sc, int32_t *ws, int *blk_out) {
    int blk = *blk_out ? *blk_out : BLK;
    if ((k % blk) || (k % 4)) return -1;
    int nbuf = (ctx && ctx->nthreads > 1 && (long)n * k >= 65536) ? ctx->nthreads : 1;
    int8_t sbuf[1024];
    int8_t *rowbuf = k * nbuf <= (int)sizeof(sbuf) ? sbuf : malloc((size_t)k * nbuf);
    if (!rowbuf) return -1;
    pack_arg a = {w, n, k, rs, cs, blk, qw, sc, ws};
    int fast = (cs == 1 && (k % 16) == 0 && (blk % 16) == 0 &&
                !getenv("E2_PACK_SCALAR") && pack_fast_ok());
    int ntile = (n + 15) / 16;
    pack_job j = {&a, fast, ntile * 16, rowbuf};
    if (nbuf > 1)
        e2_parallel(ctx, pack_work, &j);
    else
        pack_work(&j, 0, 1);
    if (rowbuf != sbuf) free(rowbuf);
    *blk_out = blk;
    return 0;
}

static int pack_mat(e2_ctx *ctx, const float *w, int n, int k, int rs, int cs,
                    int8_t **q, float **scale, int32_t **wsum, int *blk_out) {
    int blk = *blk_out ? *blk_out : BLK;
    if ((k % blk) || (k % 4)) return -1;
    int ntile = (n + 15) / 16, k4 = k / 4, nblk = k / blk;
    int8_t *qw = calloc((size_t)ntile * k4 * 64 + 64, 1);
    float *sc = malloc((size_t)ntile * 16 * sizeof(float));
    int32_t *ws = calloc((size_t)ntile * 16 * nblk, sizeof(int32_t));
    if (!qw || !sc || !ws ||
        e2_pack_mat_into(ctx, w, n, k, rs, cs, qw, sc, ws, &blk) != 0) {
        free(qw); free(sc); free(ws); return -1;
    }
    free(*q); free(*scale); free(*wsum);
    *q = qw; *scale = sc; *wsum = ws;
    *blk_out = blk;
    return 0;
}

int e2_pack_mat(const float *w, int n, int k, int rs, int cs, int8_t **q,
                float **scale, int32_t **wsum, int *blk_out) {
    return pack_mat(NULL, w, n, k, rs, cs, q, scale, wsum, blk_out);
}

int e2_pack_mat_ctx(e2_ctx *ctx, const float *w, int n, int k, int rs, int cs,
                    int8_t **q, float **scale, int32_t **wsum, int *blk_out) {
    return pack_mat(ctx, w, n, k, rs, cs, q, scale, wsum, blk_out);
}

void e2_pack_free(int8_t *q, float *scale, int32_t *wsum) {
    free(q); free(scale); free(wsum);
}
