/* kernels_i8_avx512_vnni.c — int8 GEMM on AVX512-VNNI (vpdpbusd zmm).
 *
 * Same arithmetic and the same quad-interleaved weight layout as kernels_i8_vnni.c (see int8.h).
 * The 256-bit kernel keeps 8 tokens in flight (one int32 and one float accumulator per token = 16
 * registers) and puts 8 output rows in every 32-bit lane group; the 512-bit kernel instead puts 16
 * output rows in the lanes (two 8-row groups per weight operand) and keeps 8 tokens in flight
 * across 16 registers. That is 3 + 2*MR uops per 16*4*MR MACs against 1 + 4*MR for an equally
 * register-bound paired layout, i.e. ~1.7x more MACs per uop. The per-layer weight slices are
 * L2-resident in this engine, so the uop count decides and this layout wins.
 *
 * The weight operand is the two 32-byte 8-row groups combined into one zmm, and the activation
 * operand is one 4-byte k-quad broadcast to all 16 lanes. vpdpbusd wants an unsigned first operand;
 * the stored activations are already a+128 (int8.h), so no per-token flip is needed.
 *
 * Tokens go 8 at a time; the 256-bit loop handles the 1..7 token tail. N is processed 16 output
 * rows at a time (8-row and masked-store tails), so an arbitrary claim range is correct.
 */
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

#include "int8.h"
#include "kernels.h"

#define NQ4 (E2_I8_BLOCK / 4) /* k-quads per activation scale block */
#define MR8 8                 /* tokens per 256-bit tail pass */

/* ---------------------------------------------------------------- packed-operand attention GEMM
 * See int8.h for the contract. 8 query rows x 16 output columns per step, 16 output rows per weight
 * operand, one 4-byte activation broadcast reused by every row in the tile. lda lets the caller keep
 * Q in its [S, H*D] layout instead of repacking it per head. */
void e2_i8_attn_gemm_avx512(float *C, int ldc, const uint8_t *A, int lda, const float *asc,
                            const uint8_t *Bp, const float *bsc, const int32_t *bz, int M, int Np,
                            int N, int K) {
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const int qend = (K + 3) / 4;
    for (int n0 = 0; n0 < Np; n0 += 16) {
        const int left = N - n0;
        const int nn = left >= 16 ? 16 : (left > 0 ? left : 0);
        const __mmask16 nm = (__mmask16)(nn == 16 ? 0xffffu : ((1u << nn) - 1u));
        const __m512 bscv = _mm512_maskz_loadu_ps(nm, bsc + n0);
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = M - m0 < 8 ? M - m0 : 8;
            __m512i acc[8];
            __m512 f[8];
            for (int i = 0; i < 8; i++) {
                acc[i] = _mm512_setzero_si512();
                f[i] = _mm512_setzero_ps();
            }
            for (int b = 0; b < nb; b++) {
                const int q1 = (b * (E2_I8_BLOCK / 4) + E2_I8_BLOCK / 4) < qend
                                   ? b * (E2_I8_BLOCK / 4) + E2_I8_BLOCK / 4
                                   : qend;
                for (int q = b * (E2_I8_BLOCK / 4); q < q1; q++) {
                    const __m512i bv =
                        _mm512_loadu_si512((const void *)(Bp + ((size_t)q * Np + n0) * 4));
                    for (int i = 0; i < mm; i++)
                        acc[i] = _mm512_dpbusd_epi32(
                            acc[i],
                            _mm512_set1_epi32(*(const int32_t *)(A + (size_t)(m0 + i) * lda + q * 4)),
                            bv);
                }
                const __m512i zp =
                    _mm512_slli_epi32(_mm512_loadu_si512((const void *)(bz + (size_t)b * Np + n0)), 7);
                for (int i = 0; i < mm; i++) {
                    const __m512 sc =
                        _mm512_mul_ps(_mm512_set1_ps(asc[(size_t)(m0 + i) * nb + b]), bscv);
                    f[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[i], zp)), sc, f[i]);
                    acc[i] = _mm512_setzero_si512();
                }
            }
            for (int i = 0; i < mm; i++) {
                if (nn == 16) _mm512_storeu_ps(C + (size_t)(m0 + i) * ldc + n0, f[i]);
                else if (nn > 0) _mm512_mask_storeu_ps(C + (size_t)(m0 + i) * ldc + n0, nm, f[i]);
            }
        }
    }
}

/* ---------------------------------------------------------------- 512-bit 16-row loop
 * 16 output rows x 1 token per zmm: the weight is the two 8-row groups combined (one 64-byte load
 * cannot span them, so 2 loads + insert), the activation operand is a single 4-byte broadcast,
 * reused by every token in the 8-token block. */
static int gemm_cols16_mr8(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                           int M, int N, int n) {
    const __m512 lo5 = _mm512_set1_ps(w->out_min), hi5 = _mm512_set1_ps(w->out_max);
    const int K = w->K, nb = w->nblocks, NP = w->Npad;
    const int kq = (K + 3) / 4;
    const int8_t *wp0 = w->w + (size_t)(n >> 3) * kq * 32 + (n & 7) * 4;
    const int8_t *wp1 = wp0 + (size_t)kq * 32;
    const __m512 wsc = _mm512_loadu_ps(w->scale + n);
    int m = 0;
    for (; m + 8 <= M; m += 8) {
        __m512i acc[8];
        __m512 f[8];
        for (int i = 0; i < 8; i++) {
            acc[i] = _mm512_setzero_si512();
            f[i] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; b++) {
            const int8_t *ap[8];
            for (int i = 0; i < 8; i++) ap[i] = qa + (size_t)(m + i) * K + b * E2_I8_BLOCK;
            const int8_t *wpq0 = wp0 + (size_t)b * NQ4 * 32;
            const int8_t *wpq1 = wp1 + (size_t)b * NQ4 * 32;
            for (int q = 0, off = 0; q < NQ4; q++, off += 4, wpq0 += 32, wpq1 += 32) {
                const __m512i wv = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)wpq0)),
                    _mm256_loadu_si256((const __m256i *)wpq1), 1);
                for (int i = 0; i < 8; i++)
                    acc[i] = _mm512_dpbusd_epi32(
                        acc[i], _mm512_set1_epi32(*(const int32_t *)(ap[i] + off)), wv);
            }
            const __m512i zp = _mm512_slli_epi32(
                _mm512_loadu_si512((const void *)(w->bsq + (size_t)b * NP + n)), 7);
            for (int i = 0; i < 8; i++) {
                const __m512 sc = _mm512_mul_ps(_mm512_set1_ps(a_scales[(size_t)(m + i) * nb + b]), wsc);
                f[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[i], zp)), sc, f[i]);
                acc[i] = _mm512_setzero_si512();
            }
        }
        for (int i = 0; i < 8; i++)
            _mm512_storeu_ps(y + (size_t)(m + i) * N + n, _mm512_min_ps(_mm512_max_ps(f[i], lo5), hi5));
    }
    return m;
}

/* ---------------------------------------------------------------- 256-bit tail
 * Rows n..n+rows-1 (rows in [1,8]), any M. This is the kernels_i8_vnni.c kernel: 8 tokens share
 * one weight read, with a single-token tail. EVEX-encoded here (avx512vl+avx512vnni), so it also
 * runs on CPUs that have AVX512-VNNI but not the VEX AVX-VNNI encoding. */
static void gemm_cols8_256(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                           int M, int N, int n, int rows) {
    const __m256 lo_ = _mm256_set1_ps(w->out_min), hi_ = _mm256_set1_ps(w->out_max);
    const int K = w->K, nb = w->nblocks, NP = w->Npad;
    const int kq = (K + 3) / 4;
    const int8_t *wp = w->w + (size_t)(n >> 3) * kq * 32 + (n & 7) * 4;
    const __m256 wsc = _mm256_loadu_ps(w->scale + n);
    int m = 0;
    for (; m + MR8 <= M; m += MR8) {
        __m256i acc[MR8];
        __m256 f[MR8];
        for (int i = 0; i < MR8; i++) {
            acc[i] = _mm256_setzero_si256();
            f[i] = _mm256_setzero_ps();
        }
        for (int b = 0; b < nb; b++) {
            const int8_t *ap[MR8];
            for (int i = 0; i < MR8; i++) ap[i] = qa + (size_t)(m + i) * K + b * E2_I8_BLOCK;
            const int8_t *wpq = wp + (size_t)b * NQ4 * 32;
            for (int q = 0, off = 0; q < NQ4; q++, off += 4, wpq += 32) {
                const __m256i wv = _mm256_loadu_si256((const __m256i *)wpq);
                for (int i = 0; i < MR8; i++)
                    acc[i] = _mm256_dpbusd_epi32(
                        acc[i], _mm256_set1_epi32(*(const int32_t *)(ap[i] + off)), wv);
            }
            const __m256i zp = _mm256_slli_epi32(
                _mm256_loadu_si256((const __m256i *)(w->bsq + (size_t)b * NP + n)), 7);
            for (int i = 0; i < MR8; i++) {
                const __m256 sc =
                    _mm256_mul_ps(_mm256_set1_ps(a_scales[(size_t)(m + i) * nb + b]), wsc);
                f[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i], zp)), sc, f[i]);
                acc[i] = _mm256_setzero_si256();
            }
        }
        if (rows == 8) {
            for (int i = 0; i < MR8; i++)
                _mm256_storeu_ps(y + (size_t)(m + i) * N + n, _mm256_min_ps(_mm256_max_ps(f[i], lo_), hi_));
        } else {
            float tmp[8];
            for (int i = 0; i < MR8; i++) {
                _mm256_storeu_ps(tmp, _mm256_min_ps(_mm256_max_ps(f[i], lo_), hi_));
                float *dst = y + (size_t)(m + i) * N + n;
                for (int j = 0; j < rows; j++) dst[j] = tmp[j];
            }
        }
    }
    for (; m < M; m++) {
        __m256i acc = _mm256_setzero_si256();
        __m256 f = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const int8_t *ap = qa + (size_t)m * K + b * E2_I8_BLOCK;
            const int8_t *wpq = wp + (size_t)b * NQ4 * 32;
            for (int q = 0, off = 0; q < NQ4; q++, off += 4, wpq += 32) {
                const __m256i wv = _mm256_loadu_si256((const __m256i *)wpq);
                acc = _mm256_dpbusd_epi32(acc, _mm256_set1_epi32(*(const int32_t *)(ap + off)), wv);
            }
            const __m256i zp = _mm256_slli_epi32(
                _mm256_loadu_si256((const __m256i *)(w->bsq + (size_t)b * NP + n)), 7);
            const __m256 sc = _mm256_mul_ps(_mm256_set1_ps(a_scales[(size_t)m * nb + b]), wsc);
            f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc, zp)), sc, f);
            acc = _mm256_setzero_si256();
        }
        if (rows == 8) {
            _mm256_storeu_ps(y + (size_t)m * N + n, _mm256_min_ps(_mm256_max_ps(f, lo_), hi_));
        } else {
            float tmp[8];
            _mm256_storeu_ps(tmp, _mm256_min_ps(_mm256_max_ps(f, lo_), hi_));
            float *dst = y + (size_t)m * N + n;
            for (int j = 0; j < rows; j++) dst[j] = tmp[j];
        }
    }
}

/* ---------------------------------------------------------------- 512-bit 32-row loop
 * Two 16-row operand groups per activation load: 4 tokens x 32 output rows. The activation quad is
 * loaded once and reused by both weight vectors, which halves the activation traffic the 16-row
 * kernel pays per output column - the wide projections (gate/up at N=2048) were L3-bound on it.
 * 4 tokens instead of 8 keeps the register file at 8 int32 + 8 float accumulators; the weights are
 * re-read twice as often but stay in L1. */
static int gemm_cols32_mr4(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                           int M, int N, int n) {
    const __m512 lo5 = _mm512_set1_ps(w->out_min), hi5 = _mm512_set1_ps(w->out_max);
    const int K = w->K, nb = w->nblocks, NP = w->Npad;
    const int kq = (K + 3) / 4;
    const int8_t *wp0 = w->w + (size_t)(n >> 3) * kq * 32 + (n & 7) * 4;
    const int8_t *wp1 = wp0 + (size_t)kq * 32;
    const int8_t *wp2 = wp1 + (size_t)kq * 32;
    const int8_t *wp3 = wp2 + (size_t)kq * 32;
    const __m512 wsc0 = _mm512_loadu_ps(w->scale + n);
    const __m512 wsc1 = _mm512_loadu_ps(w->scale + n + 16);
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        __m512i acc[4][2];
        __m512 f[4][2];
        for (int i = 0; i < 4; i++) {
            acc[i][0] = acc[i][1] = _mm512_setzero_si512();
            f[i][0] = f[i][1] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; b++) {
            const int8_t *ap[4];
            for (int i = 0; i < 4; i++) ap[i] = qa + (size_t)(m + i) * K + b * E2_I8_BLOCK;
            const int8_t *q0 = wp0 + (size_t)b * NQ4 * 32;
            const int8_t *q1 = wp1 + (size_t)b * NQ4 * 32;
            const int8_t *q2 = wp2 + (size_t)b * NQ4 * 32;
            const int8_t *q3 = wp3 + (size_t)b * NQ4 * 32;
            for (int q = 0, off = 0; q < NQ4; q++, off += 4, q0 += 32, q1 += 32, q2 += 32, q3 += 32) {
                const __m512i wv0 = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q0)),
                    _mm256_loadu_si256((const __m256i *)q1), 1);
                const __m512i wv1 = _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q2)),
                    _mm256_loadu_si256((const __m256i *)q3), 1);
                for (int i = 0; i < 4; i++) {
                    const __m512i av = _mm512_set1_epi32(*(const int32_t *)(ap[i] + off));
                    acc[i][0] = _mm512_dpbusd_epi32(acc[i][0], av, wv0);
                    acc[i][1] = _mm512_dpbusd_epi32(acc[i][1], av, wv1);
                }
            }
            const __m512i zp0 = _mm512_slli_epi32(
                _mm512_loadu_si512((const void *)(w->bsq + (size_t)b * NP + n)), 7);
            const __m512i zp1 = _mm512_slli_epi32(
                _mm512_loadu_si512((const void *)(w->bsq + (size_t)b * NP + n + 16)), 7);
            for (int i = 0; i < 4; i++) {
                const float as = a_scales[(size_t)(m + i) * nb + b];
                f[i][0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[i][0], zp0)),
                                          _mm512_mul_ps(_mm512_set1_ps(as), wsc0), f[i][0]);
                f[i][1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[i][1], zp1)),
                                          _mm512_mul_ps(_mm512_set1_ps(as), wsc1), f[i][1]);
                acc[i][0] = acc[i][1] = _mm512_setzero_si512();
            }
        }
        for (int i = 0; i < 4; i++) {
            _mm512_storeu_ps(y + (size_t)(m + i) * N + n, _mm512_min_ps(_mm512_max_ps(f[i][0], lo5), hi5));
            _mm512_storeu_ps(y + (size_t)(m + i) * N + n + 16, _mm512_min_ps(_mm512_max_ps(f[i][1], lo5), hi5));
        }
    }
    return m;
}

static int i8_w32(void) {
    static int e = -1;
    if (e < 0) e = getenv("E2_I8_W16") == NULL;
    return e;
}

void e2_i8_gemm_avx512(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                       int M, int N, int n0, int n1) {
    const int nb = w->nblocks;
    int n = n0;
    /* 32 output rows per pass share one activation load; only worth it while the 32-column weight
     * slice stays in L1 (K <= 1024). At K=2048 it spills and the 16-row kernel is faster. */
    if (i8_w32() && w->K <= 1024)
        for (; n + 32 <= n1; n += 32) {
        const int m = gemm_cols32_mr4(y, qa, a_scales, w, M, N, n);
        if (m < M) { /* 4-token tail over the two 16-row halves */
            gemm_cols8_256(y + (size_t)m * N, qa + (size_t)m * w->K, a_scales + (size_t)m * nb, w,
                           M - m, N, n, 8);
            gemm_cols8_256(y + (size_t)m * N, qa + (size_t)m * w->K, a_scales + (size_t)m * nb, w,
                           M - m, N, n + 8, 8);
            gemm_cols8_256(y + (size_t)m * N, qa + (size_t)m * w->K, a_scales + (size_t)m * nb, w,
                           M - m, N, n + 16, 8);
            gemm_cols8_256(y + (size_t)m * N, qa + (size_t)m * w->K, a_scales + (size_t)m * nb, w,
                           M - m, N, n + 24, 8);
        }
    }
    for (; n + 16 <= n1; n += 16) {
        int m = gemm_cols16_mr8(y, qa, a_scales, w, M, N, n);
        if (m < M) { /* token tail: both 8-row halves of this column group */
            gemm_cols8_256(y + (size_t)m * N, qa + (size_t)m * w->K,
                           a_scales + (size_t)m * nb, w, M - m, N, n, 8);
            gemm_cols8_256(y + (size_t)m * N, qa + (size_t)m * w->K,
                           a_scales + (size_t)m * nb, w, M - m, N, n + 8, 8);
        }
    }
    for (; n + 8 <= n1; n += 8) gemm_cols8_256(y, qa, a_scales, w, M, N, n, 8);
    if (n < n1) gemm_cols8_256(y, qa, a_scales, w, M, N, n, n1 - n);
}

/* Per-block symmetric activation quantization; 16 elements per step and one horizontal max. */
#define CL16(v) (clamp ? _mm512_min_ps(_mm512_max_ps((v), vlo), vhi) : (v))

void e2_quantize_blocks_avx512(int8_t *out, float *scales, const float *x, int M, int K, float b_lo,
                               float b_hi) {
    const __m512 vlo = _mm512_set1_ps(b_lo), vhi = _mm512_set1_ps(b_hi);
    const int clamp = b_lo > -INFINITY || b_hi < INFINITY;
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        int8_t *orow = out + (size_t)m * K;
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
            __m512 vmax = _mm512_setzero_ps();
            int k = k0;
            for (; k + 16 <= k1; k += 16)
                vmax = _mm512_max_ps(vmax, _mm512_and_ps(CL16(_mm512_loadu_ps(xr + k)), absmask));
            __m256 lo = _mm512_castps512_ps256(vmax), hi = _mm512_extractf32x8_ps(vmax, 1);
            __m256 sv = _mm256_max_ps(lo, hi);
            __m128 slo = _mm256_castps256_ps128(sv), shi = _mm256_extractf128_ps(sv, 1);
            __m128 s = _mm_max_ps(slo, shi);
            s = _mm_max_ps(s, _mm_shuffle_ps(s, s, 0x4e));
            s = _mm_max_ps(s, _mm_shuffle_ps(s, s, 0xb1));
            float amax = _mm_cvtss_f32(s);
            for (; k < k1; k++) {
                const float v = clamp ? (xr[k] < b_lo ? b_lo : (xr[k] > b_hi ? b_hi : xr[k])) : xr[k];
                const float a = fabsf(v);
                if (a > amax) amax = a;
            }
            const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
            const __m512 inv = _mm512_set1_ps(1.0f / sc);
            const __m512i bias = _mm512_set1_epi8((char)0x80);
            k = k0;
            for (; k + 16 <= k1; k += 16) {
                __m512i i = _mm512_cvtps_epi32(_mm512_mul_ps(CL16(_mm512_loadu_ps(xr + k)), inv));
                /* 16 x int32 -> 16 x int16 -> 16 x int8; the 64-bit lane interleave of the first
                 * pack is undone with permute4x64(0xd8) before the second. */
                __m256i w16 = _mm256_packs_epi32(_mm512_castsi512_si256(i),
                                                 _mm512_extracti64x4_epi64(i, 1));
                w16 = _mm256_permute4x64_epi64(w16, 0xd8);
                __m128i b8 = _mm_packs_epi16(_mm256_castsi256_si128(w16),
                                             _mm256_extracti128_si256(w16, 1));
                b8 = _mm_xor_si128(b8, _mm512_castsi512_si128(bias));
                _mm_storeu_si128((__m128i *)(orow + k), b8);
            }
            for (; k < k1; k++) {
                const float v = clamp ? (xr[k] < b_lo ? b_lo : (xr[k] > b_hi ? b_hi : xr[k])) : xr[k];
                int32_t r = (int32_t)lrintf(v / sc);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                orow[k] = (int8_t)(r + 128);
            }
            scales[(size_t)m * nb + b] = sc;
        }
    }
}
