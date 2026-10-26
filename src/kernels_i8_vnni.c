/* kernels_i8_vnni.c — int8 GEMM on AVX-VNNI / AVX512-VNNI (vpdpbusd). Compiled with -mavxvnni.
 *
 * Scheme (ggml-Q8_0-like, with per-row weight scales):
 *   w[n,k] = wscale[n] * b[n,k]         b int8, quad-interleaved [K/4][N][4] (see int8.h)
 *   x[m,k] = ascale[m,blk] * a[m,k]     a int8, one scale per E2_I8_BLOCK elements
 *
 * The kernel processes 8 *output rows* per 32-byte instruction: the weight load is one k-quad
 * (4 k-values) for 8 consecutive rows, and the activation is the same 4 bytes broadcast into every
 * 32-bit lane, so lane j of the accumulator holds the partial dot of output row n+j. That is what
 * the quad-interleaved layout buys: the activation operand is 4 bytes for 8 rows x 4 k-values = 32
 * int8 MACs, and it is reused by every token in flight, so activation traffic is 1/8 byte/MAC
 * regardless of the tile. E2_I8_MR tokens are accumulated at once (2 live YMM each: int32 through
 * the block, float across blocks).
 *
 * vpdpbusd wants an unsigned first operand, so the broadcast activation's sign bit is flipped (that
 * adds 128 to every byte) and the precomputed per-block weight sum - stored transposed, one 32-byte
 * load for 8 rows - is subtracted back with a shift.
 *
 * The float bookkeeping is per (token, 8-row group, block), so E2_I8_BLOCK (int8.h) is what decides
 * its share of the uop stream: at 32 it is over half of it, at 128 about a quarter.
 */
#include <immintrin.h>
#include <math.h>

#include "int8.h"
#include "kernels.h"

#define MR E2_I8_MR
#define NQ4 (E2_I8_BLOCK / 4) /* k-quads per activation scale block */

/* store 8 floats, saturated when the weight carries clamp bounds */

/* ---------------------------------------------------------------- packed-operand attention GEMM
 * The 256-bit form of the kernel in kernels_i8_avx512_vnni.c, same contract (int8.h): 8 output
 * columns per weight load, 8 query rows in flight, one 4-byte activation broadcast reused by every
 * row. It is what makes the attention of both towers int8 on AVX-VNNI-only CPUs (Arrow Lake and
 * later consumer parts have vpdpbusd but no AVX-512 at all), where the two products previously fell
 * back to the fp32 sgemm tile: that tile spends 9 of every 17 uops on loads (3.8 MACs/uop), this one
 * 5 of 13 (14.8), and both are limited by the same single dpbusd port.
 *
 * 8 rows need 16 registers for the int32 and float accumulators plus the weight operand, one more
 * than AVX2 has, so the float accumulators are a plain local array: the compiler keeps them in
 * L1-resident stack slots and reloads them once per 128-element block. Taking 6 rows instead would
 * fit in registers but costs 12.5% of the MACs and 50% more weight traffic on a 32-row block. */
void e2_i8_attn_gemm_vnni(float *C, int ldc, const uint8_t *A, int lda, const float *asc,
                          const uint8_t *Bp, const float *bsc, const int32_t *bz, int M, int Np,
                          int N, int K) {
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const int qend = (K + 3) / 4;
    for (int n0 = 0; n0 < N; n0 += 8) {
        const int nn = N - n0 < 8 ? N - n0 : 8;
        /* Bp/bsc/bz are padded to Np, a multiple of 16, so the 8-wide loads below never run off */
        const __m256 bscv = _mm256_loadu_ps(bsc + n0);
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = M - m0 < 8 ? M - m0 : 8;
            __m256i acc[8];
            __m256 f[8];
            for (int i = 0; i < 8; i++) {
                acc[i] = _mm256_setzero_si256();
                f[i] = _mm256_setzero_ps();
            }
            for (int b = 0; b < nb; b++) {
                const uint8_t *ap[8];
                for (int i = 0; i < mm; i++)
                    ap[i] = A + (size_t)(m0 + i) * lda + (size_t)b * E2_I8_BLOCK;
                const int q0 = b * NQ4;
                int q1 = q0 + NQ4;
                if (q1 > qend) q1 = qend;
                for (int q = q0, off = 0; q < q1; q++, off += 4) {
                    const __m256i bv = _mm256_loadu_si256(
                        (const __m256i *)(Bp + ((size_t)q * Np + n0) * 4));
                    for (int i = 0; i < mm; i++)
                        acc[i] = _mm256_dpbusd_epi32(
                            acc[i], _mm256_set1_epi32(e2_load_i32_tail(ap[i] + off, K - q * 4)), bv);
                }
                const __m256i zp = _mm256_slli_epi32(
                    _mm256_loadu_si256((const __m256i *)(bz + (size_t)b * Np + n0)), 7);
                for (int i = 0; i < mm; i++) {
                    const __m256 sc =
                        _mm256_mul_ps(_mm256_set1_ps(asc[(size_t)(m0 + i) * nb + b]), bscv);
                    f[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i], zp)), sc, f[i]);
                    acc[i] = _mm256_setzero_si256();
                }
            }
            for (int i = 0; i < mm; i++) {
                float *dst = C + (size_t)(m0 + i) * ldc + n0;
                if (nn == 8) {
                    _mm256_storeu_ps(dst, f[i]);
                } else {
                    float tmp[8];
                    _mm256_storeu_ps(tmp, f[i]);
                    for (int j = 0; j < nn; j++) dst[j] = tmp[j];
                }
            }
        }
    }
}
static inline void store8(float *dst, __m256 v, __m256 lo, __m256 hi) {
    _mm256_storeu_ps(dst, _mm256_min_ps(_mm256_max_ps(v, lo), hi));
}
/* One 8-row group of columns, rows [n, n+rows) of an 8-aligned n (rows in [1,8]; the weight, scale
 * and block-sum arrays are padded to Npad = ceil(N/8)*8 so the vector loads are always in bounds). */
static void gemm_cols8(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w, int M,
                       int N, int n, int rows) {
    const int K = w->K, nb = w->nblocks, NP = w->Npad;
    const int kq4 = (K + 3) / 4; /* k-quads per row; an 8-row block of them is kq4*32 bytes */
    const int8_t *wp = w->w + (size_t)(n >> 3) * kq4 * 32 + (n & 7) * 4;
    const int32_t *bs = w->bsq + n;
    const __m256 wsc = _mm256_loadu_ps(w->scale + n);
    const __m256 vlo = _mm256_set1_ps(w->out_min), vhi = _mm256_set1_ps(w->out_max);
    int m = 0;
    for (; m + MR <= M; m += MR) {
        __m256i acc[MR];
        __m256 f[MR];
        for (int i = 0; i < MR; i++) {
            acc[i] = _mm256_setzero_si256();
            f[i] = _mm256_setzero_ps();
        }
        for (int b = 0; b < nb; b++) {
            /* activation is stored sign-flipped (int8.h): the broadcast is already the unsigned
             * operand vpdpbusd wants, so no per-token flip. The k-quad offset is shared by every
             * token, the weight pointer walks one 8-row group at a time. */
            const int8_t *ap[MR];
            for (int i = 0; i < MR; i++) ap[i] = qa + (size_t)(m + i) * K + b * E2_I8_BLOCK;
            const int8_t *wpq = wp + (size_t)b * NQ4 * 32;
            for (int q = 0, off = 0; q < NQ4; q++, off += 4, wpq += 32) {
                const __m256i wv = _mm256_loadu_si256((const __m256i *)wpq);
                for (int i = 0; i < MR; i++)
                    acc[i] = _mm256_dpbusd_epi32(
                        acc[i], _mm256_set1_epi32(e2_load_i32(ap[i] + off)), wv);
            }
            /* one 8-lane zero-point and eight activation scales for the block */
            const __m256i zp = _mm256_slli_epi32(
                _mm256_loadu_si256((const __m256i *)(bs + (size_t)b * NP)), 7);
            for (int i = 0; i < MR; i++) {
                const __m256 sc =
                    _mm256_mul_ps(_mm256_set1_ps(a_scales[(size_t)(m + i) * nb + b]), wsc);
                f[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[i], zp)), sc, f[i]);
                acc[i] = _mm256_setzero_si256();
            }
        }
        if (rows == 8) {
            for (int i = 0; i < MR; i++) store8(y + (size_t)(m + i) * N + n, f[i], vlo, vhi);
        } else {
            float tmp[8];
            for (int i = 0; i < MR; i++) {
                _mm256_storeu_ps(tmp, _mm256_min_ps(_mm256_max_ps(f[i], vlo), vhi));
                float *dst = y + (size_t)(m + i) * N + n;
                for (int j = 0; j < rows; j++) dst[j] = tmp[j];
            }
        }
    }
    for (; m < M; m++) { /* token tail */
        __m256i acc = _mm256_setzero_si256();
        __m256 f = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const int8_t *ap = qa + (size_t)m * K + b * E2_I8_BLOCK;
            const int8_t *wpq = wp + (size_t)b * NQ4 * 32;
            for (int q = 0, off = 0; q < NQ4; q++, off += 4, wpq += 32) {
                const __m256i wv = _mm256_loadu_si256((const __m256i *)wpq);
                acc = _mm256_dpbusd_epi32(
                    acc, _mm256_set1_epi32(e2_load_i32(ap + off)), wv);
            }
            const __m256i zp = _mm256_slli_epi32(
                _mm256_loadu_si256((const __m256i *)(bs + (size_t)b * NP)), 7);
            const __m256 sc = _mm256_mul_ps(_mm256_set1_ps(a_scales[(size_t)m * nb + b]), wsc);
            f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc, zp)), sc, f);
            acc = _mm256_setzero_si256();
        }
        if (rows == 8) {
            store8(y + (size_t)m * N + n, f, vlo, vhi);
        } else {
            float tmp[8];
            _mm256_storeu_ps(tmp, _mm256_min_ps(_mm256_max_ps(f, vlo), vhi));
            float *dst = y + (size_t)m * N + n;
            for (int j = 0; j < rows; j++) dst[j] = tmp[j];
        }
    }
}

void e2_i8_gemm_vnni(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                     int M, int N, int n0, int n1) {
    int n = n0;
    for (; n + 8 <= n1; n += 8) gemm_cols8(y, qa, a_scales, w, M, N, n, 8);
    if (n < n1) gemm_cols8(y, qa, a_scales, w, M, N, n, n1 - n);
}

/* The tile drops output rows it cannot fill, so the entry point has to finish the range: the engine
 * splits N into arbitrary tiles (N/(8*threads) rows each), which are almost never multiples of 8.
 * Silently losing the tail is what made the first version of this tile look 1.3x faster and score
 * 0.94 cosine. */
void e2_i8_gemm_vnni_range(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                           int M, int N, int n0, int n1) {
    e2_i8_gemm_vnni(y, qa, a_scales, w, M, N, n0, n1);
}

/* Per-block symmetric activation quantization to int8. */
void e2_quantize_blocks_vnni(int8_t *out, float *scales, const float *x, int M, int K, float b_lo, float b_hi) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 vlo = _mm256_set1_ps(b_lo), vhi = _mm256_set1_ps(b_hi);
    const int clamp = b_lo > -INFINITY || b_hi < INFINITY;
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        int8_t *orow = out + (size_t)m * K;
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
            __m256 vmax = _mm256_setzero_ps();
            int k = k0;
            for (; k + 8 <= k1; k += 8) {
                __m256 v = _mm256_loadu_ps(xr + k);
                if (clamp) v = _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
                vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, absmask));
            }
            __m128 lo = _mm256_castps256_ps128(vmax), hi = _mm256_extractf128_ps(vmax, 1);
            __m128 sv = _mm_max_ps(lo, hi);
            sv = _mm_max_ps(sv, _mm_shuffle_ps(sv, sv, 0x4e));
            sv = _mm_max_ps(sv, _mm_shuffle_ps(sv, sv, 0xb1));
            float amax = _mm_cvtss_f32(sv);
            for (; k < k1; k++) {
                float v = clamp ? (xr[k] < b_lo ? b_lo : (xr[k] > b_hi ? b_hi : xr[k])) : xr[k];
                float a = fabsf(v);
                if (a > amax) amax = a;
            }
            const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
            const __m256 inv = _mm256_set1_ps(1.0f / sc);
            k = k0;
            const __m256i bias = _mm256_set1_epi8((char)0x80); /* store a+128: unsigned operand */
            for (; k + 8 <= k1; k += 8) {
                __m256 v = _mm256_loadu_ps(xr + k);
                if (clamp) v = _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
                __m256i i = _mm256_cvtps_epi32(_mm256_mul_ps(v, inv));
                __m128i pk = _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
                pk = _mm_packs_epi16(pk, pk);
                pk = _mm_xor_si128(pk, _mm256_castsi256_si128(bias));
                _mm_storel_epi64((__m128i *)(orow + k), pk);
            }
            for (; k < k1; k++) {
                float v = clamp ? (xr[k] < b_lo ? b_lo : (xr[k] > b_hi ? b_hi : xr[k])) : xr[k];
                int32_t r = (int32_t)lrintf(v / sc);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                orow[k] = (int8_t)(r + 128);
            }
            scales[(size_t)m * nb + b] = sc;
        }
    }
}
