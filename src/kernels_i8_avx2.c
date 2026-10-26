/* kernels_i8_avx2.c — int8 GEMM for AVX2 without AVX-VNNI (Haswell..Zen3), same quantization scheme
 * and the same quad-interleaved weight layout as the VNNI kernel (see int8.h).
 *
 * Both operands are int8 here. The activation is broadcast into all eight 32-bit lanes, so the
 * abs/sign trick runs per lane: |a| (one shared vector per token) x (b * sign(a)) can never
 * saturate vpmaddubsw, and every lane ends up holding one output row's partial dot, exactly like
 * the vpdpbusd path.
 */
#include <immintrin.h>
#include <math.h>

#include "int8.h"
#include "kernels.h"

#define MR E2_I8_MR
#define NQ4 (E2_I8_BLOCK / 4)

/* 4 int8 MACs per 32-bit lane, 8 rows at a time, without saturation: |a| in the unsigned position
 * and a's sign folded into b. */
static inline __m256i dot4(__m256i absa, __m256i bv, __m256i av) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(absa, _mm256_sign_epi8(bv, av)),
                             _mm256_set1_epi16(1));
}

static void gemm_cols8(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w, int M,
                       int N, int n, int rows) {
    const int K = w->K, nb = w->nblocks;
    const __m256i flip = _mm256_set1_epi8((char)0x80); /* activations are stored sign-flipped */
    const int kq4 = (K + 3) / 4;
    const int8_t *wp = w->w + (size_t)(n >> 3) * kq4 * 32 + (size_t)(n & 7) * 4;
    const float *wsc = w->scale + n;
    const __m256 lo_ = _mm256_set1_ps(w->out_min), hi_ = _mm256_set1_ps(w->out_max);
    int m = 0;
    for (; m + MR <= M; m += MR) {
        __m256i acc[MR];
        __m256 f[MR];
        for (int i = 0; i < MR; i++) {
            acc[i] = _mm256_setzero_si256();
            f[i] = _mm256_setzero_ps();
        }
        for (int b = 0; b < nb; b++) {
            const float *sp = a_scales + (size_t)m * nb + b;
            for (int q = 0; q < NQ4; q++) {
                const int k = b * E2_I8_BLOCK + q * 4;
                __m256i av[MR], u[MR];
                for (int i = 0; i < MR; i++) {
                    av[i] = _mm256_xor_si256(
                        _mm256_set1_epi32(e2_load_i32(qa + (size_t)(m + i) * K + k)), flip);
                    u[i] = _mm256_sign_epi8(av[i], av[i]); /* |a| */
                }
                const __m256i wv = _mm256_loadu_si256(
                    (const __m256i *)(wp + (size_t)(b * NQ4 + q) * 32));
                for (int i = 0; i < MR; i++) acc[i] = _mm256_add_epi32(acc[i], dot4(u[i], wv, av[i]));
            }
            for (int i = 0; i < MR; i++) {
                const __m256 sc = _mm256_mul_ps(
                    _mm256_set1_ps(sp[i * nb]), _mm256_loadu_ps(wsc));
                f[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i]), sc, f[i]);
                acc[i] = _mm256_setzero_si256();
            }
        }
        if (rows == 8) {
            for (int i = 0; i < MR; i++)
                _mm256_storeu_ps(y + (size_t)(m + i) * N + n, _mm256_min_ps(_mm256_max_ps(f[i], lo_), hi_));
        } else {
            float tmp[8];
            for (int i = 0; i < MR; i++) {
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
            for (int q = 0; q < NQ4; q++) {
                const int k = b * E2_I8_BLOCK + q * 4;
                const __m256i av = _mm256_xor_si256(
                    _mm256_set1_epi32(e2_load_i32(qa + (size_t)m * K + k)), flip);
                const __m256i wv = _mm256_loadu_si256(
                    (const __m256i *)(wp + (size_t)(b * NQ4 + q) * 32));
                acc = _mm256_add_epi32(acc,
                                       dot4(_mm256_sign_epi8(av, av), wv, av));
            }
            const __m256 sc = _mm256_mul_ps(_mm256_set1_ps(a_scales[(size_t)m * nb + b]),
                                            _mm256_loadu_ps(wsc));
            f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sc, f);
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

void e2_i8_gemm_avx2(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                     int M, int N, int n0, int n1) {
    int n = n0;
    for (; n + 8 <= n1; n += 8) gemm_cols8(y, qa, a_scales, w, M, N, n, 8);
    if (n < n1) gemm_cols8(y, qa, a_scales, w, M, N, n, n1 - n);
}

#define CL8(v) (clamp ? _mm256_min_ps(_mm256_max_ps((v), vlo), vhi) : (v))

void e2_quantize_blocks_avx2(int8_t *out, float *scales, const float *x, int M, int K, float b_lo, float b_hi) {
    const __m256 vlo = _mm256_set1_ps(b_lo), vhi = _mm256_set1_ps(b_hi);
    const int clamp = b_lo > -INFINITY || b_hi < INFINITY;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        int8_t *orow = out + (size_t)m * K;
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
            __m256 vmax = _mm256_setzero_ps();
            int k = k0;
            for (; k + 8 <= k1; k += 8)
                vmax = _mm256_max_ps(vmax, _mm256_and_ps(CL8(_mm256_loadu_ps(xr + k)), absmask));
            __m128 lo = _mm256_castps256_ps128(vmax), hi = _mm256_extractf128_ps(vmax, 1);
            __m128 sv = _mm_max_ps(lo, hi);
            sv = _mm_max_ps(sv, _mm_shuffle_ps(sv, sv, 0x4e));
            sv = _mm_max_ps(sv, _mm_shuffle_ps(sv, sv, 0xb1));
            float amax = _mm_cvtss_f32(sv);
            for (; k < k1; k++) {
                const float v = clamp ? (xr[k] < b_lo ? b_lo : (xr[k] > b_hi ? b_hi : xr[k])) : xr[k];
                const float a = fabsf(v);
                if (a > amax) amax = a;
            }
            const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
            const __m256 inv = _mm256_set1_ps(1.0f / sc);
            k = k0;
            const __m128i bias = _mm_set1_epi8((char)0x80); /* store a+128: see int8.h */
            for (; k + 8 <= k1; k += 8) {
                __m256i i = _mm256_cvtps_epi32(_mm256_mul_ps(CL8(_mm256_loadu_ps(xr + k)), inv));
                __m128i pk = _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
                pk = _mm_packs_epi16(pk, pk);
                pk = _mm_xor_si128(pk, bias);
                _mm_storel_epi64((__m128i *)(orow + k), pk);
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
