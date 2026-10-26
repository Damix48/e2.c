/* kernels_avx512.c — AVX-512F/BW/VL/DQ + FMA backend. Compiled with -mavx512f -mavx512bw
 * -mavx512vl -mavx512dq -mfma; selected at runtime. Mirrors kernels_avx2.c with 16-wide vectors.
 *
 * The blocked attention GEMM is the reason this backend exists beyond the int8 path: the sgemm
 * micro-kernel goes from an 8x8 to a 16x16 register tile, which halves the B-panel loads per MAC
 * and doubles the FMA width on the Zen 5's full-width 512-bit pipes. The elementwise/norm kernels
 * are simple 16-wide translations of the AVX2 ones.
 */
#include <immintrin.h>
#include <math.h>

#include "kernels.h"
#include "threads.h"

#define KB 256 /* k-block: 1 KB of fp32 activations */
#define NB 8   /* n-block: 8 weight rows at a time */

/* ---------------------------------------------------------------- vector primitives */
static inline __m512 exp_ps512(__m512 x) {
    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    const __m512 ln2hi = _mm512_set1_ps(0.6931471805599453f);
    const __m512 ln2lo = _mm512_set1_ps(-1.904654299957768e-9f);
    const __m512 c5 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 c4 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 c3 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 c2 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 c1 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 c0 = _mm512_set1_ps(5.0000001201e-1f);

    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, ln2hi, x);
    r = _mm512_fnmadd_ps(n, ln2lo, r);

    __m512 p = c5;
    p = _mm512_fmadd_ps(p, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, c0);
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r);
    p = _mm512_add_ps(p, _mm512_set1_ps(1.0f));

    __m512i ni = _mm512_cvtps_epi32(n);
    __m512i scale = _mm512_slli_epi32(_mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(scale));
}

static inline __m512 tanh_ps512(__m512 z) {
    __m512 a = _mm512_and_ps(z, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)));
    __m512 e = exp_ps512(_mm512_add_ps(a, a));
    __m512 r = _mm512_sub_ps(_mm512_set1_ps(1.0f),
                             _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(e, _mm512_set1_ps(1.0f))));
    __mmask16 big = _mm512_cmp_ps_mask(a, _mm512_set1_ps(15.0f), _CMP_GT_OQ);
    r = _mm512_mask_mov_ps(r, big, _mm512_set1_ps(1.0f));
    return _mm512_xor_ps(r, _mm512_and_ps(z, _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000))));
}

static inline __m512 load_bf16_16(const uint16_t *p) {
    __m256i w16 = _mm256_loadu_si256((const __m256i *)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(w16), 16));
}

static inline float hsum_ps512(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v), hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s = _mm256_add_ps(lo, hi);
    __m128 l = _mm256_castps256_ps128(s), h = _mm256_extractf128_ps(s, 1);
    __m128 r = _mm_add_ps(l, h);
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(r);
}

static inline float dot_f32_n512(const float *a, const float *b, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    size_t k = 0;
    for (; k + 16 <= n; k += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), acc);
    float s = hsum_ps512(acc);
    for (; k < n; k++) s += a[k] * b[k];
    return s;
}

static inline float dot_bf16_n512(const float *a, const uint16_t *b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), load_bf16_16(b + k), acc);
    float s = hsum_ps512(acc);
    for (; k < n; k++) s += a[k] * e2_bf16f(b[k]);
    return s;
}

/* ---------------------------------------------------------------- GEMM / GEMV */
struct gemm_task512 {
    float *y;
    const float *x;
    const void *w;
    int M, N, K;
    e2_dtype wd;
};

#define E2_WS_ROW 4096

static void gemm_range512(struct gemm_task512 *t, int n0, int n1, float *rowbuf) {
    const float *wf = (const float *)t->w;
    const uint16_t *wb = (const uint16_t *)t->w;
    for (int n = n0; n < n1; n++) {
        const float *wrow;
        if (t->wd == E2_DT_BF16) {
            for (int k = 0; k < t->K; k++) rowbuf[k] = e2_bf16f(wb[(size_t)n * t->K + k]);
            wrow = rowbuf;
        } else {
            wrow = wf + (size_t)n * t->K;
        }
        for (int m = 0; m < t->M; m++)
            t->y[(size_t)m * t->N + n] = dot_f32_n512(t->x + (size_t)m * t->K, wrow, (size_t)t->K);
    }
}

static void gemm_worker512(void *arg, int tid, int nthreads) {
    struct gemm_task512 *t = arg;
    const int per = (t->N + nthreads - 1) / nthreads;
    const int n0 = tid * per;
    int n1 = n0 + per;
    if (n1 > t->N) n1 = t->N;
    if (n0 >= n1) return;
    float rowbuf[E2_WS_ROW];
    if (t->K <= E2_WS_ROW) {
        gemm_range512(t, n0, n1, rowbuf);
    } else {
        for (int n = n0; n < n1; n++)
            for (int m = 0; m < t->M; m++) {
                float acc = 0.f;
                for (int k = 0; k < t->K; k++)
                    acc += t->x[(size_t)m * t->K + k] *
                           e2_bf16f(((const uint16_t *)t->w)[(size_t)n * t->K + k]);
                t->y[(size_t)m * t->N + n] = acc;
            }
    }
}

static void avx512_gemm_nt(float *y, const float *x, const void *w, int M, int N, int K, e2_dtype wd) {
    struct gemm_task512 t = { y, x, w, M, N, K, wd };
    e2_parallel(gemm_worker512, &t);
}

static void avx512_gemv(float *y, const void *w, const float *x, int N, int K, e2_dtype wd) {
    const float *wf = (const float *)w;
    const uint16_t *wb = (const uint16_t *)w;
    for (int n = 0; n < N; n++)
        y[n] = (wd == E2_DT_BF16) ? dot_bf16_n512(x, wb + (size_t)n * K, K)
                                  : dot_f32_n512(x, wf + (size_t)n * K, K);
}

/* ---------------------------------------------------------------- norms / activations */
static void avx512_rmsnorm(float *x, int M, int D, const float *gamma, float eps) {
    for (int m = 0; m < M; m++) {
        float *row = x + (size_t)m * D;
        __m512 ss = _mm512_setzero_ps();
        int d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 v = _mm512_loadu_ps(row + d);
            ss = _mm512_fmadd_ps(v, v, ss);
        }
        float s = hsum_ps512(ss);
        for (; d < D; d++) s += row[d] * row[d];
        float inv = 1.0f / sqrtf(s / (float)D + eps);
        __m512 vinv = _mm512_set1_ps(inv);
        d = 0;
        if (gamma) {
            for (; d + 16 <= D; d += 16)
                _mm512_storeu_ps(row + d,
                                 _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(row + d), vinv),
                                               _mm512_loadu_ps(gamma + d)));
        } else {
            for (; d + 16 <= D; d += 16)
                _mm512_storeu_ps(row + d, _mm512_mul_ps(_mm512_loadu_ps(row + d), vinv));
        }
        for (; d < D; d++) row[d] = row[d] * inv * (gamma ? gamma[d] : 1.0f);
    }
}

static void avx512_gelu_tanh(float *x, size_t n) {
    const __m512 c = _mm512_set1_ps(0.7978845608028654f);
    const __m512 c3 = _mm512_set1_ps(0.044715f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
        __m512 inner = _mm512_mul_ps(c, _mm512_fmadd_ps(c3, v3, v));
        __m512 t = tanh_ps512(inner);
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_mul_ps(half, v), _mm512_add_ps(one, t)));
    }
    for (; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

static void avx512_softmax(float *x, int n) {
    __m512 vmax = _mm512_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 16 <= n; i += 16) vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(x + i));
    float m = _mm512_reduce_max_ps(vmax);
    for (; i < n; i++) if (x[i] > m) m = x[i];

    __m512 vm = _mm512_set1_ps(m), vsum = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 e = exp_ps512(_mm512_sub_ps(_mm512_loadu_ps(x + i), vm));
        _mm512_storeu_ps(x + i, e);
        vsum = _mm512_add_ps(vsum, e);
    }
    float s = hsum_ps512(vsum);
    for (; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv = 1.0f / s;
    __m512 vinv = _mm512_set1_ps(inv);
    i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vinv));
    for (; i < n; i++) x[i] *= inv;
}

/* The max reduction must be a max, not a sum: a sum over the -inf padding lanes would poison m
 * and the exponential then clamps at 88, which changes the probabilities. */

/* softmax + per-128-block int8 quantisation in one kernel (int8.h): the only consumer of the
 * probabilities in an int8 attention is the quantiser, so doing both here removes the fp32
 * normalisation write and one full read of the (P,P) score block per head. The quantiser's own
 * max pass comes for free from the exp pass. */
void e2_softmax_quant_i8_avx512(float *x, int8_t *q8, float *scales, int rows, int n, int nb) {
    const __m512i bias = _mm512_set1_epi8((char)0x80);
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * n;
        int8_t *q = q8 + (size_t)r * n;
        float *sc = scales + (size_t)r * nb;
        int i = 0;
        __m512 vmax = _mm512_set1_ps(-INFINITY);
        for (; i + 16 <= n; i += 16) vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(row + i));
        float m = _mm512_reduce_max_ps(vmax);
        for (; i < n; i++) if (row[i] > m) m = row[i];

        const __m512 vm = _mm512_set1_ps(m);
        __m512 vsum = _mm512_setzero_ps();
        float tail_sum = 0.f;
        float bmax[32]; /* >= ceil(E2_V_MAX_PATCHES/E2_I8_BLOCK) */
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < n ? k0 + E2_I8_BLOCK : n;
            __m512 bm = _mm512_setzero_ps();
            i = k0;
            for (; i + 16 <= k1; i += 16) {
                const __m512 e = exp_ps512(_mm512_sub_ps(_mm512_loadu_ps(row + i), vm));
                _mm512_storeu_ps(row + i, e);
                bm = _mm512_max_ps(bm, e);
                vsum = _mm512_add_ps(vsum, e);
            }
            float bf = _mm512_reduce_max_ps(bm);
            for (; i < k1; i++) {
                const float e = expf(row[i] - m);
                row[i] = e;
                if (e > bf) bf = e;
                tail_sum += e;
            }
            bmax[b] = bf;
        }
        const float s = _mm512_reduce_add_ps(vsum) + tail_sum;
        const float inv = s > 0.f ? 1.0f / s : 0.f;

        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < n ? k0 + E2_I8_BLOCK : n;
            const float bf = bmax[b];
            const float rscale = bf > 0.f ? 127.0f / bf : 0.f;
            sc[b] = bf > 0.f ? bf * inv / 127.0f : 0.f;
            const __m512 vr = _mm512_set1_ps(rscale);
            i = k0;
            for (; i + 16 <= k1; i += 16) {
                const __m512i v = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(row + i), vr));
                __m256i w16 = _mm256_packs_epi32(_mm512_castsi512_si256(v),
                                                 _mm512_extracti64x4_epi64(v, 1));
                w16 = _mm256_permute4x64_epi64(w16, 0xd8);
                __m128i b8 = _mm_packs_epi16(_mm256_castsi256_si128(w16),
                                             _mm256_extracti128_si256(w16, 1));
                b8 = _mm_xor_si128(b8, _mm512_castsi512_si128(bias));
                _mm_storeu_si128((__m128i *)(q + i), b8);
            }
            for (; i < k1; i++) {
                int v = (int)lrintf(row[i] * rscale);
                if (v > 127) v = 127; else if (v < -127) v = -127;
                q[i] = (int8_t)(v + 128);
            }
        }
    }
}

static void avx512_axpy(float *y, const float *x, float a, int n) {
    const __m512 va = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}

static void avx512_rope_rows(float *x, int r0, int r1, int ngroups, int head_dim, int block,
                             const float *cs, const float *sn) {
    const int half = block / 2;
    for (int r = r0; r < r1; r++) {
        float *v = x + (size_t)r * head_dim;
        const float *c = cs + (size_t)(r / ngroups) * head_dim;
        const float *s = sn + (size_t)(r / ngroups) * head_dim;
        for (int b = 0; b < head_dim; b += block) {
            int i = 0;
            for (; i + 16 <= half; i += 16) {
                __m512 a = _mm512_loadu_ps(v + b + i);
                __m512 y = _mm512_loadu_ps(v + b + half + i);
                __m512 cc = _mm512_loadu_ps(c + b + i);
                __m512 ss = _mm512_loadu_ps(s + b + i);
                _mm512_storeu_ps(v + b + i, _mm512_fmsub_ps(a, cc, _mm512_mul_ps(y, ss)));
                _mm512_storeu_ps(v + b + half + i, _mm512_fmadd_ps(a, ss, _mm512_mul_ps(y, cc)));
            }
            for (; i < half; i++) {
                float a = v[b + i], y = v[b + half + i];
                v[b + i] = a * c[b + i] - y * s[b + i];
                v[b + half + i] = y * c[b + i] + a * s[b + i];
            }
        }
    }
}

/* ---------------------------------------------------------------- blocked f32 GEMM
 * 64x64x64 blocking with a 16x16 register tile (A and B panels stay in L1 while the tile walks
 * them). The row and column tails go through the masked micro-kernel, so no scalar fallback.
 */
static inline void sge_micro16_full(float *C, int ldc, const float *A, int lda, const float *B,
                                    int ldb, int kk, int accumulate) {
    __m512 acc[16];
    for (int i = 0; i < 16; i++)
        acc[i] = accumulate ? _mm512_loadu_ps(C + (size_t)i * ldc) : _mm512_setzero_ps();
    for (int k = 0; k < kk; k++) {
        const __m512 bv = _mm512_loadu_ps(B + (size_t)k * ldb);
        for (int i = 0; i < 16; i++)
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(A[(size_t)i * lda + k]), bv, acc[i]);
    }
    for (int i = 0; i < 16; i++) _mm512_storeu_ps(C + (size_t)i * ldc, acc[i]);
}

static inline void sge_micro16_part(float *C, int ldc, const float *A, int lda, const float *B,
                                    int ldb, int kk, int accumulate, int rows, int cols) {
    const __mmask16 cm = (__mmask16)((1u << cols) - 1u);
    __m512 acc[16];
    for (int i = 0; i < 16; i++)
        acc[i] = (i < rows && accumulate)
                     ? _mm512_maskz_loadu_ps(cm, C + (size_t)i * ldc)
                     : _mm512_setzero_ps();
    for (int k = 0; k < kk; k++) {
        const __m512 bv = _mm512_maskz_loadu_ps(cm, B + (size_t)k * ldb);
        for (int i = 0; i < 16; i++) {
            const float av = i < rows ? A[(size_t)i * lda + k] : 0.f;
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(av), bv, acc[i]);
        }
    }
    for (int i = 0; i < rows; i++)
        _mm512_mask_storeu_ps(C + (size_t)i * ldc, cm, acc[i]);
}

void e2_sgemm_nn_range_avx512(float *C, int ldc, const float *A, int lda, const float *B, int ldb,
                              int M, int N, int K, int m0, int m1) {
    for (int mb = m0; mb < m1; mb += 64) {
        const int me = m1 - mb < 64 ? m1 - mb : 64;
        for (int k0 = 0; k0 < K; k0 += 64) {
            const int kk = K - k0 < 64 ? K - k0 : 64;
            for (int n0 = 0; n0 < N; n0 += 64) {
                const int nn = N - n0 < 64 ? N - n0 : 64;
                int i = 0;
                for (; i + 16 <= me; i += 16) {
                    int j = 0;
                    for (; j + 16 <= nn; j += 16)
                        sge_micro16_full(C + (size_t)(mb + i) * ldc + n0 + j, ldc,
                                         A + (size_t)(mb + i) * lda + k0, lda,
                                         B + (size_t)k0 * ldb + n0 + j, ldb, kk, k0 != 0);
                    for (; j < nn; j += 16)
                        sge_micro16_part(C + (size_t)(mb + i) * ldc + n0 + j, ldc,
                                         A + (size_t)(mb + i) * lda + k0, lda,
                                         B + (size_t)k0 * ldb + n0 + j, ldb, kk, k0 != 0, 16,
                                         nn - j);
                }
                for (; i < me; i += 16) {
                    const int rows = me - i < 16 ? me - i : 16;
                    int j = 0;
                    for (; j + 16 <= nn; j += 16)
                        sge_micro16_part(C + (size_t)(mb + i) * ldc + n0 + j, ldc,
                                         A + (size_t)(mb + i) * lda + k0, lda,
                                         B + (size_t)k0 * ldb + n0 + j, ldb, kk, k0 != 0, rows, 16);
                    for (; j < nn; j += 16)
                        sge_micro16_part(C + (size_t)(mb + i) * ldc + n0 + j, ldc,
                                         A + (size_t)(mb + i) * lda + k0, lda,
                                         B + (size_t)k0 * ldb + n0 + j, ldb, kk, k0 != 0, rows,
                                         nn - j);
                }
            }
        }
    }
    (void)M;
}

/* ---------------------------------------------------------------- elementwise map ops */
static inline __m512 sigmoid_ps512(__m512 x) {
    return _mm512_div_ps(_mm512_set1_ps(1.0f),
                         _mm512_add_ps(_mm512_set1_ps(1.0f), exp_ps512(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}
static void avx512_silu(float *x, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(x + i, _mm512_mul_ps(v, sigmoid_ps512(v)));
    }
    for (; i < n; i++) x[i] *= 1.0f / (1.0f + expf(-x[i]));
}
static void avx512_glu(float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), sigmoid_ps512(_mm512_loadu_ps(b + i))));
    for (; i < n; i++) a[i] *= 1.0f / (1.0f + expf(-b[i]));
}

static void avx512_clamp(float *x, float lo, float hi, size_t n) {
    const __m512 vlo = _mm512_set1_ps(lo), vhi = _mm512_set1_ps(hi);
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(x + i, _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(x + i), vlo), vhi));
    for (; i < n; i++) x[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
}

static void avx512_mul(float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; i++) a[i] *= b[i];
}
static void avx512_add(float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(a + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; i++) a[i] += b[i];
}
static void avx512_scale(float *a, float s, size_t n) {
    const __m512 vs = _mm512_set1_ps(s);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), vs));
    for (; i < n; i++) a[i] *= s;
}
static void avx512_mul_rows(float *a, const float *b, int rows, int D, int stride) {
    for (int r = 0; r < rows; r++) avx512_mul(a + (size_t)r * D, b + (size_t)r * stride, (size_t)D);
}
static inline float row_inv512(const float *x, int D, float eps) {
    __m512 ss = _mm512_setzero_ps();
    int d = 0;
    for (; d + 16 <= D; d += 16) {
        const __m512 v = _mm512_loadu_ps(x + d);
        ss = _mm512_fmadd_ps(v, v, ss);
    }
    float s = hsum_ps512(ss);
    for (; d < D; d++) s += x[d] * x[d];
    return 1.0f / sqrtf(s / (float)D + eps);
}
static void avx512_norm_split(float *x, float *res, int rows, int D, const float *gamma, float eps) {
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * D;
        float *rr = res + (size_t)r * D;
        const float inv = row_inv512(xr, D, eps);
        const __m512 vinv = _mm512_set1_ps(inv);
        int d = 0;
        for (; d + 16 <= D; d += 16) {
            const __m512 v = _mm512_loadu_ps(xr + d);
            _mm512_storeu_ps(rr + d, v);
            _mm512_storeu_ps(xr + d, _mm512_mul_ps(_mm512_mul_ps(v, vinv),
                                                   _mm512_loadu_ps(gamma + d)));
        }
        for (; d < D; d++) { const float v = xr[d]; rr[d] = v; xr[d] = v * inv * gamma[d]; }
    }
}
static void avx512_norm_add(float *out, const float *src, const float *res, int rows, int D,
                            const float *gamma, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *sr = src + (size_t)r * D;
        const float *rr = res + (size_t)r * D;
        float *orow = out + (size_t)r * D;
        const float inv = row_inv512(sr, D, eps);
        const __m512 vinv = _mm512_set1_ps(inv);
        int d = 0;
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(orow + d,
                             _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(sr + d), vinv),
                                                         _mm512_loadu_ps(gamma + d)),
                                           _mm512_loadu_ps(rr + d)));
        for (; d < D; d++) orow[d] = sr[d] * inv * gamma[d] + rr[d];
    }
}

/* ---------------------------------------------------------------- backend handle */
static const e2_kernels g_avx512 = {
    .name = "avx512+fma",
    .gemm_nt = avx512_gemm_nt,
    .gemv = avx512_gemv,
    .rmsnorm = avx512_rmsnorm,
    .gelu_tanh = avx512_gelu_tanh,
    .softmax = avx512_softmax,
    .dot_f32 = dot_f32_n512,
    .axpy = avx512_axpy,
    .mul = avx512_mul,
    .add = avx512_add,
    .scale = avx512_scale,
    .mul_rows = avx512_mul_rows,
    .norm_split = avx512_norm_split,
    .norm_add = avx512_norm_add,
    .rope_rows = avx512_rope_rows,
    .silu = avx512_silu,
    .glu = avx512_glu,
    .clamp = avx512_clamp,
};

const e2_kernels *e2_avx512_kernels(void) { return &g_avx512; }
