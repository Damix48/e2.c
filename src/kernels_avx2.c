/* kernels_avx2.c — AVX2 + FMA backend. Compiled with -mavx2 -mfma; selected at runtime.
 *
 * Everything here is float32 math with bf16 weights converted on the fly, which is what the
 * checkpoint stores. The GEMM is blocked as
 *
 *     for k-block:  for n-block:  for m:  dot(x[m,k0:k0+KB], w[n,k0:k0+KB]) for all n in the block
 *
 * so the weight matrix is streamed exactly once (it is the only large operand) while the activation
 * rows stay in L1/L2. For a single query M is small (15..200) which is exactly the case where memory
 * traffic decides the runtime.
 */
#include <immintrin.h>
#include <math.h>

#include "kernels.h"
#include "threads.h"

#define KB 256 /* k-block: 1 KB of fp32 activations */
#define NB 8   /* n-block: 8 weight rows at a time */

/* ---------------------------------------------------------------- vector primitives */
static inline __m256 exp_ps(__m256 x) {
    /* Range reduction followed by exp(r) = 1 + r + r*r*P(r). */
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 ln2hi = _mm256_set1_ps(0.6931471805599453f);
    const __m256 ln2lo = _mm256_set1_ps(-1.904654299957768e-9f);
    const __m256 c5 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 c4 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 c3 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 c2 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 c1 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 c0 = _mm256_set1_ps(5.0000001201e-1f);

    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f)); /* clamp: below this, exp underflows to 0 */
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, ln2hi, x);
    r = _mm256_fnmadd_ps(n, ln2lo, r);

    __m256 p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

    /* scale by 2^n by adding n to the exponent field */
    __m256i ni = _mm256_cvtps_epi32(n);
    __m256i scale = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(scale));
}

static inline __m256 tanh_ps(__m256 z) {
    /* tanh(z) = 1 - 2/(e^{2z}+1), odd symmetry keeps precision for negative inputs */
    __m256 a = _mm256_and_ps(z, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    __m256 e = exp_ps(_mm256_add_ps(a, a));
    __m256 r = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, _mm256_set1_ps(1.0f))));
    /* saturate: for a > 15 the rational form loses precision */
    __m256 big = _mm256_set1_ps(15.0f);
    r = _mm256_blendv_ps(r, _mm256_set1_ps(1.0f), _mm256_cmp_ps(a, big, _CMP_GT_OQ));
    return _mm256_xor_ps(r, _mm256_and_ps(z, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))));
}

static inline __m256 load_bf16_8(const uint16_t *p) {
    __m128i w16 = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w16), 16));
}

/* dot of 8 floats * 8 bf16 we already converted */
static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float dot_f32_n(const float *a, const float *b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= n; k += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
    float s = hsum_ps(acc);
    for (; k < n; k++) s += a[k] * b[k];
    return s;
}

static inline float dot_bf16_n(const float *a, const uint16_t *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= n; k += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), load_bf16_8(b + k), acc);
    float s = hsum_ps(acc);
    for (; k < n; k++) s += a[k] * e2_bf16f(b[k]);
    return s;
}

/* ---------------------------------------------------------------- GEMM / GEMV */
struct gemm_task {
    float *y;
    const float *x;
    const void *w;
    int M, N, K;
    e2_dtype wd;
};

/* One weight row is streamed once and used for every activation row: the weight matrix is the only
 * large operand, so it must move exactly once. Activations stay in L1/L2 across the n loop. */
/* Stack scratch for one converted weight row; also the per-thread score buffer lives in the caller. */
#define E2_WS_ROW 4096

static void gemm_range(struct gemm_task *t, int n0, int n1, float *rowbuf) {
    const float *wf = (const float *)t->w;
    const uint16_t *wb = (const uint16_t *)t->w;
    for (int n = n0; n < n1; n++) {
        const float *wrow;
        if (t->wd == E2_DT_BF16) {
            /* convert the row once, then every token reuses it: bf16 conversion is 2 extra ops per
             * element, which would otherwise dominate an inner loop that is M tokens long */
            for (int k = 0; k < t->K; k++) rowbuf[k] = e2_bf16f(wb[(size_t)n * t->K + k]);
            wrow = rowbuf;
        } else {
            wrow = wf + (size_t)n * t->K;
        }
        for (int m = 0; m < t->M; m++)
            t->y[(size_t)m * t->N + n] = dot_f32_n(t->x + (size_t)m * t->K, wrow, (size_t)t->K);
    }
}

static void gemm_worker(void *arg, int tid, int nthreads) {
    struct gemm_task *t = arg;
    const int per = (t->N + nthreads - 1) / nthreads;
    const int n0 = tid * per;
    int n1 = n0 + per;
    if (n1 > t->N) n1 = t->N;
    if (n0 >= n1) return;
    float rowbuf[E2_WS_ROW];
    if (t->K <= E2_WS_ROW) {
        gemm_range(t, n0, n1, rowbuf);
    } else { /* very wide rows: convert per (n,m) instead of buffering */
        for (int n = n0; n < n1; n++)
            for (int m = 0; m < t->M; m++) {
                float acc = 0.f;
                for (int k = 0; k < t->K; k++)
                    acc += t->x[(size_t)m * t->K + k] * e2_bf16f(((const uint16_t *)t->w)[(size_t)n * t->K + k]);
                t->y[(size_t)m * t->N + n] = acc;
            }
    }
}

static void avx2_gemm_nt(float *y, const float *x, const void *w, int M, int N, int K, e2_dtype wd) {
    struct gemm_task t = { y, x, w, M, N, K, wd };
    e2_parallel(gemm_worker, &t);
}

static void avx2_gemv(float *y, const void *w, const float *x, int N, int K, e2_dtype wd) {
    const float *wf = (const float *)w;
    const uint16_t *wb = (const uint16_t *)w;
    for (int n = 0; n < N; n++)
        y[n] = (wd == E2_DT_BF16) ? dot_bf16_n(x, wb + (size_t)n * K, K) : dot_f32_n(x, wf + (size_t)n * K, K);
}

/* ---------------------------------------------------------------- norms / activations */
static void avx2_rmsnorm(float *x, int M, int D, const float *gamma, float eps) {
    for (int m = 0; m < M; m++) {
        float *row = x + (size_t)m * D;
        __m256 ss = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 v = _mm256_loadu_ps(row + d);
            ss = _mm256_fmadd_ps(v, v, ss);
        }
        float s = hsum_ps(ss);
        for (; d < D; d++) s += row[d] * row[d];
        float inv = 1.0f / sqrtf(s / (float)D + eps);
        __m256 vinv = _mm256_set1_ps(inv);
        d = 0;
        if (gamma) {
            for (; d + 8 <= D; d += 8)
                _mm256_storeu_ps(row + d, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row + d), vinv),
                                                       _mm256_loadu_ps(gamma + d)));
        } else {
            for (; d + 8 <= D; d += 8)
                _mm256_storeu_ps(row + d, _mm256_mul_ps(_mm256_loadu_ps(row + d), vinv));
        }
        for (; d < D; d++) row[d] = row[d] * inv * (gamma ? gamma[d] : 1.0f);
    }
}

static void avx2_gelu_tanh(float *x, size_t n) {
    const __m256 c = _mm256_set1_ps(0.7978845608028654f);
    const __m256 c3 = _mm256_set1_ps(0.044715f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
        __m256 inner = _mm256_mul_ps(c, _mm256_fmadd_ps(c3, v3, v));
        __m256 t = tanh_ps(inner);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, t)));
    }
    for (; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

static void avx2_softmax(float *x, int n) {
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 8 <= n; i += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    float m = 0.f;
    {
        __m128 lo = _mm256_castps256_ps128(vmax), hi = _mm256_extractf128_ps(vmax, 1);
        __m128 s = _mm_max_ps(lo, hi);
        s = _mm_max_ps(s, _mm_shuffle_ps(s, s, 0x4e));
        s = _mm_max_ps(s, _mm_shuffle_ps(s, s, 0xb1));
        m = _mm_cvtss_f32(s);
    }
    for (; i < n; i++) if (x[i] > m) m = x[i];

    __m256 vm = _mm256_set1_ps(m), vsum = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 e = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm));
        _mm256_storeu_ps(x + i, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float s = hsum_ps(vsum);
    for (; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv = 1.0f / s;
    __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    for (; i < n; i++) x[i] *= inv;
}

/* softmax + per-128-block int8 quantisation in one pass (int8.h). The 256-bit form of the AVX-512
 * kernel: the only consumer of the probabilities in an int8 attention is the quantiser, so the
 * exp() pass, the row sum and the per-block max are fused and the fp32 normalisation is never
 * written back as a separate array. Rows in [n, rows) are untouched. */
void e2_softmax_quant_i8_avx2(float *x, int8_t *q8, float *scales, int rows, int n, int nb) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * n;
        int8_t *q = q8 + (size_t)r * n;
        float *sc = scales + (size_t)r * nb;
        int i = 0;
        __m256 vmax = _mm256_set1_ps(-INFINITY);
        for (; i + 8 <= n; i += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + i));
        __m128 lo = _mm256_castps256_ps128(vmax), hi = _mm256_extractf128_ps(vmax, 1);
        __m128 mv = _mm_max_ps(lo, hi);
        mv = _mm_max_ps(mv, _mm_shuffle_ps(mv, mv, 0x4e));
        mv = _mm_max_ps(mv, _mm_shuffle_ps(mv, mv, 0xb1));
        float m = _mm_cvtss_f32(mv);
        for (; i < n; i++) if (row[i] > m) m = row[i];

        const __m256 vm = _mm256_set1_ps(m);
        __m256 vsum = _mm256_setzero_ps();
        float tail = 0.f;
        float bmax[32]; /* >= ceil(E2_MAX_SEQ / E2_I8_BLOCK) */
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < n ? k0 + E2_I8_BLOCK : n;
            __m256 bm = _mm256_setzero_ps();
            i = k0;
            for (; i + 8 <= k1; i += 8) {
                const __m256 e = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + i), vm));
                _mm256_storeu_ps(row + i, e);
                bm = _mm256_max_ps(bm, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            __m128 blo = _mm256_castps256_ps128(bm), bhi = _mm256_extractf128_ps(bm, 1);
            __m128 bv = _mm_max_ps(blo, bhi);
            bv = _mm_max_ps(bv, _mm_shuffle_ps(bv, bv, 0x4e));
            bv = _mm_max_ps(bv, _mm_shuffle_ps(bv, bv, 0xb1));
            float bf = _mm_cvtss_f32(bv);
            for (; i < k1; i++) {
                const float e = expf(row[i] - m);
                row[i] = e;
                if (e > bf) bf = e;
                tail += e;
            }
            bmax[b] = bf;
        }
        const float inv = 1.0f / (hsum_ps(vsum) + tail);

        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < n ? k0 + E2_I8_BLOCK : n;
            const float bf = bmax[b];
            const __m256 vr = _mm256_set1_ps(bf > 0.f ? 127.0f / bf : 0.f);
            sc[b] = bf > 0.f ? bf * inv / 127.0f : 0.f;
            i = k0;
            for (; i + 8 <= k1; i += 8) {
                const __m256i v =
                    _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(row + i), vr));
                __m128i pk = _mm_packs_epi32(_mm256_castsi256_si128(v),
                                             _mm256_extracti128_si256(v, 1));
                pk = _mm_packs_epi16(pk, pk);
                pk = _mm_xor_si128(pk, _mm256_castsi256_si128(bias));
                _mm_storel_epi64((__m128i *)(q + i), pk);
            }
            for (; i < k1; i++) {
                int v = (int)lrintf(row[i] * (bf > 0.f ? 127.0f / bf : 0.f));
                if (v > 127) v = 127; else if (v < -127) v = -127;
                q[i] = (int8_t)(v + 128);
            }
        }
    }
}

/* y += a * x: the attention value accumulation, 4 heads x S x S x head_dim per layer */
static void avx2_axpy(float *y, const float *x, float a, int n) {
    const __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}

/* rotate_half: y[0:d/2] = a*cos - b*sin ; y[d/2:] = b*cos + a*sin, 8 lanes at a time */
static void avx2_rope_rows(float *x, int r0, int r1, int ngroups, int head_dim, int block,
                           const float *cs, const float *sn) {
    const int half = block / 2;
    for (int r = r0; r < r1; r++) {
        float *v = x + (size_t)r * head_dim;
        const float *c = cs + (size_t)(r / ngroups) * head_dim;
        const float *s = sn + (size_t)(r / ngroups) * head_dim;
        for (int b = 0; b < head_dim; b += block)
            for (int i = 0; i < half; i += 8) {
                __m256 a = _mm256_loadu_ps(v + b + i);
                __m256 y = _mm256_loadu_ps(v + b + half + i);
                __m256 cc = _mm256_loadu_ps(c + b + i);
                __m256 ss = _mm256_loadu_ps(s + b + i);
                _mm256_storeu_ps(v + b + i, _mm256_fmsub_ps(a, cc, _mm256_mul_ps(y, ss)));
                _mm256_storeu_ps(v + b + half + i, _mm256_fmadd_ps(a, ss, _mm256_mul_ps(y, cc)));
            }
    }
}

/* ---------------------------------------------------------------- blocked f32 GEMM
 * C[M,N] = A[M,K] @ B[K,N], row-major with strides. 64x64x64 blocking with an 8x8 register tile:
 * the A and B panels stay in L1 while the tile walks them, which is what keeps the attention logits
 * (K=64) and the context product (N=64, K=patches) off the memory system. ~14 MACs/cycle.
 */
static inline void sge_micro8(float *C, int ldc, const float *A, int lda, const float *B, int ldb,
                              int kk, int accumulate) {
    __m256 acc[8];
    for (int i = 0; i < 8; i++)
        acc[i] = accumulate ? _mm256_loadu_ps(C + (size_t)i * ldc) : _mm256_setzero_ps();
    for (int k = 0; k < kk; k++) {
        const __m256 bv = _mm256_loadu_ps(B + (size_t)k * ldb);
        for (int i = 0; i < 8; i++)
            acc[i] = _mm256_fmadd_ps(_mm256_set1_ps(A[(size_t)i * lda + k]), bv, acc[i]);
    }
    for (int i = 0; i < 8; i++) _mm256_storeu_ps(C + (size_t)i * ldc, acc[i]);
}

void e2_sgemm_nn_range_avx2(float *C, int ldc, const float *A, int lda, const float *B, int ldb, int M, int N,
                            int K, int m0, int m1) {
    for (int mb = m0; mb < m1; mb += 64) {
        const int me = m1 - mb < 64 ? m1 - mb : 64;
        for (int k0 = 0; k0 < K; k0 += 64) {
            const int kk = K - k0 < 64 ? K - k0 : 64;
            for (int n0 = 0; n0 < N; n0 += 64) {
                const int nn = N - n0 < 64 ? N - n0 : 64;
                for (int i = 0; i + 8 <= me; i += 8) {
                    int j = 0;
                    for (; j + 8 <= nn; j += 8)
                        sge_micro8(C + (size_t)(mb + i) * ldc + n0 + j, ldc,
                                   A + (size_t)(mb + i) * lda + k0, lda,
                                   B + (size_t)k0 * ldb + n0 + j, ldb, kk, k0 != 0);
                    for (; j < nn; j++) { /* column tail */
                        float acc[8] = { 0 };
                        for (int k = 0; k < kk; k++)
                            for (int i2 = 0; i2 < 8; i2++)
                                acc[i2] += A[(size_t)(mb + i + i2) * lda + k0 + k] *
                                           B[(size_t)(k0 + k) * ldb + n0 + j];
                        for (int i2 = 0; i2 < 8; i2++) {
                            float *cp = C + (size_t)(mb + i + i2) * ldc + n0 + j;
                            *cp = k0 ? *cp + acc[i2] : acc[i2];
                        }
                    }
                }
                for (int i = (me & ~7); i < me; i++) { /* row tail */
                    for (int j = 0; j < nn; j++) {
                        float acc = 0;
                        for (int k = 0; k < kk; k++)
                            acc += A[(size_t)(mb + i) * lda + k0 + k] * B[(size_t)(k0 + k) * ldb + n0 + j];
                        float *cp = C + (size_t)(mb + i) * ldc + n0 + j;
                        *cp = k0 ? *cp + acc : acc;
                    }
                }
            }
        }
    }
    (void)M;
}


/* ---------------------------------------------------------------- elementwise map ops
 * Same arithmetic order as the scalar versions in kernels.c, so the two agree bit for bit on
 * mul/add/scale/copy and only on float summation order for the two norm fusions. */
/* sigmoid(x) = 1 / (1 + exp(-x)), on the same exp_ps the softmax uses */
static inline __m256 sigmoid_ps(__m256 x) {
    return _mm256_div_ps(_mm256_set1_ps(1.0f),
                         _mm256_add_ps(_mm256_set1_ps(1.0f), exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}
static void avx2_silu(float *x, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, sigmoid_ps(v)));
    }
    for (; i < n; i++) x[i] *= 1.0f / (1.0f + expf(-x[i]));
}
static void avx2_glu(float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), sigmoid_ps(_mm256_loadu_ps(b + i))));
    for (; i < n; i++) a[i] *= 1.0f / (1.0f + expf(-b[i]));
}

static void avx2_clamp(float *x, float lo, float hi, size_t n) {
    const __m256 vlo = _mm256_set1_ps(lo), vhi = _mm256_set1_ps(hi);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(x + i), vlo), vhi));
    for (; i < n; i++) x[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
}

static void avx2_mul(float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) a[i] *= b[i];
}
static void avx2_add(float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) a[i] += b[i];
}
static void avx2_scale(float *a, float s, size_t n) {
    const __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vs));
    for (; i < n; i++) a[i] *= s;
}
static void avx2_mul_rows(float *a, const float *b, int rows, int D, int stride) {
    for (int r = 0; r < rows; r++) avx2_mul(a + (size_t)r * D, b + (size_t)r * stride, (size_t)D);
}
/* inv for one row: the same 8-wide tree the rmsnorm kernel reduces with */
static inline float row_inv(const float *x, int D, float eps) {
    __m256 ss = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= D; d += 8) {
        const __m256 v = _mm256_loadu_ps(x + d);
        ss = _mm256_fmadd_ps(v, v, ss);
    }
    float s = hsum_ps(ss);
    for (; d < D; d++) s += x[d] * x[d];
    return 1.0f / sqrtf(s / (float)D + eps);
}
static void avx2_norm_split(float *x, float *res, int rows, int D, const float *gamma, float eps) {
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * D;
        float *rr = res + (size_t)r * D;
        const float inv = row_inv(xr, D, eps);
        const __m256 vinv = _mm256_set1_ps(inv);
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            const __m256 v = _mm256_loadu_ps(xr + d);
            _mm256_storeu_ps(rr + d, v);
            _mm256_storeu_ps(xr + d, _mm256_mul_ps(_mm256_mul_ps(v, vinv),
                                                   _mm256_loadu_ps(gamma + d)));
        }
        for (; d < D; d++) { const float v = xr[d]; rr[d] = v; xr[d] = v * inv * gamma[d]; }
    }
}
static void avx2_norm_add(float *out, const float *src, const float *res, int rows, int D,
                          const float *gamma, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *sr = src + (size_t)r * D;
        const float *rr = res + (size_t)r * D;
        float *orow = out + (size_t)r * D;
        const float inv = row_inv(sr, D, eps);
        const __m256 vinv = _mm256_set1_ps(inv);
        int d = 0;
        for (; d + 8 <= D; d += 8)
            _mm256_storeu_ps(orow + d,
                             _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(sr + d), vinv),
                                                         _mm256_loadu_ps(gamma + d)),
                                           _mm256_loadu_ps(rr + d)));
        for (; d < D; d++) orow[d] = sr[d] * inv * gamma[d] + rr[d];
    }
}

/* ---------------------------------------------------------------- backend handle */
static const e2_kernels g_avx2 = {
    .name = "avx2+fma",
    .gemm_nt = avx2_gemm_nt,
    .gemv = avx2_gemv,
    .rmsnorm = avx2_rmsnorm,
    .gelu_tanh = avx2_gelu_tanh,
    .softmax = avx2_softmax,
    .dot_f32 = dot_f32_n,
    .axpy = avx2_axpy,
    .mul = avx2_mul,
    .add = avx2_add,
    .scale = avx2_scale,
    .mul_rows = avx2_mul_rows,
    .norm_split = avx2_norm_split,
    .norm_add = avx2_norm_add,
    .rope_rows = avx2_rope_rows,
    .silu = avx2_silu,
    .glu = avx2_glu,
    .clamp = avx2_clamp,
};

const e2_kernels *e2_avx2_kernels(void) { return &g_avx2; }
