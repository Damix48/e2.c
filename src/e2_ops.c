/* e2_ops.c - elementwise, norm, rope and attention kernels. */
#include "e2_internal.h"

#include <math.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define E2_X86OPS 1
#endif

/* ------------------------------------------------------------------ */
/* SIMD building blocks                                               */
/* ------------------------------------------------------------------ */

typedef struct { float *p; int n, chunk; void (*fn)(float *, int); } split_arg;

static void split_work(void *vp, int tid, int nt) {
    split_arg *a = vp;
    int lo = tid * a->chunk;
    if (lo >= a->n) return;
    int hi = lo + a->chunk;
    if (hi > a->n) hi = a->n;
    a->fn(a->p + lo, hi - lo);
}

/* Run fn over [0,n) split across threads (only worth it above 64k elems). */
static void run_split(e2_ctx *ctx, void (*fn)(float *, int), float *p, int n) {
    if (!ctx || ctx->nthreads <= 1 || n < 65536) { fn(p, n); return; }
    int nt = ctx->nthreads;
    if (n / 65536 < nt) nt = n / 65536;
    if (nt < 1) nt = 1;
    int chunk = (int)(((long)n + nt - 1) / nt);
    chunk = (chunk + 63) & ~63;
    split_arg a = {p, n, chunk, fn};
    e2_parallel(ctx, split_work, &a);
}

static inline float gelu1(float x) {
    const float k = 0.7978845608028654f;  /* sqrt(2/pi) */
    float t = k * (x + 0.044715f * x * x * x);
    if (t > 15.0f) return x;
    return x / (1.0f + e2_fexp(-2.0f * t));
}

static void gelu_s(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = gelu1(x[i]);
}
static void silu_s(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + e2_fexp(-x[i]));
}
static void scale_s(float *x, int n, float k) { for (int i = 0; i < n; i++) x[i] *= k; }

#ifdef E2_X86OPS
/* exp(x) for 16 floats, rel. error < 2e-6 like e2_fexp. */
__attribute__((target("avx512f")))
static inline __m512 vexp(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.3f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.7f));
    const __m512 LOG2E = _mm512_set1_ps(1.4426950408889634f);
    const __m512 LN2 = _mm512_set1_ps(0.6931471805599453f);
    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, LOG2E),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, LN2, x);
    __m512 p = _mm512_set1_ps(1.0f / 120.0f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

/* 1/(1+exp(-x)) with one Newton step on a 14-bit reciprocal. */
__attribute__((target("avx512f")))
static inline __m512 vsigmoid(__m512 x) {
    __m512 d = _mm512_add_ps(_mm512_set1_ps(1.0f), vexp(_mm512_sub_ps(_mm512_setzero_ps(), x)));
    __m512 r = _mm512_rcp14_ps(d);
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(d, r, _mm512_set1_ps(2.0f)));
    return r;
}

__attribute__((target("avx512f")))
static void gelu_v(float *x, int n) {
    const __m512 k = _mm512_set1_ps(0.7978845608028654f);
    const __m512 c = _mm512_set1_ps(0.044715f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 big = _mm512_set1_ps(15.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 t = _mm512_mul_ps(k, _mm512_fmadd_ps(c, _mm512_mul_ps(v, _mm512_mul_ps(v, v)), v));
        __m512 s = vsigmoid(_mm512_mul_ps(two, _mm512_min_ps(t, big)));
        /* t > 15 -> gelu(x) = x */
        __mmask16 sat = _mm512_cmp_ps_mask(t, big, _CMP_GT_OQ);
        __m512 r = _mm512_mul_ps(v, s);
        _mm512_storeu_ps(x + i, _mm512_mask_mov_ps(r, sat, v));
    }
    for (; i < n; i++) x[i] = gelu1(x[i]);
}

__attribute__((target("avx512f")))
static void silu_v(float *x, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vsigmoid(_mm512_loadu_ps(x + i))));
    for (; i < n; i++) x[i] = x[i] / (1.0f + e2_fexp(-x[i]));
}

__attribute__((target("avx512f")))
static void scale_v(float *x, int n, float k) {
    __m512 vk = _mm512_set1_ps(k);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vk));
    for (; i < n; i++) x[i] *= k;
}

__attribute__((target("avx512f")))
static void rmsnorm_v(float *out, const float *x, const float *w, int rows, int dim, float eps) {
    float inv_dim = 1.0f / (float)dim;
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 64 <= dim; i += 64) {
            __m512 v0 = _mm512_loadu_ps(xp + i), v1 = _mm512_loadu_ps(xp + i + 16);
            __m512 v2 = _mm512_loadu_ps(xp + i + 32), v3 = _mm512_loadu_ps(xp + i + 48);
            a0 = _mm512_fmadd_ps(v0, v0, a0); a1 = _mm512_fmadd_ps(v1, v1, a1);
            a2 = _mm512_fmadd_ps(v2, v2, a2); a3 = _mm512_fmadd_ps(v3, v3, a3);
        }
        for (; i + 16 <= dim; i += 16) {
            __m512 v0 = _mm512_loadu_ps(xp + i);
            a0 = _mm512_fmadd_ps(v0, v0, a0);
        }
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1),
                                                       _mm512_add_ps(a2, a3)));
        for (; i < dim; i++) sum += xp[i] * xp[i];
        __m512 vs = _mm512_set1_ps(1.0f / sqrtf(sum * inv_dim + eps));
        i = 0;
        if (w) {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(op + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(xp + i), vs),
                                                       _mm512_loadu_ps(w + i)));
        } else {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(op + i, _mm512_mul_ps(_mm512_loadu_ps(xp + i), vs));
        }
        for (; i < dim; i++) op[i] = w ? xp[i] * (vs[0] * w[i]) : xp[i] * vs[0];
    }
}

__attribute__((target("avx512f")))
static void l2norm_v(float *x, int rows, int dim) {
    for (int r = 0; r < rows; r++) {
        float *p = x + (size_t)r * dim;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0;
        int i = 0;
        for (; i + 32 <= dim; i += 32) {
            __m512 v0 = _mm512_loadu_ps(p + i), v1 = _mm512_loadu_ps(p + i + 16);
            a0 = _mm512_fmadd_ps(v0, v0, a0); a1 = _mm512_fmadd_ps(v1, v1, a1);
        }
        for (; i + 16 <= dim; i += 16) { __m512 v0 = _mm512_loadu_ps(p + i); a0 = _mm512_fmadd_ps(v0, v0, a0); }
        float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        for (; i < dim; i++) s += p[i] * p[i];
        __m512 vs = _mm512_set1_ps(1.0f / sqrtf(s + 1e-12f));
        for (i = 0; i + 16 <= dim; i += 16)
            _mm512_storeu_ps(p + i, _mm512_mul_ps(_mm512_loadu_ps(p + i), vs));
        for (; i < dim; i++) p[i] *= vs[0];
    }
}

__attribute__((target("avx512f")))
static void add_v(float *a, const float *b, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(a + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; i++) a[i] += b[i];
}

/* Softmax over n scores: exp + normalise, vectorised. */
static void softmax_exp_v(float *sc, int n) {
    float mx = e2_span_max(sc, n);
    float sum = e2_exp_sum(sc, n, mx);
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) sc[i] *= inv;
}
#endif

static int simd_pick(void) {
#ifdef E2_X86OPS
    if (getenv("E2_OPS_SCALAR")) return 0;   /* debugging: force the scalar ops */
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return 2;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return 1;
#endif
    return 0;
}
static int g_simd = -1;
static inline int simd(void) { if (g_simd < 0) g_simd = simd_pick(); return g_simd; }

#ifdef E2_X86OPS
__attribute__((target("avx512f")))
static float span_max_v(const float *x, int n) {
    int i = 0;
    __m512 vm = _mm512_set1_ps(-INFINITY);
    for (; i + 16 <= n; i += 16) vm = _mm512_max_ps(vm, _mm512_loadu_ps(x + i));
    float mx = _mm512_reduce_max_ps(vm);
    for (; i < n; i++) if (x[i] > mx) mx = x[i];
    return mx;
}

__attribute__((target("avx512f")))
static float exp_sum_v(float *x, int n, float mx) {
    int i = 0;
    __m512 vmx = _mm512_set1_ps(mx), vs = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 e = vexp(_mm512_sub_ps(_mm512_loadu_ps(x + i), vmx));
        _mm512_storeu_ps(x + i, e);
        vs = _mm512_add_ps(vs, e);
    }
    float sum = _mm512_reduce_add_ps(vs);
    for (; i < n; i++) { x[i] = e2_fexp(x[i] - mx); sum += x[i]; }
    return sum;
}
#endif

/* ------------------------------------------------------------------ */
/* AVX2 + FMA twins of the kernels above.  Hosts without AVX-512       */
/* (Arrow Lake, Alder Lake, Zen 2/3/4) get 8 lanes instead of 1.       */
/* ------------------------------------------------------------------ */
#if defined(E2_X86OPS)
#define A2 __attribute__((target("avx2,fma")))
A2 static void rot_a2(float *x, const float *c, const float *s, int half);
A2 static void gelu_mul_a2(float *o, const float *x, const float *up, int n);

A2 static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

A2 static inline float hmax8(__m256 v) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_max_ss(s, _mm_shuffle_ps(s, s, 1)));
}

/* Same polynomial and clamping as vexp; 2^n comes from the exponent field. */
A2 static inline __m256 vexp8(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.7f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931471805599453f), x);
    __m256 p = _mm256_set1_ps(1.0f / 120.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

A2 static inline __m256 vsigmoid8(__m256 x) {
    __m256 d = _mm256_add_ps(_mm256_set1_ps(1.0f),
                             vexp8(_mm256_sub_ps(_mm256_setzero_ps(), x)));
    __m256 r = _mm256_rcp_ps(d);
    return _mm256_mul_ps(r, _mm256_fnmadd_ps(d, r, _mm256_set1_ps(2.0f)));
}

A2 static void gelu_a2(float *x, int n) {
    const __m256 k = _mm256_set1_ps(0.7978845608028654f);
    const __m256 c = _mm256_set1_ps(0.044715f);
    const __m256 big = _mm256_set1_ps(15.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 t = _mm256_mul_ps(k, _mm256_fmadd_ps(c, _mm256_mul_ps(v, _mm256_mul_ps(v, v)), v));
        __m256 s = vsigmoid8(_mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_min_ps(t, big)));
        __m256 r = _mm256_mul_ps(v, s);
        _mm256_storeu_ps(x + i, _mm256_blendv_ps(r, v, _mm256_cmp_ps(t, big, _CMP_GT_OQ)));
    }
    for (; i < n; i++) x[i] = gelu1(x[i]);
}

A2 static void silu_a2(float *x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, vsigmoid8(v)));
    }
    for (; i < n; i++) x[i] = x[i] / (1.0f + e2_fexp(-x[i]));
}

A2 static void tanh_a2(float *x, int n) {
    const __m256 two = _mm256_set1_ps(2.0f);
    /* tanh(+-20) is already +-1 in f32; clamping keeps the exp finite for the
     * +-inf sentinels callers use (see the audio attention mask) */
    const __m256 lim = _mm256_set1_ps(20.0f), nlim = _mm256_set1_ps(-20.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_max_ps(_mm256_min_ps(v, lim), nlim);
        _mm256_storeu_ps(x + i, _mm256_fmsub_ps(vsigmoid8(_mm256_mul_ps(v, two)), two,
                                                _mm256_set1_ps(1.0f)));
    }
    for (; i < n; i++) x[i] = tanhf(x[i]);
}

A2 static void gated_silu_a2(float *x, int rows, int dim) {
    for (int r = 0; r < rows; r++) {
        float *a = x + (size_t)r * 2 * dim;
        const float *b = a + dim;
        int i = 0;
        for (; i + 8 <= dim; i += 8)
            _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                                  vsigmoid8(_mm256_loadu_ps(b + i))));
        for (; i < dim; i++) a[i] *= 1.0f / (1.0f + e2_fexp(-b[i]));
    }
}

A2 static void scale_a2(float *x, int n, float k) {
    __m256 vk = _mm256_set1_ps(k);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vk));
    for (; i < n; i++) x[i] *= k;
}

A2 static void add_a2(float *a, const float *b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) a[i] += b[i];
}

A2 static float span_max_a2(const float *x, int n) {
    int i = 0;
    __m256 vm = _mm256_set1_ps(-INFINITY);
    for (; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(x + i));
    float mx = hmax8(vm);
    for (; i < n; i++) if (x[i] > mx) mx = x[i];
    return mx;
}

A2 static float exp_sum_a2(float *x, int n, float mx) {
    int i = 0;
    __m256 vmx = _mm256_set1_ps(mx), vs = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 e = vexp8(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmx));
        _mm256_storeu_ps(x + i, e);
        vs = _mm256_add_ps(vs, e);
    }
    float sum = hsum8(vs);
    for (; i < n; i++) { x[i] = e2_fexp(x[i] - mx); sum += x[i]; }
    return sum;
}

A2 static void rmsnorm_a2(float *out, const float *x, const float *w, int rows,
                          int dim, float eps) {
    float inv_dim = 1.0f / (float)dim;
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 32 <= dim; i += 32) {
            __m256 v0 = _mm256_loadu_ps(xp + i), v1 = _mm256_loadu_ps(xp + i + 8);
            __m256 v2 = _mm256_loadu_ps(xp + i + 16), v3 = _mm256_loadu_ps(xp + i + 24);
            a0 = _mm256_fmadd_ps(v0, v0, a0); a1 = _mm256_fmadd_ps(v1, v1, a1);
            a2 = _mm256_fmadd_ps(v2, v2, a2); a3 = _mm256_fmadd_ps(v3, v3, a3);
        }
        for (; i + 8 <= dim; i += 8) {
            __m256 v0 = _mm256_loadu_ps(xp + i);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
        }
        float sum = hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; i < dim; i++) sum += xp[i] * xp[i];
        float sv = 1.0f / sqrtf(sum * inv_dim + eps);
        __m256 vs = _mm256_set1_ps(sv);
        i = 0;
        if (w) {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(op + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xp + i), vs),
                                                       _mm256_loadu_ps(w + i)));
        } else {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(op + i, _mm256_mul_ps(_mm256_loadu_ps(xp + i), vs));
        }
        for (; i < dim; i++) op[i] = w ? xp[i] * (sv * w[i]) : xp[i] * sv;
    }
}

A2 static void rmsnorm_pre_a2(float *out, const float *x, const float *w, int rows,
                              int dim, float eps, float pre) {
    float inv_dim = 1.0f / (float)dim;
    __m256 vp = _mm256_set1_ps(pre);
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 32 <= dim; i += 32) {
            __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(xp + i), vp);
            __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(xp + i + 8), vp);
            __m256 v2 = _mm256_mul_ps(_mm256_loadu_ps(xp + i + 16), vp);
            __m256 v3 = _mm256_mul_ps(_mm256_loadu_ps(xp + i + 24), vp);
            a0 = _mm256_fmadd_ps(v0, v0, a0); a1 = _mm256_fmadd_ps(v1, v1, a1);
            a2 = _mm256_fmadd_ps(v2, v2, a2); a3 = _mm256_fmadd_ps(v3, v3, a3);
        }
        for (; i + 8 <= dim; i += 8) {
            __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(xp + i), vp);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
        }
        float sum = hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; i < dim; i++) { float v = xp[i] * pre; sum += v * v; }
        float sv = 1.0f / sqrtf(sum * inv_dim + eps);
        __m256 vs = _mm256_set1_ps(sv);
        i = 0;
        if (w) {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(op + i,
                    _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xp + i), vp), vs),
                                  _mm256_loadu_ps(w + i)));
        } else {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(op + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xp + i), vp), vs));
        }
        for (; i < dim; i++) {
            float s = pre * sv;
            op[i] = w ? xp[i] * s * w[i] : xp[i] * s;
        }
    }
}

A2 static void rmsnorm_add_a2(float *x, const float *src, const float *w, int rows,
                              int dim, float eps) {
    float inv_dim = 1.0f / (float)dim;
    for (int r = 0; r < rows; r++) {
        const float *sp = src + (size_t)r * dim;
        float *xp = x + (size_t)r * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 32 <= dim; i += 32) {
            __m256 v0 = _mm256_loadu_ps(sp + i), v1 = _mm256_loadu_ps(sp + i + 8);
            __m256 v2 = _mm256_loadu_ps(sp + i + 16), v3 = _mm256_loadu_ps(sp + i + 24);
            a0 = _mm256_fmadd_ps(v0, v0, a0); a1 = _mm256_fmadd_ps(v1, v1, a1);
            a2 = _mm256_fmadd_ps(v2, v2, a2); a3 = _mm256_fmadd_ps(v3, v3, a3);
        }
        for (; i + 8 <= dim; i += 8) {
            __m256 v0 = _mm256_loadu_ps(sp + i);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
        }
        float sum = hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; i < dim; i++) sum += sp[i] * sp[i];
        float sv = 1.0f / sqrtf(sum * inv_dim + eps);
        __m256 vs = _mm256_set1_ps(sv);
        i = 0;
        if (w) {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(xp + i, _mm256_add_ps(
                    _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(sp + i), vs),
                                  _mm256_loadu_ps(w + i)),
                    _mm256_loadu_ps(xp + i)));
        } else {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(xp + i, _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(sp + i), vs), _mm256_loadu_ps(xp + i)));
        }
        for (; i < dim; i++) xp[i] += w ? sp[i] * (sv * w[i]) : sp[i] * sv;
    }
}

A2 static void l2norm_a2(float *x, int rows, int dim) {
    for (int r = 0; r < rows; r++) {
        float *p = x + (size_t)r * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 32 <= dim; i += 32) {
            __m256 v0 = _mm256_loadu_ps(p + i), v1 = _mm256_loadu_ps(p + i + 8);
            __m256 v2 = _mm256_loadu_ps(p + i + 16), v3 = _mm256_loadu_ps(p + i + 24);
            a0 = _mm256_fmadd_ps(v0, v0, a0); a1 = _mm256_fmadd_ps(v1, v1, a1);
            a2 = _mm256_fmadd_ps(v2, v2, a2); a3 = _mm256_fmadd_ps(v3, v3, a3);
        }
        for (; i + 8 <= dim; i += 8) {
            __m256 v0 = _mm256_loadu_ps(p + i);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
        }
        float s = hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; i < dim; i++) s += p[i] * p[i];
        float sv = 1.0f / sqrtf(s + 1e-12f);
        __m256 vs = _mm256_set1_ps(sv);
        for (i = 0; i + 8 <= dim; i += 8)
            _mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), vs));
        for (; i < dim; i++) p[i] *= sv;
    }
}
#endif

/* max and exp(x-mx) over a span; the attention softmax needs both without the
 * normalisation (it folds 1/sum into the P x V epilogue). */
#ifdef E2_X86OPS
__attribute__((target("avx512f")))
static void tanh_v(float *x, int n) {
    const __m512 two = _mm512_set1_ps(2.0f), one = _mm512_set1_ps(1.0f);
    const __m512 lim = _mm512_set1_ps(20.0f), nlim = _mm512_set1_ps(-20.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_max_ps(_mm512_min_ps(_mm512_loadu_ps(x + i), lim), nlim);
        __m512 s = vsigmoid(_mm512_mul_ps(v, two));
        _mm512_storeu_ps(x + i, _mm512_sub_ps(_mm512_mul_ps(s, two), one));
    }
    for (; i < n; i++) x[i] = tanhf(x[i]);
}

/* x[r][d] *= sigmoid(x[r][dim+d]) -- the gated feed-forward */
__attribute__((target("avx512f")))
static void gated_silu_v(float *x, int rows, int dim) {
    for (int r = 0; r < rows; r++) {
        float *a = x + (size_t)r * 2 * dim;
        const float *b = a + dim;
        int i = 0;
        for (; i + 16 <= dim; i += 16)
            _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i),
                                                 vsigmoid(_mm512_loadu_ps(b + i))));
        for (; i < dim; i++) a[i] *= 1.0f / (1.0f + e2_fexp(-b[i]));
    }
}
#endif

void e2_tanh_span(float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) { tanh_v(x, n); return; }
    if (simd() == 1) { tanh_a2(x, n); return; }
#endif
    for (int i = 0; i < n; i++) x[i] = tanhf(x[i]);
}

void e2_gated_silu(float *x, int rows, int dim) {
#ifdef E2_X86OPS
    if (simd() >= 2) { gated_silu_v(x, rows, dim); return; }
    if (simd() == 1) { gated_silu_a2(x, rows, dim); return; }
#endif
    for (int r = 0; r < rows; r++) {
        float *a = x + (size_t)r * 2 * dim;
        for (int i = 0; i < dim; i++)
            a[i] *= 1.0f / (1.0f + e2_fexp(-a[dim + i]));
    }
}

float e2_span_max(const float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) return span_max_v(x, n);
    if (simd() == 1) return span_max_a2(x, n);
#endif
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    return mx;
}

float e2_exp_sum(float *x, int n, float mx) {
#ifdef E2_X86OPS
    if (simd() >= 2) return exp_sum_v(x, n, mx);
    if (simd() == 1) return exp_sum_a2(x, n, mx);
#endif
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = e2_fexp(x[i] - mx); sum += x[i]; }
    return sum;
}

void e2_gelu_tanh(float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) { gelu_v(x, n); return; }
    if (simd() == 1) { gelu_a2(x, n); return; }
#endif
    gelu_s(x, n);
}

void e2_gelu_tanh_ctx(e2_ctx *ctx, float *x, int n) {
    run_split(ctx, e2_gelu_tanh, x, n);
}

void e2_silu(float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) { silu_v(x, n); return; }
    if (simd() == 1) { silu_a2(x, n); return; }
#endif
    silu_s(x, n);
}

void e2_vec_scale(float *x, int n, float k) {
#ifdef E2_X86OPS
    if (simd() >= 2) { scale_v(x, n, k); return; }
    if (simd() == 1) { scale_a2(x, n, k); return; }
#endif
    scale_s(x, n, k);
}

void e2_add_inplace(float *a, const float *b, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) { add_v(a, b, n); return; }
    if (simd() == 1) { add_a2(a, b, n); return; }
#endif
    for (int i = 0; i < n; i++) a[i] += b[i];
}

#ifdef E2_X86OPS
__attribute__((target("avx512f")))
static void rmsnorm_pre_v(float *out, const float *x, const float *w, int rows,
                          int dim, float eps, float pre) {
    float inv_dim = 1.0f / (float)dim;
    __m512 vp = _mm512_set1_ps(pre);
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 64 <= dim; i += 64) {
            __m512 v0 = _mm512_mul_ps(_mm512_loadu_ps(xp + i), vp);
            __m512 v1 = _mm512_mul_ps(_mm512_loadu_ps(xp + i + 16), vp);
            __m512 v2 = _mm512_mul_ps(_mm512_loadu_ps(xp + i + 32), vp);
            __m512 v3 = _mm512_mul_ps(_mm512_loadu_ps(xp + i + 48), vp);
            a0 = _mm512_fmadd_ps(v0, v0, a0); a1 = _mm512_fmadd_ps(v1, v1, a1);
            a2 = _mm512_fmadd_ps(v2, v2, a2); a3 = _mm512_fmadd_ps(v3, v3, a3);
        }
        for (; i + 16 <= dim; i += 16) {
            __m512 v0 = _mm512_mul_ps(_mm512_loadu_ps(xp + i), vp);
            a0 = _mm512_fmadd_ps(v0, v0, a0);
        }
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1),
                                                       _mm512_add_ps(a2, a3)));
        for (; i < dim; i++) { float v = xp[i] * pre; sum += v * v; }
        __m512 vs = _mm512_set1_ps(1.0f / sqrtf(sum * inv_dim + eps));
        i = 0;
        if (w) {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(op + i,
                    _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(xp + i), vp), vs),
                                  _mm512_loadu_ps(w + i)));
        } else {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(op + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(xp + i), vp), vs));
        }
        for (; i < dim; i++)
            op[i] = w ? xp[i] * pre * (vs[0] * w[i]) : xp[i] * pre * vs[0];
    }
}

__attribute__((target("avx512f")))
static void rmsnorm_add_v(float *x, const float *src, const float *w, int rows,
                          int dim, float eps) {
    float inv_dim = 1.0f / (float)dim;
    for (int r = 0; r < rows; r++) {
        const float *sp = src + (size_t)r * dim;
        float *xp = x + (size_t)r * dim;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 64 <= dim; i += 64) {
            __m512 v0 = _mm512_loadu_ps(sp + i), v1 = _mm512_loadu_ps(sp + i + 16);
            __m512 v2 = _mm512_loadu_ps(sp + i + 32), v3 = _mm512_loadu_ps(sp + i + 48);
            a0 = _mm512_fmadd_ps(v0, v0, a0); a1 = _mm512_fmadd_ps(v1, v1, a1);
            a2 = _mm512_fmadd_ps(v2, v2, a2); a3 = _mm512_fmadd_ps(v3, v3, a3);
        }
        for (; i + 16 <= dim; i += 16) {
            __m512 v0 = _mm512_loadu_ps(sp + i);
            a0 = _mm512_fmadd_ps(v0, v0, a0);
        }
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1),
                                                       _mm512_add_ps(a2, a3)));
        for (; i < dim; i++) sum += sp[i] * sp[i];
        __m512 vs = _mm512_set1_ps(1.0f / sqrtf(sum * inv_dim + eps));
        i = 0;
        if (w) {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(xp + i, _mm512_fmadd_ps(
                    _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(sp + i), vs),
                                  _mm512_loadu_ps(w + i)),
                    _mm512_set1_ps(1.0f), _mm512_loadu_ps(xp + i)));
        } else {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(xp + i, _mm512_fmadd_ps(
                    _mm512_mul_ps(_mm512_loadu_ps(sp + i), vs),
                    _mm512_set1_ps(1.0f), _mm512_loadu_ps(xp + i)));
        }
        for (; i < dim; i++) xp[i] += w ? sp[i] * (vs[0] * w[i]) : sp[i] * vs[0];
    }
}

__attribute__((target("avx512f")))
static void gelu_mul_v3(float *o, const float *x, const float *up, int n) {
    const __m512 k = _mm512_set1_ps(0.7978845608028654f);
    const __m512 c = _mm512_set1_ps(0.044715f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 big = _mm512_set1_ps(15.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 t = _mm512_mul_ps(k, _mm512_fmadd_ps(c, _mm512_mul_ps(v, _mm512_mul_ps(v, v)), v));
        __m512 s = vsigmoid(_mm512_mul_ps(two, _mm512_min_ps(t, big)));
        __mmask16 sat = _mm512_cmp_ps_mask(t, big, _CMP_GT_OQ);
        __m512 r = _mm512_mul_ps(v, s);
        r = _mm512_mask_mov_ps(r, sat, v);
        _mm512_storeu_ps(o + i, _mm512_mul_ps(r, _mm512_loadu_ps(up + i)));
    }
    for (; i < n; i++) o[i] = gelu1(x[i]) * up[i];
}

static void gelu_mul_v(float *x, const float *up, int n)
{
    if (simd() >= 2) gelu_mul_v3(x, x, up, n);
    else if (simd() == 1) gelu_mul_a2(x, x, up, n);
}

typedef struct { const float *x, *up; uint8_t *q; float *sc; int rows, k, blk; float *tmp; } gmq_arg;
static void gelu_mul_q_work(void *vp, int tid, int nt) {
    gmq_arg *a = vp;
    int r0, r1;
    e2_split(a->rows, tid, nt, &r0, &r1);
    int nblk = a->k / a->blk;
    float *t = a->tmp + (size_t)tid * a->k;
    for (int r = r0; r < r1; r++) {
        const float *xp = a->x + (size_t)r * a->k, *up = a->up + (size_t)r * a->k;
#ifdef E2_X86OPS
        if (simd() >= 2) gelu_mul_v3(t, xp, up, a->k);
        else if (simd() == 1) gelu_mul_a2(t, xp, up, a->k);
        else
#endif
            for (int i = 0; i < a->k; i++) t[i] = gelu1(xp[i]) * up[i];
        e2_quant_rows_blk(t, 1, a->k, a->q + (size_t)r * a->k,
                          a->sc + (size_t)r * nblk, a->blk);
    }
}

/* tmp = gelu(x) * up, immediately quantised: the GEMM that consumes it reads
 * the u8 form straight away, so the f32 intermediate never reaches memory. */
void e2_gelu_mul_q(e2_ctx *ctx, const float *x, const float *up, int rows, int k,
                   uint8_t *q, float *sc, int blk) {
    if (!blk) blk = 128;
    int nt = ctx && ctx->nthreads > 1 && (long)rows * k >= 65536 ? ctx->nthreads : 1;
    if (nt > rows) nt = rows;
    size_t tneed = (size_t)nt * k;
    if (ctx && ctx->tcap < tneed * sizeof(float)) {
        free(ctx->tbuf);
        ctx->tbuf = malloc(tneed * sizeof(float));
        ctx->tcap = ctx->tbuf ? tneed * sizeof(float) : 0;
    }
    float *tmp = ctx ? ctx->tbuf : NULL;
    if (!tmp) return;
    gmq_arg a = {x, up, q, sc, rows, k, blk, tmp};
    if (nt > 1) e2_parallel(ctx, gelu_mul_q_work, &a);
    else gelu_mul_q_work(&a, 0, 1);
}

typedef struct { float *x; const float *up; int n, chunk; } mul_arg;
static void gelu_mul_work(void *vp, int tid, int nt) {
    mul_arg *a = vp;
    int lo = tid * a->chunk;
    if (lo >= a->n) return;
    int hi = lo + a->chunk;
    if (hi > a->n) hi = a->n;
    gelu_mul_v(a->x + lo, a->up + lo, hi - lo);
}
#endif

static void rmsnorm_pre_s(float *out, const float *x, const float *w, int rows,
                          int dim, float eps, float pre) {
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) { float v = xp[i] * pre; sum += v * v; }
        float s = 1.0f / sqrtf(sum / (float)dim + eps);
        for (int i = 0; i < dim; i++)
            op[i] = xp[i] * pre * (w ? s * w[i] : s);
    }
}

static void rmsnorm_add_s(float *x, const float *src, const float *w, int rows,
                          int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *sp = src + (size_t)r * dim;
        float *xp = x + (size_t)r * dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += sp[i] * sp[i];
        float s = 1.0f / sqrtf(sum / (float)dim + eps);
        for (int i = 0; i < dim; i++) xp[i] += w ? sp[i] * (s * w[i]) : sp[i] * s;
    }
}

/* Row-split driver for the norm family.  Norms are pure streaming work, so
 * splitting rows is enough to use every core; the dispatch costs 0.25 us. */
typedef struct {
    float *out; const float *x, *src, *w;
    int rows, dim; float eps, pre; int mode;   /* 0 norm, 1 pre, 2 add, 3 rope */
    const e2_rope_tab *tb; int per_pos; const int *seq_off; int nseq;
} norm_arg;

static inline void rot(float *x, const float *c, const float *s, int half);
static void rmsnorm_rope_v(float *out, const float *x, const float *w, int rows,
                           int dim, float eps, const e2_rope_tab *tb,
                           int per_pos, const int *seq_off, int nseq, int r0);
static void rmsnorm_rope_a2(float *out, const float *x, const float *w, int rows,
                            int dim, float eps, const e2_rope_tab *tb,
                            int per_pos, const int *seq_off, int nseq, int r0);

/* position of row `r` inside its sequence */
static int rope_pos(int r, int per_pos, const int *seq_off, int nseq) {
    int gt = r / per_pos;
    int s = nseq - 1;
    while (s > 0 && gt < seq_off[s]) s--;
    return gt - seq_off[s];
}

/* norm then rotate: two L1-resident passes over one row. */
static void rmsnorm_rope_span(float *out, const float *x, const float *w, int rows,
                              int dim, float eps, const e2_rope_tab *tb,
                              int per_pos, const int *seq_off, int nseq, int r0) {
    int half = dim / 2;
    float inv_dim = 1.0f / (float)dim;
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        int pos = rope_pos(r0 + r, per_pos, seq_off, nseq);
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += xp[i] * xp[i];
        float s = 1.0f / sqrtf(sum * inv_dim + eps);
        for (int i = 0; i < dim; i++) op[i] = xp[i] * s * (w ? w[i] : 1.0f);
        rot(op, tb->c + (size_t)pos * half, tb->s + (size_t)pos * half, half);
    }
}

static void norm_rows(norm_arg *a, int r0, int r1) {
    int dim = a->dim;
    float *out = a->out + (size_t)r0 * dim;
    const float *x = a->x + (size_t)r0 * dim;
    const float *src = a->src ? a->src + (size_t)r0 * dim : NULL;
    int rows = r1 - r0;
    if (a->mode == 3) {
#ifdef E2_X86OPS
        if (simd() >= 2) {
            rmsnorm_rope_v(out, x, a->w, rows, dim, a->eps, a->tb, a->per_pos,
                           a->seq_off, a->nseq, r0);
            return;
        }
        if (simd() == 1) {
            rmsnorm_rope_a2(out, x, a->w, rows, dim, a->eps, a->tb, a->per_pos,
                            a->seq_off, a->nseq, r0);
            return;
        }
#endif
        rmsnorm_rope_span(out, x, a->w, rows, dim, a->eps, a->tb, a->per_pos,
                          a->seq_off, a->nseq, r0);
        return;
    }
#ifdef E2_X86OPS
    if (simd() >= 2) {
        if (a->mode == 1) { rmsnorm_pre_v(out, x, a->w, rows, dim, a->eps, a->pre); return; }
        if (a->mode == 2) { rmsnorm_add_v(out, src, a->w, rows, dim, a->eps); return; }
        rmsnorm_v(out, x, a->w, rows, dim, a->eps);
        return;
    }
    if (simd() == 1) {
        if (a->mode == 1) { rmsnorm_pre_a2(out, x, a->w, rows, dim, a->eps, a->pre); return; }
        if (a->mode == 2) { rmsnorm_add_a2(out, src, a->w, rows, dim, a->eps); return; }
        rmsnorm_a2(out, x, a->w, rows, dim, a->eps);
        return;
    }
#endif
    if (a->mode == 1) { rmsnorm_pre_s(out, x, a->w, rows, dim, a->eps, a->pre); return; }
    if (a->mode == 2) { rmsnorm_add_s(out, src, a->w, rows, dim, a->eps); return; }
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += xp[i] * xp[i];
        float s = 1.0f / sqrtf(sum / (float)dim + a->eps);
        if (a->w)
            for (int i = 0; i < dim; i++) op[i] = xp[i] * s * a->w[i];
        else
            for (int i = 0; i < dim; i++) op[i] = xp[i] * s;
    }
}

static void norm_work(void *vp, int tid, int nt) {
    norm_arg *a = vp;
    int r0, r1;
    e2_split(a->rows, tid, nt, &r0, &r1);
    if (r0 < r1) norm_rows(a, r0, r1);
}

static void norm_run(e2_ctx *ctx, norm_arg *a) {
    if (!ctx || ctx->nthreads <= 1 || (long)a->rows * a->dim < 16384) {
        norm_rows(a, 0, a->rows);
        return;
    }
    int nt = ctx->nthreads;
    if (a->rows < nt) nt = a->rows;
    e2_parallel(ctx, norm_work, a);
}

void e2_rmsnorm_pre(e2_ctx *ctx, float *out, const float *x, const float *w,
                    int rows, int dim, float eps, float pre) {
    norm_arg a = {out, x, NULL, w, rows, dim, eps, pre, 1, NULL, 0, NULL, 0};
    norm_run(ctx, &a);
}

void e2_rmsnorm_add(e2_ctx *ctx, float *x, const float *src, const float *w,
                    int rows, int dim, float eps) {
    norm_arg a = {x, x, src, w, rows, dim, eps, 1.0f, 2, NULL, 0, NULL, 0};
    norm_run(ctx, &a);
}

void e2_gelu_mul(e2_ctx *ctx, float *x, const float *up, int n) {
    if (!ctx || ctx->nthreads <= 1 || n < 65536) {
#ifdef E2_X86OPS
        if (simd() >= 2) { gelu_mul_v3(x, x, up, n); return; }
        if (simd() == 1) { gelu_mul_a2(x, x, up, n); return; }
#endif
        for (int i = 0; i < n; i++) x[i] = gelu1(x[i]) * up[i];
        return;
    }
    int nt = ctx->nthreads;
    if (n / 65536 < nt) nt = n / 65536;
    if (nt < 1) nt = 1;
    int chunk = (int)(((long)n + nt - 1) / nt);
    chunk = (chunk + 63) & ~63;
    mul_arg a = {x, up, n, chunk};
    e2_parallel(ctx, gelu_mul_work, &a);
}

void e2_rmsnorm(e2_ctx *ctx, float *out, const float *x, const float *w, int rows,
                int dim, float eps) {
    norm_arg a = {out, x, NULL, w, rows, dim, eps, 1.0f, 0, NULL, 0, NULL, 0};
    norm_run(ctx, &a);
}

void e2_rmsnorm_rope(e2_ctx *ctx, float *x, const float *w, int rows, int dim,
                     float eps, const e2_rope_tab *tb, int per_pos,
                     const int *seq_off, int nseq) {
    if (!tb) { e2_rmsnorm(ctx, x, x, w, rows, dim, eps); return; }
    norm_arg a = {x, x, NULL, w, rows, dim, eps, 1.0f, 3, tb, per_pos, seq_off, nseq};
    norm_run(ctx, &a);
}

void e2_rmsnorm_noweight(e2_ctx *ctx, float *out, const float *x, int rows, int dim,
                         float eps) {
    e2_rmsnorm(ctx, out, x, NULL, rows, dim, eps);
}

static void l2norm_span(float *x, int rows, int dim) {
#ifdef E2_X86OPS
    if (simd() >= 2) { l2norm_v(x, rows, dim); return; }
    if (simd() == 1) { l2norm_a2(x, rows, dim); return; }
#endif
    for (int r = 0; r < rows; r++) {
        float *p = x + (size_t)r * dim;
        float s = 0.0f;
        for (int i = 0; i < dim; i++) s += p[i] * p[i];
        s = 1.0f / sqrtf(s + 1e-12f);
        for (int i = 0; i < dim; i++) p[i] *= s;
    }
}

static void l2norm_work(void *vp, int tid, int nt) {
    norm_arg *a = vp;
    int r0, r1;
    e2_split(a->rows, tid, nt, &r0, &r1);
    if (r0 < r1) l2norm_span(a->out + (size_t)r0 * a->dim, r1 - r0, a->dim);
}

void e2_l2norm_rows(e2_ctx *ctx, float *x, int rows, int dim) {
    norm_arg a = {x, x, NULL, NULL, rows, dim, 0, 1.0f, 0, NULL, 0, NULL, 0};
    if (!ctx || ctx->nthreads <= 1 || (long)rows * dim < 16384) {
        l2norm_span(x, rows, dim);
        return;
    }
    int nt = ctx->nthreads;
    if (rows < nt) nt = rows;
    e2_parallel(ctx, l2norm_work, &a);
}

/* --- parallel unary/binary elementwise ---------------------------- */

typedef struct {
    float *a; const float *b; int n, chunk, op; float k;
} ew_arg;

static void ew_span(float *a, const float *b, int n, int op, float k) {
    switch (op) {
    case 0: e2_add_inplace(a, b, n); break;
    case 1: e2_vec_scale(a, n, k); break;
    case 2: e2_silu(a, n); break;
    default: e2_gelu_tanh(a, n); break;
    }
}

static void ew_work(void *vp, int tid, int nt) {
    ew_arg *e = vp;
    int lo = tid * e->chunk;
    if (lo >= e->n) return;
    int hi = lo + e->chunk;
    if (hi > e->n) hi = e->n;
    ew_span(e->a + lo, e->b ? e->b + lo : NULL, hi - lo, e->op, e->k);
}

static void ew_run(e2_ctx *ctx, float *a, const float *b, int n, int op, float k) {
    if (!ctx || ctx->nthreads <= 1 || n < 65536) { ew_span(a, b, n, op, k); return; }
    int nt = ctx->nthreads;
    if (n / 32768 < nt) nt = n / 32768;
    if (nt < 1) nt = 1;
    int chunk = (int)(((long)n + nt - 1) / nt);
    chunk = (chunk + 63) & ~63;
    ew_arg e = {a, b, n, chunk, op, k};
    e2_parallel(ctx, ew_work, &e);
}

void e2_add_inplace_ctx(e2_ctx *ctx, float *a, const float *b, int n) {
    ew_run(ctx, a, b, n, 0, 0.0f);
}
void e2_vec_scale_ctx(e2_ctx *ctx, float *x, int n, float k) {
    ew_run(ctx, x, NULL, n, 1, k);
}
void e2_silu_ctx(e2_ctx *ctx, float *x, int n) { ew_run(ctx, x, NULL, n, 2, 0.0f); }
void e2_gelu_tanh_p(e2_ctx *ctx, float *x, int n) { ew_run(ctx, x, NULL, n, 3, 0.0f); }

void e2_softmax(float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 1 && n >= 8) { softmax_exp_v(x, n); return; }
#endif
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = e2_fexp(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void e2_dequant_row(const e2_qmat *w, int row, float *out) {
    const int8_t *q = w->q + (size_t)row * w->k;
    float s = w->scale[row];
    for (int i = 0; i < w->k; i++) out[i] = (float)q[i] * s;
}

/* ------------------------------------------------------------------ */
/* Rotary: cached cos/sin tables, amortised over all heads and layers. */
/* ------------------------------------------------------------------ */

__attribute__((target("avx512f")))
static void rot_v(float *x, const float *c, const float *s, int half) {
    int d = 0;
    for (; d + 16 <= half; d += 16) {
        __m512 c1 = _mm512_loadu_ps(c + d), s1 = _mm512_loadu_ps(s + d);
        __m512 x1 = _mm512_loadu_ps(x + d), x2 = _mm512_loadu_ps(x + d + half);
        _mm512_storeu_ps(x + d, _mm512_fmsub_ps(x1, c1, _mm512_mul_ps(x2, s1)));
        _mm512_storeu_ps(x + d + half, _mm512_fmadd_ps(x2, c1, _mm512_mul_ps(x1, s1)));
    }
    for (; d < half; d++) {
        float x1 = x[d], x2 = x[d + half];
        x[d] = x1 * c[d] - x2 * s[d];
        x[d + half] = x2 * c[d] + x1 * s[d];
    }
}

static void rot_s(float *x, const float *c, const float *s, int half) {
    for (int d = 0; d < half; d++) {
        float x1 = x[d], x2 = x[d + half];
        x[d] = x1 * c[d] - x2 * s[d];
        x[d + half] = x2 * c[d] + x1 * s[d];
    }
}

static inline void rot(float *x, const float *c, const float *s, int half) {
#ifdef E2_X86OPS
    if (simd() >= 2) { rot_v(x, c, s, half); return; }
    if (simd() == 1) { rot_a2(x, c, s, half); return; }
#endif
    rot_s(x, c, s, half);
}

/* vectorised RMSNorm + RoPE on one row (norm and rotation in one pass over
 * the row, which is still in L1 between the two halves) */
__attribute__((target("avx512f")))
static void rmsnorm_rope_v(float *out, const float *x, const float *w, int rows,
                           int dim, float eps, const e2_rope_tab *tb,
                           int per_pos, const int *seq_off, int nseq, int r0) {
    int half = dim / 2;
    float inv_dim = 1.0f / (float)dim;
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0;
        int i = 0;
        for (; i + 32 <= dim; i += 32) {
            __m512 v0 = _mm512_loadu_ps(xp + i), v1 = _mm512_loadu_ps(xp + i + 16);
            a0 = _mm512_fmadd_ps(v0, v0, a0);
            a1 = _mm512_fmadd_ps(v1, v1, a1);
        }
        for (; i + 16 <= dim; i += 16) {
            __m512 v0 = _mm512_loadu_ps(xp + i);
            a0 = _mm512_fmadd_ps(v0, v0, a0);
        }
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        for (; i < dim; i++) sum += xp[i] * xp[i];
        __m512 vs = _mm512_set1_ps(1.0f / sqrtf(sum * inv_dim + eps));
        i = 0;
        if (w) {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(op + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(xp + i), vs),
                                                       _mm512_loadu_ps(w + i)));
        } else {
            for (; i + 16 <= dim; i += 16)
                _mm512_storeu_ps(op + i, _mm512_mul_ps(_mm512_loadu_ps(xp + i), vs));
        }
        for (; i < dim; i++) op[i] = w ? xp[i] * (vs[0] * w[i]) : xp[i] * vs[0];
        int pos = rope_pos(r0 + r, per_pos, seq_off, nseq);
        rot_v(op, tb->c + (size_t)pos * half, tb->s + (size_t)pos * half, half);
    }
}

/* AVX2 RoPE rotate: same maths, 8 lanes. */
A2 static void rot_a2(float *x, const float *c, const float *s, int half) {
    int d = 0;
    for (; d + 8 <= half; d += 8) {
        __m256 c1 = _mm256_loadu_ps(c + d), s1 = _mm256_loadu_ps(s + d);
        __m256 x1 = _mm256_loadu_ps(x + d), x2 = _mm256_loadu_ps(x + d + half);
        _mm256_storeu_ps(x + d, _mm256_fmsub_ps(x1, c1, _mm256_mul_ps(x2, s1)));
        _mm256_storeu_ps(x + d + half, _mm256_fmadd_ps(x2, c1, _mm256_mul_ps(x1, s1)));
    }
    for (; d < half; d++) {
        float x1 = x[d], x2 = x[d + half];
        x[d] = x1 * c[d] - x2 * s[d];
        x[d + half] = x2 * c[d] + x1 * s[d];
    }
}

/* AVX2 RMSNorm + RoPE in one pass over the row. */
A2 static void rmsnorm_rope_a2(float *out, const float *x, const float *w, int rows,
                               int dim, float eps, const e2_rope_tab *tb,
                               int per_pos, const int *seq_off, int nseq, int r0) {
    int half = dim / 2;
    float inv_dim = 1.0f / (float)dim;
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0;
        int i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m256 v0 = _mm256_loadu_ps(xp + i), v1 = _mm256_loadu_ps(xp + i + 8);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
            a1 = _mm256_fmadd_ps(v1, v1, a1);
        }
        for (; i + 8 <= dim; i += 8) {
            __m256 v0 = _mm256_loadu_ps(xp + i);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
        }
        float sum = hsum8(_mm256_add_ps(a0, a1));
        for (; i < dim; i++) sum += xp[i] * xp[i];
        float sv = 1.0f / sqrtf(sum * inv_dim + eps);
        __m256 vs = _mm256_set1_ps(sv);
        i = 0;
        if (w) {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(op + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xp + i), vs),
                                                       _mm256_loadu_ps(w + i)));
        } else {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(op + i, _mm256_mul_ps(_mm256_loadu_ps(xp + i), vs));
        }
        for (; i < dim; i++) op[i] = w ? xp[i] * (sv * w[i]) : xp[i] * sv;
        int pos = rope_pos(r0 + r, per_pos, seq_off, nseq);
        rot_a2(op, tb->c + (size_t)pos * half, tb->s + (size_t)pos * half, half);
    }
}

/* AVX2 gelu(x) * up, and the in-place form. */
A2 static void gelu_mul_a2(float *o, const float *x, const float *up, int n) {
    const __m256 k = _mm256_set1_ps(0.7978845608028654f);
    const __m256 c = _mm256_set1_ps(0.044715f);
    const __m256 big = _mm256_set1_ps(15.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 t = _mm256_mul_ps(k, _mm256_fmadd_ps(c, _mm256_mul_ps(v, _mm256_mul_ps(v, v)), v));
        __m256 s = vsigmoid8(_mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_min_ps(t, big)));
        __m256 r = _mm256_mul_ps(v, s);
        r = _mm256_blendv_ps(r, v, _mm256_cmp_ps(t, big, _CMP_GT_OQ));
        _mm256_storeu_ps(o + i, _mm256_mul_ps(r, _mm256_loadu_ps(up + i)));
    }
    for (; i < n; i++) o[i] = gelu1(x[i]) * up[i];
}

/* Build/extend a table covering positions [0,need) for `theta`. */
static int rope_build(e2_rope_tab *tb, float theta, int half, int need) {
    if (tb->cap >= need && tb->half == half && tb->theta == theta && tb->c) return 0;
    int cap = tb->cap > need ? tb->cap : need;
    float *c = realloc(tb->c, (size_t)cap * half * sizeof(float));
    if (!c) return -1;
    float *s = realloc(tb->s, (size_t)cap * half * sizeof(float));
    if (!s) return -1;
    tb->c = c; tb->s = s; tb->cap = cap; tb->half = half; tb->theta = theta;
    for (int p = 0; p < cap; p++)
        for (int d = 0; d < half; d++) {
            float ang = (float)p * (1.0f / powf(theta, (float)(2 * d) / (float)(2 * half)));
            tb->c[(size_t)p * half + d] = cosf(ang);
            tb->s[(size_t)p * half + d] = sinf(ang);
        }
    return 0;
}

void e2_rope_free(e2_ctx *ctx) {
    for (int i = 0; i < 4; i++) {
        e2_rope_tab *tb = i < 2 ? &ctx->rope[i] : &ctx->ropev[i - 2];
        free(tb->c); free(tb->s);
        memset(tb, 0, sizeof(*tb));
    }
}

static e2_rope_tab *rope_pick(e2_rope_tab *tabs, float theta, int half, int need) {
    for (int i = 0; i < 2; i++)
        if (tabs[i].c && tabs[i].theta == theta && tabs[i].half == half) {
            if (rope_build(&tabs[i], theta, half, need)) return NULL;
            return &tabs[i];
        }
    int i = (tabs[0].c == NULL) ? 0 : (tabs[1].c == NULL ? 1 : 0);
    if (rope_build(&tabs[i], theta, half, need)) return NULL;
    return &tabs[i];
}

e2_rope_tab *e2_rope_get(e2_ctx *ctx, float theta, int half, int need) {
    return rope_pick(ctx->rope, theta, half, need);
}

typedef struct {
    float *q, *k;
    int t, heads, kv_heads, head_dim, offset;
    const float *c, *s;
} rope_arg;

static void rope_text_work(void *vp, int tid, int nt) {
    rope_arg *a = vp;
    int half = a->head_dim / 2;
    int t0 = (int)((long)a->t * tid / nt), t1 = (int)((long)a->t * (tid + 1) / nt);
    for (int pos = t0; pos < t1; pos++) {
        const float *c = a->c + (size_t)(pos + a->offset) * half;
        const float *s = a->s + (size_t)(pos + a->offset) * half;
        for (int h = 0; h < a->heads; h++)
            rot(a->q + ((size_t)pos * a->heads + h) * a->head_dim, c, s, half);
        for (int h = 0; h < a->kv_heads; h++)
            rot(a->k + ((size_t)pos * a->kv_heads + h) * a->head_dim, c, s, half);
    }
}

void e2_rope_text(e2_ctx *ctx, float *q, float *k, int t, int heads, int kv_heads,
                  int head_dim, int pos_offset, float theta) {
    int half = head_dim / 2;
    e2_rope_tab *tb = rope_pick(ctx->rope, theta, half, t + pos_offset);
    if (!tb) return;
    rope_arg a = {q, k, t, heads, kv_heads, head_dim, pos_offset, tb->c, tb->s};
    if (t * heads < 4096) { rope_text_work(&a, 0, 1); return; }
    e2_parallel(ctx, rope_text_work, &a);
}

/* vision: axial rope, tables cached per axis */
typedef struct {
    float *q, *k;
    int t, heads, head_dim;
    const int *xy;
    const float *ch, *sh, *cw, *sw;
    int half;
} vrope_arg;

static void rope_vision_work(void *vp, int tid, int nt) {
    vrope_arg *a = vp;
    int spatial = a->head_dim / 2, half = a->half;
    int t0 = (int)((long)a->t * tid / nt), t1 = (int)((long)a->t * (tid + 1) / nt);
    for (int pos = t0; pos < t1; pos++) {
        int px = a->xy[2 * pos], py = a->xy[2 * pos + 1];
        const float *ch = a->ch + (size_t)px * half, *sh = a->sh + (size_t)px * half;
        const float *cw = a->cw + (size_t)py * half, *sw = a->sw + (size_t)py * half;
        for (int which = 0; which < 2; which++) {
            float *base = which == 0 ? a->q : a->k;
            if (!base) continue;
            if (which == 1 && a->k == a->q) continue;
            for (int h = 0; h < a->heads; h++) {
                float *v = base + ((size_t)pos * a->heads + h) * a->head_dim;
                rot(v, ch, sh, half);
                rot(v + spatial, cw, sw, half);
            }
        }
    }
}

void e2_rope_vision(e2_ctx *ctx, float *q, float *k, int t, int heads, int head_dim,
                    const int *xy, float theta) {
    int spatial = head_dim / 2, half = spatial / 2;
    int mh = 0, mw = 0;
    for (int i = 0; i < t; i++) {
        if (xy[2 * i] > mh) mh = xy[2 * i];
        if (xy[2 * i + 1] > mw) mw = xy[2 * i + 1];
    }
    if (rope_build(&ctx->ropev[0], theta, half, mh + 1)) return;
    if (rope_build(&ctx->ropev[1], theta, half, mw + 1)) return;
    (void)spatial;
    vrope_arg a = {q, k, t, heads, head_dim, xy, ctx->ropev[0].c, ctx->ropev[0].s,
                   ctx->ropev[1].c, ctx->ropev[1].s, half};
    e2_parallel(ctx, rope_vision_work, &a);
}

/* ------------------------------------------------------------------ */
/* attention: R query rows share one pass over K/V.                   */
/* ------------------------------------------------------------------ */

#define ATTN_C 32           /* value columns per accumulator chunk (2 zmm) */

/* sc[r*n + j] is already normalised; accumulate into `out`. */
static void attn_pv_s(float *const out[4], const float *const q[4],
                      const float *vb, size_t vstride, int n, int dim, const float *sc,
                      int sstride) {
    (void)q;
    for (int r = 0; r < 4; r++)
        for (int d = 0; d < dim; d++) out[r][d] = 0.0f;
    for (int j = 0; j < n; j++) {
        const float *v = vb + (size_t)j * vstride;
        for (int r = 0; r < 4; r++) {
            float w = sc[(size_t)r * sstride + j];
            float *o = out[r];
            for (int d = 0; d < dim; d++) o[d] += w * v[d];
        }
    }
}

#ifdef E2_X86OPS
__attribute__((target("avx512f")))
static void attn_pv_v(float *const out[4], const float *const q[4],
                      const float *vb, size_t vstride, int n, int dim, const float *sc,
                      int sstride) {
    for (int c = 0; c < dim; c += ATTN_C) {
        int cw = dim - c; if (cw > ATTN_C) cw = ATTN_C;
        __m512 z = _mm512_setzero_ps();
        float *o0 = out[0] + c, *o1 = out[1] + c, *o2 = out[2] + c, *o3 = out[3] + c;
        __m512 a0 = z, a1 = z, a2 = z, a3 = z;
        __m512 b0 = z, b1 = z, b2 = z, b3 = z;
        for (int j = 0; j < n; j++) {
            const float *v = vb + (size_t)j * vstride + c;
            _mm_prefetch((const char *)(v + 8 * vstride), _MM_HINT_T0);
            __m512 v0 = _mm512_loadu_ps(v);
            __m512 v1 = cw > 16 ? _mm512_loadu_ps(v + 16) : z;
            const float *p = sc + j;
            __m512 w0 = _mm512_set1_ps(p[0]), w1 = _mm512_set1_ps(p[sstride]);
            __m512 w2 = _mm512_set1_ps(p[2 * sstride]), w3 = _mm512_set1_ps(p[3 * sstride]);

            a0 = _mm512_fmadd_ps(w0, v0, a0); b0 = _mm512_fmadd_ps(w0, v1, b0);
            a1 = _mm512_fmadd_ps(w1, v0, a1); b1 = _mm512_fmadd_ps(w1, v1, b1);
            a2 = _mm512_fmadd_ps(w2, v0, a2); b2 = _mm512_fmadd_ps(w2, v1, b2);
            a3 = _mm512_fmadd_ps(w3, v0, a3); b3 = _mm512_fmadd_ps(w3, v1, b3);

        }
        _mm512_storeu_ps(o0, a0); _mm512_storeu_ps(o1, a1);
        _mm512_storeu_ps(o2, a2); _mm512_storeu_ps(o3, a3);

        if (cw > 16) {
            _mm512_storeu_ps(o0 + 16, b0); _mm512_storeu_ps(o1 + 16, b1);
            _mm512_storeu_ps(o2 + 16, b2); _mm512_storeu_ps(o3 + 16, b3);

        }
    }
}

/* AVX2 twins.  Without them the f32 attention (text shorter than 32 tokens,
 * and any host without AVX-512) ran the scalar kernel at ~8 GMAC/s. */
__attribute__((target("avx2,fma")))
static void attn_pv_a2(float *const out[4], const float *const q[4],
                       const float *vb, size_t vstride, int n, int dim,
                       const float *sc, int sstride) {
    (void)q;
    int cfull = dim & ~15;
    for (int c = 0; c < cfull; c += 16) {
        __m256 z = _mm256_setzero_ps();
        float *o0 = out[0] + c, *o1 = out[1] + c, *o2 = out[2] + c, *o3 = out[3] + c;
        __m256 a0 = z, a1 = z, a2 = z, a3 = z;
        __m256 b0 = z, b1 = z, b2 = z, b3 = z;
        for (int j = 0; j < n; j++) {
            const float *v = vb + (size_t)j * vstride + c;
            _mm_prefetch((const char *)(v + 8 * vstride), _MM_HINT_T0);
            __m256 v0 = _mm256_loadu_ps(v);
            __m256 v1 = _mm256_loadu_ps(v + 8);
            const float *p = sc + j;
            __m256 w0 = _mm256_set1_ps(p[0]), w1 = _mm256_set1_ps(p[sstride]);
            __m256 w2 = _mm256_set1_ps(p[2 * sstride]), w3 = _mm256_set1_ps(p[3 * sstride]);
            a0 = _mm256_fmadd_ps(w0, v0, a0); b0 = _mm256_fmadd_ps(w0, v1, b0);
            a1 = _mm256_fmadd_ps(w1, v0, a1); b1 = _mm256_fmadd_ps(w1, v1, b1);
            a2 = _mm256_fmadd_ps(w2, v0, a2); b2 = _mm256_fmadd_ps(w2, v1, b2);
            a3 = _mm256_fmadd_ps(w3, v0, a3); b3 = _mm256_fmadd_ps(w3, v1, b3);
        }
        _mm256_storeu_ps(o0, a0); _mm256_storeu_ps(o1, a1);
        _mm256_storeu_ps(o2, a2); _mm256_storeu_ps(o3, a3);
        _mm256_storeu_ps(o0 + 8, b0); _mm256_storeu_ps(o1 + 8, b1);
        _mm256_storeu_ps(o2 + 8, b2); _mm256_storeu_ps(o3 + 8, b3);
    }
    for (int c = cfull; c < dim; c++) {          /* head_dim not a multiple of 16 */
        for (int r = 0; r < 4; r++) out[r][c] = 0.0f;
        for (int j = 0; j < n; j++) {
            const float *v = vb + (size_t)j * vstride + c;
            for (int r = 0; r < 4; r++) out[r][c] += sc[(size_t)r * sstride + j] * *v;
        }
    }
}
#endif

/* Transpose K to [kv][dim][t_pad] (zero padded to a multiple of 32 columns). */
typedef struct {
    const float *k;
    float *kt;
    int t, tpad, kv, dim, kstride;
} kt_arg;

static void kt_work(void *vp, int tid, int nt) {
    kt_arg *a = vp;
    int dim = a->dim, t = a->t, tpad = a->tpad;
    long total = (long)a->kv * ((dim + 15) / 16) * ((t + 15) / 16);
    int i0, i1;
    e2_split((int)total, tid, nt, &i0, &i1);
    long w0 = i0, w1 = i1;
    int nt_t = (t + 15) / 16, nt_d = (dim + 15) / 16;
    for (long w = w0; w < w1; w++) {
        int kh = (int)(w / (nt_t * nt_d));
        int rem = (int)(w % (nt_t * nt_d));
        int dt = rem / nt_t, tt = rem % nt_t;
        int d0 = dt * 16, j0 = tt * 16;
        int dmax = dim - d0; if (dmax > 16) dmax = 16;
        int jmax = t - j0; if (jmax > 16) jmax = 16;
        const float *src = a->k + (size_t)j0 * a->kstride + (size_t)kh * dim + d0;
        float *dst = a->kt + ((size_t)kh * dim + d0) * tpad + j0;
        for (int j = 0; j < jmax; j++) {
            const float *sp = src + (size_t)j * a->kstride;
            for (int d = 0; d < dmax; d++) dst[(size_t)d * tpad + j] = sp[d];
        }
    }
}

/* scores sc[r*tpad + j] = q[r] . kt[d][j], 32 keys at a time, 4 rows */
__attribute__((target("avx2,fma")))
static void qk_kt_a2(const float *const q[4], const float *kt, int n, int dim,
                     int tpad, float *sc) {
    const float *p0 = q[0], *p1 = q[1], *p2 = q[2], *p3 = q[3];
    int jb = 0;
    for (; jb + 16 <= n; jb += 16) {
        const float *kp = kt + jb;
        __m256 z = _mm256_setzero_ps();
        __m256 a0 = z, a1 = z, b0 = z, b1 = z;
        __m256 c0 = z, c1 = z, d0 = z, d1 = z;
        for (int d = 0; d < dim; d++, kp += tpad) {
            _mm_prefetch((const char *)(kp + 6 * tpad), _MM_HINT_T0);
            _mm_prefetch((const char *)(kp + 6 * tpad + 8), _MM_HINT_T0);
            __m256 k0 = _mm256_loadu_ps(kp);
            __m256 k1 = _mm256_loadu_ps(kp + 8);
            __m256 w0 = _mm256_set1_ps(p0[d]);
            __m256 w1 = _mm256_set1_ps(p1[d]);
            __m256 w2 = _mm256_set1_ps(p2[d]);
            __m256 w3 = _mm256_set1_ps(p3[d]);
            a0 = _mm256_fmadd_ps(w0, k0, a0); a1 = _mm256_fmadd_ps(w0, k1, a1);
            b0 = _mm256_fmadd_ps(w1, k0, b0); b1 = _mm256_fmadd_ps(w1, k1, b1);
            c0 = _mm256_fmadd_ps(w2, k0, c0); c1 = _mm256_fmadd_ps(w2, k1, c1);
            d0 = _mm256_fmadd_ps(w3, k0, d0); d1 = _mm256_fmadd_ps(w3, k1, d1);
        }
        _mm256_storeu_ps(sc + jb, a0); _mm256_storeu_ps(sc + jb + 8, a1);
        _mm256_storeu_ps(sc + tpad + jb, b0); _mm256_storeu_ps(sc + tpad + jb + 8, b1);
        _mm256_storeu_ps(sc + 2 * tpad + jb, c0); _mm256_storeu_ps(sc + 2 * tpad + jb + 8, c1);
        _mm256_storeu_ps(sc + 3 * tpad + jb, d0); _mm256_storeu_ps(sc + 3 * tpad + jb + 8, d1);
    }
    for (; jb < n; jb++)
        for (int r = 0; r < 4; r++) {
            const float *kf = kt + jb;
            float s = 0.0f;
            for (int d = 0; d < dim; d++, kf += tpad) s += q[r][d] * *kf;
            sc[(size_t)r * tpad + jb] = s;
        }
}

__attribute__((target("avx512f")))
static void qk_kt_v(const float *const q[4], const float *kt, int n, int dim, int tpad,
                    float *sc) {
    const float *p0 = q[0], *p1 = q[1], *p2 = q[2], *p3 = q[3];
    int jb = 0;
    for (; jb + 32 <= n; jb += 32) {
        const float *kp = kt + jb;
        __m512 z = _mm512_setzero_ps();
        __m512 a0 = z, a1 = z, b0 = z, b1 = z, c0 = z, c1 = z, d0 = z, d1 = z;
        for (int d = 0; d < dim; d++, kp += tpad) {
            _mm_prefetch((const char *)(kp + 6 * tpad), _MM_HINT_T0);
            _mm_prefetch((const char *)(kp + 6 * tpad + 16), _MM_HINT_T0);
            __m512 k0 = _mm512_loadu_ps(kp);
            __m512 k1 = _mm512_loadu_ps(kp + 16);
            __m512 w0 = _mm512_set1_ps(p0[d]);
            __m512 w1 = _mm512_set1_ps(p1[d]);
            __m512 w2 = _mm512_set1_ps(p2[d]);
            __m512 w3 = _mm512_set1_ps(p3[d]);
            a0 = _mm512_fmadd_ps(w0, k0, a0); a1 = _mm512_fmadd_ps(w0, k1, a1);
            b0 = _mm512_fmadd_ps(w1, k0, b0); b1 = _mm512_fmadd_ps(w1, k1, b1);
            c0 = _mm512_fmadd_ps(w2, k0, c0); c1 = _mm512_fmadd_ps(w2, k1, c1);
            d0 = _mm512_fmadd_ps(w3, k0, d0); d1 = _mm512_fmadd_ps(w3, k1, d1);
        }
        _mm512_storeu_ps(sc + jb, a0); _mm512_storeu_ps(sc + jb + 16, a1);
        _mm512_storeu_ps(sc + tpad + jb, b0); _mm512_storeu_ps(sc + tpad + jb + 16, b1);
        _mm512_storeu_ps(sc + 2 * tpad + jb, c0); _mm512_storeu_ps(sc + 2 * tpad + jb + 16, c1);
        _mm512_storeu_ps(sc + 3 * tpad + jb, d0); _mm512_storeu_ps(sc + 3 * tpad + jb + 16, d1);
    }
    for (; jb < n; jb++)
        for (int r = 0; r < 4; r++) {
            const float *kf = kt + jb;
            float s = 0.0f;
            for (int d = 0; d < dim; d++, kf += tpad) s += q[r][d] * *kf;
            sc[(size_t)r * tpad + jb] = s;
        }
}

typedef struct {
    float *out;
    const float *q, *k, *v, *kt;
    int t, tpad, heads, kv_heads, head_dim, window;
} attn_arg;

static void qk_kt_s(const float *const q[4], const float *kt, int n, int dim, int tpad,
                    float *sc) {
    for (int j = 0; j < n; j++)
        for (int r = 0; r < 4; r++) {
            float s = 0.0f;
            for (int d = 0; d < dim; d++) s += q[r][d] * kt[(size_t)d * tpad + j];
            sc[(size_t)r * tpad + j] = s;
        }
}

static void attn_work(void *vp, int tid, int nt) {
    attn_arg *a = vp;
    int head_dim = a->head_dim, tpad = a->tpad;
    int groups = a->heads / a->kv_heads;
    if (groups < 1) groups = 1;
    if (groups > 4) groups = 4;
    int P = 4 / groups;                   /* positions per work item */
    if (P < 1) P = 1;
    int nblk = (a->t + P - 1) / P;
    float *sc = malloc((size_t)4 * (tpad + 32) * sizeof(float));
    if (!sc) return;
    size_t kv_stride = (size_t)a->kv_heads * head_dim;

    long total = (long)nblk * a->kv_heads;
    int i0, i1;
    e2_split((int)total, tid, nt, &i0, &i1);
    long w0 = i0, w1 = i1;

    /* head-major: each thread stays on one K/V head so its transposed K and V
     * stay in L2 instead of thrashing L3. */
    for (long w = w0; w < w1; w++) {
        int kh = (int)(w / nblk);
        int blk = (int)(w % nblk);
        if (kh >= a->kv_heads) { kh = a->kv_heads - 1; blk = nblk - 1; }
        int i0 = blk * P;
        int np = a->t - i0; if (np > P) np = P;

        int lo = 0, hi = a->t;
        if (a->window > 0) {
            lo = i0 - a->window; if (lo < 0) lo = 0;
            hi = i0 + np - 1 + a->window + 1; if (hi > a->t) hi = a->t;
        }
        int n = hi - lo;
        int npad = (n + 31) & ~31;

        const float *q[4];
        float *out[4];
        for (int p = 0; p < P; p++) {
            int pp = p < np ? p : np - 1;
            for (int g = 0; g < groups; g++) {
                int r = p * groups + g;
                q[r] = a->q + ((size_t)(i0 + pp) * a->heads + kh * groups + g) * head_dim;
                out[r] = a->out + ((size_t)(i0 + pp) * a->heads + kh * groups + g) * head_dim;
            }
        }
        const float *kt = a->kt + ((size_t)kh * head_dim) * tpad + lo;
        const float *vb = a->v + (size_t)lo * kv_stride + (size_t)kh * head_dim;

#ifdef E2_X86OPS
        if (simd() >= 2) qk_kt_v(q, kt, npad, head_dim, tpad, sc);
        else if (simd() >= 1) qk_kt_a2(q, kt, npad, head_dim, tpad, sc);
        else
#endif
        qk_kt_s(q, kt, npad, head_dim, tpad, sc);

        for (int r = 0; r < 4; r++) {
            float *sr = sc + (size_t)r * tpad;
            int pos = i0 + ((r < P * groups) ? (r / groups) : (np - 1));
            int lo_r = 0, hi_r = a->t;
            if (a->window > 0) {
                lo_r = pos - a->window; if (lo_r < 0) lo_r = 0;
                hi_r = pos + a->window + 1; if (hi_r > a->t) hi_r = a->t;
            }
            int bad0 = lo_r - lo, bad1 = hi_r - lo;
            for (int j = 0; j < bad0; j++) sr[j] = -INFINITY;
            for (int j = bad1; j < npad; j++) sr[j] = -INFINITY;
        }
#ifdef E2_X86OPS
        if (simd() >= 2) {
            for (int r = 0; r < 4; r++) softmax_exp_v(sc + (size_t)r * tpad, npad);
        } else
#endif
        {
            for (int r = 0; r < 4; r++) e2_softmax(sc + (size_t)r * tpad, npad);
        }
#ifdef E2_X86OPS
        if (simd() >= 2) attn_pv_v(out, q, vb, kv_stride, n, head_dim, sc, tpad);
        else if (simd() >= 1) attn_pv_a2(out, q, vb, kv_stride, n, head_dim, sc, tpad);
        else
#endif
        attn_pv_s(out, q, vb, kv_stride, n, head_dim, sc, tpad);
    }
    free(sc);
}

typedef struct attn_ws attn_ws;

static void attention_f32(e2_ctx *ctx, float *out, const float *q, const float *k,
                          const float *v, int t, int heads, int kv_heads,
                          int head_dim, int window) {
    int groups = heads / kv_heads;
    if (groups < 1) groups = 1;
    if (groups > 4) groups = 4;
    int tpad = ((t + 31) & ~31) + 32;
    size_t need = (size_t)kv_heads * head_dim * tpad;
    if (ctx->ktcap < need) {
        free(ctx->ktbuf);
        ctx->ktbuf = malloc(need * sizeof(float));
        ctx->ktcap = ctx->ktbuf ? need : 0;
    }
    if (!ctx->ktbuf) return;
    memset(ctx->ktbuf, 0, need * sizeof(float));
    kt_arg ka = {k, ctx->ktbuf, t, tpad, kv_heads, head_dim, kv_heads * head_dim};
    if ((long)kv_heads * ((head_dim + 15) / 16) * ((t + 15) / 16) < 512)
        kt_work(&ka, 0, 1);
    else
        e2_parallel(ctx, kt_work, &ka);
    attn_arg a = {out, q, k, v, ctx->ktbuf, t, tpad, heads, kv_heads, head_dim, window};
    int P = 4 / groups;
    long items = (long)((t + P - 1) / P) * kv_heads;
    if (items < ctx->nthreads || (long)t * heads < 2048) attn_work(&a, 0, 1);
    else e2_parallel(ctx, attn_work, &a);
}

void e2_attention_text(e2_ctx *ctx, float *out, const float *q, const float *k,
                       const float *v, int t, int heads, int kv_heads, int head_dim,
                       int window) {
    if (t <= 0 || heads <= 0) return;
    static int noint8 = -1;
    if (noint8 < 0) noint8 = getenv("E2_ATTN_F32") ? 1 : 0;
    if (!noint8 && simd() >= 1 && t >= 32) {
        int off = 0;
        if (e2_attn_int8(ctx, out, q, k, v, t, heads, kv_heads, head_dim, window,
                         1, &off) == 0)
            return;
    }
    attention_f32(ctx, out, q, k, v, t, heads, kv_heads, head_dim, window);
}

void e2_attention_vision(e2_ctx *ctx, float *out, const float *q, const float *k,
                         const float *v, int t, int heads, int head_dim) {
    e2_attention_text(ctx, out, q, k, v, t, heads, heads, head_dim, 0);
}

void e2_attention_full(e2_ctx *ctx, float *out, const float *q, const float *k,
                       const float *v, int t, int heads, int head_dim) {
    e2_attention_text(ctx, out, q, k, v, t, heads, heads, head_dim, 0);
}





/* Batched attention over `nseq` equal-length sequences: one item list covers
 * every sequence, so a layer is still two dispatches (pack + items). */
void e2_attention_text_multi(e2_ctx *ctx, float *out, const float *q, const float *k,
                             const float *v, int t, const int *seq_off, int nseq,
                             int heads, int kv_heads, int head_dim, int window) {
    size_t qs = (size_t)heads * head_dim, ks = (size_t)kv_heads * head_dim;
    static int noint8 = -1;
    if (noint8 < 0) noint8 = getenv("E2_ATTN_F32") ? 1 : 0;
    if (!noint8 && nseq > 1 && t >= 16 && simd() >= 1 &&
        e2_attn_int8(ctx, out, q, k, v, t, heads, kv_heads, head_dim, window,
                     nseq, seq_off) == 0)
        return;
    for (int s = 0; s < nseq; s++) {
        size_t off = (size_t)seq_off[s];
        e2_attention_text(ctx, out + off * qs, q + off * qs, k + off * ks,
                          v + off * ks, t, heads, kv_heads, head_dim, window);
    }
}
