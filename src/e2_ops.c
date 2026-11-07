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
__attribute__((target("avx512f")))
static void softmax_exp_v(float *sc, int n) {
    float mx = -INFINITY;
    int i = 0;
    __m512 vm = _mm512_set1_ps(-INFINITY);
    for (; i + 16 <= n; i += 16) vm = _mm512_max_ps(vm, _mm512_loadu_ps(sc + i));
    mx = _mm512_reduce_max_ps(vm);
    for (; i < n; i++) if (sc[i] > mx) mx = sc[i];
    __m512 vmx = _mm512_set1_ps(mx);
    __m512 vs = _mm512_setzero_ps();
    for (i = 0; i + 16 <= n; i += 16) {
        __m512 e = vexp(_mm512_sub_ps(_mm512_loadu_ps(sc + i), vmx));
        _mm512_storeu_ps(sc + i, e);
        vs = _mm512_add_ps(vs, e);
    }
    float sum = _mm512_reduce_add_ps(vs);
    for (; i < n; i++) { sc[i] = e2_fexp(sc[i] - mx); sum += sc[i]; }
    __m512 vinv = _mm512_set1_ps(1.0f / sum);
    for (i = 0; i + 16 <= n; i += 16)
        _mm512_storeu_ps(sc + i, _mm512_mul_ps(_mm512_loadu_ps(sc + i), vinv));
    for (; i < n; i++) sc[i] = sc[i] / sum;
}
#endif

static int simd_pick(void) {
#ifdef E2_X86OPS
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return 2;
    if (__builtin_cpu_supports("avx2")) return 1;
#endif
    return 0;
}
static int g_simd = -1;
static inline int simd(void) { if (g_simd < 0) g_simd = simd_pick(); return g_simd; }

void e2_gelu_tanh(float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) { gelu_v(x, n); return; }
#endif
    gelu_s(x, n);
}

void e2_gelu_tanh_ctx(e2_ctx *ctx, float *x, int n) {
    run_split(ctx, e2_gelu_tanh, x, n);
}

void e2_silu(float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) { silu_v(x, n); return; }
#endif
    silu_s(x, n);
}

void e2_vec_scale(float *x, int n, float k) {
#ifdef E2_X86OPS
    if (simd() >= 2) { scale_v(x, n, k); return; }
#endif
    scale_s(x, n, k);
}

void e2_add_inplace(float *a, const float *b, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2) { add_v(a, b, n); return; }
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
static void gelu_mul_v(float *x, const float *up, int n) {
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
        _mm512_storeu_ps(x + i, _mm512_mul_ps(r, _mm512_loadu_ps(up + i)));
    }
    for (; i < n; i++) x[i] = gelu1(x[i]) * up[i];
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

void e2_rmsnorm_pre(float *out, const float *x, const float *w, int rows, int dim,
                    float eps, float pre) {
    if (pre == 1.0f) { e2_rmsnorm(out, x, w, rows, dim, eps); return; }
#ifdef E2_X86OPS
    if (simd() >= 2) { rmsnorm_pre_v(out, x, w, rows, dim, eps, pre); return; }
#endif
    rmsnorm_pre_s(out, x, w, rows, dim, eps, pre);
}

void e2_rmsnorm_add(float *x, const float *src, const float *w, int rows, int dim,
                    float eps) {
#ifdef E2_X86OPS
    if (simd() >= 2) { rmsnorm_add_v(x, src, w, rows, dim, eps); return; }
#endif
    rmsnorm_add_s(x, src, w, rows, dim, eps);
}

void e2_gelu_mul(e2_ctx *ctx, float *x, const float *up, int n) {
    if (!ctx || ctx->nthreads <= 1 || n < 65536) {
#ifdef E2_X86OPS
        if (simd() >= 2) { gelu_mul_v(x, up, n); return; }
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

void e2_rmsnorm(float *out, const float *x, const float *w, int rows, int dim, float eps) {
#ifdef E2_X86OPS
    if (simd() >= 2) { rmsnorm_v(out, x, w, rows, dim, eps); return; }
#endif
    for (int r = 0; r < rows; r++) {
        const float *xp = x + (size_t)r * dim;
        float *op = out + (size_t)r * dim;
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += xp[i] * xp[i];
        float s = 1.0f / sqrtf(sum / (float)dim + eps);
        if (w)
            for (int i = 0; i < dim; i++) op[i] = xp[i] * s * w[i];
        else
            for (int i = 0; i < dim; i++) op[i] = xp[i] * s;
    }
}

void e2_rmsnorm_noweight(float *out, const float *x, int rows, int dim, float eps) {
    e2_rmsnorm(out, x, NULL, rows, dim, eps);
}

void e2_l2norm_rows(float *x, int rows, int dim) {
#ifdef E2_X86OPS
    if (simd() >= 2) { l2norm_v(x, rows, dim); return; }
#endif
    for (int r = 0; r < rows; r++) {
        float *p = x + (size_t)r * dim;
        float s = 0.0f;
        for (int i = 0; i < dim; i++) s += p[i] * p[i];
        s = 1.0f / sqrtf(s + 1e-12f);
        for (int i = 0; i < dim; i++) p[i] *= s;
    }
}

void e2_softmax(float *x, int n) {
#ifdef E2_X86OPS
    if (simd() >= 2 && n >= 16) { softmax_exp_v(x, n); return; }
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
#endif
    rot_s(x, c, s, half);
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
    long w0 = total * tid / nt, w1 = total * (tid + 1) / nt;
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
    long w0 = total * tid / nt, w1 = total * (tid + 1) / nt;

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
        else
#endif
        attn_pv_s(out, q, vb, kv_stride, n, head_dim, sc, tpad);
    }
    free(sc);
}

typedef struct attn_ws attn_ws;
static int attn_int8(e2_ctx *ctx, float *out, const float *q, const float *k,
                     const float *v, int t, int heads, int kv_heads, int head_dim,
                     int window, attn_ws *ws, int qstride, int kstride, int ostride);

void e2_attention_text(e2_ctx *ctx, float *out, const float *q, const float *k,
                       const float *v, int t, int heads, int kv_heads, int head_dim,
                       int window) {
    if (t <= 0 || heads <= 0) return;
#ifdef E2_X86OPS
    if (simd() >= 2 && t >= 32) {
        static int noint8 = -1;
        if (noint8 < 0) noint8 = getenv("E2_ATTN_F32") ? 1 : 0;
        if (!noint8 && attn_int8(ctx, out, q, k, v, t, heads, kv_heads, head_dim,
                                 window, NULL, heads * head_dim, kv_heads * head_dim,
                                 heads * head_dim) == 0)
            return;
    }
#endif
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

void e2_attention_vision(e2_ctx *ctx, float *out, const float *q, const float *k,
                         const float *v, int t, int heads, int head_dim) {
    e2_attention_text(ctx, out, q, k, v, t, heads, heads, head_dim, 0);
}

void e2_attention_full(e2_ctx *ctx, float *out, const float *q, const float *k,
                       const float *v, int t, int heads, int head_dim) {
    e2_attention_text(ctx, out, q, k, v, t, heads, heads, head_dim, 0);
}

/* ------------------------------------------------------------------ */
/* int8 attention: scores = Q x K^T and out = P x V^T, both through    */
/* the packed-int8 GEMM (4x the f32 MAC rate, 1.6 dpbusd/cycle).       */
/* ------------------------------------------------------------------ */

#define ATTN_MAXR 512

typedef struct {
    int t, tpad, head_dim, heads, kv_heads, window, ablk, rows;
    const float *q, *k, *v;
    float *out;
    e2_qmat kq, vq;           /* per kv head, repacked inside */
    int8_t *kp; float *ks; int32_t *kw;
    int8_t *vp; float *vs; int32_t *vw;
    float *sc;                /* rows x tpad scores */
    float *qg;                /* rows x head_dim gather buffer */
    float *ob;                /* rows x head_dim output buffer */
} attn8;

/* mask + softmax one score row over [lo,hi), tpad-wide buffer */
static void attn8_softmax(float *sr, int tpad, int lo, int hi) {
    for (int j = 0; j < lo; j++) sr[j] = -INFINITY;
    for (int j = hi; j < tpad; j++) sr[j] = -INFINITY;
#ifdef E2_X86OPS
    if (simd() >= 2) { softmax_exp_v(sr, tpad); return; }
#endif
    e2_softmax(sr, tpad);
}

#ifdef E2_X86OPS
__attribute__((target("avx512f")))
static inline __m512 attn8_exp16(const float *sr, int j0, __m512 vmx,
                                 int lo, int hi) {
    __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(j0),
                                   _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
    __mmask16 m = _mm512_cmp_epi32_mask(idx, _mm512_set1_epi32(lo), _MM_CMPINT_GE) &
                  _mm512_cmp_epi32_mask(idx, _mm512_set1_epi32(hi), _MM_CMPINT_LT);
    __m512 e = vexp(_mm512_sub_ps(_mm512_loadu_ps(sr + j0), vmx));
    return _mm512_maskz_mov_ps(m, e);
}
#endif

/* Fused softmax + u8 quantisation of the PV activations: unnormalised
 * exp(x-max) with per-128-block scales, 1/sum applied after the PV GEMM.
 * Two f32 reads and one u8 write instead of five f32 passes. */
__attribute__((target("avx512f")))
static void attn8_sq(const float *sr, int tpad, int lo, int hi,
                     uint8_t *q, float *xs, float *invsum) {
#ifdef E2_X86OPS
    if (simd() >= 2) {
        int j = lo;
        __m512 vm = _mm512_set1_ps(-INFINITY);
        for (; j + 16 <= hi; j += 16) vm = _mm512_max_ps(vm, _mm512_loadu_ps(sr + j));
        float mxf = _mm512_reduce_max_ps(vm);
        for (; j < hi; j++) if (sr[j] > mxf) mxf = sr[j];
        __m512 vmx = _mm512_set1_ps(mxf);
        float sum = 0.0f;
        int nblk = tpad / 128;
        for (int b = 0; b < nblk; b++) {
            int b0 = b * 128;
            __m512 e[8];
            __m512 bmax = _mm512_setzero_ps();
            for (int k = 0; k < 8; k++) {
                e[k] = attn8_exp16(sr, b0 + k * 16, vmx, lo, hi);
                bmax = _mm512_max_ps(bmax, e[k]);
            }
            float emax = _mm512_reduce_max_ps(bmax);
            float scale = emax / 127.0f;
            if (scale < 1e-12f) scale = 1e-12f;
            xs[b] = scale;
            __m512 vinv = _mm512_set1_ps(1.0f / scale);
            __m512 v128 = _mm512_set1_ps(128.0f);
            __m512 ssum = _mm512_setzero_ps();
            for (int k = 0; k < 8; k++) {
                ssum = _mm512_add_ps(ssum, e[k]);
                __m512i iv = _mm512_cvtps_epi32(_mm512_fmadd_ps(e[k], vinv, v128));
                _mm_storeu_si128((__m128i *)(q + b0 + k * 16),
                                 _mm512_cvtusepi32_epi8(iv));
            }
            sum += _mm512_reduce_add_ps(ssum);
        }
        *invsum = sum > 0.0f ? 1.0f / sum : 0.0f;
        return;
    }
#endif
    float mx = -INFINITY;
    for (int j = lo; j < hi; j++) if (sr[j] > mx) mx = sr[j];
    float sum = 0.0f;
    int nblk = tpad / 128;
    for (int b = 0; b < nblk; b++) {
        int b0 = b * 128;
        float e[128], emax = 0.0f;
        for (int i = 0; i < 128; i++) {
            int j = b0 + i;
            e[i] = (j >= lo && j < hi) ? e2_fexp(sr[j] - mx) : 0.0f;
            if (e[i] > emax) emax = e[i];
        }
        float scale = emax / 127.0f;
        if (scale < 1e-12f) scale = 1e-12f;
        xs[b] = scale;
        float inv = 1.0f / scale;
        for (int i = 0; i < 128; i++) {
            long v = lrintf(e[i] * inv) + 128;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            q[b0 + i] = (uint8_t)v;
            sum += e[i];
        }
    }
    *invsum = sum > 0.0f ? 1.0f / sum : 0.0f;
}

struct attn_ws {
    float *kt;        /* kv_heads * head_dim * tpad */
    uint8_t *xq;      /* max(t*head_dim, t*tpad) */
    float *xs;        /* activation scales */
    float *tmp;       /* clip scratch */
};

static int attn_int8(e2_ctx *ctx, float *out, const float *q, const float *k,
                     const float *v, int t, int heads, int kv_heads, int head_dim,
                     int window, attn_ws *ws, int qstride, int kstride, int ostride) {
    int groups = heads / kv_heads;
    int ablk = (head_dim % 128 == 0) ? 128 : ((head_dim % 64 == 0) ? 64 : 0);
    if (!ablk) return -1;
    int tpad = (t + 127) & ~127;
    int rows = t < ATTN_MAXR ? t : ATTN_MAXR;
    if (rows < 1) return -1;
    size_t sbytes = (size_t)rows * tpad;
    size_t sz = sbytes * sizeof(float) + sbytes + (size_t)rows * (tpad / 128 + 1) * 4
                + (size_t)rows * head_dim * 8 + (size_t)rows * head_dim * 4;
    char *base = malloc(sz);
    if (!base) return -1;
    float *sc = (float *)base;
    float *qf = sc + sbytes;
    float *ob = qf + (size_t)rows * head_dim;
    int nblk = tpad / 128;
    int sq = 0;
#ifdef E2_X86OPS
    sq = simd() >= 2 && getenv("E2_ATTN_FUSED_SQ");
#endif
    size_t sqsz = sq ? (size_t)rows * tpad + (size_t)rows * nblk * 4
                       + (size_t)rows * 4 : 0;
    uint8_t *xqs = sq ? malloc(sqsz) : NULL;
    uint8_t *sxq = xqs;
    float *sxs = sq ? (float *)(xqs + (size_t)rows * tpad) : NULL;
    float *sinv = sq ? sxs + (size_t)rows * nblk : NULL;
    if (sq && !xqs) sq = 0;

    /* V^T via the blocked transpose so the packer walks contiguous rows */
    size_t ktn = (size_t)kv_heads * head_dim * tpad;
    float *kt;
    if (ws) {
        kt = ws->kt;
    } else {
        if (ctx->ktcap < ktn) {
            free(ctx->ktbuf);
            ctx->ktbuf = malloc(ktn * sizeof(float));
            ctx->ktcap = ctx->ktbuf ? ktn : 0;
        }
        if (!ctx->ktbuf) { free(base); return -1; }
        kt = ctx->ktbuf;
    }
    memset(kt, 0, ktn * sizeof(float));
    kt_arg ka = {v, kt, t, tpad, kv_heads, head_dim, kstride};
    if (ws || (long)kv_heads * ((head_dim + 15) / 16) * ((t + 15) / 16) < 512)
        kt_work(&ka, 0, 1);
    else
        e2_parallel(ctx, kt_work, &ka);

    e2_qmat kq = {0}, vq = {0};
    kq.n = tpad; kq.k = head_dim; kq.blk = ablk; kq.packed = 1;
    vq.n = head_dim; vq.k = tpad; vq.blk = 128; vq.packed = 1;

    for (int kh = 0; kh < kv_heads; kh++) {
        int8_t *kq_p = NULL, *vq_p = NULL;
        float *ksc = NULL, *vsc = NULL;
        int32_t *kwm = NULL, *vwm = NULL;
        int kb = ablk, vb = 128;
        if ((ws ? e2_pack_mat(k + (size_t)kh * head_dim, t, head_dim,
                              kstride, 1, &kq_p, &ksc, &kwm, &kb)
                : e2_pack_mat_ctx(ctx, k + (size_t)kh * head_dim, t, head_dim,
                                  kstride, 1, &kq_p, &ksc, &kwm, &kb))) {
            free(base); return -1;
        }
        int vrc = ws ? e2_pack_mat(kt + (size_t)kh * head_dim * tpad, head_dim,
                                   tpad, tpad, 1, &vq_p, &vsc, &vwm, &vb)
                     : e2_pack_mat_ctx(ctx, kt + (size_t)kh * head_dim * tpad,
                                       head_dim, tpad, tpad, 1, &vq_p, &vsc,
                                       &vwm, &vb);
        if (vrc) {
            e2_pack_free(kq_p, ksc, kwm); free(base); return -1;
        }
        kq.q = kq_p; kq.scale = ksc; kq.wsum = kwm;
        vq.q = vq_p; vq.scale = vsc; vq.wsum = vwm;
        e2_linear kql = {0}, vql = {0};
        kql.w = kq; vql.w = vq;
        for (int g = 0; g < groups; g++) {
            int h = kh * groups + g;
            for (int i0 = 0; i0 < t; i0 += rows) {
                int nq = t - i0; if (nq > rows) nq = rows;
                /* gather q rows for this head */
                for (int r = 0; r < nq; r++)
                    memcpy(qf + (size_t)r * head_dim,
                           q + (size_t)(i0 + r) * qstride + (size_t)h * head_dim,
                           (size_t)head_dim * sizeof(float));
                if (ws) e2_linear_forward_st(sc, qf, nq, &kql, ws->xq, ws->xs, ws->tmp);
                else e2_linear_forward(ctx, sc, qf, nq, &kql);
                if (sq) {
                    for (int r = 0; r < nq; r++) {
                        int lo_r = 0, hi_r = t;
                        if (window > 0) {
                            lo_r = i0 + r - window; if (lo_r < 0) lo_r = 0;
                            hi_r = i0 + r + window + 1; if (hi_r > t) hi_r = t;
                        }
                        attn8_sq(sc + (size_t)r * tpad, tpad, lo_r, hi_r,
                                 sxq + (size_t)r * tpad, sxs + (size_t)r * nblk,
                                 sinv + r);
                    }
                } else {
                    for (int r = 0; r < nq; r++) {
                        int32_t lo_r = 0, hi_r = t;
                        if (window > 0) {
                            lo_r = i0 + r - window; if (lo_r < 0) lo_r = 0;
                            hi_r = i0 + r + window + 1; if (hi_r > t) hi_r = t;
                        }
                        attn8_softmax(sc + (size_t)r * tpad, tpad, lo_r, hi_r);
                    }
                }
                if (sq) {
                    e2_linear_forward_pq(ws ? NULL : ctx, ob, sxq, sxs, nq, &vql);
                    for (int r = 0; r < nq; r++) {
                        float iv = sinv[r];
                        float *op = ob + (size_t)r * head_dim;
                        for (int d = 0; d < head_dim; d++) op[d] *= iv;
                    }
                } else if (ws) {
                    e2_linear_forward_st(ob, sc, nq, &vql, ws->xq, ws->xs, ws->tmp);
                } else {
                    e2_linear_forward(ctx, ob, sc, nq, &vql);
                }
                for (int r = 0; r < nq; r++)
                    memcpy(out + (size_t)(i0 + r) * ostride + (size_t)h * head_dim,
                           ob + (size_t)r * head_dim, (size_t)head_dim * sizeof(float));
            }
        }
        e2_pack_free(kq_p, ksc, kwm);
        e2_pack_free(vq_p, vsc, vwm);
    }
    free(base);
    free(xqs);
    return 0;
}


/* Batched short-sequence int8 attention: one work item per sequence.  A
 * separate e2_parallel per (head, group, block) costs more than the actual
 * 64-token attention, so the batch path runs each sequence on one worker. */
typedef struct {
    float *out; const float *q, *k, *v;
    int t, heads, kv_heads, head_dim, window, nseq;
    const int *seq_off;
    attn_ws *ws;
    int err;
} attn_multi_arg;

static void attn_multi_work(void *vp, int tid, int nt) {
    attn_multi_arg *a = vp;
    size_t qs = (size_t)a->heads * a->head_dim;
    size_t ks = (size_t)a->kv_heads * a->head_dim;
    for (int s = tid; s < a->nseq; s += nt) {
        size_t off = (size_t)a->seq_off[s];
        if (attn_int8(NULL, a->out + off * qs, a->q + off * qs, a->k + off * ks,
                      a->v + off * ks, a->t, a->heads, a->kv_heads, a->head_dim,
                      a->window, &a->ws[tid], (int)qs, (int)ks, (int)qs) != 0) {
            a->err = 1;
            return;
        }
    }
}

void e2_attention_text_multi(e2_ctx *ctx, float *out, const float *q, const float *k,
                             const float *v, int t, const int *seq_off, int nseq,
                             int heads, int kv_heads, int head_dim, int window) {
    size_t qs = (size_t)heads * head_dim, ks = (size_t)kv_heads * head_dim;
    static int noint8 = -1;
    if (noint8 < 0) noint8 = getenv("E2_ATTN_F32") ? 1 : 0;
    if (nseq <= 1 || t < 32 || t > 256 || noint8 || simd() < 2 ||
        (head_dim % 64) != 0) {
        for (int s = 0; s < nseq; s++) {
            size_t off = (size_t)seq_off[s];
            e2_attention_text(ctx, out + off * qs, q + off * qs, k + off * ks,
                              v + off * ks, t, heads, kv_heads, head_dim, window);
        }
        return;
    }
    int nt = ctx->nthreads;
    int tpad = (t + 127) & ~127;
    int ablk = (head_dim % 128 == 0) ? 128 : 64;
    size_t ktn = (size_t)kv_heads * head_dim * tpad;
    size_t qcap = (size_t)t * (head_dim > tpad ? head_dim : tpad);
    size_t sblk = (size_t)(head_dim / ablk);
    size_t pblk = (size_t)(tpad / 128);
    size_t scap = (size_t)t * (sblk > pblk ? sblk : pblk);
    attn_ws *ws = calloc((size_t)nt, sizeof(*ws));
    float *kt = malloc(ktn * sizeof(float) * nt);
    uint8_t *xq = malloc(qcap * nt);
    float *xs = malloc(scap * sizeof(float) * nt);
    float *tmp = malloc(qcap * sizeof(float) * nt);
    if (!ws || !kt || !xq || !xs || !tmp) {
        free(ws); free(kt); free(xq); free(xs); free(tmp);
        for (int s = 0; s < nseq; s++) {
            size_t off = (size_t)seq_off[s];
            e2_attention_text(ctx, out + off * qs, q + off * qs, k + off * ks,
                              v + off * ks, t, heads, kv_heads, head_dim, window);
        }
        return;
    }
    for (int i = 0; i < nt; i++) {
        ws[i].kt = kt + ktn * i;
        ws[i].xq = xq + qcap * i;
        ws[i].xs = xs + scap * i;
        ws[i].tmp = tmp + qcap * i;
    }
    attn_multi_arg a = {out, q, k, v, t, heads, kv_heads, head_dim, window, nseq,
                        seq_off, ws, 0};
    e2_parallel(ctx, attn_multi_work, &a);
    if (a.err) {
        for (int s = 0; s < nseq; s++) {
            size_t off = (size_t)seq_off[s];
            e2_attention_text(ctx, out + off * qs, q + off * qs, k + off * ks,
                              v + off * ks, t, heads, kv_heads, head_dim, window);
        }
    }
    free(ws); free(kt); free(xq); free(xs); free(tmp);
}

