/* kernels.c — scalar reference backend + backend dispatch. */
#include "dispatch.h"
#include "int8.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- shared helpers */
void e2_bf16_to_f32(float *dst, const uint16_t *src, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = e2_bf16f(src[i]);
}

float e2_dot_f32(const float *a, const float *b, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void e2_l2norm_f32(float *x, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float inv = 1.f / sqrtf(s);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void e2_meanpool_f32(float *out, const float *x, int M, int D, const uint8_t *valid) {
    memset(out, 0, sizeof(float) * (size_t)D);
    int cnt = 0;
    for (int m = 0; m < M; m++) {
        if (valid && !valid[m]) continue;
        const float *row = x + (size_t)m * D;
        for (int d = 0; d < D; d++) out[d] += row[d];
        cnt++;
    }
    if (cnt > 0) for (int d = 0; d < D; d++) out[d] /= (float)cnt;
}

/* ---------------------------------------------------------------- scalar backend */
static void scalar_gemm_nt(float *y, const float *x, const void *w, int M, int N, int K, e2_dtype wd) {
    const float *wf = (const float *)w;
    const uint16_t *wb = (const uint16_t *)w;
    for (int n = 0; n < N; n++) {
        const float *wr = wd == E2_DT_BF16 ? NULL : wf + (size_t)n * K;
        const uint16_t *wrb = wd == E2_DT_BF16 ? wb + (size_t)n * K : NULL;
        for (int m = 0; m < M; m++) {
            const float *xr = x + (size_t)m * K;
            float s = 0.f;
            if (wd == E2_DT_BF16)
                for (int k = 0; k < K; k++) s += xr[k] * e2_bf16f(wrb[k]);
            else
                for (int k = 0; k < K; k++) s += xr[k] * wr[k];
            y[(size_t)m * N + n] = s;
        }
    }
}

static void scalar_gemv(float *y, const void *w, const float *x, int N, int K, e2_dtype wd) {
    const float *wf = (const float *)w;
    const uint16_t *wb = (const uint16_t *)w;
    for (int n = 0; n < N; n++) {
        float s = 0.f;
        if (wd == E2_DT_BF16) {
            const uint16_t *wr = wb + (size_t)n * K;
            for (int k = 0; k < K; k++) s += x[k] * e2_bf16f(wr[k]);
        } else {
            const float *wr = wf + (size_t)n * K;
            for (int k = 0; k < K; k++) s += x[k] * wr[k];
        }
        y[n] = s;
    }
}

static void scalar_rmsnorm(float *x, int M, int D, const float *gamma, float eps) {
    for (int m = 0; m < M; m++) {
        float *row = x + (size_t)m * D;
        float ss = 0.f;
        for (int d = 0; d < D; d++) ss += row[d] * row[d];
        float inv = 1.f / sqrtf(ss / (float)D + eps);
        if (gamma)
            for (int d = 0; d < D; d++) row[d] = row[d] * inv * gamma[d];
        else
            for (int d = 0; d < D; d++) row[d] *= inv;
    }
}

static void scalar_gelu_tanh(float *x, size_t n) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

static void scalar_softmax(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); sum += x[i]; }
    float inv = 1.f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static float scalar_dot_f32(const float *a, const float *b, size_t n) { return e2_dot_f32(a, b, n); }

static void scalar_axpy(float *y, const float *x, float a, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

static void scalar_rope_rows(float *x, int r0, int r1, int ngroups, int head_dim, int block,
                             const float *cs, const float *sn) {
    const int half = block / 2;
    for (int r = r0; r < r1; r++) {
        float *v = x + (size_t)r * head_dim;
        const float *c = cs + (size_t)(r / ngroups) * head_dim;
        const float *s = sn + (size_t)(r / ngroups) * head_dim;
        for (int b = 0; b < head_dim; b += block)
            for (int i = 0; i < half; i++) {
                float a = v[b + i], y = v[b + half + i];
                v[b + i] = a * c[b + i] - y * s[b + i];
                v[b + half + i] = y * c[b + i] + a * s[b + i];
            }
    }
}


/* ---------------------------------------------------------------- elementwise map ops
 * Portable reference implementations; kernels_avx2.c has the vectorised versions of the same ops. */
static void scalar_mul(float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] *= b[i];
}
static void scalar_add(float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] += b[i];
}
static void scalar_scale(float *a, float s, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] *= s;
}
static void scalar_mul_rows(float *a, const float *b, int rows, int D, int stride) {
    for (int r = 0; r < rows; r++)
        for (int d = 0; d < D; d++) a[(size_t)r * D + d] *= b[(size_t)r * stride + d];
}
static float scalar_row_inv(const float *x, int D, float eps) {
    float ss = 0.f;
    for (int d = 0; d < D; d++) ss += x[d] * x[d];
    return 1.0f / sqrtf(ss / (float)D + eps);
}
static void scalar_norm_split(float *x, float *res, int rows, int D, const float *gamma, float eps) {
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * D;
        float *rr = res + (size_t)r * D;
        const float inv = scalar_row_inv(xr, D, eps);
        for (int d = 0; d < D; d++) {
            const float v = xr[d];
            rr[d] = v;
            xr[d] = v * inv * (gamma ? gamma[d] : 1.0f);
        }
    }
}
static void scalar_norm_add(float *out, const float *src, const float *res, int rows, int D,
                            const float *gamma, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *sr = src + (size_t)r * D;
        const float *rr = res + (size_t)r * D;
        float *orow = out + (size_t)r * D;
        const float inv = scalar_row_inv(sr, D, eps);
        for (int d = 0; d < D; d++)
            orow[d] = sr[d] * inv * (gamma ? gamma[d] : 1.0f) + rr[d];
    }
}

static void scalar_clamp(float *x, float lo, float hi, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
}

static const e2_kernels g_scalar = {
    .name = "scalar",
    .gemm_nt = scalar_gemm_nt,
    .gemv = scalar_gemv,
    .rmsnorm = scalar_rmsnorm,
    .gelu_tanh = scalar_gelu_tanh,
    .softmax = scalar_softmax,
    .dot_f32 = scalar_dot_f32,
    .axpy = scalar_axpy,
    .mul = scalar_mul,
    .add = scalar_add,
    .scale = scalar_scale,
    .mul_rows = scalar_mul_rows,
    .norm_split = scalar_norm_split,
    .norm_add = scalar_norm_add,
    .rope_rows = scalar_rope_rows,
    .silu = e2_silu_scalar, /* activations.c: must not be rcpss-approximated */
    .glu = e2_glu_scalar,
    .clamp = scalar_clamp,
};

/* ---------------------------------------------------------------- scalar f32 GEMM
 *
 * C[m][n] = sum_k A[m][k] * B[k][n], walked k-outer so a row of B is read once for four rows of C
 * and the inner loop is a map, not a reduction: `c[n] += a*b[n]` over n vectorises without
 * -ffast-math, which a dot product over k cannot. Every C element still accumulates over k in
 * order, so the result is bit-identical to the naive i-j-k form it replaces - the reference stays
 * the reference.
 *
 * This is the path every host without AVX2/NEON takes, both for the bf16 weight GEMMs and for the
 * fp32 attention products. The naive form read B with stride ldb, i.e. one cache line per MAC:
 * 0.4 GMAC/s on a 5 GHz core; this is 45x faster on the same shape. */
#define SG_ROW 4
void e2_sgemm_nn_range_scalar(float *C, int ldc, const float *A, int lda, const float *B, int ldb, int M,
                              int N, int K, int m0, int m1) {
    (void)M;
    int m = m0;
    for (; m + SG_ROW <= m1; m += SG_ROW) {
        const float *a[SG_ROW];
        float *c[SG_ROW];
        for (int r = 0; r < SG_ROW; r++) {
            a[r] = A + (size_t)(m + r) * lda;
            c[r] = C + (size_t)(m + r) * ldc;
        }
        for (int n = 0; n < N; n++)
            for (int r = 0; r < SG_ROW; r++) c[r][n] = 0.f;
        for (int k = 0; k < K; k++) {
            const float *b = B + (size_t)k * ldb;
            float v[SG_ROW];
            for (int r = 0; r < SG_ROW; r++) v[r] = a[r][k];
            for (int n = 0; n < N; n++) {
                const float w = b[n];
                for (int r = 0; r < SG_ROW; r++) c[r][n] += v[r] * w;
            }
        }
    }
    for (; m < m1; m++) {
        const float *arow = A + (size_t)m * lda;
        float *crow = C + (size_t)m * ldc;
        for (int n = 0; n < N; n++) crow[n] = 0.f;
        for (int k = 0; k < K; k++) {
            const float av = arow[k];
            const float *b = B + (size_t)k * ldb;
            for (int n = 0; n < N; n++) crow[n] += av * b[n];
        }
    }
}

/* ---------------------------------------------------------------- backend selection */
static const e2_kernels *g_active = &g_scalar;
static const char *g_cpu_name = "scalar";

extern const e2_kernels *e2_avx2_kernels(void); /* kernels_avx2.c, returns NULL if unavailable */
extern const e2_kernels *e2_avx512_kernels(void); /* kernels_avx512.c, returns NULL if unavailable */
extern void e2_i8_gemm_vnni_range(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
extern void e2_i8_gemm_avx2(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
extern void e2_i8_gemm_avx512(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
extern void e2_i8_attn_gemm_avx512(float *, int, const uint8_t *, int, const float *, const uint8_t *,
                                   const float *, const int32_t *, int, int, int, int);
extern void e2_softmax_quant_i8_avx512(float *, int8_t *, float *, int, int, int);
extern void e2_i8_attn_gemm_vnni(float *, int, const uint8_t *, int, const float *, const uint8_t *,
                                const float *, const int32_t *, int, int, int, int);
extern void e2_softmax_quant_i8_avx2(float *, int8_t *, float *, int, int, int);
extern void e2_quantize_blocks_vnni(int8_t *, float *, const float *, int, int, float, float);
extern void e2_quantize_blocks_avx2(int8_t *, float *, const float *, int, int, float, float);
extern void e2_quantize_blocks_avx512(int8_t *, float *, const float *, int, int, float, float);
void e2_i8_gemm_scalar(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
void e2_quantize_blocks_scalar(int8_t *, float *, const float *, int, int, float, float);
#if defined(__aarch64__)
extern void e2_i8_gemm_neon(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
extern void e2_quantize_blocks_neon(int8_t *, float *, const float *, int, int, float, float);
extern void e2_i8_attn_gemm_neon(float *, int, const uint8_t *, int, const float *, const uint8_t *,
                                 const float *, const int32_t *, int, int, int, int);
extern void e2_softmax_quant_i8_neon(float *, int8_t *, float *, int, int, int);
#endif

e2_i8_gemm_fn e2_i8_gemm = NULL;
void (*e2_sgemm_nn_range)(float *, int, const float *, int, const float *, int, int, int, int, int, int) =
    e2_sgemm_nn_range_scalar;
extern void e2_sgemm_nn_range_avx2(float *, int, const float *, int, const float *, int, int, int, int, int, int);
extern void e2_sgemm_nn_range_avx512(float *, int, const float *, int, const float *, int, int, int, int, int, int);
#if defined(__aarch64__)
extern void e2_sgemm_nn_range_neon(float *, int, const float *, int, const float *, int, int, int, int,
                                   int, int);
#endif
void (*e2_i8_quantize)(int8_t *, float *, const float *, int, int, float, float) = NULL;
e2_i8_attn_fn e2_i8_attn_gemm = NULL;
e2_sq_fn e2_softmax_quant_i8 = NULL;

/* __builtin_cpu_supports is x86-only: on any other host every SIMD feature collapses to "no". */
#if defined(__x86_64__) || defined(_M_X64)
static int cpu_has_avxvnni(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avxvnni");
#else
    return 0;
#endif
}

/* AVX512-VNNI: the 512-bit int8 kernel needs the full set (BW/VL for the masked tails). */
static int cpu_has_avx512vnni(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vnni");
#else
    return 0;
#endif
}

static int cpu_has_avx512(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
#else
    return 0;
#endif
}

static int cpu_has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
#else
/* Not x86: the probes are only referenced by the x86 rows of the registry and by the tests, so they
 * are inline stubs rather than static functions - an unused static function is a -Wall warning, and
 * on aarch64 the library uses none of them. */
static inline int cpu_has_avxvnni(void) { return 0; }
static inline int cpu_has_avx512vnni(void) { return 0; }
static inline int cpu_has_avx512(void) { return 0; }
static inline int cpu_has_avx2(void) { return 0; }
#endif

/* aarch64: the int8 kernels need the ARMv8.2 dot-product extension (vdotq_u32), which is what
 * Neoverse-N1 and everything newer have. Without it nothing here is ever called. */
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
static int cpu_has_neon_dotprod(void) { return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0; }
#endif

/* ---------------------------------------------------------------- backend registry
 *
 * One row per backend, in preference order, for each of the two ladders. They are separate because a
 * CPU can have AVX-512 without VNNI: it then runs the AVX-512 fp32 kernels and the AVX2 int8 ones.
 * A row is {probe, install, capability id}; the first probe that succeeds installs and stops the
 * walk, and the last row has no probe so the walk always terminates. Adding a backend is a row plus
 * a fill function, not another else-if.
 */
typedef struct {
    const char *name; /* E2_CPU_BACKEND value and e2_cpu_backend_name() */
    int tier;         /* scalar 0 < avx2 1 < vnni 2 < avx512 3; neon 1 on its own architecture */
    int (*probe)(void); /* NULL = unconditional */
    void (*install)(void);
    e2_backend bk; /* the capability id dispatch.c keys its rules on */
} e2_i8_backend_def;

typedef struct {
    const char *name;
    int tier; /* same scale, without vnni: there is no VNNI fp32 backend */
    int (*probe)(void);
    int (*install)(void); /* 0 = this row declined, keep walking */
} e2_f32_backend_def;

#if defined(__x86_64__) || defined(_M_X64)
static void install_i8_avx512vnni(void) {
    e2_i8_gemm = e2_i8_gemm_avx512;
    e2_i8_quantize = e2_quantize_blocks_avx512;
    e2_i8_attn_gemm = e2_i8_attn_gemm_avx512;
    e2_softmax_quant_i8 = e2_softmax_quant_i8_avx512;
}
static void install_i8_vnni(void) {
    e2_i8_gemm = e2_i8_gemm_vnni_range;
    e2_i8_quantize = e2_quantize_blocks_vnni;
    e2_i8_attn_gemm = e2_i8_attn_gemm_vnni;
    e2_softmax_quant_i8 = e2_softmax_quant_i8_avx2;
}
static void install_i8_avx2(void) {
    e2_i8_gemm = e2_i8_gemm_avx2;
    e2_i8_quantize = e2_quantize_blocks_avx2;
    e2_softmax_quant_i8 = e2_softmax_quant_i8_avx2; /* no maddubs attention kernel */
}
static int install_f32_avx512(void) {
    const e2_kernels *k = e2_avx512_kernels();
    if (!k) return 0;
    e2_sgemm_nn_range = e2_sgemm_nn_range_avx512;
    g_active = k;
    g_cpu_name = k->name;
    return 1;
}
static int install_f32_avx2(void) {
    const e2_kernels *k = e2_avx2_kernels();
    if (!k) return 0;
    e2_sgemm_nn_range = e2_sgemm_nn_range_avx2;
    g_active = k;
    g_cpu_name = k->name;
    return 1;
}
#endif

#if defined(__aarch64__)
static void install_i8_neon(void) {
    e2_i8_gemm = e2_i8_gemm_neon;
    e2_i8_quantize = e2_quantize_blocks_neon;
    e2_i8_attn_gemm = e2_i8_attn_gemm_neon;
    e2_softmax_quant_i8 = e2_softmax_quant_i8_neon;
}
static int install_f32_neon(void) {
    e2_sgemm_nn_range = e2_sgemm_nn_range_neon;
    /* ASIMD is baseline, but there is no NEON elementwise vtable, so g_active stays scalar while the
     * products and the quantizer run on NEON. The name says what actually runs - it used to report
     * "scalar" here, which is what HARDWARE.md's machine-identification line then printed. */
    g_cpu_name = "neon";
    return 1;
}
#endif

/* The portable backend is a real int8 backend now, not just a reference: the 8-row kernel and the
 * quantizer are plain C, so a host with no SIMD at all runs the int8 weights (1/2 the bytes of bf16,
 * exact integer dots) instead of the bf16 GEMMs. Everything above this row is a faster way to do the
 * same arithmetic, which is what keeps the fidelity numbers comparable across backends. */
static void install_i8_scalar(void) {
    e2_i8_gemm = e2_i8_gemm_scalar;
    e2_i8_quantize = e2_quantize_blocks_scalar;
}
static int install_f32_scalar(void) {
    e2_sgemm_nn_range = e2_sgemm_nn_range_scalar;
    g_active = &g_scalar;
    g_cpu_name = "scalar";
    return 1;
}

static const e2_i8_backend_def k_i8_backends[] = {
#if defined(__x86_64__) || defined(_M_X64)
    { "avx512", 3, cpu_has_avx512vnni, install_i8_avx512vnni, E2_BK_CPU_AVX512 },
    { "vnni", 2, cpu_has_avxvnni, install_i8_vnni, E2_BK_CPU_VNNI },
    { "avx2", 1, cpu_has_avx2, install_i8_avx2, E2_BK_CPU_AVX2 },
#endif
#if defined(__aarch64__)
    { "neon", 1, cpu_has_neon_dotprod, install_i8_neon, E2_BK_CPU_NEON },
#endif
    { "scalar", 0, NULL, install_i8_scalar, E2_BK_CPU_SCALAR },
};

static const e2_f32_backend_def k_f32_backends[] = {
#if defined(__x86_64__) || defined(_M_X64)
    { "avx512", 3, cpu_has_avx512, install_f32_avx512 },
    { "avx2", 1, cpu_has_avx2, install_f32_avx2 },
#endif
#if defined(__aarch64__)
    { "neon", 1, NULL, install_f32_neon }, /* no probe: ASIMD is in the baseline */
#endif
    { "scalar", 0, NULL, install_f32_scalar },
};

static const int k_i8_count = (int)(sizeof k_i8_backends / sizeof k_i8_backends[0]);
static const int k_f32_count = (int)(sizeof k_f32_backends / sizeof k_f32_backends[0]);

static void reset_i8_pointers(void) {
    e2_i8_gemm = NULL;
    e2_i8_quantize = NULL;
    e2_i8_attn_gemm = NULL;
    e2_softmax_quant_i8 = NULL;
}

/* Install a named backend and make the fp32 vtable follow the same tier, so a forced "avx2" behaves
 * like an AVX2-only machine rather than an AVX-512 machine with slow int8. Only a tier this CPU can
 * execute is accepted: the switch may downgrade, never invent an ISA (that is an illegal instruction,
 * not a slow run). */
static int apply_backend(const char *name) {
    const e2_i8_backend_def *row = NULL;
    for (int i = 0; i < k_i8_count; i++)
        if (strcmp(k_i8_backends[i].name, name) == 0) row = &k_i8_backends[i];
    if (!row) {
        fprintf(stderr, "e2: E2_CPU_BACKEND=%s is not a known backend\n", name);
        return -1;
    }
    if (row->probe && !row->probe()) {
        fprintf(stderr, "e2: E2_CPU_BACKEND=%s is not supported by this CPU\n", name);
        return -1;
    }
    reset_i8_pointers(); /* a downgrade must not leave the previous backend's kernels installed */
    row->install();
    e2_dispatch_set_cpu_backend(row->bk);
    for (int i = 0; i < k_f32_count; i++) {
        if (k_f32_backends[i].tier > row->tier) continue;
        if (k_f32_backends[i].probe && !k_f32_backends[i].probe()) continue;
        if (k_f32_backends[i].install()) break;
    }
    return 0;
}

int e2_cpu_backend_count(void) { return k_i8_count; }

const char *e2_cpu_backend_name(int i) {
    if (i < 0 || i >= k_i8_count) return NULL;
    if (k_i8_backends[i].probe && !k_i8_backends[i].probe()) return NULL; /* this host cannot run it */
    return k_i8_backends[i].name;
}

void e2_cpu_init(void) {
    /* The attention overrides are environment, not CPU: read them once here so no tower reads getenv
     * per layer (DISPATCH.md phase 3). e2_attn_override() flips them afterwards for an A/B or a test. */
    e2_attn_env_reload();

    reset_i8_pointers(); /* e2_cpu_init is idempotent: a second call re-selects from a blank slate */
    for (int i = 0; i < k_i8_count; i++) {
        if (k_i8_backends[i].probe && !k_i8_backends[i].probe()) continue;
        k_i8_backends[i].install();
        e2_dispatch_set_cpu_backend(k_i8_backends[i].bk);
        break;
    }
    for (int i = 0; i < k_f32_count; i++) {
        if (k_f32_backends[i].probe && !k_f32_backends[i].probe()) continue;
        if (k_f32_backends[i].install()) break;
    }

    /* E2_CPU_BACKEND downgrades the selection for an A/B; an unknown or unsupported name is reported
     * and ignored, leaving the best backend in place. */
    const char *forced = getenv("E2_CPU_BACKEND");
    if (forced && forced[0]) (void)apply_backend(forced);
}

const e2_kernels *e2_k(void) { return g_active; }
const char *e2_cpu_name(void) { return g_cpu_name; }
