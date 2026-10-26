/* kernels.c — scalar reference backend + backend dispatch. */
#include "int8.h"
#include "kernels.h"

#include <math.h>
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

static void scalar_silu(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] *= 1.0f / (1.0f + expf(-x[i]));
}
static void scalar_glu(float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] *= 1.0f / (1.0f + expf(-b[i]));
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
    .silu = scalar_silu,
    .glu = scalar_glu,
    .clamp = scalar_clamp,
};

/* ---------------------------------------------------------------- scalar f32 GEMM */
void e2_sgemm_nn_range_scalar(float *C, int ldc, const float *A, int lda, const float *B, int ldb, int M,
                              int N, int K, int m0, int m1) {
    (void)M;
    for (int m = m0; m < m1; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += A[(size_t)m * lda + k] * B[(size_t)k * ldb + n];
            C[(size_t)m * ldc + n] = acc;
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

e2_i8_gemm_fn e2_i8_gemm = NULL;
void (*e2_sgemm_nn_range)(float *, int, const float *, int, const float *, int, int, int, int, int, int) =
    e2_sgemm_nn_range_scalar;
extern void e2_sgemm_nn_range_avx2(float *, int, const float *, int, const float *, int, int, int, int, int, int);
extern void e2_sgemm_nn_range_avx512(float *, int, const float *, int, const float *, int, int, int, int, int, int);
void (*e2_i8_quantize)(int8_t *, float *, const float *, int, int, float, float) = NULL;
e2_i8_attn_fn e2_i8_attn_gemm = NULL;
e2_sq_fn e2_softmax_quant_i8 = NULL;

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

#if defined(__x86_64__) || defined(_M_X64)
static int cpu_has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
#else
static int cpu_has_avx2(void) { return 0; }
#endif

void e2_cpu_init(void) {
    /* int8 kernels are picked independently of the fp32 ones: the 512-bit path wants AVX512-VNNI,
     * the 256-bit one either AVX-VNNI or (EVEX) AVX512-VNNI, and AVX2 is the last SIMD fallback. */
    if (cpu_has_avx512vnni()) {
        e2_i8_gemm = e2_i8_gemm_avx512;
        e2_i8_quantize = e2_quantize_blocks_avx512;
        e2_i8_attn_gemm = e2_i8_attn_gemm_avx512;
        e2_softmax_quant_i8 = e2_softmax_quant_i8_avx512;
    } else if (cpu_has_avxvnni()) {
        e2_i8_gemm = e2_i8_gemm_vnni_range;
        e2_i8_quantize = e2_quantize_blocks_vnni;
        e2_i8_attn_gemm = e2_i8_attn_gemm_vnni;
        e2_softmax_quant_i8 = e2_softmax_quant_i8_avx2;
    } else if (cpu_has_avx2()) {
        e2_i8_gemm = e2_i8_gemm_avx2;
        e2_i8_quantize = e2_quantize_blocks_avx2;
        e2_softmax_quant_i8 = e2_softmax_quant_i8_avx2; /* no maddubs attention kernel yet */
    } else {
        e2_i8_gemm = e2_i8_gemm_scalar;
        e2_i8_quantize = NULL; /* scalar model path is unused; text.c falls back to bf16 GEMMs */
    }
    if (cpu_has_avx512()) {
        const e2_kernels *k = e2_avx512_kernels();
        if (k) {
            e2_sgemm_nn_range = e2_sgemm_nn_range_avx512;
            g_active = k;
            g_cpu_name = k->name;
            return;
        }
    }
    if (cpu_has_avx2()) {
        const e2_kernels *k = e2_avx2_kernels();
        if (k) {
            e2_sgemm_nn_range = e2_sgemm_nn_range_avx2;
            g_active = k;
            g_cpu_name = k->name;
            return;
        }
    }
    g_active = &g_scalar;
    g_cpu_name = "scalar";
}

const e2_kernels *e2_k(void) { return g_active; }
const char *e2_cpu_name(void) { return g_cpu_name; }
