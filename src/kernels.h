/* kernels.h — numeric kernels. Scalar reference versions live in kernels.c; SIMD backends live in
 * kernels_avx2.c and kernels_avx512.c (plus the int8 backends in kernels_i8_*.c) and are selected
 * once at startup by e2_cpu_init().
 *
 * Layout notes (chosen for the memory access patterns this model actually has):
 *   - All weights are row-major [N, K] (the safetensors layout of nn.Linear, torch stores [out,in]).
 *   - Activations are row-major [M, K] with M = tokens (the sequence), K = hidden.
 *   - gemm_nt computes y[M,N] = x[M,K] @ W[N,K]^T walking both operands along K.
 *   - bf16 weights are consumed in place (2 bytes/weight, no conversion copy): the inner loops load
 *     bf16 and convert on the fly.
 */
#ifndef E2_KERNELS_H
#define E2_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#include "int8.h"

typedef enum { E2_DT_F32 = 0, E2_DT_BF16 = 1 } e2_dtype;

typedef struct {
    const char *name;
    /* y[M,N] = x[M,K] @ w[N,K]^T ; w is bf16 or f32 */
    void (*gemm_nt)(float *y, const float *x, const void *w, int M, int N, int K, e2_dtype wdtype);
    /* y[N] = w[N,K] @ x[K] */
    void (*gemv)(float *y, const void *w, const float *x, int N, int K, e2_dtype wdtype);
    /* in-place RMSNorm over the last dim: x[M,D] *= 1/sqrt(mean(x^2)+eps), then *gamma (may be NULL) */
    void (*rmsnorm)(float *x, int M, int D, const float *gamma, float eps);
    /* in-place tanh-approximated GELU */
    void (*gelu_tanh)(float *x, size_t n);
    /* in-place softmax over n elements */
    void (*softmax)(float *x, int n);
    float (*dot_f32)(const float *a, const float *b, size_t n);
    /* y[i] += a * x[i]; the attention value accumulation is the only consumer (it used to be a
     * scalar inner loop in text.c, which no backend could vectorize because of -fno-tree-vectorize) */
    void (*axpy)(float *y, const float *x, float a, int n);
    /* Elementwise map ops. The text tower used to run these as plain loops in text.c, which is not
     * compiled with -mavx2 (the engine has to build for any x86-64), so the compiler emitted scalar
     * SSE: they are ~11 % of a document and ~8 % of a 15-token query. Same dispatch as everything
     * else, so the scalar fallback still covers non-AVX2 machines. */
    void (*mul)(float *a, const float *b, size_t n);
    void (*add)(float *a, const float *b, size_t n);
    void (*scale)(float *a, float s, size_t n);
    void (*mul_rows)(float *a, const float *b, int rows, int D, int stride);
    /* x *= sigmoid(x) and a *= sigmoid(b): the audio tower's activation points (SiLU in the
     * feed-forwards, the GLU gate in the light conv). Both use the same exp the softmax does. */
    void (*silu)(float *x, size_t n);
    void (*glu)(float *a, const float *b, size_t n);
    /* x[i] = min(max(x[i], lo), hi): the audio tower's clipped-linear output saturation, over every
     * element of a projection's output matrix */
    void (*clamp)(float *x, float lo, float hi, size_t n);
    /* res := x (pre-norm), x := x * inv * gamma, one pass over the row (the residual copy and the
     * normalisation share their reads of x, which is what makes this worth fusing) */
    void (*norm_split)(float *x, float *res, int rows, int D, const float *gamma, float eps);
    /* out := norm(src) * gamma + res */
    void (*norm_add)(float *out, const float *src, const float *res, int rows, int D,
                     const float *gamma, float eps);
    /* rotate_half RoPE for rows [r0,r1) of a [rows, head_dim] tensor; row r uses the table entry for
     * position r/ngroups (q and k are stored head-fastest, so the position is a row quotient).
     * `block` is the size of one rotation block: the text tower rotates the whole head
     * (block == head_dim, pair i with i+head_dim/2), the vision tower rotates each half separately
     * (block == head_dim/2, the reference's axial split into x and y channel groups). */
    void (*rope_rows)(float *x, int r0, int r1, int ngroups, int head_dim, int block, const float *cs,
                      const float *sn);
} e2_kernels;

/* int8 paths, selected by the same runtime detection (VNNI > AVX2 > scalar). */
extern e2_i8_gemm_fn e2_i8_gemm;
/* per-block symmetric int8 quantization; lo/hi saturate the input first (the audio tower's clipped
 * linears), so the clamp costs no extra pass. Pass -INFINITY/INFINITY for none. */
extern void (*e2_i8_quantize)(int8_t *out, float *scales, const float *x, int M, int K, float lo, float hi);

/* C[M,N] = A[M,K] @ B[K,N], row-major with strides, rows [m0,m1) only.
 * The attention logits and context products: blocked so the 64x64 A/B panels stay in L1 while an 8x8
 * register tile walks them. Backends: AVX2/FMA, scalar. */
extern void (*e2_sgemm_nn_range)(float *C, int ldc, const float *A, int lda, const float *B, int ldb,
                                 int M, int N, int K, int m0, int m1);
/* The scalar reference every other sgemm is checked against (also the fallback). */
void e2_sgemm_nn_range_scalar(float *C, int ldc, const float *A, int lda, const float *B, int ldb,
                              int M, int N, int K, int m0, int m1);

/* The scalar SiLU/GLU. In activations.c, which is compiled without -ffast-math: the reciprocal must
 * be a real division, or the vectorised and scalar halves of the same loop disagree by 1e-4. */
void e2_silu_scalar(float *x, size_t n);
void e2_glu_scalar(float *a, const float *b, size_t n);

/* Select the best backend for this CPU (AVX2/FMA if available, else scalar). Call once before use.
 * E2_CPU_BACKEND=<name> downgrades the choice for an A/B; only a backend this CPU can execute is
 * accepted, so the switch can measure the AVX2 kernels on a VNNI machine but never invent an ISA. */
void e2_cpu_init(void);
const e2_kernels *e2_k(void);
const char *e2_cpu_name(void);

/* The named backends this build carries, in preference order: e2_cpu_backend_name(i) returns NULL for
 * one this host cannot execute, so a bench can iterate and skip. The names are the E2_CPU_BACKEND
 * values ("scalar", "avx2", "vnni", "avx512", "neon", whichever apply to the architecture). */
int e2_cpu_backend_count(void);
const char *e2_cpu_backend_name(int i);

/* Shared helpers (backend-independent). */
void e2_bf16_to_f32(float *dst, const uint16_t *src, size_t n);
float e2_dot_f32(const float *a, const float *b, size_t n);
void e2_l2norm_f32(float *x, int n);
void e2_meanpool_f32(float *out, const float *x, int M, int D, const uint8_t *valid);

/* bf16 scalar conversions, reused by the SIMD backends for tails. */
static inline float e2_bf16f(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

#endif /* E2_KERNELS_H */
