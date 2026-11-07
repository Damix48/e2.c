/* e2_gpu.h - Level Zero (Intel iGPU) GEMM backend. Internal, not public. */
#ifndef E2_GPU_H
#define E2_GPU_H

#include "e2_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Create the GPU device/context/queue and load the checked-in SPIR-V module.
 * Returns 0 on success, -1 if Level Zero is unavailable.  The CPU path stays
 * authoritative: callers fall back to it whenever these return nonzero. */
int  e2_gpu_init(e2_ctx *ctx);
void e2_gpu_free(e2_ctx *ctx);

/* Packed-int8 GEMM y[m][n] = xq (u8, block-scaled) x w (packed int8).
 * Same semantics as e2_gemm_range() for the full-range case nb0=0.
 * Returns 0 if the GPU produced y, nonzero to fall back to the CPU kernel. */
int e2_gpu_gemm(e2_ctx *ctx, float *y, int ns, const uint8_t *xq, const float *xs,
                int xss, int m, int k, const e2_qmat *w, int nb0, int nb1);

/* Matrix-engine (DPAS) path: y[i] = x @ W[i]^T for several matrices sharing one
 * activation row block, quantising x on the fly.  Returns 0 on success. */
int e2_gpu_linear_multi(e2_ctx *ctx, const float *x, int m,
                        const e2_linear *const *ls, float *const *ys, int n);

#ifdef __cplusplus
}
#endif

#endif /* E2_GPU_H */
