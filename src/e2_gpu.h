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

/* ------------------------------------------------------------------ */
/* device-resident op layer (e2_gpu_ops.c)                            */
/*                                                                    */
/* Each of these *appends* a kernel to the open command list and      */
/* returns 0, or returns nonzero after flushing so the caller can run */
/* the CPU implementation.  A whole forward is one submission.        */
/* ------------------------------------------------------------------ */
int  e2_gpu_on(const e2_ctx *ctx);
/* Keep a tower on the CPU: its code still reads activations directly. */
void e2_gpu_no_dev(e2_ctx *ctx, int on);
/* Scratch the CPU allocates but a kernel also reads must be shared USM. */
void *e2_dev_alloc(e2_ctx *ctx, size_t n);
void  e2_dev_free(e2_ctx *ctx, void *p);
void *e2_arena_usm_alloc(void *opaque, size_t n);
void  e2_arena_usm_free(void *opaque, void *p);
int  e2_gpu_flush(e2_ctx *ctx);

int e2_gpu_op_rmsnorm(e2_ctx *ctx, float *out, const float *x, const float *w,
                      const float *src, int rows, int dim, float eps, float pre,
                      int mode);
int e2_gpu_op_norm_rope(e2_ctx *ctx, float *x, const float *w, int rows, int dim,
                        float eps, const e2_rope_tab *tb, int per_pos,
                        const int *seq_off, int nseq);
int e2_gpu_op_rope_text(e2_ctx *ctx, float *q, float *k, int t, int heads,
                        int kv_heads, int head_dim, int pos_offset, float theta);
int e2_gpu_op_rope_vision(e2_ctx *ctx, float *q, float *k, int t, int heads,
                          int head_dim, const int *xy, float theta);
int e2_gpu_op_attn(e2_ctx *ctx, float *out, const float *q, const float *k,
                   const float *v, int t, int heads, int kv_heads, int head_dim,
                   int window, int nseq, const int *seq_off);
int e2_gpu_op_gelu_mul_q(e2_ctx *ctx, const float *x, const float *up, int rows,
                         int k, uint8_t *q, float *sc, int blk);
int e2_gpu_op_gelu_tanh(e2_ctx *ctx, float *x, int n);
int e2_gpu_op_scale(e2_ctx *ctx, float *x, int n, float s);
int e2_gpu_op_mul_slice(e2_ctx *ctx, float *h, const float *ple, int n, int L,
                        int P, int li);
int e2_gpu_op_pos_add(e2_ctx *ctx, float *x, const int *xy, const float *tab,
                      int P, int H, long stride);
int e2_gpu_op_linear(e2_ctx *ctx, float *y, const float *x, int m, const e2_linear *l);
int e2_gpu_op_linear_pq(e2_ctx *ctx, float *y, const uint8_t *xq, const float *xs,
                        int m, const e2_linear *l);
int e2_gpu_op_linear_multi(e2_ctx *ctx, const float *x, int m,
                           const e2_linear *const *ls, float *const *ys, int n);

#ifdef __cplusplus
}
#endif

#endif /* E2_GPU_H */
