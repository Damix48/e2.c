/* e2_gpu_int.h - state shared by the Level Zero backend and its op layer.
 *
 * Two files implement the device path:
 *   e2_gpu.c      device, modules, weight cache, packed-int8 GEMM
 *   e2_gpu_ops.c  elementwise/norm/attention kernels and the command-list
 *                 batching that keeps a whole forward to one submission
 *
 * Everything here is internal; the CPU path in e2_gemm.c/e2_ops.c remains the
 * reference implementation and the fallback.
 */
#ifndef E2_GPU_INT_H
#define E2_GPU_INT_H

#include "e2_gpu.h"
#include "e2_gpu_l0.h"

/* One resident packed matrix.  Keyed by the source pointer: the model file is
 * mmap'd and never moves, so the address is a stable id. */
typedef struct {
    const int8_t *q_src;
    int n, k, blk, ntiles, nblk;
    size_t tstride;
    int8_t *q;          /* packed tiles, CPU kernel layout */
    float *scale;       /* n */
    int32_t *wsum;      /* [n][nblk] */
    int8_t *pq;         /* matrix-engine layout, n * k bytes */
    int32_t *wsumt;     /* n, sum over all k (per-row-scale GEMM) */
    int32_t *wsumk;     /* [n][nblk] (block-scale GEMM, same as wsum) */
} e2_gpu_wc;

typedef struct {
    e2_gpu_wc *tab;
    int cap, used;
} e2_gpu_wct;

/* f32/i32 model tensor copied into shared USM on first use (norm weights, rope
 * tables, position tables).  Keyed by the CPU pointer. */
typedef struct {
    const void *src;
    void *usm;
    size_t bytes;
} e2_gpu_mir;

typedef struct e2_gpu {
    ze_driver_handle_t drv;
    ze_device_handle_t dev;
    ze_context_handle_t ctx;
    ze_command_queue_handle_t queue;
    ze_command_list_handle_t imm;   /* immediate list, one-shot fallbacks */
    ze_command_list_handle_t list;  /* deferred list: a whole forward */
    ze_module_handle_t module, dmodule, omodule;
    ze_kernel_handle_t kern;        /* plain (non-DPAS) GEMM */
    ze_kernel_handle_t k_pack_u8, k_pack_wq, k_gemm, k_gemm_blk;
    ze_kernel_handle_t k_rmsnorm, k_norm_rope, k_rope_text, k_rope_vision;
    ze_kernel_handle_t k_attn, k_attn_part, k_attn_comb, k_quant_u8;
    ze_kernel_handle_t k_gelu_mul_q, k_gelu_tanh, k_scale;
    ze_kernel_handle_t k_mul_slice, k_pos_add, k_softmax;
    ze_event_pool_handle_t epool;
    ze_event_handle_t evt;
    int ready;
    int defer;                      /* command list open for batching */
    int ndisp;                      /* kernels appended to the open list */
    int use_dpas;                   /* matrix-engine GEMM available */
    int use_blk;                    /* per-128-block activation scales */
    int barrier;                    /* insert a barrier after each kernel */
    unsigned skip;                  /* E2_GPU_SKIP: op groups kept on the CPU */
    int no_dev;                     /* tower that must stay on the CPU (audio) */
    /* E2_GPU_TS=1: a timestamp after every kernel, so the CPU-side profile can
     * attribute device time per kernel instead of guessing. */
    int ts_on;
    uint64_t *ts;
    size_t ts_cap;
    int ts_n;
    double ts_res;                  /* ns per tick */
    const char *ts_name[2048];

    e2_gpu_wct wct;
    e2_gpu_mir *mir;
    int mir_cap, mir_used, mir_shift;

    /* Packed activations live as long as the forward: a GEMM in the queued
     * list may not have run yet when the next one is packed, so they cannot
     * share one buffer (sharing one was the "nondeterministic output" bug).
     * shortcut: bump-allocated per forward, so peak memory is
     * sum(m*k) over the forward -- ~230 MB at 2048 tokens. Upgrade with a small
     * event-checked ring of slots when that matters. */
    e2_arena act;
} e2_gpu;

/* One packed activation row block (f32 -> u8 -> matrix-engine layout). */
typedef struct {
    uint8_t *pack;      /* [m/8][k/32][8][32] u8 */
    float *scale;       /* [m][k/128] */
} e2_act;

/* Append a kernel and a barrier: an in-order list orders execution but does
 * not promise that one kernel's writes are visible to the next. */
int  e2_gpu_launch(e2_gpu *g, ze_kernel_handle_t k, uint32_t groups);
int  e2_gpu_launch_named(e2_gpu *g, ze_kernel_handle_t k, uint32_t groups,
                         const char *name);
/* Per-kernel device time from the E2_GPU_TS timestamps. */
void e2_gpu_ts_report(e2_gpu *g);
/* Open the forward's command list if it is not already open. */
int  e2_gpu_enq_open(e2_gpu *g);
void *e2_usm_alloc(e2_gpu *g, size_t n);
void *e2_usm_realloc(e2_gpu *g, void **p, size_t *cap, size_t need);
e2_gpu_wc *e2_gpu_weight(e2_gpu *g, const e2_qmat *w);
const void *e2_gpu_mirror(e2_gpu *g, const void *p, size_t bytes);

/* Enqueue y[m][n] = x @ w^T (quantising x on the device).  0 = enqueued. */
int e2_gpu_enq_linear(e2_gpu *g, float *y, const float *x, int m, const e2_linear *l);
/* Same, for an activation already quantised to u8 rows + block scales. */
int e2_gpu_enq_linear_pq(e2_gpu *g, float *y, const uint8_t *xq, const float *xs,
                         int m, const e2_linear *l);
/* Several linears on one activation block: pack once, launch each. */
int e2_gpu_enq_linear_list(e2_gpu *g, const float *x, int m,
                           const e2_linear *const *ls, float *const *ys, int n);
/* Pack / launch halves, for callers that share the activation. */
int e2_gpu_pack_act(e2_gpu *g, const float *x, int m, int k, e2_act *a);
int e2_gpu_pack_act_u8(e2_gpu *g, const uint8_t *q8, int m, int k, e2_act *a);
int e2_gpu_enq_act(e2_gpu *g, float *y, int m, const e2_linear *l, const e2_act *a);

#endif /* E2_GPU_INT_H */
