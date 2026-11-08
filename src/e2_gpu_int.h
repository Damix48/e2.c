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
#include "e2_gpu_pack.h"

/* One resident packed matrix.  Keyed by the source pointer: the model file is
 * mmap'd and never moves, so the address is a stable id.  Only the matrix
 * engine layout is kept: the activation is centred int8, so the DPAS needs no
 * zero-point term and there are no weight sums to carry. */
typedef struct {
    const int8_t *q_src;
    int n, k, blk, nblk;
    float *scale;       /* n */
    int8_t *pq;         /* matrix-engine layout, n * k bytes */
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
    ze_module_handle_t dmodule, bmodule, omodule;
    ze_kernel_handle_t k_gemm, k_gemm_z, k_gemm_b2, k_mlp_gu;
    /* the same three, with the large token tile (tools/gpu/e2_xmx_big.cl) */
    ze_kernel_handle_t k_gemm_ple;
    ze_kernel_handle_t k_bgemm_z, k_bgemm_b2, k_bmlp_gu, k_bgemm_ple;
    ze_kernel_handle_t k_rmsnorm, k_rmsnorm_q, k_norm_rope, k_rope_text,
        k_rope_vision;
    ze_kernel_handle_t k_attn_pack, k_attn_softmax, k_quant_s8;
    ze_kernel_handle_t k_gelu_mul_q, k_gelu_tanh, k_scale;
    ze_kernel_handle_t k_mul_slice, k_pos_add;
    ze_event_pool_handle_t epool;
    ze_event_handle_t evt;
    int ready;
    int defer;                      /* command list open for batching */
    int ndisp;                      /* kernels appended to the open list */
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

    /* The activation a norm just quantised, so the GEMM that follows does not
     * quantise it a second time (q/k/v share one).  Keyed by the f32 pointer;
     * cleared as soon as a GEMM reads a different activation. */
    struct { const float *x; int m, k; int8_t *q; float *sc; } ac;

    /* Packed activations live as long as the forward: a GEMM in the queued
     * list may not have run yet when the next one is packed, so they cannot
     * share one buffer (sharing one was the "nondeterministic output" bug).
     * shortcut: bump-allocated per forward, so peak memory is
     * sum(m*k) over the forward -- ~230 MB at 2048 tokens. Upgrade with a small
     * event-checked ring of slots when that matters. */
    e2_arena act;
} e2_gpu;

/* Bytes of score matrix the attention keeps live at once.  Above this the
 * rectangle stops fitting in L3 and the materialised scores dominate the
 * forward (90 MB per layer at 1200 patches), so the heads are walked in groups
 * small enough to stay cached. */
#define E2_GPU_ATT_BUDGET (8u << 20)

/* Quantised activation: row-major centred int8 plus one scale per 128-wide
 * block.  No repacking - the matrix engine's B operand is a row-major tile. */
typedef struct {
    int8_t *q;          /* [m][k] */
    float *scale;       /* [m][k/128] */
} e2_act;

/* Append a kernel and a barrier: an in-order list orders execution but does
 * not promise that one kernel's writes are visible to the next. */
int  e2_gpu_launch_named(e2_gpu *g, ze_kernel_handle_t k, uint32_t groups,
                         const char *name);
int  e2_gpu_launch_3d(e2_gpu *g, ze_kernel_handle_t k, uint32_t gx, uint32_t gy,
                      uint32_t gz, const char *name);
/* Per-kernel device time from the E2_GPU_TS timestamps. */
void e2_gpu_ts_report(e2_gpu *g);
/* Open the forward's command list if it is not already open. */
int  e2_gpu_enq_open(e2_gpu *g);
void *e2_usm_alloc(e2_gpu *g, size_t n);
void *e2_usm_realloc(e2_gpu *g, void **p, size_t *cap, size_t need);
e2_gpu_wc *e2_gpu_weight(e2_gpu *g, const e2_qmat *w);
const void *e2_gpu_mirror(e2_gpu *g, const void *p, size_t bytes);

/* One launch of `nd` GEMMs that share a shape and differ by a constant stride
 * (the attention heads).  0 = enqueued. */
int e2_gpu_enq_gemm_z(e2_gpu *g, float *y, const int8_t *xq, const float *xsc,
                      const int8_t *wp, const float *wsc, int m, int k, int n,
                      int xstr, int xscstr, int ystr, long zxq, long zxsc,
                      long zwp, long zwsc, long zy, int zrep, int nd, int b2);

/* Enqueue y[m][n] = x @ w^T for a model matrix.  0 = enqueued. */
int e2_gpu_gemm_add(e2_gpu *g, float *y, const int8_t *xq, const float *xsc,
                    int m, const e2_qmat *w);

/* Enqueue y = quantise(gelu(gate(x)) * up(x)) followed by down(y), the whole
 * MLP block: two launches instead of six, and no f32 intermediate. */
int e2_gpu_enq_mlp(e2_gpu *g, float *y, const float *x, int m,
                   const e2_linear *gate, const e2_linear *up,
                   const e2_linear *down);

/* The PLE block: quantise(gelu(gate(x)) * ple[token][li][feature]) feeding the
 * per-layer projection, in one launch plus the projection. */
int e2_gpu_enq_ple(e2_gpu *g, float *y, const float *x, int m,
                   const e2_linear *gate, const e2_linear *proj,
                   const float *ple, int nl, int li);

/* Enqueue y[m][n] = x @ w^T (quantising x on the device).  0 = enqueued. */
int e2_gpu_enq_linear(e2_gpu *g, float *y, const float *x, int m, const e2_linear *l);
/* Same, for an activation already quantised to int8 rows + block scales. */
int e2_gpu_enq_linear_pq(e2_gpu *g, float *y, const int8_t *xq, const float *xs,
                         int m, const e2_linear *l);
/* Enqueue the int8 attention for one sequence group: pack K and V^T, then per
 * query head a QK^T product, the masked softmax into int8 P, and the PV
 * product.  Returns nonzero if the shape has no device path (the caller falls
 * back to the CPU int8 attention). */
int e2_gpu_enq_attn(e2_gpu *g, float *out, const float *q, const float *k,
                    const float *v, int t, int heads, int kv_heads, int head_dim,
                    int window, int nseq, const int *seq_off);
/* Quantise / launch halves, for callers that share the activation. */
int e2_gpu_pack_act(e2_gpu *g, const float *x, int m, int k, e2_act *a);
int e2_gpu_enq_act(e2_gpu *g, float *y, int m, const e2_linear *l, const e2_act *a);

#endif /* E2_GPU_INT_H */
