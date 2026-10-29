/* hip.h — internal state of the AMD/HIP backend (the public entry points are in include/e2hip.h). */
#ifndef E2_HIP_INTERNAL_H
#define E2_HIP_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include <hip/hip_runtime_api.h>

#include "e2.h"
#include "e2hip.h"

/* Activation regions. Same idea as the Intel backend: one buffer sized for E2_HIP_MAX_ROWS once,
 * so every offset in a recorded graph is static and a graph stays valid for any shape that fits. */
enum {
    HA_X, HA_RES, HA_Q, HA_V, HA_CTX, HA_GATE, HA_UP, HA_TMP, HA_PLE, HA_EMB, HA_K2, HA_V2,
    HA_QAX, HA_QAXS,  /* quantized x, K=512  (4 blocks) */
    HA_QAC, HA_QACS,  /* quantized attention context, K=H*D */
    HA_QAM, HA_QAMS,  /* quantized mlp intermediate, K=2048 */
    HA_QAP, HA_QAPS,  /* quantized ple gate, K=512 */
    /* int8 attention (see e2k_attn_scores8 in kernels.hip): Q quantized per (row, head); K and
     * V^T quantized per (sequence, kv-head); scores and P materialized per (sequence, head) in
     * sliced mode, one shared slice in loop mode. */
    HA_QQ, HA_QQS, HA_KPK, HA_KSC, HA_VPK, HA_VSC, HA_SCR, HA_PB, HA_PBS,
    /* vision tower (its own recorded graph; these are the only activation regions the two towers do
     * not share, along with the pool/soft-token buffers at the end). */
    HA_VX, HA_VRES, HA_VTMP, HA_VQ, HA_VK, HA_VV, HA_VCTX, HA_VGATE, HA_VUP,
    HA_VQX, HA_VQXS,          /* quantized x / head input, K=768, nb=6 */
    HA_VQH, HA_VQHS,          /* quantized attention context, K=768, nb=6 */
    HA_VQM, HA_VQMS,          /* quantized mlp intermediate, K=3072, nb=24 */
    HA_VPP, HA_VPPS,          /* quantized pooled rows before the 768->512 projection */
    HA_VQPK, HA_VQPKS,        /* attention: Q int8 [H][Ppad][64] + per-row scales */
    HA_VKPK, HA_VKSC,         /* attention: K int8 [H][Ppad][64] + per-row scales */
    HA_VVPK, HA_VVSC,         /* attention: V^T int8 [H][64][Ppad] + per-dim scales */
    HA_VSCR,                  /* attention: fp32 scores [Ppad][Ppad], one head at a time */
    HA_VPB, HA_VPBS,          /* attention: int8 P [Ppad][Ppad] + per-128-key scales */
    HA_VPOOL,                 /* pooler output, [groups, 768] */
    HA_VSOFT, HA_VEMBIN,      /* soft tokens out / text input embeddings in */
    HA_NREGION
};

/* The int8 attention path runs for seq_len >= the HIP/text threshold in dispatch.c (shorter shapes
 * stay on the fused fp32 kernel: the pack + softmax launches do not pay for themselves there). The
 * threshold is a rule, not a constant of this backend; E2_HIP_ATTN_I8_MIN overrides it at run time. */
/* scores/P scratch cap: nseq*H*S*sup <= this many elements -> every (sequence, head) gets its own
 * slice and the products launch once; larger shapes loop (sequence, head) over one shared slice.
 * 4M elements = 16 MB fp32 scores + 4 MB int8 P, and one max-size (s,h) slice is exactly 4M. */
#define E2_HIP_A8_SLICES (4 * 1024 * 1024)
/* seq_len at which the LDS-staged attention products beat the direct-load ones on gfx1036. This is a
 * kernel-generation threshold, not a precision one, which is why it stays here and not in the table. */
#define E2_HIP_A8_LDS_MIN 512

#define E2_HIP_MAX_ROWS 2048
#define E2_HIP_MAX_NSL 4     /* k-slices of the short-sequence GEMV */
#define E2_HIP_GEMV_TM 16    /* tokens per GEMV work item (must match E2_GEMV_TM in kernels.hip) */
#define E2_HIP_MAX_SEQ E2_MAX_SEQ

/* Vision shapes: the patch count padded to 128 (the attention's quantization block), and the pool
 * group bound (2520/9 = 280; the fixed grid is what keeps the recorded graph valid for any image
 * with a given patch count). */
#define E2_HIP_VP (((E2_V_MAX_PATCHES + 127) / 128) * 128)
#define E2_HIP_VGROUPS 512

typedef struct {
    size_t off; /* bytes */
    size_t bytes;
} e2_hip_region;

typedef struct {
    int rows, seq_len, from_embeds;
    hipGraph_t graph;
    hipGraphExec_t exec;
    int kernels;
} e2_hip_slot;

struct e2_hip {
    e2_model *m; /* borrowed */
    hipStream_t stream;
    int device;
    int nograph; /* E2_HIP_NOGRAPH=1: launch every kernel directly (debugging) */

    void *wbuf;  /* quantized weights + scales (the CPU quad-interleaved layout, copied as-is) */
    void *acts;  /* all activations */
    void *part;  /* k-split partials of the short-sequence GEMV */
    void *desc;  /* tensor descriptors (device, static) */
    int32_t *hdesc; /* the same descriptors on the host (the column-grid launcher reads N) */
    void *toktab;/* the bf16 token table */

    int32_t *ids;         /* pinned host: per-call token ids */
    uint8_t *valid;       /* pinned host: per-call row validity */
    int32_t *d_ids;
    uint8_t *d_valid;
    float *d_embout;

    e2_hip_region reg[HA_NREGION];
    struct { int n[7]; } noff[E2_MAX_LAYERS]; /* byte offsets of the per-layer norm gammas */
    int final_norm_off, ple_proj_norm_off, cos_off[2], sin_off[2];
    int desc_tensors;
    /* vision tower: per-layer norm gammas (6 each), the bf16 position table, the descriptor base,
     * the per-call device uploads and the pooler's CSR lists. */
    struct { int n[6]; } vnoff[E2_V_LAYERS];
    int vpostab_off, vdesc_base, vgroups;
    void *vpix, *vpos, *vorder, *vgpatch, *vgstart, *vrope, *vpool_ng;
    double vlast_ms;
    long vkernels_last;
    int vlayer_limit; /* E2_HIP_VLAYERS: diagnostic truncation of the vision graph */
    size_t wbuf_bytes, acts_bytes;

    e2_hip_slot slots[16];
    int nslots;

    int layer_limit; /* E2_HIP_LAYERS=n: diagnostic truncation of the graph */
    int gemm_lds;    /* E2_HIP_GEMM=direct selects the old direct-load GEMM (A/B switch) */
    int attn_lds;    /* E2_HIP_ATTN=old selects the scalar per-row attention (A/B switch) */
    int attn_i8;     /* E2_HIP_ATTN=lds selects the fused fp32 blocked attention (A/B switch) */
    int attn_i8_lds; /* E2_HIP_ATTN_I8=direct selects the direct-load int8 products */
    int attn_i8_min; /* E2_HIP_ATTN_I8_MIN: seq_len threshold for the int8 attention */
    int limit;       /* E2_HIP_LIMIT=n: stop recording after n kernels */
    const char *dump_dir; /* E2_HIP_DUMP=dir: write the activation regions after a call */
    int dump_call;

    long calls, kernels, kernels_last;
    double last_ms;
    char err[1024];
};

/* forward.c */
int e2h_forward_build(e2_hip *g);
int e2h_record(e2_hip *g, int rows, int seq_len, int from_embeds, e2_hip_slot *slot);
int e2h_record_vision(e2_hip *g, int P, int ng, e2_hip_slot *slot);
int e2h_launch_vision(e2_hip *g, int P, int ng);
int e2h_launch_all(e2_hip *g, int rows, int seq_len, int from_embeds);
void e2h_dump_regions(e2_hip *g, int rows);

/* ---- kernels.hip launchers (grid geometry lives in kernels.hip so the C side stays free of it) */
void e2h_embed(hipStream_t s, float *x, const uint16_t *table, const int32_t *ids,
               const uint8_t *valid, int rows, int hidden, float scale);
void e2h_copy_rows(hipStream_t s, float *x, const float *src, int rows, int hidden);
void e2h_rowops(hipStream_t s, float *x, const float *src, float *res, int8_t *qa, float *asc,
                const float *gamma, const float *gamma2, float eps, float prescale, float scale,
                int rows, int K, int nb, int flags, const uint8_t *valid);
void e2h_gemv(hipStream_t s, const void *wbuf, const void *acts, void *part, const void *desc,
              int nsl, int rows, int nmax, int first, int count);
void e2h_gemm(hipStream_t s, const void *wbuf, const void *acts, const void *desc, int rows,
              int first, int count, int ytotal);
void e2h_gemm_lds(hipStream_t s, const void *wbuf, const void *acts, const void *desc, int rows,
                  int first, int count, int ytotal);
void e2h_norm_rope3(hipStream_t s, float *xq, const float *gq, int rq, float *xk, const float *gk,
                    int rk, float *xv, const float *gv, int rv, const float *costab,
                    const float *sintab, float eps, int hq, int hkv, int D, int seq_len);
void e2h_attn(hipStream_t s, float *ctx, const float *q, const float *k, const float *v, int8_t *qa,
              float *asc, const uint8_t *valid, int rows, int H, int D, int kv_heads, int q_stride,
              int kv_stride, int window, int seq_len);
void e2h_attn_lds(hipStream_t s, float *ctx, const float *q, const float *k, const float *v,
                  int8_t *qa, float *asc, const uint8_t *valid, int rows, int H, int D,
                  int kv_heads, int q_stride, int kv_stride, int window, int seq_len);
void e2h_attn_packk(hipStream_t s, int8_t *kpk, float *ksc, const float *k, int S, int sup,
                    int D, int KV, int kv_stride, int nseq);
void e2h_attn_packv(hipStream_t s, int8_t *vpk, float *vsc, const float *v, int S, int sup,
                    int D, int KV, int kv_stride, int nseq);
void e2h_attn_scores8(hipStream_t s, float *scr, const int8_t *qpk, const float *qsc,
                      const int8_t *kpk, const float *ksc, int S, int sup, int D, int H, int KV,
                      int q_stride, int nz);
void e2h_attn_scores8_lds(hipStream_t s, float *scr, const int8_t *qpk, const float *qsc,
                          const int8_t *kpk, const float *ksc, int S, int sup, int D, int H,
                          int KV, int q_stride, int nz);
void e2h_attn_softmax8(hipStream_t s, float *scr, int8_t *pb, float *psc, const uint8_t *valid,
                       int S, int sup, int H, int window, int nz);
void e2h_attn_ctx8(hipStream_t s, float *ctx, const int8_t *pb, const float *psc,
                   const int8_t *vpk, const float *vsc, int S, int sup, int D, int H, int KV,
                   int q_stride, int nz);
void e2h_attn_ctx8_lds(hipStream_t s, float *ctx, const int8_t *pb, const float *psc,
                       const int8_t *vpk, const float *vsc, int S, int sup, int D, int H, int KV,
                       int q_stride, int nz);
void e2h_gelu_mul_quant(hipStream_t s, float *gate, const float *up, const float *ple, int8_t *qa,
                        float *asc, int rows, int K, int nb, int stride, int ple_off);
void e2h_pool_l2(hipStream_t s, float *out, const float *emb, const uint8_t *valid, int rows,
                 int dim, int seq_len);

/* ---- vision tower kernels (kernels.hip launchers) */
void e2h_vpatch_q(hipStream_t s, int8_t *qa, float *asc, const float *pix, const int32_t *order,
                  int P, int D);
void e2h_vpos_add(hipStream_t s, float *x, const float *proj, const uint16_t *postab,
                  const int32_t *order, const int32_t *posids, int P, int D, int npos);
void e2h_vnorm_rope3(hipStream_t s, float *xq, const float *gq, float *xk, const float *gk,
                     float *xv, const float *gv, const float *costab, const float *sintab,
                     float eps, int rows, int heads);
void e2h_vattn_packq(hipStream_t s, int8_t *qpk, float *qsc, const float *q, int P, int Ppad,
                     int H);
void e2h_vattn_packk(hipStream_t s, int8_t *kpk, float *ksc, const float *k, int P, int Ppad,
                     int H);
void e2h_vattn_packv(hipStream_t s, int8_t *vpk, float *vsc, const float *v, int P, int Ppad,
                     int H);
void e2h_vattn_scores8(hipStream_t s, float *scr, const int8_t *qpk, const float *qsc,
                       const int8_t *kpk, const float *ksc, int P, int Ppad, int H, int h);
void e2h_vattn_softmax8(hipStream_t s, const float *scr, int8_t *pb, float *psc, int P, int Ppad);
void e2h_vattn_ctx8(hipStream_t s, float *ctx, const int8_t *pb, const float *psc,
                    const int8_t *vpk, const float *vsc, int P, int Ppad, int H, int h);
void e2h_vpool(hipStream_t s, float *pool, const float *x, const int32_t *gstart,
               const int32_t *gpatch, const int32_t *ngp, int D, float scale);

/* device helpers in hip.c */
int e2h_ok(e2_hip *g, hipError_t e, const char *what);
void e2h_set_error(e2_hip *g, const char *fmt, ...);
double e2h_now_ms(void);

/* E2_HIP_PROFILE=1 (with E2_HIP_NOGRAPH=1) brackets every kernel with HIP events and prints the
 * per-kernel totals when the engine is destroyed. Diagnostic only. */
extern int e2h_prof_on;
extern int e2h_prof_rec;
void e2h_prof_ev(int slot, const char *name, hipEvent_t a, hipEvent_t b);
void e2h_prof_flush(void);
void e2h_prof_report(void);

#endif /* E2_HIP_INTERNAL_H */
