/* gpu.h — internal state of the GPU backend (the public entry points are in include/e2gpu.h). */
#ifndef E2_GPU_INTERNAL_H
#define E2_GPU_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "../include/e2.h"
#include "../include/e2gpu.h"
#include "l0.h"

/* must match the defines in kernels/e2.cl */
#define E2_NM 16
/* Audio M sub-groups sharing one weight tile (k_audio_dpas4): the work group covers
 * E2_DM * E2_AMG rows. A 0.75 s clip is 19 soft tokens, so at 4 the kernel computed a 64-row tile
 * for 19 real rows - 3.4x the MACs. 2 covers them with one 32-row tile (1.7x). */
#define E2_AMG 2
#include "e2_tile.h"

/* Region of the activation buffer, in units of floats. The buffer is sized for E2_GPU_MAX_ROWS once,
 * so every offset is a compile-time constant and a recorded graph stays valid for any shape that fits
 * - which is what lets one command list be replayed instead of rebuilt per call. */
enum {
    A_X, A_RES, A_Q, A_V, A_CTX, A_GATE, A_UP, A_TMP, A_PLE, A_EMB, A_K2, A_V2,
    A_QAX, A_QAXS, /* qa bytes + scales for x, K=512, nb=4 */
    A_QAC, A_QACS, /* ctx, K=2048, nb=16 */
    A_QAM, A_QAMS, /* mlp intermediate, K=2048, nb=16 */
    A_QAP, A_QAPS, /* ple gate, K=512, nb=4 */
    /* ---- vision tower (a separate recorded graph; the regions below are the only memory the two
     * towers do not share, along with the pool/soft-token buffers at the end). */
    A_VX, A_VRES, A_VTMP, A_VQ, A_VK, A_VV, A_VCTX, A_VGATE, A_VUP,
    A_VQX, A_VQXS,           /* quantized x / head input, K=768, nb=6 */
    A_VQH, A_VQHS,           /* quantized attention context, K=768, nb=6 */
    A_VQM, A_VQMS,           /* quantized mlp intermediate, K=3072, nb=24 */
    A_VPP, A_VPPS,           /* quantized pooled rows before the 768->512 projection */
    A_VQPK, A_VQPKS,         /* attention: Q int8 [P*12,64] + per-row scales */
    A_VKPK, A_VKSC, A_VKBSQ, /* attention: K packed per head, rows = keys, K = 64 */
    A_VVPK, A_VVSC, A_VVBSQ, /* attention: V^T packed per head, rows = 64, K = keys */
    A_VSCR, A_VPB, A_VPBS,   /* attention: (unused score scratch) + the int8 P + the per-32-key scales */
    A_VSBM, A_VSBS,          /* attention: per-32-key block exp reference (max) and exp sum */
    A_VPOOL,                 /* pooler output, [groups, 768] */
    A_VSOFT, A_VEMBIN,       /* soft tokens out / text input embeddings in */
    /* ---- audio tower. Its graph is recorded separately; the Conformer's 4096-wide intermediates
     * are the largest activations here (12.3 MB at 750 soft tokens), and the int8 scratch is shared
     * by every projection in the tower because a command list serialises the uses. */
    A_AX, A_ARES, A_ATMP, A_AFF, A_ALC, A_AGLU, A_AQ, A_AK, A_AV, A_AATT,
    A_AQA, A_AQAS,           /* shared quantized activation + per-128 scales (K up to 4096) */
    A_ASOFT, A_ARELK,        /* [S,512] soft tokens / twelve cached [13,1024] relative-key blocks */
    A_NREGION
};

#define E2_GPU_MAX_ROWS 2048
#define E2_GPU_MAX_NSL 8
/* Vision shapes: the patch count padded to the two things that need it - E2_DM for the GEMM's
 * unguarded activation reads, and 128 for the key axis of the attention quantization. */
#define E2_GPU_VP (((E2_V_MAX_PATCHES + E2_DM - 1) / E2_DM) * E2_DM)
#define E2_GPU_VK (((E2_V_MAX_PATCHES + 127) / 128) * 128)
#define E2_GPU_VGROUPS 512 /* pool groups at most (2520/9 = 280; the grid is bounded by the pad) */

typedef struct {
    size_t off;  /* bytes */
    size_t bytes;
} e2_region;

typedef struct {
    int rows, seq_len, from_embeds;
    e2_graph *gr;
    int kernels;
} e2_slot;

struct e2_gpu {
    e2_l0 *l0;
    e2_model *m; /* borrowed */
    void *wbuf;  /* quantized weights, norm gammas, rope tables */
    void *acts;  /* all activations */
    void *part;  /* k-split partials of the short-sequence GEMM */
    void *toktab;
    void *desc;  /* tensor descriptors (device, static) */
    void *ids;   /* host-visible token ids */
    void *valid; /* host-visible row validity mask */
    void *embout;/* host-visible output vectors */
    e2_region reg[A_NREGION];
    struct { int n[7]; } noff[E2_MAX_LAYERS]; /* byte offsets of the per-layer norm gammas */
    int final_norm_off, ple_proj_norm_off, cos_off[2], sin_off[2];
    /* vision tower: per-layer norm gammas (6), the bf16 position table, the axial rope frequencies,
     * the per-call upload buffers and the pooler's CSR lists. */
    struct { int n[6]; } vnoff[E2_V_LAYERS];
    int vpostab_off, vropeinv_off;
    int vdesc_base; /* first descriptor index of the vision tensors */
    void *vpix, *vpos, *vorder, *vrope, *vgstart, *vgpatch, *vpool_ng;
    int vgroups;
    /* audio tower: 135 descriptors (11 per layer + 3 global), the nine norm gammas and the per-dim
     * query scale per layer, the transposed depthwise kernels, the cached relative keys, and the
     * output-projection bias. */
    int adesc_base;
    struct { int n[9]; } anoff[E2_A_LAYERS];
    int aqmult_off[E2_A_LAYERS], alc_conv_off[E2_A_LAYERS], arelk_off[E2_A_LAYERS];
    int aout_bias_off;
    void *axin, *asoft; /* host-visible encoder input / soft-token output */
    double vlast_ms;
    long vkernels_last;
    int desc_tensors;
    e2_slot slots[16];
    int nslots;
    int max_rows;
    int vlayer_limit;     /* E2_GPU_VLAYERS: diagnostic truncation of the vision graph */
    int gemm_dp4a;        /* E2_GPU_DP4A=1: use the dp4a tiled GEMM instead of the XMX one */
    int vattn_old;        /* E2_GPU_VATTN_OLD=1: vision attention the pre-fusion way (A/B) */
    int layer_limit;      /* E2_GPU_LAYERS: diagnostic truncation of the graph */
    int only;             /* E2_GPU_ONLY=1: record only the embedding gather */
    int limit;            /* E2_GPU_LIMIT: stop recording after this many kernels */
    const char *dump_dir; /* E2_GPU_DUMP: write the activation regions after a call */
    int dump_call;        /* E2_GPU_DUMP_CALL: which call to dump (default 1) */
    size_t wbuf_bytes, acts_bytes, part_bytes;
    char err[1024];
    /* per-call statistics */
    long calls, kernels, kernels_last;
    double last_ms;
};

/* offsets in the descriptor array: 9 per layer (q,k,v,o,gate,up,down,ple_gate,ple_proj) + 2 globals */
#define E2_DS_PER_LAYER 9
/* 16 int32 per tensor: the ten fields the kernels read, an output-clamp flag, its two bounds, and
 * three spare. The stride is a kernel argument only through the pointer arithmetic here, so growing
 * it is source-compatible with the existing projection kernels. */
#define E2_DS_STRIDE 16
#define E2_DS_Q 0
#define E2_DS_K 1
#define E2_DS_V 2
#define E2_DS_O 3
#define E2_DS_GATE 4
#define E2_DS_UP 5
#define E2_DS_DOWN 6
#define E2_DS_PLE_GATE 7
#define E2_DS_PLE_PROJ 8
#define E2_DS_CLAMP 10
#define E2_DS_LO 11
#define E2_DS_HI 12

/* audio: 11 projections per layer, then in_proj / out_proj / embed_audio */
#define E2_ADS_PER_LAYER 11
#define E2_ADS_Q 0
#define E2_ADS_K 1
#define E2_ADS_V 2
#define E2_ADS_POST 3
#define E2_ADS_FF1A 4
#define E2_ADS_FF1B 5
#define E2_ADS_FF2A 6
#define E2_ADS_FF2B 7
#define E2_ADS_LC_START 8
#define E2_ADS_LC_END 9
#define E2_ADS_RELK 10

#endif /* E2_GPU_INTERNAL_H */
