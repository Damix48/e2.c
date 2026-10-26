/* model.h — internal model layout: bf16 pointers into the mmap'd checkpoint + small f32 copies of
 * the norm scalars (they are read once per token per layer, so converting them once is free). */
#ifndef E2_MODEL_H
#define E2_MODEL_H

#include "../include/e2.h"
#include "threads.h"
#include "int8.h"
#include "kernels.h"
#include "st.h"

#define E2_MAX_LAYERS 24
#define E2_NUM_HEADS 4
#define E2_EMBED_DIM 768
#define E2_VOCAB 262144
#define E2_MAX_SEQ 2048
#define E2_MAX_THREADS 64

/* Vision tower (config.json -> vision_config, tensor shapes checked at load) */
#define E2_V_LAYERS 16
#define E2_V_HEADS 12
#define E2_V_HEAD_DIM 64
#define E2_V_HIDDEN 768
#define E2_V_INTER 3072
#define E2_V_PATCH 768   /* patch_size 16, 16x16x3 */
#define E2_V_POOL 3      /* pooling_kernel_size */
#define E2_V_MAX_PATCHES 2520
#define E2_V_POS 10240

/* Query rows per work item in the blocked attention, and the rows in one per-worker score block.
 * The reference materialises the whole (P,P) logits matrix per head - 21 MB at P=2304, which is
 * written, read by the softmax and read again by the context product. Blocking it keeps that traffic
 * in L2: measured 1.9x on the image attention (see vision.c). */
#define E2_V_ATT_BQ 32

/* Audio tower (config.json -> audio_config, a USM/Conformer: tensor shapes checked at load) */
#define E2_A_LAYERS 12
#define E2_A_HIDDEN 1024
#define E2_A_HEADS 8
#define E2_A_HEAD_DIM 128
#define E2_A_INTER 4096
#define E2_A_OUT 1536        /* output_proj_dims */
#define E2_A_CHUNK 12        /* attention_chunk_size */
#define E2_A_PAST 12         /* attention_context_left - 1: the relative window is d = 0..PAST-1 */
#define E2_A_RELPOS 13       /* ctx / 2 + 1 sinusoidal relative positions */
#define E2_A_MEL 128         /* mel bins (feature extractor feature_size) */
#define E2_A_FRAME 320       /* 20 ms at 16 kHz */
#define E2_A_HOP 160         /* 10 ms */
#define E2_A_FFT 512
#define E2_A_BINS 257        /* fft/2 + 1 */
#define E2_A_C0 128          /* subsampling_conv_channels */
#define E2_A_C1 32
#define E2_A_CONV_K 3
#define E2_A_LCONV_K 5
#define E2_A_MAX_SAMPLES 480000 /* the reference processor truncates at 30 s */
#define E2_A_MAX_TOK 750        /* 30 s / 40 ms per soft token */

/* tokens per GEMM pass; shared by the text and vision work-claiming GEMM regions */
#define E2_TILE_PER_THREAD 8

/* query rows per work item in the blocked attention (see text.c); also the per-worker score block */
#define E2_ATT_BQ 32

typedef struct {
    const uint16_t *q_proj, *k_proj, *v_proj, *o_proj;
    const uint16_t *gate_proj, *up_proj, *down_proj;
    const uint16_t *ple_gate, *ple_proj;
    e2_q8w q8_q, q8_k, q8_v, q8_o, q8_gate, q8_up, q8_down, q8_ple_gate, q8_ple_proj;
    float *q_norm, *k_norm, *input_ln, *post_attn_ln, *pre_ffn_ln, *post_ffn_ln, *ple_norm;
    float layer_scalar;
    int head_dim, n_kv_heads;
    int sliding;      /* 1 = sliding-window layer, 0 = full attention */
    int window;       /* sliding window size (512, inclusive) */
    int q_dim, kv_dim;
} e2_layer;

/* Per-call workspace: one allocation, fixed layout, no aliasing. */
typedef struct {
    float *x;      /* S x hidden          */
    float *res;    /* S x hidden          */
    float *ple;    /* S x 24 x ple_dim    */
    float *q;      /* S x 2048            */
    float *k;      /* S x 1024            */
    float *v;      /* S x 1024            */
    float *att;    /* S x 2048            */
    float *attn;   /* max(S, 2*window+1)  */
    float *kt;     /* [E2_NUM_HEADS * 512, S]: transposed K for the blocked attention */
    float *asblk;  /* [nthreads, E2_ATT_BQ, S]: blocked-attention scores, one block per worker */
    float *gate;   /* S x intermediate    */
    float *up;     /* S x intermediate    */
    float *ple_g;  /* S x ple_dim         */
    float *tmp;    /* S x hidden          */
    uint8_t *qa;   /* S x 2048 int8 activations, byte view of qa_words */
    float *qa_scale;
    float *emb;    /* S x 768             */
    float *qa_words;   /* ceiling(S*2048/4) words; qa is a byte view of this */
    float *valid_words; /* ceiling(S/4) words; valid is a byte view of this */
    uint8_t *valid;
    size_t bytes;
} e2_ws;

struct e2_model {
    st_file st;
    e2_dtype wdtype;

    const uint16_t *embed_tokens; /* [VOCAB, hidden] */
    const uint16_t *ple_model_proj;
    const uint16_t *embed_proj;
    e2_q8w q8_ple_model, q8_embed_proj;
    int use_int8;
    float *ple_proj_norm, *final_norm;

    e2_layer layers[E2_MAX_LAYERS];
    int n_layers, hidden, intermediate, ple_dim;
    float embed_scale, ple_scale, rms_eps;
    float rope_theta_full, rope_theta_sliding;
    float *rope_cos[2], *rope_sin[2]; /* indexed by sliding(0)/full(1) */

    void *vision;      /* vision.c */
    void *audio;       /* audio.c */
    e2_ws ws;
};

typedef struct {
    const uint16_t *q_proj, *k_proj, *v_proj, *o_proj;
    const uint16_t *gate_proj, *up_proj, *down_proj;
    e2_q8w q8_q, q8_k, q8_v, q8_o, q8_gate, q8_up, q8_down;
    float *input_ln, *post_attn_ln, *pre_ffn_ln, *post_ffn_ln, *q_norm, *k_norm;
} e2_vlayer;

/* Vision tower state. The workspace is allocated on the first image call so a text-only process
 * never pays for it. */
typedef struct {
    e2_vlayer layers[E2_V_LAYERS];
    const uint16_t *patch_proj; /* [768, 768] */
    const uint16_t *pos_table;  /* [2, 10240, 768] */
    const uint16_t *embed_proj; /* [512, 768] */
    e2_q8w q8_patch, q8_embed_proj;
    float rope_inv[E2_V_HEAD_DIM / 4]; /* axial: 100^(-2i/32), i = 0..15 */

    float *x, *res, *q, *k, *v, *att, *tmp, *gate, *up;
    float *kt;         /* transposed k for one head: [64, P] */
    float *scores;     /* attention logits block, one E2_V_ATT_BQ x P block per worker */
    /* int8 attention scratch (one head at a time; the fp32 path ignores it). Q and P are the
     * unsigned (q+128) activation operands, K and V^T the signed weight operands, packed
     * k-quad-major so one 16-key load is contiguous. */
    uint8_t *att_q8; float *att_qs;             /* Q: [P][64] + per-row scale */
    uint8_t *att_kp; float *att_ks; int32_t *att_kz; /* K: [16][Pp][4] + scale + row sum */
    uint8_t *att_vp; float *att_vs; int32_t *att_vz; /* V^T: [Pp/4][64][4] + scale + 128-key block sums */
    uint8_t *att_p8; float *att_ps;             /* P: per-worker [BQ][P] + per-128-block scales */
    float *att_q8_words, *att_kp_words, *att_vp_words, *att_p8_words, *att_kz_words, *att_vz_words;
    float *cos, *sin;  /* per-patch axial tables: [P, 64] */
    float *pool;       /* [P, 768] pooled groups, emitted in group order */
    float *emb;        /* [E2_MAX_SEQ, 512] text input embeddings for the image layout */
    uint8_t *qa;       /* int8 activations */
    float *qa_scale;
    uint8_t *seen;     /* pooler group occupancy, [max_patches] */
    float *qa_words, *order_words, *seen_words; /* byte views of qa / order / seen */
    int *order;        /* valid patch indices, gathered once per image */
    size_t bytes;
} e2_vision;

/* One of the audio tower's "clipped linears": the reference clamps the activation into a learned
 * per-tensor range on both sides of the projection (use_clipped_linears), which is also why the
 * tower tolerates int8 activations. */
typedef struct {
    const uint16_t *w;
    e2_q8w q8;
    float in_min, in_max, out_min, out_max;
} e2_alin;

typedef struct {
    e2_alin q, k, v, post;
    const uint16_t *rel_k; /* unclipped nn.Linear over the relative position encoding */
    e2_q8w q8_rel_k;
    e2_alin ff1a, ff1b;    /* feed-forward block 1: 1024 -> 4096 -> 1024 */
    e2_alin ff2a, ff2b;    /* feed-forward block 2 */
    e2_alin lc_start, lc_end;   /* light conv: GLU projection in (2048) and out (1024) */
    const uint16_t *lc_conv;    /* depthwise [1024, 1, 5], kept bf16 */
    float *lc_conv_t;           /* [5][1024]: the same kernel transposed, tap-major */
    float *n_pre_attn, *n_post_attn, *n_out;
    float *n_ff1_pre, *n_ff1_post, *n_ff2_pre, *n_ff2_post, *n_lc_pre, *n_lc_conv;
    float *qmult; /* [128] q_scale * softplus(per_dim_scale), the reference's query scaling */
} e2_alayer;

/* Audio tower state. The workspace is allocated on the first audio call, so a text-only or
 * image-only process never pays for it. */
typedef struct {
    e2_alayer layers[E2_A_LAYERS];
    const uint16_t *c0_w, *c1_w;      /* subsample convs [128,1,3,3] / [32,128,3,3] */
    float *c0_wt, *c1_wt;             /* transposed to [tap][channel] for the inner loops */
    float *c0_norm, *c1_norm;         /* LayerNorm scales (bias-free) */
    e2_alin in_proj;                  /* 1024 -> 1024 (not clipped) */
    e2_alin out_proj;                 /* 1024 -> 1536 (not clipped, has a bias) */
    const uint16_t *out_proj_b;
    e2_alin embed;                    /* 1536 -> 512 (not clipped) */
    float *pos;                       /* [E2_A_RELPOS][1024] sinusoidal relative positions */
    float *window;                    /* [E2_A_FRAME] periodic Hann */
    float *melfb;                     /* [E2_A_BINS][E2_A_MEL] */
    float *fft_cos, *fft_sin;         /* [E2_A_FFT/2] twiddles for the real FFT */
    int *fft_rev;                     /* bit-reversal permutation, [E2_A_FFT] */

    float *x, *res, *tmp, *q, *k, *v, *att, *ff, *lc, *relk;
    float *clamp8;                     /* bf16 fallback: 8 clamped rows */
    float *soft;                      /* [E2_A_MAX_TOK][512]: audio soft tokens */
    float *emb;                       /* [E2_MAX_SEQ][512]: the text-tower input block */
    uint8_t *qa;                      /* int8 activations, [E2_A_MAX_TOK][E2_A_INTER] max */
    float *qa_scale;
    float *qa_words;
    /* frontend + subsample scratch: sized by the input length, grown on demand */
    float *mel;                       /* [F][128] */
    float *c0_out, *c1_out;           /* [T/2][64][128] and [T/4][32][32] */
    size_t mel_cap, c0_cap, c1_cap;
    int ready; /* int8 weights built (lazily, so a text-only load never pays for them) */
    size_t bytes;
} e2_audio;

/* Workspace allocation for a given sequence length / image patch count. */
int e2_ws_alloc(e2_model *m, int max_seq, int max_patches);
void e2_ws_free(e2_model *m);

const uint16_t *e2_tensor_bf16(e2_model *m, const char *name, int ndim, const int64_t *dims);
/* Build the audio tower's int8 weights (lazily on the first audio call, or eagerly for the GPU
 * upload and the packed-cache writer). Returns 0 when the tower is ready. */
int e2_audio_prepare(e2_model *m);
/* one token row out of the bf16 token table, scaled by the model's embedding scale */
void e2_embed_token(e2_model *m, int id, float *dst);
/* Run the audio front end (mel -> two subsampling convs -> 1024-wide projection) and return its
 * [n_tok, 1024] output, which is the input of the twelve Conformer layers. The pointer is into the
 * audio workspace and lives until the next audio call. */
int e2_audio_frontend(e2_model *m, const float *pcm, int n_samples, const float **x, int *n_tok);
/* 1 if the tensor is available from the safetensors or from the q8cache (E2_Q8_READ) */
int e2_tensor_exists(e2_model *m, const char *name);
/* 1 when the model is being built from the packed cache (E2_Q8_READ) */
int e2_q8_is_read(void);
/* 1 when the model is being dumped to the packed cache (E2_Q8_WRITE) */
int e2_q8_is_write(void);
/* A weight pointer is usable when it is non-NULL, or when the model came from the packed cache: a
 * quantized weight has no bf16 copy there, the int8 kernels fetch the packed tensor under the same
 * name, and nothing dereferences the bf16 pointer on that path. */
static inline int e2_weight_ok(const uint16_t *p) { return p != NULL || e2_q8_is_read(); }
/* Quantizes a [N,K] bf16 weight into q, or reads it back from the cache instead of quantizing when
 * E2_Q8_READ is set, or writes the result to the cache when E2_Q8_WRITE is set. `name` is the
 * safetensors tensor name, which is also the cache key - the loader is what supplies it, so the two
 * sides cannot drift apart. */
int e2_q8w_load(e2_model *m, e2_q8w *q, const char *name, const uint16_t *w, int N, int K);
int e2_copy_bf16_f32(e2_model *m, const char *name, int64_t n, float *dst);
/* one token out of the bf16 token table, scaled (text.c; the GPU image path builds its embedding
 * block on the host, where soft tokens have no token id to gather) */
void e2_embed_token(e2_model *m, int id, float *dst);
void e2_set_error(const char *fmt, ...);

#endif /* E2_MODEL_H */
