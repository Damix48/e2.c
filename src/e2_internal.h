/* e2_internal.h - shared internals. Not part of the public API. */
#ifndef E2_INTERNAL_H
#define E2_INTERNAL_H

#include "e2.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define E2_MAX_DIM 4

enum { E2_DT_F32 = 0, E2_DT_I8 = 1, E2_DT_I32 = 2 };
enum { E2_TF_PACKED = 1 };

typedef struct {
    char name[96];
    uint32_t dtype;
    uint32_t ndim;
    int64_t shape[E2_MAX_DIM];
    uint64_t offset;
    uint64_t nbytes;
    uint32_t flags;
    uint32_t _reserved;
    const void *data;
} e2_tensor;

/* Quantized matrix, rows = output features, cols = input features. */
typedef struct {
    const int8_t *q;
    const float *scale;     /* [n] */
    const int32_t *wsum;    /* [n], packed only */
    const float *bias;      /* [n] or NULL */
    int n, k;
    int packed;
    int blk;                /* activation scale block, 0 = 128 */
    size_t tstride;         /* bytes per 16-row tile, 0 = k4*64 */
    int wsmstride;          /* row stride of wsum, 0 = nblk */
} e2_qmat;

typedef struct {
    e2_qmat w;
    const float *in_min, *in_max, *out_min, *out_max; /* clipped linears */
} e2_linear;

typedef struct {
    e2_linear q, k, v, o;
    e2_linear gate, up, down;
    e2_linear ple_gate, ple_proj;
    const float *in_norm, *post_attn_norm, *pre_ff_norm, *post_ff_norm;
    const float *ple_norm;
    const float *q_norm, *k_norm;
    const float *layer_scalar;
    int head_dim, kv_heads;
    int is_full;
} e2_text_layer;

typedef struct {
    e2_linear q, k, v, o, gate, up, down;
    const float *in_norm, *post_attn_norm, *pre_ff_norm, *post_ff_norm;
    const float *q_norm, *k_norm;
} e2_vision_layer;

typedef struct {
    e2_linear q, k, v, post, rel_k;
    e2_linear ff1a, ff1b, ff2a, ff2b;
    e2_linear lc_start, lc_end;
    const float *per_dim_scale;
    const float *ff1_pre_norm, *ff1_post_norm;
    const float *ff2_pre_norm, *ff2_post_norm;
    const float *pre_attn_norm, *post_attn_norm;
    const float *lc_pre_norm, *lc_conv_norm;
    const float *norm_out;
    const float *dw_conv;   /* [hidden * conv_k] */
} e2_audio_layer;

typedef struct {
    int flags;
    int text_layers, text_hidden, text_heads, text_kv, text_head_dim,
        text_head_dim_full, text_kv_full, text_inter, text_ple,
        text_embed_dim, text_vocab, text_sliding;
    int vision_layers, vision_hidden, vision_heads, vision_head_dim,
        vision_inter, vision_patch, vision_pos_table, vision_pool;
    int audio_layers, audio_hidden, audio_heads, audio_head_dim, audio_inter,
        audio_output_dim, audio_conv_k, audio_chunk, audio_ctx_left,
        audio_ctx_right, text_full_mask, vision_soft_tokens;
    float text_eps, text_embed_scale, text_ple_scale, vision_eps, vision_rope_theta,
          vision_pool_scale, audio_eps, audio_softcap, audio_residual,
          audio_clip, audio_qscale, audio_kscale, text_rope_theta_sliding,
          text_rope_theta_full, mel_floor;
} e2_meta;

typedef struct e2_tok e2_tok;

typedef struct {
    void *map;
    size_t size;
    e2_tensor *tensors;
    int ntensors;
    e2_meta meta;

    e2_tok *tok;

    /* text */
    e2_qmat embed;                 /* row-major int8 [vocab, hidden] */
    e2_qmat ple_proj, text_proj;
    const float *ple_norm, *final_norm;
    e2_text_layer *tl;

    /* vision */
    e2_qmat patch_proj, vis_embed;
    const float *pos_table;        /* [2, pos_table, hidden] */
    e2_vision_layer *vl;

    /* audio */
    e2_qmat aud_sub0, aud_sub1, aud_in_proj, aud_out_proj, aud_embed;
    const float *aud_conv0, *aud_conv1;
    const float *aud_sub0_norm, *aud_sub1_norm;
    e2_audio_layer *al;

    /* preprocessing tables */
    const float *mel_filters;      /* [257*128] */
    const float *hann;             /* [320] */
} e2_model;

typedef struct e2_pool e2_pool;

/* Cached cos/sin table for rotary positions: [cap][half] each. */
typedef struct {
    float *c, *s;
    int cap, half;
    float theta;
} e2_rope_tab;

typedef struct e2_arena_block e2_arena_block;
typedef struct {
    e2_arena_block *head, *cur;
    size_t used, cap;
} e2_arena;

struct e2_ctx {
    e2_model m;
    e2_arena scratch;
    /* reusable GEMM scratch (main-thread only) */
    uint8_t *qbuf; size_t qcap;
    float *sbuf; size_t scap;
    float *tbuf; size_t tcap;
    float *ktbuf; size_t ktcap;   /* transposed K for attention */
    void *abuf; size_t acap;      /* int8 attention scratch */
    e2_rope_tab rope[2];
    e2_rope_tab ropev[2];
    int nthreads;
    e2_pool *pool;
    double last_ms;
    char info[256];
};

typedef void (*e2_work_fn)(void *arg, int tid, int nthreads);

/* arena */
void *e2_arena_alloc(e2_arena *a, size_t n);
void  e2_arena_reset(e2_arena *a);
void  e2_arena_free(e2_arena *a);

/* pool */
int  e2_pool_init(e2_ctx *ctx, int nthreads);
void e2_pool_free(e2_ctx *ctx);
void e2_parallel(e2_ctx *ctx, e2_work_fn fn, void *arg);
int  e2_cpu_count(void);
/* Range of n items for worker tid of nt, weighted by that core's measured
 * throughput (a hybrid barrier ends at the slowest core). */
void e2_split(int n, int tid, int nt, int *lo, int *hi);

/* model */
const e2_tensor *e2_find(const e2_model *m, const char *name);
const float *e2_f32(const e2_model *m, const char *name, const char *what);
const float *e2_f32_opt(const e2_model *m, const char *name);
e2_qmat e2_qmat_get(const e2_model *m, const char *name);
e2_linear e2_linear_get(const e2_model *m, const char *weight_name);
e2_linear e2_linear_clip_get(const e2_model *m, const char *base);
int e2_model_load(e2_ctx *ctx, const char *model_path, const char *tok_path);

/* gemm */
void e2_quant_rows(const float *x, int m, int k, uint8_t *q, float *scale);
void e2_quant_rows_blk(const float *x, int m, int k, uint8_t *q, float *scale, int blk);
/* Pack a float [n][k] matrix into the int8 tile layout (row-major input). */
int  e2_pack_mat(const float *w, int n, int k, int rs, int cs, int8_t **q,
                 float **scale, int32_t **wsum, int *blk_out);
int  e2_pack_mat_ctx(e2_ctx *ctx, const float *w, int n, int k, int rs, int cs,
                     int8_t **q, float **scale, int32_t **wsum, int *blk_out);
void e2_pack_free(int8_t *q, float *scale, int32_t *wsum);
void e2_gemm_init(void);
double e2_gemm_calib(void);
void e2_linear_forward(e2_ctx *ctx, float *y, const float *x, int m, const e2_linear *l);
void e2_linear_forward_multi(e2_ctx *ctx, const float *x, int m,
                             const e2_linear *const *ls, float *const *ys, int n);
void e2_linear_forward_st(float *y, const float *x, int m, const e2_linear *l,
                          uint8_t *xq, float *xs, float *tmp);
void e2_linear_forward_pq(e2_ctx *ctx, float *y, const uint8_t *xq,
                          const float *xs, int m, const e2_linear *l);
void e2_gemm_range(float *y, int ns, const uint8_t *xq, const float *xs, int xss,
                   int m, int k, const e2_qmat *w, int nb0, int nb1);
int  e2_pack_mat_into(e2_ctx *ctx, const float *w, int n, int k, int rs, int cs,
                      int8_t *q, float *scale, int32_t *wsum, int *blk_out);
void e2_gemm_f32(float *y, const float *x, const float *w, int m, int k, int n);
void e2_gemm_nt_f32(float *y, const float *x, const float *w, int m, int k, int n);
void e2_dequant_row(const e2_qmat *w, int row, float *out);

/* ops */
void e2_rmsnorm(e2_ctx *ctx, float *out, const float *x, const float *w, int rows,
                int dim, float eps);
void e2_rmsnorm_pre(e2_ctx *ctx, float *out, const float *x, const float *w, int rows,
                    int dim, float eps, float pre);
void e2_rmsnorm_add(e2_ctx *ctx, float *x, const float *src, const float *w, int rows,
                    int dim, float eps);
void e2_gelu_mul(e2_ctx *ctx, float *x, const float *up, int n);
void e2_gelu_mul_q(e2_ctx *ctx, const float *x, const float *up, int rows, int k,
                   uint8_t *q, float *sc, int blk);
uint8_t *e2_quant_scratch(e2_ctx *ctx, int m, int k, float **xs);
void e2_rmsnorm_noweight(e2_ctx *ctx, float *out, const float *x, int rows, int dim,
                         float eps);
void e2_gelu_tanh(float *x, int n);
void e2_gelu_tanh_ctx(e2_ctx *ctx, float *x, int n);
void e2_silu(float *x, int n);
void e2_add_inplace(float *a, const float *b, int n);
void e2_add_inplace_ctx(e2_ctx *ctx, float *a, const float *b, int n);
void e2_vec_scale_ctx(e2_ctx *ctx, float *x, int n, float k);
void e2_silu_ctx(e2_ctx *ctx, float *x, int n);
void e2_gelu_tanh_p(e2_ctx *ctx, float *x, int n);
void e2_l2norm_rows(e2_ctx *ctx, float *x, int rows, int dim);
void e2_softmax(float *x, int n);
float e2_span_max(const float *x, int n);
void  e2_tanh_span(float *x, int n);
void  e2_gated_silu(float *x, int rows, int dim);
float e2_exp_sum(float *x, int n, float mx);
void e2_rope_free(e2_ctx *ctx);
void e2_vec_scale(float *x, int n, float k);
void e2_rope_text(e2_ctx *ctx, float *q, float *k, int t, int heads, int kv_heads,
                  int head_dim, int pos_offset, float theta);
/* Cached cos/sin table, or NULL on OOM. */
e2_rope_tab *e2_rope_get(e2_ctx *ctx, float theta, int half, int need);
/* RMSNorm over `per_pos` consecutive rows per position, then rotate each row
 * with the table.  Fuses the two passes the text encoder used to make over q
 * and k (rope was a separate read+write of both). */
void e2_rmsnorm_rope(e2_ctx *ctx, float *x, const float *w, int rows, int dim,
                     float eps, const e2_rope_tab *tb, int per_pos,
                     const int *seq_off, int nseq);
void e2_rope_vision(e2_ctx *ctx, float *q, float *k, int t, int heads, int head_dim,
                    const int *xy, float theta);
void e2_attention_text(e2_ctx *ctx, float *out, const float *q, const float *k,
                       const float *v, int t, int heads, int kv_heads, int head_dim,
                       int window);
void e2_attention_text_multi(e2_ctx *ctx, float *out, const float *q, const float *k,
                             const float *v, int t, const int *seq_off, int nseq,
                             int heads, int kv_heads, int head_dim, int window);
void e2_attention_vision(e2_ctx *ctx, float *out, const float *q, const float *k,
                         const float *v, int t, int heads, int head_dim);
void e2_attention_full(e2_ctx *ctx, float *out, const float *q, const float *k,
                       const float *v, int t, int heads, int head_dim);
/* int8 attention core (e2_attn.c): returns nonzero if it cannot handle the
 * shape, in which case the caller falls back to the f32 kernel. */
int  e2_attn_int8(e2_ctx *ctx, float *out, const float *q, const float *k,
                  const float *v, int t, int heads, int kv_heads, int head_dim,
                  int window, int nseq, const int *seq_off);
int  e2_audio_forward(e2_ctx *ctx, const float *samples, int n, float *out, int *out_tokens);
int  e2_text_forward(e2_ctx *ctx, const int *ids, int n, float *last768);
int  e2_text_forward_embeds(e2_ctx *ctx, const float *emb, int n, float *last768);
int  e2_text_forward_multi(e2_ctx *ctx, const float *emb, int n,
                           const int *seq_off, const int *seq_len, int nseq, float *out);
void e2_text_embed_ids(e2_ctx *ctx, const int *ids, int n, float *out);
int  e2_vision_forward(e2_ctx *ctx, const uint8_t *rgb, int w, int h, int max_soft,
                       float *out, int *ntok);
int  e2_image_prepare(e2_ctx *ctx, const uint8_t *src, int w, int h, int max_soft,
                      float **patches, int **pos, int *np, int *pw, int *ph);
int  e2_image_load_rgb(const char *path, uint8_t **rgb, int *w, int *h);
int  e2_text_forward_soft(e2_ctx *ctx, const int *ids, int n,
                          const float *const *soft, const int *soft_len, int nsoft,
                          float *last768);

/* tokenizer */
e2_tok *e2_tok_load(const char *path);
void e2_tok_free(e2_tok *t);
int  e2_tok_encode(e2_tok *t, const char *text, int *ids, int max_ids);
int  e2_tok_id(e2_tok *t, const char *token);

/* Fast exp: range reduction + degree-5 polynomial, rel. error < 2e-6. */
static inline float e2_fexp(float x) {
    if (x < -87.3f) return 0.0f;
    if (x > 88.7f) return 1e38f;
    float n = floorf(x * 1.4426950408889634f + 0.5f);
    float r = x - n * 0.6931471805599453f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f +
              r * (0.04166667f + r * 0.00833333f))));
    union { float f; int32_t i; } v;
    v.f = p;
    v.i += (int32_t)n << 23;
    return v.f;
}

/* misc */
int e2_fail(const char *fmt, ...);
double e2_now_ms(void);

#endif
