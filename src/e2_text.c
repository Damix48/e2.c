/* e2_text.c - EmbeddingGemma-2 bidirectional text encoder. */
#include "e2_internal.h"

#include <math.h>
#include <stdlib.h>

static void dump_tensor(const char *name, const float *p, size_t n) {
    const char *dir = getenv("E2_DUMP");
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, sizeof(float), n, f);
    fclose(f);
}

void e2_text_embed_ids(e2_ctx *ctx, const int *ids, int n, float *out) {
    const e2_qmat *w = &ctx->m.embed;
    float scale = ctx->m.meta.text_embed_scale;
    int k = w->k;
    for (int t = 0; t < n; t++) {
        int id = ids[t];
        const int8_t *q = w->q + (size_t)id * k;
        float s = w->scale[id] * scale;
        float *o = out + (size_t)t * k;
        for (int i = 0; i < k; i++) o[i] = (float)q[i] * s;
    }
}

#define PROF_T(x) do { if (prof_on) { double _t = e2_now_ms(); x; prof[_i] += e2_now_ms() - _t; } else x; } while (0)

int e2_text_forward_multi(e2_ctx *ctx, const float *emb_in, int n,
                          const int *seq_off, const int *seq_len, int nseq,
                          float *out) {
    int prof_on = getenv("E2_PROFILE") != NULL;
    double prof[16] = {0};
    int _i = 0;
    e2_model *m = &ctx->m;
    e2_meta *c = &m->meta;
    const int H = c->text_hidden, L = c->text_layers, P = c->text_ple;

    e2_arena_reset(&ctx->scratch);
    float *x = e2_arena_alloc(&ctx->scratch, (size_t)n * H * sizeof(float));
    float *h = e2_arena_alloc(&ctx->scratch, (size_t)n * H * sizeof(float));
    float *tmp = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_inter * sizeof(float));
    float *up = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_inter * sizeof(float));
    float *ple = e2_arena_alloc(&ctx->scratch, (size_t)n * L * P * sizeof(float));
    float *q = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_heads * c->text_head_dim_full * sizeof(float));
    float *k = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_kv * c->text_head_dim * sizeof(float));
    float *v = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_kv * c->text_head_dim * sizeof(float));
    float *att = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_heads * c->text_head_dim_full * sizeof(float));
    if (!x || !h || !tmp || !up || !ple || !q || !k || !v || !att) return -1;

    _i = 0; PROF_T(memcpy(x, emb_in, (size_t)n * H * sizeof(float)));
    dump_tensor("embeddings", x, (size_t)n * H);

    /* per-layer embeddings: project all layers at once from the token embeddings */
    e2_linear lple = {0}; lple.w = m->ple_proj;
    _i = 1;
    PROF_T(e2_linear_forward(ctx, ple, x, n, &lple));
    e2_vec_scale(ple, n * L * P, c->text_ple_scale);
    e2_rmsnorm(ple, ple, m->ple_norm, n * L, P, c->text_eps);
    dump_tensor("ple", ple, (size_t)n * L * P);

    for (int li = 0; li < L; li++) {
        e2_text_layer *ly = &m->tl[li];
        int hd = ly->head_dim, kvh = ly->kv_heads;
        int qdim = c->text_heads * hd;
        int kdim = kvh * hd;

        _i = 2;
        PROF_T(e2_rmsnorm(h, x, ly->in_norm, n, H, c->text_eps));
        if (li == 0) dump_tensor("l0_h", h, (size_t)n * H);

        _i = 3;
        PROF_T(e2_linear_forward(ctx, q, h, n, &ly->q));
        _i = 4;
        PROF_T(e2_linear_forward(ctx, k, h, n, &ly->k));
        _i = 5;
        PROF_T(e2_linear_forward(ctx, v, h, n, &ly->v));
        _i = 4;

        if (li == 0) dump_tensor("l0_vraw", v, (size_t)n * kvh * hd);
        e2_rmsnorm(q, q, ly->q_norm, n * c->text_heads, hd, c->text_eps);
        e2_rmsnorm(k, k, ly->k_norm, n * kvh, hd, c->text_eps);
        e2_rmsnorm(v, v, NULL, n * kvh, hd, c->text_eps);
        if (li == 0) {
            dump_tensor("l0_qpre", q, (size_t)n * c->text_heads * hd);
            dump_tensor("l0_kpre", k, (size_t)n * kvh * hd);
            dump_tensor("l0_vpre", v, (size_t)n * kvh * hd);
        }

        float theta = ly->is_full ? c->text_rope_theta_full : c->text_rope_theta_sliding;
        for (int sq = 0; sq < nseq; sq++)
            e2_rope_text(ctx, q + (size_t)seq_off[sq] * c->text_heads * hd,
                         k + (size_t)seq_off[sq] * kvh * hd, seq_len[sq],
                         c->text_heads, kvh, hd, 0, theta);
        if (li == 0 && nseq == 1) {
            dump_tensor("l0_q", q, (size_t)n * c->text_heads * hd);
            dump_tensor("l0_k", k, (size_t)n * kvh * hd);
            dump_tensor("l0_v", v, (size_t)n * kvh * hd);
        }

        int window = ly->is_full ? -1 : c->text_sliding;
        _i = 6;
        if (prof_on) {
            double _t = e2_now_ms();
            for (int sq = 0; sq < nseq; sq++)
                e2_attention_text(ctx, att + (size_t)seq_off[sq] * c->text_heads * hd,
                                  q + (size_t)seq_off[sq] * c->text_heads * hd,
                                  k + (size_t)seq_off[sq] * kvh * hd,
                                  v + (size_t)seq_off[sq] * kvh * hd,
                                  seq_len[sq], c->text_heads, kvh, hd, window);
            prof[6] += e2_now_ms() - _t;
        } else {
            for (int sq = 0; sq < nseq; sq++)
                e2_attention_text(ctx, att + (size_t)seq_off[sq] * c->text_heads * hd,
                                  q + (size_t)seq_off[sq] * c->text_heads * hd,
                                  k + (size_t)seq_off[sq] * kvh * hd,
                                  v + (size_t)seq_off[sq] * kvh * hd,
                                  seq_len[sq], c->text_heads, kvh, hd, window);
        }
        if (li == 0 && nseq == 1) dump_tensor("l0_att", att, (size_t)n * c->text_heads * hd);

        /* out projection + post-attention norm + residual */
        _i = 7;
        PROF_T(e2_linear_forward(ctx, tmp, att, n, &ly->o));
        _i = 8;
        e2_rmsnorm(tmp, tmp, ly->post_attn_norm, n, H, c->text_eps);
        e2_add_inplace(x, tmp, n * H);
        if (li == 0 && nseq == 1) dump_tensor("l0_x1", x, (size_t)n * H);

        /* MLP */
        e2_rmsnorm(h, x, ly->pre_ff_norm, n, H, c->text_eps);
        _i = 9;
        PROF_T(e2_linear_forward(ctx, tmp, h, n, &ly->gate));
        _i = 10;
        PROF_T(e2_linear_forward(ctx, up, h, n, &ly->up));
        _i = 14;
        PROF_T(e2_gelu_tanh_ctx(ctx, tmp, n * c->text_inter));
        for (size_t i = 0; i < (size_t)n * c->text_inter; i++) tmp[i] *= up[i];
        _i = 11;
        PROF_T(e2_linear_forward(ctx, h, tmp, n, &ly->down));
        e2_rmsnorm(h, h, ly->post_ff_norm, n, H, c->text_eps);
        e2_add_inplace(x, h, n * H);
        if (li == 0 && nseq == 1) dump_tensor("l0_x2", x, (size_t)n * H);

        /* PLE block: gate, gelu, multiply by this layer's per-token slice */
        _i = 12;
        PROF_T(e2_linear_forward(ctx, h, x, n, &ly->ple_gate));
        PROF_T(e2_gelu_tanh_ctx(ctx, h, n * P));
        for (int t = 0; t < n; t++) {
            const float *pl = ple + ((size_t)t * L + li) * P;
            float *hp = h + (size_t)t * P;
            for (int i = 0; i < P; i++) hp[i] *= pl[i];
        }
        _i = 13;
        PROF_T(e2_linear_forward(ctx, tmp, h, n, &ly->ple_proj));
        _i = 15;
        e2_rmsnorm(tmp, tmp, ly->ple_norm, n, H, c->text_eps);
        e2_add_inplace(x, tmp, n * H);

        if (li == 0 && nseq == 1) dump_tensor("l0_x3", x, (size_t)n * H);
        float ls = ly->layer_scalar ? ly->layer_scalar[0] : 1.0f;
        if (ls != 1.0f) e2_vec_scale(x, n * H, ls);

        if (nseq == 1) {
            char nm[32];
            snprintf(nm, sizeof(nm), "layer%d", li + 1);
            dump_tensor(nm, x, (size_t)n * H);
        }

        (void)qdim; (void)kdim;
    }

    e2_rmsnorm(h, x, m->final_norm, n, H, c->text_eps);
    float *proj = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_embed_dim * sizeof(float));
    if (!proj) return -1;
    e2_linear lproj = {0}; lproj.w = m->text_proj;
    e2_linear_forward(ctx, proj, h, n, &lproj);
    if (nseq == 1) dump_tensor("last_hidden_state", proj, (size_t)n * c->text_embed_dim);

    for (int sq = 0; sq < nseq; sq++) {
        const float *pr = proj + (size_t)seq_off[sq] * c->text_embed_dim;
        float *o = out + (size_t)sq * c->text_embed_dim;
        for (int i = 0; i < c->text_embed_dim; i++) o[i] = 0.0f;
        for (int t = 0; t < seq_len[sq]; t++)
            for (int i = 0; i < c->text_embed_dim; i++)
                o[i] += pr[(size_t)t * c->text_embed_dim + i];
        float inv = 1.0f / (float)seq_len[sq];
        for (int i = 0; i < c->text_embed_dim; i++) o[i] *= inv;
    }
    if (nseq == 1) {
        float tmpn[E2_EMBED_DIM];
        memcpy(tmpn, out, sizeof(tmpn));
        e2_normalize(tmpn, c->text_embed_dim);
        dump_tensor("embedding", tmpn, (size_t)c->text_embed_dim);
    }
    if (prof_on) {
        static const char *nm[16] = {"embed","ple_proj","rmsnorm_in","q","k","v","attn",
            "o_proj","-","mlp_gate","mlp_up","mlp_down","ple_gate","ple_proj2","gelu","other"};
        for (int i = 0; i < 16; i++) if (prof[i] > 0) fprintf(stderr, "%-10s %7.2f ms\n", nm[i], prof[i]);
    }
    return 0;
}

int e2_text_forward_embeds(e2_ctx *ctx, const float *emb, int n, float *last768) {
    int off = 0, len = n;
    return e2_text_forward_multi(ctx, emb, n, &off, &len, 1, last768);
}

int e2_text_forward(e2_ctx *ctx, const int *ids, int n, float *last768) {
    int H = ctx->m.meta.text_hidden;
    e2_arena_reset(&ctx->scratch);
    float *emb = e2_arena_alloc(&ctx->scratch, (size_t)n * H * sizeof(float));
    if (!emb) return e2_fail("oom");
    e2_text_embed_ids(ctx, ids, n, emb);
    float *tmp = malloc((size_t)n * H * sizeof(float));
    if (!tmp) return e2_fail("oom");
    memcpy(tmp, emb, (size_t)n * H * sizeof(float));
    int rc = e2_text_forward_embeds(ctx, tmp, n, last768);
    free(tmp);
    return rc;
}
