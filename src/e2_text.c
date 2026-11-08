/* e2_text.c - EmbeddingGemma-2 bidirectional text encoder. */
#include "e2_internal.h"
#include "e2_gpu.h"

#include <math.h>
#include <stdlib.h>

static e2_ctx *e2_dump_ctx;

static void dump_tensor(const char *name, const float *p, size_t n) {
    const char *dir = getenv("E2_DUMP");
    if (!dir) return;
    e2_gpu_flush(e2_dump_ctx);          /* the device may still owe this buffer */
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

/* h[t][P] *= ple[t][L][P] slice for layer li (rows are contiguous) */
typedef struct { float *h; const float *ple; int n, L, P, li; } plemul_arg;
static void plemul_work(void *vp, int tid, int nt) {
    plemul_arg *a = vp;
    int t0 = (int)((long)a->n * tid / nt), t1 = (int)((long)a->n * (tid + 1) / nt);
    int P = a->P, L = a->L, li = a->li;
    for (int t = t0; t < t1; t++) {
        const float *pl = a->ple + ((size_t)t * L + li) * P;
        float *hp = a->h + (size_t)t * P;
        for (int i = 0; i < P; i++) hp[i] *= pl[i];
    }
}

int e2_text_forward_multi(e2_ctx *ctx, const float *emb_in, int n,
                          const int *seq_off, const int *seq_len, int nseq,
                          float *out) {
    int prof_on = getenv("E2_PROFILE") != NULL;
    e2_dump_ctx = ctx;
    double prof[16] = {0};
    double p2[10] = {0};
    double _q = 0;
    double t_all = e2_now_ms();
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
    _q = e2_now_ms();
    e2_rmsnorm_pre(ctx, ple, ple, m->ple_norm, n * L, P, c->text_eps, c->text_ple_scale);
    p2[0] += e2_now_ms() - _q;
    dump_tensor("ple", ple, (size_t)n * L * P);

    for (int li = 0; li < L; li++) {
        e2_text_layer *ly = &m->tl[li];
        int hd = ly->head_dim, kvh = ly->kv_heads;
        int qdim = c->text_heads * hd;
        int kdim = kvh * hd;

        _i = 2;
        PROF_T(e2_rmsnorm(ctx, h, x, ly->in_norm, n, H, c->text_eps));
        if (li == 0) dump_tensor("l0_h", h, (size_t)n * H);

        _i = 3;
        {
            const e2_linear *ls[3] = {&ly->q, &ly->k, &ly->v};
            float *ys[3] = {q, k, v};
            PROF_T(e2_linear_forward_multi(ctx, h, n, ls, ys, 3));
        }
        _i = 4;

        if (li == 0) dump_tensor("l0_vraw", v, (size_t)n * kvh * hd);
        int window = ly->is_full ? -1 : c->text_sliding;
        int tmax = seq_len[0];
        for (int sq = 1; sq < nseq; sq++) if (seq_len[sq] > tmax) tmax = seq_len[sq];
        float theta = ly->is_full ? c->text_rope_theta_full : c->text_rope_theta_sliding;
        /* E2_NOFUSE=1 keeps the old norm-then-rope passes (debug aid: the
         * l0_qpre/l0_kpre dumps only exist on that path). */
        e2_rope_tab *rt = getenv("E2_NOFUSE") ? NULL : e2_rope_get(ctx, theta, hd / 2, tmax);
        _q = e2_now_ms();
        if (rt) {
            e2_rmsnorm_rope(ctx, q, ly->q_norm, n * c->text_heads, hd, c->text_eps,
                            rt, c->text_heads, seq_off, nseq);
            e2_rmsnorm_rope(ctx, k, ly->k_norm, n * kvh, hd, c->text_eps,
                            rt, kvh, seq_off, nseq);
            e2_rmsnorm(ctx, v, v, NULL, n * kvh, hd, c->text_eps);
        } else {
            e2_rmsnorm(ctx, q, q, ly->q_norm, n * c->text_heads, hd, c->text_eps);
            e2_rmsnorm(ctx, k, k, ly->k_norm, n * kvh, hd, c->text_eps);
            e2_rmsnorm(ctx, v, v, NULL, n * kvh, hd, c->text_eps);
            p2[2] += e2_now_ms() - _q;
            if (li == 0) {
                dump_tensor("l0_qpre", q, (size_t)n * c->text_heads * hd);
                dump_tensor("l0_kpre", k, (size_t)n * kvh * hd);
                dump_tensor("l0_vpre", v, (size_t)n * kvh * hd);
            }
            _q = e2_now_ms();
            for (int sq = 0; sq < nseq; sq++)
                e2_rope_text(ctx, q + (size_t)seq_off[sq] * c->text_heads * hd,
                             k + (size_t)seq_off[sq] * kvh * hd, seq_len[sq],
                             c->text_heads, kvh, hd, 0, theta);
        }
        p2[1] += e2_now_ms() - _q;
        if (li == 0 && nseq == 1) {
            dump_tensor("l0_q", q, (size_t)n * c->text_heads * hd);
            dump_tensor("l0_k", k, (size_t)n * kvh * hd);
            dump_tensor("l0_v", v, (size_t)n * kvh * hd);
        }

        int same_len = nseq > 1;
        for (int sq = 1; sq < nseq && same_len; sq++)
            if (seq_len[sq] != seq_len[0]) same_len = 0;
        _i = 6;
        double _t = e2_now_ms();
        if (same_len)
            e2_attention_text_multi(ctx, att, q, k, v, seq_len[0], seq_off, nseq,
                                    c->text_heads, kvh, hd, window);
        else
            for (int sq = 0; sq < nseq; sq++)
                e2_attention_text(ctx, att + (size_t)seq_off[sq] * c->text_heads * hd,
                                  q + (size_t)seq_off[sq] * c->text_heads * hd,
                                  k + (size_t)seq_off[sq] * kvh * hd,
                                  v + (size_t)seq_off[sq] * kvh * hd,
                                  seq_len[sq], c->text_heads, kvh, hd, window);
        if (prof_on) prof[6] += e2_now_ms() - _t;
        if (li == 0 && nseq == 1) dump_tensor("l0_att", att, (size_t)n * c->text_heads * hd);

        /* out projection + post-attention norm + residual */
        _i = 7;
        PROF_T(e2_linear_forward(ctx, tmp, att, n, &ly->o));
        _i = 8;
        _q = e2_now_ms();
        e2_rmsnorm_add(ctx, x, tmp, ly->post_attn_norm, n, H, c->text_eps);
        p2[3] += e2_now_ms() - _q;
        if (li == 0 && nseq == 1) dump_tensor("l0_x1", x, (size_t)n * H);

        /* MLP */
        _q = e2_now_ms();
        e2_rmsnorm(ctx, h, x, ly->pre_ff_norm, n, H, c->text_eps);
        p2[4] += e2_now_ms() - _q;
        _i = 9;
        if (e2_gpu_op_mlp(ctx, h, h, n, &ly->gate, &ly->up, &ly->down) == 0) {
            _i = 15;
            goto mlp_done;
        }
        {
            const e2_linear *ls[2] = {&ly->gate, &ly->up};
            float *ys[2] = {tmp, up};
            PROF_T(e2_linear_forward_multi(ctx, h, n, ls, ys, 2));
        }
        _i = 10;
        _q = e2_now_ms();
        float *gs = NULL;
        uint8_t *gq = e2_quant_scratch(ctx, n, c->text_inter, &gs);
        if (gq) {
            e2_gelu_mul_q(ctx, tmp, up, n, c->text_inter, gq, gs, 128);
            p2[5] += e2_now_ms() - _q;
            _i = 11;
            PROF_T(e2_linear_forward_pq(ctx, h, gq, gs, n, &ly->down));
        } else {
            e2_gelu_mul(ctx, tmp, up, n * c->text_inter);
            p2[5] += e2_now_ms() - _q;
            _i = 11;
            PROF_T(e2_linear_forward(ctx, h, tmp, n, &ly->down));
        }
mlp_done:
        _q = e2_now_ms();
        e2_rmsnorm_add(ctx, x, h, ly->post_ff_norm, n, H, c->text_eps);
        p2[6] += e2_now_ms() - _q;
        if (li == 0 && nseq == 1) dump_tensor("l0_x2", x, (size_t)n * H);

        /* PLE block: gate, gelu, multiply by this layer's per-token slice */
        _i = 12;
        /* device: gate + gelu + slice multiply + quantise + projection, with no
         * f32 intermediate at all */
        if (e2_gpu_op_ple(ctx, tmp, x, n, &ly->ple_gate, &ly->ple_proj, ple, L,
                          li) == 0) {
            _i = 15;
            goto ple_done;
        }
        PROF_T(e2_linear_forward(ctx, h, x, n, &ly->ple_gate));
        PROF_T(e2_gelu_tanh_ctx(ctx, h, n * P));
        _q = e2_now_ms();
        if (e2_gpu_op_mul_slice(ctx, h, ple, n * P, L, P, li) == 0) {
            p2[7] += e2_now_ms() - _q;
        } else if (ctx->nthreads > 1 && (long)n * P >= 65536) {
            plemul_arg pa = {h, ple, n, L, P, li};
            e2_parallel(ctx, plemul_work, &pa);
        } else {
            plemul_arg pa = {h, ple, n, L, P, li};
            plemul_work(&pa, 0, 1);
        }
        p2[7] += e2_now_ms() - _q;
        _i = 13;
        PROF_T(e2_linear_forward(ctx, tmp, h, n, &ly->ple_proj));
ple_done:
        _i = 15;
        _q = e2_now_ms();
        e2_rmsnorm_add(ctx, x, tmp, ly->ple_norm, n, H, c->text_eps);
        p2[8] += e2_now_ms() - _q;

        if (li == 0 && nseq == 1) dump_tensor("l0_x3", x, (size_t)n * H);
        float ls = ly->layer_scalar ? ly->layer_scalar[0] : 1.0f;
        _q = e2_now_ms();
        if (ls != 1.0f) e2_vec_scale_ctx(ctx, x, n * H, ls);
        p2[9] += e2_now_ms() - _q;

        if (nseq == 1) {
            char nm[32];
            snprintf(nm, sizeof(nm), "layer%d", li + 1);
            dump_tensor(nm, x, (size_t)n * H);
        }

        (void)qdim; (void)kdim;
    }

    e2_rmsnorm(ctx, h, x, m->final_norm, n, H, c->text_eps);
    float *proj = e2_arena_alloc(&ctx->scratch, (size_t)n * c->text_embed_dim * sizeof(float));
    if (!proj) return -1;
    e2_linear lproj = {0}; lproj.w = m->text_proj;
    e2_linear_forward(ctx, proj, h, n, &lproj);
    if (nseq == 1) dump_tensor("last_hidden_state", proj, (size_t)n * c->text_embed_dim);

    if (e2_gpu_flush(ctx) != 0) return e2_fail("device forward failed");
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
        double timed = 0;
        for (int i = 0; i < 15; i++) timed += prof[i];
        prof[15] = (e2_now_ms() - t_all) - timed;
        for (int i = 0; i < 16; i++) if (prof[i] > 0) fprintf(stderr, "%-10s %7.2f ms\n", nm[i], prof[i]);
        fprintf(stderr, "%-10s %7.2f ms\n", "total", e2_now_ms() - t_all);
        static const char *nm2[10] = {"ple_scale","rope","qkvnorms","postattn","preff",
            "gatedmul","postff","plemul","plenorm","layerscale"};
        for (int i = 0; i < 10; i++) if (p2[i] > 0) fprintf(stderr, "  %-8s %7.2f ms\n", nm2[i], p2[i]);
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
