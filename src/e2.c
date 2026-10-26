/* e2.c — model loading, workspace management, public entry points. */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <assert.h>

#include "model.h"
#include "q8cache.h"

void e2_text_init(e2_model *m);
int e2_vision_init(e2_model *m);
void e2_vision_free(e2_vision *v);
int e2_audio_init(e2_model *m);
void e2_audio_free(e2_audio *a);

static char g_err[512];
/* The one model this process owns: the packed-cache mapping and the worker pool behind a handle are
 * process-global (q8cache.c, threads.c), so two live handles cannot be independent. A second e2_load
 * is refused before it touches either. */
static e2_model *g_live;

/* E2_Q8_WRITE=<file> dumps the quantized model the loader builds; E2_Q8_READ=<file> builds the model
 * from that dump instead of from model.safetensors (the int8 weight set plus the tensors that stay
 * bf16/f32 - the token table, the vision position table, the norm scales). See q8cache.c. */
static int g_q8_write;
static int g_q8_read;

int e2_q8_is_write(void) { return g_q8_write; }

void e2_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

const char *e2_last_error(void) { return g_err; }

/* ---------------------------------------------------------------- tensor lookup */
const uint16_t *e2_tensor_bf16(e2_model *m, const char *name, int ndim, const int64_t *dims) {
    if (g_q8_read) {
        const uint16_t *p = q8c_get(name, Q8C_BF16, ndim, dims, NULL);
        if (p) { m->wdtype = E2_DT_BF16; return p; }
        /* a quantized weight is stored packed, not in bf16: NULL here, and e2_q8w_load (same name)
         * fills in the tensor the kernels actually use */
        if (q8c_kind_of(name) == Q8C_Q8W) return NULL;
        e2_set_error("%s", q8c_error());
        return NULL;
    }
    const st_tensor *t = st_find(&m->st, name);
    if (!t) { e2_set_error("missing tensor %s", name); return NULL; }
    if (t->ndim != ndim) {
        e2_set_error("tensor %s has rank %d, expected %d", name, t->ndim, ndim);
        return NULL;
    }
    for (int i = 0; i < ndim; i++) {
        if (dims[i] > 0 && t->shape[i] != dims[i]) {
            e2_set_error("tensor %s dim %d is %lld, expected %lld", name, i, (long long)t->shape[i],
                         (long long)dims[i]);
            return NULL;
        }
    }
    if (st_is_bf16(t)) {
        m->wdtype = E2_DT_BF16;
        const uint16_t *p = st_bf16(&m->st, t);
        if (g_q8_write) {
            int64_t d[8];
            for (int i = 0; i < ndim; i++) d[i] = t->shape[i];
            if (q8c_defer(name, Q8C_BF16, ndim, d, p, st_numel(t) * 2)) {
                e2_set_error("cache write: %s", q8c_error());
                return NULL;
            }
        }
        return p;
    }
    e2_set_error("tensor %s is %s, expected BF16 (F32 not wired up yet)", name, t->dtype);
    return NULL;
}

int e2_q8_is_read(void) { return g_q8_read; }

int e2_tensor_exists(e2_model *m, const char *name) {
    if (g_q8_read) {
        const int64_t d[1] = { 0 };
        return q8c_get(name, Q8C_BF16, 1, d, NULL) != NULL;
    }
    return st_find(&m->st, name) != NULL;
}

/* The int8 weight path in one place: quantize from the checkpoint, read the packed tensor back from
 * the cache, or write the freshly quantized one to it. Idempotent for a tensor that is already built:
 * e2_audio_prepare is lazy and retried as a whole after a failure, so re-quantizing (and, on the
 * writer path, recording the same tensor twice) would leak and duplicate. Only a complete tensor
 * counts as built - quantize() zeroes q when it fails and q8c_get_q8w leaves it untouched. */
int e2_q8w_load(e2_model *m, e2_q8w *q, const char *name, const uint16_t *w, int N, int K) {
    (void)m;
    if (q->w && q->scale && q->bsq && q->N == N && q->K == K) return 0;
    if (g_q8_read) {
        if (q8c_get_q8w(name, q)) { e2_set_error("%s", q8c_error()); return -1; }
        if (q->N != N || q->K != K) {
            e2_set_error("%s: cache is [%d,%d], expected [%d,%d]", name, q->N, q->K, N, K);
            return -1;
        }
        return 0;
    }
    if (e2_q8w_from_bf16(q, w, N, K)) return -1;
    if (g_q8_write) {
        q8c_drop(name); /* the bf16 copy the loader just fetched is not needed in the cache */
        if (q8c_put_q8w(name, q)) { e2_set_error("cache write: %s", q8c_error()); return -1; }
    }
    return 0;
}

/* copy a bf16 tensor into caller memory as f32 (used for norm scales) */
int e2_copy_bf16_f32(e2_model *m, const char *name, int64_t n, float *dst) {
    const int64_t dims[1] = { n };
    const uint16_t *src = e2_tensor_bf16(m, name, 1, dims);
    if (!src) return -1;
    e2_bf16_to_f32(dst, src, (size_t)n);
    return 0;
}

static float *copy_bf16_f32(e2_model *m, const char *name, int64_t expect) {
    const int64_t dims[1] = { expect };
    const uint16_t *src = e2_tensor_bf16(m, name, 1, dims);
    if (!src) return NULL;
    float *dst = malloc(sizeof(float) * (size_t)expect);
    if (!dst) { e2_set_error("oom: %s", name); return NULL; }
    e2_bf16_to_f32(dst, src, (size_t)expect);
    return dst;
}

/* ---------------------------------------------------------------- workspace */
/* One function defines both the size and the layout, so the two can never drift apart (they did once:
 * the score buffer was reserved for 64 worker rows but advanced by S rows, silently overflowing the
 * arena whenever S was smaller than the reservation). */
static size_t ws_layout(e2_ws *w, float *base, const e2_model *m, size_t S) {
    float *p = base;
    size_t n = 0;
#define SLOT(field, cnt)                 \
    do {                                 \
        size_t c_ = (size_t)(cnt);       \
        if (w) w->field = p;             \
        p += c_;                         \
        n += c_;                         \
    } while (0)
    const size_t D = (size_t)m->hidden, I = (size_t)m->intermediate, PD = (size_t)m->ple_dim;
    const size_t L = (size_t)m->n_layers;
    SLOT(x, S * D);
    SLOT(res, S * D);
    SLOT(ple, S * L * PD);
    SLOT(q, S * 2048);
    SLOT(k, S * 1024);
    SLOT(v, S * 1024);
    SLOT(att, S * 2048);
    SLOT(attn, E2_MAX_THREADS * E2_MAX_SEQ); /* one score row per worker */
    SLOT(kt, E2_NUM_HEADS * 512 * S);        /* blocked attention: transposed K */
    SLOT(asblk, (size_t)e2_nthreads() * E2_ATT_BQ * S);
    SLOT(gate, S * I);
    SLOT(up, S * I);
    SLOT(ple_g, S * PD);
    SLOT(tmp, S * D);
    SLOT(emb, S * E2_EMBED_DIM);
    SLOT(qa_words, (S * 2048 + 3) / 4); /* int8 activations, byte view */
    SLOT(qa_scale, S * 64);             /* per-32-block activation scales (2048/32 = 64 max) */
    SLOT(valid_words, (S + 3) / 4);
#undef SLOT
    return n;
}

int e2_ws_alloc(e2_model *m, int max_seq, int max_patches) {
    e2_ws *w = &m->ws;
    if (w->x) return 0;
    const size_t S = (size_t)max_seq;
    const size_t floats = ws_layout(NULL, NULL, m, S);
    w->bytes = floats * sizeof(float) + 64;
    void *blk = malloc(w->bytes);
    if (!blk) { e2_set_error("oom: %zu bytes of workspace", w->bytes); return -1; }
    const size_t used = ws_layout(w, (float *)blk, m, S);
    if (used != floats) { e2_set_error("workspace layout mismatch"); free(blk); memset(w, 0, sizeof *w); return -1; }
    w->qa = (uint8_t *)w->qa_words;
    w->valid = (uint8_t *)w->valid_words;
    return 0;
}

void e2_ws_free(e2_model *m) {
    if (m->ws.x) free(m->ws.x);
    memset(&m->ws, 0, sizeof m->ws);
}

/* ---------------------------------------------------------------- load */
/* Releases everything a model owns. The packed-cache mapping and the worker pool are process-global,
 * so e2_free and the failure path below close those themselves. Shared with the failure path: a
 * partial model must release exactly what a complete one does, or a failed load leaks its int8
 * weights (that was ~142 MB per load of the text tower alone). */
static void model_release(e2_model *m) {
    e2_q8w_free(&m->q8_ple_model);
    e2_q8w_free(&m->q8_embed_proj);
    for (int l = 0; l < m->n_layers; l++) {
        e2_layer *L = &m->layers[l];
        e2_q8w_free(&L->q8_q);
        e2_q8w_free(&L->q8_k);
        e2_q8w_free(&L->q8_v);
        e2_q8w_free(&L->q8_o);
        e2_q8w_free(&L->q8_gate);
        e2_q8w_free(&L->q8_up);
        e2_q8w_free(&L->q8_down);
        e2_q8w_free(&L->q8_ple_gate);
        e2_q8w_free(&L->q8_ple_proj);
        free(L->q_norm); free(L->k_norm); free(L->input_ln); free(L->post_attn_ln);
        free(L->pre_ffn_ln); free(L->post_ffn_ln); free(L->ple_norm);
    }
    free(m->ple_proj_norm);
    free(m->final_norm);
    for (int t = 0; t < 2; t++) { free(m->rope_cos[t]); free(m->rope_sin[t]); }
    e2_ws_free(m);
    e2_vision_free((e2_vision *)m->vision);
    e2_audio_free((e2_audio *)m->audio);
    st_close(&m->st);
    free(m);
}

e2_model *e2_load(const char *model_dir) {
    if (g_live) {
        e2_set_error("a model is already loaded: one live model per process, call e2_free first");
        return NULL;
    }
    if (!model_dir || !model_dir[0]) {
        e2_set_error("e2_load: model_dir is empty");
        return NULL;
    }
    char path[1024];
    const int plen = snprintf(path, sizeof path, "%s/model.safetensors", model_dir);
    if (plen < 0 || (size_t)plen >= sizeof path) { /* silently truncated would read the wrong file */
        e2_set_error("e2_load: model_dir is too long (%.200s...)", model_dir);
        return NULL;
    }

    e2_model *m = calloc(1, sizeof *m);
    if (!m) { e2_set_error("oom"); return NULL; }
    m->st.fd = -1;
    e2_cpu_init();

    const char *qw = getenv("E2_Q8_WRITE"), *qr = getenv("E2_Q8_READ");
    g_q8_write = qw && qw[0];
    g_q8_read = qr && qr[0];
    /* every failure from here on releases through `fail`, including the ones that only opened a
     * file: a half-opened load must not leave session state (the g_q8_* flags above) behind */
    if (g_q8_read && g_q8_write) {
        e2_set_error("E2_Q8_READ and E2_Q8_WRITE are exclusive");
        goto fail;
    }
    if (!g_q8_read && st_open(&m->st, path)) { e2_set_error("%s", st_error()); goto fail; }
    if (g_q8_read && q8c_read_open(qr)) { e2_set_error("%s", q8c_error()); goto fail; }
    if (g_q8_write && q8c_write_open(qw)) { e2_set_error("%s", q8c_error()); goto fail; }

    m->hidden = 512;
    m->intermediate = 2048;
    m->ple_dim = 512;
    m->rms_eps = 1e-6f;
    m->embed_scale = sqrtf((float)m->hidden);
    m->ple_scale = 1.0f / sqrtf((float)m->hidden);
    m->rope_theta_full = 1000000.0f;
    m->rope_theta_sliding = 10000.0f;

    /* count layers by probing names */
    for (int l = 0; l < E2_MAX_LAYERS; l++) {
        char name[192];
        snprintf(name, sizeof name, "language_model.layers.%d.input_layernorm.weight", l);
        if (!e2_tensor_exists(m, name)) break;
        m->n_layers++;
    }
    if (m->n_layers != E2_MAX_LAYERS) {
        e2_set_error("expected %d text layers, found %d", E2_MAX_LAYERS, m->n_layers);
        goto fail;
    }
    e2_text_init(m);

    /* fixed shapes: every text-tower dimension is known here, so the checkpoint and the packed cache
     * are checked against all of it (no wildcard second dimension) */
    const int64_t d_vocab[2] = { E2_VOCAB, m->hidden };
    m->embed_tokens = e2_tensor_bf16(m, "language_model.embed_tokens.weight", 2, d_vocab);
    const int64_t d_ple[2] = { (int64_t)m->n_layers * m->ple_dim, m->hidden };
    m->ple_model_proj = e2_tensor_bf16(m, "language_model.ple.per_layer_model_projection.weight", 2, d_ple);
    const int64_t d_proj[2] = { E2_EMBED_DIM, m->hidden };
    m->embed_proj = e2_tensor_bf16(m, "language_model.embedding_projection.weight", 2, d_proj);
    if (!e2_weight_ok(m->embed_tokens) || !e2_weight_ok(m->ple_model_proj) ||
        !e2_weight_ok(m->embed_proj))
        goto fail;
    m->ple_proj_norm = copy_bf16_f32(m, "language_model.ple.per_layer_projection_norm.weight", m->ple_dim);
    m->final_norm = copy_bf16_f32(m, "language_model.norm.weight", m->hidden);
    if (!m->ple_proj_norm || !m->final_norm) goto fail;

    char n_[14][96];
    const char *suffix[14] = {
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            "ple_block.per_layer_input_gate.weight", "ple_block.per_layer_projection.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        "ple_block.post_per_layer_input_norm.weight",
    };
    for (int l = 0; l < m->n_layers; l++) {
        e2_layer *L = &m->layers[l];
        for (int s = 0; s < 14; s++) snprintf(n_[s], sizeof n_[s], "language_model.layers.%d.%s", l, suffix[s]);

        const int64_t dq[2] = { L->q_dim, m->hidden }, dkv[2] = { L->kv_dim, m->hidden };
        const int64_t dff[2] = { m->intermediate, m->hidden }, don[2] = { m->hidden, L->q_dim };
        const int64_t ddn[2] = { m->hidden, m->intermediate }, dpl[2] = { m->ple_dim, m->hidden };
        const int64_t dpp[2] = { m->hidden, m->ple_dim };
        L->q_proj = e2_tensor_bf16(m, n_[0], 2, dq);
        L->k_proj = e2_tensor_bf16(m, n_[1], 2, dkv);
        L->v_proj = e2_tensor_bf16(m, n_[2], 2, dkv);
        L->o_proj = e2_tensor_bf16(m, n_[3], 2, don);
        L->gate_proj = e2_tensor_bf16(m, n_[4], 2, dff);
        L->up_proj = e2_tensor_bf16(m, n_[5], 2, dff);
        L->down_proj = e2_tensor_bf16(m, n_[6], 2, ddn);
        L->ple_gate = e2_tensor_bf16(m, n_[7], 2, dpl);
        L->ple_proj = e2_tensor_bf16(m, n_[8], 2, dpp);
        if (!e2_weight_ok(L->q_proj) || !e2_weight_ok(L->k_proj) || !e2_weight_ok(L->v_proj) ||
            !e2_weight_ok(L->o_proj) || !e2_weight_ok(L->gate_proj) || !e2_weight_ok(L->up_proj) ||
            !e2_weight_ok(L->down_proj) || !e2_weight_ok(L->ple_gate) || !e2_weight_ok(L->ple_proj))
            goto fail;
        L->input_ln = copy_bf16_f32(m, n_[9], m->hidden);
        L->post_attn_ln = copy_bf16_f32(m, n_[10], m->hidden);
        L->pre_ffn_ln = copy_bf16_f32(m, n_[11], m->hidden);
        L->post_ffn_ln = copy_bf16_f32(m, n_[12], m->hidden);
        L->ple_norm = copy_bf16_f32(m, n_[13], m->hidden);
        if (!L->input_ln || !L->post_attn_ln || !L->pre_ffn_ln || !L->post_ffn_ln || !L->ple_norm) goto fail;
        {
            char qn[96], kn[96];
            snprintf(qn, sizeof qn, "language_model.layers.%d.self_attn.q_norm.weight", l);
            snprintf(kn, sizeof kn, "language_model.layers.%d.self_attn.k_norm.weight", l);
            L->q_norm = copy_bf16_f32(m, qn, L->head_dim);
            L->k_norm = copy_bf16_f32(m, kn, L->head_dim);
            if (!L->q_norm || !L->k_norm) goto fail;
        }
        {
            char sn[96];
            snprintf(sn, sizeof sn, "language_model.layers.%d.layer_scalar", l);
            const int64_t d1[1] = { 1 };
            const uint16_t *ls = e2_tensor_bf16(m, sn, 1, d1);
            if (!ls) goto fail;
            L->layer_scalar = e2_bf16f(ls[0]);
        }
    }

    /* int8 weights: quantize once at load. Enabled when the CPU has AVX2 (the kernel needs it);
     * E2_INT8=0 forces the bf16 path, which is kept for A/B comparison and debugging. */
    const char *no_i8 = getenv("E2_INT8");
    m->use_int8 = (e2_i8_quantize != NULL) && !(no_i8 && no_i8[0] == '0');
    if ((g_q8_read || g_q8_write) && !m->use_int8) {
        e2_set_error("the packed cache holds the int8 weights: E2_INT8=0 cannot use it");
        goto fail;
    }
    if (m->use_int8) {
        if (e2_q8w_load(m, &m->q8_ple_model, "language_model.ple.per_layer_model_projection.weight",
                        m->ple_model_proj, m->n_layers * m->ple_dim, m->hidden) ||
            e2_q8w_load(m, &m->q8_embed_proj, "language_model.embedding_projection.weight",
                        m->embed_proj, E2_EMBED_DIM, m->hidden))
            goto fail;
        for (int l = 0; l < m->n_layers; l++) {
            e2_layer *L = &m->layers[l];
            for (int s = 0; s < 9; s++)
                snprintf(n_[s], sizeof n_[s], "language_model.layers.%d.%s", l, suffix[s]);
            if (e2_q8w_load(m, &L->q8_q, n_[0], L->q_proj, L->q_dim, m->hidden) ||
                e2_q8w_load(m, &L->q8_k, n_[1], L->k_proj, L->kv_dim, m->hidden) ||
                e2_q8w_load(m, &L->q8_v, n_[2], L->v_proj, L->kv_dim, m->hidden) ||
                e2_q8w_load(m, &L->q8_o, n_[3], L->o_proj, m->hidden, L->q_dim) ||
                e2_q8w_load(m, &L->q8_gate, n_[4], L->gate_proj, m->intermediate, m->hidden) ||
                e2_q8w_load(m, &L->q8_up, n_[5], L->up_proj, m->intermediate, m->hidden) ||
                e2_q8w_load(m, &L->q8_down, n_[6], L->down_proj, m->hidden, m->intermediate) ||
                e2_q8w_load(m, &L->q8_ple_gate, n_[7], L->ple_gate, m->ple_dim, m->hidden) ||
                e2_q8w_load(m, &L->q8_ple_proj, n_[8], L->ple_proj, m->hidden, m->ple_dim))
                goto fail;
        }
    }

    e2_pool_global_init(0); /* default: one worker per hardware thread */
    if (e2_ws_alloc(m, E2_MAX_SEQ, 0)) goto fail;
    if (e2_vision_init(m)) goto fail;
    if (e2_audio_init(m)) goto fail;
    if (g_q8_write && q8c_write_close()) { e2_set_error("%s", q8c_error()); goto fail; }
    g_live = m;
    g_err[0] = 0;
    return m;

fail:
    q8c_write_close(); /* a partial cache is useless; the reader reports the missing tensor */
    q8c_read_close();
    g_q8_read = g_q8_write = 0; /* no cache is open any more: e2_q8_is_read/write must say so */
    model_release(m);
    e2_pool_global_free(); /* this load may have built it; no handle exists to own it */
    if (!g_err[0]) e2_set_error("model load failed");
    return NULL;
}

void e2_free(e2_model *m) {
    if (!m) return;
    if (g_live == m) g_live = NULL;
    q8c_read_close(); /* the borrowed q8w arrays point into this mapping */
    q8c_write_close();
    g_q8_read = g_q8_write = 0;
    model_release(m);
    e2_pool_global_free();
}

/* ---------------------------------------------------------------- info */
void e2_info(const e2_model *m) {
    printf("EmbeddingGemma 2 engine\n");
    printf("  cpu backend : %s, %d threads, weights %s%s\n", e2_cpu_name(), e2_nthreads(),
           m->use_int8 ? "int8 (VNNI/AVX2)" : "bf16", g_q8_read ? ", packed cache" : "");
    if (g_q8_read) printf("  tensors     : from the packed cache (E2_Q8_READ)\n");
    else printf("  tensors     : %zu from safetensors\n", m->st.count);
    printf("  text        : %d layers, hidden %d, ffn %d, heads %d, ple %d\n",
           m->n_layers, m->hidden, m->intermediate, E2_NUM_HEADS, m->ple_dim);
    printf("  layer types : sliding (head_dim 256, kv 2) / full (head_dim 512, kv 1)\n");
    printf("  audio       : %s\n", m->audio ? "12 layer USM conformer, 1024 wide, 8 heads" : "not loaded");
    printf("  workspace   : %.1f MB\n", (double)m->ws.bytes / 1e6);
}
