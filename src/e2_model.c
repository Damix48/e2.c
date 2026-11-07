/* e2_model.c - model file parsing and weight binding. */
#include "e2_internal.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define E2_MAGIC "E2MODEL1"
#define E2_ENTRY_SIZE 168

static char g_err[512] = "ok";

int e2_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    return -1;
}

const char *e2_error(void) { return g_err; }

static int name_eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

const e2_tensor *e2_find(const e2_model *m, const char *name) {
    for (int i = 0; i < m->ntensors; i++)
        if (name_eq(m->tensors[i].name, name)) return &m->tensors[i];
    return NULL;
}

const float *e2_f32_opt(const e2_model *m, const char *name) {
    const e2_tensor *t = e2_find(m, name);
    if (!t || t->dtype != E2_DT_F32) return NULL;
    return (const float *)t->data;
}

const float *e2_f32(const e2_model *m, const char *name, const char *what) {
    const float *p = e2_f32_opt(m, name);
    if (!p) e2_fail("%s missing: %s", what, name);
    return p;
}

e2_qmat e2_qmat_get(const e2_model *m, const char *name) {
    e2_qmat w = {0};
    const e2_tensor *t = e2_find(m, name);
    char buf[128];
    if (!t || t->dtype != E2_DT_I8) {
        e2_fail("matrix missing: %s", name);
        return w;
    }
    w.q = (const int8_t *)t->data;
    w.n = (int)t->shape[0];
    w.k = (int)t->shape[1];
    w.packed = (t->flags & E2_TF_PACKED) != 0;
    if (w.n <= 0 || w.k <= 0) { e2_fail("bad matrix shape: %s", name); return w; }
    if (w.packed && (w.k % 128) != 0) {
        e2_fail("%s: K=%d is not a multiple of 128", name, w.k);
        return w;
    }
    snprintf(buf, sizeof(buf), "%s.scale", name);
    const e2_tensor *s = e2_find(m, buf);
    if (!s || s->dtype != E2_DT_F32) { e2_fail("scale missing: %s", buf); return w; }
    w.scale = (const float *)s->data;
    if (w.packed) {
        snprintf(buf, sizeof(buf), "%s.wsum", name);
        const e2_tensor *ws = e2_find(m, buf);
        if (!ws || ws->dtype != E2_DT_I32) { e2_fail("wsum missing: %s", buf); return w; }
        w.wsum = (const int32_t *)ws->data;
    }
    return w;
}

e2_linear e2_linear_get(const e2_model *m, const char *weight_name) {
    e2_linear l = {0};
    l.w = e2_qmat_get(m, weight_name);
    char buf[128];
    size_t n = strlen(weight_name);
    if (n > 7 && name_eq(weight_name + n - 7, ".weight")) {
        memcpy(buf, weight_name, n - 7);
        buf[n - 7] = 0;
        strncat(buf, ".bias", sizeof(buf) - strlen(buf) - 1);
        l.w.bias = e2_f32_opt(m, buf);
    }
    return l;
}

/* Audio clipped linear: base = "...ffw_layer_1", weight = base + ".linear.weight". */
e2_linear e2_linear_clip_get(const e2_model *m, const char *base) {
    e2_linear l = {0};
    char buf[128];
    snprintf(buf, sizeof(buf), "%s.linear.weight", base);
    l.w = e2_qmat_get(m, buf);
    snprintf(buf, sizeof(buf), "%s.input_min", base);
    l.in_min = e2_f32_opt(m, buf);
    snprintf(buf, sizeof(buf), "%s.input_max", base);
    l.in_max = e2_f32_opt(m, buf);
    snprintf(buf, sizeof(buf), "%s.output_min", base);
    l.out_min = e2_f32_opt(m, buf);
    snprintf(buf, sizeof(buf), "%s.output_max", base);
    l.out_max = e2_f32_opt(m, buf);
    return l;
}

static void bind_text(e2_ctx *ctx) {
    e2_model *m = &ctx->m;
    e2_meta *c = &m->meta;
    char a[160], b[160];

    snprintf(a, sizeof(a), "language_model.embed_tokens.weight");
    m->embed = e2_qmat_get(m, a);
    m->ple_proj = e2_qmat_get(m, "language_model.ple.per_layer_model_projection.weight");
    m->ple_norm = e2_f32(m, "language_model.ple.per_layer_projection_norm.weight", "ple norm");
    m->text_proj = e2_qmat_get(m, "language_model.embedding_projection.weight");
    m->final_norm = e2_f32(m, "language_model.norm.weight", "text norm");

    m->tl = calloc((size_t)c->text_layers, sizeof(e2_text_layer));
    if (!m->tl) { e2_fail("out of memory"); return; }
    for (int i = 0; i < c->text_layers; i++) {
        e2_text_layer *L = &m->tl[i];
        snprintf(a, sizeof(a), "language_model.layers.%d.self_attn.q_proj.weight", i);
        L->q = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.self_attn.k_proj.weight", i);
        L->k = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.self_attn.v_proj.weight", i);
        L->v = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.self_attn.o_proj.weight", i);
        L->o = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.mlp.gate_proj.weight", i);
        L->gate = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.mlp.up_proj.weight", i);
        L->up = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.mlp.down_proj.weight", i);
        L->down = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.ple_block.per_layer_input_gate.weight", i);
        L->ple_gate = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "language_model.layers.%d.ple_block.per_layer_projection.weight", i);
        L->ple_proj = e2_linear_get(m, a);

        snprintf(a, sizeof(a), "language_model.layers.%d.input_layernorm.weight", i);
        L->in_norm = e2_f32(m, a, "input_layernorm");
        snprintf(a, sizeof(a), "language_model.layers.%d.post_attention_layernorm.weight", i);
        L->post_attn_norm = e2_f32(m, a, "post_attention_layernorm");
        snprintf(a, sizeof(a), "language_model.layers.%d.pre_feedforward_layernorm.weight", i);
        L->pre_ff_norm = e2_f32(m, a, "pre_feedforward_layernorm");
        snprintf(a, sizeof(a), "language_model.layers.%d.post_feedforward_layernorm.weight", i);
        L->post_ff_norm = e2_f32(m, a, "post_feedforward_layernorm");
        snprintf(a, sizeof(a), "language_model.layers.%d.ple_block.post_per_layer_input_norm.weight", i);
        L->ple_norm = e2_f32(m, a, "ple_block norm");
        snprintf(a, sizeof(a), "language_model.layers.%d.self_attn.q_norm.weight", i);
        L->q_norm = e2_f32(m, a, "q_norm");
        snprintf(a, sizeof(a), "language_model.layers.%d.self_attn.k_norm.weight", i);
        L->k_norm = e2_f32(m, a, "k_norm");
        snprintf(a, sizeof(a), "language_model.layers.%d.layer_scalar", i);
        L->layer_scalar = e2_f32(m, a, "layer_scalar");

        L->is_full = (c->text_full_mask >> i) & 1;
        L->head_dim = L->is_full ? c->text_head_dim_full : c->text_head_dim;
        L->kv_heads = L->is_full ? c->text_kv_full : c->text_kv;
        (void)b;
    }
}

static void bind_vision(e2_ctx *ctx) {
    e2_model *m = &ctx->m;
    e2_meta *c = &m->meta;
    char a[160];

    m->patch_proj = e2_qmat_get(m, "vision_tower.patch_embedder.input_proj.weight");
    m->pos_table = e2_f32(m, "vision_tower.patch_embedder.position_embedding_table", "pos table");
    m->vis_embed = e2_qmat_get(m, "embed_vision.embedding_projection.weight");

    m->vl = calloc((size_t)c->vision_layers, sizeof(e2_vision_layer));
    if (!m->vl) { e2_fail("out of memory"); return; }
    for (int i = 0; i < c->vision_layers; i++) {
        e2_vision_layer *L = &m->vl[i];
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.self_attn.q_proj.linear.weight", i);
        L->q = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.self_attn.k_proj.linear.weight", i);
        L->k = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.self_attn.v_proj.linear.weight", i);
        L->v = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.self_attn.o_proj.linear.weight", i);
        L->o = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.mlp.gate_proj.linear.weight", i);
        L->gate = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.mlp.up_proj.linear.weight", i);
        L->up = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.mlp.down_proj.linear.weight", i);
        L->down = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.input_layernorm.weight", i);
        L->in_norm = e2_f32(m, a, "vision input norm");
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.post_attention_layernorm.weight", i);
        L->post_attn_norm = e2_f32(m, a, "vision post attn norm");
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.pre_feedforward_layernorm.weight", i);
        L->pre_ff_norm = e2_f32(m, a, "vision pre ff norm");
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.post_feedforward_layernorm.weight", i);
        L->post_ff_norm = e2_f32(m, a, "vision post ff norm");
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.self_attn.q_norm.weight", i);
        L->q_norm = e2_f32(m, a, "vision q norm");
        snprintf(a, sizeof(a), "vision_tower.encoder.layers.%d.self_attn.k_norm.weight", i);
        L->k_norm = e2_f32(m, a, "vision k norm");
    }
}

static void bind_audio(e2_ctx *ctx) {
    e2_model *m = &ctx->m;
    e2_meta *c = &m->meta;
    char a[160], b[160];

    /* conv kernels are 4-D; the packer stores them f32 */
    const e2_tensor *t0 = e2_find(m, "audio_tower.subsample_conv_projection.layer0.conv.weight");
    if (t0 && t0->dtype == E2_DT_F32) m->aud_conv0 = (const float *)t0->data;
    const e2_tensor *t1 = e2_find(m, "audio_tower.subsample_conv_projection.layer1.conv.weight");
    if (t1 && t1->dtype == E2_DT_F32) m->aud_conv1 = (const float *)t1->data;
    m->aud_sub0_norm = e2_f32(m, "audio_tower.subsample_conv_projection.layer0.norm.weight", "aud norm0");
    m->aud_sub1_norm = e2_f32(m, "audio_tower.subsample_conv_projection.layer1.norm.weight", "aud norm1");
    m->aud_in_proj = e2_qmat_get(m, "audio_tower.subsample_conv_projection.input_proj_linear.weight");
    m->aud_out_proj = e2_qmat_get(m, "audio_tower.output_proj.weight");
    m->aud_out_proj.bias = e2_f32_opt(m, "audio_tower.output_proj.bias");
    m->aud_embed = e2_qmat_get(m, "embed_audio.embedding_projection.weight");

    m->al = calloc((size_t)c->audio_layers, sizeof(e2_audio_layer));
    if (!m->al) { e2_fail("out of memory"); return; }
    for (int i = 0; i < c->audio_layers; i++) {
        e2_audio_layer *L = &m->al[i];
        const char *pre[10] = {
            "audio_tower.layers.%d.self_attn.q_proj",
            "audio_tower.layers.%d.self_attn.k_proj",
            "audio_tower.layers.%d.self_attn.v_proj",
            "audio_tower.layers.%d.self_attn.post",
            "audio_tower.layers.%d.feed_forward1.ffw_layer_1",
            "audio_tower.layers.%d.feed_forward1.ffw_layer_2",
            "audio_tower.layers.%d.feed_forward2.ffw_layer_1",
            "audio_tower.layers.%d.feed_forward2.ffw_layer_2",
            "audio_tower.layers.%d.lconv1d.linear_start",
            "audio_tower.layers.%d.lconv1d.linear_end",
        };
        e2_linear *dst[10] = {&L->q, &L->k, &L->v, &L->post, &L->ff1a, &L->ff1b,
                              &L->ff2a, &L->ff2b, &L->lc_start, &L->lc_end};
        for (int j = 0; j < 10; j++) {
            snprintf(a, sizeof(a), pre[j], i);
            *dst[j] = e2_linear_clip_get(m, a);
        }
        snprintf(a, sizeof(a), "audio_tower.layers.%d.self_attn.relative_k_proj.weight", i);
        L->rel_k = e2_linear_get(m, a);
        snprintf(a, sizeof(a), "audio_tower.layers.%d.self_attn.per_dim_scale", i);
        L->per_dim_scale = e2_f32(m, a, "per_dim_scale");

        snprintf(a, sizeof(a), "audio_tower.layers.%d.feed_forward1.pre_layer_norm.weight", i);
        L->ff1_pre_norm = e2_f32(m, a, "ff1 pre norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.feed_forward1.post_layer_norm.weight", i);
        L->ff1_post_norm = e2_f32(m, a, "ff1 post norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.feed_forward2.pre_layer_norm.weight", i);
        L->ff2_pre_norm = e2_f32(m, a, "ff2 pre norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.feed_forward2.post_layer_norm.weight", i);
        L->ff2_post_norm = e2_f32(m, a, "ff2 post norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.norm_pre_attn.weight", i);
        L->pre_attn_norm = e2_f32(m, a, "pre attn norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.norm_post_attn.weight", i);
        L->post_attn_norm = e2_f32(m, a, "post attn norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.norm_out.weight", i);
        L->norm_out = e2_f32(m, a, "norm out");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.lconv1d.pre_layer_norm.weight", i);
        L->lc_pre_norm = e2_f32(m, a, "lc pre norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.lconv1d.conv_norm.weight", i);
        L->lc_conv_norm = e2_f32(m, a, "lc conv norm");
        snprintf(a, sizeof(a), "audio_tower.layers.%d.lconv1d.depthwise_conv1d.weight", i);
        L->dw_conv = e2_f32(m, a, "depthwise conv");
        (void)b;
    }
}

int e2_model_load(e2_ctx *ctx, const char *model_path, const char *tok_path) {
    e2_model *m = &ctx->m;
    snprintf(g_err, sizeof(g_err), "ok");
    memset(m, 0, sizeof(*m));

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) return e2_fail("cannot open model: %s", model_path);
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return e2_fail("fstat failed"); }
    size_t fsize = (size_t)st.st_size;
    void *map = MAP_FAILED;
    /* The weights are streamed every forward, so what matters is DRAM read
     * throughput.  Anonymous huge pages beat a 4 KB-page file mapping by ~2x
     * on the small-m towers (audio: 18 vs 39 GB/s); E2_MMAP=1 keeps the old
     * path for comparison. */
    if (!getenv("E2_MMAP")) {
        map = malloc(fsize + 4096);
        if (map) {
            size_t off = 0;
            while (off < fsize) {
                ssize_t r = pread(fd, (char *)map + off, fsize - off, (off_t)off);
                if (r <= 0) break;
                off += (size_t)r;
            }
            if (off != fsize) { free(map); map = MAP_FAILED; }
            else {
                size_t aligned = (fsize + 2u * 1024 * 1024 - 1) & ~(size_t)(2 * 1024 * 1024 - 1);
                madvise(map, aligned, MADV_HUGEPAGE);
            }
        }
    }
    if (map == MAP_FAILED) {
        map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        m->mapped = 1;
    }
    close(fd);
    if (map == MAP_FAILED) return e2_fail("cannot load model: %s", model_path);
    m->map = map;
    m->size = fsize;

    if (fsize < 20 || memcmp(map, E2_MAGIC, 8) != 0)
        return e2_fail("not an E2 model file: %s", model_path);
    const uint32_t *h = (const uint32_t *)((const char *)map + 8);
    uint32_t version = h[0], ntensors = h[1];
    if (version != 1) return e2_fail("unsupported model version %u", version);

    size_t idx_bytes = (size_t)ntensors * E2_ENTRY_SIZE;
    if (20 + idx_bytes > fsize) return e2_fail("truncated model index");
    m->tensors = calloc(ntensors, sizeof(e2_tensor));
    if (!m->tensors) return e2_fail("out of memory");
    m->ntensors = (int)ntensors;
    const char *base = (const char *)map + 20;
    const char *data0 = base + idx_bytes;

    for (uint32_t i = 0; i < ntensors; i++) {
        const char *e = base + (size_t)i * E2_ENTRY_SIZE;
        e2_tensor *t = &m->tensors[i];
        memcpy(t->name, e, 96);
        t->name[95] = 0;
        const uint32_t *u = (const uint32_t *)(e + 96);
        const uint64_t *q = (const uint64_t *)(e + 104);
        t->dtype = u[0];
        t->ndim = u[1];
        for (int k = 0; k < 4; k++) t->shape[k] = (int64_t)q[k];
        t->offset = q[4];
        t->nbytes = q[5];
        t->flags = (uint32_t)q[6];
        if (t->offset + t->nbytes > fsize - (size_t)(data0 - (const char *)map))
            return e2_fail("tensor out of bounds: %s", t->name);
        t->data = data0 + t->offset;
    }

    const e2_tensor *meta = e2_find(m, "__meta__");
    const e2_tensor *par = e2_find(m, "__params__");
    if (!meta || !par || meta->dtype != E2_DT_I32 || par->dtype != E2_DT_F32)
        return e2_fail("model metadata missing");
    const int32_t *iv = (const int32_t *)meta->data;
    const float *fv = (const float *)par->data;
    e2_meta *c = &m->meta;
    c->flags             = iv[1];
    c->text_layers       = iv[2];
    c->text_hidden       = iv[3];
    c->text_heads        = iv[4];
    c->text_kv           = iv[5];
    c->text_head_dim     = iv[6];
    c->text_head_dim_full= iv[7];
    c->text_kv_full      = iv[8];
    c->text_inter        = iv[9];
    c->text_ple          = iv[10];
    c->text_embed_dim    = iv[11];
    c->text_vocab        = iv[12];
    c->text_sliding      = iv[13];
    c->vision_layers     = iv[14];
    c->vision_hidden     = iv[15];
    c->vision_heads      = iv[16];
    c->vision_head_dim   = iv[17];
    c->vision_inter      = iv[18];
    c->vision_patch      = iv[19];
    c->vision_pos_table  = iv[20];
    c->vision_pool       = iv[21];
    c->audio_layers      = iv[22];
    c->audio_hidden      = iv[23];
    c->audio_heads       = iv[24];
    c->audio_head_dim    = iv[25];
    c->audio_inter       = iv[26];
    c->audio_output_dim  = iv[27];
    c->audio_conv_k      = iv[28];
    c->audio_chunk       = iv[29];
    c->audio_ctx_left    = iv[30];
    c->audio_ctx_right   = iv[31];
    c->text_full_mask    = iv[32];
    c->vision_soft_tokens= iv[33];

    c->text_eps               = fv[0];
    c->text_embed_scale       = fv[1];
    c->text_ple_scale         = fv[2];
    c->vision_eps             = fv[3];
    c->vision_rope_theta      = fv[4];
    c->vision_pool_scale      = fv[5];
    c->audio_eps              = fv[6];
    c->audio_softcap          = fv[7];
    c->audio_residual         = fv[8];
    c->audio_clip             = fv[9];
    c->audio_qscale           = fv[10];
    c->audio_kscale           = fv[11];
    c->text_rope_theta_sliding= fv[12];
    c->text_rope_theta_full   = fv[13];
    c->mel_floor              = fv[14];

    if (!(c->flags & E2_MOD_TEXT)) return e2_fail("model has no text tower");
    bind_text(ctx);
    if (c->flags & E2_MOD_IMAGE) bind_vision(ctx);
    if (c->flags & E2_MOD_AUDIO) bind_audio(ctx);

    m->mel_filters = e2_f32_opt(m, "__mel_filters__");
    m->hann = e2_f32_opt(m, "__hann_window__");

    if (strcmp(g_err, "ok") != 0) return -1;

    if (tok_path) {
        m->tok = e2_tok_load(tok_path);
        if (!m->tok) return -1;
    }
    snprintf(ctx->info,
             sizeof(ctx->info),
             "EmbeddingGemma-2 int8 | text %dL | vision %s | audio %s",
             c->text_layers,
             (c->flags & E2_MOD_IMAGE) ? "yes" : "no",
             (c->flags & E2_MOD_AUDIO) ? "yes" : "no");
    return 0;
}
