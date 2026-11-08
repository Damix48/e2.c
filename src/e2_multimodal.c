/* e2_multimodal.c - public image/audio/video/interleaved entry points. */
#include "e2_internal.h"
#include "e2_gpu.h"

#include <stdlib.h>
#include <string.h>

int e2_image_load_rgb(const char *path, uint8_t **rgb, int *w, int *h);
int e2_audio_forward(e2_ctx *ctx, const float *samples, int n, float *out, int *out_tokens);

/* The audio tower still runs its convolutions and attention inline on the CPU,
 * so the device path must be off around it: a deferred op would leave buffers
 * half-written while that CPU code reads them. */
static int e2_audio_forward_cpu(e2_ctx *ctx, const float *samples, int n,
                                float *out, int *ntok) {
    int rc = -1;
    e2_gpu_no_dev(ctx, 1);
    rc = e2_audio_forward(ctx, samples, n, out, ntok);
    e2_gpu_no_dev(ctx, 0);
    return rc;
}

#define ID_BOS 2
#define ID_EOS 1
#define ID_BOI 255999
#define ID_EOI 258882
#define ID_BOA 256000
#define ID_EOA 258883
#define ID_IMAGE 258880
#define ID_AUDIO 258881
#define ID_VIDEO 258884

typedef struct {
    float *v;      /* soft tokens [n][hidden] */
    int n;
    /* video: per-frame arrays */
    int frames;
    float **fv;
    int *fn;
} soft_item;

static int is_placeholder(int id) {
    return id == ID_IMAGE || id == ID_AUDIO || id == ID_VIDEO;
}

static int seq_finish(e2_ctx *ctx, const float *emb, int T, float *out, int dim) {
    float e[E2_EMBED_DIM];
    int rc = e2_text_forward_embeds(ctx, emb, T, e);
    if (rc != 0) return rc;
    if (dim <= 0 || dim > E2_EMBED_DIM) dim = E2_EMBED_DIM;
    if (dim != 768 && dim != 512 && dim != 256 && dim != 128)
        return e2_fail("dim must be 128, 256, 512 or 768");
    e2_normalize(e, dim);
    memcpy(out, e, (size_t)dim * sizeof(float));
    return 0;
}

/*
 * Build the token-embedding sequence for mixed text/soft input.
 * Placeholders are expanded exactly like EmbeddingGemma2Processor:
 *   image -> BOI + s image tokens + EOI
 *   audio -> BOA + s audio tokens + EOA
 *   video -> per frame: BOI + s frame tokens + EOI
 */
static int build_and_run(e2_ctx *ctx, const char *text,
                         soft_item *items, int n_items, int *item_kind,
                         float *out, int dim) {
    e2_tok *tk = ctx->m.tok;
    const int H = ctx->m.meta.text_hidden;
    int *ids = malloc(8192 * sizeof(int));
    if (!ids) return e2_fail("oom");
    int nt = e2_tok_encode(tk, text, ids, 8192);
    if (nt < 0) { free(ids); return e2_fail("tokenization failed"); }

    int T = 2; /* BOS + EOS */
    for (int i = 0; i < nt; i++) {
        if (ids[i] == ID_IMAGE || ids[i] == ID_AUDIO) T += 2; /* wrappers */
        else if (ids[i] == ID_VIDEO) { /* per-frame wrappers added below */ }
        else T += 1;
    }
    int ii = 0;
    for (int i = 0; i < nt; i++) {
        if (!is_placeholder(ids[i])) continue;
        if (ii >= n_items) { free(ids); return e2_fail("too many placeholders"); }
        soft_item *s = &items[ii];
        if (s->frames > 0)
            for (int f = 0; f < s->frames; f++) T += 2 + s->fn[f];
        else
            T += s->n;
        ii++;
    }
    if (ii != n_items) { free(ids); return e2_fail("placeholder count mismatch"); }

    float *emb = malloc((size_t)T * H * sizeof(float));
    if (!emb) { free(ids); return e2_fail("oom"); }
    int pos = 0;
    int one = ID_BOS;
    e2_text_embed_ids(ctx, &one, 1, emb + (size_t)pos++ * H);

    ii = 0;
    for (int i = 0; i < nt; i++) {
        int id = ids[i];
        if (!is_placeholder(id)) {
            e2_text_embed_ids(ctx, &id, 1, emb + (size_t)pos++ * H);
            continue;
        }
        soft_item *s = &items[ii++];
        if (id == ID_AUDIO) {
            int b = ID_BOA, e = ID_EOA;
            e2_text_embed_ids(ctx, &b, 1, emb + (size_t)pos++ * H);
            memcpy(emb + (size_t)pos * H, s->v, (size_t)s->n * H * sizeof(float));
            pos += s->n;
            e2_text_embed_ids(ctx, &e, 1, emb + (size_t)pos++ * H);
        } else if (id == ID_VIDEO && s->frames > 0) {
            for (int f = 0; f < s->frames; f++) {
                int b = ID_BOI, e = ID_EOI;
                e2_text_embed_ids(ctx, &b, 1, emb + (size_t)pos++ * H);
                memcpy(emb + (size_t)pos * H, s->fv[f],
                       (size_t)s->fn[f] * H * sizeof(float));
                pos += s->fn[f];
                e2_text_embed_ids(ctx, &e, 1, emb + (size_t)pos++ * H);
            }
        } else {
            int b = ID_BOI, e = ID_EOI;
            e2_text_embed_ids(ctx, &b, 1, emb + (size_t)pos++ * H);
            memcpy(emb + (size_t)pos * H, s->v, (size_t)s->n * H * sizeof(float));
            pos += s->n;
            e2_text_embed_ids(ctx, &e, 1, emb + (size_t)pos++ * H);
        }
    }
    one = ID_EOS;
    e2_text_embed_ids(ctx, &one, 1, emb + (size_t)pos++ * H);
    if (pos != T) {
        fprintf(stderr, "seq mismatch pos=%d T=%d nt=%d ids:", pos, T, nt);
        for (int i = 0; i < nt; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "  items=%d n0=%d frames=%d fn0=%d\n", n_items, items[0].n, items[0].frames, items[0].frames ? items[0].fn[0] : -1);
        free(ids); free(emb); return e2_fail("sequence length mismatch");
    }
    free(ids);
    int rc = seq_finish(ctx, emb, T, out, dim);
    free(emb);
    return rc;
}

static void free_items(soft_item *it, int n) {
    for (int i = 0; i < n; i++) {
        free(it[i].v);
        if (it[i].fv) {
            for (int f = 0; f < it[i].frames; f++) free(it[i].fv[f]);
            free(it[i].fv);
            free(it[i].fn);
        }
    }
}

static int run_one_image(e2_ctx *ctx, const char *path, float *out, int dim) {
    uint8_t *rgb = NULL;
    int w = 0, h = 0;
    if (e2_image_load_rgb(path, &rgb, &w, &h) != 0) return -1;
    int rc = e2_embed_image_rgb(ctx, rgb, w, h, out, dim);
    free(rgb);
    return rc;
}

int e2_embed_image(e2_ctx *ctx, const char *path, float *out, int dim) {
    return run_one_image(ctx, path, out, dim);
}

int e2_embed_image_rgb(e2_ctx *ctx, const uint8_t *rgb, int w, int h, float *out, int dim) {
    if (!ctx || !rgb || !out) return e2_fail("invalid argument");
    if (!(ctx->m.meta.flags & E2_MOD_IMAGE)) return e2_fail("vision tower not present");
    double t0 = e2_now_ms();
    int max_soft = ctx->m.meta.vision_soft_tokens;
    float *soft = malloc((size_t)(max_soft + 8) * ctx->m.meta.text_hidden * sizeof(float));
    if (!soft) return e2_fail("oom");
    int nt = 0;
    if (e2_vision_forward(ctx, rgb, w, h, max_soft, soft, &nt) != 0) { free(soft); return -1; }
    soft_item it = {soft, nt, 0, NULL, NULL};
    int kind = ID_IMAGE;
    int rc = build_and_run(ctx, "<|image|>", &it, 1, &kind, out, dim);
    free(soft);
    ctx->last_ms = e2_now_ms() - t0;
    return rc;
}

int e2_embed_audio(e2_ctx *ctx, const float *samples, int n, float *out, int dim) {
    if (!ctx || !samples || n <= 0 || !out) return e2_fail("invalid argument");
    if (!(ctx->m.meta.flags & E2_MOD_AUDIO)) return e2_fail("audio tower not present");
    double t0 = e2_now_ms();
    int cap = n / 160 + 8;
    float *soft = malloc((size_t)cap * ctx->m.meta.text_hidden * sizeof(float));
    if (!soft) return e2_fail("oom");
    int nt = 0;
    if (e2_audio_forward_cpu(ctx, samples, n, soft, &nt) != 0) { free(soft); return -1; }
    soft_item it = {soft, nt, 0, NULL, NULL};
    int kind = ID_AUDIO;
    int rc = build_and_run(ctx, "<|audio|>", &it, 1, &kind, out, dim);
    free(soft);
    ctx->last_ms = e2_now_ms() - t0;
    return rc;
}

static int run_video(e2_ctx *ctx, const char *const *frames, int nframes, float *out, int dim) {
    if (!ctx || !frames || nframes <= 0 || !out) return e2_fail("invalid argument");
    if (!(ctx->m.meta.flags & E2_MOD_IMAGE)) return e2_fail("vision tower not present");
    float **fv = calloc((size_t)nframes, sizeof(float *));
    int *fn = calloc((size_t)nframes, sizeof(int));
    if (!fv || !fn) { free(fv); free(fn); return e2_fail("oom"); }
    int max_soft = 140;
    for (int f = 0; f < nframes; f++) {
        uint8_t *rgb = NULL;
        int w = 0, h = 0;
        if (e2_image_load_rgb(frames[f], &rgb, &w, &h) != 0) goto fail;
        fv[f] = malloc((size_t)(max_soft + 8) * ctx->m.meta.text_hidden * sizeof(float));
        if (!fv[f]) { free(rgb); goto fail; }
        if (e2_vision_forward(ctx, rgb, w, h, max_soft, fv[f], &fn[f]) != 0) { free(rgb); goto fail; }
        free(rgb);
    }
    {
        soft_item it = {NULL, 0, nframes, fv, fn};
        int kind = ID_VIDEO;
        int rc = build_and_run(ctx, "<|video|>", &it, 1, &kind, out, dim);
        free_items(&it, 1);
        return rc;
    }
fail:
    for (int f = 0; f < nframes; f++) free(fv[f]);
    free(fv);
    free(fn);
    return -1;
}

int e2_embed_video(e2_ctx *ctx, const char *const *frame_paths, int nframes,
                   float *out, int dim) {
    double t0 = e2_now_ms();
    int rc = run_video(ctx, frame_paths, nframes, out, dim);
    if (ctx) ctx->last_ms = e2_now_ms() - t0;
    return rc;
}

/* Append one vision item for an already-decoded RGB frame. */
static int make_image_item(e2_ctx *ctx, const char *path, soft_item *it) {
    uint8_t *rgb = NULL;
    int w = 0, h = 0;
    if (e2_image_load_rgb(path, &rgb, &w, &h) != 0) return -1;
    int max_soft = ctx->m.meta.vision_soft_tokens;
    it->v = malloc((size_t)(max_soft + 8) * ctx->m.meta.text_hidden * sizeof(float));
    if (!it->v) { free(rgb); return e2_fail("oom"); }
    if (e2_vision_forward(ctx, rgb, w, h, max_soft, it->v, &it->n) != 0) {
        free(rgb);
        return -1;
    }
    free(rgb);
    return 0;
}

static int make_video_item(e2_ctx *ctx, const char *const *frames, int nf, soft_item *it) {
    int H = ctx->m.meta.text_hidden;
    it->frames = nf;
    it->fv = calloc((size_t)nf, sizeof(float *));
    it->fn = calloc((size_t)nf, sizeof(int));
    if (!it->fv || !it->fn) return e2_fail("oom");
    for (int f = 0; f < nf; f++) {
        uint8_t *rgb = NULL;
        int w = 0, h = 0;
        if (e2_image_load_rgb(frames[f], &rgb, &w, &h) != 0) return -1;
        it->fv[f] = malloc(148 * H * sizeof(float));
        if (!it->fv[f]) { free(rgb); return e2_fail("oom"); }
        if (e2_vision_forward(ctx, rgb, w, h, 140, it->fv[f], &it->fn[f]) != 0) {
            free(rgb);
            return -1;
        }
        free(rgb);
    }
    return 0;
}

static int make_audio_item(e2_ctx *ctx, const float *samples, int n, soft_item *it) {
    int cap = n / 160 + 8;
    it->v = malloc((size_t)cap * ctx->m.meta.text_hidden * sizeof(float));
    if (!it->v) return e2_fail("oom");
    return e2_audio_forward_cpu(ctx, samples, n, it->v, &it->n);
}

int e2_embed(e2_ctx *ctx, const e2_input *in, float *out, int dim) {
    if (!ctx || !in || !in->text || !out) return e2_fail("invalid argument");
    double t0 = e2_now_ms();

    /* Count placeholders and build soft items in text order. */
    int cap = 1;
    for (const char *p = in->text; *p; p++) if (*p == '<') cap++;
    soft_item *items = calloc((size_t)cap, sizeof(*items));
    int *kinds = calloc((size_t)cap, sizeof(int));
    if (!items || !kinds) { free(items); free(kinds); return e2_fail("oom"); }

    int ii = 0, imgi = 0, vidi = 0, audi = 0;
    const char *p = in->text;
    while (*p) {
        int kind = 0, len = 0;
        if (!strncmp(p, "<|image|>", 9)) { kind = ID_IMAGE; len = 9; }
        else if (!strncmp(p, "<|audio|>", 9)) { kind = ID_AUDIO; len = 9; }
        else if (!strncmp(p, "<|video|>", 9)) { kind = ID_VIDEO; len = 9; }
        if (!kind) { p++; continue; }
        int rc = -1;
        if (kind == ID_IMAGE) {
            if (imgi >= in->n_images) { rc = e2_fail("missing image for placeholder"); goto fail; }
            rc = make_image_item(ctx, in->images[imgi++], &items[ii]);
        } else if (kind == ID_VIDEO) {
            if (vidi >= in->n_videos) { rc = e2_fail("missing video for placeholder"); goto fail; }
            if (in->video_frames[vidi] <= 0) { rc = e2_fail("video has no frames"); goto fail; }
            rc = make_video_item(ctx, in->videos[vidi], in->video_frames[vidi], &items[ii]);
            vidi++;
        } else {
            if (audi >= in->n_audio) { rc = e2_fail("missing audio for placeholder"); goto fail; }
            rc = make_audio_item(ctx, in->audio[audi], in->audio_len[audi], &items[ii]);
            audi++;
        }
        if (rc != 0) goto fail;
        kinds[ii++] = kind;
        p += len;
    }
    if (imgi != in->n_images || vidi != in->n_videos || audi != in->n_audio) {
        e2_fail("unused media inputs");
        goto fail;
    }
    {
        int rc = build_and_run(ctx, in->text, items, ii, kinds, out, dim);
        free_items(items, ii);
        free(items);
        free(kinds);
        ctx->last_ms = e2_now_ms() - t0;
        return rc;
    }
fail:
    free_items(items, ii);
    free(items);
    free(kinds);
    return -1;
}
