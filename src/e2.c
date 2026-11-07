/* e2.c - public API. */
#include "e2_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
/* Denormals in the residual stream are common (bf16 checkpoints) and cost
 * 10-100x on x86.  Flush-to-zero + denormals-are-zero on every thread. */
static void set_ftz(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned csr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(csr));
    csr |= 0x8040u;                       /* DAZ | FTZ */
    __asm__ __volatile__("ldmxcsr %0" :: "m"(csr));
#endif
}

e2_ctx *e2_load(const char *model_path, const char *tokenizer_path) {
    e2_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { e2_fail("out of memory"); return NULL; }
    set_ftz();
    int nt = e2_cpu_count();
    if (e2_pool_init(ctx, nt) != 0) { free(ctx); e2_fail("thread pool init failed"); return NULL; }
    if (e2_model_load(ctx, model_path, tokenizer_path) != 0) {
        e2_free(ctx);
        return NULL;
    }
    return ctx;
}

void e2_free(e2_ctx *ctx) {
    if (!ctx) return;
    e2_pool_free(ctx);
    e2_arena_free(&ctx->scratch);
    free(ctx->qbuf);
    free(ctx->sbuf);
    free(ctx->tbuf);
    free(ctx->ktbuf);
    e2_rope_free(ctx);
    e2_model *m = &ctx->m;
    e2_tok_free(m->tok);
    free(m->tl);
    free(m->vl);
    free(m->al);
    free(m->tensors);
    if (m->map) munmap(m->map, m->size);
    free(ctx);
}

int e2_modalities(const e2_ctx *ctx) { return ctx ? ctx->m.meta.flags : 0; }

void e2_set_threads(e2_ctx *ctx, int n) {
    if (!ctx) return;
    if (n <= 0) n = e2_cpu_count();
    e2_pool_free(ctx);
    e2_pool_init(ctx, n);
}

int e2_threads(const e2_ctx *ctx) { return ctx ? ctx->nthreads : 0; }

const char *e2_model_info(const e2_ctx *ctx) { return ctx ? ctx->info : ""; }
double e2_last_ms(const e2_ctx *ctx) { return ctx ? ctx->last_ms : 0.0; }

float e2_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void e2_normalize(float *v, int dim) {
    if (dim <= 0 || dim > E2_EMBED_DIM) dim = E2_EMBED_DIM;
    float s = 0.0f;
    for (int i = 0; i < dim; i++) s += v[i] * v[i];
    float inv = 1.0f / sqrtf(s + 1e-12f);
    for (int i = 0; i < dim; i++) v[i] *= inv;
}

static int finish(float *emb, float *out, int dim) {
    if (dim <= 0 || dim > E2_EMBED_DIM) dim = E2_EMBED_DIM;
    if (dim != E2_EMBED_DIM && dim != 512 && dim != 256 && dim != 128) {
        return e2_fail("dim must be 128, 256, 512 or 768");
    }
    e2_normalize(emb, dim);
    memcpy(out, emb, (size_t)dim * sizeof(float));
    return 0;
}

static const char *task_prefix(const char *task) {
    if (!strcmp(task, "search_query"))        return "task: search result | query: ";
    if (!strcmp(task, "search_document"))     return "title: none | text: ";
    if (!strcmp(task, "classification"))      return "task: classification | query: ";
    if (!strcmp(task, "clustering"))          return "task: clustering | query: ";
    if (!strcmp(task, "sentence_similarity")) return "task: sentence similarity | query: ";
    if (!strcmp(task, "code_retrieval"))      return "task: code retrieval | query: ";
    if (!strcmp(task, "question_answering"))  return "task: question answering | query: ";
    if (!strcmp(task, "fact_checking"))       return "task: fact checking | query: ";
    return NULL;
}

static int tokenize_text(e2_ctx *ctx, const char *text, int *ids, int max) {
    e2_tok *t = ctx->m.tok;
    if (!t) return e2_fail("no tokenizer loaded");
    if (max < 4) return e2_fail("sequence too long");
    ids[0] = 2;  /* <bos> */
    int n = e2_tok_encode(t, text, ids + 1, max - 2);
    if (n < 0) return e2_fail("text too long or tokenization failed");
    ids[n + 1] = 1;  /* <eos> */
    return n + 2;
}

int e2_embed_text(e2_ctx *ctx, const char *text, float *out, int dim) {
    if (!ctx || !text || !out) return e2_fail("invalid argument");
    if (!(ctx->m.meta.flags & E2_MOD_TEXT)) return e2_fail("text tower not present");
    double t0 = e2_now_ms();
    int max = 8192;
    int *ids = malloc((size_t)max * sizeof(int));
    if (!ids) return e2_fail("out of memory");
    int n = tokenize_text(ctx, text, ids, max);
    if (n < 0) { free(ids); return -1; }
    float emb[E2_EMBED_DIM];
    int rc = e2_text_forward(ctx, ids, n, emb);
    free(ids);
    if (rc != 0) return e2_fail("text forward failed");
    ctx->last_ms = e2_now_ms() - t0;
    return finish(emb, out, dim);
}

int e2_embed_text_batch(e2_ctx *ctx, const char *const *texts, int n, float *out, int dim) {
    if (!ctx || !texts || n <= 0 || !out) return e2_fail("invalid argument");
    int d = (dim <= 0 || dim > E2_EMBED_DIM) ? E2_EMBED_DIM : dim;
    if (d != 768 && d != 512 && d != 256 && d != 128)
        return e2_fail("dim must be 128, 256, 512 or 768");
    int H = ctx->m.meta.text_hidden;
    double t0 = e2_now_ms();

    int **ids = calloc((size_t)n, sizeof(int *));
    int *len = calloc((size_t)n, sizeof(int));
    int *off = calloc((size_t)n, sizeof(int));
    if (!ids || !len || !off) { free(ids); free(len); free(off); return e2_fail("oom"); }
    int total = 0;
    for (int i = 0; i < n; i++) {
        int cap = (int)strlen(texts[i]) * 2 + 16;
        if (cap < 16) cap = 16;
        ids[i] = malloc((size_t)cap * sizeof(int));
        if (!ids[i]) goto oom;
        int l = tokenize_text(ctx, texts[i], ids[i], cap);
        if (l < 0) goto oom;
        len[i] = l;
        off[i] = total;
        total += l;
    }
    {
        float *emb = malloc((size_t)total * H * sizeof(float));
        float *pooled = malloc((size_t)n * E2_EMBED_DIM * sizeof(float));
        if (!emb || !pooled) { free(emb); free(pooled); goto oom; }
        for (int i = 0; i < n; i++)
            e2_text_embed_ids(ctx, ids[i], len[i], emb + (size_t)off[i] * H);
        int rc = e2_text_forward_multi(ctx, emb, total, off, len, n, pooled);
        free(emb);
        if (rc != 0) { free(pooled); goto oom; }
        for (int i = 0; i < n; i++) {
            e2_normalize(pooled + (size_t)i * E2_EMBED_DIM, d);
            memcpy(out + (size_t)i * d, pooled + (size_t)i * E2_EMBED_DIM,
                   (size_t)d * sizeof(float));
        }
        free(pooled);
    }
    for (int i = 0; i < n; i++) free(ids[i]);
    free(ids); free(len); free(off);
    ctx->last_ms = e2_now_ms() - t0;
    return 0;
oom:
    for (int i = 0; i < n; i++) free(ids[i]);
    free(ids); free(len); free(off);
    if (strcmp(e2_error(), "ok") == 0) e2_fail("out of memory");
    return -1;
}

int e2_embed_text_task(e2_ctx *ctx, const char *task, const char *text, float *out, int dim) {
    const char *pre = task_prefix(task);
    if (!pre) return e2_fail("unknown task: %s", task);
    size_t pl = strlen(pre), tl = strlen(text);
    char *s = malloc(pl + tl + 1);
    if (!s) return e2_fail("out of memory");
    memcpy(s, pre, pl);
    memcpy(s + pl, text, tl + 1);
    int rc = e2_embed_text(ctx, s, out, dim);
    free(s);
    return rc;
}
