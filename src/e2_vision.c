/* e2_vision.c - Gemma 4 vision tower (16 bidirectional layers + pooler). */
#include "e2_internal.h"
#include "e2_gpu.h"

#include <math.h>
#include <stdlib.h>

void e2_image_free_rgb(uint8_t *p) { free(p); }

int e2_image_prepare(e2_ctx *ctx, const uint8_t *src, int w, int h, int max_soft,
                     float **patches_out, int **pos_out, int *npatches_out,
                     int *pw_out, int *ph_out);
int e2_image_load_rgb(const char *path, uint8_t **rgb, int *w, int *h);

static e2_ctx *g_vision_ctx;

static void vdump(const char *name, const float *p, size_t n) {
    const char *dir = getenv("E2_DUMP");
    if (!dir) return;
    e2_gpu_flush(g_vision_ctx);          /* the device may still owe this buffer */
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, sizeof(float), n, f);
    fclose(f);
}


int e2_vision_forward(e2_ctx *ctx, const uint8_t *rgb, int w, int h,
                      int max_soft, float *out, int *out_tokens) {
    e2_model *m = &ctx->m;
    e2_meta *c = &m->meta;
    if (!(c->flags & E2_MOD_IMAGE)) return e2_fail("model has no vision tower");

    g_vision_ctx = ctx;
    e2_arena_reset(&ctx->scratch);
    float *pat = NULL;
    int *pos = NULL, P = 0, pw = 0, ph = 0;
    if (e2_image_prepare(ctx, rgb, w, h, max_soft, &pat, &pos, &P, &pw, &ph) != 0) return -1;

    const int H = c->vision_hidden, I = c->vision_inter, D = c->vision_head_dim;
    const int heads = c->vision_heads;
    float *x = e2_arena_alloc(&ctx->scratch, (size_t)P * H * sizeof(float));
    float *hb = e2_arena_alloc(&ctx->scratch, (size_t)P * H * sizeof(float));
    float *q = e2_arena_alloc(&ctx->scratch, (size_t)P * heads * D * sizeof(float));
    float *k = e2_arena_alloc(&ctx->scratch, (size_t)P * heads * D * sizeof(float));
    float *v = e2_arena_alloc(&ctx->scratch, (size_t)P * heads * D * sizeof(float));
    float *att = e2_arena_alloc(&ctx->scratch, (size_t)P * heads * D * sizeof(float));
    float *tmp = e2_arena_alloc(&ctx->scratch, (size_t)P * I * sizeof(float));
    float *up = e2_arena_alloc(&ctx->scratch, (size_t)P * I * sizeof(float));
    if (!x || !hb || !q || !k || !v || !att || !tmp || !up) return e2_fail("oom");

    e2_linear lpatch = {0}; lpatch.w = m->patch_proj;
    e2_linear_forward(ctx, x, pat, P, &lpatch);

    /* add 2-D position embeddings: table[0][x] + table[1][y] */
    const float *tab = m->pos_table;
    size_t stride = (size_t)c->vision_pos_table * H;
    if (e2_gpu_op_pos_add(ctx, x, pos, tab, P, H, (long)stride) != 0) {
        for (int p = 0; p < P; p++) {
            int px = pos[p * 2], py = pos[p * 2 + 1];
            const float *ex = tab + (size_t)px * H;
            const float *ey = tab + stride + (size_t)py * H;
            float *xp = x + (size_t)p * H;
            for (int i = 0; i < H; i++) xp[i] += ex[i] + ey[i];
        }
    }

    vdump("vis_in", x, (size_t)P * H);
    int prof = getenv("E2_PROFILE") != NULL;
    double tp[4] = {0};
    for (int li = 0; li < c->vision_layers; li++) {
        double _t0 = e2_now_ms();
        e2_vision_layer *L = &m->vl[li];
        double _tq = e2_now_ms();
        e2_rmsnorm(ctx, hb, x, L->in_norm, P, H, c->vision_eps);
        {
            const e2_linear *ls[3] = {&L->q, &L->k, &L->v};
            float *ys[3] = {q, k, v};
            e2_linear_forward_multi(ctx, hb, P, ls, ys, 3);
        }
        e2_rmsnorm(ctx, q, q, L->q_norm, P * heads, D, c->vision_eps);
        e2_rmsnorm(ctx, k, k, L->k_norm, P * heads, D, c->vision_eps);
        e2_rmsnorm(ctx, v, v, NULL, P * heads, D, c->vision_eps);
        if (prof) tp[2] += e2_now_ms() - _tq;
        double _ta = e2_now_ms();
        e2_rope_vision(ctx, q, k, P, heads, D, pos, c->vision_rope_theta);
        e2_attention_vision(ctx, att, q, k, v, P, heads, D);
        if (prof) tp[3] += e2_now_ms() - _ta;
        if (prof) tp[0] += e2_now_ms() - _t0;

        e2_linear_forward(ctx, tmp, att, P, &L->o);
        e2_rmsnorm_add(ctx, x, tmp, L->post_attn_norm, P, H, c->vision_eps);

        e2_rmsnorm(ctx, hb, x, L->pre_ff_norm, P, H, c->vision_eps);
        /* device path: gate, up, gelu*up and the down projection without the
         * f32 intermediate (29 MB per layer at 1200 patches) */
        if (e2_gpu_op_mlp(ctx, hb, hb, P, &L->gate, &L->up, &L->down) != 0) {
        {
            const e2_linear *ls[2] = {&L->gate, &L->up};
            float *ys[2] = {tmp, up};
            e2_linear_forward_multi(ctx, hb, P, ls, ys, 2);
        }
        {
            float *gs = NULL;
            uint8_t *gq = e2_quant_scratch(ctx, P, I, &gs);
            if (gq) {
                e2_gelu_mul_q(ctx, tmp, up, P, I, gq, gs, 128);
                e2_linear_forward_pq(ctx, hb, gq, gs, P, &L->down);
            } else {
                e2_gelu_mul(ctx, tmp, up, P * I);
                e2_linear_forward(ctx, hb, tmp, P, &L->down);
            }
        }
        }
        e2_rmsnorm_add(ctx, x, hb, L->post_ff_norm, P, H, c->vision_eps);
        if (prof) tp[1] += e2_now_ms() - _t0;
        {
            char nm[32]; snprintf(nm, sizeof(nm), "vis_l%d", li + 1);
            vdump(nm, x, (size_t)P * H);
        }
    }

    /* spatial pool: bins of pool x pool patches, each divided by pool^2
     * (matches the reference, including partial edge bins). */
    int pool = c->vision_pool;
    int maxx = 0, maxy = 0;
    for (int p = 0; p < P; p++) {
        if (pos[p * 2] > maxx) maxx = pos[p * 2];
        if (pos[p * 2 + 1] > maxy) maxy = pos[p * 2 + 1];
    }
    int bx = (maxx + 1) / pool;
    int by = maxy / pool + 1;
    int nt = bx * by;
    vdump("vis_out", x, (size_t)P * H);
    if (prof) fprintf(stderr, "vision: qkv %.0f ms attn %.0f ms  total-layer %.0f ms\n", tp[2], tp[3], tp[1]);
    if (e2_gpu_flush(ctx) != 0) return e2_fail("device forward failed");
    float *pooled = e2_arena_alloc(&ctx->scratch, (size_t)nt * H * sizeof(float));
    if (!pooled) return e2_fail("oom");
    memset(pooled, 0, (size_t)nt * H * sizeof(float));
    for (int p = 0; p < P; p++) {
        int b = pos[p * 2] / pool + bx * (pos[p * 2 + 1] / pool);
        const float *xp = x + (size_t)p * H;
        float *op = pooled + (size_t)b * H;
        for (int i = 0; i < H; i++) op[i] += xp[i];
    }
    float inv = 1.0f / (float)(pool * pool);
    float scale = c->vision_pool_scale;
    for (size_t i = 0; i < (size_t)nt * H; i++) pooled[i] *= inv * scale;

    vdump("vis_pool", pooled, (size_t)nt * H);
    /* project to text space */
    e2_rmsnorm(ctx, pooled, pooled, NULL, nt, H, c->vision_eps);
    float *soft = e2_arena_alloc(&ctx->scratch, (size_t)nt * c->text_hidden * sizeof(float));
    if (!soft) return e2_fail("oom");
    e2_linear le = {0}; le.w = m->vis_embed;
    e2_linear_forward(ctx, soft, pooled, nt, &le);
    if (e2_gpu_flush(ctx) != 0) return e2_fail("device forward failed");
    vdump("image_soft", soft, (size_t)nt * c->text_hidden);
    memcpy(out, soft, (size_t)nt * c->text_hidden * sizeof(float));
    *out_tokens = nt;
    return 0;
}
