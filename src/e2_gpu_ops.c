/* e2_gpu_ops.c - the device-resident op layer.
 *
 * Every op the text/vision towers call has a device twin here.  The twin
 * *appends* a kernel to one command list instead of computing, so a whole
 * forward becomes a single submission with a single host sync.  That is what
 * round 7 showed was missing: the CPU-driven pipeline paid 0.38 ms of dispatch
 * + sync per GEMM and there are ~200 of them per forward.
 *
 * Rules:
 *  - an op returns 0 when it enqueued work, nonzero when the caller must run
 *    the CPU implementation.  Before returning nonzero it flushes, because the
 *    CPU is about to read buffers the device may still be writing.
 *  - every buffer a kernel touches must be shared USM: arena buffers are (the
 *    arena is switched to USM when the device comes up), model tensors are
 *    mirrored on first use, and the caller's output buffers are only ever
 *    written by the CPU.
 *  - the maths is a copy of src/e2_ops.c down to the constants; that file is
 *    the reference and tests/e2_test.c compares the two paths.
 */
#include "e2_gpu_int.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Every op that gives up must flush first: the CPU implementation is about to
 * read buffers the device may still be writing. */
#define E2_GPUBAD (-1)
#define OP_FAIL (e2_gpu_flush(ctx), -1)

/* E2_GPU_SKIP=attn,gelu,... keeps one op group on the CPU.  Debug aid: it
 * flushes at every skipped call, so use it to attribute time, not to measure. */
enum { SK_LINEAR = 1, SK_NORM = 2, SK_ROPE = 4, SK_ATTN = 8, SK_GELU = 16,
       SK_EW = 32 };
static unsigned skip_mask(e2_gpu *g) {
    if (g->skip) return g->skip;
    g->skip = 1u << 31;                  /* "parsed, nothing skipped" */
    const char *e = getenv("E2_GPU_SKIP");
    if (!e) return g->skip;
    unsigned m = 0;
    if (strstr(e, "linear")) m |= SK_LINEAR;
    if (strstr(e, "norm"))   m |= SK_NORM;
    if (strstr(e, "rope"))   m |= SK_ROPE;
    if (strstr(e, "attn"))   m |= SK_ATTN;
    if (strstr(e, "gelu"))   m |= SK_GELU;
    if (strstr(e, "ew"))     m |= SK_EW;
    g->skip = m ? m : 1u << 31;
    return g->skip;
}
#define SKIPPED(mask) (skip_mask(g) & (mask))

/* ------------------------------------------------------------------ */
/* batching                                                            */
/* ------------------------------------------------------------------ */

int e2_gpu_on(const e2_ctx *ctx) {
    const e2_gpu *g = ctx ? ctx->gpu : NULL;
    return g && g->ready && g->defer >= 0 && !g->no_dev;
}

void e2_gpu_no_dev(e2_ctx *ctx, int on) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (!g) return;
    if (on && !g->no_dev) e2_gpu_flush(ctx);
    g->no_dev = on;
}

/* Close the list, submit once, wait once.  Called before any CPU code reads a
 * buffer the device produced, and at the end of every public entry point. */
int e2_gpu_flush(e2_ctx *ctx) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (!g || !g->ready || !g->defer) return 0;
    g->defer = 0;
    if (!g->ndisp) return 0;
    int bad = 0, nd = g->ndisp;
    g->ndisp = 0;
    if (zeCommandListClose(g->list) != ZE_RESULT_SUCCESS) bad = 1;
    if (!bad && zeCommandQueueExecuteCommandLists(g->queue, 1, &g->list, NULL)
        != ZE_RESULT_SUCCESS) bad = 1;
    ze_result_t r = bad ? ZE_RESULT_ERROR_UNKNOWN
                        : zeCommandQueueSynchronize(g->queue, UINT64_MAX);
    zeCommandListReset(g->list);
    e2_arena_reset(&g->act);
    if (getenv("E2_GPU_TS")) e2_gpu_ts_report(g);
    if (g->ts) g->ts_n = 0;
    if (getenv("E2_GPU_PROF"))
        fprintf(stderr, "gpu flush: %d kernels submit=%s\n", nd,
                r == ZE_RESULT_SUCCESS ? "ok" : "FAIL");
    return r == ZE_RESULT_SUCCESS ? 0 : E2_GPUBAD;
}

#define enq_begin(g) e2_gpu_enq_open(g)

static uint32_t e2_up(uint32_t n, uint32_t m) { return (n + m - 1) / m * m; }

#define LAUNCH(g, k, threads, local, name) \
    (e2_gpu_launch_named(g, k, e2_up(threads, local) / local, name) == 0 \
         ? 0 : E2_GPUBAD)

/* ------------------------------------------------------------------ */
/* ops                                                                 */
/* ------------------------------------------------------------------ */

/* mode 0: out = norm(x)*w, 1: out = norm(x*pre)*w, 2: out += norm(src)*w */
int e2_gpu_op_rmsnorm(e2_ctx *ctx, float *out, const float *x, const float *w,
                      const float *src, int rows, int dim, float eps, float pre,
                      int mode) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && g->ready && SKIPPED(SK_NORM)) return OP_FAIL;
    if (!g || !g->ready || rows <= 0 || dim <= 0) return OP_FAIL;
    const float *wu = w ? e2_gpu_mirror(g, w, (size_t)dim * sizeof(float)) : NULL;
    if (w && !wu) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(wu), &wu) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(src), &src) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(out), &out) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(int), &rows) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(int), &dim) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(float), &eps) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(float), &pre) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rmsnorm, ai++, sizeof(int), &mode) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_rmsnorm, (uint32_t)rows * 32u, 64, "rmsnorm");
}

int e2_gpu_op_norm_rope(e2_ctx *ctx, float *x, const float *w, int rows, int dim,
                        float eps, const e2_rope_tab *tb, int per_pos,
                        const int *seq_off, int nseq) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && g->ready && SKIPPED(SK_ROPE)) return OP_FAIL;
    if (!g || !g->ready || rows <= 0 || !tb || !tb->c) return OP_FAIL;
    int cap = tb->cap;
    size_t n = (size_t)cap * (dim / 2) * sizeof(float);
    const float *cs = e2_gpu_mirror(g, tb->c, n);
    const float *sn = e2_gpu_mirror(g, tb->s, n);
    const float *wu = w ? e2_gpu_mirror(g, w, (size_t)dim * sizeof(float)) : NULL;
    const int *so = e2_gpu_mirror(g, seq_off, (size_t)nseq * sizeof(int));
    if (!cs || !sn || (w && !wu) || !so) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(wu), &wu) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(cs), &cs) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(sn), &sn) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(int), &rows) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(int), &dim) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(float), &eps) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(int), &per_pos) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(so), &so) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_norm_rope, ai++, sizeof(int), &nseq) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_norm_rope, (uint32_t)rows * 32u, 64, "norm_rope");
}

int e2_gpu_op_rope_text(e2_ctx *ctx, float *q, float *k, int t, int heads,
                        int kv_heads, int head_dim, int pos_offset, float theta) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (!g || !g->ready || t <= 0) return OP_FAIL;
    int hf = head_dim / 2;
    e2_rope_tab *tb = e2_rope_get(ctx, theta, hf, t + pos_offset);
    if (!tb) return OP_FAIL;
    size_t n = (size_t)tb->cap * hf * sizeof(float);
    const float *cs = e2_gpu_mirror(g, tb->c, n);
    const float *sn = e2_gpu_mirror(g, tb->s, n);
    if (!cs || !sn) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(q), &q) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(k), &k) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(cs), &cs) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(sn), &sn) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(int), &t) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(int), &heads) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(int), &kv_heads) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(int), &head_dim) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_text, ai++, sizeof(int), &pos_offset) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_rope_text, (uint32_t)((long)t * (heads + kv_heads) * hf), 64, "rope_text");
}

int e2_gpu_op_rope_vision(e2_ctx *ctx, float *q, float *k, int t, int heads,
                          int head_dim, const int *xy, float theta) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (!g || !g->ready || t <= 0) return OP_FAIL;
    int spatial = head_dim / 2, hf = spatial / 2;
    int mh = 0, mw = 0;
    for (int i = 0; i < t; i++) {
        if (xy[2 * i] > mh) mh = xy[2 * i];
        if (xy[2 * i + 1] > mw) mw = xy[2 * i + 1];
    }
    e2_rope_tab *tx = e2_rope_get_axis(ctx, 0, theta, hf, mh + 1);
    e2_rope_tab *ty = e2_rope_get_axis(ctx, 1, theta, hf, mw + 1);
    if (!tx || !ty) return OP_FAIL;
    size_t nx = (size_t)tx->cap * hf * sizeof(float);
    size_t ny = (size_t)ty->cap * hf * sizeof(float);
    const float *cx = e2_gpu_mirror(g, tx->c, nx);
    const float *sx = e2_gpu_mirror(g, tx->s, nx);
    const float *cy = e2_gpu_mirror(g, ty->c, ny);
    const float *sy = e2_gpu_mirror(g, ty->s, ny);
    const int *xyu = e2_gpu_mirror(g, xy, (size_t)t * 2 * sizeof(int));
    if (!cx || !sx || !cy || !sy || !xyu) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(q), &q) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(k), &k) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(cx), &cx) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(sx), &sx) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(cy), &cy) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(sy), &sy) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(xyu), &xyu) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(int), &t) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(int), &heads) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_rope_vision, ai++, sizeof(int), &head_dim) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_rope_vision, (uint32_t)((long)t * heads * hf), 64, "rope_vision");
}

/* int8 attention on the matrix engine (see e2_gpu_enq_attn).  A head dim with
 * no 128-wide layout, or a batch too large to score as one rectangle, stays on
 * the CPU int8 path. */
int e2_gpu_op_attn(e2_ctx *ctx, float *out, const float *q, const float *k,
                   const float *v, int t, int heads, int kv_heads, int head_dim,
                   int window, int nseq, const int *seq_off) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && g->ready && SKIPPED(SK_ATTN)) return OP_FAIL;
    if (!g || !g->ready) return OP_FAIL;
    if (e2_gpu_enq_attn(g, out, q, k, v, t, heads, kv_heads, head_dim, window,
                        nseq, seq_off) == 0)
        return 0;
    e2_gpu_flush(ctx);
    return OP_FAIL;
}

int e2_gpu_op_gelu_mul_q(e2_ctx *ctx, const float *x, const float *up, int rows,
                         int k, uint8_t *q, float *sc, int blk) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && g->ready && SKIPPED(SK_GELU)) return OP_FAIL;
    if (!g || !g->ready || rows <= 0 || (k % blk)) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_gelu_mul_q, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_gelu_mul_q, ai++, sizeof(up), &up) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_gelu_mul_q, ai++, sizeof(q), &q) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_gelu_mul_q, ai++, sizeof(sc), &sc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_gelu_mul_q, ai++, sizeof(int), &rows) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_gelu_mul_q, ai++, sizeof(int), &k) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_gelu_mul_q, ai++, sizeof(int), &blk) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_gelu_mul_q, (uint32_t)((long)rows * (k / blk) * 32), 64, "gelu_mul_q");
}

int e2_gpu_op_gelu_tanh(e2_ctx *ctx, float *x, int n) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && g->ready && SKIPPED(SK_GELU)) return OP_FAIL;
    if (!g || !g->ready || n <= 0) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_gelu_tanh, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_gelu_tanh, ai++, sizeof(int), &n) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_gelu_tanh, (uint32_t)n, 64, "gelu_tanh");
}

int e2_gpu_op_scale(e2_ctx *ctx, float *x, int n, float s) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && g->ready && SKIPPED(SK_EW)) return OP_FAIL;
    if (!g || !g->ready || n <= 0) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_scale, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_scale, ai++, sizeof(int), &n) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_scale, ai++, sizeof(float), &s) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_scale, (uint32_t)n, 64, "scale");
}

int e2_gpu_op_mul_slice(e2_ctx *ctx, float *h, const float *ple, int n, int L,
                        int P, int li) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && g->ready && SKIPPED(SK_EW)) return OP_FAIL;
    if (!g || !g->ready || n <= 0) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_mul_slice, ai++, sizeof(h), &h) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mul_slice, ai++, sizeof(ple), &ple) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mul_slice, ai++, sizeof(int), &n) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mul_slice, ai++, sizeof(int), &L) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mul_slice, ai++, sizeof(int), &P) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mul_slice, ai++, sizeof(int), &li) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_mul_slice, (uint32_t)n, 64, "mul_slice");
}

int e2_gpu_op_pos_add(e2_ctx *ctx, float *x, const int *xy, const float *tab,
                      int P, int H, long stride) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (!g || !g->ready || P <= 0) return OP_FAIL;
    const float *tu = e2_gpu_mirror(g, tab, (size_t)(2 * stride) * sizeof(float));
    const int *xu = e2_gpu_mirror(g, xy, (size_t)P * 2 * sizeof(int));
    if (!tu || !xu) return OP_FAIL;
    if (enq_begin(g) != 0) return OP_FAIL;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_pos_add, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pos_add, ai++, sizeof(xu), &xu) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pos_add, ai++, sizeof(tu), &tu) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pos_add, ai++, sizeof(int), &P) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pos_add, ai++, sizeof(int), &H) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pos_add, ai++, sizeof(long), &stride) != ZE_RESULT_SUCCESS)
        return OP_FAIL;
    return LAUNCH(g, g->k_pos_add, (uint32_t)((long)P * H), 64, "pos_add");
}

/* ---- GEMM wrappers: the CPU fallback needs the real values, so they flush. -- */

int e2_gpu_op_linear(e2_ctx *ctx, float *y, const float *x, int m, const e2_linear *l) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && SKIPPED(SK_LINEAR)) return OP_FAIL;
    if (!g) return OP_FAIL;
    if (e2_gpu_enq_linear(g, y, x, m, l) == 0) return 0;
    e2_gpu_flush(ctx);
    return OP_FAIL;
}

/* Same GEMM, activation already quantised on the device by a fused producer. */
int e2_gpu_op_linear_pq(e2_ctx *ctx, float *y, const uint8_t *xq, const float *xs,
                        int m, const e2_linear *l) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (g && SKIPPED(SK_LINEAR)) return OP_FAIL;
    if (!g || !g->ready) return OP_FAIL;
    if (e2_gpu_enq_linear_pq(g, y, (const int8_t *)xq, xs, m, l) == 0) return 0;
    e2_gpu_flush(ctx);
    return OP_FAIL;
}

/* Linear list sharing one activation block (q/k/v or gate/up): quantise once. */
int e2_gpu_op_linear_multi(e2_ctx *ctx, const float *x, int m,
                           const e2_linear *const *ls, float *const *ys, int n) {
    e2_gpu *g = e2_gpu_on(ctx) ? ctx->gpu : NULL;
    if (!g || !n) return OP_FAIL;
    if (e2_gpu_enq_linear(g, ys[0], x, m, ls[0]) != 0) { e2_gpu_flush(ctx); return OP_FAIL; }
    for (int i = 1; i < n; i++)
        if (e2_gpu_enq_linear(g, ys[i], x, m, ls[i]) != 0) return OP_FAIL;
    return 0;
}
