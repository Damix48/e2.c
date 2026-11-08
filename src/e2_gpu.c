/* e2_gpu.c - Level Zero device state, weight residency and the packed-int8
 * matrix-engine GEMM.
 *
 * Opt-in only (E2_DEVICE=gpu or e2_set_device(ctx, E2_DEVICE_GPU)).  The CPU
 * path stays the default and the reference; every entry point returns nonzero
 * on failure so the caller can fall back.
 *
 * The pipeline is device-resident: e2_gpu_ops.c appends a whole forward to one
 * command list and there is exactly one submission and one host sync per
 * embedding.  Round 7 measured 0.38 ms of dispatch+sync latency per GEMM on
 * this driver, so the previous one-submission-per-GEMM shape could never beat
 * the CPU path no matter how fast the kernel was (see PERFORMANCE.md).
 */
#include "e2_gpu_int.h"
#include "e2_gpu_pack.h"
#include "e2_gpu_spv.h"

#include <stdlib.h>
#include <string.h>

int e2_gpu_launch_3d(e2_gpu *g, ze_kernel_handle_t k, uint32_t gx, uint32_t gy,
                     uint32_t gz, const char *name) {
    ze_group_count_t gc = {gx ? gx : 1, gy ? gy : 1, gz ? gz : 1};
    /* E2_GPU_NOLAUNCH=1 keeps only the GEMMs (results are wrong, only the clock
     * is meaningful): it prices everything else against them in one run. */
    static int nol;
    if (!nol && getenv("E2_GPU_NOLAUNCH") &&
        k != g->k_gemm_z && k != g->k_gemm_b2) {
        g->ndisp++;
        return 0;
    }
    if (!nol && getenv("E2_GPU_NOLAUNCH")) nol = 1;
    if (g->ts_on && zeCommandListAppendWriteGlobalTimestamp &&
        zeCommandListAppendWriteGlobalTimestamp(g->list, &g->ts[g->ts_n], NULL, 0,
                                                NULL) == ZE_RESULT_SUCCESS) {
        if (g->ts_n < 2047) g->ts_name[g->ts_n] = name;
        g->ts_n++;
    }
    if (zeCommandListAppendLaunchKernel(g->list, k, &gc, NULL, 0, NULL)
        != ZE_RESULT_SUCCESS)
        return -1;
    if (g->barrier && zeCommandListAppendBarrier(g->list, NULL, 0, NULL)
        != ZE_RESULT_SUCCESS)
        return -1;
    g->ndisp++;
    return 0;
}

int e2_gpu_launch_named(e2_gpu *g, ze_kernel_handle_t k, uint32_t groups,
                        const char *name) {
    return e2_gpu_launch_3d(g, k, groups, 1, 1, name);
}

void e2_gpu_ts_report(e2_gpu *g) {
    if (!g->ts_on || g->ts_n < 2) return;
    double ns_per_tick = g->ts_res ? g->ts_res : 1.0;
    fprintf(stderr, "--- per-kernel device time (%d samples, %.3f ns/tick)\n",
            g->ts_n - 1, ns_per_tick);
    for (int i = 1; i < g->ts_n && i < 2048; i++) {
        double dt = (double)(g->ts[i] - g->ts[i - 1]) * ns_per_tick;
        fprintf(stderr, "TS %10.3f us  %s\n", dt / 1000.0,
                g->ts_name[i - 1] ? g->ts_name[i - 1] : "?");
    }
}

int e2_gpu_enq_open(e2_gpu *g) {
    if (!g || !g->ready) return -1;
    if (!g->defer) { g->defer = 1; g->ndisp = 0; }
    return 0;
}

/* ------------------------------------------------------------------ */
/* memory                                                              */
/* ------------------------------------------------------------------ */

void *e2_usm_alloc(e2_gpu *g, size_t n) {
    if (!n) n = 64;
    ze_device_mem_alloc_desc_t ddesc;
    ze_host_mem_alloc_desc_t hdesc;
    memset(&ddesc, 0, sizeof(ddesc));
    memset(&hdesc, 0, sizeof(hdesc));
    ddesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void *p = NULL;
    if (zeMemAllocShared(g->ctx, &ddesc, &hdesc, n, 64, g->dev, &p) != ZE_RESULT_SUCCESS)
        return NULL;
    return p;
}

void *e2_usm_realloc(e2_gpu *g, void **p, size_t *cap, size_t need) {
    if (*cap >= need && *p) return *p;
    void *n = e2_usm_alloc(g, need);
    if (!n) return NULL;
    if (*p) zeMemFree(g->ctx, *p);
    *p = n;
    *cap = need;
    return n;
}

/* Arena block backend: shared USM while the device path is on, so every
 * activation buffer a kernel reads is device-addressable. */
void *e2_arena_usm_alloc(void *opaque, size_t n) {
    e2_ctx *ctx = opaque;
    return e2_dev_alloc(ctx, n);
}

void e2_arena_usm_free(void *opaque, void *p) {
    e2_dev_free((e2_ctx *)opaque, p);
}

void *e2_dev_alloc(e2_ctx *ctx, size_t n) {
    if (ctx && ctx->gpu && ((e2_gpu *)ctx->gpu)->ready)
        return e2_usm_alloc((e2_gpu *)ctx->gpu, n);
    return malloc(n);
}

void e2_dev_free(e2_ctx *ctx, void *p) {
    if (!p) return;
    if (ctx && ctx->gpu && ((e2_gpu *)ctx->gpu)->ready) {
        zeMemFree(((e2_gpu *)ctx->gpu)->ctx, p);
        return;
    }
    free(p);
}

static void usm_free(e2_gpu *g) {
    if (!g->ctx) return;
    for (int i = 0; i < g->wct.cap; i++) {
        e2_gpu_wc *e = &g->wct.tab[i];
        if (!e->q_src) continue;
        if (e->scale) zeMemFree(g->ctx, e->scale);
        if (e->pq)    zeMemFree(g->ctx, e->pq);
    }
    for (int i = 0; i < g->mir_cap; i++)
        if (g->mir[i].src && g->mir[i].usm) zeMemFree(g->ctx, g->mir[i].usm);
    free(g->mir);
    g->mir = NULL;
    g->mir_cap = g->mir_used = 0;
    e2_arena_free(&g->act);        /* USM blocks: needs the context, so first */
    free(g->wct.tab);
    memset(&g->wct, 0, sizeof(g->wct));
}

/* Model tensors that kernels read (norm weights, rope tables, position
 * tables) live in the mmap; copy each into shared USM once. */
static unsigned mir_hash(const void *p, int cap) {
    unsigned long long h = (unsigned long long)(uintptr_t)p;
    h = (h >> 5) * 0x9E3779B97F4A7C15ull;
    return (unsigned)(h >> 40) & (unsigned)(cap - 1);
}

static void mir_insert(e2_gpu_mir *tab, int cap, e2_gpu_mir e) {
    unsigned i = mir_hash(e.src, cap);
    while (tab[i].src) i = (i + 1) & (unsigned)(cap - 1);
    tab[i] = e;
}

const void *e2_gpu_mirror(e2_gpu *g, const void *p, size_t bytes) {
    if (!p) return NULL;
    if (!g->mir_cap) {
        g->mir_cap = 512;
        g->mir = calloc((size_t)g->mir_cap, sizeof(*g->mir));
        if (!g->mir) { g->mir_cap = 0; return NULL; }
    }
    unsigned i = mir_hash(p, g->mir_cap);
    for (;;) {
        e2_gpu_mir *e = &g->mir[i];
        if (!e->src) break;
        if (e->src == p) return e->usm;
        i = (i + 1) & (unsigned)(g->mir_cap - 1);
    }
    if (g->mir_used * 4 >= g->mir_cap * 3) {
        int ncap = g->mir_cap * 2;
        e2_gpu_mir *nt = calloc((size_t)ncap, sizeof(*nt));
        if (!nt) return NULL;
        for (int k = 0; k < g->mir_cap; k++)
            if (g->mir[k].src) mir_insert(nt, ncap, g->mir[k]);
        free(g->mir);
        g->mir = nt;
        g->mir_cap = ncap;
        i = mir_hash(p, ncap);
        while (g->mir[i].src) i = (i + 1) & (unsigned)(ncap - 1);
    }
    void *u = e2_usm_alloc(g, bytes);
    if (!u) return NULL;
    memcpy(u, p, bytes);
    g->mir[i].src = p;
    g->mir[i].usm = u;
    g->mir[i].bytes = bytes;
    g->mir_used++;
    return u;
}

/* ------------------------------------------------------------------ */
/* weight residency                                                    */
/* ------------------------------------------------------------------ */

static unsigned wc_hash(const void *p, int cap) {
    unsigned long long h = (unsigned long long)(uintptr_t)p;
    h = (h >> 4) * 0x9E3779B97F4A7C15ull;
    return (unsigned)(h >> 40) & (unsigned)(cap - 1);
}

static e2_gpu_wc *wc_slot(e2_gpu_wc *tab, const int8_t *key, int cap) {
    unsigned i = wc_hash(key, cap);
    for (;;) {
        e2_gpu_wc *e = &tab[i];
        if (!e->q_src || e->q_src == key) return e;
        i = (i + 1) & (unsigned)(cap - 1);
    }
}

static int wc_grow(e2_gpu *g) {
    int ncap = g->wct.cap ? g->wct.cap * 2 : 256;
    e2_gpu_wc *nt = calloc((size_t)ncap, sizeof(*nt));
    if (!nt) return -1;
    for (int i = 0; i < g->wct.cap; i++) {
        e2_gpu_wc *old = &g->wct.tab[i];
        if (!old->q_src) continue;
        *wc_slot(nt, old->q_src, ncap) = *old;
    }
    free(g->wct.tab);
    g->wct.tab = nt;
    g->wct.cap = ncap;
    return 0;
}

/* Copy one packed matrix into shared USM the first time it is used, repacked
 * into the matrix-engine layout (src/e2_gpu_pack.h).  Shared memory on an iGPU
 * is the same DRAM, so this is one CPU pass and no PCIe hop; after that the
 * kernel reads it at full bandwidth.  Matrices whose shape has no matrix-engine
 * layout are refused, so the caller falls back to the CPU kernel. */
e2_gpu_wc *e2_gpu_weight(e2_gpu *g, const e2_qmat *w) {
    int blk = w->blk ? w->blk : 128;
    int nblk = w->k / blk;

    if (g->wct.cap && (g->wct.used + 1) * 4 >= g->wct.cap * 3 && wc_grow(g) != 0)
        return NULL;
    if (!g->wct.cap && wc_grow(g) != 0) return NULL;
    e2_gpu_wc *e = wc_slot(g->wct.tab, w->q, g->wct.cap);
    if (e->q_src == w->q) return e;                 /* already resident */

    e->scale = (float *)e2_usm_alloc(g, (size_t)w->n * sizeof(float));
    e->pq = (int8_t *)e2_usm_alloc(g, (size_t)w->n * w->k);
    if (!e->scale || !e->pq) goto fail;
    memcpy(e->scale, w->scale, (size_t)w->n * sizeof(float));
    if (e2_gpu_dpas_pack(w, e->pq) != 0) goto fail;

    e->q_src = w->q;
    e->n = w->n; e->k = w->k; e->blk = blk; e->nblk = nblk;
    g->wct.used++;
    return e;
fail:
    if (e->scale) zeMemFree(g->ctx, e->scale);
    if (e->pq)    zeMemFree(g->ctx, e->pq);
    memset(e, 0, sizeof(*e));
    return NULL;
}

/* ------------------------------------------------------------------ */
/* GEMM                                                                */
/* ------------------------------------------------------------------ */

/* Append one packed-int8 GEMM for an already-quantised activation block.
 *
 * Grid: one work group per 64 tokens x 64 features (E2_TM/E2_TN in
 * e2_xmx.cl, whose four-by-four sub-group arrangement is what sets the
 * operand traffic).  The x dimension walks tokens so that consecutive work
 * groups share a weight tile and it stays in cache. */
int e2_gpu_gemm_add(e2_gpu *g, float *y, const int8_t *xq, const float *xsc,
                    int m, const e2_qmat *w) {
    e2_gpu_wc *wc = e2_gpu_weight(g, w);
    if (!wc || !wc->pq) return -1;
    return e2_gpu_enq_gemm_z(g, y, xq, xsc, wc->pq, wc->scale, m, wc->k, wc->n,
                             wc->k, wc->nblk, wc->n, 0, 0, 0, 0, 0, 1, 1, 0);
}

int e2_gpu_enq_gemm_z(e2_gpu *g, float *y, const int8_t *xq, const float *xsc,
                      const int8_t *wp, const float *wsc, int m, int k, int n,
                      int xstr, int xscstr, int ystr, long zxq, long zxsc,
                      long zwp, long zwsc, long zy, int zrep, int nd, int b2) {
    if (e2_gpu_enq_open(g) != 0) return -1;
    const int big = m >= E2_GPU_BIG_MIN;
    ze_kernel_handle_t kern = b2 ? (big ? g->k_bgemm_b2 : g->k_gemm_b2)
                                 : (big ? g->k_bgemm_z : g->k_gemm_z);
    const int tm = big ? E2_GPU_BIG_TM : E2_GPU_TM;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(kern, ai++, sizeof(xq), &xq) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(xsc), &xsc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(wp), &wp) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(wsc), &wsc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(y), &y) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &m) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &k) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &n) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &xstr) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &xscstr) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &ystr) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(long), &zxq) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(long), &zxsc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(long), &zwp) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(long), &zwsc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(long), &zy) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &zrep) != ZE_RESULT_SUCCESS)
        return -1;
    /* x walks tokens so consecutive work groups share a weight tile */
    return e2_gpu_launch_3d(g, kern, (uint32_t)((m + tm - 1) / tm),
                            (uint32_t)((n + E2_GPU_TN - 1) / E2_GPU_TN),
                            (uint32_t)nd, "gemm") == 0 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* attention                                                           */
/* ------------------------------------------------------------------ */

/* scores = Q K^T and context = P V^T are the same matrix-engine product as
 * the projections, so the only per-layer preprocessing is packing K and V^T
 * into the A operand layout and quantising Q.  The score matrix is
 * materialised: P has to be an int8 B operand anyway, and one row's scores
 * stay in cache while the softmax pass walks them.
 *
 * Every stage is one launch for all heads: the per-head operands differ only
 * by a constant stride, so k_gemm_z and the pack/softmax kernels index them
 * with gridCountZ, and zrep folds the GQA mapping in (zrep query heads share
 * one kv head).  That is 6 launches per layer instead of 6 per head. */
int e2_gpu_enq_attn(e2_gpu *g, float *out, const float *q, const float *k,
                    const float *v, int t, int heads, int kv_heads, int head_dim,
                    int window, int nseq, const int *seq_off) {
    if (t <= 0 || heads <= 0 || kv_heads <= 0 || (heads % kv_heads)) return -1;
    if (head_dim % 128 || head_dim > 512) return -1;
    /* The score product has no per-sequence band, so a batch pays for the whole
     * t x t rectangle.  Take that only while it is cheaper than the CPU int8
     * attention plus a flush per layer, which is what the fallback costs. */
    if (nseq > 1 && t > 512) return -1;
    const int *so = NULL;
    if (nseq > 1) {
        so = e2_gpu_mirror(g, seq_off, (size_t)nseq * sizeof(int));
        if (!so) return -1;
    }
    if (e2_gpu_enq_open(g) != 0) return -1;
    if (getenv("E2_GPU_NOATTN")) return 0;   /* profiling knob: price attention */

    const int nblk = head_dim / 128;      /* d blocks, Q row scales */
    const int rpad = (t + 127) & ~127;    /* key rows in the packed K / V^T */
    const int groups = heads / kv_heads;
    const int nkeyblk = rpad / 128;
    int8_t *q8 = e2_arena_alloc(&g->act, (size_t)t * heads * head_dim);
    float *qsc = e2_arena_alloc(&g->act, (size_t)t * heads * nblk * sizeof(float));
    int8_t *kpq = e2_arena_alloc(&g->act, (size_t)kv_heads * rpad * head_dim);
    float *ksc = e2_arena_alloc(&g->act, (size_t)kv_heads * rpad * sizeof(float));
    int8_t *vpq = e2_arena_alloc(&g->act, (size_t)kv_heads * rpad * head_dim);
    float *vsc = e2_arena_alloc(&g->act, (size_t)kv_heads * head_dim * sizeof(float));
    if (!q8 || !qsc || !kpq || !ksc || !vpq || !vsc) return -1;

    /* Q -> centred int8 + one scale per 128-wide d block, all (token, head) rows */
    {
        int qrows = t * heads, kk = head_dim, blk = 128;
        uint32_t ai = 0;
        float *sc = qsc;
        if (zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(q), &q) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(q8), &q8) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(sc), &sc) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(qrows), &qrows) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(kk), &kk) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(blk), &blk) != ZE_RESULT_SUCCESS)
            return -1;
        uint32_t wgs = (uint32_t)(((long)qrows * nblk * 32 + 63) / 64);
        if (e2_gpu_launch_named(g, g->k_quant_s8, wgs, "quant_s8") != 0) return -1;
    }

    /* K and V^T, one launch each for all kv heads */
    {
        int rp = rpad, rv = t, kp = head_dim, kv = head_dim;
        int rstride = kv_heads * head_dim, kstride = 1;
        long zsrc = head_dim, zpq = (long)rpad * head_dim, zsc = rpad;
        uint32_t ai = 0;
        if (zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(k), &k) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(kpq), &kpq) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(ksc), &ksc) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &rp) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &rv) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &kp) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &kv) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &rstride) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &kstride) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(long), &zsrc) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(long), &zpq) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(long), &zsc) != ZE_RESULT_SUCCESS)
            return -1;
        if (e2_gpu_launch_3d(g, g->k_attn_pack, (uint32_t)(rpad * 32 / 64), 1,
                             (uint32_t)kv_heads, "attn_pack_k") != 0)
            return -1;

        /* V^T: rows are the head's dims, k the keys */
        rp = head_dim; rv = head_dim; kp = rpad; kv = t;
        rstride = 1; kstride = kv_heads * head_dim;
        zsc = head_dim;
        ai = 0;
        if (zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(v), &v) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(vpq), &vpq) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(vsc), &vsc) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &rp) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &rv) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &kp) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &kv) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &rstride) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(int), &kstride) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(long), &zsrc) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(long), &zpq) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_attn_pack, ai++, sizeof(long), &zsc) != ZE_RESULT_SUCCESS)
            return -1;
        if (e2_gpu_launch_3d(g, g->k_attn_pack, (uint32_t)(head_dim * 32 / 64), 1,
                             (uint32_t)kv_heads, "attn_pack_v") != 0)
            return -1;
    }

    /* The scores are the only large temporary here: 2*t*rpad*4 bytes per head
     * (7.6 MB at 1200 patches).  Materialising all heads at once turns that
     * into 90 MB that no cache holds, so walk the heads in groups whose score
     * rectangle still fits L3.  A group is a multiple of `groups` to keep the
     * GQA stride uniform. */
    const size_t per_head = (size_t)t * rpad * (sizeof(float) + 1) +
                            (size_t)t * nkeyblk * sizeof(float);
    int hg = (int)(E2_GPU_ATT_BUDGET / per_head);
    if (hg < 1) hg = 1;
    hg -= hg % groups;
    if (hg < groups) hg = groups;
    if (hg > heads) hg = heads;
    for (int h0 = 0; h0 < heads; h0 += hg) {
        const int nh = h0 + hg > heads ? heads - h0 : hg;
        float *S = e2_arena_alloc(&g->act, (size_t)nh * t * rpad * sizeof(float));
        int8_t *P = e2_arena_alloc(&g->act, (size_t)nh * t * rpad);
        float *psc = e2_arena_alloc(&g->act,
                                    (size_t)nh * t * nkeyblk * sizeof(float));
        if (!S || !P || !psc) return -1;
        /* all heads of the group score in one launch */
        if (e2_gpu_enq_gemm_z(g, S, q8 + (size_t)h0 * head_dim,
                              qsc + (size_t)h0 * nblk,
                              kpq + (size_t)(h0 / groups) * rpad * head_dim,
                              ksc + (size_t)(h0 / groups) * rpad, t, head_dim,
                              rpad, heads * head_dim, heads * nblk, rpad,
                              head_dim, nblk, (long)rpad * head_dim, rpad,
                              (long)t * rpad, groups, nh, 0) != 0)
            return -1;
        {
            int rows = t, kpad = rpad, kvalid = t, win = window, nsq = nseq;
            const int *sep = so ? so : seq_off;
            long zS = (long)t * rpad, zP = (long)t * rpad, zsc = (long)t * nkeyblk;
            uint32_t ai = 0;
            if (zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(S), &S) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(P), &P) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(psc), &psc) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(sep), &sep) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(nsq), &nsq) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(rows), &rows) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(kpad), &kpad) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(kvalid), &kvalid) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(win), &win) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(long), &zS) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(long), &zP) != ZE_RESULT_SUCCESS ||
                zeKernelSetArgumentValue(g->k_attn_softmax, ai++, sizeof(long), &zsc) != ZE_RESULT_SUCCESS)
                return -1;
            if (e2_gpu_launch_3d(g, g->k_attn_softmax, (uint32_t)(t * 32 / 64), 1,
                                 (uint32_t)nh, "attn_softmax") != 0)
                return -1;
        }
        /* context, written straight into the [token][head][dim] output */
        if (e2_gpu_enq_gemm_z(g, out + (size_t)h0 * head_dim, P, psc,
                              vpq + (size_t)(h0 / groups) * rpad * head_dim,
                              vsc + (size_t)(h0 / groups) * head_dim, t, rpad,
                              head_dim, rpad, nkeyblk, heads * head_dim,
                              (long)t * rpad, (long)t * nkeyblk,
                              (long)rpad * head_dim, head_dim, head_dim,
                              groups, nh, 0) != 0)
            return -1;
    }
    return 0;
}

int e2_gpu_enq_mlp(e2_gpu *g, float *y, const float *x, int m,
                   const e2_linear *gate, const e2_linear *up,
                   const e2_linear *down) {
    const e2_qmat *wg = &gate->w, *wu = &up->w, *wd = &down->w;
    if (!wg->packed || !wu->packed || !wd->packed) return -1;
    const int k = wg->k, n = wg->n;
    if (wu->k != k || wu->n != n || wd->k != n) return -1;
    if ((k % 128) || (n % 64) || (wd->k % 64)) return -1;
    if (gate->in_min || gate->in_max || gate->out_min || gate->out_max || wg->bias ||
        up->in_min || up->in_max || up->out_min || up->out_max || wu->bias ||
        down->in_min || down->in_max || down->out_min || down->out_max || wd->bias)
        return -1;
    e2_gpu_wc *cg = e2_gpu_weight(g, wg), *cu = e2_gpu_weight(g, wu),
              *cd = e2_gpu_weight(g, wd);
    if (!cg || !cu || !cd || !cg->pq || !cu->pq || !cd->pq) return -1;
    e2_act a;
    if (e2_gpu_pack_act(g, x, m, k, &a) != 0) return -1;

    const int nblk = n / 64;                 /* 64-wide: one column tile */
    const int inblk = k / 128;               /* the input's scale stride */
    int8_t *q = e2_arena_alloc(&g->act, (size_t)m * n);
    float *sc = e2_arena_alloc(&g->act, (size_t)m * nblk * sizeof(float));
    if (!q || !sc) return -1;
    if (e2_gpu_enq_open(g) != 0) return -1;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(a.q), &a.q) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(a.scale), &a.scale) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(cg->pq), &cg->pq) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(cg->scale), &cg->scale) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(cu->pq), &cu->pq) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(cu->scale), &cu->scale) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(q), &q) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(sc), &sc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(int), &m) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(int), &k) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(int), &n) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(int), &k) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_mlp_gu, ai++, sizeof(int), &inblk) != ZE_RESULT_SUCCESS)
        return -1;
    const int big = m >= E2_GPU_BIG_MIN;
    const int tm = big ? E2_GPU_BIG_TM : E2_GPU_TM;
    if (e2_gpu_launch_3d(g, big ? g->k_bmlp_gu : g->k_mlp_gu,
                         (uint32_t)((m + tm - 1) / tm),
                         (uint32_t)((n + E2_GPU_TN - 1) / E2_GPU_TN), 1,
                         "mlp_gu") != 0)
        return -1;
    return e2_gpu_enq_gemm_z(g, y, q, sc, cd->pq, cd->scale, m, cd->k, cd->n,
                             cd->k, nblk, cd->n, 0, 0, 0, 0, 0, 1, 1, 1);
}

int e2_gpu_enq_act(e2_gpu *g, float *y, int m, const e2_linear *l, const e2_act *a) {
    const e2_qmat *w = &l->w;
    if (m <= 0 || !w->packed || (w->k % 128) || (w->n % 8) || (w->k % 32)) return -1;
    if (l->in_min || l->in_max || l->out_min || l->out_max || w->bias) return -1;
    return e2_gpu_gemm_add(g, y, a->q, a->scale, m, w);
}

/* Arena blocks for the packed activations are shared USM: kernels read them. */
static void *act_usm_alloc(void *opaque, size_t n) {
    return e2_usm_alloc((e2_gpu *)opaque, n);
}

static void act_usm_free(void *opaque, void *p) {
    e2_gpu *g = opaque;
    if (g->ctx && p) zeMemFree(g->ctx, p);
}

/* f32 rows -> centred int8 rows + one scale per 128-wide block. */
int e2_gpu_pack_act(e2_gpu *g, const float *x, int m, int k, e2_act *a) {
    int nblk = k / 128;
    a->q = e2_arena_alloc(&g->act, (size_t)m * k);
    a->scale = e2_arena_alloc(&g->act, (size_t)m * nblk * sizeof(float));
    if (!a->q || !a->scale) return -1;
    if (e2_gpu_enq_open(g) != 0) return -1;
    int8_t *q = a->q;
    float *sc = a->scale;
    int rows = m, kk = k, blk = 128;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(q), &q) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(sc), &sc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(int), &rows) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(int), &kk) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_s8, ai++, sizeof(int), &blk) != ZE_RESULT_SUCCESS)
        return -1;
    uint32_t groups = (uint32_t)(((long)m * nblk * 32 + 63) / 64);
    return e2_gpu_launch_named(g, g->k_quant_s8, groups, "quant_s8") == 0 ? 0 : -1;
}

int e2_gpu_enq_linear(e2_gpu *g, float *y, const float *x, int m, const e2_linear *l) {
    if (g->ac.x == x && g->ac.m == m && g->ac.k == l->w.k) {
        e2_act a = { g->ac.q, g->ac.sc };
        return e2_gpu_enq_act(g, y, m, l, &a);
    }
    g->ac.x = NULL;                 /* another activation: drop the cache */
    e2_act a;
    if (e2_gpu_pack_act(g, x, m, l->w.k, &a) != 0) return -1;
    return e2_gpu_enq_act(g, y, m, l, &a);
}

int e2_gpu_enq_linear_pq(e2_gpu *g, float *y, const int8_t *xq, const float *xs,
                         int m, const e2_linear *l) {
    if (l->w.blk && l->w.blk != 128) return -1;
    e2_act a = { (int8_t *)xq, (float *)xs };
    return e2_gpu_enq_act(g, y, m, l, &a);
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* Build one tile variant of the matrix-engine module and create its kernels.
 * Both variants expose the same four entry points, so the launcher only swaps
 * the handles.  A missing variant is not fatal: the small one is required. */
static int xmx_module(e2_gpu *g, const unsigned char *spv, size_t len, int sg,
                      ze_module_handle_t *mod, ze_kernel_handle_t *k_probe,
                      ze_kernel_handle_t *k_z, ze_kernel_handle_t *k_b2,
                      ze_kernel_handle_t *k_mlp) {
    ze_module_desc_t md; memset(&md, 0, sizeof(md));
    md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    md.format = ZE_MODULE_FORMAT_IL_SPIRV;
    md.inputSize = len;
    md.pInputModule = spv;
    if (zeModuleCreate(g->ctx, g->dev, &md, mod, NULL) != ZE_RESULT_SUCCESS)
        return -1;
    static const char *xn[] = {"k_gemm_xmx", "k_gemm_z", "k_gemm_b2", "k_mlp_gu"};
    ze_kernel_handle_t *out[4] = {k_probe, k_z, k_b2, k_mlp};
    for (int i = 0; i < 4; i++) {
        ze_kernel_desc_t d; memset(&d, 0, sizeof(d));
        d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        d.pKernelName = xn[i];
        if (zeKernelCreate(*mod, &d, out[i]) != ZE_RESULT_SUCCESS ||
            zeKernelSetGroupSize(*out[i], 8, (uint32_t)sg, 1) != ZE_RESULT_SUCCESS)
            return -1;
    }
    return 0;
}

static void gpu_destroy(e2_gpu *g) {
    if (!g) return;
    usm_free(g);
    ze_kernel_handle_t ks[] = {g->k_gemm, g->k_gemm_z, g->k_gemm_b2, g->k_mlp_gu,
        g->k_bgemm_z, g->k_bgemm_b2, g->k_bmlp_gu,
        g->k_rmsnorm, g->k_rmsnorm_q, g->k_norm_rope, g->k_rope_text,
        g->k_rope_vision,
        g->k_quant_s8, g->k_gelu_mul_q, g->k_gelu_tanh, g->k_scale, g->k_mul_slice,
        g->k_pos_add, g->k_attn_pack, g->k_attn_softmax};
    for (size_t i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
        if (ks[i]) zeKernelDestroy(ks[i]);
    if (g->dmodule) zeModuleDestroy(g->dmodule);
    if (g->bmodule) zeModuleDestroy(g->bmodule);
    if (g->omodule) zeModuleDestroy(g->omodule);
    if (g->list)    zeCommandListDestroy(g->list);
    if (g->imm)     zeCommandListDestroy(g->imm);
    if (g->evt)     zeEventDestroy(g->evt);
    if (g->epool)   zeEventPoolDestroy(g->epool);
    if (g->queue)   zeCommandQueueDestroy(g->queue);
    if (g->ctx)     zeContextDestroy(g->ctx);
    memset(g, 0, sizeof(*g));
}

int e2_gpu_init(e2_ctx *ctx) {
    if (!ctx || ctx->gpu) return ctx && ctx->gpu ? 0 : -1;
    if (e2_gpu_l0_open() != 0) return -1;

    e2_gpu *g = calloc(1, sizeof(*g));
    if (!g) return -1;
    ctx->gpu = g;
    /* An in-order list orders execution but does not promise that one kernel's
     * writes reach the next one's view without a barrier; measure it. */
    g->barrier = getenv("E2_GPU_NOBARRIER") ? 0 : 1;

    if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) goto fail;

    uint32_t ndrv = 0;
    if (zeDriverGet(&ndrv, NULL) != ZE_RESULT_SUCCESS || !ndrv) goto fail;
    ze_driver_handle_t *drvs = calloc(ndrv, sizeof(*drvs));
    if (!drvs) goto fail;
    if (zeDriverGet(&ndrv, drvs) != ZE_RESULT_SUCCESS) { free(drvs); goto fail; }

    int found = 0;
    for (uint32_t d = 0; d < ndrv && !found; d++) {
        uint32_t ndev = 0;
        if (zeDeviceGet(drvs[d], &ndev, NULL) != ZE_RESULT_SUCCESS || !ndev) continue;
        ze_device_handle_t *devs = calloc(ndev, sizeof(*devs));
        if (!devs) continue;
        if (zeDeviceGet(drvs[d], &ndev, devs) != ZE_RESULT_SUCCESS) { free(devs); continue; }
        for (uint32_t e = 0; e < ndev; e++) {
            ze_device_properties_t props;
            memset(&props, 0, sizeof(props));
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (zeDeviceGetProperties(devs[e], &props) != ZE_RESULT_SUCCESS) continue;
            if (props.type == ZE_DEVICE_TYPE_GPU) {
                g->drv = drvs[d];
                g->dev = devs[e];
                found = 1;
                break;
            }
        }
        free(devs);
    }
    free(drvs);
    if (!found) goto fail;

    ze_context_desc_t cd; memset(&cd, 0, sizeof(cd));
    cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    if (zeContextCreate(g->drv, &cd, &g->ctx) != ZE_RESULT_SUCCESS) goto fail;

    /* One in-order queue + one deferred command list hold a whole forward. */
    ze_command_queue_desc_t qd; memset(&qd, 0, sizeof(qd));
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    if (zeCommandQueueCreate(g->ctx, g->dev, &qd, &g->queue) != ZE_RESULT_SUCCESS)
        goto fail;
    ze_command_list_desc_t ld; memset(&ld, 0, sizeof(ld));
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    if (zeCommandListCreate(g->ctx, g->dev, &ld, &g->list) != ZE_RESULT_SUCCESS)
        goto fail;

    if (xmx_module(g, e2_gpu_spv_e2_xmx, e2_gpu_spv_e2_xmx_len, E2_GPU_SG,
                   &g->dmodule, &g->k_gemm, &g->k_gemm_z, &g->k_gemm_b2,
                   &g->k_mlp_gu) != 0 ||
        xmx_module(g, e2_gpu_spv_e2_xmx_big, e2_gpu_spv_e2_xmx_big_len,
                   E2_GPU_BIG_SG, &g->bmodule, &g->k_gemm, &g->k_bgemm_z,
                   &g->k_bgemm_b2, &g->k_bmlp_gu) != 0)
        goto fail;                           /* no matrix engine -> CPU path */
    if (getenv("E2_GPU_PROF")) fprintf(stderr, "gpu xmx: gemm up\n");

    {
        static const char *on[] = {"k_rmsnorm", "k_rmsnorm_q", "k_norm_rope",
            "k_rope_text", "k_rope_vision", "k_quant_s8", "k_gelu_mul_q",
            "k_gelu_tanh", "k_scale", "k_mul_slice", "k_pos_add",
            "k_attn_pack", "k_attn_softmax"};
        enum { NOPK = sizeof(on) / sizeof(on[0]) };
        ze_kernel_handle_t ok[NOPK];
        ze_module_desc_t md; memset(&md, 0, sizeof(md));
        md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        md.format = ZE_MODULE_FORMAT_IL_SPIRV;
        md.inputSize = e2_gpu_spv_e2_ops_len;
        md.pInputModule = e2_gpu_spv_e2_ops;
        ze_module_build_log_handle_t mlog = NULL;
        ze_result_t mrc = zeModuleCreate(g->ctx, g->dev, &md, &g->omodule, &mlog);
        if (mrc != ZE_RESULT_SUCCESS) {
            fprintf(stderr, "e2: ops module build failed: 0x%x\n", (unsigned)mrc);
            if (mlog) {
                char buf[2048] = "";
                size_t sz = sizeof(buf) - 1;
                zeModuleBuildLogGetString(mlog, &sz, buf);
                fprintf(stderr, "e2: ops module: %s\n", buf);
            }
            goto fail;
        }
        int got = 0;
        memset(ok, 0, sizeof(ok));
        for (int i = 0; i < NOPK; i++) {
            ze_kernel_desc_t d; memset(&d, 0, sizeof(d));
            d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            d.pKernelName = on[i];
            ze_kernel_handle_t k = NULL;
            if (zeKernelCreate(g->omodule, &d, &k) != ZE_RESULT_SUCCESS ||
                zeKernelSetGroupSize(k, 64, 1, 1) != ZE_RESULT_SUCCESS) {
                if (getenv("E2_GPU_PROF"))
                    fprintf(stderr, "e2: kernel %s missing from the ops module\n", on[i]);
                break;
            }
            ok[i] = k;
            got++;
        }
        if (got < NOPK) {
            if (getenv("E2_GPU_PROF"))
                fprintf(stderr, "e2: ops module: %d/%d kernels\n", got, (int)NOPK);
            goto fail;
        }
        g->k_rmsnorm = ok[0]; g->k_rmsnorm_q = ok[1]; g->k_norm_rope = ok[2];
        g->k_rope_text = ok[3]; g->k_rope_vision = ok[4]; g->k_quant_s8 = ok[5];
        g->k_gelu_mul_q = ok[6]; g->k_gelu_tanh = ok[7]; g->k_scale = ok[8];
        g->k_mul_slice = ok[9]; g->k_pos_add = ok[10]; g->k_attn_pack = ok[11];
        g->k_attn_softmax = ok[12];
    }

    ze_event_pool_desc_t epd; memset(&epd, 0, sizeof(epd));
    epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    epd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    epd.count = 1;
    if (zeEventPoolCreate(g->ctx, &epd, 1, &g->dev, &g->epool) != ZE_RESULT_SUCCESS)
        goto fail;
    ze_event_desc_t ed; memset(&ed, 0, sizeof(ed));
    ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    ed.index = 0;
    ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ed.wait = 0;
    if (zeEventCreate(g->epool, &ed, &g->evt) != ZE_RESULT_SUCCESS) goto fail;

    /* Every activation the kernels touch must live in shared USM: switch the
     * arena and drop the malloc'd scratch so it is reallocated device-safe. */
    g->ready = 1;
    if (getenv("E2_GPU_TS")) {
        g->ts_on = 1;
        g->ts = e2_usm_alloc(g, 4096 * sizeof(uint64_t));
        ze_device_properties_t props;
        memset(&props, 0, sizeof(props));
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (zeDeviceGetProperties(g->dev, &props) == ZE_RESULT_SUCCESS)
            g->ts_res = (double)props.timerResolution;
        if (!g->ts) g->ts_on = 0;
    }
    g->act.blk_alloc = act_usm_alloc;
    g->act.blk_free = act_usm_free;
    g->act.alloc_ctx = g;
    e2_arena_free(&ctx->scratch);
    ctx->scratch.blk_alloc = e2_arena_usm_alloc;
    ctx->scratch.blk_free = e2_arena_usm_free;
    ctx->scratch.alloc_ctx = ctx;
    free(ctx->qbuf); ctx->qbuf = NULL; ctx->qcap = 0;
    free(ctx->sbuf); ctx->sbuf = NULL; ctx->scap = 0;
    free(ctx->tbuf); ctx->tbuf = NULL; ctx->tcap = 0;
    free(ctx->ktbuf); ctx->ktbuf = NULL; ctx->ktcap = 0;
    free(ctx->abuf); ctx->abuf = NULL; ctx->acap = 0;
    return 0;
fail:
    gpu_destroy(g);
    ctx->gpu = NULL;
    free(g);
    return -1;
}

void e2_gpu_free(e2_ctx *ctx) {
    if (!ctx || !ctx->gpu) return;
    e2_gpu *g = ctx->gpu;
    gpu_destroy(g);
    free(g);
    ctx->gpu = NULL;
}
