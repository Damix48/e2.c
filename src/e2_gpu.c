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
#include "e2_gpu_spv.h"

#include <stdlib.h>
#include <string.h>

#define E2_GPU_LOCAL 128

int e2_gpu_launch_named(e2_gpu *g, ze_kernel_handle_t k, uint32_t groups,
                        const char *name) {
    ze_group_count_t gc = {groups ? groups : 1, 1, 1};
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

int e2_gpu_launch(e2_gpu *g, ze_kernel_handle_t k, uint32_t groups) {
    return e2_gpu_launch_named(g, k, groups, "?");
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
        if (e->q)     zeMemFree(g->ctx, e->q);
        if (e->scale) zeMemFree(g->ctx, e->scale);
        if (e->wsum)  zeMemFree(g->ctx, e->wsum);
        if (e->pq)    zeMemFree(g->ctx, e->pq);
        if (e->wsumt) zeMemFree(g->ctx, e->wsumt);
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

/* Copy one packed matrix into shared USM the first time it is used, and build
 * the matrix-engine layout: [n/8][k/32][8 lanes][8 x u32], i.e. for one 8-column
 * block and one 32-wide k chunk, lane l holds the 4-byte k quad 4l..4l+3 of
 * each of the 8 columns.  Shared memory on an iGPU is the same DRAM, so this is
 * one CPU pass and no PCIe hop; after that the kernel reads it at full
 * bandwidth. */
e2_gpu_wc *e2_gpu_weight(e2_gpu *g, const e2_qmat *w) {
    int blk = w->blk ? w->blk : 128;
    int nblk = w->k / blk;
    int ntiles = (w->n + 15) / 16;
    size_t tstride = w->tstride ? w->tstride : (size_t)(w->k / 4) * 64;
    size_t qbytes = (size_t)ntiles * tstride;
    int wss_src = w->wsmstride ? w->wsmstride : nblk;

    if (g->wct.cap && (g->wct.used + 1) * 4 >= g->wct.cap * 3 && wc_grow(g) != 0)
        return NULL;
    if (!g->wct.cap && wc_grow(g) != 0) return NULL;
    e2_gpu_wc *e = wc_slot(g->wct.tab, w->q, g->wct.cap);
    if (e->q_src == w->q) return e;                 /* already resident */

    e->q = (int8_t *)e2_usm_alloc(g, qbytes ? qbytes : 64);
    e->scale = (float *)e2_usm_alloc(g, (size_t)w->n * sizeof(float));
    e->wsum = (int32_t *)e2_usm_alloc(g, (size_t)w->n * nblk * sizeof(int32_t));
    if (!e->q || !e->scale || !e->wsum) goto fail;
    memcpy(e->q, w->q, qbytes);
    memcpy(e->scale, w->scale, (size_t)w->n * sizeof(float));
    if (wss_src == nblk) {
        memcpy(e->wsum, w->wsum, (size_t)w->n * nblk * sizeof(int32_t));
    } else {
        for (int i = 0; i < w->n; i++)
            memcpy(e->wsum + (size_t)i * nblk, w->wsum + (size_t)i * wss_src,
                   (size_t)nblk * sizeof(int32_t));
    }

    if ((w->n % 8) == 0 && (w->k % 32) == 0) {
        int nkc = w->k / 32, nc8 = w->n / 8;
        e->pq = (int8_t *)e2_usm_alloc(g, (size_t)nc8 * nkc * 256);
        e->wsumt = (int32_t *)e2_usm_alloc(g, (size_t)w->n * sizeof(int32_t));
        if (!e->pq || !e->wsumt) goto fail;
        const int8_t *src = w->q;
        size_t ts = tstride;
        for (int cb = 0; cb < nc8; cb++)
            for (int kc = 0; kc < nkc; kc++) {
                uint8_t *dst = (uint8_t *)e->pq + ((size_t)cb * nkc + kc) * 256;
                for (int lane = 0; lane < 8; lane++) {
                    int col = cb * 8 + lane;
                    const int8_t *sp = src + (size_t)(col / 16) * ts +
                                       (size_t)(kc * 8) * 64 + (col % 16) * 4;
                    uint8_t *dp = dst + lane * 32;
                    for (int c = 0; c < 8; c++)
                        memcpy(dp + c * 4, sp + (size_t)c * 64, 4);
                }
            }
        for (int c = 0; c < w->n; c++) {
            int32_t t = 0;
            for (int b = 0; b < nblk; b++) t += e->wsum[(size_t)c * nblk + b];
            e->wsumt[c] = t;
        }
    }

    e->q_src = w->q;
    e->n = w->n; e->k = w->k; e->blk = blk; e->ntiles = ntiles;
    e->nblk = nblk; e->tstride = tstride;
    g->wct.used++;
    return e;
fail:
    if (e->q)     zeMemFree(g->ctx, e->q);
    if (e->scale) zeMemFree(g->ctx, e->scale);
    if (e->wsum)  zeMemFree(g->ctx, e->wsum);
    if (e->pq)    zeMemFree(g->ctx, e->pq);
    if (e->wsumt) zeMemFree(g->ctx, e->wsumt);
    memset(e, 0, sizeof(*e));
    return NULL;
}

/* ------------------------------------------------------------------ */
/* GEMM                                                                */
/* ------------------------------------------------------------------ */

/* Append one packed-int8 GEMM for an already-packed activation block. */
int e2_gpu_enq_act(e2_gpu *g, float *y, int m, const e2_linear *l, const e2_act *a) {
    const e2_qmat *w = &l->w;
    if (m <= 0 || !w->packed || (w->k % 32) || (w->n % 8) || (w->k % 128)) return -1;
    if (l->in_min || l->in_max || l->out_min || l->out_max || w->bias) return -1;
    e2_gpu_wc *wc = e2_gpu_weight(g, w);
    if (!wc || !wc->pq) return -1;
    if (g->use_blk ? !wc->wsum : !wc->wsumt) return -1;
    if (e2_gpu_enq_open(g) != 0) return -1;

    uint8_t *app = a->pack;
    float *sc = a->scale;
    const int m2 = m, k2 = wc->k, n2 = wc->n;
    ze_kernel_handle_t kern = g->use_blk ? g->k_gemm_blk : g->k_gemm;
    const int32_t *wt = g->use_blk ? wc->wsum : wc->wsumt;
    float *ws = wc->scale;
    uint8_t *wp = (uint8_t *)wc->pq;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(kern, ai++, sizeof(app), &app) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(sc), &sc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(wp), &wp) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(ws), &ws) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(wt), &wt) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(y), &y) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &m2) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &k2) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(kern, ai++, sizeof(int), &n2) != ZE_RESULT_SUCCESS)
        return -1;
    /* k_gemm_dpas: RB=4 row blocks x CT=2 column tiles per sub-group (32 rows);
     * k_gemm_dpas_blk: RB=2 (16 rows, it keeps an int8 and a float8 accumulator
     * per tile and 32+32 registers is what fits). */
    int rows_per_sg = g->use_blk ? 16 : 32;
    uint32_t subs = (uint32_t)(((m + rows_per_sg - 1) / rows_per_sg) * (wc->n / 16));
    return e2_gpu_launch_named(g, kern, subs ? subs : 1, g->use_blk ? "gemm_blk" : "gemm");
}

static void *act_usm_alloc(void *opaque, size_t n) {
    return e2_usm_alloc((e2_gpu *)opaque, n);
}

static void act_usm_free(void *opaque, void *p) {
    e2_gpu *g = opaque;
    if (g->ctx && p) zeMemFree(g->ctx, p);
}

/* row-major u8 -> matrix-engine layout, into a fresh forward-scoped block. */
int e2_gpu_pack_act_u8(e2_gpu *g, const uint8_t *src8, int m, int k, e2_act *a) {
    int nkc = k / 32, mblk = (m + 7) / 8;
    size_t pk = (size_t)mblk * nkc * 256;
    a->pack = e2_arena_alloc(&g->act, pk);
    if (!a->pack) return -1;
    uint8_t *src = (uint8_t *)src8, *dst = a->pack;
    int mm = m, kk = k;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(src), &src) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(dst), &dst) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(int), &mm) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(int), &kk) != ZE_RESULT_SUCCESS)
        return -1;
    return e2_gpu_launch_named(g, g->k_pack_u8, (uint32_t)((size_t)mblk * nkc), "pack_u8");
}

/* f32 rows -> u8 rows + one scale per 128-wide block -> matrix-engine layout. */
int e2_gpu_pack_act(e2_gpu *g, const float *x, int m, int k, e2_act *a) {
    int nblk = k / 128;
    uint8_t *q8 = e2_arena_alloc(&g->act, (size_t)m * k);
    a->scale = e2_arena_alloc(&g->act, (size_t)m * nblk * sizeof(float));
    if (!q8 || !a->scale) return -1;
    if (e2_gpu_enq_open(g) != 0) return -1;
    float *sc = a->scale;
    int rows = m, kk = k, blk = 128;
    uint32_t ai = 0;
    if (zeKernelSetArgumentValue(g->k_quant_u8, ai++, sizeof(x), &x) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_u8, ai++, sizeof(q8), &q8) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_u8, ai++, sizeof(sc), &sc) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_u8, ai++, sizeof(int), &rows) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_u8, ai++, sizeof(int), &kk) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_quant_u8, ai++, sizeof(int), &blk) != ZE_RESULT_SUCCESS)
        return -1;
    uint32_t groups = (uint32_t)(((long)m * nblk * 32 + 63) / 64);
    if (e2_gpu_launch_named(g, g->k_quant_u8, groups, "quant_u8") != 0) return -1;
    return e2_gpu_pack_act_u8(g, q8, m, k, a);
}

int e2_gpu_enq_linear(e2_gpu *g, float *y, const float *x, int m, const e2_linear *l) {
    e2_act a;
    if (e2_gpu_pack_act(g, x, m, l->w.k, &a) != 0) return -1;
    return e2_gpu_enq_act(g, y, m, l, &a);
}

int e2_gpu_enq_linear_list(e2_gpu *g, const float *x, int m,
                           const e2_linear *const *ls, float *const *ys, int n) {
    e2_act a;
    if (!n || e2_gpu_pack_act(g, x, m, ls[0]->w.k, &a) != 0) return -1;
    for (int i = 0; i < n; i++)
        if (e2_gpu_enq_act(g, ys[i], m, ls[i], &a) != 0) return -1;
    return 0;
}

int e2_gpu_enq_linear_pq(e2_gpu *g, float *y, const uint8_t *xq, const float *xs,
                         int m, const e2_linear *l) {
    const e2_qmat *w = &l->w;
    if (w->blk && w->blk != 128) return -1;
    e2_act a;
    if (e2_gpu_pack_act_u8(g, xq, m, w->k, &a) != 0) return -1;
    a.scale = (float *)xs;
    return e2_gpu_enq_act(g, y, m, l, &a);
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

static void gpu_destroy(e2_gpu *g) {
    if (!g) return;
    usm_free(g);
    ze_kernel_handle_t ks[] = {g->k_pack_u8, g->k_pack_wq, g->k_gemm, g->k_gemm_blk,
        g->k_rmsnorm, g->k_norm_rope, g->k_rope_text, g->k_rope_vision, g->k_attn,
        g->k_quant_u8, g->k_gelu_mul_q, g->k_gelu_tanh, g->k_scale, g->k_mul_slice,
        g->k_pos_add, g->k_softmax, g->kern};
    for (size_t i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
        if (ks[i]) zeKernelDestroy(ks[i]);
    if (g->dmodule) zeModuleDestroy(g->dmodule);
    if (g->omodule) zeModuleDestroy(g->omodule);
    if (g->module)  zeModuleDestroy(g->module);
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
    g->use_blk = 1;
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

    {
        static const char *dn[] = {"k_pack_u8", "k_pack_wq", "k_gemm_dpas", "k_gemm_dpas_blk"};
        ze_kernel_handle_t dk[4] = {0};
        ze_module_desc_t md; memset(&md, 0, sizeof(md));
        md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        md.format = ZE_MODULE_FORMAT_IL_SPIRV;
        md.inputSize = e2_gpu_spv_e2_dpas_len;
        md.pInputModule = e2_gpu_spv_e2_dpas;
        if (zeModuleCreate(g->ctx, g->dev, &md, &g->dmodule, NULL) != ZE_RESULT_SUCCESS)
            goto fail;                       /* no matrix engine -> CPU path */
        int got = 0;
        for (int i = 0; i < 4; i++) {
            ze_kernel_desc_t d; memset(&d, 0, sizeof(d));
            d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            d.pKernelName = dn[i];
            ze_kernel_handle_t k = NULL;
            if (zeKernelCreate(g->dmodule, &d, &k) != ZE_RESULT_SUCCESS ||
                zeKernelSetGroupSize(k, 8, 1, 1) != ZE_RESULT_SUCCESS)
                break;
            dk[i] = k;
            got++;
        }
        if (got < 3) goto fail;
        g->k_pack_u8 = dk[0];
        g->k_pack_wq = dk[1];
        g->k_gemm = dk[2];
        g->k_gemm_blk = (got >= 4) ? dk[3] : NULL;
        g->use_dpas = 1;
        if (!g->k_gemm_blk) g->use_blk = 0;
        if (getenv("E2_GPU_PROF"))
            fprintf(stderr, "gpu dpas: %d/4 kernels, use_blk=%d\n", got, g->use_blk);
    }

    {
        static const char *on[] = {"k_rmsnorm", "k_norm_rope", "k_rope_text",
            "k_rope_vision", "k_quant_u8", "k_gelu_mul_q", "k_gelu_tanh",
            "k_scale", "k_mul_slice", "k_pos_add", "k_softmax",
            "k_attn_part", "k_attn_comb"};
        ze_kernel_handle_t ok[13] = {0};
        ze_module_desc_t md; memset(&md, 0, sizeof(md));
        md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        md.format = ZE_MODULE_FORMAT_IL_SPIRV;
        md.inputSize = e2_gpu_spv_e2_ops_len;
        md.pInputModule = e2_gpu_spv_e2_ops;
        if (zeModuleCreate(g->ctx, g->dev, &md, &g->omodule, NULL) != ZE_RESULT_SUCCESS)
            goto fail;
        int got = 0;
        for (int i = 0; i < 13; i++) {
            ze_kernel_desc_t d; memset(&d, 0, sizeof(d));
            d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            d.pKernelName = on[i];
            ze_kernel_handle_t k = NULL;
            if (zeKernelCreate(g->omodule, &d, &k) != ZE_RESULT_SUCCESS ||
                zeKernelSetGroupSize(k, 64, 1, 1) != ZE_RESULT_SUCCESS)
                break;
            ok[i] = k;
            got++;
        }
        if (got < 13) goto fail;
        /* attention runs ATT_ROWS (4) sub-groups per work-group so a query
         * block shares its K/V lines; everything else uses one 64-thread WG */
        if (zeKernelSetGroupSize(ok[11], 128, 1, 1) != ZE_RESULT_SUCCESS) goto fail;
        g->k_rmsnorm = ok[0]; g->k_norm_rope = ok[1]; g->k_rope_text = ok[2];
        g->k_rope_vision = ok[3]; g->k_quant_u8 = ok[4]; g->k_gelu_mul_q = ok[5];
        g->k_gelu_tanh = ok[6]; g->k_scale = ok[7]; g->k_mul_slice = ok[8];
        g->k_pos_add = ok[9]; g->k_softmax = ok[10];
        g->k_attn_part = ok[11]; g->k_attn_comb = ok[12];
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
