/* e2_gpu.c - Level Zero packed-int8 GEMM backend.
 *
 * Opt-in only (E2_DEVICE=gpu or e2_set_device(ctx, E2_DEVICE_GPU)).  The CPU
 * path in e2_gemm.c stays the default and the reference; every entry point here
 * returns nonzero on any failure so the caller can fall back.
 *
 * Design: one driver/device/context/in-order queue/command list and one
 * persistent kernel (gemm_4x4 from tools/gpu/e2_gemm.cl).  Weights are copied
 * into shared USM once (cache keyed by the e2_qmat.q pointer); activations and
 * the output tile are re-uploaded/read back per call, which is the measured
 * per-call cost of the CPU -> GEMM -> CPU pipeline.
 */
#include "e2_gpu.h"
#include "e2_gpu_l0.h"
#include "e2_gpu_spv.h"

#include <stdlib.h>
#include <string.h>

#define E2_GPU_LOCAL 128

/* E2_GPU_PROF=1 splits the per-call cost into upload / submit / sync / copy:
 * the sync dominates on this part and that is what caps the GPU path. */
static double g_t_up, g_t_sub, g_t_sync, g_t_copy;
static long g_n_call;
static int g_prof = -1;

/* One uploaded packed matrix, in shared USM.  Keyed by the source pointer:
 * the model file is mmap'd and never moves, so the address is a stable id. */
typedef struct {
    const int8_t *q_src;
    int n, k, blk, ntiles, nblk;
    size_t tstride;
    int8_t *q;          /* shared USM: ntiles * tstride packed tiles */
    float *scale;       /* shared USM: n */
    int32_t *wsum;      /* shared USM: n * nblk (contiguous) */
} e2_gpu_wc;

/* Open-addressing table of uploads: the text tower cycles through ~8 matrices
 * per layer, so a one-entry cache re-copied ~90 MB per forward. */
typedef struct {
    e2_gpu_wc *tab;
    int cap, used;
} e2_gpu_wct;

typedef struct {
    ze_driver_handle_t drv;
    ze_device_handle_t dev;
    ze_context_handle_t ctx;
    ze_command_queue_handle_t queue;
    ze_command_list_handle_t list;
    ze_module_handle_t module;
    ze_kernel_handle_t kern;
    ze_event_pool_handle_t epool;
    ze_event_handle_t evt;
    int ready;

    e2_gpu_wct wct;
    uint8_t *xq; size_t xq_cap;
    float *xs; size_t xs_cap;
    float *y; size_t y_cap;
} e2_gpu;

static void usm_free(e2_gpu *g) {
    if (g->ctx) {
        for (int i = 0; i < g->wct.cap; i++) {
            e2_gpu_wc *e = &g->wct.tab[i];
            if (!e->q_src) continue;
            if (e->q)     zeMemFree(g->ctx, e->q);
            if (e->scale) zeMemFree(g->ctx, e->scale);
            if (e->wsum)  zeMemFree(g->ctx, e->wsum);
        }
        if (g->xq) zeMemFree(g->ctx, g->xq);
        if (g->xs) zeMemFree(g->ctx, g->xs);
        if (g->y)  zeMemFree(g->ctx, g->y);
    }
    free(g->wct.tab);
    memset(&g->wct, 0, sizeof(g->wct));
    g->xq = NULL;
    g->xs = NULL;
    g->y = NULL;
    g->xq_cap = g->xs_cap = g->y_cap = 0;
}

static void *usm_alloc(e2_gpu *g, size_t n) {
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

static void gpu_destroy(e2_gpu *g) {
    if (!g) return;
    if (g_prof && g_n_call)
        fprintf(stderr, "gpu calls %ld upload %.1f submit %.1f sync %.1f copy %.1f ms\n",
                g_n_call, g_t_up, g_t_sub, g_t_sync, g_t_copy);
    usm_free(g);
    if (g->kern)   zeKernelDestroy(g->kern);
    if (g->module) zeModuleDestroy(g->module);
    if (g->list)   zeCommandListDestroy(g->list);
    if (g->evt)    zeEventDestroy(g->evt);
    if (g->epool)  zeEventPoolDestroy(g->epool);
    if (g->ctx)    zeContextDestroy(g->ctx);
    memset(g, 0, sizeof(*g));
}

int e2_gpu_init(e2_ctx *ctx) {
    if (!ctx || ctx->gpu) return ctx && ctx->gpu ? 0 : -1;
    if (e2_gpu_l0_open() != 0) return -1;

    e2_gpu *g = calloc(1, sizeof(*g));
    if (!g) return -1;
    ctx->gpu = g;

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

    /* Immediate command list: appends go straight to the hardware queue, so a
     * GEMM costs one append plus one host sync instead of
     * reset/close/execute/synchronize on a deferred list (measured ~1 ms of
     * pure driver overhead per call on this iGPU). */
    ze_command_queue_desc_t qd; memset(&qd, 0, sizeof(qd));
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    if (zeCommandListCreateImmediate(g->ctx, g->dev, &qd, &g->list) != ZE_RESULT_SUCCESS)
        goto fail;

    ze_module_desc_t md; memset(&md, 0, sizeof(md));
    md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    md.format = ZE_MODULE_FORMAT_IL_SPIRV;
    md.inputSize = e2_gpu_spv_e2_gemm_len;
    md.pInputModule = e2_gpu_spv_e2_gemm;
    if (zeModuleCreate(g->ctx, g->dev, &md, &g->module, NULL) != ZE_RESULT_SUCCESS) goto fail;

    ze_kernel_desc_t kd; memset(&kd, 0, sizeof(kd));
    kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kd.pKernelName = "gemm_4x4";
    if (zeKernelCreate(g->module, &kd, &g->kern) != ZE_RESULT_SUCCESS) goto fail;
    if (zeKernelSetGroupSize(g->kern, E2_GPU_LOCAL, 1, 1) != ZE_RESULT_SUCCESS) goto fail;

    ze_event_pool_desc_t epd; memset(&epd, 0, sizeof(epd));
    epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    epd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    epd.count = 1;
    if (zeEventPoolCreate(g->ctx, &epd, 1, &g->dev, &g->epool) != ZE_RESULT_SUCCESS) goto fail;
    ze_event_desc_t ed; memset(&ed, 0, sizeof(ed));
    ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    ed.index = 0;
    ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ed.wait = 0;
    if (zeEventCreate(g->epool, &ed, &g->evt) != ZE_RESULT_SUCCESS) goto fail;

    g->ready = 1;
    return 0;
fail:
    gpu_destroy(g);
    ctx->gpu = NULL;
    free(g);
    return -1;
}

void e2_gpu_free(e2_ctx *ctx) {
    if (!ctx || !ctx->gpu) return;
    gpu_destroy(ctx->gpu);
    free(ctx->gpu);
    ctx->gpu = NULL;
}

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

/* Copy one packed matrix into shared USM the first time it is used.  Shared
 * memory on an iGPU is the same DRAM, so this is one memcpy and no PCIe hop;
 * after that the kernel reads the copy at full bandwidth. */
static e2_gpu_wc *upload_weight(e2_gpu *g, const e2_qmat *w) {
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

    e->q = (int8_t *)usm_alloc(g, qbytes ? qbytes : 64);
    e->scale = (float *)usm_alloc(g, (size_t)w->n * sizeof(float));
    e->wsum = (int32_t *)usm_alloc(g, (size_t)w->n * nblk * sizeof(int32_t));
    if (!e->q || !e->scale || !e->wsum) {
        if (e->q) zeMemFree(g->ctx, e->q);
        if (e->scale) zeMemFree(g->ctx, e->scale);
        if (e->wsum) zeMemFree(g->ctx, e->wsum);
        memset(e, 0, sizeof(*e));
        return NULL;
    }
    memcpy(e->q, w->q, qbytes);
    memcpy(e->scale, w->scale, (size_t)w->n * sizeof(float));
    if (wss_src == nblk) {
        memcpy(e->wsum, w->wsum, (size_t)w->n * nblk * sizeof(int32_t));
    } else {
        for (int i = 0; i < w->n; i++)
            memcpy(e->wsum + (size_t)i * nblk, w->wsum + (size_t)i * wss_src,
                   (size_t)nblk * sizeof(int32_t));
    }
    e->q_src = w->q;
    e->n = w->n; e->k = w->k; e->blk = blk; e->ntiles = ntiles;
    e->nblk = nblk; e->tstride = tstride;
    g->wct.used++;
    return e;
}

static int ensure_act(e2_gpu *g, const uint8_t *xq, const float *xs, int xss,
                      int m, int k) {
    size_t qbytes = (size_t)m * k;
    size_t sbytes = (size_t)m * xss * sizeof(float);
    if (g->xq_cap < qbytes) {
        uint8_t *p = (uint8_t *)usm_alloc(g, qbytes ? qbytes : 64);
        if (!p) return -1;
        if (g->xq) zeMemFree(g->ctx, g->xq);
        g->xq = p; g->xq_cap = qbytes;
    }
    if (g->xs_cap < sbytes) {
        float *p = (float *)usm_alloc(g, sbytes ? sbytes : 64);
        if (!p) return -1;
        if (g->xs) zeMemFree(g->ctx, g->xs);
        g->xs = p; g->xs_cap = sbytes;
    }
    memcpy(g->xq, xq, qbytes);
    memcpy(g->xs, xs, sbytes);
    return 0;
}


int e2_gpu_gemm(e2_ctx *ctx, float *y, int ns, const uint8_t *xq, const float *xs,
                int xss, int m, int k, const e2_qmat *w, int nb0, int nb1) {
    e2_gpu *g = ctx ? ctx->gpu : NULL;
    if (!g || !g->ready) return -1;
    if (nb0 != 0) return -1;                 /* sub-ranges stay on the CPU for now */
    if (!w->packed || (k % 128) != 0 || m <= 0 || w->n <= 0 || ns != w->n) return -1;

    int blk = w->blk ? w->blk : 128;
    int nblk = k / blk;
    int ntiles = (w->n + 15) / 16;
    if (nblk <= 0 || nb1 < ntiles) return -1;

    if (g_prof < 0) g_prof = getenv("E2_GPU_PROF") != NULL;
    double _t0 = g_prof ? e2_now_ms() : 0;
    e2_gpu_wc *wc = upload_weight(g, w);
    if (!wc) return -1;
    if (ensure_act(g, xq, xs, xss, m, k) != 0) return -1;
    double _t1 = g_prof ? e2_now_ms() : 0;
    if (g_prof) g_t_up += _t1 - _t0;

    size_t yn = (size_t)m * w->n;
    if (g->y_cap < yn * sizeof(float)) {
        float *p = (float *)usm_alloc(g, yn * sizeof(float));
        if (!p) return -1;
        if (g->y) zeMemFree(g->ctx, g->y);
        g->y = p; g->y_cap = yn * sizeof(float);
    }

    uint32_t ai = 0;
    const uint8_t *xq_u = g->xq;
    const float *xs_u = g->xs;
    const int8_t *q_u = wc->q;
    const float *scale_u = wc->scale;
    const int32_t *wsum_u = wc->wsum;
    float *y_u = g->y;
    size_t ts = wc->tstride;
    int wss = nblk;
    if (zeKernelSetArgumentValue(g->kern, ai++, sizeof(xq_u), &xq_u) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(xs_u), &xs_u) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(q_u), &q_u) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(scale_u), &scale_u) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(wsum_u), &wsum_u) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(y_u), &y_u) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(int), &m) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(int), &k) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(int), &w->n) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(int), &nblk) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(int), &xss) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(int), &wss) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->kern, ai++, sizeof(ts), &ts) != ZE_RESULT_SUCCESS)
        return -1;

    size_t wi = (size_t)((m + 3) / 4) * ntiles * 4;
    size_t groups = (wi + E2_GPU_LOCAL - 1) / E2_GPU_LOCAL;
    if (!groups) groups = 1;
    ze_group_count_t gc = {(uint32_t)groups, 1, 1};

    if (zeCommandListAppendLaunchKernel(g->list, g->kern, &gc, g->evt, 0, NULL) != ZE_RESULT_SUCCESS)
        return -1;
    if (g_prof) { g_t_sub += e2_now_ms() - _t1; _t1 = e2_now_ms(); }
    if (zeEventHostSynchronize(g->evt, UINT64_MAX) != ZE_RESULT_SUCCESS) return -1;
    if (zeEventHostReset(g->evt) != ZE_RESULT_SUCCESS) return -1;
    if (g_prof) { g_t_sync += e2_now_ms() - _t1; _t1 = e2_now_ms(); }

    memcpy(y, g->y, yn * sizeof(float));
    if (g_prof) {
        g_t_copy += e2_now_ms() - _t1;
        if (++g_n_call % 512 == 0)
            fprintf(stderr, "gpu calls %ld upload %.1f submit %.1f sync %.1f copy %.1f ms\n",
                    g_n_call, g_t_up, g_t_sub, g_t_sync, g_t_copy);
    }
    return 0;
}
