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
/* Matrix-engine GEMM: 4-5x the plain kernel's MAC rate, but the CPU-driven
 * "one submission per GEMM" pipeline pays ~0.4 ms of dispatch latency per call,
 * so it is opt-in until the tower runs device-resident (see PERFORMANCE.md). */
static int g_dpas = -1;

/* One uploaded packed matrix, in shared USM.  Keyed by the source pointer:
 * the model file is mmap'd and never moves, so the address is a stable id. */
typedef struct {
    const int8_t *q_src;
    int n, k, blk, ntiles, nblk;
    size_t tstride;
    int8_t *q;          /* shared USM: ntiles * tstride packed tiles (CPU kernel) */
    float *scale;       /* shared USM: n */
    int32_t *wsum;      /* shared USM: n * nblk (contiguous) */
    int8_t *pq;         /* shared USM: DPAS layout, n * k bytes */
    int32_t *wsumt;     /* shared USM: n, sum over all k of each column */
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
    ze_module_handle_t dmodule;     /* e2_dpas.cl: pack + matrix-engine GEMM */
    ze_kernel_handle_t k_pack_u8, k_pack_wq, k_gemm;
    ze_event_pool_handle_t epool;
    ze_event_handle_t evt;
    int ready;

    e2_gpu_wct wct;
    uint8_t *xq; size_t xq_cap;
    float *xs; size_t xs_cap;
    float *y; size_t y_cap;
    /* DPAS path: row-major u8 staging, packed activations, per-row scales */
    uint8_t *a_raw; size_t a_raw_cap;
    uint8_t *a_pack; size_t a_pack_cap;
    float *a_scale; size_t a_scale_cap;
} e2_gpu;

static void usm_free(e2_gpu *g) {
    if (g->ctx) {
        for (int i = 0; i < g->wct.cap; i++) {
            e2_gpu_wc *e = &g->wct.tab[i];
            if (!e->q_src) continue;
            if (e->q)     zeMemFree(g->ctx, e->q);
            if (e->scale) zeMemFree(g->ctx, e->scale);
            if (e->wsum)  zeMemFree(g->ctx, e->wsum);
            if (e->pq)    zeMemFree(g->ctx, e->pq);
            if (e->wsumt) zeMemFree(g->ctx, e->wsumt);
        }
        if (g->xq) zeMemFree(g->ctx, g->xq);
        if (g->xs) zeMemFree(g->ctx, g->xs);
        if (g->y)  zeMemFree(g->ctx, g->y);
        if (g->a_raw)   zeMemFree(g->ctx, g->a_raw);
        if (g->a_pack)  zeMemFree(g->ctx, g->a_pack);
        if (g->a_scale) zeMemFree(g->ctx, g->a_scale);
    }
    free(g->wct.tab);
    memset(&g->wct, 0, sizeof(g->wct));
    g->xq = NULL;
    g->xs = NULL;
    g->y = NULL;
    g->a_raw = NULL;
    g->a_pack = NULL;
    g->a_scale = NULL;
    g->xq_cap = g->xs_cap = g->y_cap = 0;
    g->a_raw_cap = g->a_pack_cap = g->a_scale_cap = 0;
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
    if (g->k_pack_u8) zeKernelDestroy(g->k_pack_u8);
    if (g->k_pack_wq) zeKernelDestroy(g->k_pack_wq);
    if (g->k_gemm)    zeKernelDestroy(g->k_gemm);
    if (g->dmodule)   zeModuleDestroy(g->dmodule);
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

    ze_module_desc_t md2; memset(&md2, 0, sizeof(md2));
    md2.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    md2.format = ZE_MODULE_FORMAT_IL_SPIRV;
    md2.inputSize = e2_gpu_spv_e2_dpas_len;
    md2.pInputModule = e2_gpu_spv_e2_dpas;

    ze_kernel_desc_t kd; memset(&kd, 0, sizeof(kd));
    kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kd.pKernelName = "gemm_4x4";
    if (zeKernelCreate(g->module, &kd, &g->kern) != ZE_RESULT_SUCCESS) goto fail;
    if (zeKernelSetGroupSize(g->kern, E2_GPU_LOCAL, 1, 1) != ZE_RESULT_SUCCESS) goto fail;

    /* Matrix-engine module.  Optional: if the device/toolchain cannot build
     * it, keep the plain kernel path (still correct, ~4x slower). */
    if (zeModuleCreate(g->ctx, g->dev, &md2, &g->dmodule, NULL) == ZE_RESULT_SUCCESS) {
        static const char *dn[3] = {"k_pack_u8", "k_pack_wq", "k_gemm_dpas"};
        ze_kernel_handle_t *dk[3] = {&g->k_pack_u8, &g->k_pack_wq, &g->k_gemm};
        for (int i = 0; i < 3; i++) {
            ze_kernel_desc_t d; memset(&d, 0, sizeof(d));
            d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            d.pKernelName = dn[i];
            if (zeKernelCreate(g->dmodule, &d, dk[i]) != ZE_RESULT_SUCCESS ||
                zeKernelSetGroupSize(*dk[i], 8, 1, 1) != ZE_RESULT_SUCCESS)
                break;
        }
    }

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
    /* Matrix-engine layout: [n/8][k/32][8 lanes][8 x u32].  Built on the CPU
     * into shared USM, so the kernel reads it directly (no upload, no repack
     * kernel, one-time cost on first use).  Skipped when n or k do not tile. */
    if ((w->n % 8) == 0 && (w->k % 32) == 0) {
        int nkc = w->k / 32, nc8 = w->n / 8;
        e->pq = (int8_t *)usm_alloc(g, (size_t)nc8 * nkc * 256);
        e->wsumt = (int32_t *)usm_alloc(g, (size_t)w->n * sizeof(int32_t));
        if (e->pq && e->wsumt) {
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
        } else {
            if (e->pq) zeMemFree(g->ctx, e->pq);
            if (e->wsumt) zeMemFree(g->ctx, e->wsumt);
            e->pq = NULL;
            e->wsumt = NULL;
        }
    }

    e->q_src = w->q;
    e->n = w->n; e->k = w->k; e->blk = blk; e->ntiles = ntiles;
    e->nblk = nblk; e->tstride = tstride;
    g->wct.used++;
    return e;
}

/* Blocking waits cost ~0.4 ms per GEMM on this driver (the sync dominates the
 * whole GPU path), so poll the host-visible event first and only fall back to
 * the blocking wait if the kernel really is long. */
static int wait_evt(e2_gpu *g) {
    for (long i = 0; i < 2000000L; i++) {
        ze_result_t r = zeEventQueryStatus(g->evt);
        if (r == ZE_RESULT_SUCCESS) return 0;
        if (r != ZE_RESULT_NOT_READY) return -1;
    }
    return zeEventHostSynchronize(g->evt, UINT64_MAX) == ZE_RESULT_SUCCESS ? 0 : -1;
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


/* ---- matrix-engine path -------------------------------------------------- */

typedef struct { const float *x; uint8_t *q; float *xs; int m, k; } qrow_arg;

/* Per-row asymmetric u8 quantisation (zero point 128).  The CPU kernel path
 * uses one scale per 128-wide block; the matrix-engine kernel accumulates the
 * whole k in int32, which needs one scale per row (and lets the zero point be
 * folded into a single per-column constant). */
static void qrow_work(void *vp, int tid, int nt) {
    qrow_arg *a = vp;
    int r0, r1;
    e2_split(a->m, tid, nt, &r0, &r1);
    int k = a->k;
    for (int r = r0; r < r1; r++) {
        const float *xp = a->x + (size_t)r * k;
        uint8_t *qp = a->q + (size_t)r * k;
        float amax = 0.0f;
        for (int i = 0; i < k; i++) {
            float v = fabsf(xp[i]);
            if (v > amax) amax = v;
        }
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / s;
        a->xs[r] = s;
        for (int i = 0; i < k; i++) {
            int v = (int)lrintf(xp[i] * inv) + 128;
            qp[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
}

static int gpu_alloc(void **p, size_t *cap, size_t need, e2_gpu *g) {
    if (*cap >= need) return 0;
    void *n = usm_alloc(g, need ? need : 64);
    if (!n) return -1;
    if (*p) zeMemFree(g->ctx, *p);
    *p = n;
    *cap = need;
    return 0;
}

/* Quantise x (CPU) into shared USM, then transpose it into the matrix-engine
 * layout on the device.  Both halves reuse the buffers across calls. */
static int ensure_pack(e2_gpu *g, e2_ctx *ctx, const float *x, int m, int k) {
    size_t raw = (size_t)m * k;
    size_t sc = (size_t)m * sizeof(float);
    int nkc = k / 32, mblk = (m + 7) / 8;
    size_t pk = (size_t)mblk * nkc * 256;
    if (gpu_alloc((void **)&g->a_raw, &g->a_raw_cap, raw, g) != 0) return -1;
    if (gpu_alloc((void **)&g->a_scale, &g->a_scale_cap, sc, g) != 0) return -1;
    if (gpu_alloc((void **)&g->a_pack, &g->a_pack_cap, pk, g) != 0) return -1;

    qrow_arg qa = {x, g->a_raw, g->a_scale, m, k};
    if (ctx->nthreads > 1 && (long)m * k >= 131072)
        e2_parallel(ctx, qrow_work, &qa);
    else
        qrow_work(&qa, 0, 1);

    uint8_t *src = g->a_raw, *dst = g->a_pack;
    uint32_t ai = 0;
    int mm = m, kk = k;
    if (zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(src), &src) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(dst), &dst) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(int), &mm) != ZE_RESULT_SUCCESS ||
        zeKernelSetArgumentValue(g->k_pack_u8, ai++, sizeof(int), &kk) != ZE_RESULT_SUCCESS)
        return -1;
    ze_group_count_t gc = {(uint32_t)((size_t)mblk * nkc), 1, 1};
    if (zeCommandListAppendLaunchKernel(g->list, g->k_pack_u8, &gc, NULL, 0, NULL)
        != ZE_RESULT_SUCCESS)
        return -1;
    return 0;
}

/* y[i] = x @ W[i]^T for several matrices sharing one activation row block.
 * Returns 0 when the device produced every y, nonzero to fall back to the CPU
 * kernel path. */
int e2_gpu_linear_multi(e2_ctx *ctx, const float *x, int m,
                        const e2_linear *const *ls, float *const *ys, int n) {
    e2_gpu *g = ctx ? ctx->gpu : NULL;
    if (g_dpas < 0) g_dpas = getenv("E2_GPU_DPAS") != NULL;
    if (!g_dpas || !g || !g->ready || !g->k_gemm || !g->dmodule) return -1;
    if (m < 32 || !ls || !n) return -1;
    const e2_qmat *w0 = &ls[0]->w;
    if (w0->k % 32 || w0->n % 16) return -1;
    for (int i = 0; i < n; i++) {
        const e2_linear *l = ls[i];
        const e2_qmat *w = &l->w;
        if (!w->packed || w->k != w0->k) return -1;
        if (w->n % 16 || (w->blk && w->blk != 128)) return -1;
        if (l->in_min || l->in_max || l->out_min || l->out_max || w->bias) return -1;
    }
    int maxn = 0;
    for (int i = 0; i < n; i++) if (ls[i]->w.n > maxn) maxn = ls[i]->w.n;
    if (gpu_alloc((void **)&g->y, &g->y_cap, (size_t)m * maxn * sizeof(float), g) != 0)
        return -1;
    double _t0 = g_prof ? e2_now_ms() : 0;
    if (ensure_pack(g, ctx, x, m, w0->k) != 0) return -1;
    if (g_prof) g_t_up += e2_now_ms() - _t0;

    for (int i = 0; i < n; i++) {
        const e2_qmat *w = &ls[i]->w;
        e2_gpu_wc *wc = upload_weight(g, w);
        if (!wc || !wc->pq || !wc->wsumt) return -1;
        int k = w->k, nn = w->n;
        uint8_t *xp = g->a_pack;
        float *xs = g->a_scale;
        uint8_t *wp = (uint8_t *)wc->pq;
        float *ws = wc->scale;
        int32_t *wt = wc->wsumt;
        float *yp = g->y;
        uint32_t ai = 0;
        if (zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(xp), &xp) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(xs), &xs) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(wp), &wp) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(ws), &ws) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(wt), &wt) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(yp), &yp) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(int), &m) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(int), &k) != ZE_RESULT_SUCCESS ||
            zeKernelSetArgumentValue(g->k_gemm, ai++, sizeof(int), &nn) != ZE_RESULT_SUCCESS)
            return -1;
        /* one sub-group per 8 work items; RB=4 x CT=2 tiles per sub-group */
        uint32_t subs = (uint32_t)(((m + 31) / 32) * (nn / 16));
        ze_group_count_t gc = {subs, 1, 1};
        double _t1 = g_prof ? e2_now_ms() : 0;
        if (zeCommandListAppendLaunchKernel(g->list, g->k_gemm, &gc, g->evt, 0, NULL)
            != ZE_RESULT_SUCCESS)
            return -1;
        if (g_prof) { g_t_sub += e2_now_ms() - _t1; _t1 = e2_now_ms(); }
        if (wait_evt(g) != 0) return -1;
        if (zeEventHostReset(g->evt) != ZE_RESULT_SUCCESS) return -1;
        if (g_prof) { g_t_sync += e2_now_ms() - _t1; _t1 = e2_now_ms(); }
        memcpy(ys[i], g->y, (size_t)m * nn * sizeof(float));
        if (g_prof) { g_t_copy += e2_now_ms() - _t1; g_n_call++; }
    }
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
    if (wait_evt(g) != 0) return -1;
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
