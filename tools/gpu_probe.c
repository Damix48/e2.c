/* gpu_probe - standalone Level Zero microbenchmark for the packed-int8 GEMM.
 * Not part of libe2.a; build with `make gpu-probe`.
 *
 * dlopens libze_loader.so.1 (no Level Zero headers or link dependency),
 * enumerates the Intel iGPU, loads the checked-in SPIR-V module and times the
 * packed-int8 GEMM against the CPU kernel from libe2.a on the same USM
 * buffers.
 */
#include "e2.h"
#include "src/e2_internal.h"
#include "src/e2_gpu_l0.h"
#include "src/e2_gpu_spv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static ze_driver_handle_t g_drv;
static ze_device_handle_t g_dev;
static ze_context_handle_t g_ctx;
static ze_command_queue_handle_t g_queue;
static ze_command_list_handle_t g_list;
static ze_module_handle_t g_module;

static void l0_ck(ze_result_t r, const char *what) {
    if (r == ZE_RESULT_SUCCESS) return;
    fprintf(stderr, "L0 %s failed: 0x%x\n", what, (unsigned)r);
    exit(1);
}

static void *usm_alloc(size_t n) {
    ze_device_mem_alloc_desc_t ddesc;
    ze_host_mem_alloc_desc_t hdesc;
    memset(&ddesc, 0, sizeof(ddesc));
    memset(&hdesc, 0, sizeof(hdesc));
    ddesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void *p = NULL;
    l0_ck(zeMemAllocShared(g_ctx, &ddesc, &hdesc, n, 64, g_dev, &p), "zeMemAllocShared");
    return p;
}

static ze_kernel_handle_t make_kernel(const char *name) {
    ze_kernel_desc_t kdesc;
    memset(&kdesc, 0, sizeof(kdesc));
    kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName = name;
    ze_kernel_handle_t k = NULL;
    l0_ck(zeKernelCreate(g_module, &kdesc, &k), name);
    return k;
}

static void set_gemm_args(ze_kernel_handle_t kern,
                          const uint8_t *xq, const float *xs,
                          const int8_t *wq, const float *wscale, const int32_t *wsum,
                          float *y, int m, int k, int n, int nblk, int xss, int wss,
                          size_t ts) {
    uint32_t i = 0;
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(xq), &xq), "arg xq");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(xs), &xs), "arg xs");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(wq), &wq), "arg wq");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(wscale), &wscale), "arg wscale");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(wsum), &wsum), "arg wsum");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(y), &y), "arg y");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(int), &m), "arg m");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(int), &k), "arg k");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(int), &n), "arg n");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(int), &nblk), "arg nblk");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(int), &xss), "arg xss");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(int), &wss), "arg wss");
    l0_ck(zeKernelSetArgumentValue(kern, i++, sizeof(ts), &ts), "arg ts");
}

static void launch_batch(ze_kernel_handle_t k, const ze_group_count_t *gc, int count) {
    l0_ck(zeCommandListReset(g_list), "reset");
    for (int i = 0; i < count; i++)
        l0_ck(zeCommandListAppendLaunchKernel(g_list, k, gc, NULL, 0, NULL), "append");
    l0_ck(zeCommandListClose(g_list), "close");
    l0_ck(zeCommandQueueExecuteCommandLists(g_queue, 1, &g_list, NULL), "execute");
    l0_ck(zeCommandQueueSynchronize(g_queue, UINT64_MAX), "sync");
}

static double now_ms(void) { return e2_now_ms(); }

typedef enum { V_1X4, V_4X4, V_2X8, V_SG, V_SG4, V_COUNT } variant;

static const char *variant_name(variant v) {
    switch (v) {
    case V_1X4: return "gemm_1x4";
    case V_4X4: return "gemm_4x4";
    case V_2X8: return "gemm_2x8";
    case V_SG:  return "gemm_sg";
    case V_SG4: return "gemm_sg4";
    default: return "?";
    }
}

static size_t work_items(variant v, int m, int n) {
    int ntiles = (n + 15) / 16;
    switch (v) {
    case V_1X4: return (size_t)m * ntiles * 4;
    case V_4X4: return (size_t)((m + 3) / 4) * ntiles * 4;
    case V_2X8: return (size_t)((m + 1) / 2) * ntiles * 2;
    case V_SG:  return (size_t)m * ntiles * 16;
    case V_SG4: return (size_t)((m + 3) / 4) * ntiles * 16;
    default: return 0;
    }
}

static int subgroup_kernel(variant v) { return v == V_SG || v == V_SG4; }

/* L2-normalized cosine between two m*n vectors. */
static double cosine(const float *a, const float *b, size_t n) {
    double d = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; i++) {
        d += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    if (aa == 0 || bb == 0) return 0;
    return d / (sqrt(aa) * sqrt(bb));
}

static void max_err(const float *a, const float *b, size_t n, double *mx, double *mae) {
    double m = 0, s = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)a[i] - b[i]);
        if (e > m) m = e;
        s += e;
    }
    *mx = m; *mae = s / (n ? n : 1);
}

/* The shapes that matter: text qkv/mlp, vision mlp, big-m projection, audio. */
static const struct { int m, k, n; } shapes[] = {
    {8, 768, 1536},
    {64, 512, 2048},
    {512, 512, 2048},
    {2048, 2048, 512},
    {1024, 768, 768},
    {266, 768, 768},
};

typedef struct {
    uint8_t *xq; float *xs;
    int8_t *wq; float *wscale; int32_t *wsum;
    float *y; float *y_cpu;
} buffers;

static void run_shape(int m, int k, int n, variant v, ze_kernel_handle_t kern,
                      buffers *b, e2_qmat *wmat) {
    int nblk = k / 128, xss = nblk, wss = nblk;
    size_t ts = (size_t)(k / 4) * 64;
    int ntiles = (n + 15) / 16;
    size_t yn = (size_t)m * n;
    set_gemm_args(kern, b->xq, b->xs, b->wq, b->wscale, b->wsum, b->y,
                  m, k, n, nblk, xss, wss, ts);

    size_t wi = work_items(v, m, n);
    int ls_best = 0;
    double best = 1e18;
    int ls_list[] = {1, 16, 32, 64, 128, 256};
    for (int li = 0; li < 6; li++) {
        int ls = ls_list[li];
        if (subgroup_kernel(v) && (ls % 16)) continue;
        if (ls == 1 && subgroup_kernel(v)) continue;
        l0_ck(zeKernelSetGroupSize(kern, (uint32_t)ls, 1, 1), "group size");
        size_t groups = (wi + ls - 1) / ls;
        if (groups == 0) groups = 1;
        ze_group_count_t gc = {(uint32_t)groups, 1, 1};
        /* warm */
        launch_batch(kern, &gc, 2);
        /* sustained throughput over one submitted batch */
        int batch = 8;
        double t0 = now_ms();
        launch_batch(kern, &gc, batch);
        double t = (now_ms() - t0) / batch;
        if (t < best) { best = t; ls_best = ls; }
    }
    double gmac = (double)m * n * k / 1e9;
    double gmacs = gmac / (best / 1e3);

    /* correctness: one fresh launch into zeroed y, vs the CPU kernel */
    memset(b->y, 0, yn * sizeof(float));
    memset(b->y_cpu, 0, yn * sizeof(float));
    l0_ck(zeKernelSetGroupSize(kern, (uint32_t)(ls_best ? ls_best : 1), 1, 1), "gs");
    size_t groups = (wi + ls_best - 1) / ls_best;
    if (groups == 0) groups = 1;
    ze_group_count_t gc = {(uint32_t)groups, 1, 1};
    launch_batch(kern, &gc, 1);
    double t0 = now_ms();
    e2_gemm_range(b->y_cpu, n, b->xq, b->xs, xss, m, k, wmat, 0, ntiles);
    double cpu_ms = now_ms() - t0;
    double cs = cosine(b->y, b->y_cpu, yn);
    double mx, mae;
    max_err(b->y, b->y_cpu, yn, &mx, &mae);

    printf("  %-11s m=%4d k=%4d n=%4d ls=%3d  %.3f ms  %7.1f GMAC/s  "
           "cpu=%.3f ms  cos=%.6f  maxerr=%.2e\n",
           variant_name(v), m, k, n, ls_best, best, gmacs, cpu_ms, cs, mx);
}

/* Fixed cost of one submission: same kernel, N launches per submission. */
static void bench_latency(void) {
    enum { M = 512, K = 512, N = 2048 };
    buffers b; memset(&b, 0, sizeof(b));
    int xss = K / 128, wss = K / 128;
    size_t ts = (size_t)(K / 4) * 64;
    int nblk = K / 128;
    size_t xqn = (size_t)M * K, ysn = (size_t)M * nblk, yn = (size_t)M * N;
    b.xq = usm_alloc(xqn);
    b.xs = usm_alloc(ysn * sizeof(float));
    b.wq = usm_alloc((size_t)N * K);
    b.wscale = usm_alloc((size_t)N * sizeof(float));
    b.wsum = usm_alloc((size_t)N * nblk * sizeof(int32_t));
    b.y = usm_alloc(yn * sizeof(float));
    memset(b.wq, 0, (size_t)N * K);
    memset(b.wsum, 0, (size_t)N * nblk * sizeof(int32_t));
    for (size_t i = 0; i < xqn; i++) b.xq[i] = (uint8_t)((i * 37u + (i >> 9)) & 0xff);
    for (size_t i = 0; i < ysn; i++) b.xs[i] = 0.5e-3f + 1e-6f * (float)(i % 1000);
    ze_kernel_handle_t kern = make_kernel("gemm_4x4");
    set_gemm_args(kern, b.xq, b.xs, b.wq, b.wscale, b.wsum, b.y,
                  M, K, N, nblk, xss, wss, ts);
    size_t wi = work_items(V_4X4, M, N);
    l0_ck(zeKernelSetGroupSize(kern, 64, 1, 1), "lat gs");
    ze_group_count_t gc = {(uint32_t)((wi + 63) / 64), 1, 1};
    launch_batch(kern, &gc, 2);
    for (int nlaunch = 1; nlaunch <= 8; nlaunch *= 2) {
        int reps = 40;
        double t0 = now_ms();
        for (int r = 0; r < reps; r++) launch_batch(kern, &gc, nlaunch);
        double t = (now_ms() - t0) / reps;
        printf("latency: %d launch(es)/submit  %.3f ms/submit  %.3f ms/launch\n",
               nlaunch, t, t / nlaunch);
    }
    zeKernelDestroy(kern);
    zeMemFree(g_ctx, b.xq); zeMemFree(g_ctx, b.xs); zeMemFree(g_ctx, b.wq);
    zeMemFree(g_ctx, b.wscale); zeMemFree(g_ctx, b.wsum); zeMemFree(g_ctx, b.y);
}

static void bench_bandwidth(void) {
    size_t bytes = 256u << 20;
    int nv = (int)(bytes / 16);
    size_t count = 8192;
    int32_t *p = usm_alloc(bytes);
    for (int i = 0; i < nv * 4; i++) p[i] = (int32_t)i;
    float *out = usm_alloc(count * sizeof(float));
    ze_kernel_handle_t k = make_kernel("bw_read");
    uint32_t ai = 0;
    const int32_t *pp = p;
    l0_ck(zeKernelSetArgumentValue(k, ai++, sizeof(pp), &pp), "bw p");
    l0_ck(zeKernelSetArgumentValue(k, ai++, sizeof(out), &out), "bw out");
    int iters = 8;
    l0_ck(zeKernelSetArgumentValue(k, ai++, sizeof(int), &nv), "bw nv");
    l0_ck(zeKernelSetArgumentValue(k, ai++, sizeof(int), &iters), "bw iters");
    l0_ck(zeKernelSetGroupSize(k, 256, 1, 1), "bw gs");
    ze_group_count_t gc = {(uint32_t)(count / 256), 1, 1};
    launch_batch(k, &gc, 2);
    double t0 = now_ms();
    launch_batch(k, &gc, 1);
    double t = now_ms() - t0;
    double total = (double)bytes * iters;
    printf("streaming read: %zu MB x %d = %.1f GB in %.3f ms = %.1f GB/s\n",
           bytes >> 20, iters, total / 1e9, t, total / 1e9 / (t / 1e3));
    zeKernelDestroy(k);
    zeMemFree(g_ctx, p);
    zeMemFree(g_ctx, out);
}

int main(void) {
    if (e2_gpu_l0_open() != 0) {
        fprintf(stderr, "no Level Zero runtime (libze_loader.so.1)\n");
        return 1;
    }
    l0_ck(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");

    uint32_t ndrv = 0;
    l0_ck(zeDriverGet(&ndrv, NULL), "zeDriverGet count");
    if (!ndrv) { fprintf(stderr, "no L0 drivers\n"); return 1; }
    ze_driver_handle_t *drvs = malloc((size_t)ndrv * sizeof(*drvs));
    l0_ck(zeDriverGet(&ndrv, drvs), "zeDriverGet");

    int found = 0;
    for (uint32_t d = 0; d < ndrv && !found; d++) {
        uint32_t ndev = 0;
        l0_ck(zeDeviceGet(drvs[d], &ndev, NULL), "zeDeviceGet count");
        if (!ndev) continue;
        ze_device_handle_t *devs = malloc((size_t)ndev * sizeof(*devs));
        l0_ck(zeDeviceGet(drvs[d], &ndev, devs), "zeDeviceGet");
        for (uint32_t e = 0; e < ndev; e++) {
            ze_device_properties_t props;
            memset(&props, 0, sizeof(props));
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            l0_ck(zeDeviceGetProperties(devs[e], &props), "zeDeviceGetProperties");
            if (props.type == ZE_DEVICE_TYPE_GPU) {
                g_drv = drvs[d];
                g_dev = devs[e];
                found = 1;
                printf("GPU: %s\n", props.name);
                printf("  vendorId=0x%x deviceId=0x%x EUs=%u slices=%u subslices=%u/slice "
                       "simdWidth=%u threads/EU=%u clock=%u MHz\n",
                       props.vendorId, props.deviceId,
                       props.numEUsPerSubslice * props.numSubslicesPerSlice * props.numSlices,
                       props.numSlices, props.numSubslicesPerSlice,
                       props.physicalEUSimdWidth, props.numThreadsPerEU, props.coreClockRate);
                uint32_t nmem = 0;
                zeDeviceGetMemoryProperties(g_dev, &nmem, NULL);
                ze_device_memory_properties_t *mem = calloc(nmem ? nmem : 1, sizeof(*mem));
                if (nmem) {
                    for (uint32_t i = 0; i < nmem; i++)
                        mem[i].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
                    l0_ck(zeDeviceGetMemoryProperties(g_dev, &nmem, mem), "mem props");
                    for (uint32_t i = 0; i < nmem; i++)
                        printf("  memory[%u]: %s %.1f GB\n", i, mem[i].name,
                               (double)mem[i].totalSize / 1e9);
                }
                free(mem);
                break;
            }
        }
        free(devs);
    }
    free(drvs);
    if (!found) { fprintf(stderr, "no GPU device\n"); return 1; }

    ze_context_desc_t cdesc; memset(&cdesc, 0, sizeof(cdesc));
    cdesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    l0_ck(zeContextCreate(g_drv, &cdesc, &g_ctx), "zeContextCreate");

    ze_command_queue_desc_t qdesc; memset(&qdesc, 0, sizeof(qdesc));
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    l0_ck(zeCommandQueueCreate(g_ctx, g_dev, &qdesc, &g_queue), "zeCommandQueueCreate");

    ze_command_list_desc_t ldesc; memset(&ldesc, 0, sizeof(ldesc));
    ldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    l0_ck(zeCommandListCreate(g_ctx, g_dev, &ldesc, &g_list), "zeCommandListCreate");

    ze_module_desc_t mdesc; memset(&mdesc, 0, sizeof(mdesc));
    mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    mdesc.inputSize = e2_gpu_spv_e2_gemm_len;
    mdesc.pInputModule = e2_gpu_spv_e2_gemm;
    ze_module_build_log_handle_t log = NULL;
    ze_result_t mr = zeModuleCreate(g_ctx, g_dev, &mdesc, &g_module, &log);
    if (mr != ZE_RESULT_SUCCESS) {
        size_t sz = 0;
        zeModuleBuildLogGetString(log, &sz, NULL);
        char *buf = malloc(sz ? sz : 1);
        if (sz) zeModuleBuildLogGetString(log, &sz, buf);
        fprintf(stderr, "zeModuleCreate failed 0x%x: %s\n", (unsigned)mr, sz ? buf : "(no log)");
        free(buf);
        if (log) zeModuleBuildLogDestroy(log);
        return 1;
    }
    if (log) zeModuleBuildLogDestroy(log);

    ze_kernel_handle_t kernels[V_COUNT];
    kernels[V_1X4] = make_kernel("gemm_1x4");
    kernels[V_4X4] = make_kernel("gemm_4x4");
    kernels[V_2X8] = make_kernel("gemm_2x8");
    kernels[V_SG]  = make_kernel("gemm_sg");
    kernels[V_SG4] = make_kernel("gemm_sg4");

    /* Real packed weights via e2_pack_mat_into, all shapes. */
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int m = shapes[s].m, k = shapes[s].k, n = shapes[s].n;
        int nblk = k / 128;
        printf("shape m=%d k=%d n=%d (%.2f GMAC)\n", m, k, n, (double)m * n * k / 1e9);
        size_t xqn = (size_t)m * k, ysn = (size_t)m * nblk, yn = (size_t)m * n;
        buffers b; memset(&b, 0, sizeof(b));
        b.xq = usm_alloc(xqn);
        b.xs = usm_alloc(ysn * sizeof(float));
        b.wq = usm_alloc((size_t)n * k);
        b.wscale = usm_alloc((size_t)n * sizeof(float));
        b.wsum = usm_alloc((size_t)n * nblk * sizeof(int32_t));
        b.y = usm_alloc(yn * sizeof(float));
        b.y_cpu = malloc(yn * sizeof(float));
        memset(b.wq, 0, (size_t)n * k);
        memset(b.wsum, 0, (size_t)n * nblk * sizeof(int32_t));
        for (size_t i = 0; i < xqn; i++)
            b.xq[i] = (uint8_t)((i * 37u + (i >> 9)) & 0xff);
        for (size_t i = 0; i < ysn; i++)
            b.xs[i] = 0.5e-3f + 1e-6f * (float)(i % 1000);

        float *w = malloc((size_t)n * k * sizeof(float));
        for (size_t i = 0; i < (size_t)n * k; i++)
            w[i] = (float)((int)(i % 503) - 251) * 0.02f;
        int blk = 128;
        if (e2_pack_mat_into(NULL, w, n, k, k, 1, b.wq, b.wscale, b.wsum, &blk) != 0) {
            fprintf(stderr, "pack failed\n"); return 1;
        }
        free(w);

        e2_qmat wm; memset(&wm, 0, sizeof(wm));
        wm.q = (const int8_t *)b.wq;
        wm.scale = b.wscale;
        wm.wsum = b.wsum;
        wm.n = n; wm.k = k; wm.packed = 1; wm.blk = blk;

        for (int vi = 0; vi < V_COUNT; vi++)
            run_shape(m, k, n, (variant)vi, kernels[vi], &b, &wm);

        zeMemFree(g_ctx, b.xq); zeMemFree(g_ctx, b.xs); zeMemFree(g_ctx, b.wq);
        zeMemFree(g_ctx, b.wscale); zeMemFree(g_ctx, b.wsum); zeMemFree(g_ctx, b.y);
        free(b.y_cpu);
    }

    bench_bandwidth();
    bench_latency();

    for (int vi = 0; vi < V_COUNT; vi++) zeKernelDestroy(kernels[vi]);
    zeModuleDestroy(g_module);
    zeCommandListDestroy(g_list);
    zeCommandQueueDestroy(g_queue);
    zeContextDestroy(g_ctx);
    e2_gpu_l0_close();
    return 0;
}
