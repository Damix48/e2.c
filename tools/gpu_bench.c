/* gpu_bench - dev-only kernel benchmark for the matrix-engine GEMM.
 * Not part of libe2.a; build with `make gpu-bench`.
 *
 * Sweeps the shape knobs the launcher can set (sub-groups per work group, the
 * k-split mode) over the tensor shapes the text tower actually uses, checks
 * every result against the CPU int8 kernel and prints MAC/s.  This is the loop
 * the kernel gets tuned in; the engine then just picks the winning setting.
 *
 *   ./gpu_bench            # all shapes
 *   ./gpu_bench 512 512 2048   # one shape (m k n)
 */
#include "e2.h"
#include "src/e2_internal.h"
#include "src/e2_gpu_l0.h"
#include "src/e2_gpu_pack.h"
#include "src/e2_gpu_spv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static ze_driver_handle_t g_drv;
static ze_device_handle_t g_dev;
static ze_context_handle_t g_ctx;
static ze_command_queue_handle_t g_queue;
static ze_command_list_handle_t g_list;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void ck(ze_result_t r, const char *what) {
    if (r == ZE_RESULT_SUCCESS) return;
    fprintf(stderr, "L0 %s failed: 0x%x\n", what, (unsigned)r);
    exit(1);
}

static void *usm(size_t n) {
    ze_device_mem_alloc_desc_t dd;
    ze_host_mem_alloc_desc_t hd;
    memset(&dd, 0, sizeof(dd));
    memset(&hd, 0, sizeof(hd));
    dd.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    hd.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void *p = NULL;
    ck(zeMemAllocShared(g_ctx, &dd, &hd, n, 64, g_dev, &p), "alloc");
    return p;
}

static float frand(void) {
    static uint32_t s = 12345;
    s = s * 1103515245u + 12345u;
    return (float)((s >> 8) & 0xffff) / 32768.0f - 1.0f;
}

typedef struct {
    ze_kernel_handle_t k;
    int gy;         /* sub-groups per work group */
    int ksplit;
    int tm, tn;     /* tiles a work group covers */
} variant;

static double run_variant(const variant *v, ze_group_count_t gc,
                          const uint8_t *xq, const float *xsc, const int8_t *pq,
                          const float *wsc, float *y,
                          int m, int k, int n, int nb, int reps) {
    uint32_t i = 0;
    ck(zeKernelSetGroupSize(v->k, 8, (uint32_t)v->gy, 1), "gs");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(xq), &xq), "a0");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(xsc), &xsc), "a1");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(pq), &pq), "a2");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(wsc), &wsc), "a3");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(y), &y), "a5");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(m), &m), "a6");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(k), &k), "a7");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(n), &n), "a8");
    int xs = k, xss = nb, ys = n;
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(xs), &xs), "a9");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(xss), &xss), "a10");
    ck(zeKernelSetArgumentValue(v->k, i++, sizeof(ys), &ys), "a11");
    /* `reps` launches inside one submission: the engine runs one submission per
     * forward, so the steady-state kernel time is what matters, not the
     * submit+sync latency (which is ~40 us and would dominate a single
     * launch). */
    double best = 1e9;
    for (int r = 0; r < 6; r++) {
        ck(zeCommandListReset(g_list), "reset");
        int bar = getenv("E2_GB_BARRIER") != NULL;
        for (int i = 0; i < reps; i++) {
            ck(zeCommandListAppendLaunchKernel(g_list, v->k, &gc, NULL, 0, NULL), "app");
            if (bar) ck(zeCommandListAppendBarrier(g_list, NULL, 0, NULL), "bar");
        }
        ck(zeCommandListClose(g_list), "close");
        double t0 = now_ms();
        ck(zeCommandQueueExecuteCommandLists(g_queue, 1, &g_list, NULL), "exec");
        ck(zeCommandQueueSynchronize(g_queue, UINT64_MAX), "sync");
        double dt = (now_ms() - t0) / reps;
        if (dt < best) best = dt;
    }
    return best;
}

static int run_shape(int m, int k, int n, int iters) {
    /* weights + reference */
    float *w = malloc(sizeof(float) * (size_t)n * k);
    float *x = malloc(sizeof(float) * (size_t)m * k);
    for (size_t i = 0; i < (size_t)n * k; i++) w[i] = 0.05f * frand();
    for (size_t i = 0; i < (size_t)m * k; i++) x[i] = frand() * 3.0f;
    int8_t *wq = NULL;
    float *wsc = NULL;
    int32_t *wsum = NULL;
    int blk = 0;
    if (e2_pack_mat(w, n, k, k, 1, &wq, &wsc, &wsum, &blk) != 0) { free(w); free(x); return -1; }
    e2_qmat qm = { wq, wsc, wsum, NULL, n, k, 1, blk, 0, 0 };
    const int nblk = k / (blk ? blk : 128);
    uint8_t *xq = malloc((size_t)m * k);
    float *xsc = malloc(sizeof(float) * (size_t)m * nblk);
    e2_quant_rows_blk(x, m, k, xq, xsc, blk ? blk : 128);

    float *yref = malloc(sizeof(float) * (size_t)m * n);
    e2_gemm_range(yref, n, xq, xsc, nblk, m, k, &qm, 0, (n + 15) / 16);

    /* centred int8 activations (q - 128) and the transposed form, both what the
     * kernel wants; the reference above used the u8 form, same arithmetic */
    int8_t *s8 = malloc((size_t)m * k);
    const int mpad = (m + 7) & ~7;
    int8_t *s8t = malloc((size_t)mpad * k);
    for (int r = 0; r < m; r++)
        for (int c = 0; c < k; c++) {
            int8_t v = (int8_t)((int)xq[(size_t)r * k + c] - 128);
            s8[(size_t)r * k + c] = v;
            s8t[((size_t)(c / 32) * mpad + r) * 32 + (c % 32)] = v;
        }
    /* device copies */
    size_t mpad_hint = 0; (void)mpad_hint;
    uint8_t *dxq = usm((size_t)m * k + (size_t)mpad * k);
    float *dxsc = usm(sizeof(float) * (size_t)m * nblk);
    int8_t *dpq = usm((size_t)n * k);
    float *dwsc = usm(sizeof(float) * n);
    float *dy = usm(sizeof(float) * (size_t)m * n);
    memcpy(dxq, s8, (size_t)m * k);
    if (getenv("E2_GB_XACT")) {
        memcpy(dxq, s8t, (size_t)mpad * k);
        mpad_hint = (size_t)mpad;
    }
    memcpy(dxsc, xsc, sizeof(float) * (size_t)m * nblk);
    memcpy(dwsc, wsc, sizeof(float) * n);
    if (e2_gpu_dpas_pack(&qm, dpq) != 0) { fprintf(stderr, "shape %dx%dx%d: no dpas\n", m, k, n); return 0; }

    ze_module_desc_t md;
    memset(&md, 0, sizeof(md));
    md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    md.format = ZE_MODULE_FORMAT_IL_SPIRV;
    md.inputSize = e2_gpu_spv_e2_xmx_len;
    md.pInputModule = e2_gpu_spv_e2_xmx;
    /* E2_GB_SPV: test a .spv built with other tile macros without touching
     * the checked-in header.  E2_GB_BUILD: IGC build flags (GRF mode). */
    static unsigned char *spv_buf = NULL;
    if (getenv("E2_GB_SPV")) {
        FILE *f = fopen(getenv("E2_GB_SPV"), "rb");
        if (!f) { fprintf(stderr, "no %s\n", getenv("E2_GB_SPV")); return -1; }
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        spv_buf = malloc((size_t)n);
        if (fread(spv_buf, 1, (size_t)n, f) != (size_t)n) return -1;
        fclose(f);
        md.inputSize = (size_t)n;
        md.pInputModule = spv_buf;
    }
    md.pBuildFlags = getenv("E2_GB_BUILD");
    ze_module_handle_t mod = NULL;
    ze_module_build_log_handle_t log = NULL;
    ze_result_t r = zeModuleCreate(g_ctx, g_dev, &md, &mod, &log);
    if (r != ZE_RESULT_SUCCESS) {
        char buf[4096] = "";
        size_t sz = sizeof(buf) - 1;
        if (log) zeModuleBuildLogGetString(log, &sz, buf);
        fprintf(stderr, "module: %s\n", buf);
        return -1;
    }
    /* the quantiser from the ops module, checked against the CPU one */
    {
        ze_module_desc_t om;
        memset(&om, 0, sizeof(om));
        om.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        om.format = ZE_MODULE_FORMAT_IL_SPIRV;
        om.inputSize = e2_gpu_spv_e2_ops_len;
        om.pInputModule = e2_gpu_spv_e2_ops;
        ze_module_handle_t omod = NULL;
        ze_kernel_desc_t qd;
        memset(&qd, 0, sizeof(qd));
        qd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        qd.pKernelName = "k_quant_s8";
        ze_kernel_handle_t qk = NULL;
        ck(zeModuleCreate(g_ctx, g_dev, &om, &omod, NULL), "ops module");
        ck(zeKernelCreate(omod, &qd, &qk), "quant kernel");
        ck(zeKernelSetGroupSize(qk, 64, 1, 1), "quant gs");
        int8_t *dq = usm((size_t)m * k);
        float *dsc = usm(sizeof(float) * (size_t)m * nblk);
        float *dx = usm(sizeof(float) * (size_t)m * k);
        memcpy(dx, x, sizeof(float) * (size_t)m * k);
        int rows = m, kk = k, blk2 = blk ? blk : 128;
        uint32_t groups = (uint32_t)(((long)m * nblk * 32 + 63) / 64);
        uint32_t qi = 0;
        ck(zeKernelSetArgumentValue(qk, qi++, sizeof(dx), &dx), "q0");
        ck(zeKernelSetArgumentValue(qk, qi++, sizeof(dq), &dq), "q1");
        ck(zeKernelSetArgumentValue(qk, qi++, sizeof(dsc), &dsc), "q2");
        ck(zeKernelSetArgumentValue(qk, qi++, sizeof(rows), &rows), "q3");
        ck(zeKernelSetArgumentValue(qk, qi++, sizeof(kk), &kk), "q4");
        ck(zeKernelSetArgumentValue(qk, qi++, sizeof(blk2), &blk2), "q5");
        ck(zeCommandListReset(g_list), "r");
        ze_group_count_t qgc = { groups, 1, 1 };
        ck(zeCommandListAppendLaunchKernel(g_list, qk, &qgc, NULL, 0, NULL), "ql");
        ck(zeCommandListClose(g_list), "c");
        ck(zeCommandQueueExecuteCommandLists(g_queue, 1, &g_list, NULL), "e");
        ck(zeCommandQueueSynchronize(g_queue, UINT64_MAX), "s");
        int bad = 0;
        for (size_t i = 0; i < (size_t)m * k; i++)
            if (dq[i] != s8[i]) bad++;
        printf("quant_s8: %d/%zu differ from the CPU quantiser (q[0]=%d want %d)\n",
               bad, (size_t)m * k, dq[0], s8[0]);
        zeKernelDestroy(qk);
        zeModuleDestroy(omod);
    }
    ze_kernel_desc_t kd;
    memset(&kd, 0, sizeof(kd));
    kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kd.pKernelName = "k_gemm_xmx";
    ze_kernel_handle_t kern = NULL;
    ck(zeKernelCreate(mod, &kd, &kern), "kernel");

    /* matrix-engine issue ceiling, same tile shape, no loads */
    {
        ze_kernel_desc_t pd;
        memset(&pd, 0, sizeof(pd));
        pd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        pd.pKernelName = "k_xmx_probe";
        ze_kernel_handle_t pk = NULL;
        ck(zeKernelCreate(mod, &pd, &pk), "probe");
        float *po = usm(8192 * sizeof(float));
        int nrep = 1024;
        ck(zeKernelSetGroupSize(pk, 8, 8, 1), "probe gs");
        ck(zeKernelSetArgumentValue(pk, 0, sizeof(po), &po), "pa0");
        ck(zeKernelSetArgumentValue(pk, 1, sizeof(nrep), &nrep), "pa1");
        ze_group_count_t pgc = { 1024, 1, 1 };
        double best = 1e9;
        for (int r = 0; r < 6; r++) {
            double t0 = now_ms();
            ck(zeCommandListReset(g_list), "reset");
            ck(zeCommandListAppendLaunchKernel(g_list, pk, &pgc, NULL, 0, NULL), "app");
            ck(zeCommandListClose(g_list), "close");
            ck(zeCommandQueueExecuteCommandLists(g_queue, 1, &g_list, NULL), "exec");
            ck(zeCommandQueueSynchronize(g_queue, UINT64_MAX), "sync");
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        /* 1024 WGs x 8 sub-groups x nrep x E2_XTB*E2_XFB DPAS x 2048 MAC */
        double n_dpas = 1024.0 * 8 * nrep * 2 * 2;
        printf("probe: %.3f ms -> %.2f G DPAS/s = %.1f TMAC/s peak\n",
               best, n_dpas / best / 1e6, n_dpas * 2048 / best / 1e9);
        zeKernelDestroy(pk);
        zeMemFree(g_ctx, po);
    }
    double gm = (double)m * k * n / 1e9;
    printf("m=%d k=%d n=%d (%.3f GMAC)\n", m, k, n, gm);
    double best = 1e9;
    const int nb = k / 128;
    const char *etm = getenv("E2_GB_TM"), *etn = getenv("E2_GB_TN");
    const int tm = etm ? atoi(etm) : E2_GPU_TM, tn = etn ? atoi(etn) : E2_GPU_TN;
    {
        {
            const char *egy = getenv("E2_GB_GY");
            int gy = egy ? atoi(egy) : E2_GPU_SG;
            variant v = { kern, gy, 0, tm, tn };
            ze_group_count_t gc;
            gc.groupCountX = (uint32_t)((m + v.tm - 1) / v.tm);
            gc.groupCountY = (uint32_t)((n + v.tn - 1) / v.tn);
            gc.groupCountZ = 1;
            memset(dy, 0, sizeof(float) * (size_t)m * n);
            double ms = run_variant(&v, gc, dxq, dxsc, dpq, dwsc, dy, m, k, n, nblk, iters);
            /* correctness */
            double dot = 0, na = 0, nbb = 0, mx = 0;
            for (size_t i = 0; i < (size_t)m * n; i++) {
                double a = yref[i], b = dy[i];
                dot += a * b; na += a * a; nbb += b * b;
                double d = fabs(a - b);
                if (d > mx) mx = d;
            }
            double cos = dot / sqrt(na * nbb + 1e-30);
            if (getenv("E2_GB_DIAG")) {
                /* which reference does the device output actually match? */
                double cT = 0, nT = 0;
                for (size_t r = 0; r < (size_t)m; r++)
                    for (size_t c = 0; c < (size_t)n; c++) {
                        if (r >= (size_t)n || c >= (size_t)m) continue;
                        double a = yref[(size_t)c * n + r], b = dy[r * n + c];
                        cT += a * b; nT += a * a;
                    }
                printf("    diag: cosT=%.6f nb=%d blk=%d\n",
                       cT / sqrt(nT * nbb + 1e-30), nb, blk);
                if (m >= 1) {
                    printf("    ref[0] :");
                    for (int c = 0; c < (n < 40 ? n : 40); c++) printf(" %7.4f", yref[c]);
                    printf("\n    dev[0] :");
                    for (int c = 0; c < (n < 40 ? n : 40); c++) printf(" %7.4f", dy[c]);
                    printf("\n    ref[1] :");
                    for (int c = 0; c < (n < 40 ? n : 40); c++) printf(" %7.4f", yref[n + c]);
                    printf("\n    dev[1] :");
                    for (int c = 0; c < (n < 40 ? n : 40); c++) printf(" %7.4f", dy[n + c]);
                    printf("\n");
                }
            }
            printf("  gy=%2d  %8.3f ms  %7.1f GMAC/s  cos=%.7f maxerr=%.2e\n",
                   gy, ms, gm / ms * 1e3, cos, mx);
            if (cos > 0.9999 && ms < best) best = ms;
        }
    }
    printf("  -> best %.3f ms (%.1f GMAC/s)\n", best, gm / best * 1e3);

    /* k_gemm_z: the head-batched form, nd = 2 writing the same tile */
    {
        ze_kernel_desc_t md;
        memset(&md, 0, sizeof(md));
        md.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        md.pKernelName = "k_gemm_z";
        ze_kernel_handle_t mk = NULL;
        if (zeKernelCreate(mod, &md, &mk) == ZE_RESULT_SUCCESS) {
            ck(zeKernelSetGroupSize(mk, 8, E2_GPU_SG, 1), "mgs");
            float *y2 = usm(sizeof(float) * (size_t)m * n);
            int nd = 2, zrep = 1;
            long z0 = 0;
            uint32_t mi = 0;
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(dxq), &dxq), "md0");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(dxsc), &dxsc), "md1");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(dpq), &dpq), "md2");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(dwsc), &dwsc), "md3");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(y2), &y2), "md4");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(m), &m), "md5");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(k), &k), "md6");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(n), &n), "md7");
            int xs = k, xss = nb, ys = n;
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(xs), &xs), "md8");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(xss), &xss), "md9");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(ys), &ys), "md10");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(long), &z0), "md11");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(long), &z0), "md12");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(long), &z0), "md13");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(long), &z0), "md14");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(long), &z0), "md15");
            ck(zeKernelSetArgumentValue(mk, mi++, sizeof(zrep), &zrep), "md16");
            memset(y2, 0, sizeof(float) * (size_t)m * n);
            ze_group_count_t mgc = {(uint32_t)((m + E2_GPU_TM - 1) / E2_GPU_TM),
                                    (uint32_t)((n + E2_GPU_TN - 1) / E2_GPU_TN),
                                    (uint32_t)nd};
            ck(zeCommandListReset(g_list), "mr");
            ck(zeCommandListAppendLaunchKernel(g_list, mk, &mgc, NULL, 0, NULL), "ml");
            ck(zeCommandListClose(g_list), "mc");
            ck(zeCommandQueueExecuteCommandLists(g_queue, 1, &g_list, NULL), "me");
            ck(zeCommandQueueSynchronize(g_queue, UINT64_MAX), "ms");
            int bad = 0;
            for (size_t i = 0; i < (size_t)m * n; i++) if (dy[i] != y2[i]) bad++;
            printf("  z-batch(nd=2): %d/%zu differ from k_gemm_xmx\n", bad,
                   (size_t)m * n);
            zeKernelDestroy(mk);
            zeMemFree(g_ctx, y2);
        } else {
            printf("  z-batch: kernel missing\n");
        }
    }
    zeKernelDestroy(kern);
    zeModuleDestroy(mod);
    return 0;
}

int main(int argc, char **argv) {
    if (e2_gpu_l0_open() != 0) { fprintf(stderr, "no Level Zero\n"); return 1; }
    ck(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");
    uint32_t nd = 0;
    ck(zeDriverGet(&nd, NULL), "drv");
    ze_driver_handle_t *dv = calloc(nd, sizeof(*dv));
    ck(zeDriverGet(&nd, dv), "drv2");
    g_drv = dv[0];
    uint32_t ne = 0;
    ck(zeDeviceGet(g_drv, &ne, NULL), "dev");
    ze_device_handle_t *de = calloc(ne, sizeof(*de));
    ck(zeDeviceGet(g_drv, &ne, de), "dev2");
    for (uint32_t i = 0; i < ne; i++) {
        ze_device_properties_t p;
        memset(&p, 0, sizeof(p));
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (zeDeviceGetProperties(de[i], &p) == ZE_RESULT_SUCCESS &&
            p.type == ZE_DEVICE_TYPE_GPU) {
            g_dev = de[i];
            printf("GPU: %s\n", p.name);
            break;
        }
    }
    if (!g_dev) { fprintf(stderr, "no GPU device\n"); return 1; }
    ze_context_desc_t cd;
    memset(&cd, 0, sizeof(cd));
    cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ck(zeContextCreate(g_drv, &cd, &g_ctx), "ctx");
    ze_command_queue_desc_t qd;
    memset(&qd, 0, sizeof(qd));
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    ck(zeCommandQueueCreate(g_ctx, g_dev, &qd, &g_queue), "queue");
    ze_command_list_desc_t ld;
    memset(&ld, 0, sizeof(ld));
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ck(zeCommandListCreate(g_ctx, g_dev, &ld, &g_list), "list");

    int iters = getenv("E2_GB_REPS") ? atoi(getenv("E2_GB_REPS")) : 40;
    if (argc >= 4) {
        run_shape(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), iters);
    } else {
        /* the shapes the 512-token text tower uses, plus the two extremes */
        static const int shapes[][3] = {
            {   8,  512,  512 }, {   8,  512, 2048 }, {   8, 1024,  512 },
            {  64,  512, 2048 }, { 128,  512, 2048 }, { 512,  512, 1024 },
            { 512,  512, 2048 }, { 512, 2048,  512 }, { 512, 1024,  512 },
            { 512,  512, 3072 }, { 270,  768,  768 }, { 512,  768,  768 },
            {2048,  512, 2048 }, {2048, 2048,  512 }, {1024, 1152,  512 },
        };
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
            run_shape(shapes[i][0], shapes[i][1], shapes[i][2], iters);
    }
    return 0;
}
