/* l0.c — see l0.h. */
#include <time.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "l0.h"

struct e2_l0 {
    ze_driver_handle_t driver;
    ze_device_handle_t dev;
    ze_context_handle_t ctx;
    ze_command_queue_handle_t queue;
    ze_module_handle_t mod;
    struct { char name[48]; ze_kernel_handle_t k; } kernels[64];
    int nkernels;
    uint32_t eus, clock;
    uint64_t timer_res; /* device timestamp cycles per second */
    int prof;
    int failed, pending;
    void *stage; /* owned copy scratch: a timed-out transfer must not reference caller memory */
    uint64_t *copy_done;
    char name[256];
    char err[2048];
};

struct e2_graph {
    ze_command_list_handle_t list;
    int barrier; /* append a barrier after each kernel: the graph is strictly sequential */
    int prof;
    void *ts; /* shared uint64: 2 timestamps per kernel */
    uint64_t *done; /* [expected, observed], the final command copies expected into observed */
    int n;    /* kernels profiled */
    int cap;
    char (*names)[24];
};

const char *e2_l0_error(e2_l0 *g) { return g->err; }

void e2_l0_set_error(e2_l0 *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g->err, sizeof g->err, fmt, ap);
    va_end(ap);
}

int e2_l0_ready(e2_l0 *g) { return !g->failed; }

int e2_l0_invalidate(e2_l0 *g, const char *reason) {
    if (!g->failed) e2_l0_set_error(g, "%s; GPU context unusable, restart worker", reason);
    g->failed = 1;
    return -1;
}

#define L0_FAIL(g, expr, r_) do { \
    e2_l0_set_error(g, "%s -> 0x%x", #expr, r_); } while (0)
#define L0_OK(g, expr) do { ze_result_t r_ = (expr); \
    if (r_ != ZE_RESULT_SUCCESS) { L0_FAIL(g, expr, r_); return -1; } } while (0)
#define L0_OK_P(g, expr) do { ze_result_t r_ = (expr); \
    if (r_ != ZE_RESULT_SUCCESS) { L0_FAIL(g, expr, r_); goto fail; } } while (0)

const char *e2_l0_device_name(e2_l0 *g) { return g->name; }
uint32_t e2_l0_eu_count(e2_l0 *g) { return g->eus; }
uint32_t e2_l0_clock_mhz(e2_l0 *g) { return g->clock; }
ze_context_handle_t e2_l0_context(e2_l0 *g) { return g->ctx; }

e2_l0 *e2_l0_create(char *err, size_t errn) {
    e2_l0 *g = calloc(1, sizeof *g);
    if (!g) { snprintf(err, errn, "oom"); return NULL; }
    if (zeInit(0) != ZE_RESULT_SUCCESS) { e2_l0_set_error(g, "zeInit failed"); goto fail; }
    uint32_t nd = 0;
    if (zeDriverGet(&nd, NULL) != ZE_RESULT_SUCCESS || !nd) { e2_l0_set_error(g, "no Level Zero driver"); goto fail; }
    ze_driver_handle_t *drivers = malloc(sizeof(ze_driver_handle_t) * nd);
    if (!drivers) { e2_l0_set_error(g, "oom for the driver list"); goto fail; }
    if (zeDriverGet(&nd, drivers) != ZE_RESULT_SUCCESS) {
        free(drivers);
        e2_l0_set_error(g, "zeDriverGet failed");
        goto fail;
    }
    uint32_t best = 0;
    int found = 0;
    for (uint32_t d = 0; d < nd; d++) {
        uint32_t n = 0;
        if (zeDeviceGet(drivers[d], &n, NULL) != ZE_RESULT_SUCCESS || !n) continue;
        ze_device_handle_t *devs = malloc(sizeof(ze_device_handle_t) * n);
        if (!devs) { free(drivers); e2_l0_set_error(g, "oom for the device list"); goto fail; }
        if (zeDeviceGet(drivers[d], &n, devs) != ZE_RESULT_SUCCESS) {
            free(devs);
            continue;
        }
        for (uint32_t i = 0; i < n; i++) {
            ze_device_properties_t p = { .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES };
            zeDeviceGetProperties(devs[i], &p);
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            const char *want = getenv("E2_GPU"); /* optional name substring filter */
            if (want && !strstr(p.name, want)) continue;
            /* prefer the device with the most EUs (the iGPU here is the only GPU) */
            uint32_t eus = p.numSlices * p.numSubslicesPerSlice * p.numEUsPerSubslice;
            if (!found || eus > g->eus) {
                g->dev = devs[i];
                g->driver = drivers[d];
                g->eus = eus;
                g->clock = p.coreClockRate;
                g->timer_res = p.timerResolution;
                snprintf(g->name, sizeof g->name, "%s", p.name);
                best = eus;
                found = 1;
            }
        }
        free(devs);
    }
    free(drivers);
    if (!found) { e2_l0_set_error(g, "no Intel GPU device found"); goto fail; }
    (void)best;
    ze_context_desc_t cdesc = { .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC };
    L0_OK_P(g, zeContextCreate(g->driver, &cdesc, &g->ctx));
    ze_command_queue_desc_t qdesc = { .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC };
    qdesc.ordinal = 0;
    qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    L0_OK_P(g, zeCommandQueueCreate(g->ctx, g->dev, &qdesc, &g->queue));
    g->prof = getenv("E2_GPU_PROFILE") != NULL;
    if (e2_l0_reload(g)) goto fail;
    return g;
fail:
    if (!g->err[0]) snprintf(g->err, sizeof g->err, "Level Zero init failed");
    snprintf(err, errn, "%s", g->err);
    e2_l0_destroy(g);
    return NULL;
}

void e2_l0_destroy(e2_l0 *g) {
    if (!g) return;
    /* A timeout is not cancellation: the device may still use every allocation and command list. */
    if (!g->pending) {
        e2_l0_free(g, g->stage);
        e2_l0_free(g, g->copy_done);
        for (int i = 0; i < g->nkernels; i++) zeKernelDestroy(g->kernels[i].k);
        if (g->mod) zeModuleDestroy(g->mod);
        if (g->queue) zeCommandQueueDestroy(g->queue);
        if (g->ctx) zeContextDestroy(g->ctx);
    }
    free(g);
}

int e2_l0_reload(e2_l0 *g) {
    if (g->failed) return -1;
    size_t n = 0;
    const unsigned char *spv = e2_l0_spirv(&n);
    if (!spv || !n) { e2_l0_set_error(g, "no embedded SPIR-V"); return -1; }
    for (int i = 0; i < g->nkernels; i++) zeKernelDestroy(g->kernels[i].k);
    g->nkernels = 0;
    if (g->mod) { zeModuleDestroy(g->mod); g->mod = NULL; }
    ze_module_desc_t mdesc = { .stype = ZE_STRUCTURE_TYPE_MODULE_DESC };
    mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    /* Only bulk GEMMs and the fused FFN need 256 GRFs; avoid halving occupancy elsewhere. */
    mdesc.pBuildFlags = "-ze-opt-large-grf-kernel k_gemm_dpas_large";
    mdesc.inputSize = n;
    mdesc.pInputModule = spv;
    ze_module_build_log_handle_t log = NULL;
    ze_result_t r = zeModuleCreate(g->ctx, g->dev, &mdesc, &g->mod, &log);
    if (r != ZE_RESULT_SUCCESS) {
        char buf[1024] = {0};
        size_t len = sizeof buf - 1;
        if (log) zeModuleBuildLogGetString(log, &len, buf);
        e2_l0_set_error(g, "zeModuleCreate -> 0x%x: %s", r, buf);
        if (log) zeModuleBuildLogDestroy(log);
        return -1;
    }
    if (log) zeModuleBuildLogDestroy(log);
    return 0;
}

ze_kernel_handle_t e2_l0_kernel(e2_l0 *g, const char *name) {
    if (g->failed) return NULL;
    for (int i = 0; i < g->nkernels; i++)
        if (!strcmp(g->kernels[i].name, name)) return g->kernels[i].k;
    if (g->nkernels >= (int)(sizeof g->kernels / sizeof g->kernels[0])) {
        e2_l0_set_error(g, "kernel table full");
        return NULL;
    }
    ze_kernel_desc_t kdesc = { .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC };
    kdesc.pKernelName = name;
    ze_kernel_handle_t k = NULL;
    ze_result_t r = zeKernelCreate(g->mod, &kdesc, &k);
    if (r != ZE_RESULT_SUCCESS) {
        e2_l0_set_error(g, "zeKernelCreate(%s) -> 0x%x", name, r);
        return NULL;
    }
    snprintf(g->kernels[g->nkernels].name, sizeof g->kernels[0].name, "%s", name);
    g->kernels[g->nkernels].k = k;
    g->nkernels++;
    return k;
}

void *e2_l0_alloc(e2_l0 *g, size_t bytes) {
    if (g->failed) return NULL;
    if (!bytes) bytes = 8;
    ze_device_mem_alloc_desc_t desc = { .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
    void *p = NULL;
    if (zeMemAllocDevice(g->ctx, &desc, bytes, 64, g->dev, &p) != ZE_RESULT_SUCCESS) {
        e2_l0_set_error(g, "zeMemAllocDevice(%zu) failed", bytes);
        return NULL;
    }
    return p;
}

void e2_l0_free(e2_l0 *g, void *p) { if (p && !g->pending) zeMemFree(g->ctx, p); }

static void *e2_l0_alloc_host(e2_l0 *g, size_t bytes) {
    if (g->failed) return NULL;
    if (!bytes) bytes = 8;
    ze_host_mem_alloc_desc_t desc = { .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };
    void *p = NULL;
    if (zeMemAllocHost(g->ctx, &desc, bytes, 64, &p) != ZE_RESULT_SUCCESS) {
        e2_l0_set_error(g, "zeMemAllocHost(%zu) failed", bytes);
        return NULL;
    }
    return p;
}

/* Memory both sides touch: the driver guarantees device access to shared memory, while host memory is
 * only promised to the CPU (reading token ids from it produced garbage rather than an error). */
void *e2_l0_alloc_shared(e2_l0 *g, size_t bytes) {
    if (g->failed) return NULL;
    if (!bytes) bytes = 8;
    ze_host_mem_alloc_desc_t hdesc = { .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };
    ze_device_mem_alloc_desc_t ddesc = { .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
    void *p = NULL;
    if (zeMemAllocShared(g->ctx, &ddesc, &hdesc, bytes, 64, g->dev, &p) != ZE_RESULT_SUCCESS) {
        e2_l0_set_error(g, "zeMemAllocShared(%zu) failed", bytes);
        return NULL;
    }
    return p;
}

static ze_result_t append_completion(ze_command_list_handle_t list, uint64_t *done) {
    ze_result_t r = zeCommandListAppendBarrier(list, NULL, 0, NULL);
    if (r == ZE_RESULT_SUCCESS)
        r = zeCommandListAppendMemoryCopy(list, done + 1, done, sizeof *done, NULL, 0, NULL);
    return r;
}

static int run_list(e2_l0 *g, ze_command_list_handle_t list, uint64_t *done) {
    if (g->failed) return -1;
    volatile uint64_t *cookie = done;
    cookie[0] = cookie[0] + 1;
    cookie[1] = ~cookie[0];
    g->pending = 1;
    ze_result_t r = zeCommandQueueExecuteCommandLists(g->queue, 1, &list, NULL);
    if (r == ZE_RESULT_SUCCESS) r = zeCommandQueueSynchronize(g->queue, E2_L0_TIMEOUT_NS);
    if (r != ZE_RESULT_SUCCESS) {
        char reason[128];
        snprintf(reason, sizeof reason, "GPU execute/wait -> 0x%x%s", r,
                 r == ZE_RESULT_NOT_READY ? " (30 s timeout)" : "");
        return e2_l0_invalidate(g, reason);
    }
    if (cookie[1] != cookie[0])
        return e2_l0_invalidate(g, "GPU submission has no fresh completion cookie");
    g->pending = 0;
    return 0;
}

/* Bounded, owned staging also keeps a timed-out DMA away from the caller's freed input/output. */
#define E2_COPY_BYTES (4u << 20)
static int copy_and_run(e2_l0 *g, void *device, void *host, size_t bytes, int reading) {
    if (g->failed) return -1;
    if (!g->stage) g->stage = e2_l0_alloc_host(g, E2_COPY_BYTES);
    if (!g->copy_done) {
        g->copy_done = e2_l0_alloc_shared(g, 2 * sizeof(uint64_t));
        if (g->copy_done) memset(g->copy_done, 0, 2 * sizeof(uint64_t));
    }
    if (!g->stage || !g->copy_done) return -1;
    for (size_t off = 0; off < bytes;) {
        const size_t n = bytes - off < E2_COPY_BYTES ? bytes - off : E2_COPY_BYTES;
        void *dev = (char *)device + off, *p = (char *)host + off;
        if (!reading) memcpy(g->stage, p, n);
        ze_command_list_desc_t desc = { .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC };
        ze_command_list_handle_t list = NULL;
        L0_OK(g, zeCommandListCreate(g->ctx, g->dev, &desc, &list));
        ze_result_t r = zeCommandListAppendMemoryCopy(list, reading ? g->stage : dev,
                                                     reading ? dev : g->stage, n, NULL, 0, NULL);
        if (r == ZE_RESULT_SUCCESS) r = append_completion(list, g->copy_done);
        if (r == ZE_RESULT_SUCCESS) r = zeCommandListClose(list);
        int rc = -1;
        if (r == ZE_RESULT_SUCCESS) rc = run_list(g, list, g->copy_done);
        else e2_l0_set_error(g, "recording memory copy -> 0x%x", r);
        if (!g->pending) zeCommandListDestroy(list);
        if (rc) return -1;
        if (reading) memcpy(p, g->stage, n);
        off += n;
    }
    return 0;
}

int e2_l0_write(e2_l0 *g, void *dst, size_t bytes, const void *src) {
    return copy_and_run(g, dst, (void *)src, bytes, 0);
}

int e2_l0_read(e2_l0 *g, const void *src, size_t bytes, void *dst) {
    return copy_and_run(g, (void *)src, dst, bytes, 1);
}

e2_graph *e2_l0_graph_begin(e2_l0 *g, int barrier) {
    if (g->failed) return NULL;
    g->err[0] = 0;
    e2_graph *gr = calloc(1, sizeof *gr);
    if (!gr) { e2_l0_set_error(g, "oom"); return NULL; }
    gr->barrier = barrier;
    gr->prof = g->prof;
    ze_command_list_desc_t ldesc = { .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC };
    if (zeCommandListCreate(g->ctx, g->dev, &ldesc, &gr->list) != ZE_RESULT_SUCCESS) {
        e2_l0_set_error(g, "zeCommandListCreate failed");
        free(gr);
        return NULL;
    }
    gr->done = e2_l0_alloc_shared(g, 2 * sizeof(uint64_t));
    if (!gr->done) {
        zeCommandListDestroy(gr->list);
        free(gr);
        return NULL;
    }
    memset(gr->done, 0, 2 * sizeof(uint64_t));
    if (gr->prof) {
        gr->cap = 1024;
        gr->ts = e2_l0_alloc_shared(g, sizeof(uint64_t) * (size_t)gr->cap * 2 + 16);
        gr->names = calloc((size_t)gr->cap, sizeof gr->names[0]);
        if (!gr->ts || !gr->names) {
            e2_l0_set_error(g, "oom for the profile buffer");
            free(gr->names);
            if (gr->ts) e2_l0_free(g, gr->ts);
            e2_l0_free(g, gr->done);
            zeCommandListDestroy(gr->list);
            free(gr);
            return NULL;
        }
    }
    return gr;
}

int e2_l0_append(e2_l0 *g, e2_graph *gr, const char *kname, uint32_t gx, uint32_t gy, uint32_t gz,
                 uint32_t lx, uint32_t ly, uint32_t lz, int nargs, ...) {
    ze_kernel_handle_t k = e2_l0_kernel(g, kname);
    if (!k) return -1;
    if (zeKernelSetGroupSize(k, lx, ly, lz) != ZE_RESULT_SUCCESS) {
        e2_l0_set_error(g, "%s: group size %ux%ux%u rejected", kname, lx, ly, lz);
        return -1;
    }
    va_list ap;
    va_start(ap, nargs);
    for (int i = 0; i < nargs; i++) {
        const void *p = va_arg(ap, const void *);
        size_t n = va_arg(ap, size_t);
        ze_result_t r = zeKernelSetArgumentValue(k, (uint32_t)i, n, p);
        if (r != ZE_RESULT_SUCCESS) {
            va_end(ap);
            e2_l0_set_error(g, "%s: argument %d -> 0x%x", kname, i, r);
            return -1;
        }
    }
    va_end(ap);
    ze_group_count_t groups = { gx, gy, gz };
    ze_result_t r = ZE_RESULT_SUCCESS;
    if (gr->prof && gr->n < gr->cap) {
        uint64_t *ts = gr->ts;
        const ze_result_t t0 =
            zeCommandListAppendWriteGlobalTimestamp(gr->list, ts + 2 * gr->n, NULL, 0, NULL);
        if (t0 != ZE_RESULT_SUCCESS) {
            /* Timestamp writes are not supported on this queue: stop profiling and take the plain
             * single-launch path below. Nothing has been appended for this kernel yet. */
            gr->prof = 0;
            e2_l0_set_error(g, "%s: profile timestamp -> 0x%x", kname, t0);
        } else {
            /* From here the kernel is in the list, so no path may append it a second time: a doubly
             * appended kernel runs twice, which silently changes anything that is not idempotent. */
            const ze_result_t lr =
                zeCommandListAppendLaunchKernel(gr->list, k, &groups, NULL, 0, NULL);
            if (lr != ZE_RESULT_SUCCESS) {
                e2_l0_set_error(g, "%s: launch -> 0x%x", kname, lr);
                return -1;
            }
            const ze_result_t t1 =
                zeCommandListAppendWriteGlobalTimestamp(gr->list, ts + 2 * gr->n + 1, NULL, 0, NULL);
            if (t1 == ZE_RESULT_SUCCESS) {
                if (gr->barrier) L0_OK(g, zeCommandListAppendBarrier(gr->list, NULL, 0, NULL));
                snprintf(gr->names[gr->n], 24, "%s", kname);
                gr->n++;
                return 0;
            }
            gr->prof = 0; /* the launch stays; the rest of the graph is timed no further */
            e2_l0_set_error(g, "%s: profile timestamp -> 0x%x", kname, t1);
            if (gr->barrier) L0_OK(g, zeCommandListAppendBarrier(gr->list, NULL, 0, NULL));
            return 0;
        }
    }
    r = zeCommandListAppendLaunchKernel(gr->list, k, &groups, NULL, 0, NULL);
    if (r == ZE_RESULT_SUCCESS && gr->barrier)
        r = zeCommandListAppendBarrier(gr->list, NULL, 0, NULL);
    if (r != ZE_RESULT_SUCCESS) {
        e2_l0_set_error(g, "%s: append -> 0x%x", kname, r);
        return -1;
    }
    return 0;
}

int e2_l0_graph_close(e2_l0 *g, e2_graph *gr) {
    if (g->failed) return -1;
    L0_OK(g, append_completion(gr->list, gr->done));
    L0_OK(g, zeCommandListClose(gr->list));
    return 0;
}

/* One submission, then wait for the queue. The queue is in-order and single, so synchronising it is
 * both simpler and more reliable than a fence: a fence created before the first execute does not
 * track that submission on this driver, which silently turns a wait into a no-op and lets host reads
 * race the kernels (that cost an afternoon - the GEMM looked like it wrote zeros). */
int e2_l0_graph_run(e2_l0 *g, e2_graph *gr) {
    if (run_list(g, gr->list, gr->done)) return -1;
    if (gr->prof) e2_l0_graph_report(g, gr);
    return 0;
}

/* Aggregate the device timestamps of the last run by kernel name. The timestamps are device timer
 * cycles; timerResolution is cycles/second. */
void e2_l0_graph_report(e2_l0 *g, e2_graph *gr) {
    if (!gr || !gr->prof || !gr->n) return;
    const uint64_t *ts = gr->ts;
    /* timerResolution is unreliable here; calibrate the device timer against the host clock once. */
    static double cyc_per_ns = 0;
    if (cyc_per_ns == 0) {
        uint64_t h1 = 0, d1 = 0, h2 = 0, d2 = 0;
        zeDeviceGetGlobalTimestamps(g->dev, &h1, &d1);
        struct timespec req = { 0, 20000000 };
        nanosleep(&req, NULL);
        zeDeviceGetGlobalTimestamps(g->dev, &h2, &d2);
        cyc_per_ns = (h2 > h1 && d2 > d1) ? (double)(d2 - d1) / (double)(h2 - h1) : 1.0;
    }
    const double cyc2ms = 1.0 / (cyc_per_ns * 1e6);
    char (*names)[24] = calloc(1024, sizeof *names);
    double *tot = calloc(1024, sizeof *tot);
    int *cnt = calloc(1024, sizeof *cnt);
    long *minv = calloc(1024, sizeof *minv);
    long *maxv = calloc(1024, sizeof *maxv);
    if (!names || !tot || !cnt || !minv || !maxv) {
        free(names); free(tot); free(cnt); free(minv); free(maxv);
        return;
    }
    int nn = 0;
    for (int i = 0; i < gr->n; i++) {
        const long d = (long)(ts[2 * i + 1] - ts[2 * i]);
        int j;
        for (j = 0; j < nn; j++)
            if (!strcmp(names[j], gr->names[i])) break;
        if (j == nn) { snprintf(names[nn], 24, "%s", gr->names[i]); minv[nn] = d; maxv[nn] = d; nn++; }
        tot[j] += d * cyc2ms;
        cnt[j]++;
        if (d < minv[j]) minv[j] = d;
        if (d > maxv[j]) maxv[j] = d;
    }
    double sum = 0;
    for (int j = 0; j < nn; j++) sum += tot[j];
    printf("[gpu prof] %d kernels, %.3f ms (sum of kernel times)\n", gr->n, sum);
    for (int j = 0; j < nn; j++)
        printf("[gpu prof] %-18s n=%4d  %8.3f ms  %6.1f%%  avg %7.1f us  min %6.1f  max %6.1f\n",
               names[j], cnt[j], tot[j], 100.0 * tot[j] / (sum > 0 ? sum : 1), 1e3 * tot[j] / cnt[j],
               minv[j] * cyc2ms * 1e3, maxv[j] * cyc2ms * 1e3);
    free(names); free(tot); free(cnt); free(minv); free(maxv);
}

void e2_l0_graph_free(e2_l0 *g, e2_graph *gr) {
    if (!gr) return;
    if (!g->pending) zeCommandListDestroy(gr->list);
    e2_l0_free(g, gr->done);
    /* Not gated on gr->prof: append() clears that flag when timestamp writes turn out to be
     * unsupported, and the buffers it allocated are still here. */
    if (gr->ts) e2_l0_free(g, gr->ts);
    free(gr->names);
    free(gr);
}
