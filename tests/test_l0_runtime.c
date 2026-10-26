/* Runtime failure injection without production test hooks. Each failed worker exits, never retries. */
#include "gpu/l0.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int g_fault, g_executions, g_frees;
static const void *g_caller;
static int g_caller_used;

ze_result_t __real_zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t, uint32_t,
    ze_command_list_handle_t *, ze_fence_handle_t);
ze_result_t __wrap_zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t q, uint32_t n,
    ze_command_list_handle_t *lists, ze_fence_handle_t fence) {
    g_executions++;
    if (g_fault == 1) return ZE_RESULT_SUCCESS; /* success, no observable execution */
    if (g_fault == 2) return ZE_RESULT_ERROR_DEVICE_LOST;
    return __real_zeCommandQueueExecuteCommandLists(q, n, lists, fence);
}
ze_result_t __real_zeCommandQueueSynchronize(ze_command_queue_handle_t, uint64_t);
ze_result_t __wrap_zeCommandQueueSynchronize(ze_command_queue_handle_t q, uint64_t timeout) {
    assert(timeout == E2_L0_TIMEOUT_NS);
    if (g_fault == 3) return ZE_RESULT_NOT_READY;
    if (g_fault == 4) return ZE_RESULT_ERROR_DEVICE_LOST;
    return __real_zeCommandQueueSynchronize(q, timeout);
}
ze_result_t __real_zeCommandListAppendMemoryCopy(ze_command_list_handle_t, void *, const void *,
    size_t, ze_event_handle_t, uint32_t, ze_event_handle_t *);
ze_result_t __wrap_zeCommandListAppendMemoryCopy(ze_command_list_handle_t l, void *dst,
    const void *src, size_t n, ze_event_handle_t signal, uint32_t nw, ze_event_handle_t *wait) {
    if (g_caller && (src == g_caller || dst == g_caller)) g_caller_used = 1;
    return __real_zeCommandListAppendMemoryCopy(l, dst, src, n, signal, nw, wait);
}
ze_result_t __real_zeMemFree(ze_context_handle_t, void *);
ze_result_t __wrap_zeMemFree(ze_context_handle_t c, void *p) {
    g_frees++;
    return __real_zeMemFree(c, p);
}

static int worker(int mode) {
    char err[512];
    e2_l0 *g = e2_l0_create(err, sizeof err);
    if (!g) {
        if (!strcmp(err, "no Level Zero driver") || !strcmp(err, "no Intel GPU device found")) return 77;
        fprintf(stderr, "create: %s\n", err); return 1;
    }
    const size_t bytes = (8u << 20) + 19; /* crosses both staging boundaries, with an unaligned tail */
    unsigned char *src = malloc(bytes), *out = malloc(bytes);
    assert(src && out);
    for (size_t i = 0; i < bytes; i++) src[i] = (unsigned char)(i * 17 + 31);
    void *dev = e2_l0_alloc(g, bytes);
    float *dst = e2_l0_alloc_shared(g, 256 * sizeof(float));
    assert(dev && dst);
    g_caller = src;
    assert(!e2_l0_write(g, dev, bytes, src));
    g_caller = out;
    assert(!e2_l0_read(g, dev, bytes, out));
    assert(!g_caller_used && !memcmp(src, out, bytes));
    e2_graph *gr = e2_l0_graph_begin(g, 1);
    assert(gr);
    const float value = .75f;
    assert(!e2_l0_append(g, gr, "k_bench_write", 1, 1, 1, 64, 1, 1, 2,
                       &dst, sizeof dst, &value, sizeof value));
    assert(!e2_l0_graph_close(g, gr));
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < 256; i++) dst[i] = -1;
        assert(!e2_l0_graph_run(g, gr));
        for (int i = 0; i < 256; i++) assert(dst[i] == value);
    }
    if (mode) {
        g_fault = (mode - 1) % 4 + 1;
        int rc;
        if (mode <= 4) rc = e2_l0_graph_run(g, gr);
        else if (mode <= 8) { g_caller = src; rc = e2_l0_write(g, dev, bytes, src); }
        else { g_caller = out; rc = e2_l0_read(g, dev, bytes, out); }
        assert(rc && !g_caller_used && !e2_l0_ready(g));
        assert(strstr(e2_l0_error(g), "restart worker"));
        if (g_fault == 1) assert(strstr(e2_l0_error(g), "completion cookie"));
        if (g_fault == 3) assert(strstr(e2_l0_error(g), "timeout"));
        const int before = g_executions;
        assert(e2_l0_graph_run(g, gr));
        assert(e2_l0_write(g, dev, bytes, src));
        assert(e2_l0_read(g, dev, bytes, out));
        assert(!e2_l0_graph_begin(g, 1) && !e2_l0_alloc(g, 16));
        assert(g_executions == before); /* no retry or hidden context recovery */
        g_frees = 0;
        e2_l0_graph_free(g, gr);
        e2_l0_free(g, dev);
        e2_l0_free(g, dst);
        e2_l0_destroy(g);
        assert(!g_frees); /* timed-out/uncertain commands may still reference those allocations */
    } else {
        e2_l0_graph_free(g, gr);
        e2_l0_free(g, dev);
        e2_l0_free(g, dst);
        e2_l0_destroy(g);
    }
    free(src); free(out);
    return 0;
}

int main(void) {
    /* Parent never initializes Level Zero. Children model disposable workers, including retained DMA
     * resources after timeout; _exit deliberately leaves those resources to process teardown. */
    for (int mode = 0; mode <= 12; mode++) {
        pid_t pid = fork(); assert(pid >= 0);
        if (!pid) { int rc = worker(mode); fflush(NULL); _exit(rc); }
        int status;
        pid_t r;
        do r = waitpid(pid, &status, 0); while (r < 0 && errno == EINTR);
        assert(r == pid);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 77) return 77;
        if (!WIFEXITED(status) || WEXITSTATUS(status)) {
            fprintf(stderr, "GPU runtime case %d failed (status %d)\n", mode, status); return 1;
        }
    }
    puts("GPU graph/copy completion, timeout, retirement and owned staging: passed");
    return 0;
}
