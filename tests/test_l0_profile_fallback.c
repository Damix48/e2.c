/* test_l0_profile_fallback.c — the E2_GPU_PROFILE append path (src/gpu/l0.c).
 *
 * Profiling brackets every kernel with device timestamp writes, and each of those appends can fail on
 * its own. What must not happen is a second launch of the same kernel: once zeCommandListAppendLaunchKernel
 * has succeeded the kernel runs, so a "fall back to a plain launch" after a later failure would run it
 * twice and silently change anything that is not idempotent. The three timestamp outcomes and a bare
 * launch failure are injected by wrapping the driver entry points at link time (no test hooks in the
 * production code); the profile buffers must be released even when append() disabled profiling.
 *
 * Needs a GPU but no model: skips with 77 when there is no device.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <level_zero/ze_api.h>

#include "gpu/l0.h"

/* ---- link-time doubles for the Level Zero entry points the append path uses */
static int ts_calls, launch_calls, barrier_calls, frees;
static int fail_ts_on_call;   /* 1-based index of the zeCommandListAppendWriteGlobalTimestamp call
                              * that fails; 0 = none */
static int fail_launch, fail_barrier;

ze_result_t __real_zeCommandListAppendWriteGlobalTimestamp(ze_command_list_handle_t, uint64_t *,
                                                          ze_event_handle_t, uint32_t,
                                                          ze_event_handle_t *);
ze_result_t __real_zeCommandListAppendLaunchKernel(ze_command_list_handle_t, ze_kernel_handle_t,
                                                  const ze_group_count_t *, ze_event_handle_t,
                                                  uint32_t, ze_event_handle_t *);
ze_result_t __real_zeCommandListAppendBarrier(ze_command_list_handle_t, ze_event_handle_t, uint32_t,
                                             ze_event_handle_t *);
ze_result_t __real_zeMemFree(ze_context_handle_t, void *);

ze_result_t __wrap_zeCommandListAppendWriteGlobalTimestamp(ze_command_list_handle_t l, uint64_t *p,
                                                           ze_event_handle_t s, uint32_t nw,
                                                           ze_event_handle_t *w) {
    if (++ts_calls == fail_ts_on_call) return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
    return __real_zeCommandListAppendWriteGlobalTimestamp(l, p, s, nw, w);
}

ze_result_t __wrap_zeCommandListAppendLaunchKernel(ze_command_list_handle_t l, ze_kernel_handle_t k,
                                                   const ze_group_count_t *g, ze_event_handle_t s,
                                                   uint32_t nw, ze_event_handle_t *w) {
    launch_calls++;
    if (fail_launch) return ZE_RESULT_ERROR_UNKNOWN;
    return __real_zeCommandListAppendLaunchKernel(l, k, g, s, nw, w);
}

ze_result_t __wrap_zeCommandListAppendBarrier(ze_command_list_handle_t l, ze_event_handle_t s,
                                              uint32_t nw, ze_event_handle_t *w) {
    barrier_calls++;
    if (fail_barrier) return ZE_RESULT_ERROR_UNKNOWN;
    return __real_zeCommandListAppendBarrier(l, s, nw, w);
}

ze_result_t __wrap_zeMemFree(ze_context_handle_t c, void *p) {
    frees++;
    return __real_zeMemFree(c, p);
}

/* Skip only explicit absence messages, not module or initialization failures. */
static int no_gpu(const char *err) {
    return !strcmp(err, "no Level Zero driver") || !strcmp(err, "no Intel GPU device found");
}

/* ---- the case matrix */
static e2_l0 *g_l0;
static void *dev_a, *dev_b;
static int failures;

static void check(int cond, const char *what) {
    if (cond) return;
    printf("FAIL %s (ts_calls=%d launch_calls=%d): %s\n", what, ts_calls, launch_calls,
           e2_l0_error(g_l0));
    failures++;
}

static void reset(int ts, int launch, int barrier) {
    e2_l0_set_error(g_l0, "%s", "");
    ts_calls = launch_calls = barrier_calls = 0;
    fail_ts_on_call = ts;
    fail_launch = launch;
    fail_barrier = barrier;
}

/* one recorded kernel: k_copy_rows from dev_a to dev_b */
static int append_copy(e2_graph *gr) {
    void *dst = dev_b, *src = dev_a;
    const int rows = 2, hid = 512;
    return e2_l0_append(g_l0, gr, "k_copy_rows", (uint32_t)((rows + 3) / 4),
                        (uint32_t)(hid / 4 / 64), 1, 64, 4, 1, 4, &dst, sizeof dst, &src, sizeof src,
                        &rows, sizeof rows, &hid, sizeof hid);
}

int main(void) {
    setenv("E2_GPU_PROFILE", "1", 1); /* profiling is what the impaired path belongs to */
    char err[512] = { 0 };
    g_l0 = e2_l0_create(err, sizeof err);
    if (!g_l0) {
        if (no_gpu(err)) {
            printf("skip: %s\n", err);
            return 77;
        }
        fprintf(stderr, "FAIL create: %s\n", err[0] ? err : "(no message)");
        return 1;
    }
    dev_a = e2_l0_alloc(g_l0, 4096);
    dev_b = e2_l0_alloc(g_l0, 4096);
    if (!dev_a || !dev_b) { printf("FAIL allocation\n"); return 1; }

    /* 1. the opening timestamp is rejected: nothing is queued yet, so profiling stops and this kernel
     *    (and the next one) is launched exactly once through the plain path. */
    {
        e2_graph *gr = e2_l0_graph_begin(g_l0, 1);
        reset(1, 0, 0);
        check(append_copy(gr) == 0, "start-timestamp failure: append succeeds");
        check(ts_calls == 1 && launch_calls == 1, "start-timestamp failure: one timestamp, one launch");
        check(e2_l0_error(g_l0)[0] != 0, "start-timestamp failure: reported");
        reset(0, 0, 0);
        check(append_copy(gr) == 0, "after the fallback: append succeeds");
        check(ts_calls == 0 && launch_calls == 1, "after the fallback: profiling off, one launch");
        check(e2_l0_graph_close(g_l0, gr) == 0, "after the fallback: close");
        e2_l0_graph_free(g_l0, gr);
    }

    /* 2. the launch itself is rejected: the error propagates, it is not retried. */
    {
        e2_graph *gr = e2_l0_graph_begin(g_l0, 1);
        reset(0, 1, 0);
        check(append_copy(gr) == -1, "launch failure: append fails");
        check(launch_calls == 1, "launch failure: not retried");
        check(strstr(e2_l0_error(g_l0), "k_copy_rows: launch ->") != NULL, "launch failure: reported");
        e2_l0_graph_free(g_l0, gr);
    }

    /* 3. the closing timestamp is rejected after the launch went in: the kernel must stay launched
     *    once. This is the regression: the old fallback appended it again. */
    {
        e2_graph *gr = e2_l0_graph_begin(g_l0, 1);
        reset(2, 0, 0);
        check(append_copy(gr) == 0, "end-timestamp failure: append succeeds");
        check(launch_calls == 1, "end-timestamp failure: the kernel is launched once, not twice");
        check(ts_calls == 2, "end-timestamp failure: both timestamps attempted");
        check(e2_l0_graph_close(g_l0, gr) == 0, "end-timestamp failure: close");
        /* 5. profiling was switched off on this graph, so its timestamp buffer must still be freed */
        frees = 0;
        e2_l0_graph_free(g_l0, gr);
        check(frees == 2, "profile buffer and completion cookie freed although profiling was switched off");
    }

    /* 4. a rejected barrier propagates as a failure and is not turned into another launch. */
    {
        e2_graph *gr = e2_l0_graph_begin(g_l0, 1);
        reset(0, 0, 1);
        check(append_copy(gr) == -1, "barrier failure: append fails");
        check(launch_calls == 1 && barrier_calls >= 1, "barrier failure: one launch, barrier attempted");
        fail_barrier = 0;
        e2_l0_graph_free(g_l0, gr);
    }

    /* a graph that kept profiling frees its timestamp buffer as well (the same path, other branch) */
    {
        e2_graph *gr = e2_l0_graph_begin(g_l0, 1);
        reset(0, 0, 0);
        check(append_copy(gr) == 0, "profiled graph: append succeeds");
        check(ts_calls == 2 && launch_calls == 1, "profiled graph: both timestamps around one launch");
        frees = 0;
        e2_l0_graph_free(g_l0, gr);
        check(frees == 2, "profiled graph: timestamp buffer and completion cookie freed");
    }

    e2_l0_free(g_l0, dev_a);
    e2_l0_free(g_l0, dev_b);
    e2_l0_destroy(g_l0);
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("OK\n");
    return 0;
}
