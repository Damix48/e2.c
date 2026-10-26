/* Host output validation must work despite -ffast-math in the GPU host library. */
#include "model.h"
#include "gpu/gpu.h"
#include "test_metrics.h"
#include "gpu_text_cases.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *g_corrupt;
static float g_bad_value;
static size_t g_corrupt_n;
static int g_executions;
ze_result_t __real_zeCommandQueueSynchronize(ze_command_queue_handle_t, uint64_t);
ze_result_t __wrap_zeCommandQueueSynchronize(ze_command_queue_handle_t q, uint64_t timeout) {
    ze_result_t r = __real_zeCommandQueueSynchronize(q, timeout);
    if (r == ZE_RESULT_SUCCESS && g_corrupt) {
        for (size_t i = 0; i < g_corrupt_n; i++) g_corrupt[i] = g_bad_value;
        g_corrupt = NULL;
    }
    return r;
}
ze_result_t __real_zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t, uint32_t,
    ze_command_list_handle_t *, ze_fence_handle_t);
ze_result_t __wrap_zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t q, uint32_t n,
    ze_command_list_handle_t *lists, ze_fence_handle_t fence) {
    g_executions++;
    return __real_zeCommandQueueExecuteCommandLists(q, n, lists, fence);
}

static void sentinel(float *out, size_t n) { for (size_t i = 0; i < n; i++) out[i] = 12345.f; }
static void untouched(const float *out, size_t n) {
    for (size_t i = 0; i < n; i++) assert(out[i] == 12345.f);
}
static e2_gpu *create(e2_model *m) {
    e2_gpu *g = e2_gpu_create(m);
    if (!g) fprintf(stderr, "GPU create: %s\n", e2_gpu_last_error());
    assert(g && !e2_gpu_last_error()[0]);
    return g;
}
static void retired(e2_gpu *g, const int32_t *ids) {
    assert(!e2_l0_ready(g->l0) && strstr(e2_gpu_last_error(), "restart worker"));
    float out[768]; sentinel(out, 768);
    const int before = g_executions;
    assert(e2_gpu_text_embed(g, ids, 4, out));
    assert(e2_gpu_text_embed_batch(g, ids, NULL, 1, 4, out));
    assert(e2_gpu_image_embed(g, NULL, NULL, 0, out));
    assert(e2_gpu_audio_embed(g, NULL, 0, out));
    assert(g_executions == before);
    untouched(out, 768);
    e2_gpu_destroy(g);
}
int main(int argc, char **argv) {
    assert(argc == 3);
    int count;
    e2_test_text_case *cases = e2_test_read_text_cases(argv[2], &count);
    assert(count >= 2 && cases[0].n == cases[1].n);
    const int seq = cases[0].n;
    int32_t *ids = malloc(2 * (size_t)seq * sizeof *ids); assert(ids);
    memcpy(ids, cases[0].ids, (size_t)seq * sizeof *ids);
    memcpy(ids + seq, cases[1].ids, (size_t)seq * sizeof *ids);
    e2_model *m = e2_load(argv[1]); assert(m);
    float out[2 * 768], ref[768];
    e2_gpu *g = create(m);
    /* Changed inputs on the same recorded graph must not replay the previous request's output. */
    for (int r = 0; r < 6; r++) {
        const int32_t *input = ids + (r % 2) * seq;
        assert(!e2_text_embed(m, input, seq, ref));
        assert(!e2_gpu_text_embed(g, input, seq, out));
        assert(e2_test_match("changed input vs FP32", out, cases[r % 2].ref, 768, .999));
        assert(e2_test_unit("changed input", out, 768));
        assert(e2_test_match("changed input", out, ref, 768, .999));
    }
    uint8_t *mask = calloc(2 * (size_t)seq, 1); assert(mask);
    memset(mask, 255, (size_t)seq); mask[0] = 2;
    assert(!e2_gpu_text_embed_batch(g, ids, mask, 2, seq, out));
    assert(e2_test_unit("live row", out, 768));
    for (int i = 768; i < 2 * 768; i++) assert(out[i] == 0.f);
    e2_gpu_destroy(g);

    const float invalid[] = {NAN, INFINITY, 0.f, 2.f};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; i++) {
        g = create(m);
        assert(!e2_gpu_text_embed(g, ids, seq, out)); /* record before injecting readback damage */
        /* Damage the second row: a valid first row must not be published on chunk failure. */
        g_corrupt = (float *)g->embout + 768;
        g_corrupt_n = i == 2 ? 768 : 1;
        g_bad_value = invalid[i];
        sentinel(out, 2 * 768);
        assert(e2_gpu_text_embed_batch(g, ids, NULL, 2, seq, out));
        assert(!g_corrupt);
        untouched(out, 2 * 768); /* validate the entire chunk before publishing any row */
        retired(g, ids);
    }
    if (m->audio) {
        g = create(m);
        float pcm[3200] = {0};
        float *soft = malloc((size_t)E2_A_MAX_TOK * 512 * sizeof(float)); assert(soft);
        int n = 0;
        assert(!e2_gpu_audio_soft(g, pcm, 3200, soft, &n));
        sentinel(soft, (size_t)E2_A_MAX_TOK * 512);
        g_corrupt = g->asoft; g_corrupt_n = 1; g_bad_value = NAN; n = -1;
        assert(e2_gpu_audio_soft(g, pcm, 3200, soft, &n));
        assert(!g_corrupt && n == -1);
        untouched(soft, (size_t)E2_A_MAX_TOK * 512);
        retired(g, ids);
        free(soft);
    }
    e2_free(m);
    free(mask); free(ids); free(cases);
    puts("GPU changed-input replay, masked zeros, invalid outputs and context retirement: passed");
    return 0;
}
