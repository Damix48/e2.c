/* Exercise every packed tower combination with the same binary; account for all GPU allocations. */
#include "model.h"
#include "gpu/gpu.h"
#include "test_metrics.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_live, g_calls, g_fail_at;
ze_result_t __real_zeMemAllocDevice(ze_context_handle_t, const ze_device_mem_alloc_desc_t *, size_t,
    size_t, ze_device_handle_t, void **);
ze_result_t __wrap_zeMemAllocDevice(ze_context_handle_t c, const ze_device_mem_alloc_desc_t *d,
    size_t n, size_t a, ze_device_handle_t dev, void **p) {
    if (++g_calls == g_fail_at) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    ze_result_t r = __real_zeMemAllocDevice(c, d, n, a, dev, p);
    if (r == ZE_RESULT_SUCCESS) g_live++;
    return r;
}
ze_result_t __real_zeMemAllocShared(ze_context_handle_t, const ze_device_mem_alloc_desc_t *,
    const ze_host_mem_alloc_desc_t *, size_t, size_t, ze_device_handle_t, void **);
ze_result_t __wrap_zeMemAllocShared(ze_context_handle_t c, const ze_device_mem_alloc_desc_t *d,
    const ze_host_mem_alloc_desc_t *h, size_t n, size_t a, ze_device_handle_t dev, void **p) {
    if (++g_calls == g_fail_at) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    ze_result_t r = __real_zeMemAllocShared(c, d, h, n, a, dev, p);
    if (r == ZE_RESULT_SUCCESS) g_live++;
    return r;
}
ze_result_t __real_zeMemAllocHost(ze_context_handle_t, const ze_host_mem_alloc_desc_t *, size_t,
    size_t, void **);
ze_result_t __wrap_zeMemAllocHost(ze_context_handle_t c, const ze_host_mem_alloc_desc_t *h,
    size_t n, size_t a, void **p) {
    if (++g_calls == g_fail_at) return ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    ze_result_t r = __real_zeMemAllocHost(c, h, n, a, p);
    if (r == ZE_RESULT_SUCCESS) g_live++;
    return r;
}
ze_result_t __real_zeMemFree(ze_context_handle_t, void *);
ze_result_t __wrap_zeMemFree(ze_context_handle_t c, void *p) {
    ze_result_t r = __real_zeMemFree(c, p);
    if (r == ZE_RESULT_SUCCESS) g_live--;
    return r;
}
static void match(const float *got, const float *ref) {
    assert(e2_test_unit("lifecycle", got, 768));
    assert(e2_test_match("GPU vs CPU", got, ref, 768, .999));
    assert(!e2_gpu_last_error()[0]);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    e2_model *m = e2_load(argv[1]); assert(m);
    float out[768], text[768], image[768], audio[768], pixels[9 * E2_V_PATCH] = {0}, pcm[3200] = {0};
    const int32_t ids[] = {2, 100, 101, 1};
    int32_t pos[18];
    for (int i = 0; i < 9; i++) { pos[2*i] = i % 3; pos[2*i+1] = i / 3; }
    assert(!e2_text_embed(m, ids, 4, text));
    if (m->vision) assert(!e2_image_embed(m, pixels, pos, 9, image));
    if (m->audio) assert(!e2_audio_embed(m, pcm, 3200, audio));
    int create_allocs = 0;
    for (int cycle = 0; cycle < 3; cycle++) {
        g_calls = 0;
        e2_gpu *g = e2_gpu_create(m);
        if (!g) fprintf(stderr, "GPU create: %s\n", e2_gpu_last_error());
        assert(g && !e2_gpu_last_error()[0]);
        create_allocs = g_calls;
        printf("cycle %d: weights=%zu workspace=%zu vision=%d audio=%d\n", cycle,
               g->wbuf_bytes, g->acts_bytes, !!m->vision, !!m->audio);
        assert(!e2_gpu_text_embed(g, ids, 4, out)); match(out, text);
        if (m->vision) {
            assert(!e2_gpu_image_embed(g, pixels, pos, 9, out)); match(out, image);
        } else {
            for (int i = 0; i < 768; i++) out[i] = 12345.f;
            assert(!g->vpix && !g->reg[A_VX].bytes);
            assert(e2_gpu_image_embed(g, pixels, pos, 9, out));
            assert(strstr(e2_gpu_last_error(), "vision tower not loaded"));
            assert(e2_gpu_text_image_embed(g, ids, 4, pixels, pos, 9, out));
            for (int i = 0; i < 768; i++) assert(out[i] == 12345.f);
        }
        if (m->audio) {
            assert(!e2_gpu_audio_embed(g, pcm, 3200, out)); match(out, audio);
        } else {
            assert(!g->axin && !g->asoft && !g->reg[A_AX].bytes);
            for (int i = 0; i < 768; i++) out[i] = 12345.f;
            assert(e2_gpu_audio_embed(g, pcm, 3200, out));
            assert(strstr(e2_gpu_last_error(), "no audio tower"));
            for (int i = 0; i < 768; i++) assert(out[i] == 12345.f);
        }
        assert(!e2_gpu_text_embed(g, ids, 4, out)); match(out, text); /* clears absent-tower errors */
        e2_gpu_destroy(g);
        assert(!g_live);
    }
    const int failures[] = {1, create_allocs / 2, create_allocs};
    for (size_t i = 0; i < sizeof failures / sizeof failures[0]; i++) {
        g_calls = 0; g_fail_at = failures[i];
        e2_gpu *g = e2_gpu_create(m);
        assert(!g && e2_gpu_last_error()[0] && !g_live);
        g_fail_at = 0;
    }
    e2_free(m);
    puts("GPU lifecycle, allocation failures and present/absent towers: passed");
    return 0;
}
