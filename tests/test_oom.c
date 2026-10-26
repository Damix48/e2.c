/* Linker fault injection; no allocator hooks in the engine API. */
#include "model.h"
#include "q8cache.h"
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
static int g_fail_at, g_count;
void *__wrap_malloc(size_t n) {
    if (g_fail_at && ++g_count == g_fail_at) return NULL;
    return __real_malloc(n);
}
void *__wrap_calloc(size_t n, size_t s) {
    if (g_fail_at && ++g_count == g_fail_at) return NULL;
    return __real_calloc(n, s);
}
static void put_u32le(uint8_t *p, uint32_t v) {
    for (int b = 0; b < 4; b++) p[b] = (uint8_t)(v >> (8*b));
}
static void audio_failures(void) {
    uint8_t wav[44 + 2048*2] = {0};
    memcpy(wav, "RIFF", 4); put_u32le(wav + 4, sizeof wav - 8);
    memcpy(wav + 8, "WAVEfmt ", 8); put_u32le(wav + 16, 16);
    wav[20] = wav[22] = 1;
    put_u32le(wav + 24, 48000); put_u32le(wav + 28, 96000);
    wav[32] = 2; wav[34] = 16;
    memcpy(wav + 36, "data", 4); put_u32le(wav + 40, 2048*2);
    float *out = malloc(sizeof(float) * E2_AUDIO_MAX_SAMPLES); assert(out);
    out[0] = out[E2_AUDIO_MAX_SAMPLES - 1] = 12345;
    for (int i = 1; i <= 2; i++) {
        g_count = 0; g_fail_at = i;
        assert(e2_audio_preprocess_wav(wav, sizeof wav, out) == -1 && e2_last_error()[0]);
        g_fail_at = 0;
        assert(out[0] == 12345 && out[E2_AUDIO_MAX_SAMPLES - 1] == 12345);
    }
#if defined(E2_HAVE_LIBSNDFILE)
    for (int i = 1; i <= 3; i++) {
        g_count = 0; g_fail_at = i;
        assert(e2_audio_preprocess(wav, sizeof wav, out) == -1 && e2_last_error()[0]);
        g_fail_at = 0;
        assert(out[0] == 12345 && out[E2_AUDIO_MAX_SAMPLES - 1] == 12345);
    }
#endif
    assert(e2_audio_preprocess(wav, sizeof wav, out) == 683 && !e2_last_error()[0]);
    free(out);
}
int e2_text_init(e2_model *m);
static void task(void *arg, int tid, int nt) { (void)tid; (void)nt; *(int *)arg += 1; }
int main(void) {
    audio_failures();
    for (int i = 1; i <= 4; i++) {
        e2_model m = { .rope_theta_sliding = 10000.f, .rope_theta_full = 1000000.f };
        g_count = 0; g_fail_at = i;
        assert(e2_text_init(&m) == -1 && e2_last_error()[0]);
        g_fail_at = 0;
        for (int j = 0; j < 2; j++) { free(m.rope_cos[j]); free(m.rope_sin[j]); }
    }
    for (int i = 1; i <= 3; i++) {
        g_count = 0; g_fail_at = i;
        assert(e2_pool_create(2) == NULL);
        g_fail_at = 0;
    }
    char path[] = "/tmp/e2-oom-cache-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    assert(write(fd, "old", 3) == 3); close(fd);
    for (int i = 1; i <= 2; i++) {
        g_count = 0; g_fail_at = i;
        assert(q8c_write_open(path) == -1);
        g_fail_at = 0;
        FILE *f = fopen(path, "rb"); assert(f);
        char old[4] = {0}; assert(fread(old, 1, 3, f) == 3); fclose(f);
        assert(!strcmp(old, "old"));
    }
    unlink(path);
    assert(!setenv("E2_THREADS", "2", 1));
    g_count = 0; g_fail_at = 2;
    assert(e2_pool_global_init(2) == 1);
    g_fail_at = 0;
    int runs = 0; e2_parallel(task, &runs); assert(runs == 1);
    e2_pool_global_free();
    puts("audio/RoPE/pool allocation failures and serial fallback OK");
    return 0;
}
