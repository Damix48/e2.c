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
int e2_text_init(e2_model *m);
static void task(void *arg, int tid, int nt) { (void)tid; (void)nt; *(int *)arg += 1; }
int main(void) {
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
    puts("RoPE/pool allocation failures and serial fallback OK");
    return 0;
}
