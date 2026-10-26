/* test_lifecycle.c — one live model per process, and load/free returns the heap.
 *
 * The packed-cache mapping and the worker pool are process-global, so a second e2_load must be
 * refused without disturbing the live model, and a failed load must leave no session state behind.
 * The heap half checks the other direction: the int8 weights of the text tower are ~570 MB of malloc
 * per load and used to leak (three cycles retained 143 -> 285 -> 427 MB).
 *
 * Usage: e2_test_lifecycle <model_dir> [cache_file]
 *   with a cache: E2_Q8_READ is set for the contract tests (fast, weights are borrowed)
 *   without one : the safetensors path quantizes, which is what the heap check needs
 */
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "e2.h"

static int g_fail;
#define CHECK(cond, ...)                        \
    do {                                        \
        if (!(cond)) {                          \
            printf("FAIL line %d: ", __LINE__); \
            printf(__VA_ARGS__);                \
            printf("\n");                       \
            g_fail++;                           \
        }                                       \
    } while (0)

#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
static size_t heap_bytes(void) { return mallinfo2().uordblks; }
#define HAVE_HEAP 1
#else
static size_t heap_bytes(void) { return 0; }
#define HAVE_HEAP 0
#endif

static void *die_oom(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "oom: %zu bytes\n", n); exit(2); }
    return p;
}

/* the live model must still run: one text embedding, checked to be a finite unit vector */
static void embed_ok(e2_model *m, const char *label) {
    const int32_t ids[4] = { 2, 1500, 2700, 1 };
    float *e = die_oom(sizeof(float) * 768);
    const int rc = e2_text_embed(m, ids, 4, e);
    CHECK(rc == 0, "%s: embed failed: %s", label, e2_last_error());
    double n2 = 0;
    int finite = 1;
    for (int i = 0; i < 768; i++) {
        n2 += (double)e[i] * e[i];
        if (!isfinite(e[i])) finite = 0;
    }
    CHECK(rc != 0 || (finite && fabs(sqrt(n2) - 1.0) < 1e-3), "%s: not a finite unit vector (|e|^2 %.6f)",
          label, n2);
    free(e);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_dir> [cache_file]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    const char *cache = argc > 2 ? argv[2] : NULL;
    /* This test chooses its own load mode, independent of benchmark/export environment. */
    if (unsetenv("E2_Q8_READ") || unsetenv("E2_Q8_WRITE") || unsetenv("E2_INT8")) return 2;

    /* ---- loader boundary: no dir, empty dir, a dir too long for the path buffer ---- */
    CHECK(e2_load(NULL) == NULL, "e2_load(NULL) was accepted");
    printf("  boundary: %s\n", e2_last_error());
    CHECK(e2_load("") == NULL, "e2_load(\"\") was accepted");
    char longdir[1100];
    memset(longdir, 'a', sizeof longdir - 1);
    longdir[sizeof longdir - 1] = 0;
    CHECK(e2_load(longdir) == NULL, "an over-long model_dir was accepted");
    printf("  boundary: %s\n", e2_last_error());
    /* none of the refusals above may have published a live model */

    /* ---- contract: one live model, refused second load, live model unaffected ---- */
    if (cache) setenv("E2_Q8_READ", cache, 1);
    e2_model *m = e2_load(dir);
    if (!m) {
        fprintf(stderr, "load failed: %s\n", e2_last_error());
        return 2; /* infrastructure, not a lifecycle failure */
    }
    embed_ok(m, "first load");

    e2_model *m2 = e2_load(dir);
    CHECK(m2 == NULL, "a second e2_load was accepted");
    CHECK(e2_last_error()[0] != 0, "the refused second load set no error");
    printf("  second load refused: %s\n", e2_last_error());
    /* refusal must not have touched the mapping or the pool: the live model still runs */
    embed_ok(m, "after the refused load");
    /* and a failed free of the refused handle must not release the live one either */
    e2_free(m2);
    embed_ok(m, "after e2_free(NULL)");
    e2_free(m);

    /* ---- a failed load leaves nothing behind: the next load still works ---- */
    const char *good = cache ? cache : NULL;
    if (good) {
        if (setenv("E2_Q8_READ", "/nonexistent/e2-test-cache.e8cq", 1)) return 2;
        CHECK(e2_load(dir) == NULL, "a load with a missing cache file was accepted");
        printf("  missing cache refused: %s\n", e2_last_error());
        if (setenv("E2_Q8_READ", good, 1)) return 2;
    } else {
        CHECK(e2_load("/nonexistent/e2-test-model-dir") == NULL, "a load from a missing dir was accepted");
        printf("  missing model refused: %s\n", e2_last_error());
    }
    m = e2_load(dir);
    CHECK(m != NULL, "load after a failed load: %s", e2_last_error());
    if (m) {
        embed_ok(m, "after a failed load");
        e2_free(m);
    }

    /* ---- heap: N load/free cycles must return to the same in-use byte count ---- */
    if (HAVE_HEAP) {
        enum { CYCLES = 3 };
        const size_t h0 = heap_bytes();
        size_t h[CYCLES + 1], peak[CYCLES + 1];
        h[0] = h0;
        for (int c = 1; c <= CYCLES; c++) {
            e2_model *mc = e2_load(dir);
            if (!mc) {
                fprintf(stderr, "load failed: %s\n", e2_last_error());
                return 2;
            }
            peak[c] = heap_bytes();
            e2_free(mc);
            h[c] = heap_bytes();
            printf("  cycle %d: %+.1f MB held by the model, %+.1f MB in use after free\n", c,
                   (double)(peak[c] - h0) / 1e6, (double)(h[c] - h0) / 1e6);
        }
        /* the first cycle may grow the arena once; after that the count has to be flat */
        const double grew = (double)(h[CYCLES] - h[1]) / 1e6;
        CHECK(h[CYCLES] <= h[1] + ((size_t)8 << 20), "heap grew %.1f MB over %d load/free cycles", grew,
              CYCLES - 1);
        printf("  heap: %+.1f MB after %d cycles (limit 8 MB)\n", grew, CYCLES);

        /* ---- the audio tower's int8 weights are built on the first audio call, not at load, so
         * they need their own load/prepare/free cycle ---- */
        const size_t a0 = heap_bytes();
        e2_model *ma = e2_load(dir);
        if (!ma) {
            fprintf(stderr, "load failed: %s\n", e2_last_error());
            return 2;
        }
        float *pcm = die_oom(sizeof(float) * 16000), *e = die_oom(sizeof(float) * 768);
        for (int i = 0; i < 16000; i++) pcm[i] = 0.1f * sinf(0.05f * (float)i);
        const int arc = e2_audio_embed(ma, pcm, 16000, e);
        if (arc == 0) {
            const size_t apeak = heap_bytes();
            e2_free(ma);
            const size_t aend = heap_bytes();
            printf("  audio: %+.1f MB held by the tower, %+.1f MB in use after free\n",
                   (double)(apeak - a0) / 1e6, (double)(aend - a0) / 1e6);
            CHECK(aend <= a0 + ((size_t)8 << 20), "heap grew %.1f MB after an audio embed",
                  (double)(aend - a0) / 1e6);
        } else {
            printf("  audio: check skipped (%s)\n", e2_last_error());
            e2_free(ma);
        }
        free(pcm);
        free(e);
    } else {
        printf("  heap: check skipped (needs glibc mallinfo2)\n");
    }

    printf("%s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
