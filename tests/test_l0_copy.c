/* test_l0_copy.c — the Level Zero copy path (src/gpu/l0.c): e2_l0_write/e2_l0_read upload and download
 * through ordinary heap or stack host buffers, including several megabytes and unaligned sizes, and
 * the audio-then-text embedding failure happened on a copy issued after a kernel submission. Every
 * copy here is checked byte for byte, so a copy that reports success without landing is visible.
 *
 * Needs a GPU but no model: self-contained, skips with 77 when there is no device.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu/l0.h"

/* Skip only explicit absence messages, not module or initialization failures. */
static int no_gpu(const char *err) {
    return !strcmp(err, "no Level Zero driver") || !strcmp(err, "no Intel GPU device found");
}

static int failures;

static void check(int cond, const char *what, size_t bytes) {
    if (cond) return;
    printf("FAIL %s (%zu bytes)\n", what, bytes);
    failures++;
}

/* device memory <-> ordinary host memory, byte for byte */
static void round_trip(e2_l0 *g, const char *what, size_t bytes) {
    unsigned char *src = malloc(bytes), *dst = malloc(bytes);
    void *dev = e2_l0_alloc(g, bytes);
    if (!src || !dst || !dev) { printf("FAIL %s: allocation (%zu bytes)\n", what, bytes); failures++; }
    if (src && dst && dev) {
        for (size_t i = 0; i < bytes; i++) src[i] = (unsigned char)(i * 37u + (i >> 11));
        memset(dst, 0xA5, bytes);
        if (e2_l0_write(g, dev, bytes, src)) {
            printf("FAIL %s: write: %s\n", what, e2_l0_error(g));
            failures++;
        } else if (e2_l0_read(g, dev, bytes, dst)) {
            printf("FAIL %s: read: %s\n", what, e2_l0_error(g));
            failures++;
        } else {
            size_t i = 0;
            while (i < bytes && dst[i] == src[i]) i++;
            if (i != bytes)
                printf("FAIL %s: first difference at byte %zu (got %02x want %02x)\n", what, i, dst[i],
                       src[i]);
            check(i == bytes, what, bytes);
        }
    }
    free(src);
    free(dst);
    if (dev) e2_l0_free(g, dev);
}

int main(void) {
    char err[512] = { 0 };
    e2_l0 *g = e2_l0_create(err, sizeof err);
    if (!g) {
        if (no_gpu(err)) {
            printf("skip: %s\n", err);
            return 77;
        }
        fprintf(stderr, "FAIL create: %s\n", err[0] ? err : "(no message)");
        return 1;
    }
    printf("l0: %s\n", e2_l0_device_name(g));

    /* Small, unaligned and multi-megabyte transfers. */
    static const size_t sizes[] = { 4, 47104, 3 * 1024 * 1024, (size_t)33 << 20 };
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        char label[64];
        snprintf(label, sizeof label, "round trip %zu", sizes[i]);
        round_trip(g, label, sizes[i]);
    }

    /* a stack scalar: the vision tower uploads its pool-group count exactly like this */
    {
        void *dev = e2_l0_alloc(g, sizeof(int32_t));
        int32_t in = 0x5A5A1234, out = 0;
        if (!dev || e2_l0_write(g, dev, sizeof in, &in) || e2_l0_read(g, dev, sizeof out, &out))
            check(0, "stack scalar copy", sizeof in);
        else
            check(out == in, "stack scalar copy", sizeof in);
        if (dev) e2_l0_free(g, dev);
    }

    /* the sequence the audio embedding fails in: a recorded kernel list runs, then an upload, then a
     * readback. A copy must not be lost because a kernel submission preceded it. */
    {
        const int rows = 23, hid = 512;
        const size_t bytes = (size_t)rows * hid * 4;
        float *src = malloc(bytes), *back = malloc(bytes);
        void *a = e2_l0_alloc(g, bytes), *b = e2_l0_alloc(g, bytes);
        if (!src || !back || !a || !b) {
            printf("FAIL after-kernel: allocation\n");
            failures++;
        } else {
            for (int i = 0; i < rows * hid; i++) src[i] = 0.25f * (float)(i % 97);
            e2_graph *gr = e2_l0_graph_begin(g, 1);
            int rc = gr ? 0 : -1;
            if (!rc) {
                void *dstp = b, *srcp = a;
                rc = e2_l0_append(g, gr, "k_copy_rows", (uint32_t)((rows + 3) / 4),
                                  (uint32_t)(hid / 4 / 64), 1, 64, 4, 1, 4, &dstp, sizeof dstp,
                                  &srcp, sizeof srcp, &rows, sizeof rows, &hid, sizeof hid);
            }
            if (!rc) rc = e2_l0_graph_close(g, gr);
            if (!rc && e2_l0_write(g, a, bytes, src)) rc = -1;
            if (!rc) rc = e2_l0_graph_run(g, gr);
            if (!rc) rc = e2_l0_read(g, b, bytes, back);
            check(rc == 0, "after-kernel copy: run", bytes);
            if (!rc) check(memcmp(back, src, bytes) == 0, "after-kernel copy: contents", bytes);
            if (!rc) {
                for (int i = 0; i < rows * hid; i++) src[i] += 1.f;
                rc = e2_l0_write(g, a, bytes, src);
            }
            if (!rc) rc = e2_l0_graph_run(g, gr);
            if (!rc) rc = e2_l0_read(g, b, bytes, back);
            check(rc == 0, "upload after kernel: run", bytes);
            if (!rc) check(memcmp(back, src, bytes) == 0, "upload after kernel: contents", bytes);
            if (rc) printf("  after-kernel: %s\n", e2_l0_error(g));
            e2_l0_graph_free(g, gr);
        }
        free(src);
        free(back);
        if (a) e2_l0_free(g, a);
        if (b) e2_l0_free(g, b);
    }

    e2_l0_destroy(g);
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("OK\n");
    return 0;
}
