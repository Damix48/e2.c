/* Golden FP32 ONNX checks, including kernel boundaries and retrieval rankings.
 * Generate with tools/dump_text_checks.py; run model_dir fixture_dir [gpu]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "e2.h"
#ifdef E2_TEST_GPU
#include "e2gpu.h"
#endif
#include "test_metrics.h"

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) return 2;
    const int gpu = argc == 4 && !strcmp(argv[3], "gpu");
    if (argc == 4 && !gpu) return 2;
#ifndef E2_TEST_GPU
    if (gpu) return 2;
#endif
    char path[4096];
    if (snprintf(path, sizeof path, "%s/text_checks.bin", argv[2]) >= (int)sizeof path) return 2;
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }
    uint32_t header[4];
    if (fread(header, sizeof header, 1, f) != 1 || header[0] != 0x54583245 ||
        header[1] > 128 || !header[2] || header[2] > 32 || header[1] < 2 * header[2] || header[3] != 768) {
        fclose(f); return 2;
    }
    const int count = (int)header[1], pairs = (int)header[2];
    float *got = calloc((size_t)count * 768, sizeof *got);
    float *refs = calloc((size_t)count * 768, sizeof *refs);
    if (!got || !refs) { free(got); free(refs); fclose(f); return 2; }
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }
#ifdef E2_TEST_GPU
    e2_gpu *g = gpu ? e2_gpu_create(m) : NULL;
    if (gpu && !g) { fprintf(stderr, "%s\n", e2_gpu_last_error()); return 1; }
#endif
    int failed = 0, loaded = 0;
    double minimum = 1;
    for (int c = 0; c < count; c++) {
        char name[65] = { 0 };
        int32_t n = 0, ids[2048];
        float *out = got + (size_t)c * 768, *ref = refs + (size_t)c * 768;
        if (fread(name, 64, 1, f) != 1 || fread(&n, 4, 1, f) != 1 || n <= 0 || n > 2048 ||
            fread(ids, 4, (size_t)n, f) != (size_t)n || fread(ref, 4, 768, f) != 768) {
            fprintf(stderr, "invalid text reference record %d\n", c); failed = 1; break;
        }
        loaded++;
        int rc;
#ifdef E2_TEST_GPU
        if (gpu) rc = e2_gpu_text_embed(g, ids, n, out);
        else
#endif
        rc = e2_text_embed(m, ids, n, out);
        if (rc || !e2_test_unit(name, out, 768) || !e2_test_match(name, out, ref, 768, 0.999)) {
            fprintf(stderr, "reference failure: %s (%d tokens)\n", name, n); failed = 1;
        }
        double cosine = e2_test_cosine(out, ref, 768);
        if (cosine < minimum) minimum = cosine;
        printf("%-16s %4d tokens cos=%.9f\n", name, n, cosine);
    }
    fclose(f);
    if (loaded >= 2 * pairs) {
        int matches = 0;
        for (int q = 0; q < pairs; q++) {
            int actual = -1, expected = -1;
            double best = -2, reference_best = -2;
            for (int d = 0; d < pairs; d++) {
                double s = e2_test_cosine(got + (size_t)q * 768, got + (size_t)(pairs + d) * 768, 768);
                double r = e2_test_cosine(refs + (size_t)q * 768, refs + (size_t)(pairs + d) * 768, 768);
                if (s > best) { best = s; actual = d; }
                if (r > reference_best) { reference_best = r; expected = d; }
            }
            if (actual != expected || expected != q) {
                fprintf(stderr, "retrieval query %d: document %d, FP32 expected %d\n", q, actual, expected);
                failed = 1;
            } else matches++;
        }
        printf("retrieval top-1: %d/%d match FP32\n", matches, pairs);
    }
    printf("%s: %d FP32 cases, %d retrieval pairs, minimum cosine %.9f\n",
           failed ? "FAILED" : "OK", count, pairs, minimum);
#ifdef E2_TEST_GPU
    if (g) e2_gpu_destroy(g);
#endif
    e2_free(m);
    free(got); free(refs);
    return failed;
}
