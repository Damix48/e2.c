/* Checked text latency/throughput: model_dir fixture_dir cpu|gpu seq_len batch reps.
 * Compile this same source against the old and new libraries for alternating A/B runs. */
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "e2.h"
#ifdef E2_BENCH_GPU
#include "e2gpu.h"
#endif
#include "../tests/test_metrics.h"

static int positive(const char *s, int maximum) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    return !errno && *s && !*end && n > 0 && n <= maximum ? (int)n : 0;
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s model_dir fixture_dir cpu|gpu seq_len batch reps\n", argv[0]);
        return 2;
    }
    const int gpu = !strcmp(argv[3], "gpu");
    const int seq = positive(argv[4], 2048), batch = positive(argv[5], 65536);
    const int reps = positive(argv[6], 10000);
    if ((!gpu && strcmp(argv[3], "cpu")) || !seq || !batch || !reps) return 2;
#ifndef E2_BENCH_GPU
    if (gpu) { fprintf(stderr, "GPU backend was not built\n"); return 2; }
#endif
    char path[4096];
    if (snprintf(path, sizeof path, "%s/text_%s.bin", argv[2], seq <= 32 ? "query" : "doc") >= (int)sizeof path)
        return 2;
    FILE *f = fopen(path, "rb");
    int32_t n = 0, dim = 0, source[2048];
    if (!f) { perror(path); return 2; }
    int ok = fread(&n, 4, 1, f) == 1 && fread(&dim, 4, 1, f) == 1 && n > 0 && n <= 2048 && dim == 768;
    if (ok) ok = fread(source, 4, (size_t)n, f) == (size_t)n;
    fclose(f);
    if (!ok) return 2;
    int32_t *ids = malloc(sizeof *ids * (size_t)batch * seq);
    float *out = malloc(sizeof *out * (size_t)batch * 768);
    double *samples = malloc(sizeof *samples * (size_t)reps);
    if (!ids || !out || !samples) { free(ids); free(out); free(samples); return 2; }
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < seq; i++) ids[(size_t)b * seq + i] = source[i % n];
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }
#ifdef E2_BENCH_GPU
    e2_gpu *g = gpu ? e2_gpu_create(m) : NULL;
    if (gpu && !g) { fprintf(stderr, "%s\n", e2_gpu_last_error()); e2_free(m); return 1; }
#endif
    float ref[768];
    /* A consistently wrong GPU result must not become its own correctness oracle. */
    int rc = e2_text_embed(m, ids, seq, ref);
    if (rc || !e2_test_unit("single reference", ref, 768)) return 1;
    double minimum_cosine = 1.0;
    for (int r = -3; r < reps; r++) {
        double t = now_ms();
#ifdef E2_BENCH_GPU
        if (gpu) rc = e2_gpu_text_embed_batch(g, ids, NULL, batch, seq, out);
        else
#endif
#ifdef E2_BENCH_CPU_BATCH
        if (batch == 1) rc = e2_text_embed(m, ids, seq, out);
        else rc = e2_text_embed_batch(m, ids, NULL, batch, seq, out);
#else
        {
            rc = 0;
            for (int b = 0; b < batch && !rc; b++)
                rc = e2_text_embed(m, ids + (size_t)b * seq, seq, out + (size_t)b * 768);
        }
#endif
        double elapsed = now_ms() - t;
        if (rc) { fprintf(stderr, "embedding failed\n"); return 1; }
        for (int b = 0; b < batch; b++) {
            const float *v = out + (size_t)b * 768;
            if (!e2_test_unit("benchmark", v, 768) || !e2_test_match("batch vs single", v, ref, 768, 0.999))
                return 1;
            double c = e2_test_cosine(v, ref, 768);
            if (c < minimum_cosine) minimum_cosine = c;
        }
        if (r >= 0) samples[r] = elapsed;
    }
    qsort(samples, (size_t)reps, sizeof *samples, e2_test_double_cmp);
    double median = (samples[(reps - 1) / 2] + samples[reps / 2]) * 0.5;
    printf("{\"backend\":\"%s\",\"seq_len\":%d,\"batch\":%d,\"reps\":%d,"
           "\"best_ms\":%.6f,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"minimum_cosine\":%.9f}\n",
           argv[3], seq, batch, reps, samples[0], median, samples[(int)ceil(reps * 0.95) - 1], minimum_cosine);
#ifdef E2_BENCH_GPU
    if (g) e2_gpu_destroy(g);
#endif
    e2_free(m);
    free(ids); free(out); free(samples);
    return 0;
}
