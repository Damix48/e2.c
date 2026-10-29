/* Checked text latency/throughput: model_dir fixture_dir cpu|gpu|backends seq_len batch reps.
 * Compile this same source against the old and new libraries for alternating A/B runs.
 *
 * "backends" sweeps every CPU backend this host can execute, one JSON line each, and reloads the model
 * between them because m->use_int8 is decided at load from the quantizer the backend installs. The
 * first backend's embedding is the reference the others are checked against. */
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "e2.h"
#include "kernels.h" /* e2_cpu_backend_count / e2_cpu_backend_name, the E2_CPU_BACKEND names */
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

/* One embedding pass over the batch on whichever backend is selected. */
static int embed_batch(e2_model *m, int gpu, void *gh, const int32_t *ids, int seq, int batch, float *out) {
#ifdef E2_BENCH_GPU
    if (gpu) return e2_gpu_text_embed_batch((e2_gpu *)gh, ids, NULL, batch, seq, out);
#endif
    (void)gh;
    (void)gpu;
#ifdef E2_BENCH_CPU_BATCH
    if (batch == 1) return e2_text_embed(m, ids, seq, out);
    return e2_text_embed_batch(m, ids, NULL, batch, seq, out);
#else
    int rc = 0;
    for (int b = 0; b < batch && !rc; b++)
        rc = e2_text_embed(m, ids + (size_t)b * seq, seq, out + (size_t)b * 768);
    return rc;
#endif
}

/* Three warm-up rounds then `reps` timed ones; every batch row must stay unit-norm and within tolerance
 * of `ref`, so a backend that is fast and wrong fails instead of setting a record. */
static int timed_runs(e2_model *m, int gpu, void *gh, const int32_t *ids, int seq, int batch, int reps,
                      float *out, const float *ref, double *samples, double *minimum_cosine) {
    for (int r = -3; r < reps; r++) {
        double t = now_ms();
        int rc = embed_batch(m, gpu, gh, ids, seq, batch, out);
        double elapsed = now_ms() - t;
        if (rc) { fprintf(stderr, "embedding failed\n"); return 1; }
        for (int b = 0; b < batch; b++) {
            const float *v = out + (size_t)b * 768;
            if (!e2_test_unit("benchmark", v, 768) || !e2_test_match("batch vs single", v, ref, 768, 0.999))
                return 1;
            double c = e2_test_cosine(v, ref, 768);
            if (c < *minimum_cosine) *minimum_cosine = c;
        }
        if (r >= 0) samples[r] = elapsed;
    }
    return 0;
}

static void report(const char *backend, int seq, int batch, int reps, double *samples, double minimum_cosine) {
    qsort(samples, (size_t)reps, sizeof *samples, e2_test_double_cmp);
    double median = (samples[(reps - 1) / 2] + samples[reps / 2]) * 0.5;
    printf("{\"backend\":\"%s\",\"seq_len\":%d,\"batch\":%d,\"reps\":%d,"
           "\"best_ms\":%.6f,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"minimum_cosine\":%.9f}\n",
           backend, seq, batch, reps, samples[0], median, samples[(int)ceil(reps * 0.95) - 1],
           minimum_cosine);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s model_dir fixture_dir cpu|gpu|backends seq_len batch reps\n", argv[0]);
        return 2;
    }
    const int sweep = strcmp(argv[3], "backends") == 0;
    const int gpu = !sweep && strcmp(argv[3], "gpu") == 0;
    const int seq = positive(argv[4], 2048), batch = positive(argv[5], 65536);
    const int reps = positive(argv[6], 10000);
    if ((!sweep && !gpu && strcmp(argv[3], "cpu")) || !seq || !batch || !reps) return 2;
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

    float ref[768];
    if (sweep) {
        /* Each load re-reads E2_CPU_BACKEND, so save the caller's choice and put it back at the end. */
        char previous[32] = { 0 };
        const char *p = getenv("E2_CPU_BACKEND");
        if (p && strlen(p) < sizeof previous) strcpy(previous, p);
        int have_ref = 0, ran = 0;
        for (int i = 0; i < e2_cpu_backend_count(); i++) {
            const char *name = e2_cpu_backend_name(i);
            if (!name) continue; /* this host cannot execute it */
            if (setenv("E2_CPU_BACKEND", name, 1)) return 2;
            e2_model *m = e2_load(argv[1]);
            if (!m) {
                /* The scalar backend is bf16, and a packed int8 cache cannot serve it: skip that row
                 * instead of failing the whole sweep, and say why. */
                fprintf(stderr, "skipped %s: %s\n", name, e2_last_error());
                continue;
            }
            if (!have_ref) {
                ok = !e2_text_embed(m, ids, seq, ref) && e2_test_unit("single reference", ref, 768);
                if (!ok) { e2_free(m); return 1; }
                have_ref = 1;
            }
            double minimum_cosine = 1.0;
            if (timed_runs(m, 0, NULL, ids, seq, batch, reps, out, ref, samples, &minimum_cosine)) {
                e2_free(m);
                return 1;
            }
            report(name, seq, batch, reps, samples, minimum_cosine);
            e2_free(m);
            ran++;
        }
        if (!ran) { fprintf(stderr, "no backend could load %s\n", argv[1]); return 1; }
        if (previous[0]) setenv("E2_CPU_BACKEND", previous, 1);
        else unsetenv("E2_CPU_BACKEND");
        free(ids); free(out); free(samples);
        return 0;
    }

    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }
#ifdef E2_BENCH_GPU
    e2_gpu *g = gpu ? e2_gpu_create(m) : NULL;
    if (gpu && !g) { fprintf(stderr, "%s\n", e2_gpu_last_error()); e2_free(m); return 1; }
#endif
    /* A consistently wrong GPU result must not become its own correctness oracle. */
    int rc = e2_text_embed(m, ids, seq, ref);
    if (rc || !e2_test_unit("single reference", ref, 768)) return 1;
    double minimum_cosine = 1.0;
#ifdef E2_BENCH_GPU
    if (timed_runs(m, gpu, g, ids, seq, batch, reps, out, ref, samples, &minimum_cosine)) return 1;
#else
    if (timed_runs(m, 0, NULL, ids, seq, batch, reps, out, ref, samples, &minimum_cosine)) return 1;
#endif
    report(argv[3], seq, batch, reps, samples, minimum_cosine);
#ifdef E2_BENCH_GPU
    if (g) e2_gpu_destroy(g);
#endif
    e2_free(m);
    free(ids); free(out); free(samples);
    return 0;
}
