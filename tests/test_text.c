/* test_text.c — run the engine on frozen fixtures and compare with the fp32 reference embedding.
 *
 * Usage: e2_test <model_dir> <fixture_dir>
 * Fixture format (written by tools/dump_fixtures.py):
 *   int32 n_ids, int32 dim, int32 ids[n_ids], float ref[dim]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "e2.h"
#include "test_metrics.h"

static float *read_fixture(const char *path, int32_t *n_ids_out, int32_t *dim_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    int32_t n, d;
    if (fread(&n, 4, 1, f) != 1 || fread(&d, 4, 1, f) != 1) { fclose(f); return NULL; }
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)n);
    float *ref = malloc(sizeof(float) * (size_t)d);
    if (fread(ids, 4, (size_t)n, f) != (size_t)n || fread(ref, 4, (size_t)d, f) != (size_t)d) {
        free(ids); free(ref); fclose(f); return NULL;
    }
    fclose(f);
    *n_ids_out = n;
    *dim_out = d;
    /* stash ids right after the reference so one pointer frees both */
    float *blk = malloc(sizeof(float) * (size_t)d + sizeof(int32_t) * (size_t)n);
    memcpy(blk, ref, sizeof(float) * (size_t)d);
    memcpy((char *)blk + sizeof(float) * (size_t)d, ids, sizeof(int32_t) * (size_t)n);
    free(ids); free(ref);
    return blk;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Fixed fmadd loop so every run reports the machine's current speed: this box throttles between
 * 1.5 and 4.5 GHz depending on power state, and absolute timings are meaningless without it. */
static double cpu_probe_gflops(void) {
    const int iters = 4000000;
    double t = now_ms();
    float acc[8] = { 0 };
    for (int i = 0; i < iters; i++)
        for (int j = 0; j < 8; j++) acc[j] = acc[j] * 0.9999f + 1.0f;
    double dt = (now_ms() - t) / 1e3;
    double flops = (double)iters * 8 * 2;
    volatile float sink = acc[0] + acc[7];
    (void)sink;
    return flops / dt / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <fixture_dir>\n", argv[0]); return 2; }
    double t0 = now_ms();
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "load failed: %s\n", e2_last_error()); return 1; }
    printf("loaded in %.2f s\n", (now_ms() - t0) / 1e3);
    e2_info(m);
    printf("  cpu probe   : %.1f scalar GFLOPS (throttling indicator)\n", cpu_probe_gflops());

    const char *names[] = { "text_query", "text_doc" };
    const char *labels[] = { "query (15 tok)", "document (196 tok)" };
    int ncases = 2, first = 0;
    if (argc > 3) {
        for (int i = 0; i < 2; i++) if (!strcmp(names[i], argv[3])) { first = i; ncases = i + 1; }
    }
    int failures = 0;
    for (int i = first; i < ncases; i++) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s.bin", argv[2], names[i]);
        int32_t n = 0, dim = 0;
        float *fx = read_fixture(path, &n, &dim);
        if (!fx) { fprintf(stderr, "cannot read fixture %s\n", path); failures++; continue; }
        const int32_t *ids = (const int32_t *)((const char *)fx + sizeof(float) * (size_t)dim);
        float *out = malloc(sizeof(float) * (size_t)dim);
        /* Check each timed output, outside the timed interval. */
        const char *re = getenv("E2_REPS");
        int reps = re ? atoi(re) : 3;
        if (reps < 1) reps = 1;
        double probe0 = cpu_probe_gflops();
        double dt = 1e18;
        double *samples = malloc(sizeof *samples * (size_t)reps);
        if (!samples) { free(out); free(fx); failures++; continue; }
        int rc = 0;
        for (int r = 0; r < reps; r++) {
            double t = now_ms();
            rc = e2_text_embed(m, ids, n, out);
            double d = now_ms() - t;
            samples[r] = d;
            if (rc || !e2_test_unit(names[i], out, (size_t)dim) ||
                !e2_test_match(names[i], out, fx, (size_t)dim, 0.999)) {
                fprintf(stderr, "%s: repetition %d failed: %s\n", names[i], r, e2_last_error());
                rc = -1;
                break;
            }
            if (d < dt) dt = d;
        }
        if (rc) { free(samples); free(out); free(fx); failures++; continue; }
        e2_test_timing(samples, reps);
        free(samples);
        double dot = 0, na = 0, nb = 0, maxdiff = 0;
        for (int d = 0; d < dim; d++) {
            dot += out[d] * fx[d];
            na += out[d] * out[d];
            nb += fx[d] * fx[d];
            double diff = fabs(out[d] - fx[d]);
            if (diff > maxdiff) maxdiff = diff;
        }
        double cos = dot / (sqrt(na) * sqrt(nb));
        printf("%-18s %4d tok  %8.2f ms (best of %d)  probe %.1f GF  cos=%.6f  max|Δ|=%.5f  %s\n", labels[i],
               n, dt, reps, probe0, cos, maxdiff, cos > 0.999 ? "PASS" : "FAIL");
        if (!(cos > 0.999)) failures++;
        free(out);
        free(fx);
    }
    e2_free(m);
    printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
