/* hip_bench.c — AMD/HIP backend benchmark: per-call latency of the text and image shapes, next to
 * the CPU engine on the same input.
 *
 * Usage: e2_hip_bench <model_dir> <fixture_dir> [--runs N] [--image-runs N]
 *
 * Reads the same fixtures the tests use (text files hold [n_ids, dim, ids..., ref...]); times single
 * embeddings, the two batch shapes and the two image entries. Reports the minimum and median of the
 * timed runs; the first call of each shape is a warmup and is not counted.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "e2.h"
#include "e2hip.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void bench_image(const char *label, const float *pixels, const int32_t *pos, int np,
                        const int32_t *ids, int n_ids, e2_model *m, e2_hip *g, int iruns);

static float *read_fixture(const char *path, int32_t *n_ids_out, int32_t *dim_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t n, d;
    if (fread(&n, 4, 1, f) != 1 || fread(&d, 4, 1, f) != 1) { fclose(f); return NULL; }
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)n);
    float *ref = malloc(sizeof(float) * (size_t)d);
    if (!ids || !ref || fread(ids, 4, (size_t)n, f) != (size_t)n ||
        fread(ref, 4, (size_t)d, f) != (size_t)d) {
        free(ids); free(ref); fclose(f); return NULL;
    }
    fclose(f);
    *n_ids_out = n;
    *dim_out = d;
    float *blk = malloc(sizeof(float) * (size_t)d + sizeof(int32_t) * (size_t)n);
    memcpy(blk, ref, sizeof(float) * (size_t)d);
    memcpy((char *)blk + sizeof(float) * (size_t)d, ids, sizeof(int32_t) * (size_t)n);
    free(ids);
    free(ref);
    return blk;
}

/* image.bin: int32 n_patches, patch_dim, n_pos, dim; pixels[]; positions[]; ref[] */
static float *read_image(const char *path, int32_t *np, int32_t *pd, int32_t *npos) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t dim = 0;
    if (fread(np, 4, 1, f) != 1 || fread(pd, 4, 1, f) != 1 || fread(npos, 4, 1, f) != 1 ||
        fread(&dim, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    float *px = malloc(sizeof(float) * (size_t)*np * *pd + sizeof(int32_t) * (size_t)*np * *npos +
                       sizeof(float) * 768);
    if (!px || fread(px, 4, (size_t)*np * *pd, f) != (size_t)*np * *pd) {
        free(px);
        fclose(f);
        return NULL;
    }
    int32_t *pos = (int32_t *)(px + (size_t)*np * *pd);
    float *ref = (float *)(pos + (size_t)*np * *npos);
    if (fread(pos, 4, (size_t)*np * *npos, f) != (size_t)*np * *npos ||
        fread(ref, 4, (size_t)dim, f) != (size_t)dim) {
        free(px);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return px;
}

/* image_caption.bin: int32 n_ids, n_patches, patch_dim, dim; ids[]; pixels[]; positions[]; ref[] */
static float *read_caption(const char *path, int32_t *n_ids, int32_t *np, int32_t *pd,
                           int32_t **ids_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t dim = 0;
    if (fread(n_ids, 4, 1, f) != 1 || fread(np, 4, 1, f) != 1 || fread(pd, 4, 1, f) != 1 ||
        fread(&dim, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)*n_ids);
    float *px = malloc(sizeof(float) * (size_t)*np * *pd + sizeof(int32_t) * (size_t)*np * 2 +
                       sizeof(float) * 768);
    if (!ids || !px || fread(ids, 4, (size_t)*n_ids, f) != (size_t)*n_ids ||
        fread(px, 4, (size_t)*np * *pd, f) != (size_t)*np * *pd) {
        free(ids);
        free(px);
        fclose(f);
        return NULL;
    }
    int32_t *pos = (int32_t *)(px + (size_t)*np * *pd);
    float *ref = (float *)(pos + (size_t)*np * 2);
    if (fread(pos, 4, (size_t)*np * 2, f) != (size_t)*np * 2 ||
        fread(ref, 4, (size_t)dim, f) != (size_t)dim) {
        free(ids);
        free(px);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *ids_out = ids;
    return px;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <fixture_dir> [--runs N]\n", argv[0]); return 2; }
    int runs = 20, iruns = 5;
    for (int i = 3; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--runs")) runs = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--image-runs")) iruns = atoi(argv[i + 1]);
    }
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "model load failed: %s\n", e2_last_error()); return 1; }
    double t0 = now_ms();
    e2_hip *g = e2_hip_create(m);
    if (!g) { fprintf(stderr, "hip init failed: %s\n", e2_hip_last_error()); return 1; }
    printf("hip: %s (upload %.2f s)\n", e2_hip_device(g), (now_ms() - t0) / 1e3);

    struct { const char *name; const char *label; } fx[] = {
        { "text_query", "query (15 tok)" },
        { "text_doc", "document (196 tok)" },
    };
    for (size_t i = 0; i < sizeof fx / sizeof fx[0]; i++) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s.bin", argv[2], fx[i].name);
        int32_t n = 0, dim = 0;
        float *blk = read_fixture(path, &n, &dim);
        if (!blk) { fprintf(stderr, "cannot read %s\n", path); continue; }
        const int32_t *ids = (const int32_t *)((const char *)blk + sizeof(float) * (size_t)dim);
        float *out = malloc(sizeof(float) * (size_t)dim);
        double *ts = malloc(sizeof(double) * (size_t)runs);
        for (int r = 0; r < 3 + runs; r++) {
            double t = now_ms();
            if (e2_hip_text_embed(g, ids, n, out)) {
                fprintf(stderr, "embed: %s\n", e2_hip_last_error());
                break;
            }
            if (r >= 3) ts[r - 3] = now_ms() - t;
        }
        qsort(ts, (size_t)runs, sizeof(double), cmp_double);
        long calls = 0, kernels = 0;
        double last = 0;
        e2_hip_stats(g, &calls, &kernels, &last);
        printf("%-18s S=%-4d  min %7.3f ms  median %7.3f ms  kernels %ld\n", fx[i].label, n,
               ts[0], ts[runs / 2], kernels);
        free(ts);
        free(out);
        free(blk);
    }

    /* batch shapes */
    {
        int32_t qids[512], dids[512];
        int nq = 0, nd = 0, dim = 0;
        char path[512];
        snprintf(path, sizeof path, "%s/text_query.bin", argv[2]);
        float *blk = read_fixture(path, &nq, &dim);
        if (blk) { memcpy(qids, (const char *)blk + sizeof(float) * (size_t)dim, sizeof(int32_t) * (size_t)nq); free(blk); }
        snprintf(path, sizeof path, "%s/text_doc.bin", argv[2]);
        blk = read_fixture(path, &nd, &dim);
        if (blk) { memcpy(dids, (const char *)blk + sizeof(float) * (size_t)dim, sizeof(int32_t) * (size_t)nd); free(blk); }
        const struct { int batch, seq; } cases[] = { { 8, 17 }, { 32, 196 } };
        for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
            const int batch = cases[ci].batch, seq = cases[ci].seq;
            const int32_t *src = seq <= 32 ? qids : dids;
            const int ns = seq <= 32 ? nq : nd;
            int32_t *ids = malloc(sizeof(int32_t) * (size_t)batch * seq);
            uint8_t *valid = malloc((size_t)batch * seq);
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < seq; i++) {
                    ids[(size_t)b * seq + i] = src[i < ns ? i : 0];
                    valid[(size_t)b * seq + i] = i < ns;
                }
            float *out = malloc(sizeof(float) * 768 * (size_t)batch);
            double *ts = malloc(sizeof(double) * (size_t)runs);
            for (int r = 0; r < 3 + runs; r++) {
                double t = now_ms();
                if (e2_hip_text_embed_batch(g, ids, valid, batch, seq, out)) {
                    fprintf(stderr, "batch: %s\n", e2_hip_last_error());
                    break;
                }
                if (r >= 3) ts[r - 3] = now_ms() - t;
            }
            qsort(ts, (size_t)runs, sizeof(double), cmp_double);
            long calls = 0, kernels = 0;
            double last = 0;
            e2_hip_stats(g, &calls, &kernels, &last);
            printf("batch %d x %-6d        min %7.2f ms  median %7.2f ms  %.1f seq/s  kernels %ld\n",
                   batch, seq, ts[0], ts[runs / 2], batch / ts[runs / 2] * 1e3, kernels);
            free(ts);
            free(out);
            free(ids);
            free(valid);
        }
    }
    printf("\nimage shapes (CPU engine vs HIP, min/median of %d runs after 1 warmup; ratio = cpu/hip):\n",
           iruns);
    {
        char path[512];
        snprintf(path, sizeof path, "%s/image.bin", argv[2]);
        int32_t np = 0, pd = 0, npos = 0;
        float *fx = read_image(path, &np, &pd, &npos);
        if (fx) {
            bench_image("image", fx, (const int32_t *)(fx + (size_t)np * pd), np, NULL, 0, m, g,
                        iruns);
            free(fx);
        } else {
            fprintf(stderr, "cannot read %s\n", path);
        }
        snprintf(path, sizeof path, "%s/image_caption.bin", argv[2]);
        int32_t n_ids = 0, np2 = 0, pd2 = 0;
        int32_t *ids = NULL;
        float *fx2 = read_caption(path, &n_ids, &np2, &pd2, &ids);
        if (fx2) {
            bench_image("image+caption", fx2, (const int32_t *)(fx2 + (size_t)np2 * pd2), np2, ids,
                        n_ids, m, g, iruns);
            free(fx2);
            free(ids);
        } else {
            fprintf(stderr, "cannot read %s\n", path);
        }
    }
    e2_hip_destroy(g);
    e2_free(m);
    return 0;
}

/* ---- image: the same two entry points on both backends, median of `iruns` after one warmup. */
static void bench_image(const char *label, const float *pixels, const int32_t *pos, int np,
                        const int32_t *ids, int n_ids, e2_model *m, e2_hip *g, int iruns) {
    float *out = malloc(sizeof(float) * 768);
    double *tc = malloc(sizeof(double) * (size_t)iruns);
    double *th = malloc(sizeof(double) * (size_t)iruns);
    for (int r = 0; r <= iruns; r++) {
        const double t = now_ms();
        if (ids) e2_text_image_embed(m, ids, n_ids, pixels, pos, np, out);
        else e2_image_embed(m, pixels, pos, np, out);
        if (r) tc[r - 1] = now_ms() - t;
    }
    for (int r = 0; r <= iruns; r++) {
        const double t = now_ms();
        if (ids) e2_hip_text_image_embed(g, ids, n_ids, pixels, pos, np, out);
        else e2_hip_image_embed(g, pixels, pos, np, out);
        if (r) th[r - 1] = now_ms() - t;
    }
    qsort(tc, (size_t)iruns, sizeof(double), cmp_double);
    qsort(th, (size_t)iruns, sizeof(double), cmp_double);
    long k = 0;
    double last = 0;
    e2_hip_vision_stats(g, &k, &last);
    printf("%-18s        cpu min %8.1f ms  median %8.1f ms   hip min %8.1f ms  median %8.1f ms  "
           "%.2fx  kernels %ld\n",
           label, tc[0], tc[iruns / 2], th[0], th[iruns / 2], tc[iruns / 2] / th[iruns / 2], k);
    free(out);
    free(tc);
    free(th);
}
