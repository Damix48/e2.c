/* test_gpu_text.c — the GPU text tower against the same frozen fixtures as the CPU test.
 *
 * Usage: e2_test_gpu_text <model_dir> <fixture_dir>
 *
 * Two comparisons per fixture: the GPU engine against the fp32 reference (the number that matters for
 * the port) and against the CPU engine on the same input (the two should agree to float noise, since
 * they run the same int8 recipe). Also reports the wall time per embedding and the kernel count of
 * the recorded graph.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "e2.h"
#include "test_metrics.h"
#include "e2gpu.h"

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
    float *blk = malloc(sizeof(float) * (size_t)d + sizeof(int32_t) * (size_t)n);
    memcpy(blk, ref, sizeof(float) * (size_t)d);
    memcpy((char *)blk + sizeof(float) * (size_t)d, ids, sizeof(int32_t) * (size_t)n);
    free(ids);
    free(ref);
    return blk;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static double cosine(const float *a, const float *b, int n) {
    const double c = e2_test_cosine(a, b, (size_t)n);
    return isfinite(c) ? c : -1.0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <fixture_dir>\n", argv[0]); return 2; }
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "model load failed: %s\n", e2_last_error()); return 1; }
    double t0 = now_ms();
    e2_gpu *g = e2_gpu_create(m);
    if (!g) { fprintf(stderr, "gpu init failed: %s\n", e2_gpu_last_error()); return 1; }
    printf("gpu: %s (upload %.2f s)\n", e2_gpu_device(g), (now_ms() - t0) / 1e3);

    const char *names[] = { "text_query", "text_doc" };
    const char *labels[] = { "query (15 tok)", "document (196 tok)" };
    int failures = 0;
    for (int i = 0; i < 2; i++) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s.bin", argv[2], names[i]);
        int32_t n = 0, dim = 0;
        float *fx = read_fixture(path, &n, &dim);
        if (!fx) { fprintf(stderr, "cannot read %s\n", path); failures++; continue; }
        const int32_t *ids = (const int32_t *)((const char *)fx + sizeof(float) * (size_t)dim);
        const float *ref = fx;
        float *gcpu = malloc(sizeof(float) * (size_t)dim);
        float *ggpu = malloc(sizeof(float) * (size_t)dim);
        double tcpu = now_ms();
        if (e2_text_embed(m, ids, n, gcpu) || !e2_test_unit("CPU text reference", gcpu, (size_t)dim) ||
            !e2_test_match("CPU text reference", gcpu, ref, (size_t)dim, 0.999)) {
            fprintf(stderr, "CPU reference failed: %s\n", e2_last_error());
            free(gcpu); free(ggpu); free(fx);
            e2_gpu_destroy(g); e2_free(m);
            return 1;
        }
        tcpu = now_ms() - tcpu;
        double best = 1e9, samples[5];
        int checked = 0;
        for (int r = 0; r < 5; r++) {
            double ts = now_ms();
            if (e2_gpu_text_embed(g, ids, n, ggpu)) {
                fprintf(stderr, "gpu: %s\n", e2_gpu_last_error());
                failures++;
                break;
            }
            double dt = now_ms() - ts;
            if (!e2_test_unit(names[i], ggpu, (size_t)dim) ||
                !e2_test_match(names[i], ggpu, ref, (size_t)dim, 0.999) ||
                !e2_test_match("GPU vs CPU", ggpu, gcpu, (size_t)dim, 0.999)) {
                failures++;
                break;
            }
            samples[checked++] = dt;
            if (getenv("E2_GPU_VERBOSE")) printf("     run %d: %.3f ms\n", r, dt);
            if (dt < best) best = dt;
        }
        e2_test_timing(samples, checked);
        long calls = 0, kernels = 0;
        double last = 0;
        e2_gpu_stats(g, &calls, &kernels, &last);
        printf("%-16s S=%-4d gpu %7.3f ms  cos(gpu,ref) %.6f  cos(gpu,cpu) %.6f  cos(cpu,ref) %.6f  "
               "kernels %ld  cpu %.2f ms\n",
               labels[i], n, best, cosine(ggpu, ref, dim), cosine(ggpu, gcpu, dim),
               cosine(gcpu, ref, dim), kernels, tcpu);
        /* the bar the engine holds itself to: 0.999 against the fp32 reference (the CPU int8 path
         * scores 0.9994-0.9995 on these fixtures). GPU and CPU differ in summation order only. */
        if (cosine(ggpu, ref, dim) < 0.999 || cosine(ggpu, gcpu, dim) < 0.999) failures++;
        free(gcpu);
        free(ggpu);
        free(fx);
    }
    /* a padded batch must reproduce the single-sequence result, and `valid` must keep the padding out
     * of the mean pool */
    {
        const int batch = 8, seq = 120;
        int32_t *ids = malloc(sizeof(int32_t) * (size_t)batch * seq);
        uint8_t *valid = calloc((size_t)batch * seq, 1);
        int32_t qids[256];
        int nq = 0;
        {
            char path[512];
            snprintf(path, sizeof path, "%s/text_query.bin", argv[2]);
            int32_t nn = 0, dd = 0;
            float *fx = read_fixture(path, &nn, &dd);
            if (fx) {
                memcpy(qids, (const char *)fx + sizeof(float) * (size_t)dd, sizeof(int32_t) * (size_t)nn);
                nq = nn;
                free(fx);
            }
        }
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < seq; i++) {
                ids[(size_t)b * seq + i] = qids[i % (nq > 0 ? nq : 1)]; /* filler */
                valid[(size_t)b * seq + i] = 1;
            }
        for (int i = nq; i < seq; i++) valid[i] = 0; /* sequence 0 is the query plus padding */
        float *bout = malloc(sizeof(float) * 768 * batch);
        if (e2_gpu_text_embed_batch(g, ids, valid, batch, seq, bout)) {
            printf("batch: %s\n", e2_gpu_last_error());
            failures++;
        } else {
            float *one = malloc(sizeof(float) * 768);
            if (!e2_gpu_text_embed(g, qids, nq, one)) {
                const double c = cosine(bout, one, 768);
                printf("batch of %d x %d (query padded): cos vs single %.6f  (tiled GEMM: %d rows)\n",
                       batch, seq, c, batch * seq);
                /* the batch runs the tiled GEMM path while the single call uses the register kernel:
                 * the same quantized inputs, so the difference is float multiply order only */
                if (c < 0.999) failures++;
            }
            free(one);
        }
        free(bout);
        free(ids);
        free(valid);
    }
    /* the same check on a batch that fits the register kernel (rows <= 16): this isolates "the batch
     * path" from "which GEMM kernel the row count selects" */
    {
        char path[512];
        int32_t qids[512];
        int nq = 0, dim = 0;
        snprintf(path, sizeof path, "%s/text_query.bin", argv[2]);
        float *fx = read_fixture(path, &nq, &dim);
        if (fx) {
            memcpy(qids, (const char *)fx + sizeof(float) * (size_t)dim, sizeof(int32_t) * (size_t)nq);
            free(fx);
            uint8_t valid[16] = { 0 };
            for (int i = 0; i < nq; i++) valid[i] = 1;
            float bout[768], one[768];
            if (!e2_gpu_text_embed_batch(g, qids, valid, 1, 15, bout) &&
                !e2_gpu_text_embed(g, qids, nq, one)) {
                const double c = cosine(bout, one, 768);
                printf("batch of 1 x 15 (query, 15 rows, register kernel): cos vs single %.6f\n", c);
                if (c < 0.9999) failures++;
            }
        }
    }
    /* Batch timing on the two shapes OpenVINO's GPU table reports for the text model, so the two
     * numbers are comparable: 8 x 17 and 32 x 196. The ids are the query fixture's repeated/padded,
     * which is enough for a timing (the embeddings are not checked here, only the batch consistency
     * case above is). 32 x 196 is 6272 rows, so this also exercises the sequence-chunked path. */
    {
        char path[512];
        int32_t qids[512], dids[512];
        int nq = 0, nd = 0;
        {
            int32_t n = 0, dim = 0;
            snprintf(path, sizeof path, "%s/text_query.bin", argv[2]);
            float *fx = read_fixture(path, &n, &dim);
            if (fx) { memcpy(qids, (const char *)fx + sizeof(float) * (size_t)dim, sizeof(int32_t) * (size_t)n); nq = n; free(fx); }
            snprintf(path, sizeof path, "%s/text_doc.bin", argv[2]);
            fx = read_fixture(path, &n, &dim);
            if (fx) { memcpy(dids, (const char *)fx + sizeof(float) * (size_t)dim, sizeof(int32_t) * (size_t)n); nd = n; free(fx); }
        }
        const struct { int batch, seq; const char *label; } cases[] = {
            { 8, 17, "8 x 17" },
            { 32, 196, "32 x 196" },
        };
        for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
            const int batch = cases[ci].batch, seq = cases[ci].seq;
            int32_t *ids = malloc(sizeof(int32_t) * (size_t)batch * seq);
            uint8_t *valid = malloc((size_t)batch * seq);
            const int32_t *src = seq <= 32 ? qids : dids;
            const int ns = seq <= 32 ? nq : nd;
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < seq; i++) {
                    ids[(size_t)b * seq + i] = src[i < ns ? i : 0];
                    valid[(size_t)b * seq + i] = i < ns;
                }
            float *out = malloc(sizeof(float) * 768 * (size_t)batch);
            double best = 1e9, samples[3];
            float expected[768];
            int bad = ns <= 0 || e2_gpu_text_embed(g, src, ns, expected);
            int checked = 0;
            for (int r = 0; !bad && r < 3; r++) {
                double t0 = now_ms();
                if (e2_gpu_text_embed_batch(g, ids, valid, batch, seq, out)) { bad = 1; break; }
                const double dt = now_ms() - t0;
                for (int b = 0; b < batch; b++)
                    if (!e2_test_unit("batch", out + (size_t)b * 768, 768) ||
                        !e2_test_match("batch vs single", out + (size_t)b * 768, expected, 768, 0.999))
                        bad = 1;
                if (bad) break;
                samples[checked++] = dt;
                if (dt < best) best = dt;
            }
            if (bad) failures++;
            e2_test_timing(samples, checked);
            long calls = 0, kernels = 0;
            double last = 0;
            e2_gpu_stats(g, &calls, &kernels, &last);
            if (bad)
                printf("batch %-9s FAILED: %s\n", cases[ci].label, e2_gpu_last_error());
            else
                printf("batch %-9s (%d sequences)  gpu %7.2f ms  %.1f seq/s  kernels %ld\n",
                       cases[ci].label, batch, best, batch / best * 1e3, kernels);
            free(ids);
            free(valid);
            free(out);
        }
    }
    e2_gpu_destroy(g);
    e2_free(m);
    printf("%s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
