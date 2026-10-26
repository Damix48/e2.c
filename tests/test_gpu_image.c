/* test_gpu_image.c — the GPU vision tower against the same frozen fixtures the CPU image test uses.
 *
 * Usage: e2_test_gpu_image <model_dir> <fixture_dir>
 *
 * Two comparisons per case: the GPU engine against the fp32 reference (the number that matters for
 * the port) and against the CPU engine on the same input. The CPU half of the vision tower is int8
 * too, but its attention is fp32 while the GPU attention is int8, so cos(gpu,cpu) is the number that
 * says whether the attention quantization moved the embedding; the reference column says whether it
 * matters.
 *
 * E2_GPU_VLAYERS=n records a diagnostic graph that stops after n layers (0 = after the patch
 * embedder) and, with E2_GPU_DUMP=<dir>, writes the activation to <dir>/gpu_vx.f32 — that is what
 * bisects the port against `E2_DUMP=<dir> e2_test_image`'s v_patch_embed.f32 / v_layer0.f32 / ...
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "e2.h"
#include "test_metrics.h"
#include "e2gpu.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static double cosine(const float *a, const float *b, int n) {
    const double c = e2_test_cosine(a, b, (size_t)n);
    return isfinite(c) ? c : -1.0;
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
    if (fread(px, 4, (size_t)*np * *pd, f) != (size_t)*np * *pd) {
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
    if (!ids || !px) { free(ids); free(px); fclose(f); return NULL; }
    if (fread(ids, 4, (size_t)*n_ids, f) != (size_t)*n_ids ||
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
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <fixture_dir>\n", argv[0]); return 2; }
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "model load failed: %s\n", e2_last_error()); return 1; }
    double t0 = now_ms();
    e2_gpu *g = e2_gpu_create(m);
    if (!g) { fprintf(stderr, "gpu init failed: %s\n", e2_gpu_last_error()); return 1; }
    printf("gpu: %s (upload %.2f s)\n", e2_gpu_device(g), (now_ms() - t0) / 1e3);
    const char *vlim = getenv("E2_GPU_VLAYERS");
    const int diagnostic = vlim && atoi(vlim) >= 0;
    int failures = 0;

    { /* image only */
        char path[512];
        snprintf(path, sizeof path, "%s/image.bin", argv[2]);
        int32_t n_patch = 0, p_dim = 0, n_pos = 0;
        float *fx = read_image(path, &n_patch, &p_dim, &n_pos);
        if (!fx) { fprintf(stderr, "cannot read %s\n", path); return 2; }
        const float *pixels = fx;
        const int32_t *pos = (const int32_t *)(fx + (size_t)n_patch * p_dim);
        const float *ref = (const float *)(pos + (size_t)n_patch * n_pos);
        float *gpu = malloc(sizeof(float) * 768);
        float *cpu = malloc(sizeof(float) * 768);
        double tcpu = now_ms();
        if (e2_image_embed(m, pixels, pos, n_patch, cpu) ||
            !e2_test_unit("CPU image reference", cpu, 768) ||
            !e2_test_match("CPU image reference", cpu, ref, 768, 0.999)) {
            fprintf(stderr, "CPU reference failed: %s\n", e2_last_error());
            free(gpu); free(cpu); free(fx);
            e2_gpu_destroy(g); e2_free(m);
            return 1;
        }
        tcpu = now_ms() - tcpu;
        double best = 1e18;
        const char *re = getenv("E2_REPS");
        int reps = re ? atoi(re) : 5;
        if (reps < 1) reps = 1;
        double *samples = malloc(sizeof *samples * (size_t)reps);
        if (!samples) return 2;
        int checked = 0;
        for (int r = 0; r < reps; r++) {
            const double t = now_ms();
            if (diagnostic) for (int d = 0; d < 768; d++) gpu[d] = 12345.f;
            if (e2_gpu_image_embed(g, pixels, pos, n_patch, gpu)) {
                /* A truncated graph can dump stages, but must not claim it produced an embedding. */
                if (diagnostic && !strcmp(e2_gpu_last_error(), "diagnostic vision graph has no embedding")) {
                    for (int d = 0; d < 768; d++) assert(gpu[d] == 12345.f);
                    break;
                }
                fprintf(stderr, "gpu: %s\n", e2_gpu_last_error());
                failures++;
                break;
            }
            if (diagnostic) {
                fprintf(stderr, "diagnostic graph incorrectly returned an embedding success\n");
                failures++;
                break;
            }
            const double d = now_ms() - t;
            if (!diagnostic && (!e2_test_unit("image", gpu, 768) ||
                !e2_test_match("image", gpu, ref, 768, 0.999) ||
                !e2_test_match("image GPU vs CPU", gpu, cpu, 768, 0.999))) {
                failures++;
                break;
            }
            samples[checked++] = d;
            if (getenv("E2_GPU_VERBOSE")) printf("     run %d: %.3f ms\n", r, d);
            if (d < best) best = d;
        }
        if (!diagnostic) e2_test_timing(samples, checked);
        free(samples);
        if (diagnostic) {
            printf("E2_GPU_VLAYERS=%s: diagnostic run only\n", getenv("E2_GPU_VLAYERS"));
        } else {
            long k = 0;
            double last = 0;
            e2_gpu_vision_stats(g, &k, &last);
            printf("image          %d patches  gpu %8.2f ms  cos(gpu,ref) %.6f  cos(gpu,cpu) %.6f  "
                   "cos(cpu,ref) %.6f  kernels %ld (cpu %.1f ms)\n",
                   n_patch, best, cosine(gpu, ref, 768), cosine(gpu, cpu, 768),
                   cosine(cpu, ref, 768), k, tcpu);
            if (cosine(gpu, ref, 768) < 0.999 || cosine(gpu, cpu, 768) < 0.999) failures++;
        }
        free(gpu);
        free(cpu);
        free(fx);
    }

    if (!diagnostic) { /* text + image */
        char path[512];
        snprintf(path, sizeof path, "%s/image_caption.bin", argv[2]);
        int32_t n_ids = 0, n_patch = 0, p_dim = 0;
        int32_t *ids = NULL;
        float *fx = read_caption(path, &n_ids, &n_patch, &p_dim, &ids);
        if (!fx) { fprintf(stderr, "cannot read %s\n", path); return 2; }
        const float *pixels = fx;
        const int32_t *pos = (const int32_t *)(fx + (size_t)n_patch * p_dim);
        const float *ref = (const float *)(pos + (size_t)n_patch * 2);
        float *gpu = malloc(sizeof(float) * 768);
        float *cpu = malloc(sizeof(float) * 768);
        double tcpu = now_ms();
        if (e2_text_image_embed(m, ids, n_ids, pixels, pos, n_patch, cpu) ||
            !e2_test_unit("CPU caption reference", cpu, 768) ||
            !e2_test_match("CPU caption reference", cpu, ref, 768, 0.999)) {
            fprintf(stderr, "CPU reference failed: %s\n", e2_last_error());
            free(gpu); free(cpu); free(fx); free(ids);
            e2_gpu_destroy(g); e2_free(m);
            return 1;
        }
        tcpu = now_ms() - tcpu;
        double best = 1e18, samples[3];
        int checked = 0;
        for (int r = 0; r < 3; r++) {
            const double t = now_ms();
            if (e2_gpu_text_image_embed(g, ids, n_ids, pixels, pos, n_patch, gpu)) {
                fprintf(stderr, "gpu: %s\n", e2_gpu_last_error());
                failures++;
                break;
            }
            const double d = now_ms() - t;
            if (!e2_test_unit("image+caption", gpu, 768) ||
                !e2_test_match("image+caption", gpu, ref, 768, 0.999) ||
                !e2_test_match("caption GPU vs CPU", gpu, cpu, 768, 0.999)) {
                failures++;
                break;
            }
            samples[checked++] = d;
            if (d < best) best = d;
        }
        e2_test_timing(samples, checked);
        printf("image+caption  %d ids      gpu %8.2f ms  cos(gpu,ref) %.6f  cos(gpu,cpu) %.6f  "
               "cos(cpu,ref) %.6f  (cpu %.1f ms)\n",
               n_ids, best, cosine(gpu, ref, 768), cosine(gpu, cpu, 768), cosine(cpu, ref, 768),
               tcpu);
        if (cosine(gpu, ref, 768) < 0.999 || cosine(gpu, cpu, 768) < 0.999) failures++;
        free(gpu);
        free(cpu);
        free(fx);
        free(ids);
    }

    printf(failures ? "FAIL (%d)\n" : "ok\n", failures);
    e2_gpu_destroy(g);
    e2_free(m);
    return failures ? 1 : 0;
}
