/* test_image.c — run the vision tower on the frozen image fixture and compare with the fp32
 * reference embedding.
 *
 * Usage: e2_test_image <model_dir> <fixture_dir>
 * Fixture format (written by tools/dump_image_fixture.py):
 *   int32 n_patches, patch_dim, n_pos, dim
 *   float pixels[n_patches * patch_dim]
 *   int32 positions[n_patches * n_pos]
 *   float ref[dim]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "e2.h"
#include "test_metrics.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <fixture_dir>\n", argv[0]); return 2; }
    char path[512];
    snprintf(path, sizeof path, "%s/image.bin", argv[2]);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    int32_t n_patch = 0, p_dim = 0, n_pos = 0, dim = 0;
    if (fread(&n_patch, 4, 1, f) != 1 || fread(&p_dim, 4, 1, f) != 1 || fread(&n_pos, 4, 1, f) != 1 ||
        fread(&dim, 4, 1, f) != 1) {
        fprintf(stderr, "bad fixture header\n");
        return 2;
    }
    float *pixels = malloc(sizeof(float) * (size_t)n_patch * p_dim);
    int32_t *pos = malloc(sizeof(int32_t) * (size_t)n_patch * n_pos);
    float *ref = malloc(sizeof(float) * (size_t)dim);
    if (fread(pixels, 4, (size_t)n_patch * p_dim, f) != (size_t)n_patch * p_dim ||
        fread(pos, 4, (size_t)n_patch * n_pos, f) != (size_t)n_patch * n_pos ||
        fread(ref, 4, (size_t)dim, f) != (size_t)dim) {
        fprintf(stderr, "truncated fixture\n");
        return 2;
    }
    fclose(f);

    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "load failed: %s\n", e2_last_error()); return 1; }
    e2_info(m);

    const char *re = getenv("E2_REPS");
    int reps = re ? atoi(re) : 3;
    if (reps < 1) reps = 1;
    float *out = malloc(sizeof(float) * (size_t)dim);
    double dt = 1e18;
    double *samples = malloc(sizeof *samples * (size_t)reps);
    if (!samples) return 2;
    int rc = 0;
    for (int r = 0; r < reps; r++) {
        double t = now_ms();
        rc = e2_image_embed(m, pixels, pos, n_patch, out);
        double d = now_ms() - t;
        samples[r] = d;
        if (rc || !e2_test_unit("image", out, (size_t)dim) ||
            !e2_test_match("image", out, ref, (size_t)dim, 0.999)) { rc = -1; break; }
        if (d < dt) dt = d;
    }
    if (!rc) e2_test_timing(samples, reps);
    free(samples);
    int failures = 0;
    if (rc) {
        fprintf(stderr, "image: %s\n", e2_last_error());
        failures++;
    } else {
        double dot = 0, na = 0, nb = 0, maxdiff = 0;
        for (int d = 0; d < dim; d++) {
            dot += out[d] * ref[d];
            na += out[d] * out[d];
            nb += ref[d] * ref[d];
            double diff = fabs(out[d] - ref[d]);
            if (diff > maxdiff) maxdiff = diff;
        }
        double cos = dot / (sqrt(na) * sqrt(nb));
        printf("image (%d patches) %8.2f ms (best of %d)  cos=%.6f  max|delta|=%.5f  %s\n", n_patch, dt,
               reps, cos, maxdiff, cos > 0.999 ? "PASS" : "FAIL");
        if (!(cos > 0.999)) failures++;
    }
    /* text + image: <|image|> placeholders expanded into the caption */
    snprintf(path, sizeof path, "%s/image_caption.bin", argv[2]);
    f = fopen(path, "rb");
    if (f) {
        int32_t n_ids = 0, np2 = 0, pd2 = 0, d2 = 0;
        if (fread(&n_ids, 4, 1, f) == 1 && fread(&np2, 4, 1, f) == 1 && fread(&pd2, 4, 1, f) == 1 &&
            fread(&d2, 4, 1, f) == 1) {
            int32_t *ids = malloc(sizeof(int32_t) * (size_t)n_ids);
            float *px = malloc(sizeof(float) * (size_t)np2 * pd2);
            int32_t *ps = malloc(sizeof(int32_t) * (size_t)np2 * 2);
            float *rf = malloc(sizeof(float) * (size_t)d2);
            if (fread(ids, 4, (size_t)n_ids, f) == (size_t)n_ids &&
                fread(px, 4, (size_t)np2 * pd2, f) == (size_t)np2 * pd2 &&
                fread(ps, 4, (size_t)np2 * 2, f) == (size_t)np2 * 2 &&
                fread(rf, 4, (size_t)d2, f) == (size_t)d2) {
                float *o2 = malloc(sizeof(float) * (size_t)d2);
                double t = now_ms();
                if (e2_text_image_embed(m, ids, n_ids, px, ps, np2, o2)) {
                    fprintf(stderr, "image_caption: %s\n", e2_last_error());
                    failures++;
                } else {
                    double dot = 0, na = 0, nb = 0, md = 0;
                    for (int d = 0; d < d2; d++) {
                        dot += o2[d] * rf[d];
                        na += o2[d] * o2[d];
                        nb += rf[d] * rf[d];
                        double diff = fabs(o2[d] - rf[d]);
                        if (diff > md) md = diff;
                    }
                    double cos = dot / (sqrt(na) * sqrt(nb));
                    printf("image_caption (%d ids) %8.2f ms  cos=%.6f  max|delta|=%.5f  %s\n", n_ids,
                           now_ms() - t, cos, md, cos > 0.999 ? "PASS" : "FAIL");
                    if (!(cos > 0.999) || !e2_test_unit("image_caption", o2, (size_t)d2)) failures++;
                }
                free(o2);
            }
            free(ids); free(px); free(ps); free(rf);
        }
        fclose(f);
    }

    free(out);
    free(ref);
    free(pos);
    free(pixels);
    e2_free(m);
    printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
