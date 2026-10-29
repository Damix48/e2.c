/* Worker for bench_compare.py; binary inputs are prepared and checked by that script.
 * Header: int32 kind(0=text,1=image,2=audio), batch, length, repeat; payload; batch*768 FP32
 * reference. Payload: text int32 ids[B,S]; image float pixels[N,768], int32 positions[N,2]; audio
 * float PCM[N]. `repeat` times a single image/audio call back to back inside one timed call; those
 * towers have no batched entry point, so it measures batch throughput, not a batched kernel. */
#include <errno.h>
#include <math.h>
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

static int number(const char *s, int min) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    return !errno && *s && !*end && n >= min && n <= 10000 ? (int)n : -1;
}
static void *read_array(FILE *f, size_t n) {
    void *p = malloc(n);
    if (!p || fread(p, 1, n, f) != n) { free(p); return NULL; }
    return p;
}
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s model_dir cpu|gpu input.bin runs warmup output.f32\n", argv[0]);
        return 2;
    }
    int gpu = !strcmp(argv[2], "gpu"), runs = number(argv[4], 1), warmup = number(argv[5], 0);
    if ((!gpu && strcmp(argv[2], "cpu")) || runs < 0 || warmup < 0) return 2;
#ifndef E2_BENCH_GPU
    if (gpu) { fprintf(stderr, "GPU backend not built\n"); return 2; }
#endif
    FILE *f = fopen(argv[3], "rb");
    int32_t h[4];
    if (!f) { perror(argv[3]); return 2; }
    if (fread(h, sizeof h, 1, f) != 1) { fclose(f); return 2; }
    int kind = h[0], batch = h[1], n = h[2], repeat = h[3];
    if (kind < 0 || kind > 2 || batch < 1 || batch > 32 || n < 1 || repeat < 1 || repeat > 16 ||
        (kind == 0 && n > 2048) || (kind == 1 && n > E2_MAX_PATCHES) ||
        (kind == 2 && n > 480000) || (kind != 0 && batch != 1) ||
        (kind == 0 && repeat != 1)) { fclose(f); return 2; }
    size_t count = kind == 0 ? (size_t)batch*n : kind == 1 ? (size_t)n*768 : (size_t)n;
    void *input = read_array(f, count*4);
    int32_t *pos = kind == 1 ? read_array(f, (size_t)n*2*4) : NULL;
    float *ref = read_array(f, (size_t)batch*768*4);
    int trailing = fgetc(f);
    fclose(f);
    float *out = malloc((size_t)batch*768*4);
    double *samples = malloc((size_t)runs*sizeof *samples);
    if (!input || (kind == 1 && !pos) || !ref || !out || !samples || trailing != EOF) return 2;
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }
#ifdef E2_BENCH_GPU
    e2_gpu *g = gpu ? e2_gpu_create(m) : NULL;
    if (gpu && !g) { fprintf(stderr, "%s\n", e2_gpu_last_error()); e2_free(m); return 1; }
#endif
    double minimum = 1.0;
    int failed = 0;
    for (int r = -warmup; r < runs; r++) {
        /* Poisoning catches unwritten output, though not internal stale intermediates. */
        for (int i = 0; i < batch*768; i++) out[i] = NAN;
        double t = now_ms();
        int rc = 0;
        for (int rep = 0; rep < repeat && !rc; rep++) {
#ifdef E2_BENCH_GPU
            if (gpu) {
                if (kind == 0) rc = e2_gpu_text_embed_batch(g, input, NULL, batch, n, out);
                else if (kind == 1) rc = e2_gpu_image_embed(g, input, pos, n, out);
                else rc = e2_gpu_audio_embed(g, input, n, out);
            } else
#endif
            {
                if (kind == 0) rc = batch == 1 ? e2_text_embed(m, input, n, out) :
                                                e2_text_embed_batch(m, input, NULL, batch, n, out);
                else if (kind == 1) rc = e2_image_embed(m, input, pos, n, out);
                else rc = e2_audio_embed(m, input, n, out);
            }
        }
        double elapsed = now_ms() - t;
        if (rc) { fprintf(stderr, "inference failed: %s\n", e2_last_error()); failed = 1; break; }
        for (int b = 0; b < batch; b++) {
            if (!e2_test_unit("benchmark", out+b*768, 768) ||
                !e2_test_match("benchmark vs FP32", out+b*768, ref+b*768, 768, 0.999)) {
                fprintf(stderr, "invalid output at iteration %d, item %d\n", r, b);
                failed = 1; break;
            }
            double c = e2_test_cosine(out+b*768, ref+b*768, 768);
            if (c < minimum) minimum = c;
        }
        if (failed) break;
        if (r >= 0) samples[r] = elapsed;
    }
    if (!failed) {
        f = fopen(argv[6], "wb");
        if (!f || fwrite(out, 4, (size_t)batch*768, f) != (size_t)batch*768) failed = 1;
        if (f && fclose(f)) failed = 1;
    }
    if (!failed) {
        printf("{\"minimum_cosine\":%.12f,\"samples_ms\":[", minimum);
        for (int r = 0; r < runs; r++) printf("%s%.6f", r ? "," : "", samples[r]);
        printf("]}\n");
    }
#ifdef E2_BENCH_GPU
    if (g) e2_gpu_destroy(g);
#endif
    e2_free(m);
    free(input); free(pos); free(ref); free(out); free(samples);
    return failed ? 1 : 0;
}
