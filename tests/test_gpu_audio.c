/* test_gpu_audio.c — the audio tower on the GPU against the frozen fp32 reference.
 *
 * Usage: e2_test_gpu_audio <model_dir> <fixture_dir>
 *
 * The fixture is the same one e2_test_audio uses (tools/dump_audio_fixture.py): 0.75 s of 16 kHz PCM,
 * its reference mel, the reference [19,512] soft-token block and the reference 768-d embeddings. The
 * front end stays on the CPU, so this compares the twelve Conformer layers - the part the GPU owns.
 */
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

static double cosine(const float *a, const float *b, size_t n, double *maxd) {
    double dot = 0, na = 0, nb = 0, md = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
        const double d = fabs((double)a[i] - b[i]);
        if (d > md) md = d;
    }
    if (maxd) *maxd = md;
    return dot / sqrt(na * nb + 1e-30);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <fixture_dir>\n", argv[0]); return 2; }
    char path[512];
    snprintf(path, sizeof path, "%s/audio.bin", argv[2]);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    int32_t hd[8] = { 0 };
    if (fread(hd, 4, 8, f) != 8) return 2;
    const int n_samples = hd[0], n_frames = hd[1], n_soft = hd[2], soft_dim = hd[5], text_len = hd[7];
    char *text = calloc((size_t)text_len + 1, 1);
    float *pcm = malloc(sizeof(float) * (size_t)n_samples);
    float *mel = malloc(sizeof(float) * (size_t)n_frames * 128);
    uint8_t *mask = malloc((size_t)n_frames);
    float *ref = malloc(sizeof(float) * (size_t)n_soft * soft_dim);
    float *got = malloc(sizeof(float) * (size_t)n_soft * soft_dim);
    if (!text || !pcm || !mel || !mask || !ref || !got) return 2;
    const int n_ids = hd[4];
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)n_ids);
    if (!ids) return 2;
    if (fread(text, 1, (size_t)text_len, f) != (size_t)text_len ||
        fread(pcm, 4, (size_t)n_samples, f) != (size_t)n_samples ||
        fread(ids, 4, (size_t)n_ids, f) != (size_t)n_ids ||
        fread(mel, 4, (size_t)n_frames * 128, f) != (size_t)n_frames * 128 ||
        fread(mask, 1, (size_t)n_frames, f) != (size_t)n_frames ||
        fread(ref, 4, (size_t)n_soft * soft_dim, f) != (size_t)n_soft * soft_dim) {
        fprintf(stderr, "truncated fixture\n");
        return 2;
    }
    fclose(f);
    (void)mel;
    (void)mask;
    (void)ids;
    printf("fixture: %d samples, %d soft tokens, \"%s\"\n", n_samples, n_soft, text);

    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "load failed: %s\n", e2_last_error()); return 1; }
    e2_gpu *g = e2_gpu_create(m);
    if (!g) { fprintf(stderr, "gpu create failed: %s\n", e2_gpu_last_error()); return 1; }
    printf("gpu: %s\n", e2_gpu_device(g));

    /* optional synthetic-length mode: e2_test_gpu_audio <model> <fixture> <seconds> */
    if (argc >= 4) {
        const double sec = atof(argv[3]);
        const int n = (int)(16000.0 * sec);
        float *sp = malloc(sizeof(float) * (size_t)n);
        float *sout = malloc(sizeof(float) * (size_t)750 * 512);
        if (!sp || !sout) return 2;
        for (int i = 0; i < n; i++) sp[i] = 0.2f * sinf(2.f * 3.14159265f * 220.f * (float)i / 16000.f);
        const char *sr = getenv("E2_REPS");
        int sreps = sr ? atoi(sr) : 3;
        if (sreps < 1) sreps = 1;
        double sbest = 1e18;
        int sn = 0;
        for (int i = 0; i < 2; i++)
            if (e2_gpu_audio_soft(g, sp, n, sout, &sn)) return 1;
        for (int r = 0; r < sreps; r++) {
            const double t0 = now_ms();
            if (e2_gpu_audio_soft(g, sp, n, sout, &sn)) {
                fprintf(stderr, "synthetic: %s\n", e2_gpu_last_error());
                return 1;
            }
            const double dt = now_ms() - t0;
            if (sn != e2_audio_tokens(n) ||
                !e2_test_match("synthetic finite soft tokens", sout, sout, (size_t)sn * 512, 0.99))
                return 1;
            if (dt < sbest) sbest = dt;
        }
        long sk = 0;
        double sl = 0;
        e2_gpu_audio_stats(g, &sk, &sl);
        printf("synthetic %.2f s -> %d tokens  %8.2f ms (best of %d, %ld kernels, graph %.2f ms)\n",
               sec, sn, sbest, sreps, sk, sl);
        free(sp);
        free(sout);
        e2_gpu_destroy(g);
        e2_free(m);
        return 0;
    }

    const char *re = getenv("E2_REPS");
    int reps = re ? atoi(re) : 5;
    if (reps < 1) reps = 1;
    double *samples = malloc(sizeof *samples * (size_t)reps);
    if (!samples) return 2;
    const char *wu = getenv("E2_GPU_WARMUP");
    const int warm = wu ? atoi(wu) : 10; /* the iGPU idles at 800 MHz; timing starts after a burst */
    double best = 1e18;
    int rc = 0, ns = 0;
    for (int i = 0; i < warm; i++)
        if (e2_gpu_audio_soft(g, pcm, n_samples, got, &ns)) return 1;
    for (int r = 0; r < reps; r++) {
        const double t0 = now_ms();
        rc = e2_gpu_audio_soft(g, pcm, n_samples, got, &ns);
        const double dt = now_ms() - t0;
        if (rc || ns != n_soft ||
            !e2_test_match("audio soft tokens", got, ref, (size_t)n_soft * soft_dim, 0.99)) {
            fprintf(stderr, "audio: %s\n", e2_gpu_last_error());
            return 1;
        }
        samples[r] = dt;
        if (dt < best) best = dt;
    }
    if (ns != n_soft) {
        fprintf(stderr, "soft-token count %d, expected %d\n", ns, n_soft);
        return 1;
    }
    e2_test_timing(samples, reps);
    free(samples);
    double md = 0;
    const double cos = cosine(got, ref, (size_t)n_soft * soft_dim, &md);
    long kernels = 0;
    double last = 0;
    e2_gpu_audio_stats(g, &kernels, &last);
    printf("gpu audio soft tokens  %8.2f ms (best of %d, %ld kernels, last run %.2f ms)\n", best, reps,
           kernels, last);
    printf("  soft tokens          cos=%.6f  max|d|=%.5f  %s\n", cos, md, cos > 0.99 ? "PASS" : "FAIL");
    if (!(cos > 0.99)) return 1;

    /* the same tokens through the GPU text tower, checked against the CPU engine's end-to-end vector */
    float ref768[768], gpu768[768];
    if (e2_audio_embed(m, pcm, n_samples, ref768)) {
        fprintf(stderr, "cpu audio embed: %s\n", e2_last_error());
        return 1;
    }
    if (!e2_test_unit("CPU audio reference", ref768, 768)) return 1;
    if (e2_gpu_audio_embed(g, pcm, n_samples, gpu768)) {
        fprintf(stderr, "gpu audio embed: %s\n", e2_gpu_last_error());
        return 1;
    }
    if (!e2_test_unit("GPU audio embedding", gpu768, 768)) return 1;
    const double ecos = cosine(gpu768, ref768, 768, &md);
    printf("  768d vs CPU engine    cos=%.6f  max|d|=%.5f  %s\n", ecos, md, ecos > 0.998 ? "PASS" : "FAIL");
    if (!(ecos > 0.998)) return 1;

    e2_gpu_destroy(g);
    e2_free(m);
    printf("OK\n");
    return 0;
}
