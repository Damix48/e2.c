/* Long-lived mixed-shape worker: every result checked, inputs change on cached graphs. */
#include "model.h"
#include "e2gpu.h"
#include "test_metrics.h"
#include "gpu_text_cases.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s: GPU='%s' CPU='%s'\n", __FILE__, __LINE__, #c, \
            e2_gpu_last_error(), e2_last_error()); exit(1); } } while (0)
enum { BATCH = 125, SEQ = 17, SAMPLE_CAP = 65536 };
static double g_samples[SAMPLE_CAP];
static size_t g_measured;
static unsigned long g_calls;
static double g_worst;
static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static void record_time(double start) {
    double ms = now_ms() - start;
    if (g_measured < SAMPLE_CAP) g_samples[g_measured++] = ms;
    if (ms > g_worst) g_worst = ms;
    g_calls++;
}
static int order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void match(const char *label, const float *out, const float *ref, double minimum) {
    CHECK(!e2_gpu_last_error()[0]);
    CHECK(e2_test_unit(label, out, 768));
    CHECK(e2_test_match(label, out, ref, 768, minimum));
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s MODEL_DIR FIXTURES SECONDS (0 = one cycle)\n", argv[0]); return 2; }
    char *end; long seconds = strtol(argv[3], &end, 10);
    if (!*argv[3] || *end || seconds < 0 || seconds > 86400) return 2;
    e2_model *m = e2_load(argv[1]); CHECK(m);
    e2_gpu *g = e2_gpu_create(m); CHECK(g);
    printf("soak device: %s; requested %ld seconds\n", e2_gpu_device(g), seconds); fflush(stdout);

    int count;
    e2_test_text_case *cases = e2_test_read_text_cases(argv[2], &count);
    float (*text_ref)[768] = calloc((size_t)count, sizeof *text_ref); CHECK(text_ref);
    for (int c = 0; c < count; c++)
        CHECK(!e2_text_embed(m, cases[c].ids, cases[c].n, text_ref[c]));
    int32_t batch_ids[BATCH * SEQ]; uint8_t valid[BATCH * SEQ];
    float batch_ref[6][768];
    float *batch_out = malloc((size_t)BATCH * 768 * sizeof(float)); CHECK(batch_out);
    for (int b = 0; b < BATCH; b++) {
        const int v = b % 6;
        for (int i = 0; i < SEQ; i++) {
            const e2_test_text_case *c = cases + v % 2;
            batch_ids[b * SEQ + i] = c->ids[i % c->n];
            valid[b * SEQ + i] = v % 3 == 2 ? 0 :
                (i < c->n && (v % 3 == 0 || i % 4 != v % 4) ? 255 : 0);
        }
        if (b < 6) CHECK(!e2_text_embed_batch(m, batch_ids + b * SEQ, valid + b * SEQ,
                                             1, SEQ, batch_ref[b]));
    }
    /* This batch spans multiple chunks, including exact-zero all-masked sequences. */
    struct {
        int np, nids;
        float *pixels, ref[768], cpu[768];
        int32_t *pos, *ids;
    } image[2] = {0};
    if (m->vision) for (int v = 0; v < 2; v++) {
        char path[4096]; snprintf(path, sizeof path, "%s/%s.bin", argv[2], v ? "image_caption" : "image");
        FILE *f = fopen(path, "rb"); CHECK(f);
        int32_t h[4]; CHECK(fread(h, sizeof h, 1, f) == 1);
        image[v].np = h[v ? 1 : 0]; image[v].nids = v ? h[0] : 0;
        CHECK(image[v].np > 0 && image[v].np <= E2_V_MAX_PATCHES && h[v ? 2 : 1] == E2_V_PATCH);
        CHECK(h[3] == 768 && image[v].nids >= 0 && image[v].nids <= 2048 && (v || h[2] == 2));
        const size_t np = (size_t)image[v].np, ni = (size_t)image[v].nids;
        image[v].pixels = malloc(np * E2_V_PATCH * sizeof(float));
        image[v].pos = malloc(np * 2 * sizeof(int32_t));
        image[v].ids = malloc((ni + 1) * sizeof(int32_t));
        CHECK(image[v].pixels && image[v].pos && image[v].ids);
        CHECK(fread(image[v].ids, 4, ni, f) == ni);
        CHECK(fread(image[v].pixels, 4, np * E2_V_PATCH, f) == np * E2_V_PATCH);
        CHECK(fread(image[v].pos, 4, np * 2, f) == np * 2);
        CHECK(fread(image[v].ref, sizeof image[v].ref, 1, f) == 1); fclose(f);
        CHECK(!(v ? e2_text_image_embed(m, image[v].ids, image[v].nids, image[v].pixels,
                                        image[v].pos, image[v].np, image[v].cpu) :
                     e2_image_embed(m, image[v].pixels, image[v].pos, image[v].np, image[v].cpu)));
    }
    float *pcm = NULL, audio_ref[2][768];
    int audio_samples[2] = {0, 480000};
    if (m->audio) {
        pcm = malloc((size_t)2 * 480000 * sizeof(float)); CHECK(pcm);
        char path[4096]; snprintf(path, sizeof path, "%s/audio.bin", argv[2]);
        FILE *f = fopen(path, "rb"); CHECK(f);
        int32_t h[8]; CHECK(fread(h, sizeof h, 1, f) == 1);
        CHECK(h[0] > 0 && h[0] <= 480000 && h[7] >= 0);
        CHECK(!fseek(f, h[7], SEEK_CUR));
        audio_samples[0] = h[0];
        CHECK(fread(pcm, 4, (size_t)h[0], f) == (size_t)h[0]); fclose(f);
        CHECK(!e2_audio_embed(m, pcm, h[0], audio_ref[0]));
        for (int i = 0; i < 480000; i++)
            pcm[480000+i] = (float)(.2 * sin(2 * 3.14159265358979323846 * 220 * i / 16000) +
                                    .1 * cos(2 * 3.14159265358979323846 * 530 * i / 16000));
        snprintf(path, sizeof path, "%s/audio_checks.bin", argv[2]);
        f = fopen(path, "rb"); CHECK(f);
        uint32_t header[4], n[2]; CHECK(fread(header, sizeof header, 1, f) == 1);
        CHECK(header[0] == 0x41323245 && header[1] == 1 && header[2] == 9 && header[3] == 768);
        CHECK(!fseek(f, 8 * (8 + 768 * 4), SEEK_CUR)); /* independent 30-second FP32 reference */
        CHECK(fread(n, sizeof n, 1, f) == 1 && n[0] == 480000 && n[1] == 750);
        CHECK(fread(audio_ref[1], sizeof audio_ref[1], 1, f) == 1); fclose(f);
    }
    double started = now_ms(), next_report = started + 60000;
    unsigned long cycles = 0;
    do {
        float out[768];
        for (int c = 0; c < count; c++) {
            double t = now_ms(); CHECK(!e2_gpu_text_embed(g, cases[c].ids, cases[c].n, out));
            record_time(t); match(cases[c].name, out, cases[c].ref, .999);
            match("soak text vs CPU", out, text_ref[c], .999);
        }
        double t = now_ms();
        CHECK(!e2_gpu_text_embed_batch(g, batch_ids, valid, BATCH, SEQ, batch_out));
        record_time(t);
        for (int b = 0; b < BATCH; b++) {
            if (b % 3 == 2) {
                for (int i = 0; i < 768; i++) CHECK(batch_out[b * 768 + i] == 0.f);
            } else match("soak masked batch", batch_out + b * 768, batch_ref[b % 6], .999);
        }
        if (m->vision) for (int v = 0; v < 2; v++) {
            t = now_ms();
            CHECK(!(v ? e2_gpu_text_image_embed(g, image[v].ids, image[v].nids, image[v].pixels,
                                               image[v].pos, image[v].np, out) :
                        e2_gpu_image_embed(g, image[v].pixels, image[v].pos, image[v].np, out)));
            record_time(t); match("soak image vs FP32", out, image[v].ref, .999);
            match("soak image vs CPU", out, image[v].cpu, .999);
        }
        if (m->audio) for (int v = 0; v < 2; v++) {
            t = now_ms();
            CHECK(!e2_gpu_audio_embed(g, pcm + (size_t)v * 480000, audio_samples[v], out));
            record_time(t); match(v ? "soak tone vs FP32" : "soak audio vs CPU",
                                  out, audio_ref[v], v ? .98 : .998);
        }
        cycles++;
        if (now_ms() >= next_report) {
            printf("soak: %lu cycles, %lu checked calls, %.1f seconds\n", cycles, g_calls,
                   (now_ms() - started) / 1000); fflush(stdout);
            next_report = now_ms() + 60000;
        }
    } while (now_ms() - started < seconds * 1000.0);
    qsort(g_samples, g_measured, sizeof g_samples[0], order);
    struct rusage usage; CHECK(!getrusage(RUSAGE_SELF, &usage));
    printf("soak passed: %lu cycles, %lu checked calls, %.2f s; latency first %zu calls "
           "p50=%.3f p95=%.3f p99=%.3f max(all)=%.3f ms; peak process RSS=%ld KiB (Linux)\n",
           cycles, g_calls, (now_ms() - started) / 1000, g_measured, g_samples[g_measured / 2],
           g_samples[(g_measured * 95 - 1) / 100], g_samples[(g_measured * 99 - 1) / 100], g_worst, usage.ru_maxrss);
    e2_gpu_destroy(g); e2_free(m); free(batch_out); free(pcm); free(text_ref); free(cases);
    for (int v = 0; v < 2; v++) { free(image[v].pixels); free(image[v].pos); free(image[v].ids); }
    return 0;
}
