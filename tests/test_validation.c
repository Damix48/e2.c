/* test_validation.c — the public input boundaries of the multimodal entry points.
 *
 * Usage: e2_test_validation <model_dir>
 *
 * Every case here is caller-supplied data the engine used to trust: token ids that index the
 * embedding table, patch positions that index the axial position table and size the pool grid, NULL
 * pointers and counts outside the documented ranges. All of them must come back as an error message
 * before the call reads a weight, leave the caller's output buffer untouched, and never crash.
 * The valid cases at the end are the other half of the contract: a legitimate image layout and a
 * real audio clip still run.
 *
 * Built with -DE2_TEST_GPU (and linked against e2_gpu) the same file also exercises the Intel GPU
 * boundaries: create(NULL), a NULL context, invalid ids including the masked padding rows of a
 * batch, NULL outputs and pointers, and the (300,300) image, all of which must be refused before a
 * graph is recorded. Those cases need no device; only the final valid text/batch calls touch it.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "e2.h"
#include "../src/model.h" /* inject a numeric failure without modifying mapped weights */
#ifdef E2_TEST_GPU
#include "e2gpu.h"
#endif

#define PATCH_DIM 768
#define N_PATCHES_MAX 2520 /* e2_image_max_patches(); only used to exceed it */
#define POS_TABLE 10240    /* the model's position table width */
#define SENTINEL 12345.0f  /* "not written": a rejected call must leave the output buffer alone */

static int failures;
static float out[2 * 768];

static void out_fill(int n) {
    for (int i = 0; i < n; i++) out[i] = SENTINEL;
}

static int out_clean(int n) {
    for (int i = 0; i < n; i++)
        if (out[i] != SENTINEL) return 0;
    return 1;
}

static void rejected_n(const char *what, int rc, const char *err, int n) {
    if (rc == 0 || !err[0]) {
        printf("  FAIL %s: accepted (rc=%d, err=\"%s\")\n", what, rc, err);
        failures++;
    } else if (!out_clean(n)) {
        printf("  FAIL %s: wrote the output buffer (\"%s\")\n", what, err);
        failures++;
    } else {
        printf("  ok   %s: %s\n", what, err);
    }
}

/* Fill the output with a sentinel, run the call, require a rejection and an untouched buffer. */
#define REJECTED(what, n, call)                       \
    do {                                              \
        out_fill(n);                                  \
        rejected_n(what, (call), e2_last_error(), n); \
    } while (0)
#ifdef E2_TEST_GPU
#define REJECTED_GPU(what, n, call)                       \
    do {                                                  \
        out_fill(n);                                      \
        rejected_n(what, (call), e2_gpu_last_error(), n); \
    } while (0)
#endif

/* A valid run has to produce a unit vector: "valid data unchanged" at the API boundary, without
 * pinning the exact numbers (the fixtures do that). */
static void accepted(const char *what, int rc, const char *err) {
    if (rc != 0 || err[0]) {
        printf("  FAIL %s: rejected (\"%s\")\n", what, err);
        failures++;
        return;
    }
    double n = 0;
    int finite = 1;
    for (int i = 0; i < 768; i++) {
        if (!isfinite(out[i])) finite = 0;
        n += (double)out[i] * out[i];
    }
    n = sqrt(n);
    if (!finite || fabs(n - 1.0) > 1e-3) {
        printf("  FAIL %s: not a unit vector (|v|=%.6f, finite=%d)\n", what, n, finite);
        failures++;
        return;
    }
    printf("  ok   %s: |v|=%.6f\n", what, n);
}

/* A k x k patch grid: the layout the HF image processor emits (dense cells from (0,0)). */
static int grid_positions(int32_t *pos, int k) {
    int n = 0;
    for (int y = 0; y < k; y++)
        for (int x = 0; x < k; x++) {
            pos[2 * n] = x;
            pos[2 * n + 1] = y;
            n++;
        }
    return n;
}

static void cpu_sections(e2_model *m, float *pixels, int32_t *pos) {
    printf("model and token ids\n");
    grid_positions(pos, 3);
    REJECTED("image NULL model", 768, e2_image_embed(NULL, pixels, pos, 9, out));
    REJECTED("text+image NULL model", 768,
             e2_text_image_embed(NULL, (int32_t[]){ 2, 258880, 1 }, 3, pixels, pos, 9, out));
    REJECTED("audio NULL model", 768, e2_audio_embed(NULL, pixels, 3200, out));
    int32_t ids[8] = { 2, 258880, 1, 0 }; /* <bos> <|image|> <eos> */
    ids[1] = -5;
    REJECTED("text+image id -5", 768, e2_text_image_embed(m, ids, 3, pixels, pos, 9, out));
    ids[1] = 258880;
    REJECTED("text+image id == vocab", 768,
             e2_text_image_embed(m, (int32_t[]){ 2, 262144, 1 }, 3, pixels, pos, 9, out));
    REJECTED("text+image NULL ids", 768, e2_text_image_embed(m, NULL, 3, pixels, pos, 9, out));
    out_fill(768);
    rejected_n("text+image NULL out", e2_text_image_embed(m, ids, 3, pixels, pos, 9, NULL),
               e2_last_error(), 0);
    REJECTED("text+audio NULL ids", 768, e2_text_audio_embed(m, NULL, 3, pixels, 3200, out));
    REJECTED("text+audio id -5", 768,
             e2_text_audio_embed(m, (int32_t[]){ 2, -5, 1 }, 3, pixels, 3200, out));
    rejected_n("text+audio NULL out", e2_text_audio_embed(m, ids, 3, pixels, 3200, NULL),
               e2_last_error(), 0);
    REJECTED("text+image+audio NULL positions", 768,
             e2_text_image_audio_embed(m, ids, 3, pixels, NULL, 9, pixels, 3200, out));
    REJECTED("text+image+audio bad id", 768,
             e2_text_image_audio_embed(m, (int32_t[]){ 2, 262144, 1 }, 3, pixels, pos, 9, pixels,
                                      3200, out));

    printf("patch positions\n");
    pos[0] = 0;
    pos[1] = 0;
    pos[2] = -1;
    pos[3] = 3; /* half-padding: neither a cell nor (-1,-1) */
    REJECTED("mixed padding pair (-1,3)", 768, e2_image_embed(m, pixels, pos, 2, out));
    pos[0] = -1;
    pos[1] = -1;
    pos[2] = -1;
    pos[3] = -1;
    REJECTED("all padding", 768, e2_image_embed(m, pixels, pos, 2, out));
    REJECTED("zero patches", 768, e2_image_embed(m, pixels, pos, 0, out));
    REJECTED("too many patches", 768, e2_image_embed(m, pixels, pos, N_PATCHES_MAX + 1, out));
    REJECTED("NULL pixels", 768, e2_image_embed(m, NULL, pos, 1, out));
    REJECTED("NULL positions", 768, e2_image_embed(m, pixels, NULL, 1, out));
    rejected_n("NULL out", e2_image_embed(m, pixels, pos, 1, NULL), e2_last_error(), 0);
    pos[0] = POS_TABLE;
    pos[1] = 0;
    REJECTED("position past the table", 768, e2_image_embed(m, pixels, pos, 1, out));
    pos[0] = 300;
    pos[1] = 300; /* one patch, a 101x101 pool grid out of a 2044-row soft-token block */
    REJECTED("single patch (300,300)", 768, e2_image_embed(m, pixels, pos, 1, out));
    /* One patch per pool group: 2520 groups fit the pool buffer but not the 2048-row soft-token
     * block the pooler emits into, so the grid is rejected before the tower writes it. */
    for (int i = 0; i < N_PATCHES_MAX; i++) {
        pos[2 * i] = 3 * i;
        pos[2 * i + 1] = 0;
    }
    REJECTED("one patch per pool group", 768, e2_image_embed(m, pixels, pos, N_PATCHES_MAX, out));

    printf("numeric image boundaries\n");
    grid_positions(pos, 3);
    const float invalid[] = { NAN, INFINITY, -INFINITY, -0.01f, 1.01f };
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; i++) {
        float saved = pixels[0]; pixels[0] = invalid[i];
        REJECTED("nonfinite/out-of-range image pixel", 768, e2_image_embed(m, pixels, pos, 9, out));
        pixels[0] = saved;
    }
    printf("valid input still runs\n");
    const int np = grid_positions(pos, 3);
    out_fill(768);
    accepted("image 3x3 grid", e2_image_embed(m, pixels, pos, np, out), e2_last_error());
    pos[2 * np] = -1; /* and a (-1,-1)-padded layout */
    pos[2 * np + 1] = -1;
    out_fill(768);
    accepted("image 3x3 grid + padding", e2_image_embed(m, pixels, pos, np + 1, out), e2_last_error());
    ids[1] = 258880;
    out_fill(768);
    accepted("text+image 3x3 grid", e2_text_image_embed(m, ids, 3, pixels, pos, np, out),
             e2_last_error());

    printf("audio\n");
    REJECTED("audio NULL pcm", 768, e2_audio_embed(m, NULL, 3200, out));
    REJECTED("audio zero samples", 768, e2_audio_embed(m, pixels, 0, out));
    REJECTED("audio too short", 768, e2_audio_embed(m, pixels, 160, out));
    rejected_n("audio NULL out", e2_audio_embed(m, pixels, 3200, NULL), e2_last_error(), 0);
    float pcm[3200];
    for (int i = 0; i < 3200; i++)
        pcm[i] = 0.5f * sinf(2.0f * 3.14159265358979f * 440.0f * (float)i / 16000.0f);
    const float bad_pcm[] = { NAN, INFINITY, -INFINITY, -1.01f, 1.01f };
    for (size_t i = 0; i < sizeof bad_pcm / sizeof bad_pcm[0]; i++) {
        pcm[0] = bad_pcm[i];
        REJECTED("nonfinite/out-of-range PCM", 768, e2_audio_embed(m, pcm, 3200, out));
        REJECTED("nonfinite text+audio PCM", 768,
                 e2_text_audio_embed(m, (int32_t[]){2,1}, 2, pcm, 3200, out));
    }
    pcm[0] = 0.f;
    out_fill(768);
    accepted("audio 0.2 s sine", e2_audio_embed(m, pcm, 3200, out), e2_last_error());
    if (m->use_int8) {
        float bad_scale[768];
        memcpy(bad_scale, m->q8_embed_proj.scale, sizeof bad_scale);
        bad_scale[0] = NAN;
        float *saved = m->q8_embed_proj.scale;
        m->q8_embed_proj.scale = bad_scale;
        REJECTED("nonfinite computed embedding", 768, e2_text_embed(m, (int32_t[]){2,1}, 2, out));
        m->q8_embed_proj.scale = saved;
    }
    REJECTED("text bad geometry", 768, e2_text_embed(m, (int32_t[]){2,1}, 0, out));
    accepted("text clears stale error", e2_text_embed(m, (int32_t[]){2,1}, 2, out), e2_last_error());
}

#ifdef E2_TEST_GPU
/* Intel GPU boundaries. Everything here except the last two calls is refused by the host before a
 * graph is recorded or the device is touched, so these cases run on any machine with the backend. */
static void gpu_section(e2_model *m, const float *pixels, int32_t *pos) {
    printf("Intel GPU boundaries\n");
    e2_gpu *g0 = e2_gpu_create(NULL); /* must be refused, not dereferenced */
    if (g0) {
        printf("  FAIL create(NULL): returned a context\n");
        failures++;
        e2_gpu_destroy(g0);
    } else {
        printf("  ok   create(NULL): %s\n", e2_gpu_last_error());
    }
    e2_gpu *g = e2_gpu_create(m);
    if (!g) {
        printf("  SKIP no Intel GPU: %s\n", e2_gpu_last_error());
        return;
    }

    int32_t ids[8] = { 2, 1234, 5678, 1 }; /* 2 sequences of 4 for the batch cases */
    const int grid = grid_positions(pos, 3);
    REJECTED_GPU("gpu NULL context (text)", 768, e2_gpu_text_embed(NULL, ids, 4, out));
    REJECTED_GPU("gpu NULL context (batch)", 2 * 768,
             e2_gpu_text_embed_batch(NULL, ids, NULL, 2, 4, out));
    REJECTED_GPU("gpu NULL context (image)", 768, e2_gpu_image_embed(NULL, pixels, pos, grid, out));
    REJECTED_GPU("gpu NULL context (text+image)", 768,
             e2_gpu_text_image_embed(NULL, ids, 4, pixels, pos, grid, out));
    REJECTED_GPU("gpu NULL context (audio)", 768, e2_gpu_audio_embed(NULL, pixels, 3200, out));
    REJECTED_GPU("gpu text NULL ids", 768, e2_gpu_text_embed(g, NULL, 4, out));
    ids[1] = -3;
    REJECTED_GPU("gpu text id -3", 768, e2_gpu_text_embed(g, ids, 4, out));
    ids[1] = 262144;
    REJECTED_GPU("gpu text id == vocab", 768, e2_gpu_text_embed(g, ids, 4, out));
    ids[1] = 1234;
    rejected_n("gpu text NULL out", e2_gpu_text_embed(g, ids, 4, NULL), e2_gpu_last_error(), 0);
    REJECTED_GPU("gpu text zero ids", 768, e2_gpu_text_embed(g, ids, 0, out));

    REJECTED_GPU("gpu batch NULL ids", 2 * 768, e2_gpu_text_embed_batch(g, NULL, NULL, 2, 4, out));
    REJECTED_GPU("gpu batch zero geometry", 2 * 768,
             e2_gpu_text_embed_batch(g, ids, NULL, 0, 4, out));
    REJECTED_GPU("gpu batch seq_len past the context", 2 * 768,
             e2_gpu_text_embed_batch(g, ids, NULL, 2, 4096, out));
    rejected_n("gpu batch NULL out", e2_gpu_text_embed_batch(g, ids, NULL, 2, 4, NULL),
               e2_gpu_last_error(), 0);
    /* A masked padding row still has to hold a valid id: its row is gathered before the mask. */
    ids[6] = -1;
    REJECTED_GPU("gpu batch invalid masked id", 2 * 768,
             e2_gpu_text_embed_batch(g, ids, (const uint8_t[]){ 1, 1, 1, 1, 1, 1, 0, 1 }, 2, 4, out));
    ids[6] = 1;

    pos[0] = 300;
    pos[1] = 300;
    REJECTED_GPU("gpu image (300,300)", 768, e2_gpu_image_embed(g, pixels, pos, 1, out));
    pos[0] = POS_TABLE;
    pos[1] = 0;
    REJECTED_GPU("gpu image position past the table", 768, e2_gpu_image_embed(g, pixels, pos, 1, out));
    pos[0] = 0;
    pos[1] = 0;
    REJECTED_GPU("gpu image NULL pixels", 768, e2_gpu_image_embed(g, NULL, pos, 1, out));
    REJECTED_GPU("gpu image NULL positions", 768, e2_gpu_image_embed(g, pixels, NULL, 1, out));
    rejected_n("gpu image NULL out", e2_gpu_image_embed(g, pixels, pos, grid, NULL),
               e2_gpu_last_error(), 0);

    REJECTED_GPU("gpu text+image NULL ids", 768,
             e2_gpu_text_image_embed(g, NULL, 3, pixels, pos, grid, out));
    REJECTED_GPU("gpu text+image bad id", 768,
             e2_gpu_text_image_embed(g, (int32_t[]){ 2, 262144, 1 }, 3, pixels, pos, grid, out));
    rejected_n("gpu text+image NULL out",
               e2_gpu_text_image_embed(g, (int32_t[]){ 2, 258880, 1 }, 3, pixels, pos, grid, NULL),
               e2_gpu_last_error(), 0);

    REJECTED_GPU("gpu audio NULL pcm", 768, e2_gpu_audio_soft(g, NULL, 3200, out, NULL));
    REJECTED_GPU("gpu audio NULL soft", 768, e2_gpu_audio_soft(g, pixels, 3200, NULL, NULL));
    rejected_n("gpu audio NULL out", e2_gpu_audio_embed(g, pixels, 3200, NULL), e2_gpu_last_error(), 0);
    REJECTED_GPU("gpu audio zero samples", 768, e2_gpu_audio_embed(g, pixels, 0, out));

    /* The only device work in this file: a valid sequence and a valid batch still return vectors. */
    printf("Intel GPU valid input still runs\n");
    out_fill(2 * 768);
    accepted("gpu text 4 ids", e2_gpu_text_embed(g, ids, 4, out), e2_gpu_last_error());
    out_fill(2 * 768);
    if (e2_gpu_text_embed_batch(g, ids, (const uint8_t[]){ 1, 1, 1, 1, 1, 1, 0, 1 }, 2, 4, out)) {
        printf("  FAIL gpu batch 2x4: %s\n", e2_gpu_last_error());
        failures++;
    } else {
        int ok = 1;
        for (int s = 0; s < 2; s++) {
            double n = 0;
            for (int d = 0; d < 768; d++) n += (double)out[s * 768 + d] * out[s * 768 + d];
            if (fabs(sqrt(n) - 1.0) > 1e-3) ok = 0;
        }
        printf("  %s gpu batch 2x4\n", ok ? "ok  " : "FAIL");
        if (!ok) failures++;
    }
    /* Longer than one pass holds (per = E2_GPU_MAX_ROWS / seq_len = 2), so this walks the chunk
     * loop twice; b0 advances by the chunk's n. */
    int32_t *long_ids = malloc(sizeof(int32_t) * 3 * 1024);
    if (!long_ids) {
        printf("  FAIL gpu batch 3x1024: oom\n");
        failures++;
    } else {
        for (int i = 0; i < 3 * 1024; i++) long_ids[i] = 1234;
        float *big = calloc(3 * 768, sizeof(float));
        if (!big || e2_gpu_text_embed_batch(g, long_ids, NULL, 3, 1024, big)) {
            printf("  FAIL gpu batch 3x1024: %s\n", e2_gpu_last_error());
            failures++;
        } else {
            printf("  ok   gpu batch 3x1024 (two chunks)\n");
        }
        free(big);
        free(long_ids);
    }
    e2_gpu_destroy(g);
}
#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_dir>\n", argv[0]);
        return 2;
    }
    if (e2_image_max_patches() != N_PATCHES_MAX) {
        fprintf(stderr, "unexpected e2_image_max_patches() = %d\n", e2_image_max_patches());
        return 2;
    }
    e2_model *m = e2_load(argv[1]);
    if (!m) {
        fprintf(stderr, "load failed: %s\n", e2_last_error());
        return 2;
    }

    /* A full padded image layout does not belong on the stack. */
    float *pixels = malloc(sizeof(float) * N_PATCHES_MAX * PATCH_DIM);
    int32_t *pos = malloc(sizeof(int32_t) * 2 * N_PATCHES_MAX);
    if (!pixels || !pos) {
        fprintf(stderr, "oom\n");
        return 2;
    }
    for (int i = 0; i < N_PATCHES_MAX * PATCH_DIM; i++) pixels[i] = 0.5f;

    cpu_sections(m, pixels, pos);
#ifdef E2_TEST_GPU
    gpu_section(m, pixels, pos);
#endif

    e2_free(m);
    free(pixels);
    free(pos);
    printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
