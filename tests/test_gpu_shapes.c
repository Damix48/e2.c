/* Graph turnover and padded/chunked attention regression. Compile without finite-only math.
 * Usage: e2_test_gpu_shapes <model_dir> <fixture_dir> [--audio] (requires an audio tower). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "gpu/gpu.h"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #c, e2_gpu_last_error()); \
    exit(1); \
} } while (0)

static void compare(const float *a, const float *b, int n, double minimum) {
    double dot = 0, aa = 0, bb = 0;
    for (int i = 0; i < n; i++) {
        CHECK(isfinite(a[i]) && isfinite(b[i]));
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    CHECK(aa > 0 && bb > 0);
    const double cosine = dot / sqrt(aa * bb);
    if (cosine < minimum) fprintf(stderr, "cosine %.9f (minimum %.6f)\n", cosine, minimum);
    CHECK(cosine >= minimum);
}

static long kernels(e2_gpu *g) {
    long k = 0;
    CHECK(!e2_gpu_stats(g, NULL, &k, NULL));
    CHECK(k > 0);
    return k;
}

static void cache_turnover(e2_gpu *g) {
    int32_t ids[64];
    for (int i = 0; i < 64; i++) ids[i] = 100 + i;
    float first[768], again[768], other[768];
    CHECK(!e2_gpu_text_embed(g, ids, 1, first));
    const long k1 = kernels(g);
    CHECK(!e2_gpu_text_embed(g, ids, 64, other));
    const long k64 = kernels(g);
    CHECK(k1 != k64); /* different projection/attention paths */
    CHECK(!e2_gpu_text_embed(g, ids, 1, again));
    CHECK(kernels(g) == k1);
    puts("text cache revisit");
    compare(first, again, 768, 0.999999);

    float pixels[9 * E2_V_PATCH] = {0}, image[768], image_again[768];
    int32_t pos[18];
    for (int i = 0; i < 9; i++) { pos[2*i] = i / 3; pos[2*i+1] = i % 3; }
    long vk = 0;
    if (g->m->vision) {
        CHECK(!e2_gpu_image_embed(g, pixels, pos, 9, image));
        CHECK(!e2_gpu_vision_stats(g, &vk, NULL));
        CHECK(vk > 0);
    }
    for (int n = 2; n <= 21; n++) {
        CHECK(!e2_gpu_text_embed(g, ids, n, other));
        compare(other, other, 768, 0.999999);
        CHECK(g->nslots <= 16);
    }
    CHECK(g->nslots == 16);
    CHECK(!e2_gpu_text_embed(g, ids, 1, again)); /* evicted shape is recorded again */
    CHECK(kernels(g) == k1);
    puts("text cache revisit");
    compare(first, again, 768, 0.999999);
    if (g->m->vision) {
        CHECK(!e2_gpu_image_embed(g, pixels, pos, 9, image_again));
        long hit = 0;
        CHECK(!e2_gpu_vision_stats(g, &hit, NULL));
        CHECK(hit == vk);
        puts("vision cache revisit");
        compare(image, image_again, 768, 0.999999);
        CHECK(!e2_gpu_text_embed(g, ids, 64, other));
        CHECK(kernels(g) == k64);
        CHECK(!e2_gpu_image_embed(g, pixels, pos, 9, image_again)); /* cache hit after text */
        CHECK(!e2_gpu_vision_stats(g, &hit, NULL));
        CHECK(hit == vk);
        puts("vision cache revisit");
        compare(image, image_again, 768, 0.999999);
    }
    puts("cache turnover: >16 text/vision shapes, evicted revisits and hit statistics passed");
}

static int read_ids(const char *dir, const char *name, int32_t *ids) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    CHECK(f);
    int32_t n, dim;
    CHECK(fread(&n, 4, 1, f) == 1 && fread(&dim, 4, 1, f) == 1);
    CHECK(n > 0 && n <= 512 && dim == 768);
    CHECK(fread(ids, 4, (size_t)n, f) == (size_t)n);
    CHECK(!fclose(f));
    return n;
}

static void padded_batch(e2_gpu *g, int seq, const int32_t corpus[2][512], const int lengths[2]) {
    const int per = E2_GPU_MAX_ROWS / seq, batch = per + 1;
    const size_t rows = (size_t)batch * seq;
    int32_t *ids = malloc(rows * sizeof *ids);
    uint8_t *valid = malloc(rows);
    float *out = malloc((size_t)batch * 768 * sizeof *out);
    float *ref = malloc((size_t)batch * 768 * sizeof *ref);
    CHECK(ids && valid && out && ref);
    for (int b = 0; b < batch; b++) {
        const int len = b % 3 == 0 ? seq : b % 3 == 1 ? seq - 1 : seq / 2;
        for (int i = 0; i < seq; i++) {
            ids[(size_t)b * seq + i] = corpus[b % 2][i % lengths[b % 2]];
            valid[(size_t)b * seq + i] = i < len;
        }
        CHECK(!e2_gpu_text_embed(g, ids + (size_t)b * seq, len, ref + (size_t)b * 768));
    }
    CHECK(!e2_gpu_text_embed_batch(g, ids, valid, batch, seq, out));
    for (int b = 0; b < batch; b++) compare(out + (size_t)b * 768, ref + (size_t)b * 768, 768, 0.999);
    const long recorded = kernels(g);
    /* Hit the graph after differently shaped singles, then check every output again. */
    float one[768];
    CHECK(!e2_gpu_text_embed(g, ids, seq, one));
    CHECK(!e2_gpu_text_embed_batch(g, ids, valid, batch, seq, out));
    CHECK(kernels(g) == recorded);
    for (int b = 0; b < batch; b++) compare(out + (size_t)b * 768, ref + (size_t)b * 768, 768, 0.999);
    printf("mixed padded batch %d x %d (chunk %d), every row vs single: passed\n", batch, seq, per);
    free(ids); free(valid); free(out); free(ref);
}

/* Arbitrary token IDs and noncontiguous masks: compare equal padded shapes so the scalar/INT8
 * attention threshold and its existing quantization error do not change the reference recipe. */
static void masked_batch(e2_gpu *g) {
    enum { S = 196, B = 5 };
    int32_t ids[B * S];
    uint8_t valid[B * S];
    float ref[B * 768], out[B * 768];
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < S; i++) {
            ids[b * S + i] = 100 + ((i * 17 + b * 31) % 1000);
            valid[b * S + i] = i < S - b && i % 4 != b % 4;
        }
        CHECK(!e2_gpu_text_embed_batch(g, ids + b * S, valid + b * S, 1, S, ref + b * 768));
    }
    CHECK(!e2_gpu_text_embed_batch(g, ids, valid, B, S, out));
    for (int b = 0; b < B; b++) compare(out + b * 768, ref + b * 768, 768, 0.999999);
    puts("noncontiguous masks, every row vs same-shape single: passed");
}

static void audio_turnover(e2_gpu *g) {
    const int samples = 3200;
    float pcm[samples];
    for (int i = 0; i < samples; i++) pcm[i] = 0.2f * sinf(2.f * 3.14159265f * 220.f * i / 16000.f);
    float *first = malloc((size_t)E2_A_MAX_TOK * 512 * sizeof(float));
    float *again = malloc((size_t)E2_A_MAX_TOK * 512 * sizeof(float));
    CHECK(first && again);
    int n = 0, nn = 0;
    CHECK(!e2_gpu_audio_soft(g, pcm, samples, first, &n));
    long ak = 0, hit = 0;
    CHECK(!e2_gpu_audio_stats(g, &ak, NULL));
    int32_t ids[22];
    for (int i = 0; i < 22; i++) ids[i] = 100 + i;
    float tmp[768];
    for (int i = 1; i <= 22; i++) CHECK(!e2_gpu_text_embed(g, ids, i, tmp));
    CHECK(!e2_gpu_audio_soft(g, pcm, samples, again, &nn));
    CHECK(n == nn && n > 0);
    CHECK(!e2_gpu_audio_stats(g, &hit, NULL));
    CHECK(hit == ak);
    compare(first, again, n * 512, 0.999999);
    CHECK(!e2_gpu_text_embed(g, ids, 22, tmp));
    CHECK(!e2_gpu_audio_soft(g, pcm, samples, again, &nn));
    CHECK(!e2_gpu_audio_stats(g, &hit, NULL));
    CHECK(hit == ak);
    compare(first, again, n * 512, 0.999999);
    free(first); free(again);
    puts("audio eviction, revisit and hit statistics passed");
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4 || (argc == 4 && strcmp(argv[3], "--audio"))) {
        fprintf(stderr, "usage: %s <model_dir> <fixture_dir> [--audio]\n", argv[0]);
        return 2;
    }
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "load: %s\n", e2_last_error()); return 1; }
    e2_gpu *g = e2_gpu_create(m);
    CHECK(g);
    printf("device: %s\n", e2_gpu_device(g));
    cache_turnover(g);
    int32_t corpus[2][512];
    const int counts[2] = {read_ids(argv[2], "text_doc", corpus[0]),
                           read_ids(argv[2], "text_query", corpus[1])};
    const int lengths[] = {63, 64, 196, 514, 1025, 2048};
    for (size_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++)
        padded_batch(g, lengths[i], corpus, counts);
    masked_batch(g);
    if (argc == 4) audio_turnover(g);
    e2_gpu_destroy(g);
    e2_free(m);
    puts("ok");
    return 0;
}
