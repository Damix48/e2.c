/* CPU batch correctness only: compile without finite-only/fast-math assumptions.
 * Usage: e2_test_text_batch <model_dir> <fixture_dir> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

int e2_text_embed_embeds(e2_model *, const float *, int, float *);
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, e2_last_error()); exit(1); } } while (0)

static void compare(const float *got, const float *ref, double bar) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < 768; i++) {
        CHECK(isfinite(got[i]) && isfinite(ref[i]));
        dot += (double)got[i] * ref[i];
        na += (double)got[i] * got[i];
        nb += (double)ref[i] * ref[i];
    }
    CHECK(fabs(na - 1) < 1e-4);
    double cos = dot / sqrt(na * nb);
    printf("  cosine %.9f\n", cos);
    CHECK(cos >= bar);
}

static void check_batch(e2_model *m, int B, int S, const int *lengths, int masked) {
    int32_t *ids = malloc(sizeof(*ids) * (size_t)B * S);
    uint8_t *valid = malloc((size_t)B * S);
    float *out = malloc(sizeof(float) * (size_t)B * 768);
    CHECK(ids && valid && out);
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < S; i++) {
            ids[b * S + i] = 100 + (b * 101 + i * 17) % 30000;
            valid[b * S + i] = i < lengths[b] ? 3 : 0;
        }
    }
    CHECK(!e2_text_embed_batch(m, ids, masked ? valid : NULL, B, S, out));
    printf("batch %d x %d masked=%d\n", B, S, masked);
    for (int b = 0; b < B; b++) {
        float single[768];
        if (!lengths[b]) {
            for (int d = 0; d < 768; d++) CHECK(out[b * 768 + d] == 0);
        } else {
            CHECK(!e2_text_embed(m, ids + b * S, lengths[b], single));
            compare(out + b * 768, single, .99999);
        }
    }
    /* A hole keeps its original position: compare a masked batch member with batch=1,
     * not a compacted single. Padding IDs must never affect a valid output. */
    if (B == 4 && S == 64) {
        valid[7] = 0;
        CHECK(!e2_text_embed_batch(m, ids, valid, B, S, out));
        float single[768];
        CHECK(!e2_text_embed_batch(m, ids, valid, 1, S, single));
        compare(out, single, .99999);
        for (int i = 0; i < B * S; i++) if (!valid[i]) ids[i] = 9999;
        CHECK(!e2_text_embed_batch(m, ids, valid, B, S, out));
        compare(out, single, .99999);
    }
    free(ids); free(valid); free(out);
}

static void check_fixture(e2_model *m, const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    CHECK(f);
    int32_t S, D;
    CHECK(fread(&S, 4, 1, f) == 1 && fread(&D, 4, 1, f) == 1);
    CHECK(S > 0 && S <= 2048 && D == 768);
    int32_t *ids = malloc(sizeof(*ids) * (size_t)S);
    CHECK(ids);
    float ref[768], single[768], batch[768];
    CHECK(fread(ids, 4, (size_t)S, f) == (size_t)S && fread(ref, 4, 768, f) == 768);
    fclose(f);
    CHECK(!e2_text_embed(m, ids, S, single));
    CHECK(!e2_text_embed_batch(m, ids, NULL, 1, S, batch));
    printf("fixture %s\n", name);
    compare(single, ref, .999);
    compare(batch, single, .99999);
    float *embeds = malloc(sizeof(float) * (size_t)S * m->hidden);
    CHECK(embeds);
    for (int i = 0; i < S; i++) e2_embed_token(m, ids[i], embeds + (size_t)i * m->hidden);
    CHECK(!e2_text_embed_embeds(m, embeds, S, batch));
    compare(batch, single, .99999);
    /* Exercise the diagnostic row-wise reference without producing dump artifacts. */
    CHECK(!setenv("E2_DUMP", "/dev/null/e2-text-test", 1));
    CHECK(!e2_text_embed(m, ids, S, batch));
    CHECK(!unsetenv("E2_DUMP"));
    compare(batch, single, .999);
    free(embeds); free(ids);
}

static void check_masked_reference(e2_model *m, const char *dir) {
    char path[1024];
    CHECK(snprintf(path, sizeof path, "%s/text_masked.bin", dir) < (int)sizeof path);
    FILE *f = fopen(path, "rb");
    CHECK(f);
    int32_t n, dim, ids[3 * 2048];
    uint8_t mask[3 * 2048];
    float ref[768], one[768], batch[3 * 768];
    CHECK(fread(&n, 4, 1, f) == 1 && fread(&dim, 4, 1, f) == 1);
    CHECK(n > 0 && n <= 2048 && dim == 768);
    CHECK(fread(ids, 4, (size_t)n, f) == (size_t)n);
    CHECK(fread(mask, 1, (size_t)n, f) == (size_t)n && fread(ref, 4, 768, f) == 768);
    fclose(f);
    for (int b = 1; b < 3; b++) {
        memcpy(ids + b * n, ids, sizeof(*ids) * (size_t)n);
        memcpy(mask + b * n, mask, (size_t)n);
    }
    CHECK(!e2_text_embed_batch(m, ids, mask, 1, n, one));
    puts("interior mask vs independent FP32 reference");
    compare(one, ref, .999);
    CHECK(!e2_text_embed_batch(m, ids, mask, 3, n, batch));
    for (int b = 0; b < 3; b++) compare(batch + b * 768, ref, .999);
}

static void check_invalid(e2_model *m) {
    int32_t ids[2] = { 1, 262144 };
    uint8_t valid[2] = { 1, 0 };
    float out[768];
    for (int i = 0; i < 768; i++) out[i] = 7;
    CHECK(e2_text_embed_batch(m, ids, valid, 1, 2, out) == -1);
    ids[1] = -1;
    CHECK(e2_text_embed_batch(m, ids, valid, 1, 2, out) == -1);
    CHECK(e2_text_embed_batch(m, ids, NULL, 0, 1, out) == -1);
    CHECK(e2_text_embed_batch(m, ids, NULL, -1, 1, out) == -1);
    CHECK(e2_text_embed_batch(m, ids, NULL, 1, 0, out) == -1);
    CHECK(e2_text_embed_batch(m, ids, NULL, 1, 2049, out) == -1);
    CHECK(e2_text_embed_batch(NULL, ids, NULL, 1, 1, out) == -1);
    CHECK(e2_text_embed_batch(m, NULL, NULL, 1, 1, out) == -1);
    CHECK(e2_text_embed_batch(m, ids, NULL, 1, 1, NULL) == -1);
    for (int i = 0; i < 768; i++) CHECK(out[i] == 7);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model_dir> <fixture_dir>\n", argv[0]); return 2; }
    e2_model *m = e2_load(argv[1]);
    CHECK(m);
    check_invalid(m);
    check_fixture(m, argv[2], "text_query");
    check_fixture(m, argv[2], "text_doc");
    check_masked_reference(m, argv[2]);
    const int tiny[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    check_batch(m, 8, 1, tiny, 0);
    const int lengths1[] = { 31, 32, 63, 64 };
    check_batch(m, 4, 64, lengths1, 1);
    const int lengths2[] = { 15, 15, 15 };
    check_batch(m, 3, 15, lengths2, 0);
    const int lengths3[] = { 15, 514, 31, 0 }; /* 2056 rows: three sequences then one */
    check_batch(m, 4, 514, lengths3, 1);
    e2_free(m);
    puts("text batch OK");
    return 0;
}
