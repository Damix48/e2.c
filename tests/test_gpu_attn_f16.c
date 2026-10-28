/* FP16 XMX attention vs an independent FP64 attention oracle, without model weights.
 * Check padded/odd sequences, both GQA layouts, holes/all-masked rows and the 2048-token window.
 * Sample query rows at tile/window boundaries; every key and context dimension is checked. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu/l0.h"
#include "gpu/e2_tile.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, e2_l0_error(g)); \
    exit(1); } } while (0)

static float random_value(uint32_t *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (float)(*seed >> 8) / 8388608.f - 1.f;
}

static float probability(uint16_t h) {
    const int exponent = (h >> 10) & 31;
    assert(!(h & 0x8000) && exponent != 31);
    return exponent ? ldexpf(1024.f + (h & 1023), exponent - 25)
                    : ldexpf((float)(h & 1023), -24);
}

static void run(e2_l0 *g, int S, int B, int D, int KV, int window, int mask) {
    const int H = 4, rows = B * S, SP = (S + 15) / 16 * 16;
    const size_t nq = (size_t)rows * H * D, nk = (size_t)rows * KV * D;
    const size_t ns = (size_t)H * rows * SP, np = (size_t)B * KV * SP * D;
    float *q = malloc(nq * 4), *k = malloc(nk * 4), *v = malloc(nk * 4);
    float *scores = malloc(ns * 4), *out = malloc(nq * 4);
    uint16_t *prob = malloc(ns * 2);
    uint8_t *valid = malloc((size_t)rows);
    double *ref = malloc((size_t)SP * sizeof *ref);
    CHECK(q && k && v && scores && out && prob && valid && ref);
    uint32_t seed = 123;
    for (size_t i = 0; i < nq; i++) q[i] = random_value(&seed) * (mask ? 1.f : .25f);
    for (size_t i = 0; i < nk; i++) {
        k[i] = random_value(&seed);
        v[i] = random_value(&seed);
    }
    for (int b = 0; b < B; b++)
        for (int j = 0; j < S; j++)
            valid[b * S + j] = !mask || (b != 2 && j % 7 != b && j < S - b);
    void *qd = e2_l0_alloc(g, nq * 4), *kd = e2_l0_alloc(g, nk * 4);
    void *vd = e2_l0_alloc(g, nk * 4), *sd = e2_l0_alloc(g, ns * 4);
    void *pd = e2_l0_alloc(g, ns * 2), *od = e2_l0_alloc(g, nq * 4);
    void *kp = e2_l0_alloc(g, np * 2), *vp = e2_l0_alloc(g, np * 2);
    void *md = e2_l0_alloc(g, (size_t)rows);
    void *qad = e2_l0_alloc(g, nq), *scd = e2_l0_alloc(g, nq / 128 * 4);
    void *qar = e2_l0_alloc(g, nq), *scr = e2_l0_alloc(g, nq / 128 * 4);
    CHECK(qd && kd && vd && sd && pd && od && kp && vp && md && qad && scd && qar && scr);
    CHECK(!e2_l0_write(g, qd, nq * 4, q));
    CHECK(!e2_l0_write(g, kd, nk * 4, k));
    CHECK(!e2_l0_write(g, vd, nk * 4, v));
    CHECK(!e2_l0_write(g, md, (size_t)rows, valid));
    void *mask_d = mask ? md : NULL;
    e2_graph *gr = e2_l0_graph_begin(g, 1);
    CHECK(gr);
    CHECK(!e2_l0_append(g, gr, "k_attn_f16_pack", (uint32_t)(np / 2 / 128), 1, 1,
        128, 1, 1, 8, &kd, sizeof kd, &vd, sizeof vd, &kp, sizeof kp, &vp, sizeof vp,
        &S, sizeof S, &SP, sizeof SP, &D, sizeof D, &KV, sizeof KV));
    for (int context = 0; context <= 1; context++) {
        const int N = context ? D : SP;
        if (context) {
            const int dump = 1;
            CHECK(!e2_l0_append(g, gr, "k_attn_f16_ctx", (uint32_t)((S + 7) / 8), D / 128,
                (uint32_t)(B * H), 8, 8, 1, 11, &vp, sizeof vp, &pd, sizeof pd, &od, sizeof od,
                &S, sizeof S, &SP, sizeof SP, &H, sizeof H, &D, sizeof D, &KV, sizeof KV,
                &qad, sizeof qad, &scd, sizeof scd, &dump, sizeof dump));
        } else {
            CHECK(!e2_l0_append(g, gr, "k_attn_f16", (uint32_t)((S + 7) / 8),
                (uint32_t)((N + 63) / 64), (uint32_t)(B * H), 8, 4, 1, 12,
                &qd, sizeof qd, &kp, sizeof kp, &vp, sizeof vp, &sd, sizeof sd,
                &pd, sizeof pd, &od, sizeof od, &S, sizeof S, &SP, sizeof SP,
                &H, sizeof H, &D, sizeof D, &KV, sizeof KV, &context, sizeof context));
            CHECK(!e2_l0_append(g, gr, "k_attn_f16_softmax", (uint32_t)((rows + 3) / 4),
                H, 1, 32, 4, 1, 7, &sd, sizeof sd, &pd, sizeof pd, &mask_d, sizeof mask_d,
                &rows, sizeof rows, &S, sizeof S, &SP, sizeof SP, &window, sizeof window));
        }
    }
    /* Independent quantizer: the fused path must not introduce a context normalization. */
    void *none = NULL;
    const int width = H * D, nb = width / 128, flags = E2_ROW_QUANT_ONLY;
    const float eps = 1e-6f, one = 1.f, zero = 0.f;
    CHECK(!e2_l0_append(g, gr, "k_rowops", (uint32_t)((rows + 3) / 4), 1, 1, 32, 4, 1,
        17, &od, sizeof od, &none, sizeof none, &none, sizeof none,
        &qar, sizeof qar, &scr, sizeof scr, &none, sizeof none, &none, sizeof none,
        &eps, sizeof eps, &one, sizeof one, &one, sizeof one, &rows, sizeof rows,
        &width, sizeof width, &nb, sizeof nb, &flags, sizeof flags, &zero, sizeof zero,
        &zero, sizeof zero, &mask_d, sizeof mask_d));
    CHECK(!e2_l0_graph_close(g, gr));
    CHECK(!e2_l0_graph_run(g, gr));
    CHECK(!e2_l0_read(g, sd, ns * 4, scores));
    CHECK(!e2_l0_read(g, pd, ns * 2, prob));
    CHECK(!e2_l0_read(g, od, nq * 4, out));
    uint8_t *quant = malloc(nq * 2);
    float *scales = malloc(nq / 128 * 4 * 2);
    CHECK(quant && scales);
    CHECK(!e2_l0_read(g, qad, nq, quant));
    CHECK(!e2_l0_read(g, qar, nq, quant + nq));
    CHECK(!e2_l0_read(g, scd, nq / 128 * 4, scales));
    CHECK(!e2_l0_read(g, scr, nq / 128 * 4, scales + nq / 128));
    for (size_t i = 0; i < nq; i++) {
        CHECK(isfinite(out[i]));
        const int active = !mask || valid[i / (H * D)];
        if (active) CHECK(abs((int)quant[i] - quant[nq + i]) <= 1);
        else CHECK(quant[i] == 128);
    }
    for (size_t i = 0; i < nq / 128; i++) {
        CHECK(isfinite(scales[i]) && scales[i] > 0.f);
        if (!mask || valid[i / (H * D / 128)])
            CHECK(fabsf(scales[i] - scales[nq / 128 + i]) < 1e-6f * scales[i]);
        else CHECK(scales[i] == 1.f);
    }
    free(quant); free(scales);
    double max_score = 0, max_prob = 0, max_ctx = 0, err2 = 0, ref2 = 0;
    const int queries[] = { 0, S > 7 ? 7 : S - 1, S > 8 ? 8 : S - 1, S / 2, S - 1 };
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++)
        for (size_t a = 0; a < sizeof queries / sizeof queries[0]; a++) {
            const int m = queries[a], row = b * S + m, kv = h * KV / H;
            const size_t off = ((size_t)h * rows + row) * SP;
            double mx = -1e300, sum = 0;
            for (int j = 0; j < SP; j++) {
                double dot = 0;
                if (j < S) for (int d = 0; d < D; d++)
                    dot += (double)q[(size_t)row * H * D + h * D + d] *
                                  k[((size_t)b * S + j) * KV * D + kv * D + d];
                CHECK(isfinite(scores[off + j]));
                max_score = fmax(max_score, fabs(scores[off + j] - dot));
                const int live = j < S && valid[row] && valid[b * S + j] &&
                                 (!window || abs(j - m) <= window);
                ref[j] = live ? dot : -1e300;
                mx = fmax(mx, ref[j]);
            }
            for (int j = 0; j < SP; j++) {
                ref[j] = ref[j] > -1e299 ? exp(ref[j] - mx) : 0;
                sum += ref[j];
            }
            for (int j = 0; j < SP; j++) {
                ref[j] = sum > 0 ? ref[j] / sum : 0;
                const float p = probability(prob[off + j]);
                if (!ref[j]) CHECK(p == 0.f);
                max_prob = fmax(max_prob, fabs(p - ref[j]));
            }
            for (int d = 0; d < D; d++) {
                double expected = 0;
                for (int j = 0; j < S; j++)
                    expected += ref[j] * v[((size_t)b * S + j) * KV * D + kv * D + d];
                const double got = out[(size_t)row * H * D + h * D + d];
                max_ctx = fmax(max_ctx, fabs(got - expected));
                err2 += (got - expected) * (got - expected);
                ref2 += expected * expected;
            }
        }
    const double relative = ref2 > 0 ? sqrt(err2 / ref2) : sqrt(err2);
    printf("FP16 attention B=%d S=%d D=%d KV=%d window=%d mask=%d: "
           "max score/prob/ctx error %.6g/%.6g/%.6g, relative L2 %.6g\n",
           B, S, D, KV, window, mask, max_score, max_prob, max_ctx, relative);
    CHECK(max_score < .01 && max_prob < .002 && max_ctx < .003 && relative < .003);
    e2_l0_graph_free(g, gr);
    void *allocations[] = { qd, kd, vd, sd, pd, od, kp, vp, md, qad, scd, qar, scr };
    for (size_t i = 0; i < sizeof allocations / sizeof allocations[0]; i++)
        e2_l0_free(g, allocations[i]);
    free(q); free(k); free(v); free(scores); free(out); free(prob); free(valid); free(ref);
}

int main(void) {
    char error[2048];
    e2_l0 *g = e2_l0_create(error, sizeof error);
    if (!g) {
        fprintf(stderr, "%s\n", error);
        return !strcmp(error, "no Level Zero driver") ||
               !strcmp(error, "no Intel GPU device found") ? 77 : 1;
    }
    run(g, 1, 1, 256, 2, 0, 0);
    run(g, 1, 1, 512, 1, 0, 0);
    run(g, 1, 3, 256, 2, 0, 1); /* all queries masked */
    run(g, 2, 3, 512, 1, 0, 1);
    run(g, 7, 3, 256, 2, 0, 1);
    run(g, 8, 2, 512, 1, 0, 1);
    run(g, 15, 1, 256, 2, 0, 0);
    run(g, 15, 3, 512, 1, 0, 1);
    run(g, 63, 1, 256, 2, 0, 0);
    run(g, 64, 1, 256, 2, 0, 0);
    run(g, 65, 3, 256, 2, 0, 1);
    run(g, 129, 2, 512, 1, 7, 1);
    run(g, 2048, 1, 256, 2, 1024, 1);
    e2_l0_destroy(g);
    return 0;
}
