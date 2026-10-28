/* Fused W8A8 gate/up -> GELU/product -> per-128 quantization vs a scalar oracle.
 * No checkpoint: exercise tail tiles, distinct scales, zero rows, both clamps and debug stores. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu/e2_tile.h"
#include "gpu/l0.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, e2_l0_error(g)); exit(1); \
} } while (0)
enum { N = 2048, K = 512, DS = 16 };
static uint32_t g_seed = 123;
static uint32_t random_u32(void) { return g_seed = g_seed * 1664525u + 1013904223u; }
static size_t aligned(size_t n) { return (n + 63) & ~(size_t)63; }

static void run(e2_l0 *g, int M, int wr, int clamps) {
    const int R = (M + 15) / 16 * 16;
    const size_t wspan = aligned((size_t)N * K + N * 4 + 4 * N * 4);
    const size_t asoff = aligned((size_t)R * K);
    const size_t yoff = aligned(asoff + (size_t)R * 4 * 4);
    const size_t uoff = yoff + (size_t)R * N * 4;
    const size_t qoff = uoff + (size_t)R * N * 4;
    const size_t soff = qoff + (size_t)R * N;
    const size_t bytes = soff + (size_t)R * 16 * 4;
    uint8_t *wb = calloc(2, wspan), *acts = malloc(bytes);
    int8_t *w = malloc((size_t)2 * N * K);
    float *ref = malloc((size_t)2 * M * N * 4);
    int32_t desc[3 * DS] = {0};
    CHECK(wb && acts && w && ref);
    memset(acts, 0xa5, bytes);
    float *as = (float *)(acts + asoff);
    for (int m = 0; m < R; m++) {
        for (int k = 0; k < K; k++) acts[(size_t)m * K + k] = m && m < M ? 1 + random_u32() % 255 : 128;
        for (int b = 0; b < 4; b++) as[m * 4 + b] = .004f * (b + 1) + .001f * (m % 3);
    }
    for (int p = 0; p < 2; p++) {
        int32_t *dd = desc + p * DS;
        dd[1] = (int32_t)(p * wspan + (size_t)N * K);
        dd[2] = dd[1] + N * 4;
        dd[3] = (int32_t)(p ? uoff : yoff);
        dd[4] = N; dd[5] = K; dd[6] = 4;
        dd[7] = 0; dd[8] = (int32_t)asoff; dd[9] = (int32_t)(p * wspan);
        dd[10] = !!(clamps & (1 << p));
        const float lo = p ? -.4f : -.75f, hi = p ? .6f : .5f;
        memcpy(dd + 11, &lo, 4); memcpy(dd + 12, &hi, 4);
        float *ws = (float *)(wb + dd[1]);
        int32_t *bs = (int32_t *)(wb + dd[2]);
        for (int n = 0; n < N; n++) {
            ws[n] = .03f + .01f * (n % 6);
            for (int k = 0; k < K; k++) {
                const int value = (int)(random_u32() % 15) - 7;
                w[((size_t)p * N + n) * K + k] = (int8_t)value;
                bs[(k / 128) * N + n] += value;
            }
            for (int kq = 0; kq < K / 4; kq++)
                memcpy(wb + p * wspan + E2_DPAS_OFF(kq, n / 8, n % 8, 0, N / 8),
                       w + ((size_t)p * N + n) * K + kq * 4, 4);
        }
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            float sum = 0.f;
            for (int b = 0; b < 4; b++) {
                int dot = 0;
                for (int k = b * 128; k < (b + 1) * 128; k++)
                    dot += ((int)acts[(size_t)m * K + k] - 128) * w[((size_t)p * N + n) * K + k];
                sum += as[m * 4 + b] * ws[n] * dot;
            }
            if (dd[10]) sum = fminf(hi, fmaxf(lo, sum));
            ref[((size_t)p * M + m) * N + n] = sum;
        }
    }
    desc[2 * DS + 7] = (int32_t)qoff;
    desc[2 * DS + 8] = (int32_t)soff;
    void *wd = e2_l0_alloc(g, 2 * wspan), *ad = e2_l0_alloc(g, bytes);
    void *dd = e2_l0_alloc(g, sizeof desc);
    CHECK(wd && ad && dd);
    CHECK(!e2_l0_write(g, wd, 2 * wspan, wb));
    CHECK(!e2_l0_write(g, ad, bytes, acts));
    CHECK(!e2_l0_write(g, dd, sizeof desc, desc));
    e2_graph *gr = e2_l0_graph_begin(g, 1);
    CHECK(gr);
    CHECK(!e2_l0_append(g, gr, "k_gemm_dpas_large_gate", (uint32_t)(R / 16), N / 128, 1,
        8, 8, 1, 5, &wd, sizeof wd, &ad, sizeof ad, &dd, sizeof dd, &M, sizeof M, &wr, sizeof wr));
    CHECK(!e2_l0_graph_close(g, gr));
    /* Replay too: the kernel must not depend on the previous activation/SLM contents. */
    for (int rep = 0; rep < 2; rep++) {
        CHECK(!e2_l0_graph_run(g, gr));
        CHECK(!e2_l0_read(g, ad, bytes, acts));
    }
    float maximum = 0.f;
    for (int m = 0; m < M; m++) for (int b = 0; b < 16; b++) {
        float values[128], mx = 0.f;
        for (int i = 0; i < 128; i++) {
            const int n = b * 128 + i;
            const float gate = ref[(size_t)m * N + n], up = ref[((size_t)M + m) * N + n];
            values[i] = .5f * gate * (1.f + tanhf(.7978845608f * (gate + .044715f * gate * gate * gate))) * up;
            mx = fmaxf(mx, fabsf(values[i]));
            if (wr) {
                const float got = ((float *)(acts + yoff))[(size_t)m * N + n];
                const float error = fabsf(got - values[i]) / (1.f + fabsf(values[i]));
                maximum = fmaxf(maximum, error);
                CHECK(isfinite(got) && error < 1e-4f);
                CHECK(fabsf(((float *)(acts + uoff))[(size_t)m * N + n] - up) < 1e-4f * (1.f + fabsf(up)));
            }
        }
        const float sc = mx > 0.f ? mx / 127.f : 1.f;
        const float got_sc = ((float *)(acts + soff))[m * 16 + b];
        CHECK(isfinite(got_sc) && got_sc > 0 && fabsf(got_sc - sc) <= 1e-4f * sc);
        for (int i = 0; i < 128; i++) {
            const float scaled = values[i] / sc;
            int expected = (int)lrintf(scaled);
            if (expected > 127) expected = 127;
            if (expected < -127) expected = -127;
            const int delta = abs((int)acts[qoff + (size_t)m * N + b * 128 + i] - (expected + 128));
            CHECK(delta == 0 || (delta == 1 && fabsf(scaled - (floorf(scaled) + .5f)) < .002f));
        }
    }
    for (size_t i = qoff + (size_t)M * N; i < soff; i++) CHECK(acts[i] == 0xa5);
    if (!wr) for (size_t i = yoff; i < qoff; i++) CHECK(acts[i] == 0xa5);
    printf("fused FFN M=%d dump=%d clamps=%d: scalar match, max relative dump error %.6g\n",
           M, wr, clamps, maximum);
    e2_l0_graph_free(g, gr);
    e2_l0_free(g, wd); e2_l0_free(g, ad); e2_l0_free(g, dd);
    free(wb); free(acts); free(w); free(ref);
}

int main(void) {
    char error[2048];
    e2_l0 *g = e2_l0_create(error, sizeof error);
    if (!g) {
        fprintf(stderr, "%s\n", error);
        return !strcmp(error, "no Level Zero driver") || !strcmp(error, "no Intel GPU device found") ? 77 : 1;
    }
    run(g, 1, 0, 0); run(g, 15, 1, 1); run(g, 17, 1, 3); run(g, 32, 0, 0);
    e2_l0_destroy(g);
    return 0;
}
