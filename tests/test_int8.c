/* test_int8.c — unit test for the int8 scheme: quantize + GEMM vs a float reference.
 *
 *   ./e2_test_int8            (self-contained, no model needed)
 * Checks the whole chain (weight quantization, per-block activation quantization, VNNI/AVX2 kernel)
 * on random data with an outlier-heavy distribution, which is the case that per-row scales handled
 * badly and per-block scales are supposed to fix.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "int8.h"
#include "kernels.h"

void e2_i8_gemm_scalar(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
void e2_i8_gemm_vnni(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
void e2_i8_gemm_avx2(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
void e2_i8_gemm_avx512(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);

static int has_avx512vnni(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vnni");
#else
    return 0;
#endif
}

static int has_avxvnni(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avxvnni");
#else
    return 0;
#endif
}

static double cos_vs(const float *y, const float *x, const float *w, int M, int N, int K) {
    double dot = 0, na = 0, nb = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float r = 0.f;
            for (int k = 0; k < K; k++) r += x[(size_t)m * K + k] * w[(size_t)n * K + k];
            double v = y[(size_t)m * N + n];
            dot += v * r; na += v * v; nb += (double)r * r;
        }
    return dot / (sqrt(na) * sqrt(nb));
}

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static int check(int M, int N, int K, int outliers) {
    float *x = malloc(sizeof(float) * (size_t)M * K);
    float *w = malloc(sizeof(float) * (size_t)N * K);
    float *y = malloc(sizeof(float) * (size_t)M * N);
    float *ref = malloc(sizeof(float) * (size_t)M * N);
    srand(1234 + M + K);
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = frand();
    for (size_t i = 0; i < (size_t)N * K; i++) w[i] = frand() * 0.1f;
    /* a few large values per row: forces a wide dynamic range inside each activation row */
    for (int m = 0; m < M; m++)
        for (int o = 0; o < outliers && o < K; o++) x[(size_t)m * K + (o * 37) % K] *= 30.0f;

    int8_t *qa = malloc((size_t)M * K);
    float *ascales = calloc((size_t)M * ((K + E2_I8_BLOCK - 1) / E2_I8_BLOCK) + 8, sizeof(float));
    e2_i8_quantize(qa, ascales, x, M, K, -INFINITY, INFINITY);

    e2_q8w qw;
    if (e2_q8w_from_f32(&qw, w, N, K)) { fprintf(stderr, "quantize failed\n"); return 1; }
    /* isolate: activation quantization alone, weight quantization alone, then the kernels */
    {
        double dot = 0, na = 0, nb = 0;
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++) {
                float dq = ascales[(size_t)m * ((K + E2_I8_BLOCK - 1) / E2_I8_BLOCK) + k / E2_I8_BLOCK] *
                            (float)((int)(uint8_t)qa[(size_t)m * K + k] - 128);
                dot += (double)dq * x[(size_t)m * K + k];
                na += (double)dq * dq;
                nb += (double)x[(size_t)m * K + k] * x[(size_t)m * K + k];
            }
        printf("    act-quant cos=%.6f   ", dot / (sqrt(na) * sqrt(nb)));
        dot = na = nb = 0;
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) {
                /* [N/8][K/4][8][4] quad-interleaved layout (int8.h) */
                double dq = (double)qw.scale[n] *
                            (double)qw.w[(size_t)(n >> 3) * ((K + 3) / 4) * 32 +
                                        (size_t)(k >> 2) * 32 + (n & 7) * 4 + (k & 3)];
                dot += dq * w[(size_t)n * K + k];
                na += dq * dq;
                nb += (double)w[(size_t)n * K + k] * w[(size_t)n * K + k];
            }
        printf("w-quant cos=%.6f\n    ", dot / (sqrt(na) * sqrt(nb)));
    }
    {
        float *ys = malloc(sizeof(float) * (size_t)M * N);
        e2_i8_gemm_scalar(ys, qa, ascales, &qw, M, N, 0, N);
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < (size_t)M * N; i++) {
            float r = 0.f;
            int m = (int)(i / N), n = (int)(i % N);
            for (int k = 0; k < K; k++) r += x[(size_t)m * K + k] * w[(size_t)n * K + k];
            dot += (double)ys[i] * r; na += (double)ys[i] * ys[i]; nb += (double)r * r;
        }
        printf("scalar-gemm cos=%.6f   ", dot / (sqrt(na) * sqrt(nb)));
        free(ys);
    }
    int kernel_bad = 0;
    {
        float *yv = has_avxvnni() ? malloc(sizeof(float) * (size_t)M * N) : NULL;
        float *ya = malloc(sizeof(float) * (size_t)M * N);
        float *y5 = has_avx512vnni() ? malloc(sizeof(float) * (size_t)M * N) : NULL;
        if (yv) e2_i8_gemm_vnni(yv, qa, ascales, &qw, M, N, 0, N);
        e2_i8_gemm_avx2(ya, qa, ascales, &qw, M, N, 0, N);
        if (y5) e2_i8_gemm_avx512(y5, qa, ascales, &qw, M, N, 0, N);
        double cv = yv ? cos_vs(yv, x, w, M, N, K) : 1.0;
        double ca = cos_vs(ya, x, w, M, N, K);
        if (yv) printf("vnni cos=%.6f  ", cv);
        else printf("vnni skipped (unsupported ISA)  ");
        printf("avx2 cos=%.6f", ca);
        if (y5) {
            double c5 = cos_vs(y5, x, w, M, N, K);
            printf("  avx512 cos=%.6f", c5);
            if (!(c5 > 0.9995)) kernel_bad = 1;
            free(y5);
        }
        printf("\n    ");
        /* Assert on every backend, not just the dispatched one: the dispatched host never runs the
         * others, so a layout bug in one of them would stay invisible. */
        if (!(cv > 0.9995) || !(ca > 0.9995)) kernel_bad = 1;
        free(yv); free(ya);
    }
    e2_i8_gemm(y, qa, ascales, &qw, M, N, 0, N);

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += x[(size_t)m * K + k] * w[(size_t)n * K + k];
            ref[(size_t)m * N + n] = acc;
        }
    /* Error is reported against the row norm: per-element relative error is meaningless when the
     * reference value happens to be near zero (which is the normal case for a random dot product). */
    double dot = 0, na = 0, nb = 0, maxrow = 0;
    for (int m = 0; m < M; m++) {
        double err2 = 0, norm2 = 0;
        for (int n = 0; n < N; n++) {
            size_t i = (size_t)m * N + n;
            dot += (double)y[i] * ref[i];
            na += (double)y[i] * y[i];
            nb += (double)ref[i] * ref[i];
            err2 += (double)(y[i] - ref[i]) * (y[i] - ref[i]);
            norm2 += (double)ref[i] * ref[i];
        }
        double row = sqrt(err2 / (norm2 > 0 ? norm2 : 1));
        if (row > maxrow) maxrow = row;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("M=%-4d N=%-6d K=%-5d outliers=%-3d  int8 vs float: cos=%.6f  row rel err=%.2e  %s\n",
           M, N, K, outliers, cos, maxrow, (cos > 0.9995 && !kernel_bad) ? "OK" : "BAD");
    free(x); free(w); free(y); free(ref); free(qa); free(ascales);
    e2_q8w_free(&qw);
    return cos > 0.9995 && !kernel_bad;
}

int main(void) {
    e2_cpu_init();
    printf("backend: %s, int8: %s\n", e2_cpu_name(), e2_i8_quantize ? "yes" : "no");
    if (!e2_i8_quantize) { puts("SKIP: no SIMD INT8 activation quantizer"); return 77; }
    int fails = 0;
    fails += !check(4, 16, 512, 0);
    fails += !check(4, 16, 512, 4);
    fails += !check(15, 64, 512, 8);
    fails += !check(15, 512, 512, 8);
    fails += !check(196, 128, 2048, 16);
    fails += !check(7, 33, 512, 3); /* odd M, odd N */
    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
