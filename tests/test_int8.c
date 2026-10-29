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
#if defined(__aarch64__)
void e2_i8_gemm_neon(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
#endif

/* Only the x86 kernels cross-check against themselves; off x86 the dispatcher never picks them. */
#if defined(__x86_64__) || defined(_M_X64)
static int has_avx512vnni(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vnni");
#else
    return 0;
#endif
}

static int has_avxvnni(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avxvnni");
#else
    return 0;
#endif
}
#endif

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

    int8_t *qa_base = malloc((size_t)M * K + 1);
    int8_t *qa = qa_base + 1; /* exercise packed loads without int32 alignment */
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
    /* The AVX2/VNNI/AVX512 kernels are not compiled outside x86-64, so their symbols only exist there. */
#if defined(__x86_64__) || defined(_M_X64)
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
#else
    printf("SIMD int8 kernels skipped (x86-only)\n    ");
#endif
#if defined(__aarch64__)
    /* Same check for the NEON kernel: the dispatcher already ran it above, but only for the shape
     * the model happens to use, and a layout bug in a tail would stay invisible that way. */
    {
        float *yn = malloc(sizeof(float) * (size_t)M * N);
        e2_i8_gemm_neon(yn, qa, ascales, &qw, M, N, 0, N);
        const double cn = cos_vs(yn, x, w, M, N, K);
        printf("neon cos=%.6f\n    ", cn);
        if (!(cn > 0.9995)) kernel_bad = 1;
        free(yn);
    }
#endif
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
    free(x); free(w); free(y); free(ref); free(qa_base); free(ascales);
    e2_q8w_free(&qw);
    return cos > 0.9995 && !kernel_bad;
}

static int check_attention_tail(int K) {
    if (!e2_i8_attn_gemm) return 0;
    const int M = 2, N = 8, Np = 16, Kp = (K + 3) / 4 * 4;
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    uint8_t *base = malloc((size_t)M * K + 1);
    uint8_t *b = calloc((size_t)Np * Kp, 1);
    float *asc = malloc(sizeof(float) * M * nb), bsc[16], out[16];
    int32_t *bz = malloc(sizeof(int32_t) * nb * Np);
    if (!base || !b || !asc || !bz) exit(2);
    uint8_t *a = base + 1;
    for (int i = 0; i < M * K; i++) a[i] = (uint8_t)(128 + i % 37);
    for (int i = 0; i < M * nb; i++) asc[i] = 1;
    for (int n = 0; n < Np; n++) {
        bsc[n] = 1;
        for (int k = 0; k < K; k++) b[((size_t)(k / 4) * Np + n) * 4 + k % 4] = 1;
        for (int block = 0; block < nb; block++) {
            int left = K - block * E2_I8_BLOCK;
            bz[block * Np + n] = left < E2_I8_BLOCK ? left : E2_I8_BLOCK;
        }
    }
    e2_i8_attn_gemm(out, N, a, K, asc, b, bsc, bz, M, Np, N, K);
    int bad = 0;
    for (int m = 0; m < M; m++) {
        float expected = 0;
        for (int k = 0; k < K; k++) expected += a[m * K + k] - 128;
        for (int n = 0; n < N; n++) if (!isfinite(out[m*N+n]) || fabsf(out[m*N+n] - expected) > 1e-4f) bad = 1;
    }
    free(base); free(b); free(asc); free(bz);
    printf("unaligned attention K=%d, exact-sized activation tail: %s\n", K, bad ? "FAIL" : "OK");
    return bad;
}

/* The fused softmax+probability quantizer against its own pieces run separately. This kernel had no
 * test on any backend, so it is checked against the reference path the callers fall back to when it
 * is NULL: K->softmax per row, then e2_i8_quantize. The two agree analytically (both scale the block
 * maximum by 127 and fold 1/sum into the block scale), so the only tolerated difference is the
 * fast polynomial exp moving a value across a rounding boundary. */
static int check_softmax_quant(int rows, int n) {
    if (!e2_softmax_quant_i8) return 0;
    const int nb = (n + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const size_t nf = (size_t)rows * n;
    float *x = malloc(sizeof(float) * nf), *ref = malloc(sizeof(float) * nf);
    int8_t *q8 = malloc(nf), *r8 = malloc(nf);
    float *sc = malloc(sizeof(float) * (size_t)rows * nb);
    float *rsc = malloc(sizeof(float) * (size_t)rows * nb);
    if (!x || !ref || !q8 || !r8 || !sc || !rsc) exit(2);
    for (size_t i = 0; i < nf; i++) x[i] = sinf((float)i * .013f) * 6.0f;
    memcpy(ref, x, sizeof(float) * nf);
    e2_softmax_quant_i8(x, q8, sc, rows, n, nb);
    for (int r = 0; r < rows; r++) e2_k()->softmax(ref + (size_t)r * n, n);
    e2_i8_quantize(r8, rsc, ref, rows, n, -INFINITY, INFINITY);

    double maxp = 0, maxs = 0;
    int maxbyte = 0, bad = 0;
    for (int r = 0; r < rows; r++) {
        /* x holds the unnormalised exponentials (the 1/sum is folded into the block scales), so
         * compare the probabilities it implies, not the raw rows */
        double sum = 0;
        for (int i = 0; i < n; i++) sum += x[(size_t)r * n + i];
        for (int i = 0; i < n; i++) {
            const double d = fabs(x[(size_t)r * n + i] / sum - (double)ref[(size_t)r * n + i]);
            if (d > maxp) maxp = d;
        }
    }
    for (size_t i = 0; i < nf; i++) {
        const int db = abs((int)(uint8_t)q8[i] - (int)(uint8_t)r8[i]);
        if (db > maxbyte) maxbyte = db;
        if (!isfinite(x[i]) || db > 1) bad = 1;
    }
    for (size_t i = 0; i < (size_t)rows * nb; i++) {
        const double d = fabs((double)sc[i] - (double)rsc[i]);
        if (d > maxs) maxs = d;
    }
    /* probabilities are ~1/n, so compare the scales relatively */
    if (!(maxs <= 1e-6 * (1.0 / n) + 1e-12)) bad = 1;
    if (!(maxp <= 2e-6)) bad = 1; /* the polynomial exp, not libm's, is the only difference */
    printf("softmax+quant rows=%-3d n=%-5d max|dP|=%.3e max|dscale|=%.3e max byte off=%d: %s\n",
           rows, n, maxp, maxs, maxbyte, bad ? "FAIL" : "OK");
    free(x); free(ref); free(q8); free(r8); free(sc); free(rsc);
    return bad;
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
    const int tails[] = {1,3,9,129};
    for (int i = 0; i < 4; i++) fails += check_attention_tail(tails[i]);
    {   /* fused softmax+quantizer: one block, exactly a block, several, and a patch-sized row */
        const int ns[] = { 32, 128, 200, 1300, 2520 };
        for (int i = 0; i < 5; i++) fails += check_softmax_quant(i ? 4 : 9, ns[i]);
    }
    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
