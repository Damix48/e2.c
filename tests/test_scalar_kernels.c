/* test_scalar_kernels.c — the two kernels a host without AVX2/NEON runs.
 *
 *   ./e2_test_scalar_kernels      (self-contained, no model needed)
 *
 * Both are written against an independent reference here, not against themselves:
 *
 *  - e2_sgemm_nn_range_scalar (blocked i-k-j) vs the naive i-j-k it replaced, over strides, odd
 *    sizes and m0/m1 sub-ranges. The k order is the same, so the comparison is (near) exact; a
 *    layout or range bug shows up as an order-1 difference.
 *  - e2_i8_gemm_scalar (8 columns at a time, transposed block tile) vs the per-column form of the
 *    documented arithmetic, over non-8-aligned column ranges, a K that is not a block multiple, a
 *    clamped tensor, and a chunked call against one call (that one must be bit-identical: it is the
 *    property the tower's N-split relies on).
 *  - e2_quantize_blocks_scalar vs the documented scheme: scale = amax/127, round to nearest,
 *    saturate, store biased by +128.
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "int8.h"
#include "kernels.h"

static uint32_t g_seed = 12345u;
static uint32_t rnd(void) { return g_seed = g_seed * 1664525u + 1013904223u; }
static float frand(float s) { return ((int32_t)(rnd() >> 8) % 2001 - 1000) / 1000.0f * s; }

/* ---- 1. the blocked f32 GEMM ------------------------------------------------------------- */
static void sgemm_naive(float *C, int ldc, const float *A, int lda, const float *B, int ldb, int N,
                        int K, int m0, int m1) {
    for (int m = m0; m < m1; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += A[(size_t)m * lda + k] * B[(size_t)k * ldb + n];
            C[(size_t)m * ldc + n] = acc;
        }
}

static void check_sgemm(int M, int N, int K, int lda, int ldb, int ldc, int m0, int m1) {
    float *A = malloc(sizeof(float) * ((size_t)(M - 1) * lda + K + 1));
    float *B = malloc(sizeof(float) * ((size_t)(K - 1) * ldb + N + 1));
    float *C = malloc(sizeof(float) * ((size_t)M * ldc + 1));
    float *R = malloc(sizeof(float) * ((size_t)M * ldc + 1));
    assert(A && B && C && R);
    for (size_t i = 0; i < (size_t)(M - 1) * lda + K; i++) A[i] = frand(1.f);
    for (size_t i = 0; i < (size_t)(K - 1) * ldb + N; i++) B[i] = frand(1.f);
    /* poison the output so a kernel that skips a row or a column cannot pass on stale zeros */
    for (size_t i = 0; i < (size_t)M * ldc; i++) C[i] = R[i] = 1234.5f;

    e2_sgemm_nn_range_scalar(C, ldc, A, lda, B, ldb, M, N, K, m0, m1);
    sgemm_naive(R, ldc, A, lda, B, ldb, N, K, m0, m1);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            const size_t i = (size_t)m * ldc + n;
            if (m < m0 || m >= m1) {
                assert(C[i] == 1234.5f); /* outside the range: untouched */
                continue;
            }
            const float d = fabsf(C[i] - R[i]), e = 1e-4f * (1.f + fabsf(R[i]));
            if (d > e) {
                fprintf(stderr, "sgemm M=%d N=%d K=%d lda=%d ldb=%d ldc=%d [%d,%d) at %d,%d: "
                                "%.9g vs %.9g\n", M, N, K, lda, ldb, ldc, m0, m1, m, n, C[i], R[i]);
                assert(0);
            }
        }
    free(A);
    free(B);
    free(C);
    free(R);
}

static void test_sgemm(void) {
    check_sgemm(64, 512, 2048, 2048, 512, 512, 0, 64);  /* the weight GEMM shape */
    check_sgemm(196, 256, 196, 196, 256, 256, 0, 196);  /* attention context product */
    check_sgemm(1, 1, 1, 1, 1, 1, 0, 1);                /* degenerate */
    check_sgemm(7, 5, 13, 20, 9, 11, 0, 7);             /* every stride padded, every size odd */
    check_sgemm(17, 33, 65, 65, 40, 40, 3, 11);         /* a sub-range, as the pool splits it */
    check_sgemm(9, 300, 300, 300, 300, 300, 0, 9);      /* N and K both not a multiple of 4 */
    printf("sgemm_nn_range_scalar: ok\n");
}

/* ---- 2. the portable int8 GEMM ----------------------------------------------------------- */
/* The per-column form of the documented arithmetic: what the 8-column kernel replaces. */
static void i8_gemm_ref(float *y, const int8_t *qa, const float *asc, const e2_q8w *w, int M, int N,
                        int n0, int n1) {
    const int K = w->K, nb = w->nblocks, kq = (K + 3) / 4;
    for (int n = n0; n < n1; n++) {
        const float ws = w->scale[n];
        for (int m = 0; m < M; m++) {
            const int8_t *arow = qa + (size_t)m * K;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                int32_t d = 0;
                const int k0 = b * E2_I8_BLOCK;
                const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
                for (int k = k0; k < k1; k++)
                    d += ((int32_t)(uint8_t)arow[k] - 128) *
                         (int32_t)w->w[(size_t)(n >> 3) * kq * 32 + (size_t)(k >> 2) * 32 +
                                       (n & 7) * 4 + (k & 3)];
                acc += (float)d * (asc[(size_t)m * nb + b] * ws);
            }
            if (w->out_min > -1e30f) {
                if (acc < w->out_min) acc = w->out_min;
                else if (acc > w->out_max) acc = w->out_max;
            }
            y[(size_t)m * N + n] = acc;
        }
    }
}

static void check_i8(int M, int N, int K, int n0, int n1, float clamp_lo, float clamp_hi) {
    e2_q8w w;
    memset(&w, 0, sizeof w);
    float *wsrc = malloc(sizeof(float) * (size_t)N * K);
    int8_t *qa = malloc((size_t)M * K);
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    float *asc = malloc(sizeof(float) * (size_t)M * nb);
    float *y = malloc(sizeof(float) * (size_t)M * N);
    float *r = malloc(sizeof(float) * (size_t)M * N);
    float *chunk = malloc(sizeof(float) * (size_t)M * N);
    assert(wsrc && qa && asc && y && r && chunk);
    for (size_t i = 0; i < (size_t)N * K; i++) wsrc[i] = frand(0.05f);
    assert(e2_q8w_from_f32(&w, wsrc, N, K) == 0);
    w.out_min = clamp_lo;
    w.out_max = clamp_hi;
    for (size_t i = 0; i < (size_t)M * K; i++) qa[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int i = 0; i < M * nb; i++) asc[i] = 0.001f + 0.01f * (float)(i % 7);
    for (size_t i = 0; i < (size_t)M * N; i++) y[i] = r[i] = chunk[i] = -777.f;

    e2_i8_gemm_scalar(y, qa, asc, &w, M, N, n0, n1);
    i8_gemm_ref(r, qa, asc, &w, M, N, n0, n1);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            const size_t i = (size_t)m * N + n;
            if (n < n0 || n >= n1) {
                assert(y[i] == -777.f); /* the column range is respected exactly */
                continue;
            }
            const float d = fabsf(y[i] - r[i]), e = 1e-5f * (1.f + fabsf(r[i]));
            if (d > e) {
                fprintf(stderr, "i8 M=%d N=%d K=%d [%d,%d) at %d,%d: %.9g vs %.9g\n", M, N, K, n0,
                        n1, m, n, y[i], r[i]);
                assert(0);
            }
        }

    /* the tower splits a GEMM along N across threads: the split must not change a single bit */
    for (int a = n0; a < n1;) {
        int b = a + 7; /* deliberately not a multiple of 8: the group masking has to cope */
        if (b > n1) b = n1;
        e2_i8_gemm_scalar(chunk, qa, asc, &w, M, N, a, b);
        a = b;
    }
    for (int m = 0; m < M; m++)
        for (int n = n0; n < n1; n++)
            assert(memcmp(&chunk[(size_t)m * N + n], &y[(size_t)m * N + n], sizeof(float)) == 0);
    e2_q8w_free(&w);
    free(wsrc);
    free(qa);
    free(asc);
    free(y);
    free(r);
    free(chunk);
}

static void test_i8_gemm(void) {
    check_i8(64, 512, 2048, 0, 512, -1e31f, 1e31f);   /* the weight GEMM shape */
    check_i8(15, 256, 512, 0, 256, -1e31f, 1e31f);    /* a query's attention context product */
    check_i8(9, 19, 200, 0, 19, -1e31f, 1e31f);       /* K not a block multiple, odd N */
    check_i8(4, 40, 384, 3, 29, -1e31f, 1e31f);       /* non-8-aligned n0, masked lanes at both ends */
    check_i8(8, 64, 256, 0, 64, -0.5f, 0.5f);         /* the audio tower's clipped linear */
    printf("i8_gemm_scalar: ok\n");
}

/* ---- 3. the portable activation quantizer ------------------------------------------------ */
static void quantize_ref(int8_t *out, float *scales, const float *x, int M, int K, float lo,
                         float hi) {
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const int clamp = lo > -INFINITY || hi < INFINITY;
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            int k1 = k0 + E2_I8_BLOCK;
            if (k1 > K) k1 = K;
            float amax = 0.f;
            for (int k = k0; k < k1; k++) {
                float v = x[(size_t)m * K + k];
                if (clamp) v = v < lo ? lo : (v > hi ? hi : v);
                if (fabsf(v) > amax) amax = fabsf(v);
            }
            const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
            const float inv = 1.0f / sc; /* the kernel multiplies by the reciprocal, like the SIMD
                                          * ones (pmulps), so v/sc and v*inv can round apart */
            for (int k = k0; k < k1; k++) {
                float v = x[(size_t)m * K + k];
                if (clamp) v = v < lo ? lo : (v > hi ? hi : v);
                int r = (int)lrintf(v * inv);
                if (r > 127) r = 127;
                else if (r < -127) r = -127;
                out[(size_t)m * K + k] = (int8_t)(r + 128);
            }
            scales[(size_t)m * nb + b] = sc;
        }
}

static void check_quantize(int M, int K, float lo, float hi) {
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    float *x = malloc(sizeof(float) * (size_t)M * K);
    int8_t *a = malloc((size_t)M * K), *b = malloc((size_t)M * K);
    float *sa = malloc(sizeof(float) * (size_t)M * nb), *sb = malloc(sizeof(float) * (size_t)M * nb);
    assert(x && a && b && sa && sb);
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = frand(3.f);
    /* a couple of rows are all zeros and one block is all zeros: scale 1, every byte 128 */
    for (int k = 0; k < K; k++) x[k] = 0.f;
    for (int m = 0; m < M; m++) x[(size_t)m * K] = 0.f;

    e2_quantize_blocks_scalar(a, sa, x, M, K, lo, hi);
    quantize_ref(b, sb, x, M, K, lo, hi);
    /* The library is built with -ffast-math, so the compiler may reassociate 1.0f/(amax/127.0f)
     * into 127.0f/amax; the AVX2/NEON quantizers are written with intrinsics and cannot be. The two
     * can therefore disagree by one quantization step exactly at a rounding boundary - so bytes are
     * compared within one unit, and the scales (a max, which is exact either way) exactly. */
    for (int i = 0; i < M * K; i++) {
        const int qa = (int)(uint8_t)a[i] - 128, qb = (int)(uint8_t)b[i] - 128;
        if (abs(qa - qb) > 1) {
            fprintf(stderr, "quantize M=%d K=%d [%.3g,%.3g] at %d: %d vs %d (x=%.9g, sc=%.9g)\n", M, K,
                    (double)lo, (double)hi, i, qa, qb, (double)x[i],
                    (double)sa[(i / K) * nb + (i % K) / E2_I8_BLOCK]);
            assert(0);
        }
    }
    for (int i = 0; i < M * nb; i++) {
        const float d = fabsf(sa[i] - sb[i]), e = 1e-6f * sb[i];
        if (d > e) {
            fprintf(stderr, "quantize scale %d: %.9g vs %.9g\n", i, (double)sa[i], (double)sb[i]);
            assert(0);
        }
    }
    /* every stored byte is the value + 128, and no value saturates outside +-127 */
    for (int m = 0; m < M; m++)
        for (int blk = 0; blk < nb; blk++) {
            const float sc = sa[(size_t)m * nb + blk];
            assert(sc > 0.f);
            const int k0 = blk * E2_I8_BLOCK;
            int k1 = k0 + E2_I8_BLOCK;
            if (k1 > K) k1 = K;
            for (int k = k0; k < k1; k++) {
                const int q = (int)(uint8_t)a[(size_t)m * K + k] - 128;
                assert(q >= -127 && q <= 127);
                const float back = (float)q * sc;
                float v = x[(size_t)m * K + k]; /* the quantizer sees the clamped value */
                if (lo > -INFINITY || hi < INFINITY) v = v < lo ? lo : (v > hi ? hi : v);
                assert(fabsf(back - v) <= 1.01f * sc);
            }
        }
    free(x);
    free(a);
    free(b);
    free(sa);
    free(sb);
}

static void test_quantize(void) {
    check_quantize(8, 512, -INFINITY, INFINITY);
    check_quantize(5, 200, -INFINITY, INFINITY); /* short last block */
    check_quantize(3, 256, -0.75f, 0.75f);       /* the audio tower clamps its inputs */
    printf("quantize_blocks_scalar: ok\n");
}

int main(void) {
    test_sgemm();
    test_i8_gemm();
    test_quantize();
    printf("all scalar kernel checks passed\n");
    return 0;
}
