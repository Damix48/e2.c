/* mb - internal microbenchmark for the int8 GEMM and elementwise helpers.
 * Not part of the library; build with `make mb`.
 *
 *   ./mb gemm m k n [threads]     # GMAC/s for one linear
 *   ./mb peak                     # pure dpbusd / f32 fma ceiling
 */
#include "src/e2_internal.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static double now(void) { return e2_now_ms(); }

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void bench_gemm(int m, int k, int n, int nt) {
    e2_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (e2_pool_init(&ctx, nt) != 0) { printf("pool fail\n"); return; }
    float *w = malloc((size_t)n * k * sizeof(float));
    float *x = malloc((size_t)m * k * sizeof(float));
    float *y = malloc((size_t)m * n * sizeof(float));
    for (long i = 0; i < (long)n * k; i++) w[i] = (float)((i % 251) - 125) * 0.01f;
    for (long i = 0; i < (long)m * k; i++) x[i] = (float)((i % 233) - 116) * 0.01f;
    int8_t *q = NULL; float *sc = NULL; int32_t *ws = NULL; int blk = 128;
    if (e2_pack_mat(w, n, k, k, 1, &q, &sc, &ws, &blk) != 0) { printf("pack fail\n"); return; }
    e2_linear l; memset(&l, 0, sizeof(l));
    l.w.q = q; l.w.scale = sc; l.w.wsum = ws; l.w.n = n; l.w.k = k; l.w.packed = 1; l.w.blk = blk;
    for (int i = 0; i < 3; i++) e2_linear_forward(&ctx, y, x, m, &l);
    double ts[7];
    for (int i = 0; i < 7; i++) {
        double t0 = now();
        e2_linear_forward(&ctx, y, x, m, &l);
        ts[i] = now() - t0;
    }
    qsort(ts, 7, sizeof(double), cmp);
    double t = ts[0];
    double gmac = (double)m * n * k / 1e9;
    printf("gemm m=%d k=%d n=%d nt=%d  %.3f ms  %.2f GMAC/s  (med %.3f)\n",
           m, k, n, nt, t, gmac / (t / 1e3), ts[3]);
    free(w); free(x); free(y); e2_pack_free(q, sc, ws);
    e2_pool_free(&ctx);
}

/* raw loop ceilings */
static volatile float g_sink_f;
static volatile int32_t g_sink_i;

#if defined(__x86_64__)
#include <immintrin.h>
__attribute__((target("avx512f,avx512bw,avx512vnni")))
static double peak_vnni(int nt) {
    const int iters = 400000000 / nt;
    double t0 = now();
    __m512i a = _mm512_set1_epi32(0x7f7f7f7f);
    __m512i acc[4] = {a, a, a, a};
    __m512i w0 = _mm512_set1_epi8(3);
    for (int i = 0; i < iters / 4; i++) {
        acc[0] = _mm512_dpbusd_epi32(acc[0], a, w0);
        acc[1] = _mm512_dpbusd_epi32(acc[1], a, w0);
        acc[2] = _mm512_dpbusd_epi32(acc[2], a, w0);
        acc[3] = _mm512_dpbusd_epi32(acc[3], a, w0);
    }
    __m512i s = _mm512_add_epi32(_mm512_add_epi32(acc[0], acc[1]), _mm512_add_epi32(acc[2], acc[3]));
    g_sink_i = _mm512_reduce_add_epi32(s);
    double t = now() - t0;
    double mac = (double)iters * 64;
    return mac / (t / 1e3) / 1e9;
}

__attribute__((target("avx512f")))
static double peak_fma(int nt) {
    const int iters = 400000000 / nt;
    double t0 = now();
    __m512 a = _mm512_set1_ps(1.0000001f);
    __m512 acc[8];
    for (int j = 0; j < 8; j++) acc[j] = a;
    for (int i = 0; i < iters / 8; i++) {
        for (int j = 0; j < 8; j++) acc[j] = _mm512_fmadd_ps(acc[j], a, a);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
    s = _mm512_add_ps(s, _mm512_add_ps(_mm512_add_ps(acc[4], acc[5]), _mm512_add_ps(acc[6], acc[7])));
    g_sink_f = _mm512_reduce_add_ps(s);
    double t = now() - t0;
    return (double)iters * 16 / (t / 1e3) / 1e9;
}
#endif

static void noop_work(void *a, int tid, int nt) { (void)a; (void)tid; (void)nt; }

static void bench_par(int nt, int reps) {
    e2_ctx ctx; memset(&ctx, 0, sizeof(ctx));
    if (e2_pool_init(&ctx, nt) != 0) { printf("pool fail\n"); return; }
    for (int i = 0; i < 100; i++) e2_parallel(&ctx, noop_work, NULL);
    double t0 = now();
    for (int i = 0; i < reps; i++) e2_parallel(&ctx, noop_work, NULL);
    double t = (now() - t0) / reps * 1e3;
    printf("parallel dispatch nt=%d: %.2f us/barrier\n", nt, t);
    e2_pool_free(&ctx);
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: mb gemm m k n [nt] | mb peak | mb row k [nt]\n"); return 1; }
    e2_gemm_init();
    if (!strcmp(argv[1], "gemm") && argc >= 5) {
        int m = atoi(argv[2]), k = atoi(argv[3]), n = atoi(argv[4]);
        int nt = argc > 5 ? atoi(argv[5]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
        bench_gemm(m, k, n, nt);
    } else if (!strcmp(argv[1], "par")) {
        int nt = argc > 2 ? atoi(argv[2]) : 8;
        bench_par(nt, 2000);
    } else if (!strcmp(argv[1], "peak")) {
#if defined(__x86_64__)
        for (int nt = 1; nt <= 16; nt *= 2)
            printf("vnni nt=%2d %.2f GMAC/s | fma %.2f GMAC/s\n", nt,
                   peak_vnni(nt), peak_fma(nt));
#endif
    } else if (!strcmp(argv[1], "row") && argc >= 3) {
        int k = atoi(argv[2]);
        int nt = argc > 3 ? atoi(argv[3]) : 16;
        float *x = calloc((size_t)nt * k, sizeof(float));
        for (int i = 0; i < nt * k; i++) x[i] = 1.0f;
        int m = 4096;
        int nblk = k / 128;
        uint8_t *q = malloc((size_t)m * k);
        float *s = malloc((size_t)m * nblk * sizeof(float));
        double t0 = now();
        for (int r = 0; r < 4; r++) e2_quant_rows(x, m, k, q, s);
        double t = (now() - t0) / 4;
        printf("quant k=%d m=%d  %.3f ms  %.1f GB/s\n", k, m, t,
               (double)m * k * 5 / (t / 1e3) / 1e9);
    }
    return 0;
}
