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

/* Naive f32 reference attention, bidirectional with a symmetric +/-window. */
static void ref_attn(float *out, const float *q, const float *k, const float *v,
                     int t, int heads, int kvh, int hd, int window) {
    int groups = heads / kvh;
    float *sc = malloc((size_t)t * sizeof(float));
    for (int h = 0; h < heads; h++) {
        int kh = h / groups;
        for (int i = 0; i < t; i++) {
            int lo = window > 0 ? (i - window < 0 ? 0 : i - window) : 0;
            int hi = window > 0 ? (i + window + 1 > t ? t : i + window + 1) : t;
            float mx = -INFINITY;
            for (int j = lo; j < hi; j++) {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += q[(size_t)i * heads * hd + h * hd + d] *
                         k[(size_t)j * kvh * hd + kh * hd + d];
                sc[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0;
            for (int j = lo; j < hi; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            for (int d = 0; d < hd; d++) {
                float a = 0;
                for (int j = lo; j < hi; j++)
                    a += sc[j] * v[(size_t)j * kvh * hd + kh * hd + d];
                out[(size_t)i * heads * hd + h * hd + d] = a / sum;
            }
        }
    }
    free(sc);
}

static float *rd(const char *p, size_t n) {
    float *b = malloc(n * 4);
    FILE *f = fopen(p, "rb");
    if (!f) { printf("cannot open %s\n", p); exit(1); }
    size_t got = fread(b, 4, n, f);
    fclose(f);
    if (got != n) { printf("%s: %zu != %zu\n", p, got, n); exit(1); }
    return b;
}

/* compare the int8 kernel against the f32 reference on real model tensors */
static void test_attn_file(const char *qp, const char *kp, const char *vp,
                           int t, int heads, int kvh, int hd, int window, int nt) {
    e2_ctx ctx; memset(&ctx, 0, sizeof(ctx));
    if (e2_pool_init(&ctx, nt) != 0) { printf("pool fail\n"); return; }
    size_t qn = (size_t)t * heads * hd, kn = (size_t)t * kvh * hd;
    float *q = rd(qp, qn), *k = rd(kp, kn), *v = rd(vp, kn);
    float *a = malloc(qn * 4), *b = malloc(qn * 4), *o = malloc(qn * 4);
    int off = 0;
    e2_attn_int8(&ctx, a, q, k, v, t, heads, kvh, hd, window, 1, &off);
    /* old implementation */
    e2_attention_text(&ctx, o, q, k, v, t, heads, kvh, hd, window);
    ref_attn(b, q, k, v, t, heads, kvh, hd, window);
    double e_new = 0, e_old = 0, e_pair = 0;
    for (size_t i = 0; i < qn; i++) {
        e_new += fabs(a[i] - b[i]); e_old += fabs(o[i] - b[i]);
        e_pair += fabs(a[i] - o[i]);
    }
    printf("real data t=%d h=%d kvh=%d hd=%d: |new-ref| %.4f  |old-ref| %.4f  |new-old| %.4f per elem\n",
           t, heads, kvh, hd, e_new / qn, e_old / qn, e_pair / qn);
    free(q); free(k); free(v); free(a); free(b); free(o);
    e2_pool_free(&ctx);
}

static void bench_attn(int t, int heads, int kvh, int hd, int window, int nt) {
    e2_ctx ctx; memset(&ctx, 0, sizeof(ctx));
    if (e2_pool_init(&ctx, nt) != 0) { printf("pool fail\n"); return; }
    size_t qn = (size_t)t * heads * hd, kn = (size_t)t * kvh * hd;
    float *q = malloc(qn * 4), *k = malloc(kn * 4), *v = malloc(kn * 4);
    float *a = malloc(qn * 4), *b = malloc(qn * 4);
    /* well-conditioned logits: q,k ~ U(-1,1)/sqrt(hd) so scores are O(1) */
    float scl = 1.0f / sqrtf((float)hd);
    for (size_t i = 0; i < qn; i++)
        q[i] = (float)((int)(i * 37 % 199) - 99) / 99.0f * scl;
    for (size_t i = 0; i < kn; i++) {
        k[i] = (float)((int)(i * 53 % 197) - 98) / 98.0f * scl;
        v[i] = (float)((int)(i * 29 % 193) - 96) / 96.0f;
    }
    int off = 0;
    double t0 = now();
    int rc = e2_attn_int8(&ctx, a, q, k, v, t, heads, kvh, hd, window, 1, &off);
    double dt = now() - t0;
    if (rc) { printf("attn int8 unsupported\n"); return; }
    ref_attn(b, q, k, v, t, heads, kvh, hd, window);
    double d = 0, nb = 0;
    size_t worst = 0;
    for (size_t i = 0; i < qn; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > d) { d = e; worst = i; }
        nb += b[i] * b[i];
    }
    int per = heads * hd;
    printf("attn t=%d h=%d kvh=%d hd=%d win=%d nt=%d  %.3f ms  maxerr %.3e rms %.3e "
           "worst row %zu head %zu\n", t, heads, kvh, hd, window, nt, dt, d,
           sqrt(nb / qn), worst / per, (worst % per) / hd);
    int bad = 0;
    for (size_t i = 0; i < qn; i++) if (fabs(a[i] - b[i]) > 0.05) bad++;
    { int first = -1, last = -1, nrow = 0;
      for (int r = 0; r < t; r++) {
        int rb = 0;
        for (int hh = 0; hh < heads * hd; hh++)
            if (fabs(a[(size_t)r * per + hh] - b[(size_t)r * per + hh]) > 0.05) { rb = 1; break; }
        if (rb) { if (first < 0) first = r; last = r; nrow++; }
      }
      printf("   bad rows %d first %d last %d\n", nrow, first, last); }
    printf("   bad %.3f%%  worst a=%.6f b=%.6f  row %zu col %zu\n",
           100.0 * bad / qn, a[worst], b[worst], worst / per, worst % per);
    for (size_t i = 0; i < 3; i++) printf("   %.6f  %.6f\n", a[i], b[i]);
    free(q); free(k); free(v); free(a); free(b);
    e2_pool_free(&ctx);
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: mb gemm m k n [nt] | mb peak | mb row k [nt]\n"); return 1; }
    e2_gemm_init();
    if (!strcmp(argv[1], "gemm") && argc >= 5) {
        int m = atoi(argv[2]), k = atoi(argv[3]), n = atoi(argv[4]);
        int nt = argc > 5 ? atoi(argv[5]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
        bench_gemm(m, k, n, nt);
    } else if (!strcmp(argv[1], "attn") && argc >= 6) {
        int t = atoi(argv[2]), h = atoi(argv[3]), kvh = atoi(argv[4]);
        int hd = atoi(argv[5]);
        int win = argc > 6 ? atoi(argv[6]) : 0;
        int nt = argc > 7 ? atoi(argv[7]) : 8;
        bench_attn(t, h, kvh, hd, win, nt);
    } else if (!strcmp(argv[1], "attnf") && argc >= 8) {
        test_attn_file(argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]),
                       atoi(argv[7]), argc > 8 ? atoi(argv[8]) : 64,
                       argc > 9 ? atoi(argv[9]) : 0, argc > 10 ? atoi(argv[10]) : 8);
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
