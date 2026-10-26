/* gpu_bench.c — device probe, kernel correctness check and microbenchmarks (Level Zero).
 *
 * What it answers, on this machine rather than on paper:
 *   --check   the int8 GEMM (short- and long-sequence paths) against the scalar CPU implementation
 *   default   one kernel's dispatch cost inside a recorded command list, the reachable bandwidth, and
 *             the throughput of the model's real projection shapes
 *
 * Everything goes through the same kernels the engine uses, so a shape that only works here cannot
 * pass unseen.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "int8.h"
#include "kernels.h"
#include "gpu/l0.h"
#include "../src/gpu/e2_tile.h" /* the same tile the kernels are built with */

#define E2_NM 16

/* rows of attention one work group handles (local_size(1)); the kernel is generic in it */
static int ATTN_GY = 4;

/* E2_BENCH_GEMM=0|1|2 forces the GEMM the shape table times (gemv / dp4a tiled / XMX dpas);
 * unset picks the kernel the engine ships (dpas for more than E2_NM tokens, gemv below).
 * E2_BENCH_GEMM_OLD=1 times k_gemm_dpas instead of k_gemm_dpas2. */
static int BENCH_MODE = -1;
static int BENCH_GY = E2_GEMM_GY; /* sub-groups per work group for the DPAS GEMM (the engine's `gy`) */


static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static e2_l0 *G;

#define OK(expr) do { if ((expr)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, e2_l0_error(G)); exit(1); } } while (0)
#define REQ(expr) do { if (!(expr)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, e2_l0_error(G)); exit(1); } } while (0)

/* k_attn_gemm against a scalar reference on the same bytes. The attention weights are packed by a
 * kernel (k_attn_packw) rather than by q8w_dpas, so this checks the DPAS A layout, the mixed-sign
 * GEMM and the +128 activation bias on their own, without the model in the way. */
static int check_attn(void) {
    /* the token tile is E2_DM tall and the loads are unguarded, so M must cover at least one tile */
    const int N = 128, K = 256, M = E2_DM > 16 ? E2_DM : 16, NQ = N / 8, nb = K / 128;
    void *wd = e2_l0_alloc(G, (size_t)N * K + 64), *wsc = e2_l0_alloc(G, (size_t)N * 4);
    void *bsq = e2_l0_alloc(G, (size_t)nb * N * 4), *qa = e2_l0_alloc(G, (size_t)M * K + 64);
    void *asc = e2_l0_alloc(G, (size_t)M * nb * 4), *y = e2_l0_alloc(G, (size_t)M * N * 4);
    uint8_t *wdh = calloc(1, (size_t)N * K + 64), *qah = calloc(1, (size_t)M * K + 64);
    float *wsch = malloc((size_t)N * 4), *asch = malloc((size_t)M * nb * 4), *yh = malloc((size_t)M * N * 4);
    int32_t *bsqh = calloc((size_t)nb * N, 4);
    int8_t w8[128][256];
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            const int v = (n * 37 + k * 11) % 101 - 50;
            w8[n][k] = (int8_t)v;
            wdh[E2_DPAS_OFF(k / 4, n >> 3, n & 7, k & 3, NQ)] = (uint8_t)v;
            bsqh[(size_t)(k / 128) * N + n] += v;
        }
        wsch[n] = 1.f;
    }
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) qah[(size_t)m * K + k] = (uint8_t)((m * 13 + k * 5) % 121 - 60 + 128);
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++) asch[(size_t)m * nb + b] = 1.f;
    OK(e2_l0_write(G, wd, (size_t)N * K + 64, wdh));
    OK(e2_l0_write(G, wsc, (size_t)N * 4, wsch));
    OK(e2_l0_write(G, bsq, (size_t)nb * N * 4, bsqh));
    OK(e2_l0_write(G, qa, (size_t)M * K + 64, qah));
    OK(e2_l0_write(G, asc, (size_t)M * nb * 4, asch));
    e2_graph *gr = e2_l0_graph_begin(G, 1);
    /* M <= E2_DM, so round the token grid up like the engine does or a tile taller than M launches
     * zero work groups and the check silently passes on zeros */
    OK(e2_l0_append(G, gr, "k_attn_gemm", (uint32_t)((M + E2_DM - 1) / E2_DM),
                    (uint32_t)((N + E2_DN - 1) / E2_DN), 1, 8, 1, 1, 13, &wd, sizeof wd,
                    &wsc, sizeof wsc, &bsq, sizeof bsq, &qa, sizeof qa, &asc, sizeof asc, &y, sizeof y,
                    &N, sizeof N, &K, sizeof K, &nb, sizeof nb, &M, sizeof M, &N, sizeof N, &K,
                    sizeof K, &nb, sizeof nb));
    OK(e2_l0_graph_close(G, gr));
    OK(e2_l0_graph_run(G, gr));
    OK(e2_l0_read(G, y, (size_t)M * N * 4, yh));
    int bad = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int acc = 0;
            for (int k = 0; k < K; k++) acc += ((int)qah[(size_t)m * K + k] - 128) * (int)w8[n][k];
            if ((int)yh[(size_t)m * N + n] != acc) {
                if (bad < 3)
                    printf("  y[%d][%d] = %g, expected %d\n", m, n, yh[(size_t)m * N + n], acc);
                bad++;
            }
        }
    printf("k_attn_gemm (N=%d K=%d M=%d): %d/%d wrong\n", N, K, M, bad, M * N);
    free(wdh); free(qah); free(wsch); free(asch); free(bsqh); free(yh);
    e2_l0_free(G, wd); e2_l0_free(G, wsc); e2_l0_free(G, bsq);
    e2_l0_free(G, qa); e2_l0_free(G, asc); e2_l0_free(G, y);
    e2_l0_graph_free(G, gr);
    return bad != 0;
}

/* one weight tensor, packed the way the engine packs it */
typedef struct {
    void *wbuf, *acts, *part, *desc, *xb;
    int32_t desc_h[12];
    e2_q8w q;
    size_t wbytes, acts_bytes, qa_off, asc_off, y_off;
} tdev;

static size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }

static void pack_weights(tdev *t, const float *w, int N, int K) {
    if (e2_q8w_from_f32(&t->q, w, N, K)) { printf("quantize failed\n"); exit(1); }
    const int KQ = K / 4, nb = t->q.nblocks;
    const size_t kqw = (size_t)KQ * N * 4;
    const size_t dpas_off = ((kqw + (size_t)N * 4 + (size_t)nb * N * 4 + 63) & ~(size_t)63);
    const size_t total = dpas_off + (size_t)N * K;
    uint8_t *blob = calloc(1, total);
    for (int n = 0; n < N; n++) {
        const int8_t *row = t->q.w + (size_t)(n >> 3) * KQ * 32 + (n & 7) * 4;
        for (int kq = 0; kq < KQ; kq++) memcpy(blob + ((size_t)kq * N + n) * 4, row + (size_t)kq * 32, 4);
    }
    memcpy(blob + kqw, t->q.scale, sizeof(float) * (size_t)N);
    int32_t *bsq = (int32_t *)(blob + kqw + (size_t)N * 4);
    for (int b = 0; b < nb; b++)
        for (int n = 0; n < N; n++) bsq[(size_t)b * N + n] = t->q.bsq[(size_t)b * t->q.Npad + n];
    /* the DPAS A layout, same as q8w_dpas() in gpu.c */
    for (int n = 0; n < N; n++)
        for (int kq = 0; kq < KQ; kq++) {
            const int8_t *row = t->q.w + (size_t)(n >> 3) * KQ * 32 + (n & 7) * 4;
            memcpy(blob + dpas_off + E2_DPAS_OFF(kq, n >> 3, n & 7, 0, N / 8), row + (size_t)kq * 32,
                   4);
        }

    const size_t qa_off = 0, asc_off = (size_t)4096 * K, y_off = align16(asc_off + (size_t)4096 * 16 * 4);
    const size_t acts_bytes = y_off + sizeof(float) * (size_t)8192 * N + (size_t)32 * K * 4;
    t->qa_off = qa_off;
    t->asc_off = asc_off;
    t->y_off = y_off;
    t->wbytes = total;
    t->acts_bytes = acts_bytes;
    t->wbuf = e2_l0_alloc(G, total);
    t->acts = e2_l0_alloc(G, acts_bytes);
    t->xb = e2_l0_alloc(G, sizeof(float) * (size_t)8192 * K);
    t->part = e2_l0_alloc(G, (size_t)8 * E2_NM * (size_t)N * 4 * 4);
    t->desc = e2_l0_alloc(G, sizeof t->desc_h);
    REQ(t->wbuf && t->acts && t->xb && t->part && t->desc);
    OK(e2_l0_write(G, t->wbuf, total, blob));
    free(blob);
    memset(t->desc_h, 0, sizeof t->desc_h);
    t->desc_h[0] = 0;
    t->desc_h[1] = (int32_t)kqw;
    t->desc_h[2] = (int32_t)(kqw + (size_t)N * 4);
    t->desc_h[3] = (int32_t)t->y_off;
    t->desc_h[7] = (int32_t)t->qa_off;
    t->desc_h[8] = (int32_t)t->asc_off;
    t->desc_h[9] = (int32_t)dpas_off;
    t->desc_h[4] = N;
    t->desc_h[5] = K;
    t->desc_h[6] = nb;
    OK(e2_l0_write(G, t->desc, sizeof t->desc_h, t->desc_h));
}

static void free_tdev(tdev *t) {
    e2_q8w_free(&t->q);
    e2_l0_free(G, t->wbuf);
    e2_l0_free(G, t->acts);
    e2_l0_free(G, t->xb);
    e2_l0_free(G, t->part);
    e2_l0_free(G, t->desc);
}

/* x (M x K floats) -> quantized activations in t->acts, then the GEMM into t->acts+0.
 * mode: 0 = k_gemv_split (short sequences), 1 = k_gemm_tiled (dp4a), 2 = k_gemm_dpas (XMX).
 * with_quant=0 records the GEMM alone, so the two can be timed apart. */
static e2_graph *record_case(tdev *t, const float *x, int M, int mode, int with_quant) {
    const int N = t->q.N, K = t->q.K, nb = t->q.nblocks, nsl = 4;
    OK(e2_l0_write(G, t->xb, sizeof(float) * (size_t)M * K, x));
    e2_graph *gr = e2_l0_graph_begin(G, 1);
    REQ(gr);
    void *xr = t->xb, *src = NULL, *res = NULL, *qa = (char *)t->acts + t->qa_off;
    void *asc = (char *)t->acts + t->asc_off, *gam = NULL, *gam2 = NULL, *valid = NULL;
    const float eps = 1e-6f, prescale = 1.f, scale = 1.f;
    const int rows = M, Kd = K, nbd = nb, flags = 16 | 2;
    if (with_quant)
        OK(e2_l0_append(G, gr, "k_rowops", (uint32_t)((M + 3) / 4), 1, 1, 32, 4, 1, 15, &xr, sizeof xr,
                        &src, sizeof src, &res, sizeof res, &qa, sizeof qa, &asc, sizeof asc, &gam,
                        sizeof gam, &gam2, sizeof gam2, &eps, sizeof eps, &prescale, sizeof prescale,
                        &scale, sizeof scale, &rows, sizeof rows, &Kd, sizeof Kd, &nbd, sizeof nbd,
                        &flags, sizeof flags, &valid, sizeof valid));
    if (mode == 2 || mode == 3) {
        /* 2 = k_gemm_dpas2 (shipped), 3 = k_gemm_dpas_slm (SLM-staged B). E2_BENCH_GEMM_OLD=1 times
         * k_gemm_dpas instead of k_gemm_dpas2. E2_BENCH_GY is the sub-groups per work group. */
        REQ(N % E2_DN == 0);
        const int mpad = (M + E2_DM - 1) / E2_DM * E2_DM;
        const char *kname = mode == 3 ? "k_gemm_dpas_slm"
                                      : (getenv("E2_BENCH_GEMM_OLD") ? "k_gemm_dpas" : "k_gemm_dpas2");
        OK(e2_l0_append(G, gr, kname, (uint32_t)(mpad / E2_DM),
                        (uint32_t)((N + E2_DN * BENCH_GY - 1) / (E2_DN * BENCH_GY)), 1, 8,
                        (uint32_t)BENCH_GY, 1, 4, &t->wbuf,
                        sizeof t->wbuf, &t->acts, sizeof t->acts, &t->desc, sizeof t->desc, &M,
                        sizeof M));
    } else if (mode == 1) {
        /* k_gemm_tiled's k loop is unguarded, so M must be a multiple of its 32-row tile */
        const int mpad = (M + E2_BM - 1) / E2_BM * E2_BM;
        OK(e2_l0_append(G, gr, "k_gemm_tiled", (uint32_t)(mpad / E2_BM), (uint32_t)((N + E2_BN - 1) / E2_BN), 1,
                        E2_GEMM_NG, E2_GEMM_MG, 1, 4, &t->wbuf, sizeof t->wbuf, &t->acts,
                        sizeof t->acts, &t->desc, sizeof t->desc, &mpad, sizeof mpad));
    } else {
        OK(e2_l0_append(G, gr, "k_gemv_split", (uint32_t)((N + 127) / 128), (uint32_t)nsl, 1, 128, 1, 1,
                        7, &t->wbuf, sizeof t->wbuf, &t->acts, sizeof t->acts, &t->part,
                        sizeof t->part, &t->desc, sizeof t->desc, &nsl, sizeof nsl, &rows,
                        sizeof rows, &N, sizeof N));
        OK(e2_l0_append(G, gr, "k_gemv_reduce", (uint32_t)((N + 127) / 128), 1, 1, 128, 1, 1, 6,
                        &t->acts, sizeof t->acts, &t->part, sizeof t->part, &t->desc, sizeof t->desc,
                        &nsl, sizeof nsl, &rows, sizeof rows, &N, sizeof N));
    }
    OK(e2_l0_graph_close(G, gr));
    return gr;
}

/* the engine launches q|k|v in ONE kernel (grid.z = 3): check that path too, since a single-tensor
 * check cannot catch a descriptor-index bug */
static int check_multi(void) {
    const int N = 1024, K = 512, count = 3;
    const int Ms[] = { 15, 21, 32, 40, 196 };
    int fails = 0;
    for (int mi = 0; mi < 5; mi++) {
        const int M = Ms[mi];
        const int tiled = M > E2_NM;
        e2_q8w qs[3];
        float *ws[3];
        void *desc = e2_l0_alloc(G, (size_t)count * 12 * 4);
        int32_t dh[count * 12];
        memset(dh, 0, sizeof dh);
        size_t woff = 0;
        uint8_t *blob = calloc(1, (size_t)count * 3 << 20);
        size_t blobsz = (size_t)count * 3 << 20;
        for (int t = 0; t < count; t++) {
            ws[t] = malloc(sizeof(float) * (size_t)N * K);
            for (size_t i = 0; i < (size_t)N * K; i++)
                ws[t][i] = (float)((int)(i * (7919 + t * 104729) % 977) - 488) / 60.f;
            e2_q8w_from_f32(&qs[t], ws[t], N, K);
            const int KQ = K / 4, nb = qs[t].nblocks;
            const size_t kqw = (size_t)KQ * N * 4, sz = kqw + (size_t)N * 4 + (size_t)nb * N * 4;
            woff = (woff + 63) & ~(size_t)63;
            uint8_t *dst = blob + woff;
            for (int n = 0; n < N; n++) {
                const int8_t *row = qs[t].w + (size_t)(n >> 3) * KQ * 32 + (n & 7) * 4;
                for (int kq = 0; kq < KQ; kq++) memcpy(dst + ((size_t)kq * N + n) * 4, row + (size_t)kq * 32, 4);
            }
            memcpy(dst + kqw, qs[t].scale, sizeof(float) * (size_t)N);
            int32_t *bsq = (int32_t *)(dst + kqw + (size_t)N * 4);
            for (int b = 0; b < nb; b++)
                for (int n = 0; n < N; n++) bsq[(size_t)b * N + n] = qs[t].bsq[(size_t)b * qs[t].Npad + n];
            dh[t * 12 + 0] = (int32_t)woff;
            dh[t * 12 + 1] = (int32_t)(woff + kqw);
            dh[t * 12 + 2] = (int32_t)(woff + kqw + (size_t)N * 4);
            dh[t * 12 + 4] = N;
            dh[t * 12 + 5] = K;
            dh[t * 12 + 6] = nb;
            woff += sz;
        }
        if (woff > blobsz) { printf("blob too small\n"); return 1; }
        void *wbuf = e2_l0_alloc(G, woff);
        REQ(wbuf && desc);
        OK(e2_l0_write(G, wbuf, woff, blob));
        free(blob);
        /* outputs and activations: one region per tensor */
        const size_t qa_off = 0, asc_off = (size_t)4096 * K;
        const size_t y_off = asc_off + (size_t)4096 * 16 * 4;
        void *acts = e2_l0_alloc(G, y_off + (size_t)count * 8192 * N * 4 + (size_t)32 * K * 4);
        void *xb = e2_l0_alloc(G, sizeof(float) * (size_t)M * K);
        void *part = e2_l0_alloc(G, (size_t)count * 8 * E2_NM * N * 4 * 4);
        REQ(acts && xb && part);
        for (int t = 0; t < count; t++) {
            dh[t * 12 + 3] = (int32_t)(y_off + (size_t)t * 8192 * N * 4);
            dh[t * 12 + 7] = (int32_t)qa_off;
            dh[t * 12 + 8] = (int32_t)asc_off;
        }
        OK(e2_l0_write(G, desc, sizeof dh, dh));
        float *x = malloc(sizeof(float) * (size_t)M * K);
        for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (float)((int)(i * 131 % 600) - 300) / 40.f;
        OK(e2_l0_write(G, xb, sizeof(float) * (size_t)M * K, x));
        void *xr = xb, *src = NULL, *res = NULL, *qa = acts;
        void *asc = (char *)acts + asc_off, *gam = NULL, *gam2 = NULL, *valid = NULL;
        const float eps = 1e-6f, prescale = 1.f, scale = 1.f;
        const int rows = M, Kd = K;
        e2_graph *gr = e2_l0_graph_begin(G, 1);
        REQ(gr);
        OK(e2_l0_append(G, gr, "k_rowops", (uint32_t)((M + 3) / 4), 1, 1, 32, 4, 1, 15, &xr, sizeof xr,
                        &src, sizeof src, &res, sizeof res, &qa, sizeof qa, &asc, sizeof asc, &gam,
                        sizeof gam, &gam2, sizeof gam2, &eps, sizeof eps, &prescale, sizeof prescale,
                        &scale, sizeof scale, &rows, sizeof rows, &Kd, sizeof Kd,
                        &((int){ K / 128 }), sizeof(int), &((int){ 16 | 2 }), sizeof(int), &valid,
                        sizeof valid));
        if (tiled) {
            const int mpad = (M + E2_BM - 1) / E2_BM * E2_BM;
            OK(e2_l0_append(G, gr, "k_gemm_tiled", (uint32_t)(mpad / E2_BM), (uint32_t)((N + E2_BN - 1) / E2_BN),
                            (uint32_t)count, E2_GEMM_NG, E2_GEMM_MG, 1, 4, &wbuf, sizeof wbuf, &acts,
                            sizeof acts, &desc, sizeof desc, &mpad, sizeof mpad));
        }
        else {
            const int nsl = 4;
            OK(e2_l0_append(G, gr, "k_gemv_split", (uint32_t)((N + 127) / 128), (uint32_t)nsl,
                            (uint32_t)count, 128, 1, 1, 7, &wbuf, sizeof wbuf, &acts, sizeof acts, &part,
                            sizeof part, &desc, sizeof desc, &nsl, sizeof nsl, &rows, sizeof rows, &N,
                            sizeof N));
            OK(e2_l0_append(G, gr, "k_gemv_reduce", (uint32_t)((N + 127) / 128), (uint32_t)count, 1, 128,
                            1, 1, 6, &acts, sizeof acts, &part, sizeof part, &desc, sizeof desc, &nsl,
                            sizeof nsl, &rows, sizeof rows, &N, sizeof N));
        }
        OK(e2_l0_graph_close(G, gr));
        OK(e2_l0_graph_run(G, gr));
        for (int t = 0; t < count; t++) {
            float *out = malloc(sizeof(float) * (size_t)M * N);
            int8_t *qa_h = calloc((size_t)M * K, 1);
            float *asc_h = calloc((size_t)M * qs[t].nblocks, sizeof(float));
            e2_i8_quantize(qa_h, asc_h, x, M, K, -INFINITY, INFINITY);
            float *ref = calloc((size_t)M * N, sizeof(float));
            e2_i8_gemm_scalar(ref, qa_h, asc_h, &qs[t], M, N, 0, N);
            OK(e2_l0_read(G, (char *)acts + y_off + (size_t)t * 8192 * N * 4,
                          sizeof(float) * (size_t)M * N, out));
            double dot = 0, na = 0, n2 = 0;
            for (size_t i = 0; i < (size_t)M * N; i++) {
                dot += (double)out[i] * ref[i];
                na += (double)out[i] * out[i];
                n2 += (double)ref[i] * ref[i];
            }
            const double cos = dot / sqrt(na * n2);
            if (!(cos > 0.9999999)) {
                printf("  MULTI MISMATCH tensor %d %s M=%d: cos %.6f  first gpu %.4f ref %.4f\n", t,
                       tiled ? "tiled" : "gemv", M, cos, out[0], ref[0]);
                fails++;
            }
            free(out); free(qa_h); free(asc_h); free(ref);
        }
        for (int t = 0; t < count; t++) { e2_q8w_free(&qs[t]); free(ws[t]); }
        e2_l0_graph_free(G, gr);
        e2_l0_free(G, wbuf); e2_l0_free(G, acts); e2_l0_free(G, xb); e2_l0_free(G, part);
        e2_l0_free(G, desc);
        free(x);
    }
    printf("multi-tensor check: %s\n", fails ? "FAILED" : "q|k|v style launches match too");
    return fails;
}

static int check_gemm(int gemm_mode) {
    int fails = 0;
    const int cases[][2] = { { 64, 512 }, { 32, 128 }, { 512, 768 }, { 1024, 256 }, { 2048, 512 } };
    for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
        const int N = cases[ci][0], K = cases[ci][1];
        float *w = malloc(sizeof(float) * (size_t)N * K);
        for (size_t i = 0; i < (size_t)N * K; i++) w[i] = (float)((int)(i * 7919 % 977) - 488) / 60.f;
        tdev t;
        memset(&t, 0, sizeof t);
        pack_weights(&t, w, N, K);
        const int nb = t.q.nblocks;
        for (int M = 1; M <= 40; M = M < 16 ? M + 1 : (M < 20 ? 21 : M + 13)) {
            const int tiled = M > E2_NM;
            const int mode = tiled ? gemm_mode : 0;
            float *x = malloc(sizeof(float) * (size_t)M * K);
            for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (float)((int)(i * 131 % 600) - 300) / 40.f;
            int8_t *qa_h = calloc((size_t)M * K, 1);
            float *asc_h = calloc((size_t)M * nb, sizeof(float));
            e2_i8_quantize(qa_h, asc_h, x, M, K, -INFINITY, INFINITY);
            float *ref = calloc((size_t)M * N, sizeof(float));
            e2_i8_gemm_scalar(ref, qa_h, asc_h, &t.q, M, N, 0, N);
            e2_graph *gr = record_case(&t, x, M, mode, 1);
            OK(e2_l0_graph_run(G, gr));
            float *out = malloc(sizeof(float) * (size_t)M * N);
            OK(e2_l0_read(G, (char *)t.acts + t.y_off, sizeof(float) * (size_t)M * N, out));
            double dot = 0, na = 0, n2 = 0, maxd = 0, maxr = 0;
            for (size_t i = 0; i < (size_t)M * N; i++) {
                dot += (double)out[i] * ref[i];
                na += (double)out[i] * out[i];
                n2 += (double)ref[i] * ref[i];
                double dd = fabs(out[i] - ref[i]);
                if (dd > maxd) maxd = dd;
                if (fabs(ref[i]) > maxr) maxr = fabs(ref[i]);
            }
            const double cos = dot / sqrt(na * n2), rel = maxd / maxr;
            if (!(cos > 0.9999999 && rel < 1e-5)) {
                printf("  MISMATCH %s N=%d K=%d M=%d: cos %.9f maxdiff/max %.3g\n",
                       mode == 3 ? "dpas-slms" : (mode == 2 ? "dpas" : (mode == 1 ? "tiled" : "gemv")), N, K, M, cos, rel);
                if (fails == 0) {
                    for (int i = 0; i < 6; i++) printf("      n=%d gpu %.4f ref %.4f\n", i, out[i], ref[i]);
                    /* are the inputs right? */
                    uint8_t *qg = malloc((size_t)M * K);
                    float *ag = malloc(sizeof(float) * (size_t)M * nb);
                    OK(e2_l0_read(G, (char *)t.acts + t.qa_off, (size_t)M * K, qg));
                    OK(e2_l0_read(G, (char *)t.acts + t.asc_off, sizeof(float) * (size_t)M * nb, ag));
                    int bad = 0, first = -1;
                    for (size_t i = 0; i < (size_t)M * K; i++)
                        if (qg[i] != (uint8_t)qa_h[i]) { bad++; if (first < 0) first = (int)i; }
                    printf("      quantized bytes differing: %d/%zu\n", bad, (size_t)M * K);
                    if (first >= 0) {
                        const int r = first / K;
                        int per_row[64] = { 0 };
                        for (size_t i = 0; i < (size_t)M * K; i++)
                            if (qg[i] != (uint8_t)qa_h[i]) per_row[i / K]++;
                        printf("      first at row %d k %d: gpu %d cpu %d  asc gpu %.8g cpu %.8g\n", r,
                               first % K, qg[first], (uint8_t)qa_h[first], ag[r * nb], asc_h[r * nb]);
                        printf("      per-row differing bytes:");
                        for (int i = 0; i < M && i < 64; i++) printf(" %d", per_row[i]);
                        printf("\n      k values differing in row %d:", r);
                        int shown = 0;
                        for (int k = 0; k < K && shown < 12; k++)
                            if (qg[r * K + k] != (uint8_t)qa_h[r * K + k]) {
                                printf(" k%d(g%d/c%d)", k, qg[r * K + k], (uint8_t)qa_h[r * K + k]);
                                shown++;
                            }
                        printf("\n      input x row %d:", r);
                        for (int k = 0; k < 6; k++) printf(" %.4f", x[r * K + k]);
                        printf("\n");
                    }
                    free(qg); free(ag);
                }
                fails++;
            }
            free(out); free(x); free(qa_h); free(asc_h); free(ref);
            e2_l0_graph_free(G, gr);
        }
        free_tdev(&t);
        free(w);
    }
    printf("kernel check (gemm mode %d): %s\n", gemm_mode,
           fails ? "FAILED" : "GEMMs match the scalar CPU implementation");
    return fails;
}

static void print_clock(void) {
    FILE *f = fopen("/sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt/gt0/rps_cur_freq_mhz", "r");
    if (!f) f = fopen("/sys/class/drm/card1/gt/gt0/rps_cur_freq_mhz", "r");
    if (!f) return;
    int mhz = 0;
    if (fscanf(f, "%d", &mhz) == 1) printf("[gpu clock %d MHz]\n", mhz);
    fclose(f);
}

static double bench_shape_q(tdev *t, int M, int reps, double *gbs, double *tmacs, int with_quant);

/* The iGPU idles at RP1 (800 MHz on this machine) and only boosts to RP0 (2350) after a few seconds
 * of sustained work, so a microbenchmark that packs weights on the CPU between runs measures the idle
 * clock and reports a third of the kernel's real speed. This keeps the GPU busy long enough to
 * boost before anything is timed. */
static void warmup(void) {
    float *w = malloc(sizeof(float) * (size_t)1024 * 512);
    for (size_t i = 0; i < (size_t)1024 * 512; i++)
        w[i] = (float)((int)(i * 7919 % 977) - 488) / 60.f;
    tdev t;
    memset(&t, 0, sizeof t);
    pack_weights(&t, w, 1024, 512);
    for (int i = 0; i < 600; i++) {
        double g, m;
        bench_shape_q(&t, 512, 1, &g, &m, 0);
    }
    free_tdev(&t);
    free(w);
    printf("warm-up done: ");
    print_clock();
}

static double bench_shape(tdev *t, int M, int reps, double *gbs, double *tmacs) {
    return bench_shape_q(t, M, reps, gbs, tmacs, 1);
}

static double bench_shape_q(tdev *t, int M, int reps, double *gbs, double *tmacs, int with_quant) {
    const int N = t->q.N, K = t->q.K;
    float *x = malloc(sizeof(float) * (size_t)M * K);
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (float)((i * 37 % 1000) - 500) / 250.f;
    const int mode = BENCH_MODE >= 0 ? BENCH_MODE : (M > E2_NM ? 2 : 0);
    e2_graph *gr = record_case(t, x, M, mode, with_quant);
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        OK(e2_l0_graph_run(G, gr));
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    double bytes = (double)t->wbytes + (double)M * K + (double)M * t->q.nblocks * 4 + (double)M * N * 4;
    *gbs = bytes / (best * 1e6);
    *tmacs = (double)M * N * K / (best * 1e6);
    e2_l0_graph_free(G, gr);
    free(x);
    return best;
}

int main(int argc, char **argv) {
    const int check_only = argc > 1 && !strcmp(argv[1], "--check");
    if (getenv("E2_BENCH_ATTN_GY")) ATTN_GY = atoi(getenv("E2_BENCH_ATTN_GY"));
    if (getenv("E2_BENCH_GEMM")) BENCH_MODE = atoi(getenv("E2_BENCH_GEMM"));
    if (getenv("E2_BENCH_GY")) BENCH_GY = atoi(getenv("E2_BENCH_GY"));
    e2_cpu_init();
    char err[4096];
    G = e2_l0_create(err, sizeof err);
    if (!G) { printf("Level Zero init failed:\n%s\n", err); return 1; }
    printf("device: %s | %u EU | %u MHz\n", e2_l0_device_name(G), e2_l0_eu_count(G), e2_l0_clock_mhz(G));
    if (check_attn()) return 1;
    if (check_gemm(1) || check_gemm(2) || check_gemm(3) || check_multi()) return 1;
    if (check_only) return 0;
    /* everything below is timed, so boost the GPU clock first (see warmup()) */
    warmup();
    /* E2_BENCH_FAST=1 skips the dispatch/bandwidth/attention probes: they are minutes of memory-heavy
     * work that makes SLPC drop the clock before the shape table is reached, which is exactly what a
     * tile sweep must not have between two configurations. */
    const int fast = getenv("E2_BENCH_FAST") != NULL;
    /* dispatch cost inside a recorded list: the number that decides how many kernels a graph may use */
    if (!fast) {
        void *wb = e2_l0_alloc(G, 4096);
        const int taps = 64;
        for (int bar = 0; bar < 2; bar++) {
            printf("2048 kernels %s barrier:", bar ? "with" : "no  ");
            double prev = 0;
            for (int K = 64; K <= 2048; K *= 4) {
                e2_graph *gr = e2_l0_graph_begin(G, bar);
                for (int i = 0; i < K; i++)
                    OK(e2_l0_append(G, gr, "k_bench_work", 8, 1, 1, 128, 1, 1, 2, &wb, sizeof wb, &taps,
                                    sizeof taps));
                OK(e2_l0_graph_close(G, gr));
                double best = 1e9;
                for (int r = 0; r < 3; r++) {
                    double t0 = now_ms();
                    OK(e2_l0_graph_run(G, gr));
                    if (now_ms() - t0 < best) best = now_ms() - t0;
                }
                /* the loop's last K is 1024 (64*4*4), so "the marginal cost" is printed there -
                 * comparing against 2048 never fired and the probe reported nothing */
                if (K * 4 > 2048)
                    printf(" %.2f us/kernel (marginal at K=%d)  ", (best - prev) / (K - K / 4) * 1e3, K);
                prev = best;
                e2_l0_graph_free(G, gr);
            }
            printf("\n");
        }
        e2_l0_free(G, wb);
    }
    /* bandwidth: reads and writes of device memory, at the sizes that separate L2 from DRAM. The
     * model's weight stream is a 1 MB-plus buffer read once per M-block, so the DRAM number is the
     * one that bounds the GEMM, and it is worth knowing where L2 stops helping. */
    if (!fast) {
        const size_t big = 2048u << 20;
        void *a = e2_l0_alloc(G, big), *b = e2_l0_alloc(G, 1u << 20);
        float fill = 1.f;
        e2_graph *gr = e2_l0_graph_begin(G, 0);
        OK(e2_l0_append(G, gr, "k_bench_write", (uint32_t)(big / 16 / 256), 1, 1, 256, 1, 1, 2, &a,
                        sizeof a, &fill, sizeof fill));
        OK(e2_l0_graph_close(G, gr));
        double t0 = now_ms();
        OK(e2_l0_graph_run(G, gr));
        printf("write 256 MB:       %7.2f ms (%5.1f GB/s)\n", now_ms() - t0,
               big / (now_ms() - t0) * 1e3 / 1e9);
        e2_l0_graph_free(G, gr);
                const int mbs[] = { 2, 8, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048 };
        for (size_t mi = 0; mi < sizeof mbs / sizeof mbs[0]; mi++) {
            const int mb = mbs[mi];
            const size_t sz = (size_t)mb << 20;
            const size_t n = sz / 16 / 256;
            char lbl[32];
            snprintf(lbl, sizeof lbl, "read %zu MB", sz >> 20);
            gr = e2_l0_graph_begin(G, 0);
            OK(e2_l0_append(G, gr, "k_bench_read", (uint32_t)n, 1, 1, 256, 1, 1, 2, &a, sizeof a, &b,
                            sizeof b));
            OK(e2_l0_graph_close(G, gr));
            double best = 1e9;
            for (int r = 0; r < 5; r++) {
                t0 = now_ms();
                OK(e2_l0_graph_run(G, gr));
                const double dt = now_ms() - t0;
                if (dt < best) best = dt;
            }
            printf("%-20s %10.3f %10.1f\n", lbl, best, sz / best * 1e3 / 1e9);
            e2_l0_graph_free(G, gr);
        }
        e2_l0_free(G, a);
        e2_l0_free(G, b);
    }
    /* attention on its own, at the shapes the text tower uses */
    if (!fast) {
        /* The shapes the tower uses: a document (one sequence, no clipping), the two head dimensions,
         * and the batch chunk. qstride is H*D and kvstride is KV*D, as gpu.c passes them. */
        struct { int rows, H, D, KV, qstride, kvstride, window, seq; } ac[] = {
            { 196, 4, 256, 2, 1024, 512, 512, 0 },   { 196, 4, 512, 1, 2048, 512, 0, 0 },
            { 960, 4, 256, 2, 1024, 512, 512, 0 },   { 960, 4, 512, 1, 2048, 512, 0, 0 },
            { 2048, 4, 256, 2, 1024, 512, 512, 0 },  { 1960, 4, 256, 2, 1024, 512, 512, 196 },
        };
        const size_t attn_bytes = (size_t)8192 * 2048 * 4;
        void *qb = e2_l0_alloc(G, attn_bytes), *kb = e2_l0_alloc(G, attn_bytes), *vb = e2_l0_alloc(G, attn_bytes);
        void *ctx = e2_l0_alloc(G, attn_bytes), *qa = e2_l0_alloc(G, attn_bytes), *asc = e2_l0_alloc(G, attn_bytes);
        {
            /* every row of every shape must land inside its buffer, or the probe measures a cache */
            void *z = calloc(1, attn_bytes);
            e2_l0_write(G, qb, attn_bytes, z); e2_l0_write(G, kb, attn_bytes, z);
            e2_l0_write(G, vb, attn_bytes, z); e2_l0_write(G, ctx, attn_bytes, z);
            e2_l0_write(G, qa, attn_bytes, z); e2_l0_write(G, asc, attn_bytes, z);
            free(z);
        }
        /* the engine always passes a mask, so the probe does too (all-ones: nothing is skipped) */
        void *valid = NULL;
        {
            unsigned char *vb = calloc(8192, 1);
            memset(vb, 1, 8192);
            valid = e2_l0_alloc(G, 8192);
            e2_l0_write(G, valid, 8192, vb);
            free(vb);
        }
        printf("\n%-34s %9s %9s %10s\n", "attention shape", "ms", "GMAC/s", "eff MAC/cyc");
        for (size_t i = 0; i < sizeof ac / sizeof ac[0]; i++) {
            const int rows = ac[i].rows, H = ac[i].H, D = ac[i].D, KV = ac[i].KV, win = ac[i].window;
            const int sq = ac[i].seq ? ac[i].seq : rows;
            e2_graph *gr = e2_l0_graph_begin(G, 1);
            int seql = ac[i].seq;
            OK(e2_l0_append(G, gr, "k_attn", (uint32_t)((rows + ATTN_GY - 1) / ATTN_GY), (uint32_t)H, 1,
                            32, (uint32_t)ATTN_GY, 1, 15,
                            &ctx, sizeof ctx, &qb, sizeof qb, &kb, sizeof kb, &vb, sizeof vb, &qa,
                            sizeof qa, &asc, sizeof asc, &valid, sizeof valid, &rows, sizeof rows, &H,
                            sizeof H, &D, sizeof D, &KV, sizeof KV, &ac[i].qstride,
                            sizeof ac[i].qstride, &ac[i].kvstride, sizeof ac[i].kvstride, &win,
                            sizeof win, &seql, sizeof seql));
            OK(e2_l0_graph_close(G, gr));
            double best = 1e9;
            for (int r = 0; r < 20; r++) {
                const double t0 = now_ms();
                OK(e2_l0_graph_run(G, gr));
                const double dt = now_ms() - t0;
                if (dt < best) best = dt;
            }
            double pairs = 0;
            for (int r = 0; r < rows; r++) {
                const int sb = ac[i].seq;
                const int pos = sb ? r % sb : r;
                const int slen = sb ? sb : rows;
                const int lo = (win > 0 && pos - win > 0) ? pos - win : 0;
                const int hi = (win > 0 && pos + win + 1 < slen) ? pos + win + 1 : slen;
                pairs += hi - lo;
            }
            const double macs = 2.0 * pairs * H * D;
            char lbl[64];
            snprintf(lbl, sizeof lbl, "rows=%d H=%d D=%d %s seq=%d", rows, H, D, win ? "sliding" : "full", ac[i].seq);
            printf("%-34s %9.3f %9.1f %10.3f\n", lbl, best, macs / best / 1e6,
                   macs / (best * 1e-3) / (128.0 * 2.35e9));
            e2_l0_graph_free(G, gr);
        }
        e2_l0_free(G, qb); e2_l0_free(G, kb); e2_l0_free(G, vb);
        e2_l0_free(G, ctx); e2_l0_free(G, qa); e2_l0_free(G, asc);
    }

    /* the matrix engine's ceiling for exactly this kernel structure: same tile, same loop shape, no
     * operand loads. Printing it next to the shape table says whether a shape is compute- or load- or
     * dispatch-bound instead of leaving it to inference; the three work-per-sub-group settings are
     * there because the real GEMMs run only 64-256 DPAS per sub-group (a document's K=512 shape is
     * 64), and the peak probe loses a factor of ~1.7 at that size to the sub-group's own overhead. */
    {
        const int nwg = 2048;
        void *out = e2_l0_alloc(G, (size_t)nwg * 4);
        REQ(out);
        printf("peak issue: k_bench_dpas, same tile and loop as k_gemm_dpas, no operand loads\n");
        for (int rs = 0; rs < 3; rs++) {
            const int reps = rs == 0 ? 1 : (rs == 1 ? 4 : 16);
            const int nb = 4; /* the real kernel's block count for K=512 */
            e2_graph *gr = e2_l0_graph_begin(G, 1);
            OK(e2_l0_append(G, gr, "k_bench_dpas", (uint32_t)nwg, 1, 1, 8, 4, 1, 3, &out, sizeof out,
                            &nb, sizeof nb, &reps, sizeof reps));
            OK(e2_l0_graph_close(G, gr));
            double best = 1e9;
            for (int r = 0; r < 5; r++) {
                const double t0 = now_ms();
                OK(e2_l0_graph_run(G, gr));
                const double dt = now_ms() - t0;
                if (dt < best) best = dt;
            }
            /* 128 is E2_BLOCK (the kernel's k-block, = the engine's quant block); 4 k-steps of 32 */
            const double dsp = (double)nwg * 4.0 * reps * nb * (128 / 32) * (E2_NTB * E2_NFB);
            printf("  %5d DPAS/sub-group: %8.3f ms  %7.1f GMAC/s  %5.1f G DPAS/s\n",
                   reps * nb * (128 / 32) * (E2_NTB * E2_NFB), best, dsp * 2048.0 / best / 1e6,
                   dsp / best / 1e6);
            e2_l0_graph_free(G, gr);
        }
        e2_l0_free(G, out);
    }
    struct { const char *what; int N, K; } shapes[] = {
        { "q_proj  sliding (1024x512)", 1024, 512 },
        { "q_proj  full    (2048x512)", 2048, 512 },
        { "o_proj  sliding (512x1024)", 512, 1024 },
        { "gate_proj         (2048x512)", 2048, 512 },
        { "down_proj         (512x2048)", 512, 2048 },
        { "ple_model      (12288x512)", 12288, 512 },
        { "vision mlp       (3072x768)", 3072, 768 },
    };
    printf("clock before shapes: ");
    print_clock();
    printf("\n%-30s %6s %10s %8s %9s %8s  %s\n", "shape", "M", "ms", "GB/s", "TMAC/s", "kernel",
           "(quant+gemm | gemm alone)");
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        const int N = shapes[s].N, K = shapes[s].K;
        float *w = malloc(sizeof(float) * (size_t)N * K);
        for (size_t i = 0; i < (size_t)N * K; i++) w[i] = (float)((int)(i * 7919 % 977) - 488) / 60.f;
        tdev t;
        memset(&t, 0, sizeof t);
        pack_weights(&t, w, N, K);
        static int Ms[64] = { 1, 8, 16, 32 };
        int nm = 4;
        if (getenv("E2_BENCH_MS")) {
            char *sp = strdup(getenv("E2_BENCH_MS")), *sv;
            nm = 0;
            for (char *tok = strtok_r(sp, ",", &sv); tok && nm < 64; tok = strtok_r(NULL, ",", &sv))
                Ms[nm++] = atoi(tok);
            free(sp);
        }
        for (int mi = 0; mi < nm; mi++) {
            double gbs, tmacs, gbs2, tmacs2;
            const int M = Ms[mi];
            const double ms = bench_shape(&t, M, 200, &gbs, &tmacs);
            const double ms2 = bench_shape_q(&t, M, 200, &gbs2, &tmacs2, 0);
            const int bm = BENCH_MODE >= 0 ? BENCH_MODE : (M > E2_NM ? 2 : 0);
            printf("%-30s %6d %10.3f %8.1f %9.2f %s  | gemm only %10.3f %8.1f %9.2f\n", shapes[s].what,
                   M, ms, gbs, tmacs, bm == 3 ? "slms" : (bm == 2 ? "dpas" : (bm == 1 ? "tiled" : "gemv")), ms2, gbs2, tmacs2);
        }
        free_tdev(&t);
        free(w);
    }
    e2_l0_destroy(G);
    return 0;
}
