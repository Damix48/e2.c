/* e2_kernel_bench.c — per-kernel throughput per CPU backend.
 *
 * The CPU counterpart of tools/bench/hip_kernel_bench.hip: answers "which kernel is slow", which the
 * engine-level benches cannot. The weight GEMM shape is HARDWARE.md's 64x2048x512; the attention
 * shapes are the vision tower's (64 rows x 2304 keys x 64 deep, then the 64-deep context product over
 * 2304 keys). Kernels that need a quantizer are skipped on the scalar backend, where the engine does
 * not run int8 at all.
 *
 * usage: e2_kernel_bench cpu|backends reps
 *   cpu:      the default backend, or the one E2_CPU_BACKEND forces
 *   backends: every backend this host can execute, one JSON line set each
 *
 * Pin it to one core (`taskset -c 0` on a hybrid x86, `E2_THREADS` does not apply here): on Arrow
 * Lake the same vnni GEMM measured 63 GMAC/s unpinned and 229.4 GMAC/s pinned, because the first run
 * landed on an E-core. That is the whole difference between HARDWARE.md's table and a number that
 * looks like a regression.
 *
 * One JSON line per (backend, kernel): {"backend","kernel","m","n","k","best_ms","rate","unit",
 * "check"}. `check` compares against the scalar reference where one is exported (the weight GEMM and
 * the fp32 GEMM); the attention GEMM has no exported reference, so its line says "skip".
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dispatch.h"
#include "int8.h"
#include "kernels.h"

static uint32_t g_seed = 12345u;
static uint32_t rnd(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed;
}
static float frand(float scale) { return ((int32_t)(rnd() >> 8) % 2001 - 1000) / 1000.0f * scale; }

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static const char *active_name(void) {
    switch (e2_cpu_backend()) {
    case E2_BK_CPU_SCALAR: return "scalar";
    case E2_BK_CPU_AVX2: return "avx2";
    case E2_BK_CPU_VNNI: return "vnni";
    case E2_BK_CPU_AVX512: return "avx512";
    case E2_BK_CPU_NEON: return "neon";
    default: return "?";
    }
}

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double d = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)a[i] - b[i]);
        if (e > d) d = e;
    }
    return d;
}

static void report(const char *kernel, int m, int n, int k, double ms, double rate, const char *unit,
                   const char *check) {
    printf("{\"backend\":\"%s\",\"kernel\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"best_ms\":%.6f,"
           "\"rate\":%.1f,\"unit\":\"%s\",\"check\":\"%s\"}\n",
           active_name(), kernel, m, n, k, ms, rate, unit, check);
}

static double rate_gmac(int m, int n, int k, double ms) {
    return (double)m * n * k / 1e9 / (ms / 1e3);
}
static double rate_melem(size_t count, double ms) { return (double)count / 1e6 / (ms / 1e3); }

static void run_all(int reps) {
    const int M = 64, N = 512, K = 2048; /* HARDWARE.md's 64x2048x512 weight GEMM */
    const int nblocks = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    float *wsrc = malloc(sizeof *wsrc * (size_t)N * K);
    float *x = malloc(sizeof *x * (size_t)M * K);
    float *C = malloc(sizeof *C * (size_t)M * N);
    float *ref = malloc(sizeof *ref * (size_t)M * N);
    int8_t *qa = NULL;
    float *asc = malloc(sizeof *asc * (size_t)M * nblocks);
    float *A = malloc(sizeof *A * (size_t)M * K);
    float *B = malloc(sizeof *B * (size_t)K * N);
    if (!wsrc || !x || !C || !ref || !asc || !A || !B) { free(wsrc); free(x); free(C); free(ref); free(asc); free(A); free(B); return; }
    for (int i = 0; i < N * K; i++) wsrc[i] = frand(0.05f);
    for (int i = 0; i < M * K; i++) x[i] = frand(2.0f);
    for (int i = 0; i < M * K; i++) A[i] = frand(2.0f);
    for (int i = 0; i < K * N; i++) B[i] = frand(0.05f);

    /* ---- int8 weight GEMM (needs the quantizer, which the scalar backend does not have) ---- */
    if (e2_i8_gemm && e2_i8_quantize) {
        e2_q8w w;
        memset(&w, 0, sizeof w);
        qa = malloc((size_t)M * K);
        if (!qa || e2_q8w_from_f32(&w, wsrc, N, K)) {
            e2_q8w_free(&w);
            free(qa);
            goto done;
        }
        e2_i8_quantize(qa, asc, x, M, K, -INFINITY, INFINITY);
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t = now_ms();
            e2_i8_gemm(C, qa, asc, &w, M, N, 0, N);
            double d = now_ms() - t;
            if (d < best) best = d;
        }
        e2_i8_gemm_scalar(ref, qa, asc, &w, M, N, 0, N);
        report("i8_gemm", M, N, K, best, rate_gmac(M, N, K, best), "gmac_s",
               max_abs_diff(C, ref, (size_t)M * N) < 1e-2 ? "ok" : "MISMATCH");

        /* the packer is the text attention crossover's bottleneck; it runs at load and per head */
        best = 1e30;
        for (int r = 0; r < reps; r++) {
            e2_q8w p;
            memset(&p, 0, sizeof p);
            double t = now_ms();
            int rc = e2_q8w_from_f32(&p, wsrc, N, K);
            double d = now_ms() - t;
            e2_q8w_free(&p);
            if (rc) break;
            if (d < best) best = d;
        }
        report("q8w_from_f32", N, K, 0, best, rate_melem((size_t)N * K, best), "melem_s", "skip");

        best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t = now_ms();
            e2_i8_quantize(qa, asc, x, M, K, -INFINITY, INFINITY);
            double d = now_ms() - t;
            if (d < best) best = d;
        }
        report("quantize_blocks", M, K, 0, best, rate_melem((size_t)M * K, best), "melem_s", "skip");
        e2_q8w_free(&w);
    }

    /* ---- fp32 GEMM (the attention products on the non-int8 paths) ---- */
    {
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t = now_ms();
            e2_sgemm_nn_range(C, N, A, K, B, N, M, N, K, 0, M);
            double d = now_ms() - t;
            if (d < best) best = d;
        }
        e2_sgemm_nn_range_scalar(ref, N, A, K, B, N, M, N, K, 0, M);
        report("sgemm_nn_range", M, N, K, best, rate_gmac(M, N, K, best), "gmac_s",
               max_abs_diff(C, ref, (size_t)M * N) < 1e-2 * (float)K ? "ok" : "MISMATCH");
    }

    /* ---- the packed-operand attention GEMM, both products ---- */
    if (e2_i8_attn_gemm) {
        const int AM = 64, AKEYS = 2304, AD = 64;
        const int nbl = (AKEYS + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
        const int Np = (AKEYS + 15) / 16 * 16;
        uint8_t *act = malloc((size_t)AM * AKEYS);
        uint8_t *Bp = malloc((size_t)AKEYS * Np);
        float *bsc = malloc(sizeof *bsc * (size_t)AKEYS);
        int32_t *bz = malloc(sizeof *bz * (size_t)nbl * AKEYS);
        float *ac = malloc(sizeof *ac * (size_t)AM * AKEYS);
        float *asc2 = malloc(sizeof *asc2 * (size_t)AM * nbl);
        if (!act || !Bp || !bsc || !bz || !ac || !asc2) {
            free(act); free(Bp); free(bsc); free(bz); free(ac); free(asc2);
            goto done;
        }
        for (size_t i = 0; i < (size_t)AKEYS * AKEYS; i++) Bp[i] = (uint8_t)rnd();
        for (size_t i = 0; i < (size_t)AM * AKEYS; i++) act[i] = (uint8_t)rnd();
        for (size_t i = 0; i < (size_t)AM * nbl; i++) asc2[i] = 0.01f;
        for (int i = 0; i < AKEYS; i++) bsc[i] = 0.01f;
        memset(bz, 0, sizeof *bz * (size_t)nbl * AKEYS);
        /* logits: 64 rows x 2304 keys, 64 deep */
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t = now_ms();
            e2_i8_attn_gemm(ac, AKEYS, act, AD, asc2, Bp, bsc, bz, AM, AKEYS, AKEYS, AD);
            double d = now_ms() - t;
            if (d < best) best = d;
        }
        report("attn_gemm_logits", AM, AKEYS, AD, best, rate_gmac(AM, AKEYS, AD, best), "gmac_s", "skip");
        /* context: 64 rows x 64 deep, 2304 keys */
        best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t = now_ms();
            e2_i8_attn_gemm(ac, 64, act, AKEYS, asc2, Bp, bsc, bz, AM, 64, 64, AKEYS);
            double d = now_ms() - t;
            if (d < best) best = d;
        }
        report("attn_gemm_context", AM, 64, AKEYS, best, rate_gmac(AM, 64, AKEYS, best), "gmac_s", "skip");
        free(act); free(Bp); free(bsc); free(bz); free(ac); free(asc2);
    }

done:
    free(wsrc); free(x); free(C); free(ref); free(asc); free(A); free(B); free(qa);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s cpu|backends reps\n", argv[0]);
        return 2;
    }
    const int sweep = strcmp(argv[1], "backends") == 0;
    const int reps = atoi(argv[2]);
    if ((!sweep && strcmp(argv[1], "cpu")) || reps <= 0 || reps > 10000) return 2;

    if (!sweep) {
        e2_cpu_init();
        run_all(reps);
        return 0;
    }
    char previous[32] = { 0 };
    const char *p = getenv("E2_CPU_BACKEND");
    if (p && strlen(p) < sizeof previous) strcpy(previous, p);
    for (int i = 0; i < e2_cpu_backend_count(); i++) {
        const char *name = e2_cpu_backend_name(i);
        if (!name) continue; /* this host cannot execute it */
        if (setenv("E2_CPU_BACKEND", name, 1)) return 2;
        e2_cpu_init();
        run_all(reps);
    }
    if (previous[0]) setenv("E2_CPU_BACKEND", previous, 1);
    else unsetenv("E2_CPU_BACKEND");
    return 0;
}
