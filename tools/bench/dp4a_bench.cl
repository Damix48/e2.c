/* dp4a_bench.cl — standalone copy of k_gemm_tiled from src/gpu/kernels/e2.cl.
 *
 * Not part of the engine build. It measures the DP4A projection GEMM on a GPU without
 * XMX/DPAS (the N100's Xe-LP, for example), to decide whether a DP4A-only backend
 * variant is worth writing. The kernel body is intentionally identical to e2.cl;
 * e2_tile.h supplies the tile shape the host also uses for the NDRange.
 */
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define E2_BLOCK 128
#define E2_DS_STRIDE 16

#include "e2_tile.h"

static inline int e2_dp4a(uchar4 a, char4 b) { return dot(a, b); }

/* The comment block in e2.cl explains the layout and why local_id(0) is N. Kept here verbatim
 * enough that the two copies can be diffed by eye. */
__kernel void k_gemm_tiled(__global const char *wbuf, __global const uchar *acts,
                           __global const int *desc, int M) {
    const int t = get_global_id(2);
    __global const int *dd = desc + t * E2_DS_STRIDE;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int m0 = get_group_id(0) * E2_BM + ty * E2_GEMM_TM;
    const int n0 = get_group_id(1) * E2_BN + tx * E2_GEMM_TN;
    if (m0 >= M || n0 >= N) return;
    const int nlim = (n0 + E2_GEMM_TN) < N ? E2_GEMM_TN : N - n0;
    __global const char *w = wbuf + dd[0] + (size_t)n0 * 4;
    __global const float *wsc = (__global const float *)(wbuf + dd[1]);
    __global const int *bsq = (__global const int *)(wbuf + dd[2]);
    const __global uchar *qa = acts + dd[7];
    __global const float *asc = (__global const float *)(acts + dd[8]);
    __global float *y = (__global float *)((__global char *)acts + dd[3]);
    float wsv[E2_GEMM_TN];
#pragma unroll
    for (int j = 0; j < E2_GEMM_TN; j++) wsv[j] = wsc[n0 + j];
    float yacc[E2_GEMM_TM][E2_GEMM_TN];
#pragma unroll
    for (int i = 0; i < E2_GEMM_TM; i++)
#pragma unroll
        for (int j = 0; j < E2_GEMM_TN; j++) yacc[i][j] = 0.f;
    for (int b = 0; b < nb; b++) {
        const int kb = b * E2_BLOCK;
        int acc[E2_GEMM_TM][E2_GEMM_TN];
#pragma unroll
        for (int i = 0; i < E2_GEMM_TM; i++)
#pragma unroll
            for (int j = 0; j < E2_GEMM_TN; j++) acc[i][j] = 0;
#pragma unroll
        for (int kk = 0; kk < E2_BLOCK; kk += 4) {
            const int k = kb + kk;
            uchar4 av[E2_GEMM_TM];
#pragma unroll
            for (int i = 0; i < E2_GEMM_TM; i++)
                av[i] = *(__global const uchar4 *)(qa + (size_t)(m0 + i) * K + k);
            const char16 w16 = *(__global const char16 *)(w + (size_t)(k >> 2) * N * 4);
#pragma unroll
            for (int j = 0; j < E2_GEMM_TN; j++) {
                const char4 wv = j == 0 ? w16.s0123 : j == 1 ? w16.s4567 : j == 2 ? w16.s89ab : w16.scdef;
#pragma unroll
                for (int i = 0; i < E2_GEMM_TM; i++) acc[i][j] += e2_dp4a(av[i], wv);
            }
        }
#pragma unroll
        for (int i = 0; i < E2_GEMM_TM; i++)
#pragma unroll
            for (int j = 0; j < E2_GEMM_TN; j++) {
                const int corr = 128 * bsq[(size_t)b * N + (n0 + j)];
                yacc[i][j] += asc[(size_t)(m0 + i) * nb + b] * wsv[j] * (float)(acc[i][j] - corr);
            }
    }
#pragma unroll
    for (int i = 0; i < E2_GEMM_TM; i++)
        if (m0 + i < M)
#pragma unroll
            for (int j = 0; j < E2_GEMM_TN; j++)
                if (j < nlim) y[(size_t)(m0 + i) * N + (n0 + j)] = yacc[i][j];
}
