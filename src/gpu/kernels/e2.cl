/* e2.cl — GPU kernels for the e2 engine (Intel GPU, Level Zero).
 *
 * Quantization is identical to the CPU engine (src/int8.h) so both produce the same numbers: int8
 * weights with a per-output-row scale (wsc) and a per-128-block sum (bsq), int8 activations with a
 * per-128-block scale (asc) stored biased by +128, so a dot product is a plain dp4a(uchar, char):
 *
 *     y[m,n] = sum_b  asc[m,b] * wsc[n] * ( dp4a_block(m,n,b) - 128 * bsq[b,n] )
 *
 * The weight layout is the GPU's own, not the CPU's quad-interleaved one. The k-quads of a row are
 * contiguous:
 *
 *     Wg[(k/4)*N*4 + n*4 + (k&3)]
 *
 * so one k-quad of every row is a contiguous run. A sub-group of 32 lanes reading the same k-quad of
 * 32 consecutive rows is then one 128-byte access, and a 4x4 register tile loads its four weight rows
 * (four k values each) with a single 16-byte load. The CPU layout was picked for vpdpbusd; this one
 * for 128-byte accesses and low instruction counts.
 *
 * Projections are described by an int32 descriptor array (8 entries per tensor) so that one launch can
 * run several projections sharing an input - which is what q|k|v and gate|up are:
 *
 *     desc[t*16+0..2] = byte offsets of w, wsc, bsq inside the weight buffer
 *     desc[t*16+3]    = byte offset of the output y inside the activation buffer
 *     desc[t*16+4..6] = N, K, block count
 *     desc[t*16+7..8] = byte offsets of the input activations and their scales
 *
 * The input and the output of a projection are different regions (the input is the previous kernel's
 * quantized output), which is exactly the distinction a single offset cannot express.
 */
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_intel_subgroup_matrix_multiply_accumulate
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#endif

#define E2_BLOCK 128
#define E2_NM 16 /* tokens the short-sequence GEMM handles (see k_gemv_split) */
#define E2_DS_STRIDE 16 /* int32 per projection descriptor (gpu.h) */

static inline int e2_dp4a(uchar4 a, char4 b) { return dot(a, b); }
static inline float e2_bf16(ushort v) { return as_float((uint)v << 16); }
static inline float4 e2_bf16x4(const __global ushort *p) {
    return (float4)(e2_bf16(p[0]), e2_bf16(p[1]), e2_bf16(p[2]), e2_bf16(p[3]));
}

/* sub-group butterflies (cl_intel_subgroups) */
static inline float e2_sg_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, intel_sub_group_shuffle_xor(v, o));
    return v;
}
static inline float e2_sg_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += intel_sub_group_shuffle_xor(v, o);
    return v;
}

/* ------------------------------------------------------------------ token ids -> embeddings
 * Gathers the rows this call needs out of the bf16 token table and scales them, which is what the
 * CPU engine's e2_embed_token does. Keeps 268 MB of table out of the per-call host traffic. */
__kernel void k_embed(__global float *x, __global const ushort *table, __global const int *ids,
                      __global const uchar *valid, int rows, int hidden, float scale) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int d = (get_group_id(1) * get_local_size(0) + get_local_id(0)) * 4;
    if (row >= rows || d >= hidden) return;
    __global float *dst = x + (size_t)row * hidden + d;
    if (valid && !valid[row]) {
        *(__global float4 *)dst = (float4)(0.f, 0.f, 0.f, 0.f);
        return;
    }
    *(__global float4 *)dst = e2_bf16x4(table + (size_t)ids[row] * hidden + d) * scale;
}

/* ------------------------------------------------------------------ int8 GEMM for M <= 16
 * One work item per (output row, k-slice). The k dimension is split `nsl` ways so that small N (the
 * 512-row projections are half of this model) still fills 128 EUs: a one-lane-per-row GEMV with no
 * split leaves the machine at 12 GB/s because too little memory-level parallelism is in flight.
 *
 * The inner loop walks 16 k values (four k-quads) at a time: the four weight quads of the row are four
 * loads and the sixteen activation values of a token are one 16-byte load, so the ratio is 4 dp4a per
 * 2 loads with 16 independent token chains - the earlier 4-quads-at-a-time version issued 8 loads for
 * 4 dp4a and ran at 8 GB/s, which over 24 layers is the whole inference. The per-block zero-point
 * correction applies to its own accumulator, so no cross-lane reduction over k is needed; the slices
 * are summed by k_gemv_reduce.
 *
 * The token loop runs all E2_NM rows unguarded. Writing the `m < M` guard on the accumulator made IGC
 * emit a compare, a predicated dp4a into a scratch and an extra add per element - 40+ instructions and
 * ~1.1 MAC per instruction for a 64 MAC step. Rows M..E2_NM-1 read quantized bytes that no live row
 * produced and their partials are never reduced (k_gemv_reduce uses the real M), so they cost nothing
 * but a few wasted lanes.
 *
 * The k loop inside a block has a compile-time trip count (E2_BLOCK/16), which is what keeps IGC from
 * predicating it and from splitting each accumulate into a scratch dp4a plus an add; that needs K to be
 * a multiple of E2_BLOCK, which every K in this model is and which gpu.c checks at load. */
__kernel void k_gemv_split(__global const char *wbuf, __global const uchar *acts,
                           __global float *part, __global const int *desc, int nsl, int M, int nstride) {
    const int t = get_global_id(2);
    const int sl = get_global_id(1);
    if (sl >= nsl) return;
    __global const int *dd = desc + t * E2_DS_STRIDE;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int n = get_global_id(0);
    if (n >= N) return;
    const int b0 = (nb * sl) / nsl, b1 = (nb * (sl + 1)) / nsl;
    __global const char *w = wbuf + dd[0];
    __global const float *wsc = (__global const float *)(wbuf + dd[1]);
    __global const int *bsq = (__global const int *)(wbuf + dd[2]);
    const __global uchar *qa = acts + dd[7];
    __global const float *asc = (__global const float *)(acts + dd[8]);
    const float ws = wsc[n];
    float yacc[E2_NM];
#pragma unroll
    for (int m = 0; m < E2_NM; m++) yacc[m] = 0.f;
    for (int b = b0; b < b1; b++) {
        const int kb = b * E2_BLOCK;
        int acc[E2_NM];
#pragma unroll
        for (int m = 0; m < E2_NM; m++) acc[m] = 0;
#pragma unroll
        for (int kk = 0; kk < E2_BLOCK; kk += 16) {
            const int k = kb + kk;
            const int q = k >> 2;
            const char4 w0 = *(__global const char4 *)(w + ((size_t)q * N + n) * 4);
            const char4 w1 = *(__global const char4 *)(w + ((size_t)(q + 1) * N + n) * 4);
            const char4 w2 = *(__global const char4 *)(w + ((size_t)(q + 2) * N + n) * 4);
            const char4 w3 = *(__global const char4 *)(w + ((size_t)(q + 3) * N + n) * 4);
#pragma unroll
            for (int m = 0; m < E2_NM; m++) {
                const uchar16 av = *(__global const uchar16 *)(qa + (size_t)m * K + k);
                /* four separate accumulations, not a sum of four dots: IGC turns `a += d4+d4+d4+d4`
                 * into four predicated dp4a into a scratch plus four adds, and a plain `+=` into one */
                acc[m] += e2_dp4a(av.s0123, w0);
                acc[m] += e2_dp4a(av.s4567, w1);
                acc[m] += e2_dp4a(av.s89ab, w2);
                acc[m] += e2_dp4a(av.scdef, w3);
            }
        }
        const int corr = 128 * bsq[(size_t)b * N + n];
#pragma unroll
        for (int m = 0; m < E2_NM; m++) yacc[m] += asc[(size_t)m * nb + b] * ws * (float)(acc[m] - corr);
    }
    /* E2_NM stride, not M: the partial buffer is laid out for a full 16-row tile whatever M is, so
     * every slice writes the same span and two slices cannot race on rows M..E2_NM-1. */
    __global float *p = part + ((size_t)t * nsl + sl) * E2_NM * nstride;
#pragma unroll
    for (int m = 0; m < E2_NM; m++) p[(size_t)m * nstride + n] = yacc[m];
}

__kernel void k_gemv_reduce(__global float *acts, __global const float *part,
                            __global const int *desc, int nsl, int M, int nstride) {
    const int t = get_global_id(1);
    const int N = desc[t * E2_DS_STRIDE + 4];
    const int n = get_global_id(0);
    if (n >= N) return;
    __global float *y = (__global float *)((__global char *)acts + desc[t * E2_DS_STRIDE + 3]);
    for (int m = 0; m < M; m++) {
        float s = 0.f;
        for (int sl = 0; sl < nsl; sl++)
            s += part[((size_t)t * nsl + sl) * E2_NM * nstride + (size_t)m * nstride + n];
        y[(size_t)m * N + n] = s;
    }
}

/* ------------------------------------------------------------------ int8 GEMM for M > E2_NM
 * Tiled over M and N with a 4x4 register tile per work item. The four weight rows of a tile step are
 * 16 contiguous bytes in this layout, so one char16 load feeds the whole k-quad.
 *
 * The k loop is unguarded, and that is the whole difference between this kernel and a much slower one:
 * with `if (i < mlim && j < nlim)` around the accumulate, IGC emits a compare, a predicated dp4a into a
 * scratch register and a separate add for every one of the 16 products - roughly 55 instructions and
 * 1.1 MAC per instruction for a 64 MAC step, which capped this kernel at ~0.7 TMAC/s whatever the
 * shape. Unguarded it is 5 loads and 16 dp4a, ~3.05 MAC per instruction.
 *
 * The cost is that the kernel needs the caller to keep its padding promises:
 *   - M is rounded up to E2_GEMM_TM (gpu.c passes ceil(rows/E2_BM)*E2_BM, gpu_bench does the same).
 *     Rows M..m0+E2_GEMM_TM-1 read activation bytes belonging to whatever region follows, which is why
 *     the activation buffer carries a tail pad (see layout() in gpu.c); the results are never stored.
 *   - N is a multiple of E2_GEMM_TN. Every projection in this model is (512, 768, 1024, 2048, 3072,
 *     12288), and a weight row block is quad-major, so a load that runs past row N-1 lands on the next
 *     k-quad of the same tensor rather than out of the buffer.
 * The store is still guarded, but once, outside the k loop, where it is free. */
#include "e2_tile.h"

__kernel void k_gemm_tiled(__global const char *wbuf, __global const uchar *acts,
                           __global const int *desc, int M) {
    const int t = get_global_id(2);
    __global const int *dd = desc + t * E2_DS_STRIDE;
    const int N = dd[4], K = dd[5], nb = dd[6];
    /* tx is local_id(0) on purpose: IGC assigns the lanes of a SIMD8 to consecutive local_id(0)
     * values, so with tx fastest the eight lanes of one instruction share a ty, which makes the four
     * activation loads a broadcast (one line) instead of eight scattered rows (eight lines). The weight
     * load goes the other way - eight lanes reading 16 contiguous bytes each - which is what the
     * quad-major layout was built for. */
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

/* ------------------------------------------------------------------ int8 GEMM on the XMX (DPAS) engine
 * The dot-product kernels above are dp4a; this one uses the matrix engine
 * (intel_sub_group_u8_i8_matrix_mad_k32): one sub-group of 8 lanes computes an E2_DM x E2_DN tile,
 * one DPAS per 8 rows x 8 columns x 32 k. Sub-group size 8 is required by the builtin, not chosen.
 *
 * The int8 recipe is unchanged (same per-row weight scales, same per-128-block activation scales and
 * zero-points). DPAS has no zero-point input, and that is why the operand types are the mixed signed
 * and unsigned variant: the activation byte is stored as q = a + 128, so s8(w) x u8(q) gives
 * sum(w*a) + 128*sum(w), and 128*sum(w) is exactly the per-block weight sum (bsq) the dot-product path
 * already subtracts.
 *
 * Operand layout is the ISA's (documentation/visa/instructions/DPAS.md). With exec size 8 and s8:
 *   - A (src2, 8 x 32): lane l, dword r = A[row r][k = 4l .. 4l+3]. The weights are the A operand, and
 *     q8w_dpas() in gpu.c lays them out so lane l's eight dwords are one 32-byte run: element r is
 *     W[n0+r][4l..4l+3]. One k-step's eight runs are stored lanes-contiguous (E2_DPAS_OFF in
 *     e2_tile.h), so the load's eight per-lane addresses are 32 bytes apart and coalesce; the older
 *     quad-major order put them N*4 bytes apart. Loading it is a single 32-byte vector load, which is
 *     what matters - eight 4-byte loads per operand cost more than the DPAS they feed (measured: 6
 *     loads per DPAS ran at 2.5 TMAC/s, the same kernel with one load per operand runs at 12).
 *   - B (src1, 32 x 8): lane n holds one token, dword d = X[token n][4d .. 4d+3], as unsigned bytes. A
 *     row-major int8 activation is exactly that, so the B operand is also one 32-byte vector load.
 *   - D: lane n, dword r = C[feature r][token n].
 *
 * The accumulator is integer and per 128-block, because the activation scale changes per block: after
 * each block the four k-steps are folded into the fp32 accumulator with asc[m,b] * wsc[n]. That
 * epilogue is the reason the kernel is E2_DM=32 rather than 128 rows: one fp32 accumulator per output
 * element is what the tile costs in registers.
 *
 * M is the caller's responsibility: the activation loads are unguarded over the E2_DM tile, so the
 * caller pads M to E2_DM and the stores are guarded on the real M. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_gemm_dpas(
    __global const char *wbuf, __global const uchar *acts, __global const int *desc, int M) {
    const int t = get_global_id(2);
    __global const int *dd = desc + t * E2_DS_STRIDE;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int m0 = get_group_id(0) * E2_DM;                                    /* first token */
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * E2_DN;         /* first feature */
    if (n0 >= N) return;
    __global const uint *wt = (__global const uint *)(wbuf + dd[9]); /* DPAS A layout, 32B aligned */
    __global const float *wsc = (__global const float *)(wbuf + dd[1]);
    __global const int *bsq = (__global const int *)(wbuf + dd[2]);
    __global const uchar *qa = acts + dd[7];
    __global const float *asc = (__global const float *)(acts + dd[8]);
    __global float *y = (__global float *)((__global char *)acts + dd[3]);
    const int nfb0 = n0 >> 3; /* first feature block, N is a multiple of 8 */
    const int nqb = N >> 3;   /* feature blocks per k-quad */

    float yacc[E2_NTB][E2_NFB][8];
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++)
#pragma unroll
            for (int r = 0; r < 8; r++) yacc[tc][fb][r] = 0.f;

    for (int b = 0; b < nb; b++) {
        const int kb = b * E2_BLOCK;
        int8 acc[E2_NTB][E2_NFB];
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) acc[tc][fb] = (int8)(0);
#pragma unroll
        for (int kw = 0; kw < E2_BLOCK; kw += 32) {
            const int k = kb + kw;
            /* A: lane l covers the k-quad (k/4 + l) of every feature block */
            int8 av[E2_NFB];
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                /* the k-step's 256-byte run, lane l's own 32 bytes inside it (E2_DPAS_OFF; the quads
                 * were N*4 = nqb*32 bytes apart in the layout this replaced) */
                av[fb] = *(__global const int8 *)(wt +
                    ((size_t)(k >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 64 + (size_t)lane * 8);
#pragma unroll
            for (int tc = 0; tc < E2_NTB; tc++) {
                /* B: lane n is token (m0 + tc*8 + n), 32 contiguous quantized bytes of its row */
                const uint8 bv = *(__global const uint8 *)(qa + (size_t)(m0 + tc * 8 + lane) * K + k);
#pragma unroll
                for (int fb = 0; fb < E2_NFB; fb++)
                    acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
            }
        }
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++) {
            const int row = m0 + tc * 8 + lane;
            const float a = asc[(size_t)row * nb + b];
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    const int n = n0 + fb * 8 + r;
                    yacc[tc][fb][r] += a * wsc[n] * (float)(acc[tc][fb][r] - 128 * bsq[(size_t)b * N + n]);
                }
        }
    }
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++) {
        const int row = m0 + tc * 8 + lane;
        if (row < M)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
#pragma unroll
                for (int r = 0; r < 8; r++) y[(size_t)row * N + n0 + fb * 8 + r] = yacc[tc][fb][r];
    }
}

/* The projections' GEMM, second cut. Same arithmetic, same DPAS tile and same operand layout as
 * k_gemm_dpas above; the one change is the epilogue. The per-feature weight scale and the per-128-key
 * zero-point sum are eight contiguous values each, but the first version loaded them one element at a
 * time and accumulated in scalar fp32 - 32 loads and 32 FMAs per feature block per k-block, against
 * the k-loop's own 16 loads. One 32-byte vload8 each and a float8 accumulate is the same change that
 * took k_vattn_scores from 150 to 50 ms when the vision tower landed; it had never been carried over
 * here. Measured by alternating binaries in one session: document 14.1 vs 15.1 ms, 8x17 batch 9.7 vs
 * 10.0 ms, 32x196 batch 406 vs 449 ms, image neutral-to-better.
 *
 * A rewrite that hoisted the operand addresses into base pointers with 32-bit offsets was tried and is
 * *not* here: it measured worse on the vision shapes (image 257.5 vs 256.2 ms) and no better on text.
 * The disassembly is mostly mov and address arithmetic around the dpas, but the compiler already
 * schedules that better than a hand-rolled pointer (see engine/README.md, the rejected pointer form).
 *
 * The wrappers specialize the token tile to 8 (short text) or 16 rows (all other uses). */
static inline void e2_gemm_dpas2(
    __global const char *wbuf, __global const uchar *acts, __global const int *desc, int M, int ntb) {
    const int t = get_global_id(2);
    __global const int *dd = desc + t * E2_DS_STRIDE;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int m0 = get_group_id(0) * (ntb * 8);
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * E2_DN;
    if (n0 >= N) return;
    const int nfb0 = n0 >> 3;
    const int nqb = N >> 3;
    __global const uint *wt = (__global const uint *)(wbuf + dd[9]); /* DPAS A layout */
    __global const uchar *qa = acts + dd[7];
    __global const float *wsc = (__global const float *)(wbuf + dd[1]);
    __global const int *bsq = (__global const int *)(wbuf + dd[2]);
    __global const float *asc = (__global const float *)(acts + dd[8]);
    __global float *y = (__global float *)((__global char *)acts + dd[3]);

    float8 yacc[E2_NTB][E2_NFB];
#pragma unroll
    for (int tc = 0; tc < ntb; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) yacc[tc][fb] = (float8)(0.f);

    for (int b = 0; b < nb; b++) {
        const int kb = b * E2_BLOCK;
        int8 acc[E2_NTB][E2_NFB];
#pragma unroll
        for (int tc = 0; tc < ntb; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) acc[tc][fb] = (int8)(0);
#pragma unroll
        for (int kw = 0; kw < E2_BLOCK; kw += 32) {
            const int k = kb + kw;
            int8 av[E2_NFB];
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                av[fb] = *(__global const int8 *)(wt +
                    ((size_t)(k >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 64 + (size_t)lane * 8);
#pragma unroll
            for (int tc = 0; tc < ntb; tc++) {
                const uint8 bv = *(__global const uint8 *)(qa + (size_t)(m0 + tc * 8 + lane) * K + k);
#pragma unroll
                for (int fb = 0; fb < E2_NFB; fb++)
                    acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
            }
        }
        /* the eight weight scales and the eight zero-point sums of a feature block are one 32-byte
         * load each, and the eight results one 32-byte store (the change k_vattn_scores got, which
         * took it from 150 to 50 ms; scalar here it is ~30 loads and 32 fp32 ops per feature block) */
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) {
            const int n = n0 + fb * 8;
            const float8 w = vload8(0, wsc + n);
            const int8 bs = vload8(0, bsq + b * N + n);
#pragma unroll
            for (int tc = 0; tc < ntb; tc++) {
                const int row = m0 + tc * 8 + lane;
                const float a = asc[row * nb + b];
                yacc[tc][fb] += (float8)(a)*w * convert_float8(acc[tc][fb] - (int8)(128) * bs);
            }
        }
    }
    const int clamped = dd[10];
    const float clo = as_float(dd[11]), chi = as_float(dd[12]);
#pragma unroll
    for (int tc = 0; tc < ntb; tc++) {
        const int row = m0 + tc * 8 + lane;
        if (row < M)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) {
                float8 v = yacc[tc][fb];
                if (clamped) v = clamp(v, (float8)(clo), (float8)(chi));
                vstore8(v, 0, y + row * N + n0 + fb * 8);
            }
    }
}

__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_gemm_dpas2(
    __global const char *wbuf, __global const uchar *acts, __global const int *desc, int M) {
    e2_gemm_dpas2(wbuf, acts, desc, M, 2);
}

__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_gemm_dpas_small(
    __global const char *wbuf, __global const uchar *acts, __global const int *desc, int M) {
    e2_gemm_dpas2(wbuf, acts, desc, M, 1);
}

/* Same 16x16 arithmetic; the separate name selects 256 GRFs for bulk calls in l0.c. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_gemm_dpas_large(
    __global const char *wbuf, __global const uchar *acts, __global const int *desc, int M) {
    e2_gemm_dpas2(wbuf, acts, desc, M, 2);
}

/* Gate/up share an input; keep their outputs in registers/SLM through GELU and quantization.
 * Eight subgroups cover one 128-feature activation block; two query tiles share each weight load. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_gemm_dpas_large_gate(
    __global const char *wbuf, __global uchar *acts, __global const int *desc, int M, int wr) {
    const int lane = get_local_id(0), sg = get_local_id(1);
    const int m0 = get_group_id(0) * 16, nbase = get_group_id(1) * 128;
    const int n0 = nbase + sg * 16, N = 2048;
    __global const uchar *qa = acts + desc[7];
    __global const float *asc = (__global const float *)(acts + desc[8]);
    float8 sums[2][2][2];
#pragma unroll
    for (int p = 0; p < 2; p++)
#pragma unroll
        for (int tc = 0; tc < 2; tc++)
#pragma unroll
            for (int fb = 0; fb < 2; fb++) sums[p][tc][fb] = (float8)(0.f);
    for (int b = 0; b < 4; b++) {
        int8 acc[2][2][2];
#pragma unroll
        for (int p = 0; p < 2; p++)
#pragma unroll
            for (int tc = 0; tc < 2; tc++)
#pragma unroll
                for (int fb = 0; fb < 2; fb++) acc[p][tc][fb] = (int8)(0);
#pragma unroll
        for (int kw = 0; kw < 128; kw += 32) {
            int8 av[2][2];
#pragma unroll
            for (int p = 0; p < 2; p++) {
                __global const uint *wt = (__global const uint *)(wbuf + desc[p * E2_DS_STRIDE + 9]);
#pragma unroll
                for (int fb = 0; fb < 2; fb++)
                    av[p][fb] = vload8(0, (__global const int *)wt +
                        ((size_t)((b * 128 + kw) / 32) * (N / 8) + n0 / 8 + fb) * 64 + lane * 8);
            }
#pragma unroll
            for (int tc = 0; tc < 2; tc++) {
                const uint8 bv = vload8(0, (__global const uint *)(qa +
                    (size_t)(m0 + tc * 8 + lane) * 512 + b * 128 + kw));
#pragma unroll
                for (int p = 0; p < 2; p++)
#pragma unroll
                    for (int fb = 0; fb < 2; fb++)
                        acc[p][tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[p][fb], bv, acc[p][tc][fb]);
            }
        }
#pragma unroll
        for (int p = 0; p < 2; p++) {
            __global const float *ws = (__global const float *)(wbuf + desc[p * E2_DS_STRIDE + 1]);
            __global const int *bs = (__global const int *)(wbuf + desc[p * E2_DS_STRIDE + 2]);
#pragma unroll
            for (int fb = 0; fb < 2; fb++) {
                const float8 w = vload8(0, ws + n0 + fb * 8);
                const int8 corr = vload8(0, bs + b * N + n0 + fb * 8) * (int8)(128);
#pragma unroll
                for (int tc = 0; tc < 2; tc++)
                    sums[p][tc][fb] += asc[(m0 + tc * 8 + lane) * 4 + b] * w *
                                      convert_float8(acc[p][tc][fb] - corr);
            }
        }
    }
    __local float tile[16 * 128];
#pragma unroll
    for (int tc = 0; tc < 2; tc++)
#pragma unroll
        for (int fb = 0; fb < 2; fb++) {
            float8 g = sums[0][tc][fb], up = sums[1][tc][fb];
            if (desc[10]) g = clamp(g, (float8)(as_float(desc[11])), (float8)(as_float(desc[12])));
            if (desc[E2_DS_STRIDE + 10])
                up = clamp(up, (float8)(as_float(desc[E2_DS_STRIDE + 11])), (float8)(as_float(desc[E2_DS_STRIDE + 12])));
            const float8 u = .7978845608f * (g + .044715f * g * g * g);
            const float8 e = native_exp(-2.f * fabs(u));
            const float8 t = copysign((1.f - e) / (1.f + e), u);
            g = .5f * g * (1.f + t) * up;
            vstore8(g, 0, tile + (tc * 8 + lane) * 128 + sg * 16 + fb * 8);
            const int row = m0 + tc * 8 + lane;
            if (wr && row < M) {
                vstore8(g, 0, (__global float *)(acts + desc[3]) + row * N + n0 + fb * 8);
                vstore8(up, 0, (__global float *)(acts + desc[E2_DS_STRIDE + 3]) + row * N + n0 + fb * 8);
            }
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global const int *down = desc + 2 * E2_DS_STRIDE;
#pragma unroll
    for (int tc = 0; tc < 2; tc++) {
        const int row = m0 + tc * 8 + sg;
        if (row < M) {
            const float16 v = vload16(0, tile + (tc * 8 + sg) * 128 + lane * 16);
            const float16 a = fabs(v);
            float8 a8 = fmax(a.lo, a.hi);
            float4 a4 = fmax(a8.lo, a8.hi);
            float mx = fmax(fmax(a4.x, a4.y), fmax(a4.z, a4.w));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) mx = fmax(mx, intel_sub_group_shuffle_xor(mx, o));
            const float sc = mx > 0.f ? mx / 127.f : 1.f;
            const int16 r = clamp(convert_int16_rte(v / sc), (int16)(-127), (int16)(127));
            vstore16(convert_uchar16(r + (int16)(128)), 0,
                     acts + down[7] + (size_t)row * N + nbase + lane * 16);
            if (!lane) ((__global float *)(acts + down[8]))[row * 16 + nbase / 128] = sc;
        }
    }
}

/* ------------------------------------------------------------------ audio projections on XMX
 * The text tower's 16x16 tile re-reads the weights once per token tile, which is the wrong trade for
 * a 19-token audio clip: mpad becomes 32 and the whole 305 MB tower is streamed twice. This tile
 * keeps the same 16x16 DPAS work per sub-group but stacks four M sub-groups on one N sub-group, so
 * four token tiles consume each weight tile. The four sub-groups sit in one work group and the three
 * followers hit the first one's weight lines in L1; the activation side is unchanged (16 rows per
 * sub-group, 32 bytes per lane per k-step). For a 19-token clip two of the four are active, so the
 * weights are read exactly once - the same traffic as a GEMV, without the GEMV's N-fold activation
 * re-read.
 */
#define E2_AMG 2  /* M sub-groups per N sub-group: work group covers E2_DM*E2_AMG rows */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_audio_dpas4(
    __global const char *wbuf, __global const uchar *acts, __global float *part,
    __global const int *desc, int M, int ksplit, int pstride) {
    const int t = get_global_id(2) / ksplit;
    const int ks = get_global_id(2) % ksplit;
    __global const int *dd = desc + t * E2_DS_STRIDE;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int msub = sg & (E2_AMG - 1), nsub = sg / E2_AMG;
    const int ng = get_local_size(1) / E2_AMG;
    const int m0 = get_group_id(0) * (E2_DM * E2_AMG) + msub * E2_DM;
    const int n0 = (get_group_id(1) * ng + nsub) * E2_DN;
    if (m0 >= M || n0 >= N) return;
    const int b0 = (nb * ks) / ksplit, b1 = (nb * (ks + 1)) / ksplit;
    const int nfb0 = n0 >> 3, nqb = N >> 3;
    __global const uint *wt = (__global const uint *)(wbuf + dd[9]);
    __global const uchar *qa = acts + dd[7];
    __global const float *wsc = (__global const float *)(wbuf + dd[1]);
    __global const int *bsq = (__global const int *)(wbuf + dd[2]);
    __global const float *asc = (__global const float *)(acts + dd[8]);
    __global float *y = (__global float *)((__global char *)acts + dd[3]);
    float8 yacc[E2_NTB][E2_NFB];
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) yacc[tc][fb] = (float8)(0.f);
    for (int b = b0; b < b1; b++) {
        const int kb = b * E2_BLOCK;
        int8 acc[E2_NTB][E2_NFB];
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) acc[tc][fb] = (int8)(0);
#pragma unroll
        for (int kw = 0; kw < E2_BLOCK; kw += 32) {
            const int k = kb + kw;
            int8 av[E2_NFB];
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                av[fb] = *(__global const int8 *)(wt +
                    ((size_t)(k >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 64 + (size_t)lane * 8);
#pragma unroll
            for (int tc = 0; tc < E2_NTB; tc++) {
                const uint8 bv = *(__global const uint8 *)(qa + (size_t)(m0 + tc * 8 + lane) * K + k);
#pragma unroll
                for (int fb = 0; fb < E2_NFB; fb++)
                    acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
            }
        }
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) {
            const int n = n0 + fb * 8;
            const float8 w = vload8(0, wsc + n);
            const int8 bs = vload8(0, bsq + b * N + n);
#pragma unroll
            for (int tc = 0; tc < E2_NTB; tc++) {
                const int row = m0 + tc * 8 + lane;
                const float a = asc[row * nb + b];
                yacc[tc][fb] += (float8)(a)*w * convert_float8(acc[tc][fb] - (int8)(128) * bs);
            }
        }
    }
    const int clamped = dd[10];
    const float clo = as_float(dd[11]), chi = as_float(dd[12]);
    __global float *pp = part + ((size_t)t * ksplit + ks) * pstride;
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++) {
        const int row = m0 + tc * 8 + lane;
        if (row < M)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) {
                float8 v = yacc[tc][fb];
                if (ksplit == 1) {
                    if (clamped) v = clamp(v, (float8)(clo), (float8)(chi));
                    vstore8(v, 0, y + row * N + n0 + fb * 8);
                } else {
                    vstore8(v, 0, pp + (size_t)row * N + n0 + fb * 8);
                }
            }
    }
}

/* Sum the split-K partials and apply the output clamp once, after the sum. One work item per output
 * column, looping the rows and slices; the result is the same tensor the unsplit kernel writes. */
__kernel void k_audio_reduce_k(__global const float *part, __global uchar *acts,
                               __global const int *desc, int count, int ksplit, int rows, int N,
                               int pstride) {
    const int n = get_global_id(0);
    const int t = get_global_id(1);
    if (n >= N || t >= count) return;
    __global const int *dd = desc + t * E2_DS_STRIDE;
    __global float *y = (__global float *)((__global char *)acts + dd[3]);
    const int clamped = dd[10];
    const float clo = as_float(dd[11]), chi = as_float(dd[12]);
    __global const float *base = part + (size_t)t * ksplit * pstride;
    for (int row = 0; row < rows; row++) {
        float s = 0.f;
        for (int ks = 0; ks < ksplit; ks++)
            s += base[(size_t)ks * pstride + (size_t)row * N + n];
        if (clamped) s = s < clo ? clo : (s > chi ? chi : s);
        y[(size_t)row * N + n] = s;
    }
}

/* The same GEMM with the work group's activation tile staged in shared local memory. The sixteen
 * sub-groups of a work group cover 256 features of ONE token tile, so all sixteen issue the same
 * per-lane B gathers from global - a row-major activation, eight rows per sub-group at a K-byte
 * stride. This kernel replaces those 16x redundant global reads with one coalesced cooperative load
 * of the E2_DM x E2_BLOCK tile into SLM (double-buffered, so the next block's global load overlaps
 * the current block's DPAS) and then reads B from SLM. A and the epilogue are unchanged. The
 * out-of-range sub-groups cannot return early any more - the barrier needs every work item - so they
 * clamp their feature block and discard the result instead. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_gemm_dpas_slm(
    __global const char *wbuf, __global const uchar *acts, __global const int *desc, int M) {
    const int t = get_global_id(2);
    __global const int *dd = desc + t * E2_DS_STRIDE;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int lid = sg * 8 + lane;              /* 0 .. 8*gy-1 */
    const int nwg = get_local_size(1) * 8;
    const int m0 = get_group_id(0) * E2_DM;
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * E2_DN;
    const int nok = n0 < N;                     /* barrier: no early return, clamp instead */
    const int nfb0 = (nok ? n0 : N - E2_DN) >> 3;
    const int nqb = N >> 3;
    __global const uint *wt = (__global const uint *)(wbuf + dd[9]);
    __global const uchar *qa = acts + dd[7];
    __global const float *wsc = (__global const float *)(wbuf + dd[1]);
    __global const int *bsq = (__global const int *)(wbuf + dd[2]);
    __global const float *asc = (__global const float *)(acts + dd[8]);
    __global float *y = (__global float *)((__global char *)acts + dd[3]);

    __local uchar bslm[2][E2_DM * E2_BLOCK];

    float8 yacc[E2_NTB][E2_NFB];
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) yacc[tc][fb] = (float8)(0.f);

    /* stage block 0, one 16-byte row-slice per work item (E2_DM rows x 8 slices) */
    for (int i = lid; i < E2_DM * 8; i += nwg) {
        const int row = i >> 3, col = (i & 7) * (E2_BLOCK / 8);
        *(__local uchar16 *)(bslm[0] + row * E2_BLOCK + col) =
            *(__global const uchar16 *)(qa + (size_t)(m0 + row) * K + col);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int b = 0; b < nb; b++) {
        const int kb = b * E2_BLOCK;
        /* stage block b+1 while block b computes */
        if (b + 1 < nb)
            for (int i = lid; i < E2_DM * 8; i += nwg) {
                const int row = i >> 3, col = (i & 7) * (E2_BLOCK / 8);
                *(__local uchar16 *)(bslm[(b + 1) & 1] + row * E2_BLOCK + col) =
                    *(__global const uchar16 *)(qa + (size_t)(m0 + row) * K +
                                                (size_t)(b + 1) * E2_BLOCK + col);
            }
        __local const uchar *bsrc = bslm[b & 1];
        int8 acc[E2_NTB][E2_NFB];
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) acc[tc][fb] = (int8)(0);
#pragma unroll
        for (int kw = 0; kw < E2_BLOCK; kw += 32) {
            const int k = kb + kw;
            int8 av[E2_NFB];
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                av[fb] = *(__global const int8 *)(wt +
                    ((size_t)(k >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 64 + (size_t)lane * 8);
#pragma unroll
            for (int tc = 0; tc < E2_NTB; tc++) {
                const uint8 bv = *(__local const uint8 *)(bsrc + (tc * 8 + lane) * E2_BLOCK + kw);
#pragma unroll
                for (int fb = 0; fb < E2_NFB; fb++)
                    acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
            }
        }
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) {
            const int n = n0 + fb * 8;
            const float8 w = vload8(0, wsc + n);
            const int8 bs = vload8(0, bsq + b * N + n);
#pragma unroll
            for (int tc = 0; tc < E2_NTB; tc++) {
                const int row = m0 + tc * 8 + lane;
                const float a = asc[row * nb + b];
                yacc[tc][fb] += (float8)(a)*w * convert_float8(acc[tc][fb] - (int8)(128) * bs);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nok)
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++) {
            const int row = m0 + tc * 8 + lane;
            if (row < M)
#pragma unroll
                for (int fb = 0; fb < E2_NFB; fb++)
                    vstore8(yacc[tc][fb], 0, y + row * N + n0 + fb * 8);
        }
}

/* ------------------------------------------------------------------ peak-issue probe
 * The same tile and loop shape as k_gemm_dpas with every operand load removed, so what it measures is
 * the matrix engine itself rather than the memory that feeds it. Comparing the two times says whether
 * a shape is compute-bound or load-bound; a probe that is not built from the real kernel's structure
 * is what made the two removed "peak issue" kernels report numbers seventy times too low. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_bench_dpas(__global float *out,
                                                                         int nb, int reps) {
    const int lane = get_local_id(0);
    int8 acc[E2_NTB][E2_NFB];
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) acc[tc][fb] = (int8)(get_global_id(0) + lane + fb);
    int8 av[E2_NFB];
    uint8 bv[E2_NTB];
#pragma unroll
    for (int fb = 0; fb < E2_NFB; fb++) av[fb] = (int8)(lane + fb);
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++) bv[tc] = (uint8)(lane + tc + 1);
    for (int r = 0; r < reps; r++)
        for (int b = 0; b < nb; b++)
#pragma unroll
            for (int kw = 0; kw < E2_BLOCK; kw += 32)
#pragma unroll
                for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
                    for (int fb = 0; fb < E2_NFB; fb++)
                        acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv[tc], acc[tc][fb]);
    int s = 0;
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) s += acc[tc][fb].s0 + acc[tc][fb].s7;
    if (lane == 0 && get_local_id(1) == 0) out[get_global_id(0)] = (float)s;
}

/* ------------------------------------------------------------------ row-wise normalization
 * One sub-group per row: lane l holds elements [b*128 + l*4, +4) of block b, so a block's largest
 * magnitude is a 32-lane butterfly and the quantized bytes of a block are one 128-byte contiguous
 * store. Rows of any length are covered by walking the blocks.
 *
 * flags: 1 = copy the pre-norm source to `res`, 2 = quantize the result into qa/asc,
 *        4 = read the source from `src`, 8 = add `res` after the norm (residual),
 *        16 = skip the norm and only quantize (the copy/quantize path for ctx),
 *        32 = skip the norm and don't touch the row (not used yet),
 *       128 = clamp the post-norm value to [clamp_lo, clamp_hi] before storing/quantizing (the audio
 *             tower's clipped linears: their input bound is part of the trained numerics),
 *       256 = apply `scale` before the residual, not after it (the audio FFN's 0.5 scaling)
 * `scale` multiplies the final value unless 256 moved it before the residual. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_rowops(__global float *x, __global const float *src, __global float *res,
                       __global uchar *qa, __global float *asc, __global const float *gamma,
                       __global const float *gamma2, float eps, float prescale, float scale,
                       int rows, int K, int nb, int flags, float clamp_lo, float clamp_hi,
                       __global const uchar *valid) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= rows) return;
    if (!(flags & 64) && valid && !valid[row]) return;
    __global float *rowp = x + (size_t)row * K;
    __global const float *sp = (flags & 4) ? src + (size_t)row * K : rowp;
    if (flags & 1) {
        for (int b = 0; b < nb; b++) {
            int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            *(__global float4 *)(res + (size_t)row * K + k) = *(__global const float4 *)(sp + k);
        }
    }
    float inv = 1.f;
    if (!(flags & 16)) {
        float ss = 0.f;
        for (int b = 0; b < nb; b++) {
            int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            float4 v = *(__global const float4 *)(sp + k) * prescale;
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        inv = 1.f / sqrt(e2_sg_sum(ss) / (float)K + eps);
    }
    for (int b = 0; b < nb; b++) {
        int k = b * E2_BLOCK + lane * 4;
        if (k >= K) break;
        float4 n = *(__global const float4 *)(sp + k) * prescale;
        if (!(flags & 16)) {
            n *= inv;
            if (gamma) n *= *(__global const float4 *)(gamma + k);
        }
        if (flags & 128) n = clamp(n, (float4)(clamp_lo), (float4)(clamp_hi));
        if (flags & 256) n *= scale;
        if (flags & 8) n += *(__global const float4 *)(res + (size_t)row * K + k);
        if (!(flags & 256)) n *= scale;
        *(__global float4 *)(rowp + k) = n;
        if (flags & 32) /* the o_proj -> ffn path: the value is also the next residual */
            *(__global float4 *)(res + (size_t)row * K + k) = n;
        if ((flags & 2) && !(flags & 32)) {
            float4 a = fabs(n);
            float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
            float sc = amax > 0.f ? amax / 127.f : 1.f;
            int4 r = convert_int4_rte(n / sc);
            r = clamp(r, (int4)(-127), (int4)(127));
            *(__global uchar4 *)(qa + (size_t)row * K + k) = convert_uchar4(r + (int4)(128));
            if (lane == 0) asc[(size_t)row * nb + b] = sc;
        }
    }
    if (flags & 32) { /* second half: x := norm(x) * gamma2, quantized for the FFN */
        float ss = 0.f;
        for (int b = 0; b < nb; b++) {
            int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            float4 v = *(__global const float4 *)(rowp + k);
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float inv2 = 1.f / sqrt(e2_sg_sum(ss) / (float)K + eps);
        for (int b = 0; b < nb; b++) {
            int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            float4 n = *(__global const float4 *)(rowp + k) * inv2;
            if (gamma2) n *= *(__global const float4 *)(gamma2 + k);
            *(__global float4 *)(rowp + k) = n;
            float4 a = fabs(n);
            float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
            float sc = amax > 0.f ? amax / 127.f : 1.f;
            int4 r = convert_int4_rte(n / sc);
            r = clamp(r, (int4)(-127), (int4)(127));
            *(__global uchar4 *)(qa + (size_t)row * K + k) = convert_uchar4(r + (int4)(128));
            if (lane == 0) asc[(size_t)row * nb + b] = sc;
        }
    }
}

/* ------------------------------------------------------------------ q/k/v normalization + rope
 * A row is one head of one token: the row index is token*heads + head, so the position for the rope
 * table is row/heads and the head's D elements are contiguous.
 *
 * The rotation pairs element i with i + D/2 (the reference's rotate_half). The reference reads both
 * elements of a pair before writing either, and the pairs straddle a lane boundary, so the partner is
 * loaded into a register in the same pass - reading it back after the store would give the rotated
 * value and a silently wrong embedding. */
static inline int e2_sg_sum_i(int v);

/* Constant quant modes: 0 = FP32, 1 = per-128 Q, 2 = per-head K plus per-128 block sums. */
static inline void e2_norm_rope_row(__global float *x, int row, int D, __global const float *gamma,
                                    float eps, int rope, __global const float *cs,
                                    __global const float *sn, int lane, int quant,
                                    __global uchar *qa, __global float *asc, __global int *bsq,
                                    int N, int key, int live, int wr) {
    __global float *p = x + (size_t)row * D;
    if (quant && !live) {
        for (int b = 0; b < D / 128; b++) {
            const int d = b * 128 + lane * 4;
            if (quant == 1) {
                vstore4((uchar4)(128), 0, qa + (size_t)row * D + d);
                if (!lane) asc[(size_t)row * (D / 128) + b] = 1.f;
            } else {
                vstore4((uchar4)(0), 0, qa + E2_DPAS_OFF(d / 4, key / 8, key % 8, 0, N / 8));
                if (!lane) { *asc = 1.f; bsq[b * N] = 0; }
            }
            if (wr) vstore4((float4)(0.f), 0, p + d);
        }
        return;
    }
    float4 out[4];
    const int halfd = D / 2;
    const int niter = D / 128; /* D <= 512, so at most four float4 per lane */
    const int off0 = lane * 4;
    float4 v0 = (float4)0, v1 = (float4)0, v2 = (float4)0, v3 = (float4)0;
    float ss = 0.f;
    if (niter > 0) { v0 = *(__global const float4 *)(p + off0);         ss += dot(v0, v0); }
    if (niter > 1) { v1 = *(__global const float4 *)(p + off0 + 128);   ss += dot(v1, v1); }
    if (niter > 2) { v2 = *(__global const float4 *)(p + off0 + 256);   ss += dot(v2, v2); }
    if (niter > 3) { v3 = *(__global const float4 *)(p + off0 + 384);   ss += dot(v3, v3); }
    const float inv = 1.f / sqrt(e2_sg_sum(ss) / (float)D + eps);
    const int step = halfd / 128; /* how many float4 separate a pair (1 or 2) */
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (niter <= j) break;
        const int d = off0 + j * 128;
        const float4 a = ((j == 0) ? v0 : (j == 1) ? v1 : (j == 2) ? v2 : v3) * inv;
        if (!rope) {
            const float4 g = gamma ? *(__global const float4 *)(gamma + d) : (float4)1.f;
            *(__global float4 *)(p + d) = a * g;
            continue;
        }
        const int first = d < halfd;
        const int pj = first ? j + step : j - step;
        const float4 y = ((pj == 0) ? v0 : (pj == 1) ? v1 : (pj == 2) ? v2 : v3) * inv;
        const int pd = first ? d + halfd : d - halfd;
        const float4 g = gamma ? *(__global const float4 *)(gamma + d) : (float4)1.f;
        const float4 gy = gamma ? *(__global const float4 *)(gamma + pd) : (float4)1.f;
        const float4 c = *(__global const float4 *)(cs + d);
        const float4 s4 = *(__global const float4 *)(sn + d);
        const float4 value = first ? (a * g * c - y * gy * s4) : (a * g * c + y * gy * s4);
        if (!quant || wr) *(__global float4 *)(p + d) = value;
        if (quant == 1) {
            const float4 ab = fabs(value);
            const float mx = e2_sg_max(fmax(fmax(ab.x, ab.y), fmax(ab.z, ab.w)));
            const float sc = mx > 0.f ? mx / 127.f : 1.f;
            const int4 r = clamp(convert_int4_rte(value / sc), (int4)(-127), (int4)(127));
            vstore4(convert_uchar4(r + (int4)(128)), 0, qa + (size_t)row * D + d);
            if (!lane) asc[(size_t)row * niter + j] = sc;
        } else if (quant == 2) out[j] = value;
    }
    if (quant == 2) {
        float mx = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= niter) break;
            const float4 a = fabs(out[j]);
            mx = fmax(mx, fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
        }
        mx = e2_sg_max(mx);
        const float sc = mx > 0.f ? mx / 127.f : 1.f;
        if (!lane) *asc = sc;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= niter) break;
            const int d = j * 128 + lane * 4;
            const int4 r = clamp(convert_int4_rte(out[j] / sc), (int4)(-127), (int4)(127));
            vstore4(convert_uchar4(r & (int4)(255)), 0,
                    qa + E2_DPAS_OFF(d / 4, key / 8, key % 8, 0, N / 8));
            const int sum = e2_sg_sum_i(r.x + r.y + r.z + r.w);
            if (!lane) bsq[j * N] = sum;
        }
    }
}

/* q, k and v normalization in one launch: three independent row ranges, all sub-group uniform, so the
 * sub-group reductions inside e2_norm_rope_row stay valid. K and V have fewer head rows than Q, so
 * the tail work items only do the q part. */
static inline void e2_norm_rope3(
    __global float *xq, __global const float *gq, int rq, __global float *xk,
    __global const float *gk, int rk, __global float *xv, __global const float *gv, int rv,
    __global const float *costab, __global const float *sintab, float eps, int hq, int hkv, int D,
    int seq_len) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row < rq) {
        const int pos = (row / hq) % seq_len;
        e2_norm_rope_row(xq, row, D, gq, eps, 1, costab + (size_t)pos * D, sintab + (size_t)pos * D,
                         lane, 0, 0, 0, 0, 0, 0, 1, 1);
    }
    if (row < rk) {
        const int pos = (row / hkv) % seq_len;
        e2_norm_rope_row(xk, row, D, gk, eps, 1, costab + (size_t)pos * D, sintab + (size_t)pos * D,
                         lane, 0, 0, 0, 0, 0, 0, 1, 1);
    }
    if (row < rv) {
        e2_norm_rope_row(xv, row, D, gv, eps, 0, costab, sintab, lane, 0, 0, 0, 0, 0, 0, 1, 1);
    }
}

__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_norm_rope3(
    __global float *xq, __global const float *gq, int rq, __global float *xk,
    __global const float *gk, int rk, __global float *xv, __global const float *gv, int rv,
    __global const float *costab, __global const float *sintab, float eps, int hq, int hkv, int D,
    int seq_len) {
    if (D == 256)
        e2_norm_rope3(xq, gq, rq, xk, gk, rk, xv, gv, rv, costab, sintab, eps, hq, hkv, 256, seq_len);
    else
        e2_norm_rope3(xq, gq, rq, xk, gk, rk, xv, gv, rv, costab, sintab, eps, hq, hkv, 512, seq_len);
}

/* Quantize normalized/rotated Q and pack K while their values are still in registers. */
static inline void e2_norm_rope3_i8(
    __global float *xq, __global const float *gq, int rq, __global float *xk,
    __global const float *gk, int rk, __global float *xv, __global const float *gv, int rv,
    __global const float *costab, __global const float *sintab, float eps, int H, int KV, int D,
    int S, __global uchar *qa, __global float *asc, __global uchar *kp, __global float *ks,
    __global int *kb, int SP, __global const uchar *valid, int wr) {
    const int r = get_group_id(0) * get_local_size(1) + get_local_id(1), lane = get_local_id(0);
    if (r < rq) {
        const int token = r / H, pos = token % S, live = !valid || valid[token];
        e2_norm_rope_row(xq, r, D, gq, eps, 1, costab + (size_t)pos * D, sintab + (size_t)pos * D,
                         lane, 1, qa, asc, 0, 0, 0, live, wr);
    }
    const int batches = rk / (KV * S);
    if (r < batches * KV * SP) {
        const int key = r / KV % SP, head = r % KV, seq = r / (KV * SP), kvz = seq * KV + head;
        const int token = seq * S + key;
        const int live = key < S && (!valid || valid[token]);
        e2_norm_rope_row(xk, key < S ? token * KV + head : 0, D, gk, eps, 1,
                         costab + (size_t)min(key, S - 1) * D, sintab + (size_t)min(key, S - 1) * D,
                         lane, 2, kp + (size_t)kvz * SP * D, ks + (size_t)kvz * SP + key,
                         kb + (size_t)kvz * (D / 128) * SP + key, SP, key, live, wr && key < S);
    }
    if (r < rv)
        e2_norm_rope_row(xv, r, D, gv, eps, 0, costab, sintab, lane, 0, 0, 0, 0, 0, 0, 1, 1);
}

__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_norm_rope3_i8(
    __global float *xq, __global const float *gq, int rq, __global float *xk,
    __global const float *gk, int rk, __global float *xv, __global const float *gv, int rv,
    __global const float *costab, __global const float *sintab, float eps, int H, int KV, int D,
    int S, __global uchar *qa, __global float *asc, __global uchar *kp, __global float *ks,
    __global int *kb, int SP, __global const uchar *valid, int wr) {
    if (D == 256)
        e2_norm_rope3_i8(xq, gq, rq, xk, gk, rk, xv, gv, rv, costab, sintab, eps, H, KV, 256, S,
                         qa, asc, kp, ks, kb, SP, valid, wr);
    else
        e2_norm_rope3_i8(xq, gq, rq, xk, gk, rk, xv, gv, rv, costab, sintab, eps, H, KV, 512, S,
                         qa, asc, kp, ks, kb, SP, valid, wr);
}

/* Pack two FP16 DPAS A operands; each lane loads eight pairs contiguously in the products.
 * SP includes zero keys through a multiple of sixteen. */
__kernel void k_attn_f16_pack(__global const float *k, __global const float *v,
    __global int *kp, __global int *vp, int S, int SP, int D, int KV) {
    const int i = get_global_id(0);
    const int z = i / (SP * D / 2), t = i % (SP * D / 2);
    const int seq = z / KV, h = z % KV;
    const int j = t / (D / 2), d = t % (D / 2);
    half2 a = (half2)(0);
    if (j < S) a = convert_half2_rte(vload2(0, k + ((size_t)seq * S + j) * KV * D + h * D + d * 2));
    kp[(size_t)z * SP * D / 2 + ((size_t)(d / 8) * (SP / 8) + j / 8) * 64 + (d % 8) * 8 + j % 8] = as_int(a);
    const int jp = t / D, f = t % D;
    half2 b = (half2)(0);
    if (jp * 2 < S) b.s0 = convert_half_rte(v[((size_t)seq * S + jp * 2) * KV * D + h * D + f]);
    if (jp * 2 + 1 < S) b.s1 = convert_half_rte(v[((size_t)seq * S + jp * 2 + 1) * KV * D + h * D + f]);
    vp[(size_t)z * SP * D / 2 + ((size_t)(jp / 8) * (D / 8) + f / 8) * 64 + (jp % 8) * 8 + f % 8] = as_int(b);
}

/* FP16 XMX attention with FP32 accumulation/softmax; projections remain W8A8.
 * DPAS operand distribution follows Khronos cl_intel_subgroup_matrix_multiply_accumulate.
 * Each subgroup computes eight queries by sixteen keys (or context dimensions).
 * Sequence tiles are padded independently: a subgroup must never mix two sequences. */
static inline void e2_attn_f16(
    __global const float *q, __global const int *k, __global const int *v,
    __global float *scores, __global const half *prob, __global float *ctx,
    int S, int SP, int H, int D, int KV, int context, __local float *tile, int wr) {
    const int lane = get_local_id(0), sg = get_local_id(1);
    const int m = get_group_id(0) * 8 + lane;
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * 16;
    const int z = get_group_id(2), h = z % H, seq = z / H, kv = h * KV / H;
    const int N = context ? D : SP, K = context ? SP : D;
    if (n0 >= N) return;
    float8 acc0 = (float8)(0.f), acc1 = (float8)(0.f);
    for (int kk = 0; kk < K; kk += 16) {
        __global const int *packed = (context ? v : k) + (size_t)(seq * KV + kv) * SP * D / 2;
        const size_t off = ((size_t)(kk / 16) * (N / 8) + n0 / 8) * 64 + lane * 8;
        const int8 a0 = vload8(0, packed + off);
        const int8 a1 = vload8(0, packed + off + 64);
        int8 b = (int8)(0);
        if (m < S) {
            if (context) b = as_int8(vload16(0, prob + ((size_t)h * (get_num_groups(2) / H) * S + seq * S + m) * SP + kk));
            else b = as_int8(convert_half16_rte(vload16(0, q + ((size_t)seq * S + m) * H * D + h * D + kk)));
        }
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b, acc1);
    }
    if (tile) {
        vstore8(acc0, 0, tile + lane * 128 + sg * 16);
        vstore8(acc1, 0, tile + lane * 128 + sg * 16 + 8);
    }
    if (m < S) {
        if (context && (!tile || wr)) {
            __global float *dst = ctx + ((size_t)seq * S + m) * H * D + h * D + n0;
            vstore8(acc0, 0, dst);
            vstore8(acc1, 0, dst + 8);
        } else if (!context) {
            __global float *dst = scores + ((size_t)h * (get_num_groups(2) / H) * S + seq * S + m) * SP + n0;
            vstore8(acc0, 0, dst);
            vstore8(acc1, 0, dst + 8);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_attn_f16(
    __global const float *q, __global const int *k, __global const int *v,
    __global float *scores, __global const half *prob, __global float *ctx,
    int S, int SP, int H, int D, int KV, int context) {
    e2_attn_f16(q, k, v, scores, prob, ctx, S, SP, H, D, KV, context, 0, 0);
}

/* One workgroup owns a 128-d context block, so quantization needs no global context round trip. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_attn_f16_ctx(
    __global const int *v, __global const half *prob, __global float *ctx,
    int S, int SP, int H, int D, int KV, __global uchar *qa, __global float *asc, int wr) {
    __local float tile[8 * 128];
    e2_attn_f16(0, 0, v, 0, prob, ctx, S, SP, H, D, KV, 1, tile, wr);
    barrier(CLK_LOCAL_MEM_FENCE);
    const int lane = get_local_id(0), sg = get_local_id(1);
    const int m = get_group_id(0) * 8 + sg, z = get_group_id(2);
    if (m >= S) return;
    const int row = z / H * S + m, h = z % H, nbase = get_group_id(1) * 128;
    const float16 value = vload16(0, tile + sg * 128 + lane * 16);
    const float16 a = fabs(value);
    const float8 a8 = fmax(a.lo, a.hi);
    const float4 a4 = fmax(a8.lo, a8.hi);
    float mx = fmax(fmax(a4.x, a4.y), fmax(a4.z, a4.w));
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) mx = fmax(mx, intel_sub_group_shuffle_xor(mx, o));
    const float sc = mx > 0.f ? mx / 127.f : 1.f;
    const int16 r = clamp(convert_int16_rte(value / sc), (int16)(-127), (int16)(127));
    vstore16(convert_uchar16(r + (int16)(128)), 0,
             qa + (size_t)row * H * D + h * D + nbase + lane * 16);
    if (!lane) asc[(size_t)row * (H * D / 128) + h * (D / 128) + nbase / 128] = sc;
}

__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_attn_f16_softmax(
    __global const float *scores, __global half *prob, __global const uchar *valid,
    int rows, int S, int SP, int window) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int h = get_group_id(1), lane = get_local_id(0);
    if (row >= rows) return;
    const int base = row / S * S, pos = row % S;
    const int lo = window > 0 ? max(0, pos - window) : 0;
    const int hi = window > 0 ? min(S, pos + window + 1) : S;
    const int active = !valid || valid[row];
    const size_t off = ((size_t)h * rows + row) * SP;
    float mx = -3.0e38f;
    for (int j = lane; j < S; j += 32)
        if (active && j >= lo && j < hi && (!valid || valid[base + j]))
            mx = fmax(mx, scores[off + j]);
    mx = e2_sg_max(mx);
    float sum = 0.f;
    for (int j = lane; j < S; j += 32)
        if (active && j >= lo && j < hi && (!valid || valid[base + j]))
            sum += native_exp(scores[off + j] - mx);
    sum = e2_sg_sum(sum);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int j = lane; j < SP; j += 32) {
        const float p = active && j >= lo && j < hi && (!valid || valid[base + j])
                            ? native_exp(scores[off + j] - mx) * inv : 0.f;
        prob[off + j] = convert_half_rte(p);
    }
}

/* ------------------------------------------------------------------ int8 attention on the matrix engine
 * Scores = Q K^T, FP32 softmax, context = P V, with P materialized as unsigned int8. Both products
 * share the projection GEMM's DPAS operand layout (weights in A, activations in B, one 32-byte
 * vector load per k-step) and the explicit-pointer helper below:
 *
 *     C[m][n] = sum_k W[n][k] * X[m][k]
 *
 * with W the packed K (scores) or V^T (context) and X the packed Q or the P the softmax wrote.
 *
 * Why P is materialised instead of an online softmax in registers: the context accumulator of a
 * flash-attention kernel is 8 query rows x D columns, and a DPAS sub-group is 8 lanes, so the
 * accumulator alone would be D (256 or 512) registers per lane. The online softmax cannot be made
 * to fit, and a two-pass form (row max, then write P) does not need it: both passes are over the
 * same K block and the K re-read is a fraction of the traffic the blocked kernel was doing.
 *
 * Masking (sliding window, valid rows) happens in the softmax, not in the products: the products
 * run over the whole padded key range and a masked key simply gets p = 0. That is one wasted
 * product for the keys a window excludes, which is cheap on the XMX engine and keeps the products
 * a plain GEMM. */

static inline int e2_sg_sum_i(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += intel_sub_group_shuffle_xor(v, o);
    return v;
}

/* Quantize a f32 matrix into the DPAS A-operand layout (the host-side q8w_dpas in gpu.c, on the
 * device). The matrix has N rows of K values and the source element (n,k) is src[n*sn + k*sk], so the
 * same kernel packs K (sn = the row stride, sk = 1, rows = keys) and V^T (sn = 1, sk = the row
 * stride, rows = feature dims). `rlen` and `klen` are the real row and k counts: the layout pads
 * rows to a multiple of 8 and keys to a multiple of 128, and everything past those is written as
 * zero, so a padded key contributes nothing to either product.
 *
 * One sub-group per row, lane l covers the k-quad (b*32 + l) of block b. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_attn_packw(
    __global const float *src, __global uchar *wd, __global float *wsc, __global int *bsq, int N,
    int K, int sn, int sk, int rlen, int klen, int slen, int KV, __global const uchar *valid) {
    const int z = get_group_id(2), seq = z / KV, kv = z % KV;
    const int D = sk == 1 ? K : N;
    const int stride = sk == 1 ? sn : sk;
    src += (size_t)seq * slen * stride + kv * D;
    wd += (size_t)z * N * K;
    wsc += (size_t)z * N;
    bsq += (size_t)z * (K / 128) * N;
    const int n = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (n >= N) return;
    const int live = n < rlen && (sk != 1 || valid[seq * slen + n]);
    __global const float *row = src + (size_t)n * sn;
    float amax = 0.f;
    if (live)
        for (int k = lane; k < klen; k += 32)
            if (sk == 1 || valid[seq * slen + k]) amax = fmax(amax, fabs(row[(size_t)k * sk]));
    amax = e2_sg_max(amax);
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    if (lane == 0) wsc[n] = sc;
    const int nqb = N >> 3, nb = K >> 7;
    for (int b = 0; b < nb; b++) {
        const int kq = b * 32 + lane, k0 = kq * 4;
        int4 r = (int4)(0);
        if (live && k0 < klen) {
            float4 v = (float4)(0.f);
            if (sk == 1 && k0 + 3 < klen)
                v = *(__global const float4 *)(row + k0);
            else
                for (int j = 0; j < 4; j++) {
                    const int k = k0 + j;
                    if (k < klen && (sk == 1 || valid[seq * slen + k]))
                        ((float *)&v)[j] = row[(size_t)k * sk];
                }
            r = clamp(convert_int4_rte(v / sc), (int4)(-127), (int4)(127));
        }
        *(__global char4 *)(wd + E2_DPAS_OFF(kq, n >> 3, n & 7, 0, nqb)) = convert_char4(r);
        const int blk = e2_sg_sum_i(r.x + r.y + r.z + r.w);
        if (lane == 0) bsq[(size_t)b * N + n] = blk;
    }
}

/* The two attention products, one kernel: `wd` is the packed int8 weight matrix in the DPAS A
 * layout, `qa` the int8 activations biased by +128, and the result is
 *
 *     y[m][n] = sum_b asc[m,b] * wsc[n] * ( dp4a_block - zero_point * bsq[b,n] )
 *
 * Q uses zero point 128; nonnegative probabilities use the full unsigned range with zero point 0.
 *
 * which is the same arithmetic as k_gemm_dpas, with the descriptor indirection replaced by
 * explicit pointers because the attention weights are packed at run time, per layer and per
 * sequence, and the output row stride is not N (the context stride is q_stride). */
static inline void e2_attn_gemm(
    __global const uchar *wd, __global const float *wsc, __global const int *bsq,
    __global const uchar *qa, __global const float *asc, __global float *y, int N, int K, int nb,
    int M, int ystride, int xstride, int astride, int zero_point, __local float *tile, int wr) {
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int m0 = get_group_id(0) * E2_DM;
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * E2_DN;
    if (n0 >= N) return;
    const int nfb0 = n0 >> 3, nqb = N >> 3;
    float8 yacc[E2_NTB][E2_NFB];
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) yacc[tc][fb] = (float8)(0.f);
    for (int b = 0; b < nb; b++) {
        const int kb = b * E2_BLOCK;
        int8 acc[E2_NTB][E2_NFB];
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) acc[tc][fb] = (int8)(0);
#pragma unroll
        for (int kw = 0; kw < E2_BLOCK; kw += 32) {
            const int k = kb + kw;
            int8 av[E2_NFB];
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                /* same k-step-major run as k_gemm_dpas, in bytes (wd is uchar*, not uint*) */
                av[fb] = *(__global const int8 *)(wd +
                    ((size_t)(k >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 256 + (size_t)lane * 32);
#pragma unroll
            for (int tc = 0; tc < E2_NTB; tc++) {
                const int row = m0 + tc * 8 + lane;
                const uint8 bv = row < M ?
                    *(__global const uint8 *)(qa + (size_t)row * xstride + k) : (uint8)(zero_point);
#pragma unroll
                for (int fb = 0; fb < E2_NFB; fb++)
                    acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
            }
        }
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) {
            const int n = n0 + fb * 8;
            const float8 w = vload8(0, wsc + n);
            const int8 bs = vload8(0, bsq + b * N + n);
#pragma unroll
            for (int tc = 0; tc < E2_NTB; tc++) {
                const int row = m0 + tc * 8 + lane;
                const float a = row < M ? asc[row * astride + b] : 0.f;
                yacc[tc][fb] += (float8)(a)*w * convert_float8(acc[tc][fb] - (int8)(zero_point) * bs);
            }
        }
    }
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++) {
        const int row = m0 + tc * 8 + lane;
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) {
            if (tile) vstore8(yacc[tc][fb], 0, tile + (tc * 8 + lane) * 128 + sg * E2_DN + fb * 8);
            if (row < M && (!tile || wr))
                vstore8(yacc[tc][fb], 0, y + row * ystride + n0 + fb * 8);
        }
    }
}

/* Retained for vision and the explicit-pointer kernel check. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_attn_gemm(
    __global const uchar *wd, __global const float *wsc, __global const int *bsq,
    __global const uchar *qa, __global const float *asc, __global float *y, int N, int K, int nb,
    int M, int ystride, int xstride, int astride) {
    e2_attn_gemm(wd, wsc, bsq, qa, asc, y, N, K, nb, M, ystride, xstride, astride, 128, 0, 1);
}

/* Each z is a (sequence, query head); GQA heads share packed KV, never score/P scratch.
 * Only `tile` query rows are materialized per z, independently of the full sequence length. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_attn_batch_gemm(
    __global const uchar *wd, __global const float *wsc, __global const int *bsq,
    __global const uchar *qa, __global const float *asc, __global float *y,
    int S, int sup, int D, int H, int KV, int q0, int tile) {
    const int z = get_group_id(2), seq = z / H, h = z % H;
    const int kvz = seq * KV + h * KV / H, nb = D / 128;
    wd += (size_t)kvz * sup * D;
    wsc += (size_t)kvz * sup;
    bsq += (size_t)kvz * nb * sup;
    qa += ((size_t)seq * S * H + q0 * H + h) * D;
    asc += ((size_t)seq * S * H + q0 * H + h) * nb;
    y += (size_t)z * tile * sup;
    e2_attn_gemm(wd, wsc, bsq, qa, asc, y, sup, D, nb, min(tile, S - q0),
                 sup, H * D, H * nb, 128, 0, 1);
}

/* Each workgroup quantizes sixteen query rows by one 128-feature context block. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_attn_i8_ctx(
    __global const uchar *wd, __global const float *wsc, __global const int *bsq,
    __global const uchar *prob, __global const float *psc, __global float *ctx,
    int S, int sup, int D, int H, int KV, int q0, int tile,
    __global uchar *qa, __global float *asc, int wr) {
    __local float block[E2_DM * 128];
    const int z = get_group_id(2), seq = z / H, h = z % H;
    const int kvz = seq * KV + h * KV / H, nb = sup / 128;
    const int M = min(tile, S - q0);
    e2_attn_gemm(wd + (size_t)kvz * D * sup, wsc + (size_t)kvz * D,
                 bsq + (size_t)kvz * nb * D, prob + (size_t)z * tile * sup,
                 psc + (size_t)z * tile * nb, ctx + ((size_t)(seq * S + q0) * H + h) * D,
                 D, sup, nb, M, H * D, sup, nb, 0, block, wr);
    barrier(CLK_LOCAL_MEM_FENCE);
    const int lane = get_local_id(0), sg = get_local_id(1), nbase = get_group_id(1) * 128;
#pragma unroll
    for (int t = 0; t < E2_NTB; t++) {
        const int m = get_group_id(0) * E2_DM + sg + t * 8;
        if (m >= M) continue;
        const float16 value = vload16(0, block + (sg + t * 8) * 128 + lane * 16);
        const float16 a = fabs(value);
        const float8 a8 = fmax(a.lo, a.hi);
        const float4 a4 = fmax(a8.lo, a8.hi);
        float mx = fmax(fmax(a4.x, a4.y), fmax(a4.z, a4.w));
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) mx = fmax(mx, intel_sub_group_shuffle_xor(mx, o));
        const float sc = mx > 0.f ? mx / 127.f : 1.f;
        const int16 r = clamp(convert_int16_rte(value / sc), (int16)(-127), (int16)(127));
        const int row = seq * S + q0 + m;
        vstore16(convert_uchar16(r + (int16)(128)), 0,
                 qa + (size_t)row * H * D + h * D + nbase + lane * 16);
        if (!lane) asc[(size_t)row * (H * D / 128) + h * (D / 128) + nbase / 128] = sc;
    }
}

/* scores -> P. One sub-group per row: the row's scores are reduced for the max and then for the
 * sum over the sequence's visible keys, and the result is quantized per 128-key block.
 *
 * The block scale is the only subtlety. With q = round(255 * e_k / max_e_b) and asc[m,b] =
 * max_e_b / (255 * l) the dequantized value asc*(q) is e_k/l, i.e. the softmax weight, and no
 * separate renormalisation is needed: every p_k is rounded against its own 128-key block's largest
 * value, which is the same recipe the activation quantization uses everywhere else in the engine.
 * A masked key writes byte 0 (the context product has no zero-point correction) and a block with
 * nothing visible gets asc = 0, so a fully masked row produces an all-zero context. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_attn_softmax(
    __global float *sc, __global uchar *p, __global float *asc, __global const uchar *valid, int M,
    int N, int slen, int window, int H, int nb, int q0, int tile) {
    const int z = get_group_id(2), kbase = (z / H) * slen;
    sc += (size_t)z * tile * N;
    p += (size_t)z * tile * N;
    asc += (size_t)z * tile * nb;
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= M) return;
    const int pos = q0 + row;
    if (valid && !valid[kbase + pos]) {
        for (int b = 0; b < nb; b++) {
            vstore4((uchar4)(0), 0, p + (size_t)row * N + b * 128 + lane * 4);
            if (!lane) asc[(size_t)row * nb + b] = 0.f;
        }
        return;
    }
    const int lo = (window > 0 && pos - window > 0) ? pos - window : 0;
    const int hi = (window > 0 && pos + window + 1 < slen) ? pos + window + 1 : slen;
    __global float *srow = sc + (size_t)row * N;
    float m = -3.0e38f;
    for (int b = 0; b < nb; b++) {
        const int j0 = b * 128 + lane * 4;
        const float4 v = *(__global const float4 *)(srow + j0);
        float4 z = (float4)(-3.0e38f);
        if (j0 >= lo && j0 + 3 < hi && (!valid || (valid[kbase + j0] && valid[kbase + j0 + 1] &&
                                                  valid[kbase + j0 + 2] && valid[kbase + j0 + 3]))) {
            z = v;
        } else {
            z.x = (j0 + 0 >= lo && j0 + 0 < hi && (!valid || valid[kbase + j0 + 0])) ? v.x : -3.0e38f;
            z.y = (j0 + 1 >= lo && j0 + 1 < hi && (!valid || valid[kbase + j0 + 1])) ? v.y : -3.0e38f;
            z.z = (j0 + 2 >= lo && j0 + 2 < hi && (!valid || valid[kbase + j0 + 2])) ? v.z : -3.0e38f;
            z.w = (j0 + 3 >= lo && j0 + 3 < hi && (!valid || valid[kbase + j0 + 3])) ? v.w : -3.0e38f;
        }
        m = fmax(m, fmax(fmax(z.x, z.y), fmax(z.z, z.w)));
    }
    m = e2_sg_max(m);
    float l = 0.f, maxb[16];
    for (int b = 0; b < nb && b < 16; b++) {
        const int j0 = b * 128 + lane * 4;
        const float4 v = *(__global const float4 *)(srow + j0);
        float4 e = (float4)(0.f);
        if (j0 >= lo && j0 + 3 < hi && (!valid || (valid[kbase + j0] && valid[kbase + j0 + 1] &&
                                                  valid[kbase + j0 + 2] && valid[kbase + j0 + 3]))) {
#pragma unroll
            for (int t = 0; t < 4; t++) e[t] = exp(v[t] - m);
        } else {
            e.x = (j0 + 0 >= lo && j0 + 0 < hi && (!valid || valid[kbase + j0 + 0])) ? exp(v.x - m) : 0.f;
            e.y = (j0 + 1 >= lo && j0 + 1 < hi && (!valid || valid[kbase + j0 + 1])) ? exp(v.y - m) : 0.f;
            e.z = (j0 + 2 >= lo && j0 + 2 < hi && (!valid || valid[kbase + j0 + 2])) ? exp(v.z - m) : 0.f;
            e.w = (j0 + 3 >= lo && j0 + 3 < hi && (!valid || valid[kbase + j0 + 3])) ? exp(v.w - m) : 0.f;
        }
        maxb[b] = e2_sg_max(fmax(fmax(e.x, e.y), fmax(e.z, e.w)));
        l += e.x + e.y + e.z + e.w;
        *(__global float4 *)(srow + j0) = e;
    }
    l = e2_sg_sum(l);
    __global uchar *prow = p + (size_t)row * N;
    for (int b = 0; b < nb && b < 16; b++) {
        const int j0 = b * 128 + lane * 4;
        const float4 e = *(__global const float4 *)(srow + j0);
        const float mb = maxb[b];
        int4 q = (int4)(0);
        if (mb > 0.f) q = clamp(convert_int4_rte(e * (255.f / mb)), (int4)(0), (int4)(255));
        *(__global uchar4 *)(prow + j0) = convert_uchar4(q);
        if (lane == 0) asc[(size_t)row * nb + b] = (mb > 0.f && l > 0.f) ? mb / (255.f * l) : 0.f;
    }
}

/* ------------------------------------------------------------------ gated-MLP / PLE epilogue
 * gate = gelu(gate) * up (or * this layer's PLE slice), then quantize the result: it is the input of
 * the next projection, so the elementwise product and the activation quantization are one pass. Four
 * elements per lane per block, like k_rowops.
 *
 * One *block* per sub-group, not one row: every block's scale comes from a sub-group max over the
 * block alone, so the blocks of a row are independent. They used to be a serial loop inside one
 * sub-group per row, which put 4 work groups on a 15-row call and left the launch latency-bound -
 * this kernel measured a flat 17 us for K=2048 from 16 rows to 64, i.e. the row count did not
 * matter at all because the per-row chain of nb dependent block iterations did. The grid is
 * rows*nb/4 work groups now (784 for a 196-token document instead of 49). */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_gelu_mul_quant(
    __global float *gate, __global const float *up, __global const float *ple, __global uchar *qa,
    __global float *asc, int rows, int K, int nb, int stride, int ple_off, int wr) {
    const int item = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int row = item / nb, b = item - row * nb;
    const int lane = get_local_id(0);
    if (row >= rows) return;
    const int k = b * E2_BLOCK + lane * 4;
    if (k >= K) return;
    __global float *gp = gate + (size_t)row * K;
    float4 g = *(__global const float4 *)(gp + k);
    const float4 u = 0.7978845608f * (g + 0.044715f * g * g * g);
    const float4 e = native_exp(-2.f * fabs(u));
    const float4 t = copysign((1.f - e) / (1.f + e), u);
    g = 0.5f * g * (1.f + t);
    if (up) g *= *(__global const float4 *)(up + (size_t)row * K + k);
    if (ple) g *= *(__global const float4 *)(ple + (size_t)row * stride + ple_off + k);
    /* the fp32 result has no consumer (the down projection reads the int8 copy); only the region
     * dump wants it, and storing it was a third of this kernel's traffic */
    if (wr) *(__global float4 *)(gp + k) = g;
    float4 a = fabs(g);
    float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
    float sc = amax > 0.f ? amax / 127.f : 1.f;
    int4 r = convert_int4_rte(g / sc);
    r = clamp(r, (int4)(-127), (int4)(127));
    *(__global uchar4 *)(qa + (size_t)row * K + k) = convert_uchar4(r + (int4)(128));
    if (lane == 0) asc[(size_t)row * nb + b] = sc;
}

/* ------------------------------------------------------------------ audio tower
 * The conformer's attention is not the text tower's: the window is twelve keys (the query plus
 * eleven frames back), the logits carry a relative-position term, and the softmax is capped with
 * 50*tanh(s/50). The matrix engine is the wrong tool at twelve keys - packing Q/K would cost more
 * than the products - so one sub-group owns (query row, head): it holds 4 head dims per lane, walks
 * the window once for the capped logits, once for the softmax and once for the value accumulation,
 * then clamps and quantizes the context for the post projection. That is the whole attention block
 * in one launch, including the relative term (q . (k + relk), the same sum as q.k + q.relk).
 */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_audio_attn(
    __global const float *q, __global const float *k, __global const float *v,
    __global const float *relk, __global const float *qmult, __global float *ctx,
    __global uchar *qa, __global float *asc, int rows, float cap, float kscale, float clo,
    float chi) {
    const int p = get_group_id(0);
    const int h = get_local_id(1);
    const int lane = get_local_id(0);
    if (p >= rows) return;
    const int D = 128, HD = 8 * D;
    const int hoff = h * D + lane * 4;
    const __global float4 *q4 = (const __global float4 *)(q + (size_t)p * HD + hoff);
    float4 qv = *q4 * *(__global const float4 *)(qmult + lane * 4);
    float sc[12];
    int nk = p + 1 < 12 ? p + 1 : 12;
    float mx = -3.0e38f;
    for (int d = 0; d < nk; d++) {
        const __global float4 *k4 =
            (const __global float4 *)(k + (size_t)(p - d) * HD + hoff);
        const __global float4 *r4 =
            (const __global float4 *)(relk + (size_t)(12 - d) * HD + hoff);
        float4 t = qv * (*k4 * kscale + *r4);
        float s = (t.x + t.y) + (t.z + t.w);
        s = e2_sg_sum(s);
        s = cap * tanh(s / cap);
        sc[d] = s;
        mx = fmax(mx, s);
    }
    float sum = 0.f;
    for (int d = 0; d < nk; d++) {
        const float e = exp(sc[d] - mx);
        sc[d] = e;
        sum += e;
    }
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    float4 acc = (float4)(0.f);
    for (int d = 0; d < nk; d++) {
        const __global float4 *v4 =
            (const __global float4 *)(v + (size_t)(p - d) * HD + hoff);
        acc += (sc[d] * inv) * *v4;
    }
    acc = clamp(acc, (float4)(clo), (float4)(chi));
    *(__global float4 *)(ctx + (size_t)p * HD + hoff) = acc;
    float4 a = fabs(acc);
    const float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
    const float s = amax > 0.f ? amax / 127.f : 1.f;
    int4 r = clamp(convert_int4_rte(acc / s), (int4)(-127), (int4)(127));
    *(__global uchar4 *)(qa + (size_t)p * HD + hoff) = convert_uchar4(r + (int4)(128));
    if (lane == 0) asc[(size_t)p * 8 + h] = s;
}

/* Two RMSNorm/residual stages in one pass. Three places in the layer are exactly this pair:
 *   FF1 post-norm * 0.5 + residual, then the attention pre-norm and its int8 quantization;
 *   post-attention norm + residual, then the light-conv pre-norm and its quantization;
 *   FF2 post-norm * 0.5 + residual, then the layer's final norm.
 * Each row is held in eight float4 registers across both reductions, so the intermediate never goes
 * back to memory and two launches become one. flags: 2 = quantize the second result, 128 = clamp it.
 */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_audio_norm2(
    __global const float *src, __global float *res, __global float *res_out, __global float *x,
    __global uchar *qa, __global float *asc, __global const float *g1, __global const float *g2,
    int rows, int K, int nb, int flags, float scale, float clo, float chi) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= rows) return;
    __global const float *sp = src + (size_t)row * K;
    float4 v[8];
    float ss = 0.f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        v[b] = *(__global const float4 *)(sp + k);
        ss += v[b].x * v[b].x + v[b].y * v[b].y + v[b].z * v[b].z + v[b].w * v[b].w;
    }
    const float inv1 = 1.f / sqrt(e2_sg_sum(ss) / (float)K + 1e-6f);
    ss = 0.f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 y = v[b] * inv1 * *(__global const float4 *)(g1 + k) * scale;
        y += *(__global const float4 *)(res + (size_t)row * K + k);
        v[b] = y;
        if (res_out) *(__global float4 *)(res_out + (size_t)row * K + k) = y;
        ss += y.x * y.x + y.y * y.y + y.z * y.z + y.w * y.w;
    }
    const float inv2 = 1.f / sqrt(e2_sg_sum(ss) / (float)K + 1e-6f);
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 n = v[b] * inv2 * *(__global const float4 *)(g2 + k);
        if (flags & 128) n = clamp(n, (float4)(clo), (float4)(chi));
        *(__global float4 *)(x + (size_t)row * K + k) = n;
        if (flags & 2) {
            float4 a = fabs(n);
            const float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
            const float sc = amax > 0.f ? amax / 127.f : 1.f;
            int4 r = clamp(convert_int4_rte(n / sc), (int4)(-127), (int4)(127));
            *(__global uchar4 *)(qa + (size_t)row * K + k) = convert_uchar4(r + (int4)(128));
            if (lane == 0) asc[(size_t)row * nb + b] = sc;
        }
    }
}

/* SiLU, input clamp and int8 quantization in one pass: the audio feed-forward's activation point.
 * rows x K, one sub-group per row, four values per lane per 128-element block. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_audio_silu_quant(
    __global float *x, __global uchar *qa, __global float *asc, int rows, int K, int nb,
    float clo, float chi) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= rows) return;
    __global float *xp = x + (size_t)row * K;
    for (int b = 0; b < nb; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 v = *(__global float4 *)(xp + k);
        v = v / (1.f + exp(-v));
        v = clamp(v, (float4)(clo), (float4)(chi));
        *(__global float4 *)(xp + k) = v;
        float4 a = fabs(v);
        const float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        int4 r = clamp(convert_int4_rte(v / sc), (int4)(-127), (int4)(127));
        *(__global uchar4 *)(qa + (size_t)row * K + k) = convert_uchar4(r + (int4)(128));
        if (lane == 0) asc[(size_t)row * nb + b] = sc;
    }
}

/* light conv, first half: clamp the 2048-wide projection and GLU it down to 1024. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_audio_glu(
    __global const float *lc, __global float *out, int rows, float clo, float chi) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= rows) return;
    __global const float *src = lc + (size_t)row * 2048;
    __global float *dst = out + (size_t)row * 1024;
    for (int k = lane * 4; k < 1024; k += 128) {
        float4 a = clamp(*(__global const float4 *)(src + k), (float4)(clo), (float4)(chi));
        float4 b = clamp(*(__global const float4 *)(src + 1024 + k), (float4)(clo), (float4)(chi));
        *(__global float4 *)(dst + k) = a / (1.f + exp(-b));
    }
}

/* light conv, second half: causal depthwise k=5 -> RMSNorm -> SiLU -> clamp -> int8. The convolution
 * is 1024 channels wide but only five taps deep, so one sub-group per row keeps every tap's 128-byte
 * line resident while the five accumulate. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_audio_dwconv(
    __global const float *x, __global const float *w, __global const float *gamma,
    __global uchar *qa, __global float *asc, int rows, float eps, float clo, float chi) {
    const int s = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (s >= rows) return;
    float4 cv[8];
    float ss = 0.f;
    for (int b = 0; b < 8; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 a = (float4)(0.f);
        for (int j = 0; j < 5; j++) {
            const int r = s - 4 + j;
            if (r < 0) continue;
            a += *(__global const float4 *)(x + (size_t)r * 1024 + k) *
                 *(__global const float4 *)(w + (size_t)j * 1024 + k);
        }
        cv[b] = a;
        ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    const float inv = 1.f / sqrt(e2_sg_sum(ss) / 1024.f + eps);
    for (int b = 0; b < 8; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 v = cv[b] * inv * *(__global const float4 *)(gamma + k);
        v = v / (1.f + exp(-v));
        v = clamp(v, (float4)(clo), (float4)(chi));
        float4 a = fabs(v);
        const float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        int4 r = clamp(convert_int4_rte(v / sc), (int4)(-127), (int4)(127));
        *(__global uchar4 *)(qa + (size_t)s * 1024 + k) = convert_uchar4(r + (int4)(128));
        if (lane == 0) asc[(size_t)s * 8 + b] = sc;
    }
}

/* light conv close: clamp the 1024-wide projection, add the pre-conv residual, store the layer. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_audio_add_clamp(
    __global const float *src, __global const float *res, __global float *out, int rows, float clo,
    float chi) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= rows) return;
    for (int k = lane * 4; k < 1024; k += 128) {
        float4 v = clamp(*(__global const float4 *)(src + (size_t)row * 1024 + k),
                         (float4)(clo), (float4)(chi));
        v += *(__global const float4 *)(res + (size_t)row * 1024 + k);
        *(__global float4 *)(out + (size_t)row * 1024 + k) = v;
    }
}

/* head close: out_proj bias + scale-free RMSNorm over 1536, quantized for embed_audio. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_audio_bias_norm(
    __global float *x, __global const float *bias, __global uchar *qa, __global float *asc,
    int rows, float eps) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= rows) return;
    float4 v[12];
    float ss = 0.f;
#pragma unroll
    for (int b = 0; b < 12; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        v[b] = *(__global float4 *)(x + (size_t)row * 1536 + k) +
               *(__global const float4 *)(bias + k);
        ss += v[b].x * v[b].x + v[b].y * v[b].y + v[b].z * v[b].z + v[b].w * v[b].w;
    }
    const float inv = 1.f / sqrt(e2_sg_sum(ss) / 1536.f + eps);
#pragma unroll
    for (int b = 0; b < 12; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 n = v[b] * inv;
        *(__global float4 *)(x + (size_t)row * 1536 + k) = n;
        float4 a = fabs(n);
        const float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        int4 r = clamp(convert_int4_rte(n / sc), (int4)(-127), (int4)(127));
        *(__global uchar4 *)(qa + (size_t)row * 1536 + k) = convert_uchar4(r + (int4)(128));
        if (lane == 0) asc[(size_t)row * 12 + b] = sc;
    }
}

/* ------------------------------------------------------------------ output: mean pool + L2
 * One work group per sequence: the mean over its valid rows, then the L2 norm (the reference head
 * pools first and normalizes second). */
__kernel void k_pool_l2(__global float *out, __global const float *emb, __global const uchar *valid,
                        int rows, int dim, int seq_len) {
    const int seq = get_group_id(0);
    const int lane = get_local_id(0);
    const int sb = seq_len > 0 ? seq_len : rows;
    const int r0 = seq_len > 0 ? seq * sb : 0;
    const int r1 = seq_len > 0 ? (seq + 1) * sb : rows;
    __local float buf[1024];
    __local int cntl;
    /* The mean is a reduction over rows for every dim, so the parallelism is over the 768 dims: 256
     * lanes x 4 dims covers it in one pass. It used to be 128 scalar lanes, i.e. one work group and
     * eight per cent of one EU for the whole pooler. */
    if (lane == 0) {
        int c = 0;
        for (int r = r0; r < r1; r++) if (!valid || valid[r]) c++;
        cntl = c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float inv_cnt = cntl > 0 ? 1.f / (float)cntl : 0.f;
    const int d0 = lane * 4;
    const int nf = dim - d0 >= 4 ? 4 : (dim - d0 > 0 ? dim - d0 : 0);
    float4 acc = (float4)(0.f);
    for (int r = r0; r < r1; r++) {
        if (valid && !valid[r]) continue;
        const __global float *p = emb + (size_t)r * dim + d0;
        if (nf == 4) {
            acc += *(__global const float4 *)p;
        } else {
            for (int i = 0; i < nf; i++) acc[i] += p[i];
        }
    }
    acc *= inv_cnt;
    if (nf == 4) {
        vstore4(acc, 0, buf + d0);
    } else {
        for (int i = 0; i < nf; i++) buf[d0 + i] = acc[i];
    }
    __local float red[256];
    red[lane] = acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) {
        float t = 0.f;
        for (int i = 0; i < 256; i++) t += red[i];
        red[0] = t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float inv = 1.f / sqrt(red[0] + 1e-12f);
    if (nf == 4) {
        vstore4(vload4(0, buf + d0) * inv, 0, out + (size_t)seq * dim + d0);
    } else {
        for (int i = 0; i < nf; i++) out[(size_t)seq * dim + d0 + i] = buf[d0 + i] * inv;
    }
}

/* ================================================================== vision tower
 * Gemma4 vision encoder: 16 layers, hidden 768, 12 heads x head_dim 64, ffn 3072, full bidirectional
 * attention, axial 2D RoPE (channels 0..31 rotate by the x coordinate, 32..63 by y).
 *
 * Two structural differences from the text tower shape the kernels here:
 *
 *  - The padding mask is a gather, not a mask. The host builds `order`, the indices of the patches
 *    whose position ids are not (-1,-1), and everything downstream runs on a dense [P,768] tensor.
 *    The reference materialises a (1,1,2520,2520) bias inside every layer; neither the engine's CPU
 *    path nor this one ever builds it.
 *  - head_dim is 64, which is below the engine's 128-element activation quantization block. The two
 *    attention products therefore do not share the text tower's recipe: the scores product quantizes
 *    per (row, head) with a single 64-wide block (k_vattn_packk writes a k-quad per lane, two k-steps
 *    in total) and so needs no block loop at all, while the context product keeps the 128-wide recipe
 *    because its k dimension is the *key* count, which is long.
 *
 * Every GEMM in the tower (all seven projections per layer, 768 or 3072 wide, K 768 or 3072) is a
 * multiple of 128 and goes through k_gemm_dpas unchanged; only the attention and the two ends of the
 * tower need kernels of their own. */

/* Patch gather + 2*(x-0.5) + activation quantization, one pass. The rescale is the image processor's,
 * done here so the host never touches the 7.7 MB of pixels, and the gather is why no kernel
 * downstream needs a padding mask. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_vpatch_q(
    __global uchar *qa, __global float *asc, __global const float *pixels, __global const int *order,
    int P, int D) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= P) return;
    const int nb = D / E2_BLOCK;
    __global const float *src = pixels + (size_t)order[row] * D;
    __global uchar *qp = qa + (size_t)row * D;
    for (int b = 0; b < nb; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 v = *(__global const float4 *)(src + k);
        v = 2.0f * (v - (float4)(0.5f));
        float4 a = fabs(v);
        const float amax = e2_sg_max(fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        int4 r = clamp(convert_int4_rte(v / sc), (int4)(-127), (int4)(127));
        *(__global uchar4 *)(qp + k) = convert_uchar4(r + (int4)(128));
        if (lane == 0) asc[(size_t)row * nb + b] = sc;
    }
}

/* The patch embedder's position term: x = proj + pos_table[x] + pos_table[E2_V_POS + y] (the table is
 * bf16 and stays on the device, 31 MB of it). */
__kernel void k_vpos_add(__global float *x, __global const float *proj,
                         __global const ushort *postab, __global const int *order,
                         __global const int *posids, int P, int D, int npos) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int d = (get_group_id(1) * get_local_size(0) + get_local_id(0)) * 4;
    if (row >= P || d >= D) return;
    const int i = order[row];
    const __global ushort *ex = postab + (size_t)posids[2 * i] * D + d;
    const __global ushort *ey = postab + ((size_t)npos + (size_t)posids[2 * i + 1]) * D + d;
    const float4 p = *(__global const float4 *)(proj + (size_t)row * D + d);
    *(__global float4 *)(x + (size_t)row * D + d) =
        p + e2_bf16x4(ex) + e2_bf16x4(ey);
}

/* q/k/v RMSNorm + axial 2D RoPE, one launch, one sub-group per (patch, head) row of 64 elements
 * (two per lane). Rows are (patch*12 + head), so the rope table for a row is at row/heads.
 *
 * The rotation pairs element i with i+16 inside each 32-half (the reference's rotate_half applied to
 * the x-half and the y-half separately), which is exactly the sub-group lane xor 8: lane l holds
 * elements 2l and 2l+1, its partner holds 2(l^8) and 2(l^8)+1. Lanes with bit 3 clear are the first
 * element of their pair, lanes with it set the second, so the same three lines cover all four
 * quarters. `first = !(lane & 8)` is that distinction; the table index is picked by bit 4 (the x half
 * is 0..31, the y half 32..63) and the within-half index by the low three bits. */
static inline void e2_vnorm_rope_row(__global float *p, __global const float *gamma, float eps,
                                     int rope, __global const float *cs, __global const float *sn,
                                     int lane) {
    const float2 v = *(__global const float2 *)(p + lane * 2);
    const float inv = 1.f / sqrt(e2_sg_sum(dot(v, v)) * (1.f / 64.f) + eps);
    const float2 g = gamma ? *(__global const float2 *)(gamma + lane * 2) : (float2)(1.f, 1.f);
    const float2 a = v * inv * g;
    if (!rope) {
        *(__global float2 *)(p + lane * 2) = a;
        return;
    }
    const float2 y = (float2)(intel_sub_group_shuffle(a.s0, lane ^ 8),
                              intel_sub_group_shuffle(a.s1, lane ^ 8));
    const int ci = ((lane >> 4) & 1) * 32 + (lane & 7) * 2;
    const float2 c = *(__global const float2 *)(cs + ci);
    const float2 s = *(__global const float2 *)(sn + ci);
    *(__global float2 *)(p + lane * 2) =
        (lane & 8) ? (a * c + y * s) : (a * c - y * s);
}

__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_vnorm_rope3(
    __global float *xq, __global const float *gq, __global float *xk, __global const float *gk,
    __global float *xv, __global const float *gv, __global const float *costab,
    __global const float *sintab, float eps, int rows, int heads) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= rows) return;
    const int pos = row / heads;
    e2_vnorm_rope_row(xq + (size_t)row * 64, gq, eps, 1, costab + (size_t)pos * 64,
                      sintab + (size_t)pos * 64, lane);
    e2_vnorm_rope_row(xk + (size_t)row * 64, gk, eps, 1, costab + (size_t)pos * 64,
                      sintab + (size_t)pos * 64, lane);
    e2_vnorm_rope_row(xv + (size_t)row * 64, gv, eps, 0, costab, sintab, lane);
}

/* Q -> the context product's activation layout: int8 rows of 64, one per (patch, head), biased by
 * +128 like every other activation in the engine. Rows are head-major ([head][patch][64]) so that the
 * scores product's B-operand load is 32 contiguous bytes per lane at a 64-byte stride - packed
 * patch-major it would stride by the whole hidden (768 B) and every lane of a load would land on its
 * own cache line. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_vattn_packq(
    __global const float *q, __global uchar *qa, __global float *asc, int rows, int P, int H) {
    const int r = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (r >= rows) return;
    const int p = r / H, h = r % H;
    const float2 v = *(__global const float2 *)(q + (size_t)r * 64 + lane * 2);
    const float amax = e2_sg_max(fmax(fabs(v.s0), fabs(v.s1)));
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    int2 ri = clamp(convert_int2_rte(v / sc), (int2)(-127), (int2)(127));
    __global uchar *dst = qa + ((size_t)h * P + p) * 64 + lane * 2;
    *(__global uchar2 *)dst = convert_uchar2(ri + (int2)(128));
    if (lane == 0) asc[(size_t)h * P + p] = sc;
}

/* K -> the scores product's *weight* operand (the DPAS A layout, rows = keys, K = 64). Rows are
 * (head * Ppad + key) so one launch packs all twelve heads, each into its own Ppad*64-byte block;
 * keys past the real patch count are written as zero, which is what lets the softmax ignore them by
 * masking rather than the product needing a guard.
 *
 * The whole K dimension is one 64-element quantization block, so this is two k-steps: lane l writes
 * the k-quad l for l < 16 and nothing for l >= 16, and the row's per-block sum is the row sum. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_vattn_packk(
    __global const float *k, __global uchar *wd, __global float *wsc, __global int *bsq, int P,
    int Ppad, int D) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    const int n = row % Ppad, h = row / Ppad;
    const int live = (n < P) && (lane < 16);
    __global const float *src = k + (size_t)n * D + (size_t)h * 64;
    float4 v = (float4)(0.f);
    if (live) v = *(__global const float4 *)(src + lane * 4);
    const float amax = e2_sg_max(fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w))));
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    if (lane == 0) wsc[(size_t)h * Ppad + n] = sc;
    int4 r = (live) ? clamp(convert_int4_rte(v / sc), (int4)(-127), (int4)(127)) : (int4)(0);
    if (lane < 16)
        *(__global char4 *)(wd + (size_t)h * Ppad * 64 + E2_DPAS_OFF(lane, n >> 3, n & 7, 0, Ppad >> 3)) =
            convert_char4(r);
    const int blk = e2_sg_sum_i(r.x + r.y + r.z + r.w);
    if (lane == 0) bsq[(size_t)h * Ppad + n] = blk;
}

/* V^T -> the context product's weight operand, 64 feature rows of K = keys, 128 keys per
 * quantization block (the same recipe k_attn_packw uses, specialised to the 64-row matrix). Rows are
 * (head * 64 + feature) so one launch covers all twelve heads. The source is read down a column
 * (element (feature, key) is v[key*768 + head*64 + feature]) and is not contiguous, so the four k
 * values of a quad are four scalar loads - the same path the text tower's packer takes. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_vattn_packv(
    __global const float *v, __global uchar *wd, __global float *wsc, __global int *bsq, int P,
    int Ppad, int D, int nb) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    const int n = row & 63, h = row >> 6;
    __global const float *sr = v + n + (size_t)h * 64;
    float amax = 0.f;
    for (int k = lane; k < P; k += 32) amax = fmax(amax, fabs(sr[(size_t)k * D]));
    amax = e2_sg_max(amax);
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    if (lane == 0) wsc[row] = sc;
    for (int b = 0; b < nb; b++) {
        const int kq = b * 32 + lane, k0 = kq * 4;
        float4 vv = (float4)(0.f);
        for (int j = 0; j < 4; j++) {
            const int k = k0 + j;
            if (k < P) ((float *)&vv)[j] = sr[(size_t)k * D];
        }
        const int4 r = clamp(convert_int4_rte(vv / sc), (int4)(-127), (int4)(127));
        *(__global char4 *)(wd + (size_t)h * Ppad * 64 + E2_DPAS_OFF(kq, n >> 3, n & 7, 0, 8)) =
            convert_char4(r);
        const int blk = e2_sg_sum_i(r.x + r.y + r.z + r.w);
        if (lane == 0) bsq[((size_t)h * nb + b) * 64 + n] = blk;
    }
}

/* ------------------------------------------------- fused vision softmax (the fast path)
 * The three-kernel form below writes the fp16 score matrix, reads it back to exp and quantize, and
 * writes the int8 P - a full round trip of an S x S matrix per (head, layer). This one does scores,
 * exp and the int8 P in a single launch and never materialises the score matrix.
 *
 * It works because of the tile: a lane holds the whole 32-key score block of its query row (four
 * int8 vectors, one per E2_SNFB key block), so the block max and the exp sum are per-lane reductions
 * with no shuffle, and the 32 quantized bytes of the row are one contiguous store - the same
 * coalesced store the fp16 form needed, at half the bytes.
 *
 * The exp reference is the *block's own* max, not the global row max. exp(v - mb) then lands in
 * (0, 1] and cannot overflow whatever the score magnitude is, and the block only needs its own
 * max to get there. The consequence is that the block scale is not the final softmax weight: the
 * true value is exp(v_k - M)/l with M the row max and l the row normaliser, so the context's
 * per-block scale is asc[row][b] = exp(mb_b - M) / (127 * l), which is what k_vattn_pasc computes
 * from the (max, sum) pair this kernel writes per block. From there the context is the usual
 * asc * (P . V) with P stored unbiased in [0, 127], so the i8 x u8 product needs no zero-point
 * correction at all and k_vattn_ctx has no bsq term. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_vattn_psoft(
    __global const uchar *wd, __global const float *wsc, __global const int *bsq,
    __global const uchar *qa, __global const float *asc, __global uchar *p,
    __global float *bmax, __global float *bsum, int N, int M, int ypitch, int xstride,
    int astride, int slen) {
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int m0 = get_group_id(0) * E2_SDM;
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * E2_SDN;
    if (n0 >= N) return;
    const int nfb0 = n0 >> 3, nqb = N >> 3;
    int8 acc[E2_SNTB][E2_SNFB];
#pragma unroll
    for (int tc = 0; tc < E2_SNTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_SNFB; fb++) acc[tc][fb] = (int8)(0);
    /* k = head dim = 64 for this tower: two DPAS k-steps, no block loop */
#pragma unroll
    for (int kw = 0; kw < 64; kw += 32) {
        int8 av[E2_SNFB];
#pragma unroll
        for (int fb = 0; fb < E2_SNFB; fb++)
            av[fb] = *(__global const int8 *)(wd +
                ((size_t)(kw >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 256 + (size_t)lane * 32);
#pragma unroll
        for (int tc = 0; tc < E2_SNTB; tc++) {
            const uint8 bv =
                *(__global const uint8 *)(qa + (size_t)(m0 + tc * 8 + lane) * xstride + kw);
#pragma unroll
            for (int fb = 0; fb < E2_SNFB; fb++)
                acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
        }
    }
    const int kb = n0 >> 5;            /* the 32-key block this tile writes */
    const int nb32 = N >> 5;
#pragma unroll
    for (int tc = 0; tc < E2_SNTB; tc++) {
        const int row = m0 + tc * 8 + lane;
        if (row >= M) continue;
        const float a = asc[(size_t)row * astride];
        float8 f[E2_SNFB];
        float mb = -3.0e38f;
        /* pass 1: the 32 scores of this lane's (row, key block), masked, and their max */
#pragma unroll
        for (int fb = 0; fb < E2_SNFB; fb++) {
            const int n = n0 + fb * 8;
            const float8 w = vload8(0, wsc + n);
            const int8 bs = vload8(0, bsq + n);
            float8 v = convert_float8(acc[tc][fb] - (int8)(128) * bs) * w * a;
#pragma unroll
            for (int i = 0; i < 8; i++)
                if (n + i >= slen) v[i] = -3.0e38f;
            f[fb] = v;
#pragma unroll
            for (int i = 0; i < 8; i++) mb = fmax(mb, v[i]);
        }
        const int live = mb > -1.0e38f;
        float s = 0.f;
        __global uchar *prow = p + ((size_t)kb * ypitch + row) * 32;
        /* pass 2: exp against the block max, sum, and the 32 quantized bytes (0..127, unbiased -
         * the context's i8 x u8 product then needs no zero-point correction) */
#pragma unroll
        for (int fb = 0; fb < E2_SNFB; fb++) {
            float8 e = (float8)(0.f);
            if (live) e = exp(f[fb] - mb);
            s += e.s0 + e.s1 + e.s2 + e.s3 + e.s4 + e.s5 + e.s6 + e.s7;
            const int8 q = clamp(convert_int8_rte(e * (float8)(127.f)), (int8)(0), (int8)(127));
            *(__global uchar8 *)(prow + fb * 8) = convert_uchar8_sat(q);
        }
        bmax[(size_t)row * nb32 + kb] = live ? mb : -3.0e38f;
        bsum[(size_t)row * nb32 + kb] = s;
    }
}

/* The block maxima and exp sums k_vattn_psoft wrote, turned into the context's per-block scale:
 * M = max_b mb_b, l = sum_b exp(mb_b - M) * sum_b, asc[row][b] = exp(mb_b - M) / (127 l).
 * One sub-group per query row; a row is 76 blocks, so two passes over lanes. */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_vattn_pasc(
    __global const float *bmax, __global const float *bsum, __global float *asc, int M, int nb) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= M) return;
    const __global float *mb = bmax + (size_t)row * nb;
    const __global float *sb = bsum + (size_t)row * nb;
    float m = -3.0e38f;
    for (int b = lane; b < nb; b += 32) m = fmax(m, mb[b]);
    m = e2_sg_max(m);
    float l = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const float x = mb[b];
        if (x > -1.0e38f) l += exp(x - m) * sb[b];
    }
    l = e2_sg_sum(l);
    for (int b = lane; b < nb; b += 32) {
        const float x = mb[b];
        asc[(size_t)row * nb + b] = (x > -1.0e38f && l > 0.f) ? exp(x - m) / (127.f * l) : 0.f;
    }
}

/* context = P V on the matrix engine, one 32-key block at a time. Same operand layout as
 * k_attn_gemm (V^T in the DPAS A layout, P as the B operand), but the k block is 32 keys rather than
 * 128 (the block k_vattn_psoft quantized against), P comes from the key-blocked layout that kernel
 * writes, and P is unbiased in [0, 127] so the mixed-sign product is sum(V * P) with no bsq term. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_vattn_ctx(
    __global const uchar *wd, __global const float *wsc, __global const uchar *p,
    __global const float *asc, __global float *y, int N, int K, int nb, int M, int ystride,
    int pstride, int astride) {
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int m0 = get_group_id(0) * E2_DM;
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * E2_DN;
    if (n0 >= N) return;
    const int nfb0 = n0 >> 3, nqb = N >> 3;
    float8 yacc[E2_NTB][E2_NFB];
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++) yacc[tc][fb] = (float8)(0.f);
    float8 wv[E2_NFB]; /* the weight scales do not depend on the key block: hoist them out */
#pragma unroll
    for (int fb = 0; fb < E2_NFB; fb++) wv[fb] = vload8(0, wsc + n0 + fb * 8);
    for (int ks = 0; ks < nb; ks++) {
        const int k = ks * 32;
        int8 acc[E2_NTB][E2_NFB];
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++) acc[tc][fb] = (int8)(0);
        int8 av[E2_NFB];
#pragma unroll
        for (int fb = 0; fb < E2_NFB; fb++)
            av[fb] = *(__global const int8 *)(wd +
                ((size_t)(k >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 256 + (size_t)lane * 32);
        __global const uchar *pb = p + (size_t)ks * pstride * 32;
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++) {
            const uint8 bv = *(__global const uint8 *)(pb + (size_t)(m0 + tc * 8 + lane) * 32);
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
        }
#pragma unroll
        for (int tc = 0; tc < E2_NTB; tc++) {
            const int row = m0 + tc * 8 + lane;
            const float a = asc[(size_t)row * astride + ks];
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                yacc[tc][fb] += (float8)(a)*wv[fb] * convert_float8(acc[tc][fb]);
        }
    }
#pragma unroll
    for (int tc = 0; tc < E2_NTB; tc++) {
        const int row = m0 + tc * 8 + lane;
        if (row < M)
#pragma unroll
            for (int fb = 0; fb < E2_NFB; fb++)
                vstore8(yacc[tc][fb], 0, y + row * ystride + n0 + fb * 8);
    }
}

/* scores = Q K^T on the matrix engine: the same arithmetic as k_attn_gemm with the k dimension
 * fixed at 64, i.e. two DPAS k-steps and one quantization block per row, so there is no block loop
 * and no per-block epilogue. `wd` is the packed K, `qa` the packed Q, and the result is
 * y[m][n] = asc[m] * wsc[n] * (dpas - 128*bsq[n]).
 *
 * The tile is E2_SDM x E2_SDN rather than the text kernel's 16x16: at K=64 a 16x16 tile issues four
 * operand sends for four DPAS, and this shape is bound by those sends (see e2_tile.h).
 *
 * The score matrix is stored *blocked by 32 keys* and in fp16: scr[(n/32)*ypitch + m]*32 + n%32.
 * Row-major, the eight lanes of a store are 2560 floats (10 KB) apart and each writes 32 bytes, so
 * every store allocates and half-fills a cache line; blocked, one lane's 32 columns are one
 * contiguous run and the eight lanes cover a contiguous kilobyte. The kernel is *entirely*
 * store-bound either way (deleting the store takes it from 167 to 36 ms over the tower), so what
 * matters most is the width: a score is a softmax logit of magnitude ~30, where fp16 carries an
 * absolute resolution of 0.015 - a quarter of one int8 step in the P matrix it feeds - and halving
 * the matrix halves both this kernel's traffic and the softmax's.
 *
 * It also writes `bmax[m][n/32]`, the maximum of the 32-key block, masked to the real keys (>= slen
 * is -3e38). That is the value the softmax would otherwise need a whole pass over the fp16 row to
 * rediscover; it costs one extra fmax chain and one 4-byte store per block here. */
__attribute__((intel_reqd_sub_group_size(8))) __kernel void k_vattn_scores(
    __global const uchar *wd, __global const float *wsc, __global const int *bsq,
    __global const uchar *qa, __global const float *asc, __global float *y, __global float *bmax,
    int N, int M, int ypitch, int xstride, int astride, int slen) {
    const int lane = get_local_id(0);
    const int sg = get_local_id(1);
    const int m0 = get_group_id(0) * E2_SDM;
    const int n0 = (get_group_id(1) * get_local_size(1) + sg) * E2_SDN;
    if (n0 >= N) return;
    const int nfb0 = n0 >> 3, nqb = N >> 3;
    int8 acc[E2_SNTB][E2_SNFB];
#pragma unroll
    for (int tc = 0; tc < E2_SNTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_SNFB; fb++) acc[tc][fb] = (int8)(0);
#pragma unroll
    for (int kw = 0; kw < 64; kw += 32) {
        int8 av[E2_SNFB];
#pragma unroll
        for (int fb = 0; fb < E2_SNFB; fb++)
            av[fb] = *(__global const int8 *)(wd +
                ((size_t)(kw >> 5) * nqb + (size_t)nfb0 + (size_t)fb) * 256 + (size_t)lane * 32);
#pragma unroll
        for (int tc = 0; tc < E2_SNTB; tc++) {
            const uint8 bv =
                *(__global const uint8 *)(qa + (size_t)(m0 + tc * 8 + lane) * xstride + kw);
#pragma unroll
            for (int fb = 0; fb < E2_SNFB; fb++)
                acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
        }
    }
    /* The epilogue is vectorized because at K=64 it is the kernel: a scalar form costs six
     * instructions and two global loads per output element, which measured 150 -> 101 ms over the
     * tower with the two loads deleted alone. The eight weight scales and the eight zero-point sums
     * of a feature block are one 32-byte load each, the accumulator is already a vector, and the
     * eight results are one 16-byte store. */
    const int kb = n0 >> 5; /* the whole 32-key tile lives in one block */
    __global half *yb = (__global half *)y;
    /* Per-32-key block maxima, one per (row, block), masked to the real keys. The softmax reads
     * these to get the row max without a pass over the score row, so it reads the row once. */
    float bmm[E2_SNTB];
#pragma unroll
    for (int tc = 0; tc < E2_SNTB; tc++) bmm[tc] = -3.0e38f;
#pragma unroll
    for (int fb = 0; fb < E2_SNFB; fb++) {
        const int n = n0 + fb * 8;
        const float8 w = vload8(0, wsc + n);
        const int8 bs = vload8(0, bsq + n);
#pragma unroll
        for (int tc = 0; tc < E2_SNTB; tc++) {
            const int row = m0 + tc * 8 + lane;
            if (row >= M) continue;
            const float a = asc[(size_t)row * astride];
            const float8 f = convert_float8(acc[tc][fb] - (int8)(128) * bs) * w * a;
            float8 fm = f;
#pragma unroll
            for (int i = 0; i < 8; i++)
                if (n + i >= slen) fm[i] = -3.0e38f;
            bmm[tc] = fmax(bmm[tc], fmax(fmax(fm.s0, fm.s1), fmax(fmax(fm.s2, fm.s3),
                                                   fmax(fmax(fm.s4, fm.s5), fmax(fm.s6, fm.s7)))));
            vstore_half8(f, 0, yb + ((size_t)kb * ypitch + row) * 32 + fb * 8);
        }
    }
    const int nb32 = N >> 5;
#pragma unroll
    for (int tc = 0; tc < E2_SNTB; tc++) {
        const int row = m0 + tc * 8 + lane;
        if (row < M) bmax[(size_t)row * nb32 + kb] = bmm[tc];
    }
}

/* scores -> P, one sub-group per query row. No window and no valid mask (the tower is dense after the
 * gather), so the only masking is the padded key tail: keys >= slen stay at -3e38, which contributes
 * nothing to the max and exp()s to zero.
 *
 * One pass, and the row max is free. `k_vattn_scores` writes a per-32-key block maximum next to the
 * scores (region A_VSBM), so this kernel derives the row max with one sub-group max and the
 * per-128-key block maxima with one shuffle each, then reads the score row exactly once to exp and
 * quantize it. The quantized byte is round(127 e_k / max_e_b) and the per-row scale is
 * max_e_b / (127 l), so the byte does not depend on l at all - only the scale does, and that is 20
 * floats written after the sum. It reads the blocked fp16 score layout k_vattn_scores writes, 8
 * contiguous bytes per lane.
 *
 * Measured, this kernel is bound by none of the obvious things: deleting the exp is ~2 % of it, and
 * spreading the block-max reads over the lanes (the form below) vs reading all four per lane is
 * within noise. What is left is the score row's DRAM access pattern - the row's 128-key blocks are
 * a whole padded row apart in the layout the scores kernel's coalesced store needs - and that is
 * therefore where the remaining time is (see engine/README.md, *Vision*). */
__attribute__((intel_reqd_sub_group_size(32))) __kernel void k_vattn_softmax(
    __global const float *sc, __global const float *bmax, __global uchar *p, __global float *asc,
    int M, int N, int ypitch, int slen, int nb) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int lane = get_local_id(0);
    if (row >= M) return;
    const int nb32 = N >> 5;
    /* One lane per 128-key block holds that block's max (nb <= 32 here, the padded key count over
     * 128), so any lane can fetch any block's max with a shuffle instead of re-loading it. */
    float myb = -3.0e38f;
    if (lane < nb) {
        const float4 bv = vload4(0, bmax + (size_t)row * nb32 + lane * 4);
        myb = fmax(fmax(bv.x, bv.y), fmax(bv.z, bv.w));
    }
    const float m = e2_sg_max(myb);
    float l = 0.f, mb[24];
    const int nreg = nb < 24 ? nb : 24;
    __global uchar *prow = p + (size_t)row * N;
    for (int b = 0; b < nreg; b++) {
        /* the block's exp-space max is exp(block max - row max), so no scan is needed */
        const float bm = intel_sub_group_shuffle(myb, b);
        mb[b] = (bm > -1.0e38f) ? exp(bm - m) : 0.f;
        const int j0 = b * E2_BLOCK + lane * 4;
        const float4 v = vload_half4(0, (__global const half *)sc + (size_t)(j0 >> 5) * ypitch * 32 +
                                            (size_t)row * 32 + (j0 & 31));
        float4 e = (float4)(0.f);
        e.x = (j0 + 0 < slen) ? exp(v.x - m) : 0.f;
        e.y = (j0 + 1 < slen) ? exp(v.y - m) : 0.f;
        e.z = (j0 + 2 < slen) ? exp(v.z - m) : 0.f;
        e.w = (j0 + 3 < slen) ? exp(v.w - m) : 0.f;
        l += e.x + e.y + e.z + e.w;
        int4 q = (int4)(0);
        if (mb[b] > 0.f) q = clamp(convert_int4_rte(e * (127.f / mb[b])), (int4)(0), (int4)(127));
        *(__global uchar4 *)(prow + j0) = convert_uchar4(q + (int4)(128));
    }
    l = e2_sg_sum(l);
    if (lane == 0)
        for (int b = 0; b < nreg; b++)
            asc[(size_t)row * nb + b] = (mb[b] > 0.f && l > 0.f) ? mb[b] / (127.f * l) : 0.f;
}


/* The 3x3 pooler: one work group per output group, summing the patches the host listed for it in
 * CSR form (`gstart`/`gpatch`) and scaling by sqrt(hidden)/k^2. The CSR form is what makes the
 * groups that straddle a padding boundary - and the groups the image does not have at all - free:
 * the host emits only the groups that exist, in order. */
__kernel void k_vpool(__global float *pool, __global const float *x, __global const int *gstart,
                      __global const int *gpatch, __global const int *ngp, int D, float scale) {
    const int g = get_group_id(0);
    const int lane = get_local_id(0);
    if (g >= *ngp) return;
    const int a = gstart[g], b = gstart[g + 1];
    for (int d = lane; d < D; d += 128) {
        float s = 0.f;
        for (int i = a; i < b; i++) s += x[(size_t)gpatch[i] * D + d];
        pool[(size_t)g * D + d] = s * scale;
    }
}

/* The image path's replacement for k_embed: the text tower's input embedding block is built on the
 * host (vision soft tokens scattered into the <|image|> slots and the special tokens gathered from
 * the bf16 table) and copied in, because there is no token id for a soft token. */
__kernel void k_copy_rows(__global float *dst, __global const float *src, int rows, int hidden) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);
    const int d = (get_group_id(1) * get_local_size(0) + get_local_id(0)) * 4;
    if (row >= rows || d >= hidden) return;
    *(__global float4 *)(dst + (size_t)row * hidden + d) =
        *(__global const float4 *)(src + (size_t)row * hidden + d);
}

/* ------------------------------------------------------------------ probes (tools/gpu_bench.c) */
__kernel void k_bench_work(__global float *out, int taps) {
    float a = out[get_global_id(0) & 63];
    for (int i = 0; i < taps; i++) a = fma(a, 1.000001f, 0.f);
    out[get_global_id(0) & 63] = a;
}
__kernel void k_bench_write(__global float4 *p, float v) { p[get_global_id(0)] = (float4)(v, v, v, v); }
__kernel void k_bench_read(__global const float4 *p, __global float *out) {
    float4 v = p[get_global_id(0)];
    if (v.x + v.y + v.z + v.w == 1234567.f) out[0] = 1.f;
}
/* MLP probe: `iters` independent loads in flight per work item, and the work group walks a
 * contiguous range (lane stride 1) so the stream is sequential instead of 8 MB-strided. The
 * single-load version above is latency-limited and the strided one thrashes DRAM pages; neither is
 * the machine's bandwidth. */
__kernel void k_bench_read_mlp(__global const float4 *p, __global float *out, int iters) {
    const size_t g = get_group_id(0) * get_local_size(0), l = get_local_id(0);
    const size_t span = get_local_size(0) * (size_t)iters;
    float4 acc = (float4)(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
    for (int i = 0; i < iters; i++) acc += p[g * (size_t)iters + (size_t)i * get_local_size(0) + l];
    if (acc.x + acc.y + acc.z + acc.w == 1234567.f) out[0] = 1.f;
    (void)span;
}
__kernel void k_bench_copy_mlp(__global const float4 *p, __global float4 *q, int iters) {
    const size_t g = get_group_id(0) * get_local_size(0), l = get_local_id(0);
#pragma unroll 4
    for (int i = 0; i < iters; i++)
        q[g * (size_t)iters + (size_t)i * get_local_size(0) + l] =
            p[g * (size_t)iters + (size_t)i * get_local_size(0) + l];
}
