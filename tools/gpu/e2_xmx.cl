/* e2_xmx.cl - packed-int8 GEMM on the Xe matrix engine (DPAS).
 *
 * This file needs IGC's OpenCL frontend (ocloc) because the int8 matrix
 * multiply builtin is an Intel extension; tools/e2_spv.sh detects it.
 *
 * Arithmetic (identical to src/e2_gemm.c, so the results match the CPU path):
 *
 *   y[m][n] = sum_b xblk[m][b] * wsc[n] * ( sum_k q[m][k]*w[n][k] - 128*wsum[b][n] )
 *
 * with b over 128-wide k blocks, q the activation biased by +128 and wsum[b][n]
 * the sum of the 128 int8 weights of that block.  The int32 accumulator spans
 * exactly one 128 block, then the epilogue folds in that block's activation
 * scale, so it never grows past 2^24.
 *
 * Operands.  A is the weight matrix, B the activation, which is the orientation
 * where both are one 32-byte vector load per lane per k step:
 *   A (8 features x 32 k, s8):  ap[fb][kstep][lane][r*4+c]
 *     lane l holds k = 4l..4l+3 of feature r of the 8-feature block fb.
 *   B (32 k x 8 tokens, u8):    plain row-major activations, lane = token,
 *     32 contiguous bytes of that token's row.
 *   D: lane = token, dword r = feature, so the epilogue writes y[row][col]
 *     with an 8-float store per feature block.
 * The activation is never repacked; only the weights are, once per model at
 * load time (src/e2_gpu_pack.h).
 *
 * Tiling.  A sub-group (8 lanes) computes E2_XTB*8 tokens x E2_XFB*8 features
 * with XTB*XFB int8 and the same number of float8 accumulators (2x2 is 64+64
 * registers, which is what fits).  Four sub-groups share the token tile and
 * four share the feature tile, so one work group covers E2_TM x E2_TN and the
 * operand traffic per GEMM is m/E2_TM weight passes plus n/E2_TN activation
 * passes: the tile size is what sets the arithmetic intensity, and 64x64 keeps
 * this kernel compute-bound rather than load-bound (32 tokens x 16 features was
 * 1.2 TMAC/s, 8x under the engine's 10.5 TMAC/s issue limit).
 *
 * The row tile below E2_TM tokens is padding, which costs nothing at small m:
 * those shapes are weight-bandwidth bound, and every work group still streams
 * its weight tile exactly once.
 */

#ifndef E2_XTB
#define E2_XTB 2   /* 8-token blocks per sub-group */
#endif
#ifndef E2_XFB
#define E2_XFB 2   /* 8-feature blocks per sub-group */
#endif
#ifndef E2_RSG
#define E2_RSG 8   /* sub-groups sharing the feature tile */
#endif
#ifndef E2_CSG
#define E2_CSG 4   /* sub-groups sharing the token tile */
#endif
#define E2_TM  (E2_RSG * E2_XTB * 8)   /* tokens per work group, mode A */
#define E2_TN  (E2_CSG * E2_XFB * 8)   /* features per work group */

/* A operand (weight) byte offset for one lane: the feature block is the outer
 * dimension, so a sub-group streams its weight tile as contiguous runs.  Kept
 * in bytes on a char* base - an int8* base would scale the offset by 32. */
#define XW(base, fb, kstep, lane) \
    ((size_t)((fb) * nkc + (kstep)) * 256 + (size_t)(lane) * 32)

/* One 128-wide k block: 4 DPAS per (token block, feature block).  ntokens is
 * the number of real 8-token blocks in this sub-group's row tile (1 or 2). */
static inline void xmx_block(__global const char *wp, __global const uchar *xq,
                             __global const float *xsc, __global const float *wsc,
                             float8 facc[E2_XTB][E2_XFB],
                             int b, int lane, int nkc, int nblk, int m, int xstr,
                             int xscstr, int mtok, int k, int fb0, int n0, int ntb)
{
    int8 acc[E2_XTB][E2_XFB];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++) acc[tc][fb] = (int8)(0);
#pragma unroll
    for (int t = 0; t < 4; t++) {
        const int kstep = b * 4 + t;
        int8 av[E2_XFB];
#ifdef E2_DBG_NOLOAD
        /* benchmark only: operands from registers, to price the loads */
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++) av[fb] = (int8)(lane + fb + kstep);
#else
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++)
            av[fb] = *(const __global int8 *)(wp + XW(wp, fb0 + fb, kstep, lane));
#endif
#pragma unroll
        for (int tc = 0; tc < E2_XTB; tc++) {
            const int row = mtok + tc * 8;
            int8 bv;
#ifdef E2_DBG_NOLOAD
            bv = (int8)(lane + tc + kstep);
#else
#ifdef E2_DBG_XACT
            /* benchmark only: activation transposed to [kstep][token][32 B] so a
             * sub-group's eight lane loads are one contiguous 256-byte run */
            bv = *(const __global int8 *)(xq + ((size_t)kstep * ((m + 7) & ~7) +
                                               (size_t)row) * 32);
#else
            bv = row < m
                ? *(const __global int8 *)(xq + (size_t)row * xstr + kstep * 32)
                : (int8)(0);
#endif
#endif
#pragma unroll
            for (int fb = 0; fb < E2_XFB; fb++)
                acc[tc][fb] = intel_sub_group_i8_i8_matrix_mad_k32(av[fb], bv, acc[tc][fb]);
        }
    }
#ifdef E2_DBG_NOEPI
    /* benchmark only: drop the per-block scale epilogue to price it */
#pragma unroll
    for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
        for (int tc = 0; tc < ntb; tc++) facc[tc][fb] += convert_float8(acc[tc][fb]);
    return;
#endif
#pragma unroll
    for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
        for (int tc = 0; tc < ntb; tc++) {
            const int row = mtok + tc * 8;
            const float a = xsc[(size_t)(row < m ? row : m - 1) * xscstr + b];
            facc[tc][fb] += (float8)(a) * convert_float8(acc[tc][fb]);
        }
}

/* Store this sub-group's tile: lane = row, dword = feature.  The per-column
 * weight scale is applied here, once, instead of per k block. */
static inline void xmx_store(__global float *y, const float8 facc[E2_XTB][E2_XFB],
                             __global const float *wsc, int m0, int lane, int ystr,
                             int n0, int m, int ntb)
{
#pragma unroll
    for (int tc = 0; tc < ntb; tc++) {
        const int row = m0 + tc * 8 + lane;
        if (row < m)
#pragma unroll
            for (int fb = 0; fb < E2_XFB; fb++)
                vstore8(facc[tc][fb] * vload8(0, wsc + n0 + fb * 8), 0,
                        y + (size_t)row * ystr + n0 + fb * 8);
    }
}

/* mode A: work group = E2_TM tokens x E2_TN features. */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_gemm_xmx(__global const uchar *xq, __global const float *xsc,
                         __global const char *wp, __global const float *wsc,
                         __global float *y,
                         const int m, const int k, const int n,
                         const int xstr, const int xscstr, const int ystr)
{
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int nkc = k >> 5, nblk = k >> 7;
    const int m0 = get_group_id(0) * E2_TM + (sg / E2_CSG) * (E2_XTB * 8);
    if (m0 >= m) return;
    const int n0 = get_group_id(1) * E2_TN + (sg % E2_CSG) * (E2_XFB * 8);
    if (n0 >= n) return;
    const int mtok = m0 + lane;
    const int ntb = m0 + 8 >= m ? 1 : E2_XTB;

    float8 facc[E2_XTB][E2_XFB];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++) facc[tc][fb] = (float8)(0.0f);

    for (int b = 0; b < nblk; b++)
        xmx_block(wp, xq, xsc, wsc, facc, b, lane, nkc, nblk, m, xstr, xscstr,
                  mtok, k, n0 >> 3, n0, ntb);
    xmx_store(y, facc, wsc, m0, lane, ystr, n0, m, ntb);
}

/* Matrix-engine issue probe: the same tile shape and DPAS mix as k_gemm_xmx
 * with the operand loads removed, so what it measures is the engine itself.
 * dpas/s * 2048 is the device's int8 MAC ceiling. */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_xmx_probe(__global float *out, const int nrep)
{
    const int lane = get_sub_group_local_id();
    int8 acc[E2_XTB][E2_XFB];
    int8 av[E2_XFB];
    uint8 bv[E2_XTB];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++) bv[tc] = (uint8)(lane + tc);
#pragma unroll
    for (int fb = 0; fb < E2_XFB; fb++) av[fb] = (int8)(lane + fb);
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++) acc[tc][fb] = (int8)(0);
    for (int r = 0; r < nrep; r++)
#pragma unroll
        for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_XFB; fb++)
                acc[tc][fb] = intel_sub_group_i8_u8_matrix_mad_k32(av[fb], bv[tc], acc[tc][fb]);
    float s = 0.0f;
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++)
            for (int r = 0; r < 8; r++) s += (float)acc[tc][fb][r];
    out[get_global_id(0)] = s;
}
