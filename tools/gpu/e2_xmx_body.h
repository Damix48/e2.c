/* e2_xmx_body.h - the matrix-engine GEMM body, shared by the tile variants.
 *
 * Included by e2_xmx.cl (small token tile, best below ~384 rows) and
 * e2_xmx_big.cl (large token tile, which is where the weight passes stop
 * dominating).  The tile macros must be defined before inclusion; nothing here
 * may hardcode them.
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
#define E2_TM  (E2_RSG * E2_XTB * 8)   /* tokens per work group */
#define E2_TN  (E2_CSG * E2_XFB * 8)   /* features per work group */
#define E2_NT  (E2_RSG * E2_XTB * 8)   /* local token slots for the SLM reduce */

/* A operand (weight) byte offset for one lane: the feature block is the outer
 * dimension, so a sub-group streams its weight tile as contiguous runs.  Kept
 * in bytes on a char* base - an int8* base would scale the offset by 32. */
#define XW(base, fb, kstep, lane) \
    ((size_t)((fb) * nkc + (kstep)) * 256 + (size_t)(lane) * 32)

/* One activation block of BSTEP DPAS k-steps, then that block's scale.
 * `ntb` is the number of real 8-token blocks in this sub-group's row tile.
 * Instantiated twice because the block width is what the k loop is unrolled
 * around: 128-wide for the projections, 64-wide for the MLP intermediate
 * (one work-group column tile), which is finer and therefore at least as
 * accurate. */
#define DEF_XMX_BLOCK(NAME, BSTEP)                                            \
static inline void NAME(const __global char *wp, const __global char *xq,      \
                        const __global float *xsc,                            \
                        float8 facc[E2_XTB][E2_XFB], int b, int lane,          \
                        int nkc, int m, int xstr, int xscstr, int mtok,        \
                        int fb0, int ntb)                                     \
{                                                                             \
    int8 acc[E2_XTB][E2_XFB];                                                 \
    _Pragma("unroll")                                                         \
    for (int tc = 0; tc < E2_XTB; tc++)                                       \
        _Pragma("unroll")                                                     \
        for (int fb = 0; fb < E2_XFB; fb++) acc[tc][fb] = (int8)(0);           \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < (BSTEP); t++) {                                       \
        const int kstep = b * (BSTEP) + t;                                    \
        int8 av[E2_XFB];                                                      \
        _Pragma("unroll")                                                     \
        for (int fb = 0; fb < E2_XFB; fb++)                                   \
            av[fb] = *(const __global int8 *)(wp + XW(wp, fb0 + fb, kstep, lane)); \
        _Pragma("unroll")                                                     \
        for (int tc = 0; tc < E2_XTB; tc++) {                                 \
            const int row = mtok + tc * 8;                                    \
            int8 bv = row < m                                                  \
                ? *(const __global int8 *)(xq + (size_t)row * xstr + kstep * 32) \
                : (int8)(0);                                                   \
            _Pragma("unroll")                                                 \
            for (int fb = 0; fb < E2_XFB; fb++)                               \
                acc[tc][fb] = intel_sub_group_i8_i8_matrix_mad_k32(av[fb], bv, \
                                                                  acc[tc][fb]); \
        }                                                                     \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int fb = 0; fb < E2_XFB; fb++)                                       \
        _Pragma("unroll")                                                     \
        for (int tc = 0; tc < ntb; tc++) {                                    \
            const int row = mtok + tc * 8;                                    \
            const float a = xsc[(size_t)(row < m ? row : m - 1) * xscstr + b]; \
            facc[tc][fb] += (float8)(a) * convert_float8(acc[tc][fb]);         \
        }                                                                     \
}
DEF_XMX_BLOCK(xmx_block4, 4)
DEF_XMX_BLOCK(xmx_block2, 2)

/* Store this sub-group's tile: lane = row, dword = feature.  The per-column
 * weight scale is applied here, once, instead of per k block. */
static inline void xmx_store(__global float *y, const float8 facc[E2_XTB][E2_XFB],
                             __global const float *wsc, int m0, int lane, int ystr,
                             int n0, int m, int ntb)
{
#ifdef E2_DBG_NOSTORE
    /* benchmark only: price the epilogue store.  The guarded store keeps the
     * accumulator live so the compiler cannot delete the whole GEMM. */
    if (m0 == -1 && n0 == -1)
        y[0] = facc[0][0][0] * wsc[0] + lane + ystr + m + ntb;
    return;
#endif
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

static inline void xmx_gemm(const __global char *xq, const __global float *xsc,
                            const __global char *wp, const __global float *wsc,
                            __global float *y, int m, int k, int n, int xstr,
                            int xscstr, int ystr, uint gx, uint gy)
{
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int nkc = k >> 5, nblk = k >> 7;
    const int m0 = gx * E2_TM + (sg / E2_CSG) * (E2_XTB * 8);
    if (m0 >= m) return;
    const int n0 = gy * E2_TN + (sg % E2_CSG) * (E2_XFB * 8);
    if (n0 >= n) return;
    const int mtok = m0 + lane;
    const int ntb = m0 + 8 >= m ? 1 : E2_XTB;

    float8 facc[E2_XTB][E2_XFB];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++) facc[tc][fb] = (float8)(0.0f);

    for (int b = 0; b < nblk; b++)
        xmx_block4(wp, xq, xsc, facc, b, lane, nkc, m, xstr, xscstr, mtok,
                   n0 >> 3, ntb);
    xmx_store(y, facc, wsc, m0, lane, ystr, n0, m, ntb);
}

/* mode A: work group = E2_TM tokens x E2_TN features, one GEMM. */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_gemm_xmx(__global const char *xq, __global const float *xsc,
                         __global const char *wp, __global const float *wsc,
                         __global float *y,
                         const int m, const int k, const int n,
                         const int xstr, const int xscstr, const int ystr)
{
    xmx_gemm(xq, xsc, wp, wsc, y, m, k, n, xstr, xscstr, ystr,
             get_group_id(0), get_group_id(1));
}

/* Head-batched form: gridCountZ indexes a run of GEMMs that share one shape
 * and differ only by a constant stride (the attention heads).  zrep folds the
 * GQA layout in: every zrep-th head steps the weight operand, which is how two
 * query heads share one kv head.  A descriptor array would be tidier but a
 * pointer stored in global memory does not survive the SPIR-V path. */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_gemm_z(__global const char *xq, __global const float *xsc,
                       __global const char *wp, __global const float *wsc,
                       __global float *y, const int m, const int k, const int n,
                       const int xstr, const int xscstr, const int ystr,
                       const long zxq, const long zxsc, const long zwp,
                       const long zwsc, const long zy, const int zrep)
{
    const uint z = get_group_id(2);
    if (z) {
        xq += (size_t)z * zxq;
        xsc += (size_t)z * zxsc;
        y += (size_t)z * zy;
        const uint w = z / (uint)zrep;
        wp += (size_t)w * zwp;
        wsc += (size_t)w * zwsc;
    }
    xmx_gemm(xq, xsc, wp, wsc, y, m, k, n, xstr, xscstr, ystr,
             get_group_id(0), get_group_id(1));
}

/* Same GEMM, 64-wide activation blocks: the input of the MLP down projection,
 * which k_mlp_gu quantised one 64-feature tile at a time. */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_gemm_b2(__global const char *xq, __global const float *xsc,
                        __global const char *wp, __global const float *wsc,
                        __global float *y,
                        const int m, const int k, const int n,
                        const int xstr, const int xscstr, const int ystr,
                        const long zxq, const long zxsc, const long zwp,
                        const long zwsc, const long zy, const int zrep)
{
    const uint z = get_group_id(2);
    if (z) {
        xq += (size_t)z * zxq;
        xsc += (size_t)z * zxsc;
        y += (size_t)z * zy;
        const uint w = z / (uint)zrep;
        wp += (size_t)w * zwp;
        wsc += (size_t)w * zwsc;
    }
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int nkc = k >> 5, nblk = k >> 6;
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
        xmx_block2(wp, xq, xsc, facc, b, lane, nkc, m, xstr, xscstr, mtok,
                   n0 >> 3, ntb);
    xmx_store(y, facc, wsc, m0, lane, ystr, n0, m, ntb);
}

/* ------------------------------------------------------------------ */
/* fused MLP: quantise(gelu(gate) * up)                                */
/*                                                                     */
/* One work group covers E2_TM tokens x E2_TN features of the MLP      */
/* intermediate.  It runs the DPAS loop twice (gate, then up) and       */
/* combines the two in registers, so the f32 intermediate that used to  */
/* be 8 MB per layer never reaches memory.  The int8 output uses        */
/* 64-wide blocks, which is one work-group column tile: the block max   */
/* is reduced across the E2_CSG sub-groups that share the token tile    */
/* through a small SLM table.                                           */
/* ------------------------------------------------------------------ */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_mlp_gu(__global const char *xq, __global const float *xsc,
                       __global const char *wpg, __global const float *wscg,
                       __global const char *wpu, __global const float *wscu,
                       __global char *outq, __global float *outsc,
                       const int m, const int k, const int n,
                       const int xstr, const int xscstr)
{
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int csg = sg % E2_CSG;
    const int nkc = k >> 5, nblk = k >> 7;   /* input: 128-wide blocks */
    const int m0 = get_group_id(0) * E2_TM + (sg / E2_CSG) * (E2_XTB * 8);
    const int n0 = get_group_id(1) * E2_TN + csg * (E2_XFB * 8);
    /* a work group whose token or feature tile is empty still has to reach the
     * barrier below, so this cannot return early */
    const int active = (m0 < m && n0 < n);
    const int mtok = m0 + lane;
    const int ntb = m0 + 8 >= m ? 1 : E2_XTB;
    const int fb0 = n0 >> 3;

    /* up first, kept in float; then gate, combined in place */
    float8 acc[E2_XTB][E2_XFB], gacc[E2_XTB][E2_XFB];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++) acc[tc][fb] = gacc[tc][fb] = (float8)(0.0f);
    if (active) {
        for (int b = 0; b < nblk; b++)
            xmx_block4(wpu, xq, xsc, acc, b, lane, nkc, m, xstr, xscstr, mtok,
                       fb0, ntb);
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
            for (int tc = 0; tc < ntb; tc++)
                acc[tc][fb] *= vload8(0, wscu + n0 + fb * 8);
        for (int b = 0; b < nblk; b++)
            xmx_block4(wpg, xq, xsc, gacc, b, lane, nkc, m, xstr, xscstr, mtok,
                       fb0, ntb);
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
            for (int tc = 0; tc < ntb; tc++)
                gacc[tc][fb] *= vload8(0, wscg + n0 + fb * 8);
        /* r = gelu(gate) * up, back into acc */
#pragma unroll
        for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
            for (int fb = 0; fb < E2_XFB; fb++) {
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    float g = gacc[tc][fb][r];
                    float t = 0.7978845608028654f * (g + 0.044715f * g * g * g);
                    g = t > 15.0f ? g : g / (1.0f + exp(-2.0f * t));
                    acc[tc][fb][r] *= g;
                }
            }
    }

    /* per-row max of this sub-group's E2_XFB*8 features, reduced across the
     * sub-groups that share the token tile: a 64-wide block is exactly one
     * work-group column tile, which is why the down projection uses 64-wide
     * scales. */
    __local float sm[E2_NT][E2_CSG];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++) {
        const int lr = (sg / E2_CSG) * (E2_XTB * 8) + tc * 8 + lane;
        const int row = m0 + tc * 8 + lane;
        float mx = 0.0f;
        if (active && row < m) {
#pragma unroll
            for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
                for (int r = 0; r < 8; r++) mx = fmax(mx, fabs(acc[tc][fb][r]));
        }
        sm[lr][csg] = mx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!active) return;
    const int blk = n0 >> 6;
    const int nsc = n >> 6;
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++) {
        const int lr = (sg / E2_CSG) * (E2_XTB * 8) + tc * 8 + lane;
        const int row = m0 + tc * 8 + lane;
        if (row >= m) continue;
        float mx = 0.0f;
#pragma unroll
        for (int j = 0; j < E2_CSG; j++) mx = fmax(mx, sm[lr][j]);
        float s = mx / 127.0f;
        if (s < 1e-12f) s = 1e-12f;
        outsc[(size_t)row * nsc + blk] = s;
        const float inv = 1.0f / s;
        __global char *qp = outq + (size_t)row * n + n0;
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
                int v = (int)rint(acc[tc][fb][r] * inv);
                qp[fb * 8 + r] = (char)(v < -127 ? -127 : (v > 127 ? 127 : v));
            }
    }
}

/* ------------------------------------------------------------------ */
/* PLE block: quantise(gelu(x @ w^T) * ple[token][layer][feature])     */
/*                                                                     */
/* The PLE gate projection is immediately followed by a gelu, a        */
/* per-layer slice multiply and a quantise, so the f32 result is a     */
/* pure pass-through: fusing the three into the GEMM epilogue removes  */
/* three read+write passes over 1 MB per layer.  The int8 output uses  */
/* 64-wide blocks (one work-group column tile), which is what           */
/* k_gemm_b2 expects.                                                   */
/* ------------------------------------------------------------------ */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_gemm_ple(__global const char *xq, __global const float *xsc,
                         __global const char *wp, __global const float *wsc,
                         __global const float *ple, __global char *outq,
                         __global float *outsc,
                         const int m, const int k, const int n,
                         const int xstr, const int xscstr,
                         const int nl, const int li)
{
    const int lane = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int csg = sg % E2_CSG;
    const int nkc = k >> 5, nblk = k >> 7;
    const int m0 = get_group_id(0) * E2_TM + (sg / E2_CSG) * (E2_XTB * 8);
    const int n0 = get_group_id(1) * E2_TN + csg * (E2_XFB * 8);
    const int active = (m0 < m && n0 < n);
    const int mtok = m0 + lane;
    const int ntb = m0 + 8 >= m ? 1 : E2_XTB;
    const int fb0 = n0 >> 3;

    float8 facc[E2_XTB][E2_XFB];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++)
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++) facc[tc][fb] = (float8)(0.0f);
    if (active) {
        for (int b = 0; b < nblk; b++)
            xmx_block4(wp, xq, xsc, facc, b, lane, nkc, m, xstr, xscstr, mtok,
                       fb0, ntb);
        /* gelu(x*wsc) * ple, then the block max of the 64-feature tile */
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
            for (int tc = 0; tc < ntb; tc++) {
                const int row = mtok + tc * 8;
                float8 w = vload8(0, wsc + n0 + fb * 8);
                const __global float *pl = ple + ((size_t)row * nl + li) * n;
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    float g = facc[tc][fb][r] * w[r];
                    float t = 0.7978845608028654f * (g + 0.044715f * g * g * g);
                    g = t > 15.0f ? g : g / (1.0f + exp(-2.0f * t));
                    facc[tc][fb][r] = g * pl[n0 + fb * 8 + r];
                }
            }
    }

    __local float sm[E2_NT][E2_CSG];
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++) {
        const int lr = (sg / E2_CSG) * (E2_XTB * 8) + tc * 8 + lane;
        const int row = m0 + tc * 8 + lane;
        float mx = 0.0f;
        if (active && row < m) {
#pragma unroll
            for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
                for (int r = 0; r < 8; r++) mx = fmax(mx, fabs(facc[tc][fb][r]));
        }
        sm[lr][csg] = mx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!active) return;
    const int blk = n0 >> 6;
    const int nsc = n >> 6;
#pragma unroll
    for (int tc = 0; tc < E2_XTB; tc++) {
        const int lr = (sg / E2_CSG) * (E2_XTB * 8) + tc * 8 + lane;
        const int row = m0 + tc * 8 + lane;
        if (row >= m) continue;
        float mx = 0.0f;
#pragma unroll
        for (int j = 0; j < E2_CSG; j++) mx = fmax(mx, sm[lr][j]);
        float s = mx / 127.0f;
        if (s < 1e-12f) s = 1e-12f;
        outsc[(size_t)row * nsc + blk] = s;
        const float inv = 1.0f / s;
        __global char *qp = outq + (size_t)row * n + n0;
#pragma unroll
        for (int fb = 0; fb < E2_XFB; fb++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
                int v = (int)rint(facc[tc][fb][r] * inv);
                qp[fb * 8 + r] = (char)(v < -127 ? -127 : (v > 127 ? 127 : v));
            }
    }
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
