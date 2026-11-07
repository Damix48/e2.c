/* e2_gemm.cl - packed-int8 GEMM kernels for the Level Zero probe.
 *
 * Layout (identical to src/e2_gemm.c / tools/e2_pack.py):
 *   y[m][n] = sum over 128-wide k blocks of (int32 dot over that block) *
 *             act_scale[m][block] * wscale[n]
 *   weights: [n/16 tiles][k/4][16 rows][4 k] int8, tile stride ts bytes
 *   activations: u8 row-major [m][k], zero point 128, one f32 scale per block
 *   wsum[n][block]: precomputed i32 = sum of the 128 int8 weights in the block
 *
 * The int32 accumulator per block is exact (max |sum| < 2^24), then converted
 * to f32 once per block, exactly like the CPU kernel.
 */

/* cl_khr_integer_dot_product dot() is unusable on this stack, see the note
 * below; the kernels therefore use plain byte multiply-add only. */

/* NOTE (kept as a record): cl_khr_integer_dot_product dot() does NOT work for
 * this packed format on this toolchain:
 *   - dot(char4, uchar4) (mixed, the natural translation) silently lowers to
 *     unsigned OpDot and computes unsigned x unsigned, wrong for signed int8
 *     weights (spirv-dis confirms OpDot %uint with v4uchar operands).
 *   - dot(char4, char4) with a centered activation lowers to an external call
 *     `__spirv_Dot(Dv4_c, Dv4_c)` that IGC cannot link:
 *     "undefined reference to _Z11__spirv_DotDv4_cS_" -> zeModuleCreate fails
 *     with ZE_RESULT_ERROR_MODULE_BUILD_FAILURE.
 * So the dot() variant is dropped from the module; plain byte multiply-add and
 * the sub-group tile are the viable kernels. */

/* Plain 4-element byte multiply-add tiled kernel: ROWS activation rows x COLS
 * output columns per work-item.  COLS must divide 16 (the tile row count). */
#define E2_DEF_GEMM_PLAIN(name, ROWS, COLS)                                    \
__kernel void name(__global const uchar *xq,                                   \
                   __global const float *xs,                                   \
                   __global const char *wq,                                    \
                   __global const float *wscale,                               \
                   __global const int *wsum,                                   \
                   __global float *y,                                          \
                   const int m, const int k, const int n,                      \
                   const int nblk, const int xss, const int wss,               \
                   const ulong ts)                                             \
{                                                                              \
    int gid = get_global_id(0);                                                \
    int ntiles = (n + 15) / 16;                                                \
    int per_tile = 16 / (COLS);                                                \
    int per_row = ntiles * per_tile;                                           \
    int rg = gid / per_row;                                                    \
    int rem = gid % per_row;                                                   \
    int nb = rem / per_tile;                                                   \
    int cq = rem % per_tile;                                                   \
    int r0 = rg * (ROWS);                                                      \
    if (r0 >= m) return;                                                       \
    int rows = m - r0; if (rows > (ROWS)) rows = (ROWS);                      \
    int n0 = nb * 16 + cq * (COLS);                                            \
    int cols = n - n0; if (cols > (COLS)) cols = (COLS);                      \
    if (cols <= 0) return;                                                     \
    __global const uchar *ap = xq + (size_t)r0 * k;                            \
    __global const char *wp = wq + (size_t)nb * ts + cq * (COLS) * 4;          \
    for (int b = 0; b < nblk; b++) {                                           \
        int acc[ROWS][COLS];                                                   \
        for (int rr = 0; rr < (ROWS); rr++)                                    \
            for (int c = 0; c < (COLS); c++)                                   \
                acc[rr][c] = -128 * wsum[(size_t)(n0 + c) * wss + b];          \
        for (int kb = b * 32; kb < (b + 1) * 32; kb++) {                       \
            char4 wv[COLS];                                                    \
            for (int c = 0; c < (COLS); c++)                                   \
                wv[c] = *(__global const char4 *)(wp + (size_t)kb * 64 + c * 4); \
            for (int rr = 0; rr < (ROWS); rr++) {                              \
                uchar4 a = *(__global const uchar4 *)(ap + (size_t)rr * k +    \
                                                       (size_t)kb * 4);        \
                for (int c = 0; c < (COLS); c++) {                             \
                    acc[rr][c] += a.x * wv[c].x + a.y * wv[c].y +              \
                                  a.z * wv[c].z + a.w * wv[c].w;               \
                }                                                              \
            }                                                                  \
        }                                                                      \
        for (int rr = 0; rr < rows; rr++) {                                    \
            float s = xs[(size_t)(r0 + rr) * xss + b];                         \
            __global float *yp = y + (size_t)(r0 + rr) * n + n0;               \
            for (int c = 0; c < cols; c++) {                                   \
                float v = (float)acc[rr][c] * s * wscale[n0 + c];              \
                if (b == 0) yp[c] = v; else yp[c] += v;                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

E2_DEF_GEMM_PLAIN(gemm_1x4, 1, 4)
E2_DEF_GEMM_PLAIN(gemm_4x4, 4, 4)
E2_DEF_GEMM_PLAIN(gemm_2x8, 2, 8)

/* Sub-group kernel: a 16-lane sub-group handles one full 16-row weight tile
 * for one activation row.  Each lane reads its own 4 weight bytes (the 16
 * lanes together form the 64-byte tile row, coalesced) and the 4 activation
 * bytes are loaded once then broadcast across the sub-group. */
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_sg(__global const uchar *xq,
                      __global const float *xs,
                      __global const char *wq,
                      __global const float *wscale,
                      __global const int *wsum,
                      __global float *y,
                      const int m, const int k, const int n,
                      const int nblk, const int xss, const int wss,
                      const ulong ts)
{
    int lane = get_sub_group_local_id();
    int gsub = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    int ntiles = (n + 15) / 16;
    int r = gsub / ntiles;
    int nb = gsub % ntiles;
    if (r >= m) return;
    int n0 = nb * 16 + lane;
    if (n0 >= n) return;
    __global const uchar *ap = xq + (size_t)r * k;
    __global const char *wp = wq + (size_t)nb * ts;
    float sc = wscale[n0];
    for (int b = 0; b < nblk; b++) {
        int acc = -128 * wsum[(size_t)n0 * wss + b];
        for (int kb = b * 32; kb < (b + 1) * 32; kb++) {
            uint a = 0;
            if (lane == 0)
                a = *(__global const uint *)(ap + (size_t)kb * 4);
            a = sub_group_broadcast(a, 0);
            uchar4 av = as_uchar4(a);
            char4 wv = as_char4(*(__global const uchar4 *)(wp + (size_t)kb * 64 + lane * 4));
            acc += av.x * wv.x + av.y * wv.y + av.z * wv.z + av.w * wv.w;
        }
        float s = xs[(size_t)r * xss + b];
        float v = (float)acc * s * sc;
        __global float *yp = y + (size_t)r * n + n0;
        if (b == 0) *yp = v; else *yp += v;
    }
}

/* Sub-group kernel, 4 activation rows x 16 output columns per sub-group: the
 * per-lane weight load is reused across 4 rows. */
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_sg4(__global const uchar *xq,
                       __global const float *xs,
                       __global const char *wq,
                       __global const float *wscale,
                       __global const int *wsum,
                       __global float *y,
                       const int m, const int k, const int n,
                       const int nblk, const int xss, const int wss,
                       const ulong ts)
{
    int lane = get_sub_group_local_id();
    int gsub = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    int ntiles = (n + 15) / 16;
    int rg = gsub / ntiles;
    int nb = gsub % ntiles;
    int r0 = rg * 4;
    if (r0 >= m) return;
    int rows = m - r0; if (rows > 4) rows = 4;
    int n0 = nb * 16 + lane;
    if (n0 >= n) return;
    __global const uchar *ap = xq + (size_t)r0 * k;
    __global const char *wp = wq + (size_t)nb * ts;
    float sc = wscale[n0];
    for (int b = 0; b < nblk; b++) {
        int acc[4];
        for (int rr = 0; rr < 4; rr++) acc[rr] = -128 * wsum[(size_t)n0 * wss + b];
        for (int kb = b * 32; kb < (b + 1) * 32; kb++) {
            uint av[4];
            for (int rr = 0; rr < 4; rr++) {
                uint a = 0;
                if (lane == 0)
                    a = *(__global const uint *)(ap + (size_t)rr * k + (size_t)kb * 4);
                av[rr] = sub_group_broadcast(a, 0);
            }
            char4 wv = as_char4(*(__global const uchar4 *)(wp + (size_t)kb * 64 + lane * 4));
            for (int rr = 0; rr < 4; rr++) {
                uchar4 a = as_uchar4(av[rr]);
                acc[rr] += a.x * wv.x + a.y * wv.y + a.z * wv.z + a.w * wv.w;
            }
        }
        for (int rr = 0; rr < rows; rr++) {
            float s = xs[(size_t)(r0 + rr) * xss + b];
            float v = (float)acc[rr] * s * sc;
            __global float *yp = y + (size_t)(r0 + rr) * n + n0;
            if (b == 0) *yp = v; else *yp += v;
        }
    }
}

/* Streaming read: sum int4 vectors over a buffer to find the DRAM ceiling. */
__kernel void bw_read(__global const int4 *p, __global float *out,
                      const int nv, const int iters)
{
    int gid = get_global_id(0);
    int n = get_global_size(0);
    int4 acc = (int4)(0, 0, 0, 0);
    for (int it = 0; it < iters; it++)
        for (int i = gid; i < nv; i += n)
            acc += p[i];
    out[gid] = (float)(acc.x + acc.y + acc.z + acc.w);
}
