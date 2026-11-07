/* e2_dpas.cl - packed-int8 GEMM on the Xe matrix engine (DPAS).
 *
 * Compiled by IGC's frontend (ocloc) because the int8 matrix multiply builtin
 * is an Intel extension that clang's OpenCL frontend does not know:
 *
 *     ocloc compile -file tools/gpu/e2_dpas.cl -device <id> -out_dir <dir>
 *
 * Layouts (both operands, same shape): [block][kchunk][lane][8 x u32]
 *   block  = 8 rows (activations) or 8 output columns (weights)
 *   kchunk = 32 k values, lane = which group of 4 k, component = row/column
 *   activations: lane l, component i -> row 8*block+i, k = 32*kchunk+4*l..+3
 *   weights:     lane l, component i -> col 8*block+l, k = 32*kchunk+4*i..+3
 * which is exactly what intel_sub_group_u8_i8_matrix_mad_k32 wants on a device
 * with subgroup size 8: A (M=8 rows x K=32) with 4 k per 32-bit component per
 * lane, B (K=32 x N=8) with column = lane, and C = 8 rows of column `lane`.
 */

/* f32 rows -> packed u8, one f32 scale per row (zero point 128). */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_pack_a(__global const float *x, __global uchar *dst,
                       __global const float *xs, const int m, const int k)
{
    int gid = get_global_id(0);
    int lane = gid & 7;
    int rest = gid >> 3;
    int nkc = k >> 5;
    int kc = rest % nkc;
    int rb = rest / nkc;
    int base = rb * 8;
    __global uint *out = (__global uint *)(dst + (size_t)((rb * nkc + kc) * 8 + lane) * 32);
    int k0 = kc * 32 + 4 * lane;
    for (int r = 0; r < 8; r++) {
        int row = base + r;
        uint v = 0;
        if (row < m) {
            float inv = 1.0f / xs[row];
            const __global float *xp = x + (size_t)row * k + k0;
            uchar4 q;
            q.x = (uchar)clamp((int)rint(xp[0] * inv) + 128, 0, 255);
            q.y = (uchar)clamp((int)rint(xp[1] * inv) + 128, 0, 255);
            q.z = (uchar)clamp((int)rint(xp[2] * inv) + 128, 0, 255);
            q.w = (uchar)clamp((int)rint(xp[3] * inv) + 128, 0, 255);
            v = as_uint(q);
        }
        out[r] = v;
    }
}

/* packed u8 rows (already quantised, row-major) -> DPAS layout */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_pack_u8(__global const uchar *src, __global uchar *dst,
                        const int m, const int k)
{
    int gid = get_global_id(0);
    int lane = gid & 7;
    int rest = gid >> 3;
    int nkc = k >> 5;
    int kc = rest % nkc;
    int rb = rest / nkc;
    int base = rb * 8;
    __global uint *out = (__global uint *)(dst + (size_t)((rb * nkc + kc) * 8 + lane) * 32);
    int k0 = kc * 32 + 4 * lane;
    for (int r = 0; r < 8; r++) {
        int row = base + r;
        out[r] = (row < m) ? *(const __global uint *)(src + (size_t)row * k + k0) : 0u;
    }
}

/* Model weight tile (int8, [n/16][k/4][16][4] with tile stride ts) -> DPAS
 * layout.  Also sums each column so the activation zero point can be removed
 * with one multiply-add per output. */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_pack_wq(__global const char *src, __global uchar *dst,
                        __global int *wsum, const int n, const int k,
                        const ulong ts)
{
    int gid = get_global_id(0);
    int lane = gid & 7;
    int rest = gid >> 3;
    int nkc = k >> 5;
    int kc = rest % nkc;
    int cb = rest / nkc;
    int col = cb * 8 + lane;
    int tile = col >> 4;
    int in_tile = col & 15;
    __global uint *out = (__global uint *)(dst + (size_t)((cb * nkc + kc) * 8 + lane) * 32);
    __global const char *sp = src + (size_t)tile * ts + (size_t)(kc * 8) * 64 + in_tile * 4;
    int s = 0;
    for (int c = 0; c < 8; c++) {
        uint v = (col < n) ? *(const __global uint *)(sp + (size_t)c * 64) : 0u;
        out[c] = v;
        char4 q = as_char4(v);
        s += q.x + q.y + q.z + q.w;
    }
    if (col < n) atomic_add(&wsum[col], s);   /* zeroed by the caller */
}

/* y[m][n] = (sum_k a*b - 128*sum(w)) * xscale[row] * wscale[col]
 * RB=4 row blocks x CT=2 column tiles per sub-group. */
#define RB 4
#define CT 2
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void k_gemm_dpas(__global const uchar *xp, __global const float *xscale,
                          __global const uchar *wp, __global const float *wscale,
                          __global const int *wsum, __global float *y,
                          const int m, const int k, const int n)
{
    int lane = get_sub_group_local_id();
    int sg = get_global_id(0) >> 3;
    int nkc = k >> 5;
    int cts = n / (8 * CT);
    int rg = sg / cts;
    int cg = sg - rg * cts;
    int rb0 = rg * RB;
    int ct0 = cg * CT;
    int8 acc[RB][CT];
    for (int a = 0; a < RB; a++)
        for (int b = 0; b < CT; b++) acc[a][b] = (int8)(0);
    __global const uchar *ap = xp + (size_t)rb0 * 256 * nkc + lane * 32;
    __global const uchar *bp = wp + (size_t)ct0 * 256 * nkc + lane * 32;
    for (int kc = 0; kc < nkc; kc++) {
        uint8 av[RB];
        for (int a = 0; a < RB; a++)
            av[a] = vload8(0, (const __global uint *)(ap + ((size_t)a * nkc + kc) * 256));
        for (int b = 0; b < CT; b++) {
            int8 bv = as_int8(vload8(0, (const __global uint *)(bp + ((size_t)b * nkc + kc) * 256)));
            for (int a = 0; a < RB; a++)
                acc[a][b] = intel_sub_group_u8_i8_matrix_mad_k32(av[a], bv, acc[a][b]);
        }
    }
    for (int b = 0; b < CT; b++) {
        int col = 8 * (ct0 + b) + lane;
        if (col >= n) continue;
        int8 corr = (int8)(128 * wsum[col]);
        float ws = wscale[col];
        for (int a = 0; a < RB; a++) {
            int r0 = 8 * (rb0 + a);
            int8 v = acc[a][b] - corr;
            for (int r = 0; r < 8; r++)
                if (r0 + r < m)
                    y[(size_t)(r0 + r) * n + col] = (float)v[r] * xscale[r0 + r] * ws;
        }
    }
}
