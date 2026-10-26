/* kernels.hip — HIP kernels for the e2 engine on AMD GPUs (RDNA/RDNA2/RDNA3).
 *
 * The int8 arithmetic is the engine's: int8 weights (per-output-row scale, quad-interleaved
 * [N/8][K/4][8][4] which this path reuses untouched) times int8 activations (per-128-block scale).
 * The one difference to the CPU and Intel-GPU paths is the activation bias: they store q+128 so the
 * unsigned-byte dp4a (vpdpbusd / intel u8 x i8 DPAS) can be used, and pay one 128*sum(w) zero-point
 * correction per block. RDNA's V_DOT4 is signed x signed, so this path stores q directly and the
 * correction disappears:
 *
 *     y[m,n] = sum_b asc[m,b] * wsc[n] * sdot4_acc(qa[m,b], w)
 *
 * The weight layout makes each work item's four output columns one aligned 16-byte load
 * (w[kq*8 + n&7 .. +3] inside an 8-row block), and four k-quads of one activation row one 16-byte
 * load, so the inner loop is 8 vector loads and 64 V_DOT4 per 4x4x16 tile step.
 *
 * Every kernel here is written for a 32-lane wave and reduces with __shfl_xor_sync, so the block
 * sizes are multiples of 32 and two waves per row are used where the Intel kernels used a 32-lane
 * sub-group. */
#include <hip/hip_runtime.h>
#include <stdint.h>

#define E2_BLOCK 128
#define E2_GEMV_TM 16
#define E2_GEMM_TM 8
#define E2_GEMM_TN 4
#define E2_GEMM_NG 64 /* work-item columns per block: one wave spans 32 of them */
#define E2_GEMM_MG 4  /* work-item rows per block */

/* ------------------------------------------------------------------ helpers */
static inline __device__ int e2_dot4_acc(unsigned a, int w, int c) {
#if defined(__HIP_PLATFORM_AMD__)
    return __builtin_amdgcn_sdot4((int)a, w, c, false);
#else
    int r = c;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int av = (int)(signed char)((a >> (8 * i)) & 0xff);
        const int wv = (int)(signed char)((w >> (8 * i)) & 0xff);
        r += av * wv;
    }
    return r;
#endif
}

static inline __device__ float e2_bf16f(unsigned short v) {
    return __uint_as_float((unsigned)v << 16);
}

static inline __device__ float e2_wave_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffull, v, o);
    return v;
}

static inline __device__ float e2_wave_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffull, v, o));
    return v;
}

/* ------------------------------------------------------------------ token ids -> embeddings */
extern "C" __global__ void e2k_embed(float *x, const unsigned short *table, const int *ids,
                                     const unsigned char *valid, int rows, int hidden, float scale) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int d = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (row >= rows || d >= hidden) return;
    float4 *dst = (float4 *)(x + (size_t)row * hidden + d);
    if (valid && !valid[row]) {
        *dst = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }
    const unsigned short *src = table + (size_t)ids[row] * hidden + d;
    *dst = make_float4(e2_bf16f(src[0]), e2_bf16f(src[1]), e2_bf16f(src[2]), e2_bf16f(src[3])) *
           scale;
}

/* copy of a float block (the soft-token input of the image path) */
extern "C" __global__ void e2k_copy_rows(float *x, const float *src, int rows, int hidden) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int d = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (row >= rows || d >= hidden) return;
    *(float4 *)(x + (size_t)row * hidden + d) =
        *(const float4 *)(src + (size_t)row * hidden + d);
}

/* ------------------------------------------------------------------ row-wise normalization
 * One wave per row, lane l holding elements [b*128 + l*4, +4) of block b. Same flags as the Intel
 * backend's k_rowops: 1 copy src->res, 2 quantize, 4 source is `src`, 8 add res after the norm,
 * 16 skip the norm (quantize only), 32 the o_proj->ffn second half, 64 ignore the valid mask. */
extern "C" __global__ void e2k_rowops(float *x, const float *src, float *res, signed char *qa,
                                      float *asc, const float *gamma, const float *gamma2, float eps,
                                      float prescale, float scale, int rows, int K, int nb,
                                      int flags, const unsigned char *valid) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    if (!(flags & 64) && valid && !valid[row]) return;
    float *rowp = x + (size_t)row * K;
    const float *sp = (flags & 4) ? src + (size_t)row * K : rowp;
    if (flags & 1) {
        for (int b = 0; b < nb; b++) {
            const int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            *(float4 *)(res + (size_t)row * K + k) = *(const float4 *)(sp + k);
        }
    }
    float inv = 1.f;
    if (!(flags & 16)) {
        float ss = 0.f;
        for (int b = 0; b < nb; b++) {
            const int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            const float4 v = *(const float4 *)(sp + k) * prescale;
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        inv = 1.f / sqrtf(e2_wave_sum(ss) / (float)K + eps);
    }
    for (int b = 0; b < nb; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        if (k >= K) break;
        float4 n = *(const float4 *)(sp + k) * prescale;
        if (!(flags & 16)) {
            n *= inv;
            if (gamma) n *= *(const float4 *)(gamma + k);
        }
        if (flags & 8) n += *(const float4 *)(res + (size_t)row * K + k);
        n *= scale;
        *(float4 *)(rowp + k) = n;
        if (flags & 32) *(float4 *)(res + (size_t)row * K + k) = n;
        if (flags & 2) {
            const float4 a = make_float4(fabsf(n.x), fabsf(n.y), fabsf(n.z), fabsf(n.w));
            const float amax = e2_wave_max(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
            const float sc = amax > 0.f ? amax / 127.f : 1.f;
            int4 r = make_int4((int)rintf(n.x / sc), (int)rintf(n.y / sc), (int)rintf(n.z / sc),
                               (int)rintf(n.w / sc));
            r.x = r.x > 127 ? 127 : (r.x < -127 ? -127 : r.x);
            r.y = r.y > 127 ? 127 : (r.y < -127 ? -127 : r.y);
            r.z = r.z > 127 ? 127 : (r.z < -127 ? -127 : r.z);
            r.w = r.w > 127 ? 127 : (r.w < -127 ? -127 : r.w);
            *(char4 *)(qa + (size_t)row * K + k) =
                make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z, (signed char)r.w);
            if (lane == 0) asc[(size_t)row * nb + b] = sc;
        }
    }
    if (flags & 32) {
        float ss = 0.f;
        for (int b = 0; b < nb; b++) {
            const int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            const float4 v = *(const float4 *)(rowp + k);
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float inv2 = 1.f / sqrtf(e2_wave_sum(ss) / (float)K + eps);
        for (int b = 0; b < nb; b++) {
            const int k = b * E2_BLOCK + lane * 4;
            if (k >= K) break;
            float4 n = *(const float4 *)(rowp + k) * inv2;
            if (gamma2) n *= *(const float4 *)(gamma2 + k);
            *(float4 *)(rowp + k) = n;
            const float4 a = make_float4(fabsf(n.x), fabsf(n.y), fabsf(n.z), fabsf(n.w));
            const float amax = e2_wave_max(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
            const float sc = amax > 0.f ? amax / 127.f : 1.f;
            int4 r = make_int4((int)rintf(n.x / sc), (int)rintf(n.y / sc), (int)rintf(n.z / sc),
                               (int)rintf(n.w / sc));
            r.x = r.x > 127 ? 127 : (r.x < -127 ? -127 : r.x);
            r.y = r.y > 127 ? 127 : (r.y < -127 ? -127 : r.y);
            r.z = r.z > 127 ? 127 : (r.z < -127 ? -127 : r.z);
            r.w = r.w > 127 ? 127 : (r.w < -127 ? -127 : r.w);
            *(char4 *)(qa + (size_t)row * K + k) =
                make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z, (signed char)r.w);
            if (lane == 0) asc[(size_t)row * nb + b] = sc;
        }
    }
}

/* ------------------------------------------------------------------ int8 GEMV for M <= 16
 * One work item per output row n; the M tokens are register accumulators, so the activation word
 * (one per token per k-quad) is a wave broadcast and the weight word is one coalesced load per
 * k-quad. The k dimension is split `nsl` ways so short shapes still fill the machine; the slices are
 * summed by e2k_gemv_reduce. */
extern "C" __global__ void e2k_gemv_split(const signed char *wbuf, const signed char *acts,
                                          float *part, const int *desc, int nsl, int M,
                                          int nstride) {
    const int t = blockIdx.z;
    const int sl = blockIdx.y;
    const int *dd = desc + t * 12;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int KQ = K >> 2;
    const signed char *w = wbuf + dd[0];
    const float *wsc = (const float *)(wbuf + dd[1]);
    const signed char *qa = acts + dd[7];
    const float *asc = (const float *)(acts + dd[8]);
    const float ws = wsc[n];
    const int wrow = (n >> 3) * (KQ * 8) + (n & 7);
    const int b0 = (nb * sl) / nsl, b1 = (nb * (sl + 1)) / nsl;
    float yacc[E2_GEMV_TM];
#pragma unroll
    for (int m = 0; m < E2_GEMV_TM; m++) yacc[m] = 0.f;
    for (int b = b0; b < b1; b++) {
        const int kb = b * E2_BLOCK;
        int acc[E2_GEMV_TM];
#pragma unroll
        for (int m = 0; m < E2_GEMV_TM; m++) acc[m] = 0;
#pragma unroll
        for (int kk = 0; kk < E2_BLOCK; kk += 4) {
            const int kq = (kb + kk) >> 2;
            const int wv = ((const int *)w)[wrow + (size_t)kq * 8];
#pragma unroll
            for (int m = 0; m < E2_GEMV_TM; m++) {
                const unsigned av = *(const unsigned *)(qa + (size_t)m * K + kq * 4);
                acc[m] = e2_dot4_acc(av, wv, acc[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < E2_GEMV_TM; m++)
            yacc[m] += asc[(size_t)m * nb + b] * ws * (float)acc[m];
    }
    float *p = part + ((size_t)t * nsl + sl) * E2_GEMV_TM * nstride;
    if (nsl == 1) {
        /* one slice: no partial buffer, no reduce kernel */
        float *y = (float *)((char *)acts + dd[3]);
        for (int m = 0; m < M; m++) y[(size_t)m * N + n] = yacc[m];
        return;
    }
#pragma unroll
    for (int m = 0; m < E2_GEMV_TM; m++) p[(size_t)m * nstride + n] = yacc[m];
}

extern "C" __global__ void e2k_gemv_reduce(float *acts, const float *part, const int *desc, int nsl,
                                           int M, int nstride) {
    const int t = blockIdx.y;
    const int N = desc[t * 12 + 4];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float *y = (float *)((char *)acts + desc[t * 12 + 3]);
    for (int m = 0; m < M; m++) {
        float s = 0.f;
        for (int sl = 0; sl < nsl; sl++)
            s += part[((size_t)t * nsl + sl) * E2_GEMV_TM * nstride + (size_t)m * nstride + n];
        y[(size_t)m * N + n] = s;
    }
}

/* ------------------------------------------------------------------ int8 GEMM for M > 16
 * 4x4 register tile per work item, one 16-byte activation load per row (four k-quads) and one
 * 16-byte weight load per k-quad (four output columns), 64 V_DOT4 per 16x4x4 tile step. The
 * per-128-block epilogue folds the activation scale in before the accumulator is reused. */
extern "C" __global__ void e2k_gemm(const signed char *wbuf, const signed char *acts,
                                    const int *desc, int M, int first, int count) {
    /* The column grid spans the group's tensors back to back, ceil(N/256) blocks each, so a narrow
     * tensor (k/v at N=512 next to q at N=2048) does not compute discarded column blocks. */
    int yb = blockIdx.y, t = first;
    while (t - first + 1 < count) {
        const int nt =
            (desc[t * 12 + 4] + E2_GEMM_NG * E2_GEMM_TN - 1) / (E2_GEMM_NG * E2_GEMM_TN);
        if (yb < nt) break;
        yb -= nt;
        t++;
    }
    const int *dd = desc + t * 12;
    const int N = dd[4], K = dd[5], nb = dd[6];
    const int KQ = K >> 2;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int m0 = blockIdx.x * (E2_GEMM_MG * E2_GEMM_TM) + ty * E2_GEMM_TM;
    const int n0 = yb * (E2_GEMM_NG * E2_GEMM_TN) + tx * E2_GEMM_TN;
    if (n0 >= N) return;
    const signed char *w = wbuf + dd[0];
    const float *wsc = (const float *)(wbuf + dd[1]);
    const signed char *qa = acts + dd[7];
    const float *asc = (const float *)(acts + dd[8]);
    float *y = (float *)((char *)acts + dd[3]);
    const int wrow = (n0 >> 3) * (KQ * 8) + (n0 & 7);
    const float4 wsv = *(const float4 *)(wsc + n0);
    float yacc[E2_GEMM_TM][E2_GEMM_TN];
#pragma unroll
    for (int i = 0; i < E2_GEMM_TM; i++)
#pragma unroll
        for (int j = 0; j < E2_GEMM_TN; j++) yacc[i][j] = 0.f;
    /* a warp whose whole 8-row tile is past M skips the K loop (the tail block is partial) */
    if (m0 < M)
    for (int b = 0; b < nb; b++) {
        const int kb = b * E2_BLOCK;
        int acc[E2_GEMM_TM][E2_GEMM_TN];
#pragma unroll
        for (int i = 0; i < E2_GEMM_TM; i++)
#pragma unroll
            for (int j = 0; j < E2_GEMM_TN; j++) acc[i][j] = 0;
#pragma unroll
        for (int kk = 0; kk < E2_BLOCK; kk += 16) {
            const int kq = (kb + kk) >> 2;
            uint4 av[E2_GEMM_TM];
#pragma unroll
            for (int i = 0; i < E2_GEMM_TM; i++)
                av[i] = *(const uint4 *)(qa + (size_t)(m0 + i) * K + kb + kk);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int4 wv = *(const int4 *)((const int *)w + wrow + (size_t)(kq + q) * 8);
                const unsigned *ap = (const unsigned *)&av[0];
#pragma unroll
                for (int i = 0; i < E2_GEMM_TM; i++) {
                    const unsigned a = ap[i * 4 + q];
                    acc[i][0] = e2_dot4_acc(a, wv.x, acc[i][0]);
                    acc[i][1] = e2_dot4_acc(a, wv.y, acc[i][1]);
                    acc[i][2] = e2_dot4_acc(a, wv.z, acc[i][2]);
                    acc[i][3] = e2_dot4_acc(a, wv.w, acc[i][3]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < E2_GEMM_TM; i++) {
            const float a = asc[(size_t)(m0 + i) * nb + b];
#pragma unroll
            for (int j = 0; j < E2_GEMM_TN; j++)
                yacc[i][j] += a * ((float *)&wsv)[j] * (float)acc[i][j];
        }
    }
#pragma unroll
    for (int i = 0; i < E2_GEMM_TM; i++) {
        if (m0 + i >= M) break;
        if (n0 + E2_GEMM_TN <= N) {
            *(float4 *)(y + (size_t)(m0 + i) * N + n0) =
                make_float4(yacc[i][0], yacc[i][1], yacc[i][2], yacc[i][3]);
        } else {
            for (int j = 0; j < E2_GEMM_TN; j++)
                if (n0 + j < N) y[(size_t)(m0 + i) * N + n0 + j] = yacc[i][j];
        }
    }
}

/* ------------------------------------------------------------------ int8 GEMM, LDS-tiled
The direct kernel above re-reads each activation 16x (once per column group) and each weight 4x
(once per row group) out of L1, and with only 4 SIMD units there are not enough resident waves to
hide the resulting latency: the profile shows ~200 GMAC/s against a measured ~1 TMAC/s dot4 ceiling.
This kernel stages both operands in shared memory once per work group and double-buffers the next
K-block while the current one computes, so the inner loop reads LDS (broadcast activations,
conflict-free vector weights) instead of global.

Tile: 32 rows x 128 columns x 128 K (one whole quantization block) per work group, 256 threads.
Per thread: 4 rows x 4 columns; the A operand is one LDS.128 per row per four k-quads (the row
address is a wave broadcast) and the B operand one LDS.128 per k-quad (four conflict-free lanes per
phase), 64 v_dot4 per 16x4x4 step.

Layouts in LDS: A[row][k] with a 144-byte row stride (16-byte aligned and bank-shifted so the
staging stores do not collide); B[k][n] row-major with the byte of k-quad `k` at (k*128 + n)*4,
which is exactly how the quad-interleaved global layout maps 32-byte runs. */
#define E2_LDS_BM 32
#define E2_LDS_BN 128
#define E2_LDS_BK 128
#define E2_LDS_TM 4
#define E2_LDS_TN 4
#define E2_LDS_TX 32
#define E2_LDS_TY 8
#define E2_LDS_THREADS (E2_LDS_TX * E2_LDS_TY)
#define E2_LDS_APAD 144 /* bytes per A row: 128 + 16, keeps 16-byte alignment and shifts banks */
/* 1 = stage the next K-block into the second buffer while the current one computes (41 KB of LDS,
 * one work group per CU); 0 = one buffer (21 KB, three work groups per CU, staging latency hidden
 * by the other work groups instead). Measured on gfx1036: single-buffering is 6-13 % faster on
 * every projection shape, so it is the default; -DE2_LDS_DB=1 selects the overlap variant for A/B. */
#ifndef E2_LDS_DB
#define E2_LDS_DB 0
#endif

extern "C" __global__ void e2k_gemm_lds(const signed char *wbuf, const signed char *acts,
                                        const int *desc, int M, int first, int count) {
    /* Per-tensor column blocks, back to back (see e2k_gemm). */
    int yb = blockIdx.y, t = first;
    while (t - first + 1 < count) {
        const int nt = (desc[t * 12 + 4] + E2_LDS_BN - 1) / E2_LDS_BN;
        if (yb < nt) break;
        yb -= nt;
        t++;
    }
    const int *dd = desc + t * 12;
    const int N = dd[4], K = dd[5], nb = dd[6], Npad = dd[10];
    const int KQ = K >> 2;
    const signed char *w = wbuf + dd[0];
    const float *wsc = (const float *)(wbuf + dd[1]);
    const signed char *qa = acts + dd[7];
    const float *asc = (const float *)(acts + dd[8]);
    float *y = (float *)((char *)acts + dd[3]);

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * E2_LDS_TX + tx;
    const int mb = blockIdx.x * E2_LDS_BM;
    const int nblk = yb * E2_LDS_BN;
    const int n0 = nblk + tx * E2_LDS_TN;
    const int mloc = ty * E2_LDS_TM;

    __shared__ signed char sA[E2_LDS_DB + 1][E2_LDS_BM * E2_LDS_APAD];
    __shared__ signed char sB[E2_LDS_DB + 1][E2_LDS_BK * E2_LDS_BN];

    /* staging maps, constant across K-blocks */
    const int arow = tid >> 3, acol = (tid & 7) * 16;
    const int ngroups = Npad >> 3;

    float4 wsv = *(const float4 *)(wsc + (n0 < N ? n0 : N - 4));
    float yacc[E2_LDS_TM][E2_LDS_TN];
#pragma unroll
    for (int i = 0; i < E2_LDS_TM; i++)
#pragma unroll
        for (int j = 0; j < E2_LDS_TN; j++) yacc[i][j] = 0.f;

#define E2_STAGE(buf, b)                                                                          \
    do {                                                                                          \
        const int kb = (b) * E2_LDS_BK;                                                           \
        *(uint4 *)(sA[buf] + arow * E2_LDS_APAD + acol) =                                        \
            *(const uint4 *)(qa + (size_t)(mb + arow) * K + kb + acol);                           \
        _Pragma("unroll") for (int j = 0; j < 2; j++) {                                           \
            const int cj = tid + j * E2_LDS_THREADS;                                              \
            const int kq = cj >> 4;                                                               \
            const int gg = cj & 15;                                                               \
            int g = (nblk >> 3) + gg;                                                             \
            if (g >= ngroups) g = ngroups - 1;                                                    \
            const int *src = (const int *)w + ((size_t)g * KQ + ((kb) >> 2) + kq) * 8;            \
            const int4 v0 = *(const int4 *)src;                                                   \
            const int4 v1 = *(const int4 *)(src + 4);                                             \
            signed char *dst = sB[buf] + ((kq * E2_LDS_BN) + (gg * 8)) * 4;                       \
            *(int4 *)dst = v0;                                                                    \
            *(int4 *)(dst + 16) = v1;                                                             \
        }                                                                                         \
    } while (0)

#if E2_LDS_DB
    E2_STAGE(0, 0);
    __syncthreads();
#endif

    for (int b = 0; b < nb; b++) {
#if E2_LDS_DB
        const int buf = b & 1;
        if (b + 1 < nb) E2_STAGE(buf ^ 1, b + 1);
#else
        const int buf = 0;
        E2_STAGE(0, b);
        __syncthreads();
#endif

        /* The last M-block is usually partial (196 rows -> a 4-valid-row tail): a thread whose
         * whole 4-row tile is past M skips the dot products and waits at the barrier instead.
         * ty is uniform per warp, so the branch does not diverge. Staging stays cooperative. */
        if (mb + mloc < M) {
            int acc[E2_LDS_TM][E2_LDS_TN];
#pragma unroll
            for (int i = 0; i < E2_LDS_TM; i++)
#pragma unroll
                for (int j = 0; j < E2_LDS_TN; j++) acc[i][j] = 0;
            const signed char *sap = sA[buf];
            const signed char *sbp = sB[buf];
#pragma unroll
            for (int q4 = 0; q4 < E2_LDS_BK / 16; q4++) {
                uint4 av[E2_LDS_TM];
#pragma unroll
                for (int i = 0; i < E2_LDS_TM; i++)
                    av[i] = *(const uint4 *)(sap + (mloc + i) * E2_LDS_APAD + q4 * 16);
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const int4 bv = *(const int4 *)(sbp + ((q4 * 4 + q) * E2_LDS_BN + tx * E2_LDS_TN) * 4);
                    const unsigned *ap = (const unsigned *)&av[0];
#pragma unroll
                    for (int i = 0; i < E2_LDS_TM; i++) {
                        const unsigned a = ap[i * 4 + q];
                        acc[i][0] = e2_dot4_acc(a, bv.x, acc[i][0]);
                        acc[i][1] = e2_dot4_acc(a, bv.y, acc[i][1]);
                        acc[i][2] = e2_dot4_acc(a, bv.z, acc[i][2]);
                        acc[i][3] = e2_dot4_acc(a, bv.w, acc[i][3]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < E2_LDS_TM; i++) {
                const float a = asc[(size_t)(mb + mloc + i) * nb + b];
#pragma unroll
                for (int j = 0; j < E2_LDS_TN; j++)
                    yacc[i][j] += a * ((const float *)&wsv)[j] * (float)acc[i][j];
            }
        }
        __syncthreads();
    }
#undef E2_STAGE

#pragma unroll
    for (int i = 0; i < E2_LDS_TM; i++) {
        const int row = mb + mloc + i;
        if (row >= M) break;
        if (n0 + E2_LDS_TN <= N) {
            *(float4 *)(y + (size_t)row * N + n0) =
                make_float4(yacc[i][0], yacc[i][1], yacc[i][2], yacc[i][3]);
        } else {
            for (int j = 0; j < E2_LDS_TN; j++)
                if (n0 + j < N) y[(size_t)row * N + n0 + j] = yacc[i][j];
        }
    }
}

/* ------------------------------------------------------------------ q/k/v norm + rope
 * One wave per (token, head) row of D elements; lane l holds the four elements at each 128-chunk's
 * l*4. The rope pairs element d with d + D/2 and both are held in registers before either is
 * written, so the in-place rotation cannot read a rotated value. */
static inline __device__ void e2_norm_rope_row(float *p, int D, const float *gamma, float eps,
                                               int rope, const float *cs, const float *sn, int lane) {
    const int halfd = D / 2;
    const int niter = D / 128;
    const int off0 = lane * 4;
    float4 v0 = make_float4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
    float ss = 0.f;
    if (niter > 0) { v0 = *(const float4 *)(p + off0);         ss += v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w; }
    if (niter > 1) { v1 = *(const float4 *)(p + off0 + 128);   ss += v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w; }
    if (niter > 2) { v2 = *(const float4 *)(p + off0 + 256);   ss += v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w; }
    if (niter > 3) { v3 = *(const float4 *)(p + off0 + 384);   ss += v3.x * v3.x + v3.y * v3.y + v3.z * v3.z + v3.w * v3.w; }
    const float inv = 1.f / sqrtf(e2_wave_sum(ss) / (float)D + eps);
    const int step = halfd / 128;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (niter <= j) break;
        const int d = off0 + j * 128;
        const float4 base = ((j == 0) ? v0 : (j == 1) ? v1 : (j == 2) ? v2 : v3);
        const float4 a = make_float4(base.x * inv, base.y * inv, base.z * inv, base.w * inv);
        if (!rope) {
            const float4 g = gamma ? *(const float4 *)(gamma + d) : make_float4(1, 1, 1, 1);
            *(float4 *)(p + d) =
                make_float4(a.x * g.x, a.y * g.y, a.z * g.z, a.w * g.w);
            continue;
        }
        const int first = d < halfd;
        const int pj = first ? j + step : j - step;
        const float4 pbase = ((pj == 0) ? v0 : (pj == 1) ? v1 : (pj == 2) ? v2 : v3);
        const float4 yy =
            make_float4(pbase.x * inv, pbase.y * inv, pbase.z * inv, pbase.w * inv);
        const int pd = first ? d + halfd : d - halfd;
        const float4 g = gamma ? *(const float4 *)(gamma + d) : make_float4(1, 1, 1, 1);
        const float4 gy = gamma ? *(const float4 *)(gamma + pd) : make_float4(1, 1, 1, 1);
        const float4 c = *(const float4 *)(cs + d);
        const float4 s4 = *(const float4 *)(sn + d);
        const float sg = first ? -1.f : 1.f;
        *(float4 *)(p + d) = make_float4(
            a.x * g.x * c.x + sg * yy.x * gy.x * s4.x, a.y * g.y * c.y + sg * yy.y * gy.y * s4.y,
            a.z * g.z * c.z + sg * yy.z * gy.z * s4.z, a.w * g.w * c.w + sg * yy.w * gy.w * s4.w);
    }
}

extern "C" __global__ void e2k_norm_rope3(float *xq, const float *gq, int rq, float *xk,
                                          const float *gk, int rk, float *xv, const float *gv,
                                          int rv, const float *costab, const float *sintab,
                                          float eps, int hq, int hkv, int D, int seq_len) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row < rq) {
        const int pos = (row / hq) % seq_len;
        e2_norm_rope_row(xq + (size_t)row * D, D, gq, eps, 1,
                         costab + (size_t)pos * D, sintab + (size_t)pos * D, lane);
    }
    if (row < rk) {
        const int pos = (row / hkv) % seq_len;
        e2_norm_rope_row(xk + (size_t)row * D, D, gk, eps, 1,
                         costab + (size_t)pos * D, sintab + (size_t)pos * D, lane);
    }
    if (row < rv)
        e2_norm_rope_row(xv + (size_t)row * D, D, gv, eps, 0, costab, sintab, lane);
}

/* ------------------------------------------------------------------ attention
 * One work item per (query row, head): lanes span the head dimension, the score is a 32-lane
 * butterfly, the softmax is online, and the context is quantized on the way out because it is the
 * o-projection's input. Specialised on D/32 = 8 (sliding layers) and 16 (full layers). */
#define E2_ATTN_BODY(DS)                                                                          \
    do {                                                                                          \
        float4 *o4 = (float4 *)(out + lane * DS);                                                 \
        const float4 *qp = (const float4 *)(q + (size_t)row * q_stride + h * D + lane * DS);      \
        float4 qs[DS / 4], acc[DS / 4];                                                           \
        _Pragma("unroll") for (int t = 0; t < DS / 4; t++) {                                      \
            qs[t] = qp[t];                                                                        \
            acc[t] = make_float4(0.f, 0.f, 0.f, 0.f);                                             \
        }                                                                                         \
        float m = -3.0e38f, l = 0.f;                                                              \
        for (int j = lo; j < hi; j++) {                                                           \
            const int jrow = j0 + j;                                                              \
            if (valid && !valid[jrow]) continue;                                                  \
            const float4 *kp = (const float4 *)(k + (size_t)jrow * kv_stride + kvh * D + lane * DS); \
            const float4 *vp = (const float4 *)(v + (size_t)jrow * kv_stride + kvh * D + lane * DS); \
            float4 d4 = make_float4(0.f, 0.f, 0.f, 0.f);                                          \
            _Pragma("unroll") for (int t = 0; t < DS / 4; t++) {                                  \
                d4.x += qs[t].x * kp[t].x;                                                        \
                d4.y += qs[t].y * kp[t].y;                                                        \
                d4.z += qs[t].z * kp[t].z;                                                        \
                d4.w += qs[t].w * kp[t].w;                                                        \
            }                                                                                     \
            float dots = (d4.x + d4.y) + (d4.z + d4.w);                                           \
            dots = e2_wave_sum(dots);                                                             \
            const float mn = fmaxf(m, dots);                                                      \
            const float alpha = expf(m - mn);                                                     \
            const float p = expf(dots - mn);                                                      \
            l = l * alpha + p;                                                                    \
            m = mn;                                                                               \
            _Pragma("unroll") for (int t = 0; t < DS / 4; t++) {                                  \
                acc[t].x = acc[t].x * alpha + p * vp[t].x;                                        \
                acc[t].y = acc[t].y * alpha + p * vp[t].y;                                        \
                acc[t].z = acc[t].z * alpha + p * vp[t].z;                                        \
                acc[t].w = acc[t].w * alpha + p * vp[t].w;                                        \
            }                                                                                     \
        }                                                                                         \
        const float inv = l > 0.f ? 1.f / l : 0.f;                                                \
        _Pragma("unroll") for (int t = 0; t < DS / 4; t++)                                        \
            o4[t] = make_float4(acc[t].x * inv, acc[t].y * inv, acc[t].z * inv, acc[t].w * inv);  \
        if (qa) {                                                                                 \
            const int nb = q_stride / E2_BLOCK;                                                   \
            _Pragma("unroll") for (int b = 0; b < DS / 4; b++) {                                  \
                const int kk = b * 128 + lane * 4;                                                \
                const float4 n = *(const float4 *)(out + kk);                                     \
                const float4 a = make_float4(fabsf(n.x), fabsf(n.y), fabsf(n.z), fabsf(n.w));     \
                const float amax = e2_wave_max(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));          \
                const float sc = amax > 0.f ? amax / 127.f : 1.f;                                 \
                int4 r = make_int4((int)rintf(n.x / sc), (int)rintf(n.y / sc),                    \
                                   (int)rintf(n.z / sc), (int)rintf(n.w / sc));                   \
                r.x = r.x > 127 ? 127 : (r.x < -127 ? -127 : r.x);                                \
                r.y = r.y > 127 ? 127 : (r.y < -127 ? -127 : r.y);                                \
                r.z = r.z > 127 ? 127 : (r.z < -127 ? -127 : r.z);                                \
                r.w = r.w > 127 ? 127 : (r.w < -127 ? -127 : r.w);                                \
                *(char4 *)(qa + (size_t)row * q_stride + h * D + kk) =                           \
                    make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z,              \
                               (signed char)r.w);                                                 \
                if ((lane & 7) == 0) asc[(size_t)row * nb + (h * D + kk) / E2_BLOCK] = sc;        \
            }                                                                                     \
        }                                                                                         \
    } while (0)

extern "C" __global__ void e2k_attn(float *ctx, const float *q, const float *k, const float *v,
                                    signed char *qa, float *asc, const unsigned char *valid,
                                    int rows, int H, int D, int kv_heads, int q_stride,
                                    int kv_stride, int window, int seq_len) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    const int DS = D / 32;
    if (row >= rows || DS > 16) return;
    float *out = ctx + (size_t)row * q_stride + h * D;
    if (valid && !valid[row]) {
        for (int t = 0; t < DS; t++) out[lane * DS + t] = 0.f;
        if (qa) {
            const int nb = q_stride / E2_BLOCK;
            for (int b = 0; b < DS / 4; b++) {
                const int kk = b * 128 + lane * 4;
                *(char4 *)(qa + (size_t)row * q_stride + h * D + kk) = make_char4(0, 0, 0, 0);
                if ((lane & 7) == 0) asc[(size_t)row * nb + (h * D + kk) / E2_BLOCK] = 1.f;
            }
        }
        return;
    }
    const int slen = seq_len > 0 ? seq_len : rows;
    const int sb = seq_len > 0 ? seq_len : 0;
    const int seq = sb ? row / sb : 0;
    const int pos = sb ? row - seq * sb : row;
    const int j0 = seq * sb;
    const int lo = (window > 0 && pos - window > 0) ? pos - window : 0;
    const int hi = (window > 0 && pos + window + 1 < slen) ? pos + window + 1 : slen;
    const int kvh = (h * kv_heads) / H;
    if (DS == 8) {
        E2_ATTN_BODY(8);
    } else if (DS == 16) {
        E2_ATTN_BODY(16);
    } else {
        for (int t = 0; t < DS; t++) out[lane * DS + t] = 0.f;
    }
}

/* ------------------------------------------------------------------ attention, fused + LDS-staged
The scalar kernel above is the document's top cost because one warp per (query row, head)
re-reads K and V for every query row: ~11 GB of K/V traffic for a 196-token call, 8x more than the
blocked form needs. This kernel is the blocked one: one work group is 8 query rows of one
(sequence, head), K and V are staged in LDS once per key block and shared by all eight warps, and
the online softmax + context accumulation stays in registers, so nothing of size S x S exists.

The per-lane element mapping is 4*lane + 128*t rather than lane*(D/32) + 4*t: the four lanes of one
LDS.128 read then sit 16 bytes apart, so both the activation loads and the staged K/V reads are
conflict-free (the old mapping put 8 lanes on the same bank). Each 128-element chunk t is exactly
one quantization block, which also makes the context quantization a single per-chunk wave max.

LDS: 8 keys x 512 floats x 2 (K and V) = 32 KB, so two work groups fit on a CU. */
#define E2_ATTN_BK 8
#define E2_ATTN_D4MAX 128 /* float4 per staged key row at D = 512 */

extern "C" __global__ void e2k_attn_lds(float *ctx, const float *q, const float *k, const float *v,
                                       signed char *qa, float *asc, const unsigned char *valid,
                                       int rows, int H, int D, int kv_heads, int q_stride,
                                       int kv_stride, int window, int seq_len) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int i = blockIdx.x * E2_ATTN_BK + ty; /* position in the sequence */
    const int DS = D / 32;
    if (DS > 16) return;
    const int kbase = s * seq_len;
    const int row = kbase + i;
    const int kvh = (h * kv_heads) / H;
    const int D4 = D >> 2;
    const int N4 = DS / 4; /* float4 per lane */

    __shared__ float4 sK[E2_ATTN_BK * E2_ATTN_D4MAX];
    __shared__ float4 sV[E2_ATTN_BK * E2_ATTN_D4MAX];

    const int tid = ty * 32 + tx;
    int kb0 = 0, kb1 = seq_len;
    if (window > 0) {
        const int minpos = blockIdx.x * E2_ATTN_BK;
        const int maxpos = minpos + E2_ATTN_BK - 1 < seq_len ? minpos + E2_ATTN_BK - 1 : seq_len - 1;
        kb0 = minpos - window > 0 ? ((minpos - window) / E2_ATTN_BK) * E2_ATTN_BK : 0;
        kb1 = maxpos + window + 1 < seq_len ? maxpos + window + 1 : seq_len;
    }

    const int live = i < seq_len && !(valid && !valid[row]);
    const int lo = (window > 0 && i - window > 0) ? i - window : 0;
    const int hi = (window > 0 && i + window + 1 < seq_len) ? i + window + 1 : seq_len;

    float4 qs[4], acc[4];
#pragma unroll
    for (int t = 0; t < 4; t++) {
        qs[t] = make_float4(0.f, 0.f, 0.f, 0.f);
        acc[t] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (live)
#pragma unroll
        for (int t = 0; t < 4; t++)
            if (t < N4)
                qs[t] = *(const float4 *)(q + (size_t)row * q_stride + h * D + 4 * tx + 128 * t);
    float m = -3.0e38f, l = 0.f;

    for (int kb = kb0; kb < kb1; kb += E2_ATTN_BK) {
        for (int idx = tid; idx < E2_ATTN_BK * D4; idx += 256) {
            const int r = idx / D4, c4 = idx % D4;
            const int j = kb + r;
            if (j < seq_len) {
                sK[r * E2_ATTN_D4MAX + c4] =
                    *(const float4 *)(k + ((size_t)kbase + j) * kv_stride + kvh * D + c4 * 4);
                sV[r * E2_ATTN_D4MAX + c4] =
                    *(const float4 *)(v + ((size_t)kbase + j) * kv_stride + kvh * D + c4 * 4);
            }
        }
        __syncthreads();
        if (live) {
            for (int jj = 0; jj < E2_ATTN_BK; jj++) {
                const int j = kb + jj;
                if (j >= seq_len) break;
                if (j < lo || j >= hi) continue;
                if (valid && !valid[kbase + j]) continue;
                float4 d4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
                for (int t = 0; t < 4; t++)
                    if (t < N4) {
                        const float4 kv = sK[jj * E2_ATTN_D4MAX + tx + 32 * t];
                        d4.x += qs[t].x * kv.x;
                        d4.y += qs[t].y * kv.y;
                        d4.z += qs[t].z * kv.z;
                        d4.w += qs[t].w * kv.w;
                    }
                /* base-2 softmax: the scores are scaled by log2(e) once and the two exponentials
                 * become single SFU v_exp_f32 ops (expf is that plus a range-reduction fma chain) */
                const float dots = e2_wave_sum((d4.x + d4.y) + (d4.z + d4.w)) * 1.4426950408889634f;
                const float mn = fmaxf(m, dots);
                const float alpha = exp2f(m - mn);
                const float p = exp2f(dots - mn);
                l = l * alpha + p;
                m = mn;
#pragma unroll
                for (int t = 0; t < 4; t++)
                    if (t < N4) {
                        const float4 vv = sV[jj * E2_ATTN_D4MAX + tx + 32 * t];
                        acc[t].x = acc[t].x * alpha + p * vv.x;
                        acc[t].y = acc[t].y * alpha + p * vv.y;
                        acc[t].z = acc[t].z * alpha + p * vv.z;
                        acc[t].w = acc[t].w * alpha + p * vv.w;
                    }
            }
        }
        __syncthreads();
    }

    if (!(i < seq_len)) return;
    float *out = ctx + (size_t)row * q_stride + h * D;
    if (!live) {
#pragma unroll
        for (int t = 0; t < 4; t++)
            if (t < N4) *(float4 *)(out + 4 * tx + 128 * t) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (qa) {
            const int nb = q_stride / 128;
#pragma unroll
            for (int t = 0; t < 4; t++)
                if (t < N4) {
                    *(char4 *)(qa + (size_t)row * q_stride + h * D + 4 * tx + 128 * t) =
                        make_char4(0, 0, 0, 0);
                    if (tx == 0) asc[(size_t)row * nb + (h * D) / 128 + t] = 1.f;
                }
        }
        return;
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int t = 0; t < 4; t++) {
        if (t >= N4) break;
        const float4 n = make_float4(acc[t].x * inv, acc[t].y * inv, acc[t].z * inv, acc[t].w * inv);
        *(float4 *)(out + 4 * tx + 128 * t) = n;
        if (qa) {
            const int nb = q_stride / 128;
            const float a = fmaxf(fmaxf(fabsf(n.x), fabsf(n.y)), fmaxf(fabsf(n.z), fabsf(n.w)));
            const float amax = e2_wave_max(a);
            const float sc = amax > 0.f ? amax / 127.f : 1.f;
            int4 r = make_int4((int)rintf(n.x / sc), (int)rintf(n.y / sc), (int)rintf(n.z / sc),
                               (int)rintf(n.w / sc));
            r.x = r.x > 127 ? 127 : (r.x < -127 ? -127 : r.x);
            r.y = r.y > 127 ? 127 : (r.y < -127 ? -127 : r.y);
            r.z = r.z > 127 ? 127 : (r.z < -127 ? -127 : r.z);
            r.w = r.w > 127 ? 127 : (r.w < -127 ? -127 : r.w);
            *(char4 *)(qa + (size_t)row * q_stride + h * D + 4 * tx + 128 * t) =
                make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z, (signed char)r.w);
            if (tx == 0) asc[(size_t)row * nb + (h * D) / 128 + t] = sc;
        }
    }
}

/* ------------------------------------------------------------------ attention on the dot4 datapath
 * The fused fp32 kernel above is instruction/latency bound (the profile puts it at ~46 %% of a
 * document and rising with S^2): one key costs two 32-byte loads per lane plus a 32-lane butterfly
 * and two exponentials for 16 fp32 MAC. This path is the Intel backend's decomposition instead,
 * with RDNA's signed x signed V_DOT4 in place of the XMX DPAS (so no +128 activation bias and no
 * per-block weight sums anywhere):
 *
 *   1. e2k_attn_packk    K quantized per (sequence, kv-head): one scale per key row, int8 [sup][D]
 *   2. e2k_attn_vscale/e2k_attn_vquant  V^T quantized per (sequence, kv-head): one scale per
 *                        feature dim, int8 [D][sup] - transposed so the context product reduces
 *                        over contiguous keys (two kernels, both coalesced)
 *   3. e2k_attn_scores8  scores = Q K^T through v_dot4 (Q comes quantized from e2k_rowops)
 *   4. e2k_attn_softmax8 masked softmax; P quantized per 128-key block with scale maxb/(127*l),
 *                        so the dequantized value IS the softmax weight (no separate 1/l)
 *   5. e2k_attn_ctx8     ctx = P V through v_dot4, fp32 out; a rowops quantizes it for o_proj
 *
 * Masking (sliding window, padding, keys past S) happens only in the softmax: the products run
 * over the whole padded key range [0,sup) and a masked key gets p = 0, which keeps the products
 * plain GEMM shapes. K rows past S are zero-filled by packk, so their scores are exact zeros
 * before the mask.
 *
 * Work-item mapping for the two products: one warp is 8 output rows x 32 output columns, the lane
 * is the column (key for the scores, feature dim for the context). Per 16-unit reduction step the
 * lane loads one uint4 of its own column's operand (out of L2: K rows and VT rows are 50-100 KB
 * per (sequence, kv-head) and stay resident) plus 8 broadcast uint4 of the row operand, and
 * retires 32 v_dot4 - so the dot4 pipe, not the load pipe, is the limit. Per-128-block activation
 * scales fold in after each block, exactly like e2k_gemm_lds. */
#define E2_A8_TM 8  /* output rows per warp in both attention products */

extern "C" __global__ void e2k_attn_packk(signed char *kpk, float *ksc, const float *k, int S,
                                          int sup, int D, int KV, int kv_stride) {
    const int s = blockIdx.z / KV;
    const int kv = blockIdx.z - s * KV;
    const int lane = threadIdx.x;
    const int j = blockIdx.x * blockDim.y + threadIdx.y;
    if (j >= sup) return;
    const int live = j < S;
    const float *src = k + ((size_t)(s * S + (live ? j : 0)) * kv_stride + kv * D);
    const int nc = D >> 7; /* 128-element chunks: 2 (sliding) or 4 (full) */
    float4 v0 = make_float4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
    if (live) {
        v0 = *(const float4 *)(src + lane * 4);
        if (nc > 1) v1 = *(const float4 *)(src + 128 + lane * 4);
        if (nc > 2) v2 = *(const float4 *)(src + 256 + lane * 4);
        if (nc > 3) v3 = *(const float4 *)(src + 384 + lane * 4);
    }
    float amax = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v2.x), fabsf(v2.y)), fmaxf(fabsf(v2.z), fabsf(v2.w))));
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v3.x), fabsf(v3.y)), fmaxf(fabsf(v3.z), fabsf(v3.w))));
    amax = e2_wave_max(amax);
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    signed char *dst = kpk + ((size_t)blockIdx.z * sup + j) * D;
#define E2_PACKK_Q(vv, off)                                                                       \
    do {                                                                                          \
        int4 r = make_int4((int)rintf(vv.x / sc), (int)rintf(vv.y / sc), (int)rintf(vv.z / sc),   \
                           (int)rintf(vv.w / sc));                                                \
        r.x = r.x > 127 ? 127 : (r.x < -127 ? -127 : r.x);                                        \
        r.y = r.y > 127 ? 127 : (r.y < -127 ? -127 : r.y);                                        \
        r.z = r.z > 127 ? 127 : (r.z < -127 ? -127 : r.z);                                        \
        r.w = r.w > 127 ? 127 : (r.w < -127 ? -127 : r.w);                                        \
        *(char4 *)(dst + (off) + lane * 4) =                                                      \
            make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z, (signed char)r.w);   \
    } while (0)
    E2_PACKK_Q(v0, 0);
    if (nc > 1) E2_PACKK_Q(v1, 128);
    if (nc > 2) E2_PACKK_Q(v2, 256);
    if (nc > 3) E2_PACKK_Q(v3, 384);
#undef E2_PACKK_Q
    if (lane == 0) ksc[(size_t)blockIdx.z * sup + j] = sc;
}

/* V^T quantization, split in two so both passes are coalesced: e2k_attn_vscale computes the
 * per-dim row maxima with each lane owning one dim (the warp then reads 32 consecutive dims = one
 * 128-byte line per key; a lane-per-key mapping reads one element per 2 KB-strided line instead,
 * which is what made this kernel 20x its coalesced cost), and e2k_attn_vquant transposes a 32x32
 * tile through LDS with the same lane-per-dim row reads. */
extern "C" __global__ void e2k_attn_vscale(float *vsc, const float *v, int S, int D, int KV,
                                         int kv_stride) {
    const int s = blockIdx.z / KV;
    const int kv = blockIdx.z - s * KV;
    const int d = blockIdx.x * 128 + threadIdx.y * 32 + threadIdx.x;
    if (d >= D) return;
    const float *col = v + ((size_t)s * S * kv_stride + kv * D + d);
    float amax = 0.f;
    for (int j = 0; j < S; j++) amax = fmaxf(amax, fabsf(col[(size_t)j * kv_stride]));
    vsc[(size_t)blockIdx.z * D + d] = amax > 0.f ? amax / 127.f : 1.f;
}

extern "C" __global__ void e2k_attn_vquant(signed char *vpk, const float *vsc, const float *v, int S,
                                         int sup, int D, int KV, int kv_stride) {
    const int s = blockIdx.z / KV;
    const int kv = blockIdx.z - s * KV;
    const int tx = threadIdx.x, ty = threadIdx.y; /* tx = dim, ty = key within the 32-key strip */
    const int d = blockIdx.x * 32 + tx;
    const int j0 = blockIdx.y * 32;
    __shared__ signed char tile[32][33]; /* column-padded so both the stores and the transposed reads are conflict-free */
    const float *src = v + ((size_t)s * S * kv_stride + kv * D);
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const int j = j0 + ty + p * 8;
        int q = 0;
        if (j < S) {
            const float sc = vsc[(size_t)blockIdx.z * D + d];
            q = (int)rintf(src[(size_t)j * kv_stride + d] / sc);
            q = q > 127 ? 127 : (q < -127 ? -127 : q);
        }
        tile[tx][ty + p * 8] = (signed char)q;
    }
    __syncthreads();
    signed char *dst = vpk + ((size_t)blockIdx.z * D + blockIdx.x * 32) * sup + j0;
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const int row = ty + p * 8; /* dim within the 32-dim block */
        if (blockIdx.x * 32 + row < D) dst[(size_t)row * sup + tx] = tile[row][tx];
    }
}

/* scores[m][j] = ksc[j] * sum_b qsc[m,b] * dot4(Q_b[m], K_b[j]) for one or all (sequence, head).
 * grid.z = nseq*H (sliced mode: every (s,h) gets its own scores slice) or 1 (loop mode: the host
 * pre-offsets every pointer to one (s,h)). */
extern "C" __global__ void e2k_attn_scores8(float *scr, const signed char *qpk, const float *qsc,
                                            const signed char *kpk, const float *ksc, int S,
                                            int sup, int D, int H, int KV, int q_stride) {
    const int h = blockIdx.z % H;
    const int s = blockIdx.z / H;
    const int kvh = (h * KV) / H;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int m0 = blockIdx.x * E2_A8_TM;
    const int j = blockIdx.y * (32 * 8) + warp * 32 + lane;
    if (j >= sup) return;
    const int nb = D >> 7;
    const signed char *qp = qpk + (size_t)s * S * q_stride + h * D;
    const float *qs = qsc + ((size_t)s * S * H + h) * nb;
    const signed char *kp = kpk + ((size_t)(s * KV + kvh) * sup + j) * D;
    const float ks = ksc[(size_t)(s * KV + kvh) * sup + j];
    float *srow = scr + ((size_t)blockIdx.z * S) * sup;
    float yacc[E2_A8_TM];
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++) yacc[r] = 0.f;
    for (int b = 0; b < nb; b++) {
        int acc[E2_A8_TM];
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++) acc[r] = 0;
#pragma unroll
        for (int st = 0; st < 128; st += 16) {
            const uint4 kv = *(const uint4 *)(kp + b * 128 + st);
#pragma unroll
            for (int r = 0; r < E2_A8_TM; r++) {
                const uint4 qv = *(const uint4 *)(qp + (size_t)(m0 + r) * q_stride + b * 128 + st);
                acc[r] = e2_dot4_acc(qv.x, (int)kv.x, acc[r]);
                acc[r] = e2_dot4_acc(qv.y, (int)kv.y, acc[r]);
                acc[r] = e2_dot4_acc(qv.z, (int)kv.z, acc[r]);
                acc[r] = e2_dot4_acc(qv.w, (int)kv.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++)
            yacc[r] += qs[(size_t)(m0 + r) * (H * nb) + b] * (float)acc[r];
    }
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++)
        if (m0 + r < S) srow[(size_t)(m0 + r) * sup + j] = ks * yacc[r];
}

/* scores -> P. One wave per query row: the row max, then exp + per-128-key-block maxima + the row
 * sum, then the quantization. Masked keys (window, padding, j >= S) get e = 0; a block with
 * nothing visible gets scale 0, so its P bytes are 0 and the context product skips no work but
 * adds exactly nothing. The block maxima live in psc until the row sum is known, then are
 * rescaled in place to maxb/(127*l) - the dequantized P is then e/l, the softmax weight. */
extern "C" __global__ void e2k_attn_softmax8(float *scr, signed char *pb, float *psc,
                                             const unsigned char *valid, int S, int sup, int H,
                                             int window) {
    const int s = blockIdx.z / H;
    const int lane = threadIdx.x;
    const int m = blockIdx.x * blockDim.y + threadIdx.y;
    if (m >= S) return;
    const int nbv = sup >> 7;
    float *srow = scr + ((size_t)blockIdx.z * S + m) * sup;
    signed char *prow = pb + ((size_t)blockIdx.z * S + m) * sup;
    float *psrow = psc + ((size_t)blockIdx.z * S + m) * nbv;
    const unsigned char *vv = valid ? valid + (size_t)s * S : NULL;
    const int lo = (window > 0 && m - window > 0) ? m - window : 0;
    const int hi = (window > 0 && m + window + 1 < S) ? m + window + 1 : S;
#define E2_VIS(jj) ((jj) >= lo && (jj) < hi && (jj) < S && (!vv || vv[jj]))
    float mx = -3.0e38f;
    for (int b = 0; b < nbv; b++) {
        const int j0 = b * 128 + lane * 4;
        const float4 v = *(const float4 *)(srow + j0);
        const float a0 = E2_VIS(j0 + 0) ? v.x : -3.0e38f;
        const float a1 = E2_VIS(j0 + 1) ? v.y : -3.0e38f;
        const float a2 = E2_VIS(j0 + 2) ? v.z : -3.0e38f;
        const float a3 = E2_VIS(j0 + 3) ? v.w : -3.0e38f;
        mx = fmaxf(mx, fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)));
    }
    mx = e2_wave_max(mx);
    /* one exp pass: the block maxima go to a per-lane array (every lane has mb after the
     * butterfly; a psc round-trip would be a cross-lane store->load with no fence) and P is
     * quantized on the spot - its scale 127/mb does not need the row sum. */
    float mbv[16];
    float l = 0.f;
    for (int b = 0; b < nbv; b++) {
        const int j0 = b * 128 + lane * 4;
        const float4 v = *(const float4 *)(srow + j0);
        float4 e = make_float4(0.f, 0.f, 0.f, 0.f);
        if (E2_VIS(j0 + 0)) e.x = exp2f((v.x - mx) * 1.4426950408889634f);
        if (E2_VIS(j0 + 1)) e.y = exp2f((v.y - mx) * 1.4426950408889634f);
        if (E2_VIS(j0 + 2)) e.z = exp2f((v.z - mx) * 1.4426950408889634f);
        if (E2_VIS(j0 + 3)) e.w = exp2f((v.w - mx) * 1.4426950408889634f);
        const float mb = e2_wave_max(fmaxf(fmaxf(e.x, e.y), fmaxf(e.z, e.w)));
        mbv[b] = mb;
        const float sc = mb > 0.f ? 127.f / mb : 0.f;
        const int4 r = make_int4((int)rintf(e.x * sc), (int)rintf(e.y * sc), (int)rintf(e.z * sc),
                                 (int)rintf(e.w * sc));
        *(char4 *)(prow + j0) = make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z,
                                           (signed char)r.w);
        l += e.x + e.y + e.z + e.w;
    }
    l = e2_wave_sum(l);
    const float inv = l > 0.f ? 1.f / (127.f * l) : 0.f;
    for (int b = 0; b < nbv; b++)
        if (lane == 0) psrow[b] = mbv[b] > 0.f ? mbv[b] * inv : 0.f;
#undef E2_VIS
}

/* ctx[m][d] = vsc[d] * sum_b psc[m,b] * dot4(P_b[m], VT_b[d]): same warp shape as the scores
 * kernel with the roles swapped - the lane is now the feature dim and the reduction runs over
 * keys, which is why packv stores V transposed. */
extern "C" __global__ void e2k_attn_ctx8(float *ctx, const signed char *pb, const float *psc,
                                         const signed char *vpk, const float *vsc, int S, int sup,
                                         int D, int H, int KV, int q_stride) {
    const int h = blockIdx.z % H;
    const int s = blockIdx.z / H;
    const int kvh = (h * KV) / H;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int m0 = blockIdx.x * E2_A8_TM;
    const int d = (blockIdx.y * 8 + warp) * 32 + lane;
    if (d >= D) return;
    const int nbv = sup >> 7;
    const signed char *pp = pb + (size_t)blockIdx.z * S * sup;
    const float *ps = psc + (size_t)blockIdx.z * S * nbv;
    const signed char *vp = vpk + ((size_t)(s * KV + kvh) * D + d) * sup;
    const float vs = vsc[(size_t)(s * KV + kvh) * D + d];
    float *out = ctx + (size_t)s * S * q_stride + h * D + d;
    float yacc[E2_A8_TM];
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++) yacc[r] = 0.f;
    for (int b = 0; b < nbv; b++) {
        int acc[E2_A8_TM];
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++) acc[r] = 0;
#pragma unroll
        for (int st = 0; st < 128; st += 16) {
            const uint4 vv = *(const uint4 *)(vp + b * 128 + st);
#pragma unroll
            for (int r = 0; r < E2_A8_TM; r++) {
                const uint4 pv = *(const uint4 *)(pp + (size_t)(m0 + r) * sup + b * 128 + st);
                acc[r] = e2_dot4_acc(pv.x, (int)vv.x, acc[r]);
                acc[r] = e2_dot4_acc(pv.y, (int)vv.y, acc[r]);
                acc[r] = e2_dot4_acc(pv.z, (int)vv.z, acc[r]);
                acc[r] = e2_dot4_acc(pv.w, (int)vv.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++)
            yacc[r] += ps[(size_t)(m0 + r) * nbv + b] * (float)acc[r];
    }
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++)
        if (m0 + r < S) out[(size_t)(m0 + r) * q_stride] = vs * yacc[r];
}

/* ------------------------------------------------------------------ the two attention products, LDS-staged
 * The direct kernels above put the per-lane operand (K rows for the scores, VT rows for the context)
 * straight through L1: a warp's working set is 32 rows x D bytes = 16 KB at D = 512, so with two
 * dozen warps resident the 16 KB L1 is a pure streaming buffer and every 16-byte load is an L2
 * round trip. Same story as the projections, same fix as e2k_gemm_lds: stage both operands in
 * shared memory once per 128-element block (global reads coalesced along the reduction dim, LDS
 * rows padded 16 bytes so the per-lane reads are conflict-free), double-buffer-free because the
 * staging is one small burst per block and the block loop already has the two barriers.
 *
 * Work group: 16 rows x 128 columns x all of the reduction dim, 256 threads (8 warps).
 *   scores: columns are keys,     the lane's operand is a K row   (row-major [key][dim])
 *   ctx:    columns are dims,     the lane's operand is a VT row  (row-major [dim][key])
 * Warp w covers rows r0 + 8*(w&1) and columns 32*(w>>1); the row operand is staged once and
 * broadcast to all four warps that share those rows. LDS: 128*144 + 16*144 = 20.25 KB -> three
 * work groups per CU. */
#define E2_A8L_BM 16  /* query rows per work group */
#define E2_A8L_BN 128 /* columns per work group (keys for scores, dims for ctx) */
#define E2_A8L_PAD 144 /* bytes per LDS row: 128 + 16, keeps 16-byte alignment and shifts banks */

extern "C" __global__ void e2k_attn_scores8_lds(float *scr, const signed char *qpk,
                                               const float *qsc, const signed char *kpk,
                                               const float *ksc, int S, int sup, int D, int H,
                                               int KV, int q_stride) {
    const int h = blockIdx.z % H;
    const int s = blockIdx.z / H;
    const int kvh = (h * KV) / H;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 32 + tx;
    const int lane = tx;
    const int nb = D >> 7;
    const int m0 = blockIdx.x * E2_A8L_BM;
    const int kbase = blockIdx.y * E2_A8L_BN;
    const int rgroup = ty & 1, kgroup = ty >> 1;
    const signed char *qp = qpk + (size_t)s * S * q_stride + h * D;
    const float *qs = qsc + ((size_t)s * S * H + h) * nb;
    const signed char *kp = kpk + (size_t)(s * KV + kvh) * sup * D + (size_t)kbase * D;
    const float *kss = ksc + (size_t)(s * KV + kvh) * sup + kbase;
    float *srow = scr + ((size_t)blockIdx.z * S + m0) * sup + kbase;

    __shared__ signed char sA[E2_A8L_BM * E2_A8L_PAD];
    __shared__ signed char sB[E2_A8L_BN * E2_A8L_PAD];
    /* staging maps (constant across blocks): A = 16 rows x 128 bytes (128 threads),
     * B = 128 rows x 128 bytes (256 threads x 4 passes) */
    const int arow = tid >> 3, acol = (tid & 7) * 16;

    float yacc[E2_A8_TM];
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++) yacc[r] = 0.f;
    for (int b = 0; b < nb; b++) {
        if (tid < E2_A8L_BM * 8)
            *(uint4 *)(sA + arow * E2_A8L_PAD + acol) =
                *(const uint4 *)(qp + (size_t)(m0 + arow) * q_stride + b * 128 + acol);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int brow = (tid >> 3) + p * 32, bcol = (tid & 7) * 16;
            *(uint4 *)(sB + brow * E2_A8L_PAD + bcol) =
                *(const uint4 *)(kp + (size_t)brow * D + b * 128 + bcol);
        }
        __syncthreads();
        int acc[E2_A8_TM];
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++) acc[r] = 0;
        const signed char *spb = sB + (kgroup * 32 + lane) * E2_A8L_PAD;
        const signed char *spa = sA + (rgroup * 8) * E2_A8L_PAD;
#pragma unroll
        for (int st = 0; st < 128; st += 16) {
            const uint4 kv = *(const uint4 *)(spb + st);
#pragma unroll
            for (int r = 0; r < E2_A8_TM; r++) {
                const uint4 qv = *(const uint4 *)(spa + r * E2_A8L_PAD + st);
                acc[r] = e2_dot4_acc(qv.x, (int)kv.x, acc[r]);
                acc[r] = e2_dot4_acc(qv.y, (int)kv.y, acc[r]);
                acc[r] = e2_dot4_acc(qv.z, (int)kv.z, acc[r]);
                acc[r] = e2_dot4_acc(qv.w, (int)kv.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++)
            yacc[r] += qs[(size_t)(m0 + rgroup * 8 + r) * (H * nb) + b] * (float)acc[r];
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++) {
        const int row = m0 + rgroup * 8 + r;
        if (row < S)
            srow[(size_t)(rgroup * 8 + r) * sup + kgroup * 32 + lane] =
                kss[kgroup * 32 + lane] * yacc[r];
    }
}

extern "C" __global__ void e2k_attn_ctx8_lds(float *ctx, const signed char *pb, const float *psc,
                                             const signed char *vpk, const float *vsc, int S,
                                             int sup, int D, int H, int KV, int q_stride) {
    const int h = blockIdx.z % H;
    const int s = blockIdx.z / H;
    const int kvh = (h * KV) / H;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 32 + tx;
    const int lane = tx;
    const int nbv = sup >> 7;
    const int m0 = blockIdx.x * E2_A8L_BM;
    const int dbase = blockIdx.y * E2_A8L_BN;
    const int rgroup = ty & 1, dgroup = ty >> 1;
    const signed char *pp = pb + (size_t)blockIdx.z * S * sup;
    const float *ps = psc + (size_t)blockIdx.z * S * nbv;
    const signed char *vp = vpk + ((size_t)(s * KV + kvh) * D + dbase) * sup;
    const float *vss = vsc + (size_t)(s * KV + kvh) * D + dbase;
    float *out = ctx + (size_t)s * S * q_stride + h * D + dbase;

    __shared__ signed char sA[E2_A8L_BM * E2_A8L_PAD];
    __shared__ signed char sB[E2_A8L_BN * E2_A8L_PAD];
    const int arow = tid >> 3, acol = (tid & 7) * 16;

    float yacc[E2_A8_TM];
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++) yacc[r] = 0.f;
    for (int b = 0; b < nbv; b++) {
        if (tid < E2_A8L_BM * 8)
            *(uint4 *)(sA + arow * E2_A8L_PAD + acol) =
                *(const uint4 *)(pp + (size_t)(m0 + arow) * sup + b * 128 + acol);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int brow = (tid >> 3) + p * 32, bcol = (tid & 7) * 16;
            *(uint4 *)(sB + brow * E2_A8L_PAD + bcol) =
                *(const uint4 *)(vp + (size_t)brow * sup + b * 128 + bcol);
        }
        __syncthreads();
        int acc[E2_A8_TM];
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++) acc[r] = 0;
        const signed char *spb = sB + (dgroup * 32 + lane) * E2_A8L_PAD;
        const signed char *spa = sA + (rgroup * 8) * E2_A8L_PAD;
#pragma unroll
        for (int st = 0; st < 128; st += 16) {
            const uint4 vv = *(const uint4 *)(spb + st);
#pragma unroll
            for (int r = 0; r < E2_A8_TM; r++) {
                const uint4 pv = *(const uint4 *)(spa + r * E2_A8L_PAD + st);
                acc[r] = e2_dot4_acc(pv.x, (int)vv.x, acc[r]);
                acc[r] = e2_dot4_acc(pv.y, (int)vv.y, acc[r]);
                acc[r] = e2_dot4_acc(pv.z, (int)vv.z, acc[r]);
                acc[r] = e2_dot4_acc(pv.w, (int)vv.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < E2_A8_TM; r++)
            yacc[r] += ps[(size_t)(m0 + rgroup * 8 + r) * nbv + b] * (float)acc[r];
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < E2_A8_TM; r++) {
        const int row = m0 + rgroup * 8 + r;
        if (row < S)
            out[(size_t)row * q_stride + dgroup * 32 + lane] = vss[dgroup * 32 + lane] * yacc[r];
    }
}

/* ------------------------------------------------------------------ gated-MLP / PLE epilogue */
extern "C" __global__ void e2k_gelu_mul_quant(float *gate, const float *up, const float *ple,
                                              signed char *qa, float *asc, int rows, int K, int nb,
                                              int stride, int ple_off) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    float *gp = gate + (size_t)row * K;
    for (int b = 0; b < nb; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        if (k >= K) break;
        float4 g = *(float4 *)(gp + k);
        const float4 t = make_float4(
            tanhf(0.7978845608f * (g.x + 0.044715f * g.x * g.x * g.x)),
            tanhf(0.7978845608f * (g.y + 0.044715f * g.y * g.y * g.y)),
            tanhf(0.7978845608f * (g.z + 0.044715f * g.z * g.z * g.z)),
            tanhf(0.7978845608f * (g.w + 0.044715f * g.w * g.w * g.w)));
        g = make_float4(0.5f * g.x * (1.f + t.x), 0.5f * g.y * (1.f + t.y),
                        0.5f * g.z * (1.f + t.z), 0.5f * g.w * (1.f + t.w));
        if (up) {
            const float4 u = *(const float4 *)(up + (size_t)row * K + k);
            g = make_float4(g.x * u.x, g.y * u.y, g.z * u.z, g.w * u.w);
        }
        if (ple) {
            const float4 u = *(const float4 *)(ple + (size_t)row * stride + ple_off + k);
            g = make_float4(g.x * u.x, g.y * u.y, g.z * u.z, g.w * u.w);
        }
        *(float4 *)(gp + k) = g;
        const float4 a = make_float4(fabsf(g.x), fabsf(g.y), fabsf(g.z), fabsf(g.w));
        const float amax = e2_wave_max(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        int4 r = make_int4((int)rintf(g.x / sc), (int)rintf(g.y / sc), (int)rintf(g.z / sc),
                           (int)rintf(g.w / sc));
        r.x = r.x > 127 ? 127 : (r.x < -127 ? -127 : r.x);
        r.y = r.y > 127 ? 127 : (r.y < -127 ? -127 : r.y);
        r.z = r.z > 127 ? 127 : (r.z < -127 ? -127 : r.z);
        r.w = r.w > 127 ? 127 : (r.w < -127 ? -127 : r.w);
        *(char4 *)(qa + (size_t)row * K + k) =
            make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z, (signed char)r.w);
        if (lane == 0) asc[(size_t)row * nb + b] = sc;
    }
}

/* ------------------------------------------------------------------ output: mean pool + L2
 * Launched with exactly 128 threads (four waves): the cross-wave reduction is unrolled over a
 * compile-time 4, which is both faster and what the codegen wants. A runtime `blockDim.x / 32`
 * bound here miscompiles the tail loop on this ROCm/LLVM (the read values come out zero; a minimal
 * reproducer isolates it), so the wave count is a constant. */
extern "C" __global__ void e2k_pool_l2(float *out, const float *emb, const unsigned char *valid,
                                       int rows, int dim, int seq_len) {
    const int seq = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float buf[1024];
    __shared__ float red[4];
    const int sb = seq_len > 0 ? seq_len : rows;
    const int r0 = seq_len > 0 ? seq * sb : 0;
    const int r1 = seq_len > 0 ? (seq + 1) * sb : rows;
    int cnt = 0;
    for (int r = r0; r < r1; r++)
        if (!valid || valid[r]) cnt++;
    float ss = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float s = 0.f;
        for (int r = r0; r < r1; r++) {
            if (valid && !valid[r]) continue;
            s += emb[(size_t)r * dim + d];
        }
        buf[d] = cnt > 0 ? s / (float)cnt : 0.f;
        ss += buf[d] * buf[d];
    }
    ss = e2_wave_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float t = (red[0] + red[1]) + (red[2] + red[3]);
        red[0] = t;
    }
    __syncthreads();
    const float inv = 1.f / sqrtf(red[0] + 1e-12f);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) out[(size_t)seq * dim + d] = buf[d] * inv;
}



/* ------------------------------------------------------------------ vision tower
 * The Gemma4 encoder: patch gather + quantize, the position term, per-head RMSNorm + axial 2D RoPE,
 * int8 attention (D = 64) and the 3x3 pooler. The projections reuse e2k_gemm_lds/e2k_gemv: the
 * weights are the CPU quad-interleaved int8 matrix and every projection's K is a multiple of 128.
 *
 * The attention head dim is 64, which is *not* a multiple of the E2_BLOCK=128 every text kernel is
 * built around, so the q/k/v packing and the three attention products are their own kernels. The
 * packed layouts are head-major ([h][row][64] for Q and K, [h][64][key] for V^T) so one head's
 * operands are contiguous; the scores/P scratch is one head at a time and the host loops the heads. */

/* Patch count padded to the attention's 128-key quantization block. Must match E2_HIP_VP in hip.h. */
#define E2_VPAD 2560
/* Pool grid bound, must match E2_HIP_VGROUPS in hip.h. */
#define E2_VGROUPS 512

/* Patch gather + 2*(x-0.5) + activation quantization, one row per warp. Same rescale as the Intel
 * k_vpatch_q, done here so the host never touches the 7.7 MB of pixels. */
extern "C" __global__ void e2k_vpatch_q(signed char *qa, float *asc, const float *pixels,
                                        const int *order, int P, int D) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= P) return;
    const int nb = D >> 7;
    const float *src = pixels + (size_t)order[row] * D;
    signed char *qp = qa + (size_t)row * D;
    for (int b = 0; b < nb; b++) {
        const int k = b * E2_BLOCK + lane * 4;
        float4 v = *(const float4 *)(src + k);
        v = make_float4(2.f * (v.x - 0.5f), 2.f * (v.y - 0.5f), 2.f * (v.z - 0.5f),
                        2.f * (v.w - 0.5f));
        const float4 a = make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
        const float amax = e2_wave_max(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        int4 r = make_int4((int)rintf(v.x / sc), (int)rintf(v.y / sc), (int)rintf(v.z / sc),
                           (int)rintf(v.w / sc));
        r.x = r.x > 127 ? 127 : (r.x < -127 ? -127 : r.x);
        r.y = r.y > 127 ? 127 : (r.y < -127 ? -127 : r.y);
        r.z = r.z > 127 ? 127 : (r.z < -127 ? -127 : r.z);
        r.w = r.w > 127 ? 127 : (r.w < -127 ? -127 : r.w);
        *(char4 *)(qp + k) =
            make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z, (signed char)r.w);
        if (lane == 0) asc[(size_t)row * nb + b] = sc;
    }
}

/* x = proj + pos_table[x] + pos_table[E2_V_POS + y], the bf16 table stayin on the device. */
extern "C" __global__ void e2k_vpos_add(float *x, const float *proj, const unsigned short *postab,
                                        const int *order, const int *posids, int P, int D,
                                        int npos) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int d = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (row >= P || d >= D) return;
    const int i = order[row];
    const unsigned short *ex = postab + (size_t)posids[2 * i] * D + d;
    const unsigned short *ey = postab + ((size_t)npos + (size_t)posids[2 * i + 1]) * D + d;
    const float4 p = *(const float4 *)(proj + (size_t)row * D + d);
    *(float4 *)(x + (size_t)row * D + d) =
        make_float4(p.x + e2_bf16f(ex[0]) + e2_bf16f(ey[0]),
                    p.y + e2_bf16f(ex[1]) + e2_bf16f(ey[1]),
                    p.z + e2_bf16f(ex[2]) + e2_bf16f(ey[2]),
                    p.w + e2_bf16f(ex[3]) + e2_bf16f(ey[3]));
}

/* q/k/v RMSNorm + axial 2D RoPE, one wave per (patch, head) row of 64 elements (two per lane).
 * Ported from the Intel k_vnorm_rope3: the rotation pairs element i with i+16 inside each 32-half,
 * which is lane xor 8, and the table index is picked by bit 4 (x-half 0..31, y-half 32..63). */
static inline __device__ void e2_vnorm_rope_row(float *p, const float *gamma, float eps, int rope,
                                                const float *cs, const float *sn, int lane) {
    const float2 v = *(const float2 *)(p + lane * 2);
    const float inv = 1.f / sqrtf(e2_wave_sum(v.x * v.x + v.y * v.y) * (1.f / 64.f) + eps);
    const float2 g = gamma ? *(const float2 *)(gamma + lane * 2) : make_float2(1.f, 1.f);
    const float2 a = make_float2(v.x * inv * g.x, v.y * inv * g.y);
    if (!rope) {
        *(float2 *)(p + lane * 2) = a;
        return;
    }
    const float2 y = make_float2(__shfl_xor_sync(0xffffffffull, a.x, 8),
                                 __shfl_xor_sync(0xffffffffull, a.y, 8));
    const int ci = ((lane >> 4) & 1) * 32 + (lane & 7) * 2;
    const float2 c = *(const float2 *)(cs + ci);
    const float2 s = *(const float2 *)(sn + ci);
    *(float2 *)(p + lane * 2) = (lane & 8)
                                    ? make_float2(a.x * c.x + y.x * s.x, a.y * c.y + y.y * s.y)
                                    : make_float2(a.x * c.x - y.x * s.x, a.y * c.y - y.y * s.y);
}

extern "C" __global__ void e2k_vnorm_rope3(float *xq, const float *gq, float *xk, const float *gk,
                                           float *xv, const float *gv, const float *costab,
                                           const float *sintab, float eps, int rows, int heads) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    const int pos = row / heads;
    e2_vnorm_rope_row(xq + (size_t)row * 64, gq, eps, 1, costab + (size_t)pos * 64,
                      sintab + (size_t)pos * 64, lane);
    e2_vnorm_rope_row(xk + (size_t)row * 64, gk, eps, 1, costab + (size_t)pos * 64,
                      sintab + (size_t)pos * 64, lane);
    e2_vnorm_rope_row(xv + (size_t)row * 64, gv, eps, 0, costab, sintab, lane);
}

/* Q -> int8 [H][Ppad][64], one scale per (patch, head) row. Rows are head-major so the scores
 * kernel's key-side operand is a contiguous 64-byte row per head. Source q is [P][768] (row r =
 * patch*H + head maps to q + r*64 because 768 = H*64). */
extern "C" __global__ void e2k_vattn_packq(signed char *qpk, float *qsc, const float *q, int rows,
                                           int Ppad, int H) {
    const int r = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (r >= rows) return;
    const int p = r / H, h = r % H;
    const float2 v = *(const float2 *)(q + (size_t)r * 64 + lane * 2);
    const float amax = e2_wave_max(fmaxf(fabsf(v.x), fabsf(v.y)));
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    int2 ri = make_int2((int)rintf(v.x / sc), (int)rintf(v.y / sc));
    ri.x = ri.x > 127 ? 127 : (ri.x < -127 ? -127 : ri.x);
    ri.y = ri.y > 127 ? 127 : (ri.y < -127 ? -127 : ri.y);
    signed char *dst = qpk + ((size_t)h * Ppad + p) * 64 + lane * 2;
    *(char2 *)dst = make_char2((signed char)ri.x, (signed char)ri.y);
    if (lane == 0) qsc[(size_t)h * Ppad + p] = sc;
}

/* K -> int8 [H][Ppad][64], keys past P zero-filled (the softmax masks them; zero rows score zero). */
extern "C" __global__ void e2k_vattn_packk(signed char *kpk, float *ksc, const float *k, int P,
                                           int Ppad, int H) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int h = row / Ppad, j = row - h * Ppad;
    if (h >= H) return;
    const int live = j < P;
    float2 v = make_float2(0.f, 0.f);
    if (live) v = *(const float2 *)(k + (size_t)j * 768 + (size_t)h * 64 + lane * 2);
    const float amax = e2_wave_max(fmaxf(fabsf(v.x), fabsf(v.y)));
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    int2 ri = make_int2(0, 0);
    if (live) {
        ri = make_int2((int)rintf(v.x / sc), (int)rintf(v.y / sc));
        ri.x = ri.x > 127 ? 127 : (ri.x < -127 ? -127 : ri.x);
        ri.y = ri.y > 127 ? 127 : (ri.y < -127 ? -127 : ri.y);
    }
    signed char *dst = kpk + ((size_t)h * Ppad + j) * 64 + lane * 2;
    *(char2 *)dst = make_char2((signed char)ri.x, (signed char)ri.y);
    if (lane == 0) ksc[(size_t)h * Ppad + j] = sc;
}

/* V^T -> int8 [H][64][Ppad] + one scale per (head, dim): the context product reduces over
 * contiguous keys. Two passes (dim maxima, then the LDS-transposed quantize), both coalesced along
 * the head dim, exactly like the text packv pair. */
extern "C" __global__ void e2k_vattn_vscale(float *vsc, const float *v, int P, int H) {
    const int h = blockIdx.x;
    const int d = blockIdx.y * 32 + threadIdx.x;
    if (h >= H) return;
    float amax = 0.f;
    for (int j = 0; j < P; j++) amax = fmaxf(amax, fabsf(v[(size_t)j * 768 + (size_t)h * 64 + d]));
    vsc[(size_t)h * 64 + d] = amax > 0.f ? amax / 127.f : 1.f;
}

extern "C" __global__ void e2k_vattn_vquant(signed char *vpk, const float *vsc, const float *v,
                                           int P, int Ppad, int H) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int h = blockIdx.z;
    const int d = blockIdx.x * 32 + tx;
    const int j0 = blockIdx.y * 32;
    __shared__ signed char tile[32][33];
    for (int p = 0; p < 4; p++) {
        const int j = j0 + ty + p * 8;
        int q = 0;
        if (j < P) {
            const float sc = vsc[(size_t)h * 64 + d];
            q = (int)rintf(v[(size_t)j * 768 + (size_t)h * 64 + d] / sc);
            q = q > 127 ? 127 : (q < -127 ? -127 : q);
        }
        tile[tx][ty + p * 8] = (signed char)q;
    }
    __syncthreads();
    signed char *dst = vpk + ((size_t)h * 64 + blockIdx.x * 32) * Ppad + j0;
    for (int p = 0; p < 4; p++) {
        const int row = ty + p * 8;
        dst[(size_t)row * Ppad + tx] = tile[row][tx];
    }
}

/* scores[m][j] = qsc[m] * ksc[j] * sdot4(Q[m], K[j]) for one head. One warp is 8 query rows x 32
 * keys, the lane is the key; the head dim is 64 = 16 v_dot4 per row pair. */
extern "C" __global__ void e2k_vattn_scores8(float *scr, const signed char *qpk, const float *qsc,
                                            const signed char *kpk, const float *ksc, int P,
                                            int Ppad, int h) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int m0 = blockIdx.x * 8;
    const int j = blockIdx.y * 256 + warp * 32 + lane;
    if (j >= Ppad) return;
    const signed char *qp = qpk + (size_t)h * Ppad * 64;
    const signed char *kp = kpk + ((size_t)h * Ppad + j) * 64;
    const uint4 k0 = *(const uint4 *)(kp), k1 = *(const uint4 *)(kp + 16);
    const uint4 k2 = *(const uint4 *)(kp + 32), k3 = *(const uint4 *)(kp + 48);
    const float ks = ksc[(size_t)h * Ppad + j];
    const float *qss = qsc + (size_t)h * Ppad;
    float *srow = scr + (size_t)m0 * Ppad + j;
#pragma unroll
    for (int r = 0; r < 8; r++) {
        const signed char *qrow = qp + (size_t)(m0 + r) * 64;
        const uint4 q0 = *(const uint4 *)(qrow), q1 = *(const uint4 *)(qrow + 16);
        const uint4 q2 = *(const uint4 *)(qrow + 32), q3 = *(const uint4 *)(qrow + 48);
        int acc = e2_dot4_acc(q0.x, (int)k0.x, 0);
        acc = e2_dot4_acc(q0.y, (int)k0.y, acc);
        acc = e2_dot4_acc(q0.z, (int)k0.z, acc);
        acc = e2_dot4_acc(q0.w, (int)k0.w, acc);
        acc = e2_dot4_acc(q1.x, (int)k1.x, acc);
        acc = e2_dot4_acc(q1.y, (int)k1.y, acc);
        acc = e2_dot4_acc(q1.z, (int)k1.z, acc);
        acc = e2_dot4_acc(q1.w, (int)k1.w, acc);
        acc = e2_dot4_acc(q2.x, (int)k2.x, acc);
        acc = e2_dot4_acc(q2.y, (int)k2.y, acc);
        acc = e2_dot4_acc(q2.z, (int)k2.z, acc);
        acc = e2_dot4_acc(q2.w, (int)k2.w, acc);
        acc = e2_dot4_acc(q3.x, (int)k3.x, acc);
        acc = e2_dot4_acc(q3.y, (int)k3.y, acc);
        acc = e2_dot4_acc(q3.z, (int)k3.z, acc);
        acc = e2_dot4_acc(q3.w, (int)k3.w, acc);
        if (m0 + r < P) srow[(size_t)r * Ppad] = ks * qss[m0 + r] * (float)acc;
    }
}

/* scores -> P, one warp per query row. No window and no validity mask (the gather removed the
 * padding); the only masking is the padded key tail j >= P, which stays at -3e38 and exps to 0.
 * The per-128-key block scale written is maxb/(127*l), so the dequantized int8 P is the softmax
 * weight and the context product needs no separate 1/l. */
extern "C" __global__ void e2k_vattn_softmax8(const float *scr, signed char *pb, float *psc, int P,
                                             int Ppad) {
    const int m = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (m >= P) return;
    const int nbv = Ppad >> 7;
    const float *srow = scr + (size_t)m * Ppad;
    signed char *prow = pb + (size_t)m * Ppad;
    float *psrow = psc + (size_t)m * nbv;
    float mx = -3.0e38f;
    for (int b = 0; b < nbv; b++) {
        const int j0 = b * E2_BLOCK + lane * 4;
        const float4 v = *(const float4 *)(srow + j0);
        const float a0 = (j0 + 0 < P) ? v.x : -3.0e38f;
        const float a1 = (j0 + 1 < P) ? v.y : -3.0e38f;
        const float a2 = (j0 + 2 < P) ? v.z : -3.0e38f;
        const float a3 = (j0 + 3 < P) ? v.w : -3.0e38f;
        mx = fmaxf(mx, fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)));
    }
    mx = e2_wave_max(mx);
    float mbv[E2_VPAD / E2_BLOCK];
    float l = 0.f;
    for (int b = 0; b < nbv; b++) {
        const int j0 = b * E2_BLOCK + lane * 4;
        const float4 v = *(const float4 *)(srow + j0);
        float4 e = make_float4(0.f, 0.f, 0.f, 0.f);
        if (j0 + 0 < P) e.x = exp2f((v.x - mx) * 1.4426950408889634f);
        if (j0 + 1 < P) e.y = exp2f((v.y - mx) * 1.4426950408889634f);
        if (j0 + 2 < P) e.z = exp2f((v.z - mx) * 1.4426950408889634f);
        if (j0 + 3 < P) e.w = exp2f((v.w - mx) * 1.4426950408889634f);
        const float mb = e2_wave_max(fmaxf(fmaxf(e.x, e.y), fmaxf(e.z, e.w)));
        mbv[b] = mb;
        const float sc = mb > 0.f ? 127.f / mb : 0.f;
        const int4 r = make_int4((int)rintf(e.x * sc), (int)rintf(e.y * sc), (int)rintf(e.z * sc),
                                 (int)rintf(e.w * sc));
        *(char4 *)(prow + j0) = make_char4((signed char)r.x, (signed char)r.y, (signed char)r.z,
                                           (signed char)r.w);
        l += e.x + e.y + e.z + e.w;
    }
    l = e2_wave_sum(l);
    const float inv = l > 0.f ? 1.f / (127.f * l) : 0.f;
    if (lane == 0)
        for (int b = 0; b < nbv; b++) psrow[b] = mbv[b] > 0.f ? mbv[b] * inv : 0.f;
}

/* ctx[m][h*64+d] = vsc[d] * sum_b psc[m,b] * dot4(P_b[m], VT_b[d]). One warp is 8 query rows x 32
 * dims (two warps cover the head dim), the lane is the feature dim. */
extern "C" __global__ void e2k_vattn_ctx8(float *ctx, const signed char *pb, const float *psc,
                                         const signed char *vpk, const float *vsc, int P, int Ppad,
                                         int h) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int m0 = blockIdx.x * 8;
    const int d = warp * 32 + lane;
    if (d >= 64) return;
    const int nbv = Ppad >> 7;
    const signed char *pp = pb + (size_t)m0 * Ppad;
    const float *ps = psc + (size_t)m0 * nbv;
    const signed char *vp = vpk + ((size_t)h * 64 + d) * Ppad;
    const float vs = vsc[(size_t)h * 64 + d];
    float yacc[8];
#pragma unroll
    for (int r = 0; r < 8; r++) yacc[r] = 0.f;
    for (int b = 0; b < nbv; b++) {
        int acc[8];
#pragma unroll
        for (int r = 0; r < 8; r++) acc[r] = 0;
#pragma unroll
        for (int st = 0; st < E2_BLOCK; st += 16) {
            const uint4 vv = *(const uint4 *)(vp + b * E2_BLOCK + st);
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const uint4 pv = *(const uint4 *)(pp + (size_t)r * Ppad + b * E2_BLOCK + st);
                acc[r] = e2_dot4_acc(pv.x, (int)vv.x, acc[r]);
                acc[r] = e2_dot4_acc(pv.y, (int)vv.y, acc[r]);
                acc[r] = e2_dot4_acc(pv.z, (int)vv.z, acc[r]);
                acc[r] = e2_dot4_acc(pv.w, (int)vv.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < 8; r++) yacc[r] += ps[(size_t)r * nbv + b] * (float)acc[r];
    }
#pragma unroll
    for (int r = 0; r < 8; r++)
        if (m0 + r < P) ctx[(size_t)(m0 + r) * 768 + (size_t)h * 64 + d] = vs * yacc[r];
}

/* The 3x3 pooler: one work group per output group, summing the patches the host listed for it in
 * CSR form. The count is read from the device so the fixed grid is valid for any image. */
extern "C" __global__ void e2k_vpool(float *pool, const float *x, const int *gstart,
                                     const int *gpatch, const int *ngp, int D, float scale) {
    const int g = blockIdx.x;
    const int lane = threadIdx.x;
    if (g >= *ngp) return;
    const int a = gstart[g], b = gstart[g + 1];
    for (int d = lane; d < D; d += 128) {
        float s = 0.f;
        for (int i = a; i < b; i++) s += x[(size_t)gpatch[i] * D + d];
        pool[(size_t)g * D + d] = s * scale;
    }
}

/* ---- E2_HIP_PROFILE: one hipEvent pair per launch, flushed after the profiled call */
extern "C" int e2h_prof_rec;
extern "C" void e2h_prof_ev(int slot, const char *name, hipEvent_t a, hipEvent_t b);
#define E2H_PROF(slot, kname, launch)                                                              \
    do {                                                                                           \
        if (e2h_prof_rec) {                                                                        \
            hipEvent_t _a, _b;                                                                     \
            hipEventCreate(&_a);                                                                   \
            hipEventCreate(&_b);                                                                   \
            hipEventRecord(_a, s);                                                                 \
            launch;                                                                                \
            hipEventRecord(_b, s);                                                                 \
            e2h_prof_ev(slot, kname, _a, _b);                                                      \
        } else {                                                                                   \
            launch;                                                                                \
        }                                                                                          \
    } while (0)

/* ================================================================== launchers
 * The grid geometry lives here so the C host stays free of kernel-shape details. */
extern "C" void e2h_embed(hipStream_t s, float *x, const uint16_t *table, const int32_t *ids,
                          const uint8_t *valid, int rows, int hidden, float scale) {
    dim3 grid((rows + 3) / 4, (hidden / 4 + 63) / 64, 1);
    dim3 block(64, 4, 1);
    E2H_PROF(0, "embed", (e2k_embed<<<grid, block, 0, s>>>(x, table, ids, valid, rows, hidden, scale)));
}

extern "C" void e2h_copy_rows(hipStream_t s, float *x, const float *src, int rows, int hidden) {
    dim3 grid((rows + 3) / 4, (hidden / 4 + 63) / 64, 1);
    dim3 block(64, 4, 1);
    E2H_PROF(1, "copy_rows", (e2k_copy_rows<<<grid, block, 0, s>>>(x, src, rows, hidden)));
}

extern "C" void e2h_rowops(hipStream_t s, float *x, const float *src, float *res, int8_t *qa,
                           float *asc, const float *gamma, const float *gamma2, float eps,
                           float prescale, float scale, int rows, int K, int nb, int flags,
                           const uint8_t *valid) {
    const int gy = 4;
    dim3 grid((rows + gy - 1) / gy, 1, 1);
    dim3 block(32, gy, 1);
    E2H_PROF(2, "rowops", (e2k_rowops<<<grid, block, 0, s>>>(x, src, res, qa, asc, gamma, gamma2, eps, prescale, scale,
                                      rows, K, nb, flags, valid)));
}

extern "C" void e2h_gemv(hipStream_t s, const void *wbuf, const void *acts, void *part,
                         const void *desc, int nsl, int rows, int nmax, int first, int count) {
    const int *dd = (const int *)desc + (size_t)first * 12;
    int nreal = 0;
    for (int i = 0; i < count; i++) {
        const int N = dd[i * 12 + 4];
        if (N > nreal) nreal = N;
    }
    dim3 grid((nreal + 63) / 64, (unsigned)nsl, (unsigned)count);
    E2H_PROF(3, "gemv_split", (e2k_gemv_split<<<grid, 64, 0, s>>>((const signed char *)wbuf, (const signed char *)acts,
                                       (float *)part, dd, nsl, rows, nmax)));
    if (nsl > 1) {
        dim3 grid2((nmax + 127) / 128, (unsigned)count, 1);
        E2H_PROF(4, "gemv_reduce", (e2k_gemv_reduce<<<grid2, 128, 0, s>>>((float *)acts, (const float *)part, dd, nsl, rows, nmax)));
    }
}

extern "C" void e2h_gemm(hipStream_t s, const void *wbuf, const void *acts, const void *desc,
                         int rows, int first, int count, int ytotal) {
    dim3 grid((rows + E2_GEMM_MG * E2_GEMM_TM - 1) / (E2_GEMM_MG * E2_GEMM_TM), (unsigned)ytotal, 1);
    dim3 block(E2_GEMM_NG, E2_GEMM_MG, 1);
    E2H_PROF(5, "gemm", (e2k_gemm<<<grid, block, 0, s>>>((const signed char *)wbuf,
        (const signed char *)acts, (const int *)desc, rows, first, count)));
}

extern "C" void e2h_gemm_lds(hipStream_t s, const void *wbuf, const void *acts, const void *desc,
                            int rows, int first, int count, int ytotal) {
    dim3 grid((rows + E2_LDS_BM - 1) / E2_LDS_BM, (unsigned)ytotal, 1);
    dim3 block(E2_LDS_TX, E2_LDS_TY, 1);
    E2H_PROF(10, "gemm_lds", (e2k_gemm_lds<<<grid, block, 0, s>>>((const signed char *)wbuf,
        (const signed char *)acts, (const int *)desc, rows, first, count)));
}

extern "C" void e2h_norm_rope3(hipStream_t s, float *xq, const float *gq, int rq, float *xk,
                               const float *gk, int rk, float *xv, const float *gv, int rv,
                               const float *costab, const float *sintab, float eps, int hq, int hkv,
                               int D, int seq_len) {
    int rows = rq > rk ? rq : rk;
    if (rv > rows) rows = rv;
    dim3 grid((rows + 1) / 2, 1, 1);
    dim3 block(32, 2, 1);
    E2H_PROF(6, "norm_rope", (e2k_norm_rope3<<<grid, block, 0, s>>>(xq, gq, rq, xk, gk, rk, xv, gv, rv, costab, sintab, eps,
                                          hq, hkv, D, seq_len)));
}

extern "C" void e2h_attn(hipStream_t s, float *ctx, const float *q, const float *k, const float *v,
                         int8_t *qa, float *asc, const uint8_t *valid, int rows, int H, int D,
                         int kv_heads, int q_stride, int kv_stride, int window, int seq_len) {
    dim3 grid((rows + 3) / 4, (unsigned)H, 1);
    dim3 block(32, 4, 1);
    E2H_PROF(7, "attn", (e2k_attn<<<grid, block, 0, s>>>(ctx, q, k, v, qa, asc, valid, rows, H, D, kv_heads, q_stride,
                                    kv_stride, window, seq_len)));
}

extern "C" void e2h_attn_lds(hipStream_t s, float *ctx, const float *q, const float *k,
                             const float *v, int8_t *qa, float *asc, const uint8_t *valid, int rows,
                             int H, int D, int kv_heads, int q_stride, int kv_stride, int window,
                             int seq_len) {
    const int slen = seq_len > 0 ? seq_len : rows;
    const int nseq = seq_len > 0 ? rows / seq_len : 1;
    dim3 grid((slen + E2_ATTN_BK - 1) / E2_ATTN_BK, (unsigned)H, (unsigned)nseq);
    dim3 block(32, 8, 1);
    E2H_PROF(11, "attn_lds", (e2k_attn_lds<<<grid, block, 0, s>>>(
        ctx, q, k, v, qa, asc, valid, rows, H, D, kv_heads, q_stride, kv_stride, window, slen)));
}

extern "C" void e2h_attn_packk(hipStream_t s, int8_t *kpk, float *ksc, const float *k, int S,
                               int sup, int D, int KV, int kv_stride, int nseq) {
    dim3 grid((sup + 3) / 4, 1, (unsigned)(nseq * KV));
    dim3 block(32, 4, 1);
    E2H_PROF(12, "attn_packk", (e2k_attn_packk<<<grid, block, 0, s>>>(kpk, ksc, k, S, sup, D, KV, kv_stride)));
}

extern "C" void e2h_attn_packv(hipStream_t s, int8_t *vpk, float *vsc, const float *v, int S,
                               int sup, int D, int KV, int kv_stride, int nseq) {
    dim3 grid((D + 127) / 128, 1, (unsigned)(nseq * KV));
    dim3 block(32, 4, 1);
    E2H_PROF(13, "attn_vscale", (e2k_attn_vscale<<<grid, block, 0, s>>>(vsc, v, S, D, KV, kv_stride)));
    dim3 grid2((D + 31) / 32, (sup + 31) / 32, (unsigned)(nseq * KV));
    dim3 block2(32, 8, 1);
    E2H_PROF(19, "attn_vquant", (e2k_attn_vquant<<<grid2, block2, 0, s>>>(vpk, vsc, v, S, sup, D, KV, kv_stride)));
}

extern "C" void e2h_attn_scores8(hipStream_t s, float *scr, const int8_t *qpk, const float *qsc,
                                 const int8_t *kpk, const float *ksc, int S, int sup, int D, int H,
                                 int KV, int q_stride, int nz) {
    dim3 grid((S + E2_A8_TM - 1) / E2_A8_TM, (sup + 255) / 256, (unsigned)nz);
    dim3 block(32, 8, 1);
    E2H_PROF(14, "attn_scores8", (e2k_attn_scores8<<<grid, block, 0, s>>>(scr, qpk, qsc, kpk, ksc, S, sup, D, H, KV, q_stride)));
}

extern "C" void e2h_attn_softmax8(hipStream_t s, float *scr, int8_t *pb, float *psc,
                                  const uint8_t *valid, int S, int sup, int H, int window,
                                  int nz) {
    dim3 grid((S + 3) / 4, 1, (unsigned)nz);
    dim3 block(32, 4, 1);
    E2H_PROF(15, "attn_softmax8", (e2k_attn_softmax8<<<grid, block, 0, s>>>(scr, pb, psc, valid, S, sup, H, window)));
}

extern "C" void e2h_attn_ctx8(hipStream_t s, float *ctx, const int8_t *pb, const float *psc,
                              const int8_t *vpk, const float *vsc, int S, int sup, int D, int H,
                              int KV, int q_stride, int nz) {
    dim3 grid((S + E2_A8_TM - 1) / E2_A8_TM, (D + 255) / 256, (unsigned)nz);
    dim3 block(32, 8, 1);
    E2H_PROF(16, "attn_ctx8", (e2k_attn_ctx8<<<grid, block, 0, s>>>(ctx, pb, psc, vpk, vsc, S, sup, D, H, KV, q_stride)));
}

extern "C" void e2h_attn_scores8_lds(hipStream_t s, float *scr, const int8_t *qpk,
                                     const float *qsc, const int8_t *kpk, const float *ksc, int S,
                                     int sup, int D, int H, int KV, int q_stride, int nz) {
    dim3 grid((S + E2_A8L_BM - 1) / E2_A8L_BM, (sup + E2_A8L_BN - 1) / E2_A8L_BN, (unsigned)nz);
    dim3 block(32, 8, 1);
    E2H_PROF(17, "attn_scores8l", (e2k_attn_scores8_lds<<<grid, block, 0, s>>>(
        scr, qpk, qsc, kpk, ksc, S, sup, D, H, KV, q_stride)));
}

extern "C" void e2h_attn_ctx8_lds(hipStream_t s, float *ctx, const int8_t *pb, const float *psc,
                                 const int8_t *vpk, const float *vsc, int S, int sup, int D, int H,
                                 int KV, int q_stride, int nz) {
    dim3 grid((S + E2_A8L_BM - 1) / E2_A8L_BM, (D + E2_A8L_BN - 1) / E2_A8L_BN, (unsigned)nz);
    dim3 block(32, 8, 1);
    E2H_PROF(18, "attn_ctx8l", (e2k_attn_ctx8_lds<<<grid, block, 0, s>>>(
        ctx, pb, psc, vpk, vsc, S, sup, D, H, KV, q_stride)));
}

extern "C" void e2h_gelu_mul_quant(hipStream_t s, float *gate, const float *up, const float *ple,
                                   int8_t *qa, float *asc, int rows, int K, int nb, int stride,
                                   int ple_off) {
    dim3 grid((rows + 3) / 4, 1, 1);
    dim3 block(32, 4, 1);
    E2H_PROF(8, "gelu", (e2k_gelu_mul_quant<<<grid, block, 0, s>>>(gate, up, ple, qa, asc, rows, K, nb, stride, ple_off)));
}

extern "C" void e2h_pool_l2(hipStream_t s, float *out, const float *emb, const uint8_t *valid,
                            int rows, int dim, int seq_len) {
    dim3 grid((unsigned)(seq_len > 0 ? rows / seq_len : 1), 1, 1);
    E2H_PROF(9, "pool_l2", (e2k_pool_l2<<<grid, 128, 0, s>>>(out, emb, valid, rows, dim, seq_len)));
}

/* ---- vision tower launchers */
extern "C" void e2h_vpatch_q(hipStream_t s, int8_t *qa, float *asc, const float *pix,
                             const int32_t *order, int P, int D) {
    const int gy = 4;
    dim3 grid((P + gy - 1) / gy, 1, 1);
    dim3 block(32, gy, 1);
    E2H_PROF(20, "vpatch_q", (e2k_vpatch_q<<<grid, block, 0, s>>>((signed char *)qa, asc, pix, order, P, D)));
}

extern "C" void e2h_vpos_add(hipStream_t s, float *x, const float *proj, const uint16_t *postab,
                             const int32_t *order, const int32_t *posids, int P, int D, int npos) {
    dim3 grid((P + 3) / 4, (D / 4 + 63) / 64, 1);
    dim3 block(64, 4, 1);
    E2H_PROF(21, "vpos_add", (e2k_vpos_add<<<grid, block, 0, s>>>(x, proj, postab, order, posids,
                                                                P, D, npos)));
}

extern "C" void e2h_vnorm_rope3(hipStream_t s, float *xq, const float *gq, float *xk,
                                const float *gk, float *xv, const float *gv, const float *costab,
                                const float *sintab, float eps, int rows, int heads) {
    dim3 grid((rows + 1) / 2, 1, 1);
    dim3 block(32, 2, 1);
    E2H_PROF(22, "vnorm_rope", (e2k_vnorm_rope3<<<grid, block, 0, s>>>(xq, gq, xk, gk, xv, gv,
                                                                     costab, sintab, eps, rows, heads)));
}

extern "C" void e2h_vattn_packq(hipStream_t s, int8_t *qpk, float *qsc, const float *q, int rows,
                               int Ppad, int H) {
    dim3 grid((rows + 3) / 4, 1, 1);
    dim3 block(32, 4, 1);
    E2H_PROF(23, "vattn_packq", (e2k_vattn_packq<<<grid, block, 0, s>>>((signed char *)qpk, qsc, q, rows, Ppad, H)));
}

extern "C" void e2h_vattn_packk(hipStream_t s, int8_t *kpk, float *ksc, const float *k, int P,
                               int Ppad, int H) {
    const int rows = H * Ppad;
    dim3 grid((rows + 3) / 4, 1, 1);
    dim3 block(32, 4, 1);
    E2H_PROF(24, "vattn_packk", (e2k_vattn_packk<<<grid, block, 0, s>>>((signed char *)kpk, ksc, k, P, Ppad, H)));
}

extern "C" void e2h_vattn_packv(hipStream_t s, int8_t *vpk, float *vsc, const float *v, int P,
                               int Ppad, int H) {
    dim3 grid(H, 2, 1);
    dim3 block(32, 1, 1);
    E2H_PROF(25, "vattn_vscale", (e2k_vattn_vscale<<<grid, block, 0, s>>>(vsc, v, P, H)));
    dim3 grid2(2, (Ppad + 31) / 32, (unsigned)H);
    dim3 block2(32, 8, 1);
    E2H_PROF(26, "vattn_vquant", (e2k_vattn_vquant<<<grid2, block2, 0, s>>>((signed char *)vpk, vsc, v, P, Ppad, H)));
}

extern "C" void e2h_vattn_scores8(hipStream_t s, float *scr, const int8_t *qpk, const float *qsc,
                                 const int8_t *kpk, const float *ksc, int P, int Ppad, int H,
                                 int h) {
    dim3 grid((P + 7) / 8, (Ppad + 255) / 256, 1);
    dim3 block(32, 8, 1);
    E2H_PROF(27, "vattn_scores", (e2k_vattn_scores8<<<grid, block, 0, s>>>(scr, qpk, qsc, kpk, ksc, P, Ppad, h)));
}

extern "C" void e2h_vattn_softmax8(hipStream_t s, const float *scr, int8_t *pb, float *psc, int P,
                                  int Ppad) {
    dim3 grid((P + 3) / 4, 1, 1);
    dim3 block(32, 4, 1);
    E2H_PROF(28, "vattn_softmax", (e2k_vattn_softmax8<<<grid, block, 0, s>>>(scr, (signed char *)pb, psc, P, Ppad)));
}

extern "C" void e2h_vattn_ctx8(hipStream_t s, float *ctx, const int8_t *pb, const float *psc,
                              const int8_t *vpk, const float *vsc, int P, int Ppad, int H, int h) {
    dim3 grid((P + 7) / 8, 1, 1);
    dim3 block(32, 2, 1);
    E2H_PROF(29, "vattn_ctx", (e2k_vattn_ctx8<<<grid, block, 0, s>>>(ctx, pb, psc, vpk, vsc, P, Ppad, h)));
}

extern "C" void e2h_vpool(hipStream_t s, float *pool, const float *x, const int32_t *gstart,
                         const int32_t *gpatch, const int32_t *ngp, int D, float scale) {
    E2H_PROF(30, "vpool", (e2k_vpool<<<E2_VGROUPS, 128, 0, s>>>(pool, x, gstart, gpatch, ngp, D, scale)));
}
