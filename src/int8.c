/* int8.c — weight quantization at load time + the scalar int8 GEMM fallback. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "int8.h"
#include "kernels.h"

/* Source access: `ld` is the source's leading dimension for the strided variant, so a projection
 * output [S][kv_dim] can be quantized per head without copying it into a contiguous [S][D] block. */
struct src {
    const void *p;
    size_t ld;
};
typedef float (*e2_getfn)(const struct src *, size_t n, size_t k);

/* Packs straight into the quad-interleaved layout (see int8.h): the four quantized values of one
 * k-quad of one row are one 4-byte store, and the 8 rows of a block are 4 bytes apart. */
static int quantize(e2_q8w *q, int N, int K, e2_getfn get, const struct src *ctx) {
    memset(q, 0, sizeof *q);
    q->N = N;
    q->K = K;
    const int nblocks = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    q->nblocks = nblocks;
    const int kq = (K + 3) / 4; /* k-quads per row; one 8-row block is kq*32 bytes */
    const int Npad = (N + 7) & ~7; /* the kernel loads 8 weight rows at once, tail included */
    q->out_min = -1e31f;
    q->out_max = 1e31f;
    q->Npad = Npad;
    q->w = calloc((size_t)(Npad / 8) * kq * 32, 1);
    q->scale = calloc((size_t)Npad, sizeof(float));
    q->bsq = calloc((size_t)nblocks * Npad, sizeof(int32_t));
    if (!q->w || !q->scale || !q->bsq) {
        /* leave nothing half-built: a retry (the lazy audio path) must not adopt these arrays */
        free(q->w);
        free(q->scale);
        free(q->bsq);
        memset(q, 0, sizeof *q);
        return -1;
    }
    for (int n = 0; n < N; n++) {
        float amax = 0.f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(get(ctx, (size_t)n, (size_t)k));
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / scale;
        for (int b = 0; b < nblocks; b++) {
            int32_t sum = 0;
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
            for (int k = k0; k < k1; k++) {
                int32_t r = (int32_t)lrintf(get(ctx, (size_t)n, (size_t)k) * inv);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                q->w[(size_t)(n >> 3) * kq * 32 + (size_t)(k >> 2) * 32 + (n & 7) * 4 +
                     (k & 3)] = (int8_t)r;
                sum += r;
            }
            q->bsq[(size_t)b * Npad + n] = sum;
        }
        q->scale[n] = scale;
    }
    return 0;
}

static float get_f32(const struct src *s, size_t n, size_t k) {
    return ((const float *)s->p)[n * s->ld + k];
}
static float get_bf16(const struct src *s, size_t n, size_t k) {
    return e2_bf16f(((const uint16_t *)s->p)[n * s->ld + k]);
}

int e2_q8w_from_f32(e2_q8w *q, const float *w, int N, int K) {
    const struct src s = { w, (size_t)K };
    return quantize(q, N, K, get_f32, &s);
}

int e2_q8w_from_bf16(e2_q8w *q, const uint16_t *w, int N, int K) {
    const struct src s = { w, (size_t)K };
    return quantize(q, N, K, get_bf16, &s);
}

int e2_q8w_from_f32_ld(e2_q8w *q, const float *w, int N, int K, int ld) {
    const struct src s = { w, (size_t)ld };
    return quantize(q, N, K, get_f32, &s);
}

void e2_q8w_free(e2_q8w *q) {
    if (!q->borrowed) {
        free(q->w);
        free(q->scale);
        free(q->bsq);
    }
    memset(q, 0, sizeof *q);
}

/* ---------------------------------------------------------------- scalar kernels
 * Activations arrive as int8 (symmetric, per E2_I8_BLOCK scale); the reference form of the dot is
 * sum_k x*w = sum_blocks a_scale[b]*w_scale * sum_k a_i8*b_i8, so the scalar kernel is a plain
 * per-block int32 dot scaled by the product of the block scales.
 *
 * The weight layout stores 8 output rows in one 32-byte quad (8 n x 4 k) - what the AVX2 and NEON
 * kernels consume with a single load - so a single row's weights are strided by 4 bytes and a
 * straightforward portable loop reads them one byte at a time: 7.8 GMAC/s. This walks one
 * (8 columns x 128 k) block into a k-major scratch tile first (a 1 KB transpose, once per block,
 * reused by every token) and then runs eight contiguous 128-byte dots, which the compiler can
 * vectorise without -march: 24 GMAC/s, 12x the per-row loop it replaces and 9x the bf16 GEMM on the
 * same host. The k loop stays exact int32, so the result is bit-identical to the per-row form and
 * this remains the reference the SIMD kernels are checked against.
 */
#define I8_COLS 8
static void gemm_cols8(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w, int M,
                       int N, int g, int j0, int lanes) {
    const int K = w->K, nb = w->nblocks;
    const int kq = (K + 3) / 4;      /* k-quads per row; one 8-row block is kq * 32 bytes */
    const int nq = E2_I8_BLOCK / 4;  /* quads in a full block */
    const int8_t *wbase = w->w + (size_t)(g >> 3) * kq * 32;
    const float *wsc = w->scale + g;
    int8_t tile[I8_COLS * E2_I8_BLOCK]; /* one block of 8 columns, k-major, 1 KB */
    const int jn = j0 + lanes;

    for (int m = 0; m < M; m++) {
        float *yr = y + (size_t)m * N + g;
        for (int j = j0; j < jn; j++) yr[j] = 0.f;
    }
    for (int b = 0; b < nb; b++) {
        int nk = K - b * E2_I8_BLOCK; /* a short last block is only possible for a K that is not a
                                       * block multiple, which the SIMD kernels forbid */
        if (nk > E2_I8_BLOCK) nk = E2_I8_BLOCK;
        const int8_t *wb = wbase + (size_t)(b * nq) * 32;
        for (int q = 0; q < (nk + 3) / 4; q++)
            for (int j = 0; j < I8_COLS; j++)
                memcpy(tile + (size_t)j * E2_I8_BLOCK + (size_t)q * 4, wb + (size_t)q * 32 + 4 * j, 4);
        for (int m = 0; m < M; m++) {
            const int8_t *ab = qa + (size_t)m * K + (size_t)b * E2_I8_BLOCK;
            const float as = a_scales[(size_t)m * nb + b];
            float *yr = y + (size_t)m * N + g;
            for (int j = j0; j < jn; j++) {
                const int8_t *wr = tile + (size_t)j * E2_I8_BLOCK;
                int32_t d = 0;
                /* offset-binary activations: the stored byte is the value + 128 */
                for (int k = 0; k < nk; k++) d += ((uint8_t)ab[k] - 128) * wr[k];
                yr[j] += (float)d * (as * wsc[j]);
            }
        }
    }
    if (w->out_min > -1e30f)
        for (int m = 0; m < M; m++)
            for (int j = j0; j < jn; j++) {
                float *v = y + (size_t)m * N + g + j;
                if (*v < w->out_min) *v = w->out_min;
                else if (*v > w->out_max) *v = w->out_max;
            }
}

void e2_i8_gemm_scalar(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                       int M, int N, int n0, int n1) {
    /* Column ranges are arbitrary, but a group starts at the 8-row pack that holds it: n0 is
     * 8-aligned in every hot caller (the attention's `lo &= ~7`, the towers' 16-column claims), and
     * a partial group just computes 8 lanes and stores the ones asked for. */
    for (int n = n0; n < n1;) {
        const int g = n & ~(I8_COLS - 1);
        const int j0 = n - g;
        int lanes = I8_COLS - j0;
        if (lanes > n1 - n) lanes = n1 - n;
        gemm_cols8(y, qa, a_scales, w, M, N, g, j0, lanes);
        n += lanes;
    }
}

/* ---------------------------------------------------------------- activation quantizer
 * The portable reference for e2_quantize_blocks_avx2/_vnni/_neon and the one the scalar backend
 * installs now that it has an int8 path: amax over the (clamped) block, scale = amax/127, round to
 * nearest, saturate to +-127, store biased by +128 (see int8.h). The two loops are a max-reduction
 * and a map, so both vectorise without -ffast-math. */
void e2_quantize_blocks_scalar(int8_t *out, float *scales, const float *x, int M, int K, float b_lo,
                               float b_hi) {
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const int clamp = b_lo > -INFINITY || b_hi < INFINITY;
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        int8_t *orow = out + (size_t)m * K;
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            int k1 = k0 + E2_I8_BLOCK;
            if (k1 > K) k1 = K;
            float amax = 0.f;
            for (int k = k0; k < k1; k++) {
                float v = xr[k];
                if (clamp) v = v < b_lo ? b_lo : (v > b_hi ? b_hi : v);
                const float a = fabsf(v);
                if (a > amax) amax = a;
            }
            const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
            const float inv = 1.0f / sc;
            for (int k = k0; k < k1; k++) {
                float v = xr[k];
                if (clamp) v = v < b_lo ? b_lo : (v > b_hi ? b_hi : v);
                int32_t r = (int32_t)lrintf(v * inv);
                if (r > 127) r = 127;
                else if (r < -127) r = -127;
                orow[k] = (int8_t)(r + 128);
            }
            scales[(size_t)m * nb + b] = sc;
        }
    }
}
