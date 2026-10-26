/* int8.c — weight quantization at load time + the scalar int8 GEMM fallback. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "int8.h"
#include "kernels.h"

typedef float (*e2_getfn)(const void *, size_t);

/* Packs straight into the quad-interleaved layout (see int8.h): the four quantized values of one
 * k-quad of one row are one 4-byte store, and the 8 rows of a block are 4 bytes apart. */
static int quantize(e2_q8w *q, int N, int K, e2_getfn get, const void *ctx) {
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
            float a = fabsf(get(ctx, (size_t)n * K + k));
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / scale;
        for (int b = 0; b < nblocks; b++) {
            int32_t sum = 0;
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
            for (int k = k0; k < k1; k++) {
                int32_t r = (int32_t)lrintf(get(ctx, (size_t)n * K + k) * inv);
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

static float get_f32(const void *ctx, size_t i) { return ((const float *)ctx)[i]; }
static float get_bf16(const void *ctx, size_t i) { return e2_bf16f(((const uint16_t *)ctx)[i]); }

int e2_q8w_from_f32(e2_q8w *q, const float *w, int N, int K) {
    return quantize(q, N, K, get_f32, (const void *)w);
}

int e2_q8w_from_bf16(e2_q8w *q, const uint16_t *w, int N, int K) {
    return quantize(q, N, K, get_bf16, (const void *)w);
}

void e2_q8w_free(e2_q8w *q) {
    if (!q->borrowed) {
        free(q->w);
        free(q->scale);
        free(q->bsq);
    }
    memset(q, 0, sizeof *q);
}

/* ---------------------------------------------------------------- scalar kernel
 * Activations arrive as int8 (symmetric, per E2_I8_BLOCK scale); the reference form of the dot is
 * sum_k x*w = sum_blocks a_scale[b]*w_scale * sum_k a_i8*b_i8, so the scalar kernel is a plain
 * per-block int32 dot scaled by the product of the block scales. */
void e2_i8_gemm_scalar(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                       int M, int N, int n0, int n1) {
    const int K = w->K, nb = w->nblocks, kq = (K + 3) / 4;
    for (int n = n0; n < n1; n++) {
        const float ws = w->scale[n];
        for (int m = 0; m < M; m++) {
            const int8_t *arow = qa + (size_t)m * K;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                int32_t d = 0;
                const int k0 = b * E2_I8_BLOCK;
                const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
                for (int k = k0; k < k1; k++)
                    d += ((int32_t)(uint8_t)arow[k] - 128) *
                         (int32_t)w->w[(size_t)(n >> 3) * kq * 32 + (size_t)(k >> 2) * 32 +
                                       (n & 7) * 4 + (k & 3)];
                acc += (float)d * (a_scales[(size_t)m * nb + b] * ws);
            }
            if (w->out_min > -1e30f) {
                if (acc < w->out_min) acc = w->out_min;
                else if (acc > w->out_max) acc = w->out_max;
            }
            y[(size_t)m * N + n] = acc;
        }
    }
}
