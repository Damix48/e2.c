/* kernels_i8_neon.c — int8 GEMM and per-block activation quantizer for aarch64 (ARMv8.2 dot
 * product). Same quantization scheme and the same quad-interleaved weight layout as the VNNI and
 * AVX2 kernels (see int8.h); only the instruction changes.
 *
 * vdotq_s32 is the ARMv8.2 dot product: four 4-byte int8 products accumulated into four int32 lanes.
 * Both operands are signed there (the mixed u8 x i8 form is vusdotq_s32, which needs the i8mm
 * extension), so the biased activations int8.h stores are un-biased on the fly with an xor 0x80 -
 * the exact inverse of the flip the AVX2 kernel applies. That makes the accumulator the true
 * sum_k a_i8 * b_i8, so unlike vpdpbusd there is no 128 * sum_k b correction to apply and bsq is
 * never read. No saturation trick either: vdotq_s32 does not have vpmaddubsw's behaviour.
 *
 * The layout hands the kernel 8 output rows as one 32-byte vector, which is two 128-bit halves:
 * lanes 0..3 of the first are rows n..n+3, lanes 0..3 of the second are rows n+4..n+7. So one
 * k-quad is one activation broadcast plus two vdotq_s32, exactly as the AVX2 kernel does it with
 * two 256-bit maddubs.
 *
 * MR is 4 here, not the 8 the x86 kernels use: each token keeps two int32 accumulators (one per
 * 4-row half) and two float accumulators, i.e. four of the 32 NEON registers, and 8 tokens would
 * want all 32 of them before the weight and scale vectors get one. Weights are re-read once per MR
 * tokens, so this costs a document-sized GEMM nothing (the k-quad stream is L2-resident) and only
 * matters for single-token traffic.
 */
#include <arm_neon.h>
#include <math.h>

#include "int8.h"
#include "kernels.h"

#define MR 4
#define NQ4 (E2_I8_BLOCK / 4)
/* the four-quad inner loop and its 16-byte activation load assume this */

/* The four activation bytes of one k-quad, un-biased and broadcast into all four 32-bit lanes. */
static inline int8x16_t act4(const uint8_t *p) {
    return vreinterpretq_s8_u32(vdupq_n_u32((uint32_t)e2_load_i32(p) ^ 0x80808080u));
}

/* One 8-output-row block of the weight matrix, for output columns [n, n+8). `rows` is 8 except for
 * the last partial block, where only the first `rows` lanes are written back. */
static void gemm_cols8(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w, int M,
                       int N, int n, int rows) {
    const int K = w->K, nb = w->nblocks;
    const int kq4 = (K + 3) / 4;
    const int8_t *wp = w->w + (size_t)(n >> 3) * kq4 * 32 + (size_t)(n & 7) * 4;
    const float *wsc = w->scale + n;
    const float32x4_t vlo = vdupq_n_f32(w->out_min), vhi = vdupq_n_f32(w->out_max);
    int m = 0;
    for (; m + MR <= M; m += MR) {
        int32x4_t acl[MR], ach[MR];
        float32x4_t fl[MR], fh[MR];
        for (int i = 0; i < MR; i++) {
            acl[i] = ach[i] = vdupq_n_s32(0);
            fl[i] = fh[i] = vdupq_n_f32(0.f);
        }
        for (int b = 0; b < nb; b++) {
            const float *sp = a_scales + (size_t)m * nb + b;
            for (int q = 0; q < NQ4; q++) {
                const int k = b * E2_I8_BLOCK + q * 4;
                /* wp is offset by (n&7)*4, so these are never 16-byte aligned: vld1q_s8 is the
                 * form that is allowed to be unaligned. */
                const int8_t *wq = wp + (size_t)(b * NQ4 + q) * 32;
                const int8x16_t wl = vld1q_s8(wq), wh = vld1q_s8(wq + 16);
                for (int i = 0; i < MR; i++) {
                    const int8x16_t av =
                        act4((const uint8_t *)qa + (size_t)(m + i) * K + k);
                    acl[i] = vdotq_s32(acl[i], av, wl);
                    ach[i] = vdotq_s32(ach[i], av, wh);
                }
            }
            const float32x4_t sl = vld1q_f32(wsc), sh = vld1q_f32(wsc + 4);
            for (int i = 0; i < MR; i++) {
                const float32x4_t s = vdupq_n_f32(sp[i * nb]);
                fl[i] = vfmaq_f32(fl[i], vcvtq_f32_s32(acl[i]), vmulq_f32(s, sl));
                fh[i] = vfmaq_f32(fh[i], vcvtq_f32_s32(ach[i]), vmulq_f32(s, sh));
                acl[i] = ach[i] = vdupq_n_s32(0);
            }
        }
        if (rows == 8) {
            for (int i = 0; i < MR; i++) {
                float *dst = y + (size_t)(m + i) * N + n;
                vst1q_f32(dst, vminq_f32(vmaxq_f32(fl[i], vlo), vhi));
                vst1q_f32(dst + 4, vminq_f32(vmaxq_f32(fh[i], vlo), vhi));
            }
        } else {
            float tmp[8];
            for (int i = 0; i < MR; i++) {
                vst1q_f32(tmp, vminq_f32(vmaxq_f32(fl[i], vlo), vhi));
                vst1q_f32(tmp + 4, vminq_f32(vmaxq_f32(fh[i], vlo), vhi));
                float *dst = y + (size_t)(m + i) * N + n;
                for (int j = 0; j < rows; j++) dst[j] = tmp[j];
            }
        }
    }
    for (; m < M; m++) {
        int32x4_t acl = vdupq_n_s32(0), ach = vdupq_n_s32(0);
        float32x4_t fl = vdupq_n_f32(0.f), fh = vdupq_n_f32(0.f);
        for (int b = 0; b < nb; b++) {
            for (int q = 0; q < NQ4; q++) {
                const int k = b * E2_I8_BLOCK + q * 4;
                const int8_t *wq = wp + (size_t)(b * NQ4 + q) * 32;
                const int8x16_t av = act4((const uint8_t *)qa + (size_t)m * K + k);
                acl = vdotq_s32(acl, av, vld1q_s8(wq));
                ach = vdotq_s32(ach, av, vld1q_s8(wq + 16));
            }
            const float32x4_t s = vdupq_n_f32(a_scales[(size_t)m * nb + b]);
            fl = vfmaq_f32(fl, vcvtq_f32_s32(acl), vmulq_f32(s, vld1q_f32(wsc)));
            fh = vfmaq_f32(fh, vcvtq_f32_s32(ach), vmulq_f32(s, vld1q_f32(wsc + 4)));
            acl = ach = vdupq_n_s32(0);
        }
        if (rows == 8) {
            float *dst = y + (size_t)m * N + n;
            vst1q_f32(dst, vminq_f32(vmaxq_f32(fl, vlo), vhi));
            vst1q_f32(dst + 4, vminq_f32(vmaxq_f32(fh, vlo), vhi));
        } else {
            float tmp[8];
            vst1q_f32(tmp, vminq_f32(vmaxq_f32(fl, vlo), vhi));
            vst1q_f32(tmp + 4, vminq_f32(vmaxq_f32(fh, vlo), vhi));
            float *dst = y + (size_t)m * N + n;
            for (int j = 0; j < rows; j++) dst[j] = tmp[j];
        }
    }
}

void e2_i8_gemm_neon(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                     int M, int N, int n0, int n1) {
    int n = n0;
    for (; n + 8 <= n1; n += 8) gemm_cols8(y, qa, a_scales, w, M, N, n, 8);
    if (n < n1) gemm_cols8(y, qa, a_scales, w, M, N, n, n1 - n);
}

/* ---------------------------------------------------------------- packed-operand attention GEMM
 * The NEON form of the kernel in kernels_i8_avx512_vnni.c / kernels_i8_vnni.c; same contract
 * (int8.h). It is what turns the vision tower's attention int8 on aarch64, where it was 8.5 s of a
 * 13.9 s image because this symbol was NULL and vattn_int8_ok() therefore said no.
 *
 * The activation is the biased (a+128) form int8.h stores, so it is un-biased on the fly with the
 * same xor 0x80 as the weight GEMM above - which means `bz` is never read: the accumulator already
 * is the true sum_k a_i8*b_i8, so the 128*sum(b) correction this kernel exists to apply elsewhere
 * disappears. vdotq_s32 is also why there is no |a|/sign(a) trick and no saturation to dodge.
 *
 * The B operand is packed k-quad-major, so 8 output columns are 32 contiguous bytes = two 128-bit
 * halves (columns n0..n0+3 and n0+4..n0+7). Tile is 4 query rows x 8 columns: 8 int32 and 8 float
 * accumulators plus the two weight vectors, 19 of the 32 registers. The row loop is split into a
 * fixed 4-row body and a 1-row tail, like the weight GEMM, so `i < 4` stays a constant bound and the
 * accumulators stay in registers.
 */
#define AT_BLOCK (E2_I8_BLOCK / 4) /* k-quads per activation scale block */

/* One k-quad of one token: broadcast its 4 un-biased activation bytes into all lanes. */
static inline int8x16_t act4i(const uint8_t *p) {
    return vreinterpretq_s8_u32(vdupq_n_u32((uint32_t)e2_load_i32(p) ^ 0x80808080u));
}

/* Columns [n0, n0+nn) of C, all M rows. `bz` is unused: see the xor note above. */
static void attn_cols8(float *C, int ldc, const uint8_t *A, int lda, const float *asc,
                       const uint8_t *Bp, const float *bsc, int M, int Np, int K, int n0, int nn) {
    const int nblocks = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const int qend = (K + 3) / 4;
    const int qfull = K >> 2; /* quads whose 4 k all exist in the row */
    const float32x4_t scl = vld1q_f32(bsc + n0), sch = vld1q_f32(bsc + n0 + 4);
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        int32x4_t al[4], ah[4];
        float32x4_t fl[4], fh[4];
        for (int i = 0; i < 4; i++) {
            al[i] = ah[i] = vdupq_n_s32(0);
            fl[i] = fh[i] = vdupq_n_f32(0.f);
        }
        for (int b = 0; b < nblocks; b++) {
            const int q0 = b * AT_BLOCK;
            int q1 = q0 + AT_BLOCK;
            if (q1 > qend) q1 = qend;
            int q1f = q1 < qfull ? q1 : qfull;
            for (int q = q0; q < q1f; q++) {
                const int8_t *bp = (const int8_t *)(Bp + ((size_t)q * Np + n0) * 4);
                const int8x16_t w0 = vld1q_s8(bp), w1 = vld1q_s8(bp + 16);
                for (int i = 0; i < 4; i++) {
                    const int8x16_t av = act4i(A + (size_t)(m + i) * lda + (size_t)q * 4);
                    al[i] = vdotq_s32(al[i], av, w0);
                    ah[i] = vdotq_s32(ah[i], av, w1);
                }
            }
            if (q1f < q1) { /* one quad of 1..3 k left at the end of the row: too rare to vectorize */
                const int k0 = q1f * 4, kn = K - k0;
                for (int i = 0; i < 4; i++) {
                    const uint8_t *ap = A + (size_t)(m + i) * lda + k0;
                    int32_t d[8];
                    for (int j = 0; j < 8; j++) {
                        const int8_t *wp = (const int8_t *)(Bp + ((size_t)q1f * Np + n0 + j) * 4);
                        int32_t sum = 0;
                        for (int t = 0; t < kn; t++) sum += ((int32_t)ap[t] - 128) * (int32_t)wp[t];
                        d[j] = sum;
                    }
                    al[i] = vaddq_s32(al[i], vld1q_s32(d));
                    ah[i] = vaddq_s32(ah[i], vld1q_s32(d + 4));
                }
            }
            for (int i = 0; i < 4; i++) {
                const float32x4_t s = vdupq_n_f32(asc[(size_t)(m + i) * nblocks + b]);
                fl[i] = vfmaq_f32(fl[i], vcvtq_f32_s32(al[i]), vmulq_f32(s, scl));
                fh[i] = vfmaq_f32(fh[i], vcvtq_f32_s32(ah[i]), vmulq_f32(s, sch));
                al[i] = ah[i] = vdupq_n_s32(0);
            }
        }
        if (nn == 8) {
            for (int i = 0; i < 4; i++) {
                float *dst = C + (size_t)(m + i) * ldc + n0;
                vst1q_f32(dst, fl[i]);
                vst1q_f32(dst + 4, fh[i]);
            }
        } else {
            for (int i = 0; i < 4; i++) {
                float tmp[8];
                vst1q_f32(tmp, fl[i]);
                vst1q_f32(tmp + 4, fh[i]);
                float *dst = C + (size_t)(m + i) * ldc + n0;
                for (int j = 0; j < nn; j++) dst[j] = tmp[j];
            }
        }
    }
    for (; m < M; m++) { /* row tail */
        int32x4_t al = vdupq_n_s32(0), ah = vdupq_n_s32(0);
        float32x4_t fl = vdupq_n_f32(0.f), fh = vdupq_n_f32(0.f);
        for (int b = 0; b < nblocks; b++) {
            const int q0 = b * AT_BLOCK;
            int q1 = q0 + AT_BLOCK;
            if (q1 > qend) q1 = qend;
            const int q1f = q1 < qfull ? q1 : qfull;
            for (int q = q0; q < q1f; q++) {
                const int8_t *bp = (const int8_t *)(Bp + ((size_t)q * Np + n0) * 4);
                const int8x16_t av = act4i(A + (size_t)m * lda + (size_t)q * 4);
                al = vdotq_s32(al, av, vld1q_s8(bp));
                ah = vdotq_s32(ah, av, vld1q_s8(bp + 16));
            }
            if (q1f < q1) {
                const int k0 = q1f * 4, kn = K - k0;
                const uint8_t *ap = A + (size_t)m * lda + k0;
                int32_t d[8];
                for (int j = 0; j < 8; j++) {
                    const int8_t *wp = (const int8_t *)(Bp + ((size_t)q1f * Np + n0 + j) * 4);
                    int32_t sum = 0;
                    for (int t = 0; t < kn; t++) sum += ((int32_t)ap[t] - 128) * (int32_t)wp[t];
                    d[j] = sum;
                }
                al = vaddq_s32(al, vld1q_s32(d));
                ah = vaddq_s32(ah, vld1q_s32(d + 4));
            }
            const float32x4_t s = vdupq_n_f32(asc[(size_t)m * nblocks + b]);
            fl = vfmaq_f32(fl, vcvtq_f32_s32(al), vmulq_f32(s, scl));
            fh = vfmaq_f32(fh, vcvtq_f32_s32(ah), vmulq_f32(s, sch));
            al = ah = vdupq_n_s32(0);
        }
        float tmp[8];
        vst1q_f32(tmp, fl);
        vst1q_f32(tmp + 4, fh);
        float *dst = C + (size_t)m * ldc + n0;
        for (int j = 0; j < nn; j++) dst[j] = tmp[j];
    }
}

void e2_i8_attn_gemm_neon(float *C, int ldc, const uint8_t *A, int lda, const float *asc,
                          const uint8_t *Bp, const float *bsc, const int32_t *bz, int M, int Np,
                          int N, int K) {
    (void)bz;
    for (int n0 = 0; n0 < N; n0 += 8) {
        const int nn = N - n0 < 8 ? N - n0 : 8;
        attn_cols8(C, ldc, A, lda, asc, Bp, bsc, M, Np, K, n0, nn);
    }
}

/* ---------------------------------------------------------------- fused softmax + probability quantizer
 * Same contract as the AVX2/AVX-512 versions (int8.h): x[0..n) is softmaxed in place, the unbiased
 * (q+128) probabilities land in q8 with one scale per E2_I8_BLOCK, rows in [rows, ...) untouched.
 *
 * Fusing is the point: the rows are written once and read once. Done separately - what the callers
 * fall back to when this is NULL - the score row is written by the logits GEMM, read and rewritten by
 * softmax, then read again by the quantizer, and for a 2304-patch image that matrix is ~1 G values.
 *
 * The exponential is the same range-reduction + degree-6 polynomial the AVX2 kernel uses, not libm's
 * expf: the scalar softmax calls `_ZGVnN4v_expf`, which is accurate and expensive, and softmax only
 * needs the values to be right to ~1e-7 relative before they are quantized to 7 bits anyway.
 */
static inline float32x4_t exp_ps_neon(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2hi = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t ln2lo = vdupq_n_f32(-1.904654299957768e-9f);
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    const float32x4_t t = vmulq_f32(x, log2e);
    const float32x4_t n = vrndnq_f32(t); /* round to nearest, like _MM_FROUND_TO_NEAREST_INT */
    float32x4_t r = vfmsq_f32(x, n, ln2hi); /* x - n*ln2hi */
    r = vfmsq_f32(r, n, ln2lo);

    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vfmaq_f32(r, p, vmulq_f32(r, r));
    p = vaddq_f32(p, vdupq_n_f32(1.0f));
    /* scale by 2^n by adding n to the exponent field */
    const int32x4_t ni = vcvtq_s32_f32(n);
    const float32x4_t scale =
        vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
    return vmulq_f32(p, scale);
}

void e2_softmax_quant_i8_neon(float *x, int8_t *q8, float *scales, int rows, int n, int nb) {
    const int32x4_t bias = vdupq_n_s32(0x80808080); /* store q+128: see int8.h */
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * n;
        int8_t *q = q8 + (size_t)r * n;
        float *sc = scales + (size_t)r * nb;
        int i = 0;
        float32x4_t vmax = vdupq_n_f32(-INFINITY);
        for (; i + 4 <= n; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(row + i));
        float m = vmaxvq_f32(vmax);
        for (; i < n; i++)
            if (row[i] > m) m = row[i];

        const float32x4_t vm = vdupq_n_f32(m);
        float32x4_t vsum = vdupq_n_f32(0.f);
        float tail = 0.f;
        /* n is a key/patch count here, not a sequence length: up to E2_MAX_PATCHES/128 = 20 */
        float bmax[32];
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < n ? k0 + E2_I8_BLOCK : n;
            float32x4_t bm = vdupq_n_f32(0.f);
            i = k0;
            for (; i + 4 <= k1; i += 4) {
                const float32x4_t e = exp_ps_neon(vsubq_f32(vld1q_f32(row + i), vm));
                vst1q_f32(row + i, e);
                bm = vmaxq_f32(bm, e);
                vsum = vaddq_f32(vsum, e);
            }
            float bf = vmaxvq_f32(bm);
            for (; i < k1; i++) {
                const float e = expf(row[i] - m);
                row[i] = e;
                if (e > bf) bf = e;
                tail += e;
            }
            bmax[b] = bf;
        }
        const float inv = 1.0f / (vaddvq_f32(vsum) + tail);

        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < n ? k0 + E2_I8_BLOCK : n;
            const float bf = bmax[b];
            const float32x4_t vr = vdupq_n_f32(bf > 0.f ? 127.0f / bf : 0.f);
            sc[b] = bf > 0.f ? bf * inv / 127.0f : 0.f;
            i = k0;
            for (; i + 16 <= k1; i += 16) {
                const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + i), vr));
                const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + i + 4), vr));
                const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + i + 8), vr));
                const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + i + 12), vr));
                const int8x8_t lo = vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1)));
                const int8x8_t hi = vqmovn_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3)));
                const int8x16_t s8 = vcombine_s8(lo, hi);
                vst1q_s8(q + i, vreinterpretq_s8_u8(
                                    veorq_u8(vreinterpretq_u8_s8(s8),
                                             vreinterpretq_u8_s32(bias))));
            }
            for (; i < k1; i++) {
                int v = (int)lrintf(row[i] * (bf > 0.f ? 127.0f / bf : 0.f));
                if (v > 127) v = 127;
                else if (v < -127) v = -127;
                q[i] = (int8_t)(v + 128);
            }
        }
    }
}

/* Per-E2_I8_BLOCK symmetric int8 quantization, storing a+128 (see int8.h). The float->int32 step
 * rounds to nearest-even, which is what the scalar reference's lrintf does, so the bytes match the
 * reference exactly and the GEMM tests compare bit-for-bit rather than approximately. */
void e2_quantize_blocks_neon(int8_t *out, float *scales, const float *x, int M, int K, float b_lo,
                             float b_hi) {
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const float32x4_t vlo = vdupq_n_f32(b_lo), vhi = vdupq_n_f32(b_hi);
    const int clamp = b_lo > -INFINITY || b_hi < INFINITY;
    const uint8x16_t bias = vdupq_n_u8(0x80);
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        int8_t *orow = out + (size_t)m * K;
        for (int b = 0; b < nb; b++) {
            const int k0 = b * E2_I8_BLOCK;
            const int k1 = (k0 + E2_I8_BLOCK) < K ? (k0 + E2_I8_BLOCK) : K;
            float32x4_t vmax = vdupq_n_f32(0.f);
            int k = k0;
            for (; k + 16 <= k1; k += 16) {
                float32x4_t a0 = vld1q_f32(xr + k), a1 = vld1q_f32(xr + k + 4);
                float32x4_t a2 = vld1q_f32(xr + k + 8), a3 = vld1q_f32(xr + k + 12);
                if (clamp) {
                    a0 = vminq_f32(vmaxq_f32(a0, vlo), vhi);
                    a1 = vminq_f32(vmaxq_f32(a1, vlo), vhi);
                    a2 = vminq_f32(vmaxq_f32(a2, vlo), vhi);
                    a3 = vminq_f32(vmaxq_f32(a3, vlo), vhi);
                }
                vmax = vmaxq_f32(vmax, vabsq_f32(a0));
                vmax = vmaxq_f32(vmax, vabsq_f32(a1));
                vmax = vmaxq_f32(vmax, vabsq_f32(a2));
                vmax = vmaxq_f32(vmax, vabsq_f32(a3));
            }
            float amax = vmaxvq_f32(vmax);
            for (; k < k1; k++) {
                const float v = clamp ? (xr[k] < b_lo ? b_lo : (xr[k] > b_hi ? b_hi : xr[k])) : xr[k];
                const float a = fabsf(v);
                if (a > amax) amax = a;
            }
            const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
            const float32x4_t inv = vdupq_n_f32(1.0f / sc);
            k = k0;
            for (; k + 16 <= k1; k += 16) {
                float32x4_t a0 = vld1q_f32(xr + k), a1 = vld1q_f32(xr + k + 4);
                float32x4_t a2 = vld1q_f32(xr + k + 8), a3 = vld1q_f32(xr + k + 12);
                if (clamp) {
                    a0 = vminq_f32(vmaxq_f32(a0, vlo), vhi);
                    a1 = vminq_f32(vmaxq_f32(a1, vlo), vhi);
                    a2 = vminq_f32(vmaxq_f32(a2, vlo), vhi);
                    a3 = vminq_f32(vmaxq_f32(a3, vlo), vhi);
                }
                const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(a0, inv));
                const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(a1, inv));
                const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(a2, inv));
                const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(a3, inv));
                const int8x8_t lo = vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1)));
                const int8x8_t hi = vqmovn_s16(vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3)));
                /* store a+128, see int8.h */
                vst1q_s8(orow + k, vreinterpretq_s8_u8(
                                       veorq_u8(vreinterpretq_u8_s8(vcombine_s8(lo, hi)), bias)));
            }
            for (; k < k1; k++) {
                const float v = clamp ? (xr[k] < b_lo ? b_lo : (xr[k] > b_hi ? b_hi : xr[k])) : xr[k];
                int32_t r = (int32_t)lrintf(v / sc);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                orow[k] = (int8_t)(r + 128);
            }
            scales[(size_t)m * nb + b] = sc;
        }
    }
}
