/* kernels_neon.c — the fp32 kernels the text and vision towers still run on the CPU, for aarch64.
 *
 * NEON (ASIMD) is mandatory in the aarch64 baseline, so unlike the AVX2/AVX-512 backends this file
 * needs no -march flag and no CPUID gate: on this architecture it is always usable.
 *
 * The one kernel here that matters for throughput is the blocked f32 GEMM. Both text attention
 * products go through it (logits = Q K^T, context = P V) and on a scalar host they were 75 % of a
 * 196-token document, so the structure is deliberately identical to the AVX2 kernel's: 64x64x64
 * blocking with the A and B panels resident in L1, and an 8x8 register tile walking them. The tile
 * is 8 rows x 8 columns, held as 16 float32x4 accumulators - twice the AVX2 register count for the
 * same tile because a NEON register is half as wide, which is why the aarch64 machine can afford it
 * (32 vector registers against AVX2's 16).
 *
 * The inner loop is 8 broadcasts + 16 FMA per k: one b vector pair is reused by all 8 rows, which is
 * the same "1/MR of the B traffic" trade the int8 kernel makes.
 */
#include <arm_neon.h>

#include "kernels.h"

/* C[0..8) rows, 8 columns, K depth. `accumulate` is set on the k-blocks after the first. */
static inline void sge_micro8(float *C, int ldc, const float *A, int lda, const float *B, int ldb,
                              int kk, int accumulate) {
    float32x4_t acc[8][2];
    for (int i = 0; i < 8; i++) {
        acc[i][0] = accumulate ? vld1q_f32(C + (size_t)i * ldc) : vdupq_n_f32(0.f);
        acc[i][1] = accumulate ? vld1q_f32(C + (size_t)i * ldc + 4) : vdupq_n_f32(0.f);
    }
    for (int k = 0; k < kk; k++) {
        const float *brow = B + (size_t)k * ldb;
        const float32x4_t b0 = vld1q_f32(brow), b1 = vld1q_f32(brow + 4);
        for (int i = 0; i < 8; i++) {
            const float32x4_t av = vdupq_n_f32(A[(size_t)i * lda + k]);
            acc[i][0] = vfmaq_f32(acc[i][0], av, b0);
            acc[i][1] = vfmaq_f32(acc[i][1], av, b1);
        }
    }
    for (int i = 0; i < 8; i++) {
        vst1q_f32(C + (size_t)i * ldc, acc[i][0]);
        vst1q_f32(C + (size_t)i * ldc + 4, acc[i][1]);
    }
}

void e2_sgemm_nn_range_neon(float *C, int ldc, const float *A, int lda, const float *B, int ldb,
                            int M, int N, int K, int m0, int m1) {
    for (int mb = m0; mb < m1; mb += 64) {
        const int me = m1 - mb < 64 ? m1 - mb : 64;
        for (int k0 = 0; k0 < K; k0 += 64) {
            const int kk = K - k0 < 64 ? K - k0 : 64;
            for (int n0 = 0; n0 < N; n0 += 64) {
                const int nn = N - n0 < 64 ? N - n0 : 64;
                for (int i = 0; i + 8 <= me; i += 8) {
                    int j = 0;
                    for (; j + 8 <= nn; j += 8)
                        sge_micro8(C + (size_t)(mb + i) * ldc + n0 + j, ldc,
                                   A + (size_t)(mb + i) * lda + k0, lda,
                                   B + (size_t)k0 * ldb + n0 + j, ldb, kk, k0 != 0);
                    for (; j < nn; j++) { /* column tail */
                        float acc[8] = { 0 };
                        for (int k = 0; k < kk; k++)
                            for (int i2 = 0; i2 < 8; i2++)
                                acc[i2] += A[(size_t)(mb + i + i2) * lda + k0 + k] *
                                           B[(size_t)(k0 + k) * ldb + n0 + j];
                        for (int i2 = 0; i2 < 8; i2++) {
                            float *cp = C + (size_t)(mb + i + i2) * ldc + n0 + j;
                            *cp = k0 ? *cp + acc[i2] : acc[i2];
                        }
                    }
                }
                for (int i = (me & ~7); i < me; i++) { /* row tail */
                    for (int j = 0; j < nn; j++) {
                        float acc = 0;
                        for (int k = 0; k < kk; k++)
                            acc += A[(size_t)(mb + i) * lda + k0 + k] * B[(size_t)(k0 + k) * ldb + n0 + j];
                        float *cp = C + (size_t)(mb + i) * ldc + n0 + j;
                        *cp = k0 ? *cp + acc : acc;
                    }
                }
            }
        }
    }
    (void)M;
}
