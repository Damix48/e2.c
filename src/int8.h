/* int8.h — weight-only int8 quantization (per output row) plus per-row int8 activation quantization.
 *
 * Scheme (the same arithmetic ggml's Q8_0 uses, generalized to per-row scales):
 *
 *   w[n,k] = sw[n] * b[n,k]                      b int8,  sw[n] = max|w[n,:]|/127
 *   x[m,k] = s[m]  * (a[m,k] - 128)              a uint8, s[m]  = max|x[m,:]|/127
 *
 *   sum_k x*w = s[m]*sw[n] * ( sum_k a*b - 128 * sum_k b )
 *
 * `sum_k b` is a per-row constant (rowsum), so the zero-point correction is one int32 subtract per
 * output element. That lets the hot loop be a pure uint8 x int8 dot product, which is what
 * vpdpbusd (AVX-VNNI / AVX512-VNNI) and vpmaddubsw (AVX2) do natively.
 *
 * Weights are stored *quad-interleaved* in 8-row blocks: for the block of output rows starting at
 * n8 = n & ~7, the k-quads of the row's k are at
 *
 *     w[((n>>3)*(K/4) + k/4)*32 + (n&7)*4 + k%4]
 *
 * i.e. [N/8][K/4][8][4]: each (k-quad, 8 rows) is one 32-byte vector, and consecutive k-quads of an
 * 8-row block are adjacent, so the kernel streams sequentially through 4 KB per K=512 block instead
 * of stepping N*4 bytes per k-quad (which no prefetcher follows and which cost the 15-token query
 * 1.5 ms the first time this layout was tried with a [K/4][N][4] stride).
 *
 * Interleaving is what makes the kernel cheap: one 32-byte load covers 8 output rows' worth of one k-quad, so a
 * vpdpbusd whose eight 32-bit lanes are eight *output rows* advances 4 k-values for 8 rows at once
 * (4x4 = 16 int32 lane-MACs, 32 int8 MACs). The alternative - lanes = output columns of a 32-byte
 * activation block - needs MR*NR float and int32 accumulators to keep the activation re-reads down,
 * and then the operand traffic is (1/MR + 1/NR) bytes/MAC with MR*NR capped at ~8 by the register
 * file. Here the activation is a 4-byte broadcast reused by every accumulared row, so its traffic is
 * 1/8 byte/MAC no matter how many tokens are in flight, and only the weight side scales with MR.
 * Measured on the reference machine, one layer of the text tower (M=200, K=512, N=8192, 14 threads):
 * 961 GMAC/s interleaved vs 764 for the row-major 2x4 tile, and the gap widens with thread count
 * because it is the shared-L3 traffic that saturates.
 *
 * The per-block zero-point is stored transposed (bsq[blk*N + n]) for the same reason: the kernel
 * needs it as eight lanes of eight rows, which is one contiguous 32-byte load.
 *
 * Per-row activation scales would make the whole sum one int32 dot and cost 0.0014 fidelity on the
 * document case (0.9989, below the 0.999 bar), so scales stay per E2_I8_BLOCK and the float FMA runs
 * once per block per row-group.
 */
#ifndef E2_INT8_H
#define E2_INT8_H

#include <stdint.h>
#include <string.h>

/* Packed byte operands need not have int32 alignment (e.g. attention rows of width 9). */
static inline int32_t e2_load_i32(const void *p) {
    int32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}
static inline int32_t e2_load_i32_tail(const void *p, int n) {
    if (n >= 4) return e2_load_i32(p);
    int32_t v = 0;
    memcpy(&v, p, (size_t)n);
    return v;
}

/* Activation/weight quantization block. The kernel converts each block's int32 dot to float and
 * applies the scale, so this is a direct uop/throughput knob: 32 keeps the most fidelity, 128 is
 * ~1.5x faster on the int8 GEMM (both towers), 256 is another 1.15x but costs 0.0006 cosine on the
 * document (0.99886, below the 0.999 bar this engine holds itself to), and a per-row scale would
 * cost 0.0014. Measured on the reference machine, one thread, M=192 K=512 N=8192: b32 91 GMAC/s,
 * b128 125, b256 154, per-row (=K) 165. Must divide every K in the model (512, 768, 1024, 2048,
 * 3072). */
#ifndef E2_I8_BLOCK
#define E2_I8_BLOCK 128
#endif

/* Tokens per pass; one weight pass per E2_I8_MR tokens is what the interleaved kernel's traffic
 * costs, and it is the register file that caps it (2 live YMM per token: int32 through the block,
 * float across blocks). 8 is also the padded-token unit, and E2_MAX_SEQ is a multiple of it. */
#ifndef E2_I8_MR
#define E2_I8_MR 8
#endif

typedef struct {
    int8_t *w;         /* [N/8][K/4][8][4] quad-interleaved, Npad * ceil(K/4) * 4 bytes */
    int32_t *bsq;      /* [nblocks][N]: per-block weight sum, transposed for a 8-row vector load */
    float *scale;      /* N: per-row weight scale */
    int N, K, nblocks, Npad; /* Npad = N rounded up to 8: the kernel reads 8 rows at a time */
    /* Optional output saturation, applied at the kernel's store: the audio tower's clipped linears
     * clamp their result to a learned per-tensor range, and doing it here costs two uops per 8
     * floats instead of a second pass over M*N (which was 60 ms of an 11 s clip). out_min > -1e30
     * means "clipped"; the default is no clamp. */
    float out_min, out_max;
    /* 1 when the arrays point into the q8cache mapping (E2_Q8_READ) instead of the heap: e2_q8w_free
     * must not free them, and the mapping must outlive the model. */
    int borrowed;
} e2_q8w;

/* y[m,n] for n in [n0,n1) = x[m,:] @ w[n,:]^T. Splitting along N is what lets the text tower run a
 * GEMM on the thread pool without duplicating any weight traffic. */
typedef void (*e2_i8_gemm_fn)(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                              int M, int N, int n0, int n1);

/* Quantize a bf16/f32 tensor with shape [N, K] into q (allocates). Returns 0 on success. */
int e2_q8w_from_f32(e2_q8w *q, const float *w, int N, int K);
int e2_q8w_from_bf16(e2_q8w *q, const uint16_t *w, int N, int K);
/* Same, from a strided f32 source: element (n,k) is src[n*ld + k]. Lets a projection output
 * [S][kv_dim] be quantized one head at a time without copying it into a contiguous block. */
int e2_q8w_from_f32_ld(e2_q8w *q, const float *w, int N, int K, int ld);
void e2_q8w_free(e2_q8w *q);

/* Precondition for every e2_i8_gemm backend: K must be a multiple of E2_I8_BLOCK. The SIMD kernels
 * walk k-quads up to the end of each block without looking at K (the scalar reference does not need
 * this, which is exactly why it can hide), so a short K reads into the next 8-row group. Keeping
 * that let the kernels keep a constant trip count on the inner loop; a caller whose K is arbitrary
 * - the attention's context product uses K = keys - zero-pads the weight rows and the activation
 * rows up to the block. Every projection in the model already has a K that is a multiple of 128. */
/* The scalar reference GEMM (int8 weights, biased int8 activations); the backends and the GPU
 * compare against it, and it is the fallback on machines without VNNI/AVX2. */
void e2_i8_gemm_scalar(float *y, const int8_t *qa, const float *a_scales, const e2_q8w *w,
                       int M, int N, int n0, int n1);

/* Packed-operand int8 GEMM shared by the attention of both towers:
 *
 *   C[m][n] = bsc[n] * sum_b asc[m][b] * ( A[m] . B[n] - 128 * bz[b][n] )
 *
 * A is the unsigned (biased) activation [M][K] with row stride lda, B is the signed weight packed
 * k-quad-major as Bp[((k>>2)*Np + n)*4 + (k&3)] (Np = N rounded up to 16), bsc is one scale per
 * output column and bz the per-128-block column sums the +128 bias needs. It is the same arithmetic
 * as the weight GEMM, just with an operand layout the attention's Q/K/P/V^T packs share - the
 * logits product (A=Q, B=K^T), the context product (A=P, B=V^T) and the vision's 64-deep head all
 * go through the one kernel. NULL when the CPU has no AVX512-VNNI. */
typedef void (*e2_i8_attn_fn)(float *C, int ldc, const uint8_t *A, int lda, const float *asc,
                              const uint8_t *Bp, const float *bsc, const int32_t *bz, int M, int Np,
                              int N, int K);
extern e2_i8_attn_fn e2_i8_attn_gemm;

/* softmax + per-128-block int8 quantisation of the probabilities, fused so the row is never written
 * back as fp32 and never read twice for the quantiser: x[r][0..n) is softmaxed in place and the
 * unbiased int8 (q+128) lands in q8 with scales[r][b] = block_max*inv/127. Rows in [n, rows) are
 * untouched. NULL when the CPU has no AVX512-VNNI (callers then run softmax then e2_i8_quantize). */
typedef void (*e2_sq_fn)(float *x, int8_t *q8, float *scales, int rows, int n, int nb);
extern e2_sq_fn e2_softmax_quant_i8;

#endif /* E2_INT8_H */
