/* The tiled int8 GEMM's shape, in one place because two languages compile it: kernels/e2.cl (OpenCL,
 * built by ocloc with -I pointing here) and the C host (gpu.h, and gpu_bench, which includes this
 * header directly). Keeping them in separate files is how a stale E2_BM silently halved the grid
 * during the last round - the kernel still ran, and returned wrong numbers.
 *
 * Plain #defines only - this header is included by both C and OpenCL C. */
#ifndef E2_TILE_H
#define E2_TILE_H

/* Workgroup: MG lanes along M (local_id(1)) and NG along N (local_id(0)). N is the fast local
 * dimension on purpose - IGC gives the lanes of one SIMD8 consecutive local_id(0) values, so with N
 * fastest the eight lanes of an instruction share a row and the activation load is a broadcast
 * instead of eight scattered rows (measured 10-30 %).
 *
 * MG=2 (BM=8 rows) rather than 8 or 16: halving the rows per work group doubles the number of work
 * groups, and this kernel wants work items more than it wants reuse. A sweep of 25 combinations put
 * this at 2.5x the 8x16/4x4 shape that was picked by hand. */
#define E2_GEMM_MG 2
#define E2_GEMM_NG 128

/* Register tile per lane: TM rows by TN columns. TN=2 beats TN=4 at every size even though a char16
 * weight load covers four rows in this layout and so wastes half of each load; the smaller tile wins
 * because it doubles the work items. TM=8 is 2x slower than TM=4 (32 accumulators run out of
 * registers before they run out of instructions) and 2x2 is 1.5x slower than 4x2. */
#define E2_GEMM_TM 4
#define E2_GEMM_TN 2

/* Tile covered by one workgroup. */
#define E2_BM (E2_GEMM_MG * E2_GEMM_TM)
#define E2_BN (E2_GEMM_NG * E2_GEMM_TN)

/* Sub-groups per DPAS GEMM work group (local size (8, E2_GEMM_GY, 1)). It is a pure operand-reuse
 * knob: the E2_GEMM_GY sub-groups of a work group cover E2_GEMM_GY * E2_DN features of the same
 * token tile, so the B (activation) tile is read once into L1 and shared instead of re-read by every
 * feature work group. No single value wins everywhere: the large-M batch (2048-row launches) and the
 * image (2520 rows) are 3-4 % faster at 16, while the small-M document (196 rows) is 4 % faster at
 * 32. The engine picks per launch by row count (`gemm_gy` in gpu.c: E2_GEMM_GY_SMALL below 512 rows,
 * E2_GEMM_GY above); E2_GPU_GY forces one value for the A/B. 16 is also the value that divides every
 * N the towers use (512/768/1024/2048/3072/12288) into exact work groups. Evaluated end to end on
 * the shipped k_gemm_dpas2: document 14.0 -> 12.9 ms, 32x196 batch 400 -> 392 ms, image 194 -> 189
 * ms, 8x17 batch 9.6 -> 9.4 ms. 8/48/64 measured no better than these. */
#define E2_GEMM_GY 16
#define E2_GEMM_GY_SMALL 32

/* XMX (DPAS) GEMM shape. The sub-group of 8 lanes computes an E2_DM (tokens) x E2_DN (features)
 * tile out of 8x8 DPAS blocks: E2_NTB token blocks of 8 rows and E2_NFB feature blocks of 8 columns.
 *
 * Which operand is which matters. The weights go in the A operand and the activations in B, because
 * that is the orientation where both operands are ONE 32-byte vector load per k-step: the B operand's
 * per-lane register is 32 contiguous bytes of a row-major quantized activation, and the A operand's is
 * 32 contiguous bytes of a weight layout a host-side repack produces (q8w_dpas in gpu.c). The other way
 * round the activations would have to be stored in the DPAS A layout by every quantization kernel.
 *
 * E2_DM is the knob: it is how many tokens share one weight read, so the weights are re-read M/E2_DM
 * times. E2_DN only affects how many tokens share one activation read. Both cost registers - the fp32
 * accumulation of the per-128-block activation scales is 8 x E2_NTB x E2_NFB registers, the same again
 * for the integer accumulators. A sweep of the pairs puts 16x16 first on every shape the text tower
 * uses. */
#define E2_DM 16
#define E2_DN 16
#define E2_NTB (E2_DM / 8)
#define E2_NFB (E2_DN / 8)

/* DPAS A operand byte offset: one k-step (8 quads) at a time, 8 lanes contiguous, so the eight
 * addresses of one operand load are 32 bytes apart instead of N*4. */
#define E2_DPAS_OFF(kq, rb, r8, kk, nqb) \
    (((size_t)((kq) >> 3) * (size_t)(nqb) + (size_t)(rb)) * 256 + (size_t)((kq) & 7) * 32 + \
     (size_t)(r8) * 4 + (size_t)(kk))

/* The vision tower's attention is the one shape this tile is not right for, because its head dim is
 * 64: the scores product is only two DPAS k-steps deep, so the operand loads, not the matrix engine,
 * set the pace (four 32-byte sends per four DPAS at 16x16). A 32-token by 32-key tile makes that eight
 * sends per sixteen DPAS and halves the weight re-reads on top; the context product's N is the 64-wide
 * head dim, so it takes a 64-token tile for the same reason on the V side. Both cost 16 integer plus
 * 32 float registers per lane, which is what the text tile's 16x16 was avoiding at K=512. */
#define E2_SDM 32
#define E2_SDN 32
#define E2_SNTB (E2_SDM / 8)
#define E2_SNFB (E2_SDN / 8)
#define E2_CDM 64
#define E2_CDN 16
#define E2_CNTB (E2_CDM / 8)
#define E2_CNFB (E2_CDN / 8)

#endif /* E2_TILE_H */
