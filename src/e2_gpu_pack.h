/* e2_gpu_pack.h - the one definition of the matrix-engine weight layout.
 *
 * Shared by the backend (src/e2_gpu.c) and the dev-only kernel benchmark
 * (tools/gpu_bench.c) so the two cannot drift: a stale layout still runs and
 * returns wrong numbers, which is how a whole round gets lost.
 *
 * Layout (E2_XFB/E2_XTB in tools/gpu/e2_xmx.cl):
 *   pq[(fb * nkc + kstep) * 256 + lane * 32 + r * 4 + c]  int8
 *   fb    = output feature / 8,  r = output feature % 8
 *   kstep = k / 32,              lane = (k / 4) % 8,   c = k % 4
 * i.e. the DPAS A operand: lane l holds k = 4l..4l+3 of each of the 8 features
 * of its block.  Feature blocks are the outer dimension so one sub-group
 * streams its whole weight tile as a contiguous run.
 */
#ifndef E2_GPU_PACK_H
#define E2_GPU_PACK_H

#include "e2_internal.h"

/* Matrix-engine tile geometry.  These MUST equal E2_XTB/E2_XFB/E2_RSG/E2_CSG in
 * tools/gpu/e2_xmx.cl: the kernel computes the tile, the launcher only sizes
 * the grid, and a mismatch that still runs would return wrong numbers.  RSG
 * sub-groups share a feature tile and CSG sub-groups a token tile, so one work
 * group covers E2_GPU_TM tokens by E2_GPU_TN features and the operand traffic
 * per GEMM is m/E2_GPU_TM weight passes plus n/E2_GPU_TN activation passes.
 * 128x64 at 256 threads measured best end to end: 64x64 (Windows 55.5 ms at
 * text 512) -> 128x64 (52.5) -> 256x64 (58.5), and 128 tokens is where the
 * weight tile starts being reused instead of re-streamed. */
#define E2_GPU_XTB 2
#define E2_GPU_XFB 2
#define E2_GPU_RSG 8
#define E2_GPU_CSG 4
#define E2_GPU_TM  (E2_GPU_RSG * E2_GPU_XTB * 8)   /* 128 tokens per work group */
#define E2_GPU_TN  (E2_GPU_CSG * E2_GPU_XFB * 8)   /* 64 features per work group */
#define E2_GPU_SG  (E2_GPU_RSG * E2_GPU_CSG)       /* sub-groups per work group */

/* Second tile variant (tools/gpu/e2_xmx_big.cl): the same kernel with a bigger
 * token tile, which is what the long rows need because the weight passes then
 * drop from m/128 to m/256.  The launcher picks by row count; below the
 * crossover the bigger tile just idles most of the machine. */
#define E2_GPU_BIG_RSG 16
#define E2_GPU_BIG_CSG 4
#define E2_GPU_BIG_TM  (E2_GPU_BIG_RSG * 2 * 8)    /* 256 tokens */
#define E2_GPU_BIG_TN  (E2_GPU_BIG_CSG * 2 * 8)    /* 64 features */
#define E2_GPU_BIG_SG  (E2_GPU_BIG_RSG * E2_GPU_BIG_CSG)
#define E2_GPU_BIG_MIN 768                        /* rows: measured crossover */

/* Repack w into the matrix-engine A layout; pq needs n*k bytes.  Returns 0, or
 * -1 if the shape has no matrix-engine layout. */
static inline int e2_gpu_dpas_pack(const e2_qmat *w, int8_t *pq)
{
    if ((w->n % 8) || (w->k % 32)) return -1;
    const int nkc = w->k / 32, nqb = w->n / 8;
    const size_t ts = w->tstride ? w->tstride : (size_t)(w->k / 4) * 64;
    for (int fb = 0; fb < nqb; fb++)
        for (int kstep = 0; kstep < nkc; kstep++) {
            uint8_t *dst = (uint8_t *)pq + ((size_t)fb * nkc + kstep) * 256;
            for (int lane = 0; lane < 8; lane++)
                for (int r = 0; r < 8; r++) {
                    const int col = fb * 8 + r;
                    const int k = (kstep * 8 + lane) * 4;
                    const int8_t *sp = w->q + (size_t)(col / 16) * ts +
                                       (size_t)(k / 4) * 64 + (col % 16) * 4;
                    memcpy(dst + lane * 32 + r * 4, sp, 4);
                }
        }
    return 0;
}

#endif /* E2_GPU_PACK_H */
