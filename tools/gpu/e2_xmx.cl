/* e2_xmx.cl - packed-int8 GEMM on the Xe matrix engine (DPAS), small tile.
 *
 * One work group covers E2_TM tokens x E2_TN features.  Two tile variants are
 * built from e2_xmx_body.h and the launcher picks by row count:
 *
 *   file           E2_TM  E2_TN  threads   best at (measured, GEMM phase only)
 *   e2_xmx.cl        128     64   256      m <= 512  (3.47 ms at m=128)
 *   e2_xmx_big.cl    256     64   512      m >= 1024 (22.2 vs 32.0 ms at 2048)
 *
 * A sweep of every (RSG,CSG) pair up to 1024 threads is in PERFORMANCE.md;
 * 128x64 was already the best small-m tile, and only the long rows want the
 * bigger token tile.
 */
#define E2_DPAS_BODY
#define E2_RSG 8    /* sub-groups sharing the feature tile */
#define E2_CSG 4    /* sub-groups sharing the token tile */
#include "e2_xmx_body.h"
