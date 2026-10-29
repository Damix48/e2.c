/* activations.c — the scalar SiLU and GLU, compiled without -ffast-math.
 *
 * These two are the only kernels where -ffast-math is a correctness problem rather than a speed
 * knob: GCC turns `1.0f / (1.0f + expf(-x))` into `rcpps` (12-bit mantissa, ~1e-4 relative) in the
 * vectorised body while leaving `divss` in the scalar tail, so the same input gets two different
 * answers depending on its position in the row. On the audio tower that was worth 0.0028 of cosine
 * against the fixture (0.9996 -> 0.9982) - it amplified through twelve conformer layers.
 *
 * It lives in its own translation unit because taking the flag off all of kernels.c costs 17 % on
 * the aarch64 NEON path and 3.8x on the bf16 GEMMs (measured on the Ampere A1): -ffast-math is what
 * lets the compiler vectorise the reductions in `scalar_gemm_nt`, `scalar_rmsnorm` and friends, and
 * on aarch64 the elementwise vtable stays scalar, so they are on every path. Everything else in
 * kernels.c keeps it.
 */
#include <math.h>

#include "kernels.h"

void e2_silu_scalar(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] *= 1.0f / (1.0f + expf(-x[i]));
}

void e2_glu_scalar(float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] *= 1.0f / (1.0f + expf(-b[i]));
}
