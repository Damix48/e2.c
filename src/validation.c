/* Trust-boundary checks must not inherit the kernels' finite-only assumptions. */
#include <math.h>
#include "validation.h"

int e2_valid_floats(const float *p, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++)
        if (!isfinite(p[i]) || p[i] < lo || p[i] > hi) return 0;
    return 1;
}

int e2_valid_embedding(const float *p) {
    double norm = 0;
    for (int i = 0; i < E2_EMBED_DIM; i++) {
        if (!isfinite(p[i])) return 0;
        norm += (double)p[i] * p[i];
    }
    return fabs(norm - 1.0) < 1e-3;
}
