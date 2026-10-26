/* Regression for SIMD exp range reduction: the polynomial is 1+r+r*r*P(r),
 * not 1+r*P(r). Compile checks without -ffast-math so NaNs cannot pass. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static void check_softmax(const e2_kernels *K, const char *name) {
    const int lengths[] = { 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 513, 514, 2048 };
    const float edge = .34657359027997265f;
    const float values[] = { 0, edge - 1e-6f, edge + 1e-6f, -edge - 1e-6f,
                             -edge + 1e-6f, -1, -10, -87, -1000, -1e30f, -INFINITY };
    double maxdiff = 0;
    for (int ni = 0; ni < 13; ni++) {
        int n = lengths[ni];
        float x[2048];
        double ref[2048], sum = 0;
        for (int i = 0; i < n; i++) {
            x[i] = values[i % 11];
            ref[i] = exp((double)x[i] - edge - 1e-6f);
            sum += ref[i];
        }
        K->softmax(x, n);
        double got_sum = 0;
        for (int i = 0; i < n; i++) {
            CHECK(isfinite(x[i]) && x[i] >= 0);
            got_sum += x[i];
            double d = fabs(x[i] - ref[i] / sum);
            if (d > maxdiff) maxdiff = d;
            CHECK(d < 3e-7);
        }
        CHECK(fabs(got_sum - 1) < 2e-6);
    }
    /* Straddle every nearest-integer range-reduction boundary of negative exp. */
    for (int b = 0; b < 125; b++) {
        float a[32], c[32];
        for (int i = 0; i < 32; i++) a[i] = c[i] = 0;
        a[0] = -(b + .5f) * .6931471805599453f - 1e-6f;
        c[0] = -(b + .5f) * .6931471805599453f + 1e-6f;
        K->softmax(a, 32); K->softmax(c, 32);
        CHECK(fabs(a[0] - c[0]) < 2e-7);
    }
    float gelu[2048];
    for (int i = 0; i < 2048; i++) gelu[i] = (i - 1024) * .01f;
    K->gelu_tanh(gelu, 2048);
    for (int i = 0; i < 2048; i++) {
        double x = (i - 1024) * .01f;
        double ref = .5 * x * (1 + tanh(.7978845608028654 * (x + .044715 * x * x * x)));
        CHECK(isfinite(gelu[i]) && fabs(gelu[i] - ref) < 3e-6);
    }
    printf("%s softmax/GELU accuracy and continuity OK; max probability diff %.9g\n", name, maxdiff);
}

int main(void) {
    e2_cpu_init();
    check_softmax(e2_k(), "selected CPU");
#if defined(__x86_64__)
    extern const e2_kernels *e2_avx2_kernels(void);
    extern const e2_kernels *e2_avx512_kernels(void);
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        check_softmax(e2_avx2_kernels(), "AVX2");
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("fma"))
        check_softmax(e2_avx512_kernels(), "AVX512");
    else puts("AVX512 execution skipped: unsupported CPU");
#endif
    return 0;
}
