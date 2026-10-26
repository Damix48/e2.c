/* Test-side checks deliberately compile without -ffast-math. */
#ifndef E2_TEST_METRICS_H
#define E2_TEST_METRICS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static inline double e2_test_cosine(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return NAN;
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return na > 0 && nb > 0 ? dot / sqrt(na * nb) : NAN;
}

static inline int e2_test_match(const char *label, const float *got, const float *ref,
                                size_t n, double minimum) {
    const double c = e2_test_cosine(got, ref, n);
    if (isfinite(c) && c > minimum) return 1;
    fprintf(stderr, "%s: invalid result (cos=%.9f, required > %.6f)\n", label, c, minimum);
    return 0;
}

static inline int e2_test_unit(const char *label, const float *x, size_t n) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) { fprintf(stderr, "%s: non-finite embedding\n", label); return 0; }
        ss += (double)x[i] * x[i];
    }
    if (fabs(ss - 1.0) < 0.001) return 1;
    fprintf(stderr, "%s: embedding norm squared %.9f is not one\n", label, ss);
    return 0;
}

static inline int e2_test_double_cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static inline void e2_test_timing(double *samples, int n) {
    if (n <= 0) return;
    qsort(samples, (size_t)n, sizeof *samples, e2_test_double_cmp);
    const double median = (samples[(n - 1) / 2] + samples[n / 2]) * 0.5;
    const int p95 = (int)ceil(0.95 * n) - 1;
    printf("  timing: best %.3f ms, median %.3f ms, p95 %.3f ms (%d checked runs)\n",
           samples[0], median, samples[p95], n);
}

#endif
