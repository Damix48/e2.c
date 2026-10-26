#include <assert.h>
#include "test_metrics.h"

int main(void) {
    const float one[] = { 1, 0 }, other[] = { 0, 1 };
    const float zero[] = { 0, 0 }, nan[] = { NAN, 0 }, inf[] = { INFINITY, 0 };
    assert(e2_test_cosine(one, one, 2) == 1);
    assert(e2_test_cosine(one, other, 2) == 0);
    assert(isnan(e2_test_cosine(zero, one, 2)));
    assert(isnan(e2_test_cosine(nan, one, 2)));
    assert(isnan(e2_test_cosine(inf, one, 2)));
    assert(e2_test_match("valid", one, one, 2, 0.999));
    assert(e2_test_unit("valid", one, 2));
    assert(!e2_test_match("expected NaN rejection", nan, one, 2, 0.999));
    assert(!e2_test_match("expected infinity rejection", inf, one, 2, 0.999));
    assert(!e2_test_unit("expected zero rejection", zero, 2));
    puts("metrics OK");
    return 0;
}
