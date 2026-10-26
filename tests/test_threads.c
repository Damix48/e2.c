#include "../src/threads.c"
#include "validation.h"
#include <assert.h>
#include <math.h>

int main(void) {
#if defined(__linux__)
    const char *cases[] = { "max 100000", "100000 100000", "150000 100000", "0 100000", "bad", "100000 0" };
    const int want[] = {64,1,2,64,64,64};
    for (int i = 0; i < 6; i++) {
        FILE *f = tmpfile(); assert(f); fputs(cases[i], f); rewind(f);
        assert(quota_threads(f) == want[i]); fclose(f);
    }
    cpu_set_t original;
    if (!sched_getaffinity(0, sizeof original, &original)) {
        cpu_set_t one; CPU_ZERO(&one);
        for (int i = 0; i < CPU_SETSIZE; i++) if (CPU_ISSET(i, &original)) { CPU_SET(i, &one); break; }
        assert(!sched_setaffinity(0, sizeof one, &one));
        unsetenv("E2_THREADS"); assert(e2_pool_global_init(0) == 1); e2_pool_global_free();
        assert(!sched_setaffinity(0, sizeof original, &original));
    }
#endif
    float embedding[768] = {1.f};
    assert(e2_valid_embedding(embedding));
    embedding[0] = NAN; assert(!e2_valid_embedding(embedding));
    embedding[0] = INFINITY; assert(!e2_valid_embedding(embedding));
    embedding[0] = 0; assert(!e2_valid_embedding(embedding));
    float input[] = {0.f, 1.f, -1.f};
    assert(e2_valid_floats(input, 3, -1, 1));
    assert(!e2_valid_floats(input, 3, 0, 1));
    input[2] = NAN; assert(!e2_valid_floats(input, 3, -1, 1));
    puts("CPU budget and strict finite-value checks OK");
    return 0;
}
