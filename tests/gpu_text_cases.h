/* Shared, trusted golden text inputs for GPU replay and soak checks. */
#ifndef E2_TEST_GPU_TEXT_CASES_H
#define E2_TEST_GPU_TEXT_CASES_H
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char name[65];
    int32_t n, ids[2048];
    float ref[768];
} e2_test_text_case;

static e2_test_text_case *e2_test_read_text_cases(const char *dir, int *count) {
    char path[4096];
    int len = snprintf(path, sizeof path, "%s/text_checks.bin", dir);
    assert(len > 0 && (size_t)len < sizeof path);
    FILE *f = fopen(path, "rb"); assert(f);
    uint32_t h[4]; assert(fread(h, sizeof h, 1, f) == 1);
    assert(h[0] == 0x54583245 && h[1] > 1 && h[1] <= 128 && h[3] == 768);
    e2_test_text_case *cases = calloc(h[1], sizeof *cases); assert(cases);
    for (uint32_t i = 0; i < h[1]; i++) {
        e2_test_text_case *c = cases + i;
        assert(fread(c->name, 64, 1, f) == 1 && fread(&c->n, 4, 1, f) == 1);
        assert(c->n > 0 && c->n <= 2048);
        assert(fread(c->ids, 4, (size_t)c->n, f) == (size_t)c->n);
        assert(fread(c->ref, sizeof c->ref, 1, f) == 1);
    }
    assert(fgetc(f) == EOF); fclose(f);
    *count = (int)h[1];
    return cases;
}
#endif
