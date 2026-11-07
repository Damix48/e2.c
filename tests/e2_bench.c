#include "e2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "models";
    char mp[512], tp[512];
    snprintf(mp, sizeof(mp), "%s/e2-text.bin", dir);
    snprintf(tp, sizeof(tp), "%s/e2-tokenizer.bin", dir);
    e2_ctx *c = e2_load(mp, tp);
    if (!c) { fprintf(stderr, "%s\n", e2_error()); return 1; }
    const char *t[8] = {
        "What causes the northern lights?",
        "The northern lights are caused by charged particles from the sun.",
        "hello world", "a quick brown fox jumps over the lazy dog",
        "retrieval augmented generation", "int8 matrix multiplication",
        "machine learning", "the quick brown fox"
    };
    for (int N = 1; N <= 128; N *= 2) {
        const char **arr = malloc((size_t)N * sizeof(char *));
        for (int i = 0; i < N; i++) arr[i] = t[i % 8];
        float *out = malloc((size_t)N * 768 * sizeof(float));
        e2_embed_text_batch(c, arr, N, out, 768);  /* warm */
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            double t0 = now_ms();
            e2_embed_text_batch(c, arr, N, out, 768);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        printf("N=%3d  %8.2f ms  %6.3f ms/text\n", N, best, best / N);
        free(arr); free(out);
    }
    e2_free(c);
    return 0;
}
