/* Audio row padding and exactly-once work claims, without a checkpoint. */
#include "../src/audio.c"
#include <assert.h>
#include <unistd.h>

static _Atomic int g_seen[64];
static void count_gemm(float *y, const int8_t *x, const float *s, const e2_q8w *w,
                       int M, int N, int n0, int n1) {
    (void)y; (void)x; (void)s; (void)w; (void)M; (void)N; (void)n1;
    atomic_fetch_add(&g_seen[n0], 1);
    if (n0 == 0) usleep(100); /* expose a late worker resetting the column counter */
}
int main(void) {
    e2_cpu_init();
    if (!e2_i8_quantize) return 77;
    e2_audio a = {0};
    e2_model m = { .use_int8 = 1 };
    e2_alin lin = { .in_min = -1e31f, .in_max = 1e31f };
    assert(!aws_alloc(&a));
    const int Mp = (E2_A_MAX_TOK + E2_I8_MR - 1) / E2_I8_MR * E2_I8_MR;
    assert(a.q - a.tmp >= Mp * E2_A_OUT);
    assert(a.clamp8 - a.relk >= 16 * E2_A_HIDDEN);
    assert((uint8_t *)a.qa_scale - a.qa >= (ptrdiff_t)Mp * E2_A_INTER);
    const int sizes[] = { 1, 7, 8, 9, 13, 17, 743, 744, 745, 749, 750 };
    float *x = malloc(sizeof(float) * Mp * E2_A_INTER);
    float *w = malloc(sizeof(float) * 8 * E2_A_INTER);
    float *y = malloc(sizeof(float) * Mp * 8);
    assert(x && w && y);
    for (int i = 0; i < Mp * E2_A_INTER; i++) x[i] = 1.f;
    for (int i = 0; i < 8 * E2_A_INTER; i++) w[i] = 1.f;
    assert(!e2_q8w_from_f32(&lin.q8, w, 8, E2_A_INTER));
    for (int nt = 1; nt <= 4; nt *= 2) {
        e2_pool_global_init(nt);
        for (size_t c = 0; c < sizeof sizes / sizeof sizes[0]; c++) {
            int M = sizes[c];
            agemm_run(&m, &a, y, x, &lin, M, 8, E2_A_INTER);
            for (int i = 0; i < M * 8; i++) assert(fabsf(y[i] - E2_A_INTER) < .01f);
        }
        e2_i8_gemm_fn real = e2_i8_gemm;
        e2_i8_gemm = count_gemm;
        for (int r = 0; r < 100; r++) {
            for (int j = 0; j < 64; j++) atomic_store(&g_seen[j], 0);
            agemm_run(&m, &a, y, x, &lin, 16, 64, 32);
            for (int j = 0; j < 64; j++) assert(atomic_load(&g_seen[j]) == (j % 8 == 0));
        }
        e2_i8_gemm = real;
        e2_pool_global_free();
    }
    e2_q8w_free(&lin.q8);
    free(a.x); free(x); free(w); free(y);
    puts("audio padding and exactly-once scheduling OK");
    return 0;
}
