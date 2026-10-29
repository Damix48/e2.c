/* Test-only dispatch override: exercise the complete tower, not just isolated GEMMs. */
#define e2_cpu_init e2_cpu_init_auto
#include "../src/kernels.c"
#undef e2_cpu_init
#include "e2.h"
#include "test_metrics.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static int g_test_backend;
#if defined(__aarch64__) /* compiled into the same TU as kernels.c above, so declare these by hand */
extern void e2_i8_gemm_neon(float *, const int8_t *, const float *, const e2_q8w *, int, int, int, int);
extern void e2_quantize_blocks_neon(int8_t *, float *, const float *, int, int, float, float);
static int has_neon_dotprod(void) { return cpu_has_neon_dotprod(); }
#else
static int has_neon_dotprod(void) { return 0; }
#endif
void e2_cpu_init(void) {
    e2_cpu_init_auto();
    /* Keep dispatch.c's idea of the backend in step with the pointers faked below: text.c now asks
     * the table rather than testing e2_i8_quantize, so a stale id would pick int8 with no quantizer. */
    switch (g_test_backend) {
    case 0: e2_dispatch_set_cpu_backend(E2_BK_CPU_SCALAR); break;
    case 1: e2_dispatch_set_cpu_backend(E2_BK_CPU_AVX2); break;
    case 2: e2_dispatch_set_cpu_backend(E2_BK_CPU_VNNI); break;
    case 3: e2_dispatch_set_cpu_backend(E2_BK_CPU_AVX512); break;
    case 4: e2_dispatch_set_cpu_backend(E2_BK_CPU_NEON); break;
    default: break;
    }
    if (g_test_backend == 0) {
        g_active = &g_scalar;
        e2_sgemm_nn_range = e2_sgemm_nn_range_scalar;
        e2_i8_quantize = NULL; e2_i8_attn_gemm = NULL; e2_softmax_quant_i8 = NULL;
    } else if (g_test_backend == 1 || g_test_backend == 2) {
#if defined(__x86_64__) || defined(_M_X64) /* the AVX2/VNNI kernels are x86-only */
        g_active = e2_avx2_kernels();
        e2_sgemm_nn_range = e2_sgemm_nn_range_avx2;
        e2_i8_gemm = g_test_backend == 1 ? e2_i8_gemm_avx2 : e2_i8_gemm_vnni_range;
        e2_i8_quantize = g_test_backend == 1 ? e2_quantize_blocks_avx2 : e2_quantize_blocks_vnni;
        e2_i8_attn_gemm = g_test_backend == 1 ? NULL : e2_i8_attn_gemm_vnni;
        e2_softmax_quant_i8 = e2_softmax_quant_i8_avx2;
#endif
#if defined(__aarch64__)
    } else if (g_test_backend == 4) {
        e2_i8_gemm = e2_i8_gemm_neon;
        e2_i8_quantize = e2_quantize_blocks_neon;
#endif
    }
}
int main(int argc, char **argv) {
    assert(argc == 3);
    unsetenv("E2_Q8_READ"); unsetenv("E2_Q8_WRITE"); unsetenv("E2_INT8");
    /* this test compares backend outputs, so no ambient precision override may leak in */
    unsetenv("E2_ATTN_TEXT"); unsetenv("E2_ATTN_IMAGE");
    unsetenv("E2_ATTN_I8"); unsetenv("E2_VATTN_OFF");
    const char *names[] = {"scalar bf16", "AVX2 int8", "AVX-VNNI int8", "AVX512-VNNI int8", "NEON int8"};
    const int nbackends = 5;
    for (int backend = 0; backend < nbackends; backend++) {
        if ((backend == 1 && !cpu_has_avx2()) || (backend == 2 && !cpu_has_avxvnni()) ||
            (backend == 3 && !cpu_has_avx512vnni()) || (backend == 4 && !has_neon_dotprod())) {
            printf("unsupported: %s\n", names[backend]);
            continue;
        }
        g_test_backend = backend;
        e2_model *m = e2_load(argv[1]); assert(m);
        const char *fixtures[] = {"text_query", "text_doc"};
        for (int c = 0; c < 2; c++) {
            char path[1024]; snprintf(path, sizeof path, "%s/%s.bin", argv[2], fixtures[c]);
            FILE *f = fopen(path, "rb"); assert(f);
            int32_t n, dim, ids[2048]; float ref[768], out[768];
            assert(fread(&n, 4, 1, f) == 1 && fread(&dim, 4, 1, f) == 1 && n > 0 && n <= 2048 && dim == 768);
            assert(fread(ids, 4, (size_t)n, f) == (size_t)n && fread(ref, 4, 768, f) == 768); fclose(f);
            assert(!e2_text_embed(m, ids, n, out));
            assert(e2_test_unit(names[backend], out, 768) && e2_test_match(names[backend], out, ref, 768, .999));
            printf("%s %s cosine %.9f\n", names[backend], fixtures[c], e2_test_cosine(out, ref, 768));
        }
        e2_free(m);
    }
    return 0;
}
