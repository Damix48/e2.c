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
void e2_cpu_init(void) {
    e2_cpu_init_auto();
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
    }
}
int main(int argc, char **argv) {
    assert(argc == 3);
    unsetenv("E2_Q8_READ"); unsetenv("E2_Q8_WRITE"); unsetenv("E2_INT8");
    const char *names[] = {"scalar bf16", "AVX2 int8", "AVX-VNNI int8", "AVX512-VNNI int8"};
    for (int backend = 0; backend < 4; backend++) {
        if ((backend == 1 && !cpu_has_avx2()) || (backend == 2 && !cpu_has_avxvnni()) ||
            (backend == 3 && !cpu_has_avx512vnni())) { printf("unsupported: %s\n", names[backend]); continue; }
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
