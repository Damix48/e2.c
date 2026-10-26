/* Independent FP32 audio embeddings: chunk boundaries and the full 30-second input. */
#include "model.h"
#include "e2gpu.h"
#include "test_metrics.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    assert(argc == 3);
    char path[1024]; snprintf(path, sizeof path, "%s/audio_checks.bin", argv[2]);
    FILE *f = fopen(path, "rb"); assert(f);
    uint32_t h[4]; assert(fread(h, sizeof h, 1, f) == 1);
    assert(h[0] == 0x41323245 && h[1] == 1 && h[2] == 9 && h[3] == 768);
    float *pcm = malloc(sizeof(float) * 480000); assert(pcm);
    for (int i = 0; i < 480000; i++)
        pcm[i] = (float)(.2 * sin(2 * 3.14159265358979323846 * 220 * i / 16000) +
                         .1 * cos(2 * 3.14159265358979323846 * 530 * i / 16000));
    e2_model *m = e2_load(argv[1]); assert(m);
    e2_gpu *g = e2_gpu_create(m);
    if (!g) { fprintf(stderr, "GPU create: %s\n", e2_gpu_last_error()); return 1; }
    double minimum = 1;
    for (uint32_t c = 0; c < h[2]; c++) {
        uint32_t n[2]; float ref[768], cpu[768], out[768];
        assert(fread(n, sizeof n, 1, f) == 1 && n[0] <= 480000);
        assert(fread(ref, sizeof ref, 1, f) == 1);
        assert(e2_audio_tokens((int)n[0]) == (int)n[1]);
        assert(!e2_audio_embed(m, pcm, (int)n[0], cpu));
        if (e2_gpu_audio_embed(g, pcm, (int)n[0], out)) {
            fprintf(stderr, "%u samples: %s\n", n[0], e2_gpu_last_error()); return 1;
        }
        /* Same disclosed 0.98 int8 tone-corpus floor as CPU, not a relaxed short-fixture gate. */
        assert(e2_test_unit("GPU audio boundary", out, 768));
        assert(e2_test_match("GPU audio vs FP32", out, ref, 768, .98));
        assert(e2_test_match("CPU audio vs FP32", cpu, ref, 768, .98));
        const double cos = e2_test_cosine(out, ref, 768);
        if (cos < minimum) minimum = cos;
        printf("%u samples, %u soft tokens: GPU vs FP32 %.9f, CPU vs FP32 %.9f, GPU vs CPU %.9f\n",
               n[0], n[1], cos, e2_test_cosine(cpu, ref, 768), e2_test_cosine(out, cpu, 768));
    }
    assert(fgetc(f) == EOF); fclose(f);
    printf("GPU audio reference minimum cosine %.9f\n", minimum);
    e2_gpu_destroy(g); e2_free(m); free(pcm);
    return 0;
}
