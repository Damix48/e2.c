/* Independent FP32 references at audio chunk/tile boundaries and the 30-second limit. */
#include "model.h"
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
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }
    /* Long steady tones are a quantization stress test, not the short speech-like fixture.
     * Keep the independent bf16 oracle strict; publish the int8 floor rather than hiding its drift. */
    double bar = m->use_int8 ? .98 : .999;
    double minimum = 1;
    int failed = 0;
    for (uint32_t c = 0; c < h[2]; c++) {
        uint32_t n[2]; float ref[768], out[768];
        assert(fread(n, sizeof n, 1, f) == 1 && n[0] <= 480000);
        assert(fread(ref, sizeof ref, 1, f) == 1);
        assert(e2_audio_tokens((int)n[0]) == (int)n[1]);
        int rc = e2_audio_embed(m, pcm, (int)n[0], out);
        if (rc || !e2_test_unit("audio boundary", out, 768) ||
            !e2_test_match("audio boundary", out, ref, 768, bar)) {
            fprintf(stderr, "%u samples: %s\n", n[0], e2_last_error());
            if (rc) return 1;
            failed = 1;
        }
        double cos = e2_test_cosine(out, ref, 768); if (cos < minimum) minimum = cos;
        printf("%u samples, %u soft tokens: cosine %.9f\n", n[0], n[1], cos);
    }
    printf("audio reference minimum cosine %.9f\n", minimum);
    e2_free(m); free(pcm); fclose(f);
    return failed;
}
