/* e2_test - self-checks that do not need external assets. */
#include "e2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef E2_USE_WEBP
#include <webp/encode.h>
#endif

static void write_wav16(const char *path, const short *pcm, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    unsigned char h[44] = {0};
    memcpy(h, "RIFF", 4);
    unsigned v = 36 + n * 2; memcpy(h + 4, &v, 4);
    memcpy(h + 8, "WAVEfmt ", 8);
    v = 16; memcpy(h + 16, &v, 4);
    unsigned short one = 1, ch = 1, bits = 16, align = 2;
    unsigned sr = 16000, br = 32000;
    memcpy(h + 20, &one, 2); memcpy(h + 22, &ch, 2);
    memcpy(h + 24, &sr, 4); memcpy(h + 28, &br, 4);
    memcpy(h + 32, &align, 2); memcpy(h + 34, &bits, 2);
    memcpy(h + 36, "data", 4);
    v = n * 2; memcpy(h + 40, &v, 4);
    fwrite(h, 1, 44, f);
    fwrite(pcm, 2, (size_t)n, f);
    fclose(f);
}

static int fails;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

static float norm(const float *v, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "models";
    char tokp[512], textp[512], fullp[512];
    snprintf(tokp, sizeof(tokp), "%s/e2-tokenizer.bin", dir);
    snprintf(textp, sizeof(textp), "%s/e2-text.bin", dir);
    snprintf(fullp, sizeof(fullp), "%s/e2-text+image+audio.bin", dir);

    char msg[256];
    e2_ctx *txt = e2_load(textp, tokp);
    if (!txt) { printf("SKIP: %s\n", e2_error()); return 0; }

    CHECK(e2_modalities(txt) == E2_MOD_TEXT, "text model flags");
    float a[768], b[768], c[768];
    CHECK(e2_embed_text(txt, "What causes the northern lights?", a, 768) == 0, "embed text");
    CHECK(fabsf(norm(a, 768) - 1.0f) < 1e-4f, "text embedding normalized");
    CHECK(e2_embed_text(txt, "What causes the northern lights?", b, 768) == 0, "embed text 2");
    float d = 0;
    for (int i = 0; i < 768; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    CHECK(d == 0.0f, "deterministic");

    const char *texts[4] = {"hello", "a longer sentence about retrieval", "x", ""};
    float batch[4 * 768];
    CHECK(e2_embed_text_batch(txt, texts, 4, batch, 768) == 0, "batch");
    for (int k = 0; k < 4; k++) {
        CHECK(e2_embed_text(txt, texts[k], c, 768) == 0, "single for batch");
        d = 0;
        for (int i = 0; i < 768; i++) d = fmaxf(d, fabsf(c[i] - batch[k * 768 + i]));
        snprintf(msg, sizeof(msg), "batch matches single [%d]", k);
        CHECK(d == 0.0f, msg);
    }

    float t128[128], t256[256];
    CHECK(e2_embed_text(txt, "hello", t128, 128) == 0, "dim 128");
    CHECK(fabsf(norm(t128, 128) - 1.0f) < 1e-4f, "dim 128 normalized");
    CHECK(e2_embed_text(txt, "hello", t256, 256) == 0, "dim 256");
    CHECK(fabsf(norm(t256, 256) - 1.0f) < 1e-4f, "dim 256 normalized");
    CHECK(e2_embed_text(txt, "hello", b, 100) != 0, "invalid dim rejected");

    uint8_t rgb[64 * 64 * 3];
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            uint8_t *p = rgb + (y * 64 + x) * 3;
            p[0] = (uint8_t)(x * 4); p[1] = (uint8_t)(y * 4); p[2] = 100;
        }
    CHECK(e2_embed_image_rgb(txt, rgb, 64, 64, b, 768) != 0, "text-only rejects image");
    e2_free(txt);

    e2_ctx *full = e2_load(fullp, tokp);
    if (!full) { printf("SKIP full model: %s\n", e2_error()); return fails ? 1 : 0; }
    CHECK((e2_modalities(full) & E2_MOD_IMAGE) != 0, "full has vision");
    CHECK((e2_modalities(full) & E2_MOD_AUDIO) != 0, "full has audio");
    CHECK(e2_embed_image_rgb(full, rgb, 64, 64, a, 768) == 0, "image embed");
    CHECK(fabsf(norm(a, 768) - 1.0f) < 1e-4f, "image normalized");

    float pcm[16000];
    for (int i = 0; i < 16000; i++) pcm[i] = 0.2f * sinf(2.0f * 3.14159265f * 440.0f * i / 16000.0f);
    CHECK(e2_embed_audio(full, pcm, 16000, a, 768) == 0, "audio embed");
    CHECK(fabsf(norm(a, 768) - 1.0f) < 1e-4f, "audio normalized");

    short spcm[16000];
    for (int i = 0; i < 16000; i++) spcm[i] = (short)(pcm[i] * 32767.0f);
    write_wav16("/tmp/e2_test.wav", spcm, 16000);
    float *loaded = NULL; int ln = 0;
    if (e2_audio_read("/tmp/e2_test.wav", &loaded, &ln) == 0) {
        CHECK(ln == 16000, "audio_read length");
        CHECK(e2_embed_audio(full, loaded, ln, b, 768) == 0, "audio file embed");
        CHECK(fabsf(norm(b, 768) - 1.0f) < 1e-4f, "audio file normalized");
        free(loaded);
    } else {
        printf("note: libsndfile not enabled, skipping audio file test\n");
    }

#ifdef E2_USE_WEBP
    uint8_t *webp = NULL;
    size_t wsz = WebPEncodeRGB(rgb, 64, 64, 64 * 3, 80.0f, &webp);
    if (wsz) {
        FILE *wf = fopen("/tmp/e2_test.webp", "wb");
        fwrite(webp, 1, wsz, wf);
        fclose(wf);
        WebPFree(webp);
        CHECK(e2_embed_image(full, "/tmp/e2_test.webp", b, 768) == 0, "webp image embed");
        CHECK(fabsf(norm(b, 768) - 1.0f) < 1e-4f, "webp normalized");
    }
#endif
    e2_free(full);

    if (fails) { printf("%d check(s) failed\n", fails); return 1; }
    printf("all e2 tests passed\n");
    return 0;
}
