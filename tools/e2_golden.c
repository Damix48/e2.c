/* e2_golden - reference embeddings for the C# binding tests.
 *
 * The inputs are synthetic and byte-reproducible by any language (no trig, no
 * files), so the C# tests rebuild the same buffers and compare embeddings.
 *
 * usage: e2_golden <model> <tokenizer> <out.txt> [dim]
 */
#include "e2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 128
#define H 128
#define NSAMP 64000 /* 4 s at 16 kHz */

static const char *TEXTS[] = {
    "hello world",
    "a brown fox jumps over the lazy dog",
    "caff\xc3\xa8 \xe2\x98\x95 unicode",
};

static void emit(FILE *f, const char *name, const float *v, int dim) {
    fprintf(f, "%s %d\n", name, dim);
    for (int i = 0; i < dim; i++) fprintf(f, "%.9g%c", v[i], i + 1 == dim ? '\n' : ' ');
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model tokenizer out.txt [dim]\n", argv[0]);
        return 2;
    }
    int dim = argc > 4 ? atoi(argv[4]) : 768;

    e2_ctx *ctx = e2_load(argv[1], argv[2]);
    if (!ctx) { fprintf(stderr, "load: %s\n", e2_error()); return 1; }
    int mods = e2_modalities(ctx);

    FILE *f = fopen(argv[3], "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }

    float emb[E2_EMBED_DIM];
    char name[256];

    for (size_t i = 0; i < sizeof(TEXTS) / sizeof(TEXTS[0]); i++) {
        if (e2_embed_text(ctx, TEXTS[i], emb, dim)) {
            fprintf(stderr, "text: %s\n", e2_error());
            return 1;
        }
        snprintf(name, sizeof(name), "text_default:%s", TEXTS[i]);
        emit(f, name, emb, dim);
    }

    if (e2_embed_text_task(ctx, "search_query", "what is e2", emb, dim)) {
        fprintf(stderr, "task: %s\n", e2_error());
        return 1;
    }
    emit(f, "task:search_query:what is e2", emb, dim);

    if (mods & E2_MOD_IMAGE) {
        unsigned char *rgb = malloc((size_t)W * H * 3);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                unsigned char *p = rgb + ((size_t)y * W + x) * 3;
                p[0] = (unsigned char)((x * 7 + y * 13) & 255);
                p[1] = (unsigned char)((x * 11 + y * 3) & 255);
                p[2] = (unsigned char)((x * 5 + y * 17) & 255);
            }
        if (e2_embed_image_rgb(ctx, rgb, W, H, emb, dim)) {
            fprintf(stderr, "image: %s\n", e2_error());
            return 1;
        }
        free(rgb);
        emit(f, "image:rgb128x128", emb, dim);
    }

    if (mods & E2_MOD_AUDIO) {
        float *pcm = malloc((size_t)NSAMP * sizeof(float));
        /* / 512, not / 400: a power of two keeps the sample exactly
         * representable, so a host built with -ffast-math (x*0.0025f instead of
         * x/400.0f) produces the same buffer as any other language. */
        for (int i = 0; i < NSAMP; i++) pcm[i] = (float)((i % 200) - 100) / 512.0f;
        if (e2_embed_audio(ctx, pcm, NSAMP, emb, dim)) {
            fprintf(stderr, "audio: %s\n", e2_error());
            return 1;
        }
        free(pcm);
        emit(f, "audio:square4s", emb, dim);
    }

    fclose(f);
    e2_free(ctx);
    return 0;
}
