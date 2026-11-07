/* e2cli - small command line front-end for libe2. */
#include "e2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_wav(const char *path, int *n_out, int *sr_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fclose(f); return NULL;
    }
    int format = 0, channels = 1, sr = 16000, bits = 16;
    float *out = NULL;
    long total = 0;
    for (;;) {
        unsigned char ck[8];
        if (fread(ck, 1, 8, f) != 8) break;
        unsigned sz = ck[4] | (ck[5] << 8) | (ck[6] << 16) | ((unsigned)ck[7] << 24);
        if (!memcmp(ck, "fmt ", 4)) {
            unsigned char b[16];
            if (sz < 16 || fread(b, 1, 16, f) != 16) break;
            format = b[0] | (b[1] << 8);
            channels = b[2] | (b[3] << 8);
            sr = b[4] | (b[5] << 8) | (b[6] << 16) | ((unsigned)b[7] << 24);
            bits = b[14] | (b[15] << 8);
            if (sz > 16) fseek(f, (long)sz - 16, SEEK_CUR);
        } else if (!memcmp(ck, "data", 4)) {
            int bytes = bits / 8;
            long frames = sz / (bytes * (channels ? channels : 1));
            out = malloc((size_t)frames * sizeof(float));
            if (!out) break;
            for (long i = 0; i < frames; i++) {
                float acc = 0;
                for (int c = 0; c < channels; c++) {
                    if (format == 3 && bits == 32) {
                        float v; if (fread(&v, 4, 1, f) != 1) v = 0;
                        acc += v;
                    } else if (bits == 16) {
                        short v; if (fread(&v, 2, 1, f) != 1) v = 0;
                        acc += v / 32768.0f;
                    } else if (bits == 8) {
                        unsigned char v; if (fread(&v, 1, 1, f) != 1) v = 128;
                        acc += (v - 128) / 128.0f;
                    } else {
                        unsigned char v[4] = {0};
                        fread(v, 1, 4, f);
                        acc += 0;
                    }
                }
                out[i] = acc / (channels ? channels : 1);
                total = i + 1;
            }
            break;
        } else {
            fseek(f, (long)sz, SEEK_CUR);
        }
        if (sz & 1) fseek(f, 1, SEEK_CUR);
    }
    fclose(f);
    if (!out) return NULL;
    *n_out = (int)total;
    *sr_out = sr;
    return out;
}

static void print_emb(const float *e, int dim, double ms) {
    printf("dim=%d time=%.2f ms\n", dim, ms);
    for (int i = 0; i < dim; i++) printf("%.8f%c", e[i], (i % 8 == 7) ? '\n' : ' ');
    if (dim % 8) printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage:\n"
            "  %s text  MODEL TOKENIZER \"text\" [threads] [dim]\n"
            "  %s batch MODEL TOKENIZER FILE_WITH_ONE_TEXT_PER_LINE [threads] [dim]\n"
            "  %s image MODEL TOKENIZER FILE.png|.jpg\n"
            "  %s audio MODEL TOKENIZER FILE.wav\n"
            "  %s video MODEL TOKENIZER FRAME1 FRAME2 ...\n"
            "  %s mm    MODEL TOKENIZER \"text with <|image|>/<|audio|>/<|video|>\" ...\n",
            argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    const char *cmd = argv[1];
    e2_ctx *ctx = e2_load(argv[2], argv[3]);
    if (!ctx) { fprintf(stderr, "load: %s\n", e2_error()); return 1; }

    int argi = 4, dim = 768, nth = 0;
    if (!strcmp(cmd, "text")) {
        if (argc < 5) { fprintf(stderr, "text missing\n"); return 2; }
        if (argc > 5) nth = atoi(argv[5]);
        if (argc > 6) dim = atoi(argv[6]);
        if (nth) e2_set_threads(ctx, nth);
        float *e = malloc((size_t)dim * sizeof(float));
        if (e2_embed_text(ctx, argv[4], e, dim)) { fprintf(stderr, "%s\n", e2_error()); return 1; }
        print_emb(e, dim, e2_last_ms(ctx));
        free(e);
    } else if (!strcmp(cmd, "batch")) {
        if (argc < 5) { fprintf(stderr, "batch missing\n"); return 2; }
        if (argc > 5) nth = atoi(argv[5]);
        if (argc > 6) dim = atoi(argv[6]);
        if (nth) e2_set_threads(ctx, nth);
        FILE *f = fopen(argv[4], "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[4]); return 1; }
        char **lines = NULL; int n = 0, cap = 0; char buf[65536];
        while (fgets(buf, sizeof(buf), f)) {
            size_t l = strlen(buf);
            while (l && (buf[l-1] == '\n' || buf[l-1] == '\r')) buf[--l] = 0;
            if (!l) continue;
            if (n == cap) { cap = cap ? cap * 2 : 64; lines = realloc(lines, (size_t)cap * sizeof(char *)); }
            lines[n++] = strdup(buf);
        }
        fclose(f);
        float *e = malloc((size_t)n * dim * sizeof(float));
        if (e2_embed_text_batch(ctx, (const char *const *)lines, n, e, dim)) {
            fprintf(stderr, "%s\n", e2_error()); return 1;
        }
        printf("n=%d dim=%d time=%.2f ms (%.3f ms/text)\n", n, dim,
               e2_last_ms(ctx), e2_last_ms(ctx) / (n ? n : 1));
        for (int i = 0; i < n; i++) {
            printf("%s", lines[i]);
            for (int j = 0; j < 3; j++) printf(" %.6f", e[(size_t)i * dim + j]);
            printf("\n");
            free(lines[i]);
        }
        free(lines); free(e);
    } else if (!strcmp(cmd, "image")) {
        if (argc < 5) return 2;
        float e[768];
        if (e2_embed_image(ctx, argv[4], e, 768)) { fprintf(stderr, "%s\n", e2_error()); return 1; }
        print_emb(e, 768, e2_last_ms(ctx));
    } else if (!strcmp(cmd, "audio")) {
        if (argc < 5) return 2;
        int n = 0;
        float *s = NULL;
        if (e2_audio_read(argv[4], &s, &n) != 0) {
            int sr = 0;
            s = read_wav(argv[4], &n, &sr);
            if (s && sr != 16000) fprintf(stderr, "warning: sample rate %d, expected 16000\n", sr);
        }
        if (!s) { fprintf(stderr, "cannot read audio %s\n", argv[4]); return 1; }
        float e[768];
        if (e2_embed_audio(ctx, s, n, e, 768)) { fprintf(stderr, "%s\n", e2_error()); return 1; }
        print_emb(e, 768, e2_last_ms(ctx));
        free(s);
    } else if (!strcmp(cmd, "video")) {
        if (argc < 6) return 2;
        float e[768];
        if (e2_embed_video(ctx, (const char *const *)&argv[4], argc - 4, e, 768)) {
            fprintf(stderr, "%s\n", e2_error()); return 1;
        }
        print_emb(e, 768, e2_last_ms(ctx));
    } else if (!strcmp(cmd, "mm")) {
        if (argc < 5) return 2;
        /* remaining args: images, then videos (comma separated frames), then audio wav */
        const char *text = argv[argi++];
        const char *images[16]; int ni = 0;
        const char *audio_paths[8]; int na = 0;
        while (argi < argc) {
            if (strstr(argv[argi], ".wav")) audio_paths[na++] = argv[argi++];
            else images[ni++] = argv[argi++];
        }
        float *aud[8]; int alen[8] = {0};
        for (int i = 0; i < na; i++) {
            if (e2_audio_read(audio_paths[i], &aud[i], &alen[i]) != 0) {
                int sr = 0;
                aud[i] = read_wav(audio_paths[i], &alen[i], &sr);
            }
            if (!aud[i]) { fprintf(stderr, "bad audio\n"); return 1; }
        }
        e2_input in = {0};
        in.text = text;
        in.images = images; in.n_images = ni;
        in.audio = (const float *const *)aud; in.audio_len = alen; in.n_audio = na;
        float e[768];
        if (e2_embed(ctx, &in, e, 768)) { fprintf(stderr, "%s\n", e2_error()); return 1; }
        print_emb(e, 768, e2_last_ms(ctx));
        for (int i = 0; i < na; i++) free(aud[i]);
    } else {
        fprintf(stderr, "unknown command %s\n", cmd);
        return 2;
    }
    e2_free(ctx);
    return 0;
}
