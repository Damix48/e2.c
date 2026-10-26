/* test_audio.c — run the audio tower on the frozen fixture, check the tokenizer layout, and compare
 * the final embeddings of text+audio and text+image+audio with the fp32 reference.
 *
 * Usage: e2_test_audio <model_dir> <fixture_dir>
 * Fixture format (written by tools/dump_audio_fixture.py):
 *   int32 n_samples, n_mel_frames, n_soft, dim, n_ids, soft_dim, n_img_soft, text_len
 *   char  text[text_len]                  the prompt, with one `<|audio|>` marker
 *   float pcm[n_samples]                  16 kHz mono
 *   int32 ids[n_ids]                      text+audio layout, as the processor builds it
 *   float mel[n_mel_frames * 128]         reference mel features
 *   uint8 mask[n_mel_frames]              reference frame mask
 *   float soft[n_soft * soft_dim]         reference audio soft tokens
 *   float ref_ta[dim]                     reference 768d of text+audio
 *   float ref_tia[dim]                    reference 768d of text+image+audio
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "e2.h"
#include "test_metrics.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int g_fail;

static void put_u32le(uint8_t *p, uint32_t v) {
    for (int b = 0; b < 4; b++) p[b] = (uint8_t)(v >> (8*b));
}
/* The existing FP32 embedding fixtures must also pass when PCM comes from file bytes. */
static int file_roundtrip(float *pcm, int n) {
    size_t bytes = (size_t)n * 4;
    uint8_t *wav = calloc(44 + bytes, 1);
    float *decoded = malloc(sizeof(float) * E2_AUDIO_MAX_SAMPLES);
    if (!wav || !decoded) { free(wav); free(decoded); return -1; }
    memcpy(wav, "RIFF", 4); put_u32le(wav + 4, (uint32_t)bytes + 36);
    memcpy(wav + 8, "WAVEfmt ", 8); put_u32le(wav + 16, 16);
    wav[20] = 3; wav[22] = 1; /* IEEE float, mono */
    put_u32le(wav + 24, E2_AUDIO_SAMPLE_RATE); put_u32le(wav + 28, E2_AUDIO_SAMPLE_RATE * 4);
    wav[32] = 4; wav[34] = 32;
    memcpy(wav + 36, "data", 4); put_u32le(wav + 40, (uint32_t)bytes);
    for (int i = 0; i < n; i++) { uint32_t v; memcpy(&v, pcm + i, 4); put_u32le(wav + 44 + 4*i, v); }
    int count = e2_audio_preprocess(wav, 44 + bytes, decoded);
    int ok = count == n && !memcmp(decoded, pcm, bytes);
    if (ok) memcpy(pcm, decoded, bytes);
    else fprintf(stderr, "audio file preprocessing: %s\n", e2_last_error());
    free(wav); free(decoded);
    return ok ? 0 : -1;
}

/* the int8 recipe costs ~3e-4 of cosine against the fp32 reference on every tower here (the text
 * document scores 0.99942, the image 0.99976), so the bar is the same 0.999 the other tests use */
/* per-stage activations are compared for localisation, not as the fidelity bar: the int8 tower loses
 * ~3e-3 of cosine per stage near the end and the text tower recovers it in the pooled embedding
 * (measured: l11 0.9971, soft 0.9962, final 768d 0.9995) */
static void score_stage(const char *what, const float *got, const float *ref, size_t n) {
    double dot = 0, na = 0, nb = 0, md = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)got[i] * ref[i];
        na += (double)got[i] * got[i];
        nb += (double)ref[i] * ref[i];
        const double d = fabs((double)got[i] - ref[i]);
        if (d > md) md = d;
    }
    const double cos = dot / sqrt(na * nb + 1e-30);
    printf("  %-20s cos=%.6f  max|d|=%.5f  %s\n", what, cos, md, cos > 0.99 ? "ok" : "DIVERGED");
    if (!(cos > 0.99)) g_fail = 1;
}

static void score(const char *what, const float *got, const float *ref, size_t n) {
    double dot = 0, na = 0, nb = 0, md = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)got[i] * ref[i];
        na += (double)got[i] * got[i];
        nb += (double)ref[i] * ref[i];
        const double d = fabs((double)got[i] - ref[i]);
        if (d > md) md = d;
    }
    const double cos = dot / sqrt(na * nb + 1e-30);
    printf("  %-20s cos=%.6f  max|d|=%.5f  %s\n", what, cos, md, cos > 0.999 ? "PASS" : "FAIL");
    if (!(cos > 0.999)) g_fail = 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <fixture_dir> [noimage]\n"
                        "  noimage: skip the text+image+audio case, for a cache without a vision tower\n",
                argv[0]);
        return 2;
    }
    const int with_image = !(argc > 3 && !strcmp(argv[3], "noimage"));
    char path[512];
    snprintf(path, sizeof path, "%s/audio.bin", argv[2]);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    int32_t hd[8] = { 0 };
    if (fread(hd, 4, 8, f) != 8) { fprintf(stderr, "bad fixture header\n"); return 2; }
    const int n_samples = hd[0], n_frames = hd[1], n_soft = hd[2], dim = hd[3], n_ids = hd[4],
              soft_dim = hd[5], n_img_soft = hd[6], text_len = hd[7];
    char *text = calloc((size_t)text_len + 1, 1);
    float *pcm = malloc(sizeof(float) * (size_t)n_samples);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)n_ids);
    float *mel = malloc(sizeof(float) * (size_t)n_frames * 128);
    uint8_t *mask = malloc((size_t)n_frames);
    float *soft = malloc(sizeof(float) * (size_t)n_soft * soft_dim);
    float *ref_ta = malloc(sizeof(float) * dim), *ref_tia = malloc(sizeof(float) * dim);
    int nvalid = 0;
    if (!text || !pcm || !ids || !mel || !mask || !soft || !ref_ta || !ref_tia) return 2;
    if (fread(text, 1, (size_t)text_len, f) != (size_t)text_len ||
        fread(pcm, 4, (size_t)n_samples, f) != (size_t)n_samples ||
        fread(ids, 4, (size_t)n_ids, f) != (size_t)n_ids ||
        fread(mel, 4, (size_t)n_frames * 128, f) != (size_t)n_frames * 128 ||
        fread(mask, 1, (size_t)n_frames, f) != (size_t)n_frames ||
        fread(soft, 4, (size_t)n_soft * soft_dim, f) != (size_t)n_soft * soft_dim ||
        fread(ref_ta, 4, (size_t)dim, f) != (size_t)dim ||
        fread(ref_tia, 4, (size_t)dim, f) != (size_t)dim) {
        fprintf(stderr, "truncated fixture\n");
        return 2;
    }
    fclose(f);
    if (file_roundtrip(pcm, n_samples)) return 1;
    for (int i = 0; i < n_frames; i++) nvalid += mask[i] != 0;
    printf("fixture: %d samples, %d mel frames (%d valid), %d soft tokens, %d ids: \"%s\"\n", n_samples,
           n_frames, nvalid, n_soft, n_ids, text);

    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "load failed: %s\n", e2_last_error()); return 1; }
    e2_info(m);

    const char *re = getenv("E2_REPS");
    int reps = re ? atoi(re) : 3;
    if (reps < 1) reps = 1;
    double *samples = malloc(sizeof *samples * (size_t)reps);
    if (!samples) return 2;
    float *out = malloc(sizeof(float) * dim);

    /* 0. the tokenizer: the markers a caller writes must expand to the reference's own id layout,
     *    for the audio-only prompt and for a prompt with both placeholders */
    e2_tokenizer *tok = e2_tokenizer_load(argv[1]);
    int32_t mixed[1024];
    int n_mixed = 0;
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", e2_last_error()); g_fail = 1; }
    else {
        char prompt[1024];
        int32_t got[1024];
        const int32_t n_audio = n_soft;
        snprintf(prompt, sizeof prompt, "%s<|audio|>", text);
        int n = e2_encode_mma(tok, prompt, NULL, 0, &n_audio, 1, got, 1024);
        if (n != n_ids || memcmp(got, ids, sizeof(int32_t) * (size_t)n_ids)) {
            printf("  tokenizer layout MISMATCH (n=%d, expected %d)\n", n, n_ids);
            g_fail = 1;
        } else {
            printf("  tokenizer layout      %d ids, text+audio matches the reference PASS\n", n);
        }
        /* text + image + audio in one prompt: <|image|> then <|audio|> */
        const int32_t n_img = n_img_soft;
        snprintf(prompt, sizeof prompt, "%s<|image|><|audio|>", text);
        n_mixed = e2_encode_mma(tok, prompt, &n_img, 1, &n_audio, 1, mixed, 1024);
        if (n_mixed < 0) { printf("  tokenizer mixed prompt: %s\n", e2_last_error()); g_fail = 1; }
        else {
            /* the same ids the manual splice below builds, so the reference embedding applies */
            /* one image block (2 wrapper ids + its soft slots) and the same audio block: the
             * fixture's text+audio ids plus that image block in the middle */
            const int ok = n_mixed == n_ids + 2 + n_img_soft;
            printf("  tokenizer layout      %d ids, text+image+audio %s\n", n_mixed,
                   ok ? "PASS" : "MISMATCH");
            if (!ok) g_fail = 1;
        }
    }

    /* 1. the audio tower on its own: <bos> <|audio> soft <audio|> <eos> */
    double dt = 1e18;
    int rc = 0;
    for (int r = 0; r < reps; r++) {
        const double t0 = now_ms();
        rc = e2_audio_embed(m, pcm, n_samples, out);
        const double d = now_ms() - t0;
        if (rc || !e2_test_unit("audio only", out, (size_t)dim)) return 1;
        samples[r] = d;
        if (d < dt) dt = d;
    }
    if (rc) { fprintf(stderr, "e2_audio_embed: %s\n", e2_last_error()); return 1; }
    printf("audio only          %8.2f ms (best of %d)\n", dt, reps);
    e2_test_timing(samples, reps);

    /* 2. text+audio through the announced ids */
    dt = 1e18;
    for (int r = 0; r < reps; r++) {
        const double t0 = now_ms();
        rc = e2_text_audio_embed(m, ids, n_ids, pcm, n_samples, out);
        const double d = now_ms() - t0;
        if (rc || !e2_test_unit("text+audio", out, (size_t)dim) ||
            !e2_test_match("text+audio", out, ref_ta, (size_t)dim, 0.999)) return 1;
        samples[r] = d;
        if (d < dt) dt = d;
    }
    if (rc) { fprintf(stderr, "e2_text_audio_embed: %s\n", e2_last_error()); return 1; }
    printf("text+audio          %8.2f ms (best of %d, %d ids)\n", dt, reps, n_ids);
    score("text+audio", out, ref_ta, dim);
    e2_test_timing(samples, reps);

    /* 3. text+image+audio: the image fixture of the image test, spliced before the audio */
    if (with_image) {
        char ipath[512];
        snprintf(ipath, sizeof ipath, "%s/image.bin", argv[2]);
        FILE *fi = fopen(ipath, "rb");
        if (!fi) {
            fprintf(stderr, "cannot open %s (needed for text+image+audio)\n", ipath);
            g_fail = 1;
        } else {
            int32_t ih[4];
            if (fread(ih, 4, 4, fi) != 4) return 2;
            float *pixels = malloc(sizeof(float) * (size_t)ih[0] * ih[1]);
            int32_t *ppos = malloc(sizeof(int32_t) * (size_t)ih[0] * ih[2]);
            float *iref = malloc(sizeof(float) * ih[3]);
            if (fread(pixels, 4, (size_t)ih[0] * ih[1], fi) != (size_t)ih[0] * ih[1] ||
                fread(ppos, 4, (size_t)ih[0] * ih[2], fi) != (size_t)ih[0] * ih[2] ||
                fread(iref, 4, (size_t)ih[3], fi) != (size_t)ih[3])
                return 2;
            fclose(fi);
            /* the ids the tokenizer built for "<prose><|image|><|audio|>" */
            if (n_mixed > 0) {
                const int32_t *ids2 = mixed;
                const int total = n_mixed;
                dt = 1e18;
                for (int r = 0; r < reps; r++) {
                    const double t0 = now_ms();
                    rc = e2_text_image_audio_embed(m, ids2, total, pixels, ppos, ih[0], pcm, n_samples, out);
                    const double d = now_ms() - t0;
                    if (rc || !e2_test_unit("text+image+audio", out, (size_t)dim) ||
                        !e2_test_match("text+image+audio", out, ref_tia, (size_t)dim, 0.999)) return 1;
                    samples[r] = d;
                    if (d < dt) dt = d;
                }
                if (rc) { fprintf(stderr, "e2_text_image_audio_embed: %s\n", e2_last_error()); g_fail = 1; }
                else {
                    printf("text+image+audio    %8.2f ms (best of %d, %d ids)\n", dt, reps, total);
                    score("text+image+audio", out, ref_tia, dim);
                    e2_test_timing(samples, reps);
                }
            } else {
                printf("text+image+audio    skipped (no mixed-prompt ids)\n");
            }
            free(pixels);
            free(ppos);
            free(iref);
        }
    }

    /* 4. stage bisection: E2_DUMP=<dir> writes the engine's stages, E2_REF_ACT=<ref_act dir> compares */
    const char *dump = getenv("E2_DUMP"), *act = getenv("E2_REF_ACT");
    if (dump) {
        char p2[512];
        float *buf = malloc(sizeof(float) * 750 * 4096);
        const char *names[] = { "mel", "conv", "l0", "l1", "l3", "l6", "l9", "l11", "out_proj", "soft" };
        const int widths[] = { 128, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1536, 512 };
        const int rows[] = { -1, 19, 19, 19, 19, 19, 19, 19, 19, 19 };
        printf("stages (%s%s):\n", dump, act ? "" : ", no E2_REF_ACT: only the sizes are checked");
        for (int i = 0; i < 10; i++) {
            snprintf(p2, sizeof p2, "%s/a_%s.f32", dump, names[i]);
            FILE *fo = fopen(p2, "rb");
            if (!fo) { printf("  %-10s (not written)\n", names[i]); continue; }
            /* "mel" holds the valid mel frames; everything else is S rows of its own width */
            const int r = rows[i] > 0 ? rows[i] : nvalid;
            const size_t n = (size_t)r * (size_t)widths[i];
            const size_t got = fread(buf, 4, n, fo);
            fclose(fo);
            if (got != n) { printf("  %-10s size mismatch (%zu of %zu)\n", names[i], got, n); g_fail = 1; continue; }
            if (!act) continue;
            snprintf(p2, sizeof p2, "%s/a_%s.f32", act, names[i]);
            fo = fopen(p2, "rb");
            if (!fo) { printf("  %-10s (no reference)\n", names[i]); continue; }
            float *ref = malloc(sizeof(float) * n);
            const size_t rn = fread(ref, 4, n, fo);
            fclose(fo);
            if (rn != n) { printf("  %-10s reference size mismatch\n", names[i]); g_fail = 1; }
            else score_stage(names[i], buf, ref, n);
            free(ref);
        }
        free(buf);
    }

    if (tok) e2_tokenizer_free(tok);
    free(out);
    free(samples);
    free(text); free(pcm); free(ids); free(mel); free(mask); free(soft); free(ref_ta); free(ref_tia);
    e2_free(m);
    printf("%s\n", g_fail ? "FAILED" : "OK");
    return g_fail;
}
