/* e2_bench_full - latency benchmark driven by a tab-separated spec file.
 *
 * usage: e2_bench_full SPEC
 *
 * SPEC lines, "\t" separated:
 *   model      PATH
 *   tokenizer  PATH
 *   threads    N
 *   iters      N
 *   text       LABEL  TEXT
 *   batch      LABEL  TEXT1  TEXT2  ...
 *   image      LABEL  PATH
 *   audio      LABEL  PATH
 *   video      LABEL  FRAME1 FRAME2 ...
 *
 * Prints one "LABEL<TAB>best_ms<TAB>median_ms" line per measurement.
 * The spec is written by tools/bench_compare.py; run that instead.
 *
 * Image time includes PNG/JPEG decode, audio time excludes WAV/FLAC decode
 * (the tower takes raw PCM), matching how the two entry points are used.
 */
#include "e2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINES  4096
#define MAX_FIELDS 256

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct {
    e2_ctx *ctx;
    const char *text;
    const char *const *paths;
    int npaths;
    const float *pcm;
    int npcm;
} job;

static int op_text(void *p, float *out) {
    job *j = p;
    return e2_embed_text(j->ctx, j->text, out, E2_EMBED_DIM);
}

static int op_batch(void *p, float *out) {
    job *j = p;
    return e2_embed_text_batch(j->ctx, j->paths, j->npaths, out, E2_EMBED_DIM);
}

static int op_image(void *p, float *out) {
    job *j = p;
    return e2_embed_image(j->ctx, j->paths[0], out, E2_EMBED_DIM);
}

static int op_audio(void *p, float *out) {
    job *j = p;
    return e2_embed_audio(j->ctx, j->pcm, j->npcm, out, E2_EMBED_DIM);
}

static int op_video(void *p, float *out) {
    job *j = p;
    return e2_embed_video(j->ctx, (const char *const *)j->paths, j->npaths, out,
                          E2_EMBED_DIM);
}

/* 2 warm-up runs, then `iters` timed runs. Writes best and median. */
static int measure(const char *label, int (*fn)(void *, float *), job *j,
                   int iters, int nout) {
    if (iters < 1) iters = 1;
    float *out = malloc(sizeof(float) * E2_EMBED_DIM * (size_t)nout);
    double *t = malloc(sizeof(double) * (size_t)iters);
    if (!out || !t) {
        printf("%s\terr\tout of memory\n", label);
        free(out);
        free(t);
        return -1;
    }
    for (int i = 0; i < 2; i++)
        if (fn(j, out)) goto fail;
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        if (fn(j, out)) goto fail;
        t[i] = now_ms() - t0;
    }
    qsort(t, iters, sizeof *t, cmp_double);
    printf("%s\t%.3f\t%.3f\n", label, t[0], t[iters / 2]);
    free(out);
    free(t);
    return 0;
fail:
    printf("%s\terr\t%s\n", label, e2_error());
    free(out);
    free(t);
    return -1;
}

static int split(char *s, char **v, int max) {
    int n = 0;
    for (char *tok = strtok(s, "\t"); tok; tok = strtok(NULL, "\t")) {
        if (n == max) break;
        v[n++] = tok;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s SPEC\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    static char *lines[MAX_LINES];
    int nlines = 0;
    char *line = NULL;
    size_t cap = 0;
    while (nlines < MAX_LINES && getline(&line, &cap, f) > 0) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n == 0 || line[0] == '#') continue;
        lines[nlines++] = strdup(line);
    }
    free(line);
    fclose(f);

    const char *model = NULL, *tokpath = NULL;
    int threads = 0, iters = 3;
    char *v[MAX_FIELDS];
    for (int i = 0; i < nlines; i++) {
        char *copy = strdup(lines[i]);
        int nf = split(copy, v, MAX_FIELDS);
        if (nf >= 2 && !strcmp(v[0], "model")) model = strdup(v[1]);
        else if (nf >= 2 && !strcmp(v[0], "tokenizer")) tokpath = strdup(v[1]);
        else if (nf >= 2 && !strcmp(v[0], "threads")) threads = atoi(v[1]);
        else if (nf >= 2 && !strcmp(v[0], "iters")) iters = atoi(v[1]);
        free(copy);
    }
    if (!model || !tokpath) {
        fprintf(stderr, "spec needs model and tokenizer lines\n");
        return 2;
    }
    e2_ctx *ctx = e2_load(model, tokpath);
    if (!ctx) {
        fprintf(stderr, "%s\n", e2_error());
        return 1;
    }
    if (threads > 0) e2_set_threads(ctx, threads);
    printf("threads\t%d\n", e2_threads(ctx));

    for (int i = 0; i < nlines; i++) {
        char *copy = strdup(lines[i]);
        int nf = split(copy, v, MAX_FIELDS);
        job j = {ctx, NULL, NULL, 0, NULL, 0};
        if (nf >= 3 && !strcmp(v[0], "text")) {
            j.text = v[2];
            measure(v[1], op_text, &j, iters, 1);
        } else if (nf >= 3 && !strcmp(v[0], "batch")) {
            j.paths = (const char *const *)&v[2];
            j.npaths = nf - 2;
            measure(v[1], op_batch, &j, iters, j.npaths);
        } else if (nf >= 3 && !strcmp(v[0], "image")) {
            j.paths = (const char *const *)&v[2];
            j.npaths = 1;
            measure(v[1], op_image, &j, iters, 1);
        } else if (nf >= 3 && !strcmp(v[0], "audio")) {
            float *pcm = NULL;
            int npcm = 0;
            if (e2_audio_read(v[2], &pcm, &npcm) != 0) {
                printf("%s\terr\t%s\n", v[1], e2_error());
            } else {
                j.pcm = pcm;
                j.npcm = npcm;
                measure(v[1], op_audio, &j, iters, 1);
                free(pcm);
            }
        } else if (nf >= 3 && !strcmp(v[0], "video")) {
            j.paths = (const char *const *)&v[2];
            j.npaths = nf - 2;
            measure(v[1], op_video, &j, iters, 1);
        }
        free(copy);
    }

    for (int i = 0; i < nlines; i++) free(lines[i]);
    e2_free(ctx);
    return 0;
}
