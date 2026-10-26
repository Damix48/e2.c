/* embed_cli — batch text -> embedding bridge for evaluation harnesses (MTEB).
 *
 * usage: embed_cli MODEL_DIR TEXTS.bin OUT.f32 OUT.ids
 *
 * TEXTS.bin is [uint32 len][len bytes] repeated (no NUL). OUT.f32 receives n_texts x 768 float32,
 * and OUT.ids receives, per text, [uint32 n_ids][n_ids int32] so a second engine can be run on
 * exactly the same token ids. Nothing is printed on success.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "e2.h"

static int rd(void *dst, size_t n, FILE *f) { return fread(dst, 1, n, f) == n; }

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s model_dir texts.bin out.f32 out.ids\n", argv[0]);
        return 2;
    }
    FILE *fi = fopen(argv[2], "rb");
    if (!fi) { perror(argv[2]); return 2; }
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) { perror(argv[3]); return 2; }
    FILE *fj = fopen(argv[4], "wb");
    if (!fj) { perror(argv[4]); return 2; }

    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }
    e2_tokenizer *tk = e2_tokenizer_load(argv[1]);
    if (!tk) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }

    float *emb = malloc(sizeof(float) * 768);
    if (!emb) return 1;
    for (;;) {
        uint32_t n = 0;
        size_t got = fread(&n, sizeof n, 1, fi);
        if (got != 1) break; /* clean EOF between records */
        char *text = malloc((size_t)n + 1);
        if (!text || !rd(text, n, fi)) { free(text); fprintf(stderr, "truncated record\n"); return 1; }
        text[n] = 0;
        int max_ids = 2 + 3 * (int)n;
        int32_t *ids = malloc(sizeof(int32_t) * (size_t)max_ids);
        if (!ids) return 1;
        int nid = e2_encode(tk, text, ids, max_ids);
        if (nid <= 0) { fprintf(stderr, "tokenize: %s\n", e2_last_error()); return 1; }
        if (nid > 2048) { /* the engine's sequence limit: keep the head and the closing token */
            ids[2047] = ids[nid - 1];
            nid = 2048;
        }
        if (e2_text_embed(m, ids, nid, emb)) { fprintf(stderr, "embed: %s\n", e2_last_error()); return 1; }
        if (fwrite(emb, sizeof(float), 768, fo) != 768) return 1;
        const uint32_t u = (uint32_t)nid;
        if (fwrite(&u, sizeof u, 1, fj) != 1) return 1;
        if (fwrite(ids, sizeof(int32_t), (size_t)nid, fj) != (size_t)nid) return 1;
        free(ids);
        free(text);
    }
    free(emb);
    e2_tokenizer_free(tk);
    e2_free(m);
    fclose(fi);
    fclose(fo);
    fclose(fj);
    return 0;
}
