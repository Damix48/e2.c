/* A real exported cache with a missing/raw wrong-shaped tensor must fail at load, not inference. */
#include "e2.h"
#include "kernels.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static long entry(const char *path, const char *name) {
    FILE *f = fopen(path, "rb"); assert(f);
    uint32_t h[4]; assert(fread(h, sizeof h, 1, f) == 1);
    assert(!fseek(f, h[3], SEEK_SET));
    for (uint32_t i = 0; i < h[2]; i++) {
        unsigned char e[152]; assert(fread(e, sizeof e, 1, f) == 1);
        if (!strcmp((char *)e, name)) { fclose(f); return (long)h[3] + i * 152; }
    }
    fclose(f); assert(!"tensor not found"); return -1;
}
static void patch(const char *path, long at, const void *p, size_t n) {
    FILE *f = fopen(path, "r+b"); assert(f);
    assert(!fseek(f, at, SEEK_SET) && fwrite(p, n, 1, f) == 1); assert(!fclose(f));
}
int main(int argc, char **argv) {
    assert(argc == 2);
    e2_cpu_init();
    if (!e2_i8_quantize) { puts("SKIP: packed caches require an int8 backend"); return 77; }
    char dir[] = "/tmp/e2-model-load-XXXXXX"; assert(mkdtemp(dir));
    char cache[512]; snprintf(cache, sizeof cache, "%s/model.e8cq", dir);
    unsetenv("E2_INT8"); unsetenv("E2_Q8_READ");
    assert(!setenv("E2_Q8_WRITE", cache, 1) && !setenv("E2_Q8_TOWERS", "all", 1));
    e2_model *m = e2_load(argv[1]);
    if (!m) { fprintf(stderr, "%s\n", e2_last_error()); return 1; }
    e2_free(m); unsetenv("E2_Q8_WRITE"); unsetenv("E2_Q8_TOWERS");
    assert(!setenv("E2_Q8_READ", cache, 1));
    const char *names[] = { "language_model.embed_tokens.weight",
        "vision_tower.patch_embedder.position_embedding_table",
        "audio_tower.layers.11.feed_forward2.ffw_layer_2.linear.weight" };
    for (int i = 0; i < 3; i++) {
        long at = entry(cache, names[i]); char x = 'X';
        patch(cache, at, &x, 1);
        assert(e2_load(argv[1]) == NULL && e2_last_error()[0]);
        patch(cache, at, names[i], 1);
        if (i < 2) {
            /* Preserve the byte count: structural cache validation alone cannot catch this. */
            int64_t bad[2] = { i == 0 ? 131072 : 1, i == 0 ? 1024 : 20480 };
            patch(cache, at + 104, bad, sizeof bad);
            assert(e2_load(argv[1]) == NULL && e2_last_error()[0]);
            int64_t good[2] = { i == 0 ? 262144 : 2, i == 0 ? 512 : 10240 };
            patch(cache, at + 104, good, sizeof good);
        }
    }
    m = e2_load(argv[1]); assert(m && !e2_last_error()[0]);
    float out[768]; int32_t ids[] = {2,1234,1};
    assert(!e2_text_embed(m, ids, 3, out));
    e2_free(m); unlink(cache); rmdir(dir);
    puts("missing/wrong-shaped raw and lazy audio cache tensors refused at load OK");
    return 0;
}
