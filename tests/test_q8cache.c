/* test_q8cache.c — the packed int8 cache round-trips and refuses a corrupt or incompatible file
 * before anything dereferences it. Self-contained: it writes its own small cache into a temp dir, so
 * it needs no model, no fixtures and no arguments. Usage: e2_test_q8cache */
#include "test_tmpdir.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "model.h" /* e2_q8w_load: the lazy loader's retry unit */
#include "q8cache.h"

#define NAME_Q8W "test.q8w"
#define NAME_BF16 "test.bf16"
#define NAME_SCALAR "test.scalar" /* 0-d: the audio clamp bounds are stored like this */
#define TEST_N 16
#define TEST_K 96 /* not a multiple of E2_I8_BLOCK: nblocks must round up to 1 */

static int g_fail;
#define CHECK(cond, ...)                              \
    do {                                              \
        if (!(cond)) {                                \
            printf("FAIL line %d: ", __LINE__);       \
            printf(__VA_ARGS__);                      \
            printf("\n");                             \
            g_fail++;                                 \
        }                                             \
    } while (0)

/* entry layout of the on-disk index (see q8c_entry in q8cache.c) */
enum { E_NAME = 0, E_KIND = 96, E_NDIM = 100, E_DIMS = 104, E_NBYTES = 136, E_OFF = 144, E_SIZE = 152 };

static void make_q8w(e2_q8w *q) {
    float *w = malloc(sizeof(float) * TEST_N * TEST_K);
    for (int n = 0; n < TEST_N; n++)
        for (int k = 0; k < TEST_K; k++) w[(size_t)n * TEST_K + k] = (float)((n * 37 + k * 11) % 101 - 50) * 0.02f;
    if (e2_q8w_from_f32(q, w, TEST_N, TEST_K)) g_fail++;
    free(w);
}

static int write_valid(const char *path, const e2_q8w *q) {
    static const uint16_t bf16[3] = { 0x3f80, 0x4000, 0xc000 };
    static const uint16_t scalar = 0x3f80;
    const int64_t d[1] = { 3 };
    if (q8c_write_open(path)) return -1;
    /* the loader's own sequence for a quantized weight: the bf16 copy it fetched is deferred, then
     * dropped and replaced by the packed one under the same name (no duplicate record) */
    if (q8c_defer(NAME_Q8W, Q8C_BF16, 2, (const int64_t[2]){ TEST_N, TEST_K }, bf16, sizeof bf16)) return -1;
    if (q8c_drop(NAME_Q8W)) return -1;
    if (q8c_put_q8w(NAME_Q8W, q)) return -1;
    if (q8c_defer(NAME_BF16, Q8C_BF16, 1, d, bf16, sizeof bf16)) return -1;
    if (q8c_defer(NAME_SCALAR, Q8C_BF16, 0, NULL, &scalar, sizeof scalar)) return -1;
    return q8c_write_close();
}

static uint32_t entry_count(const char *path) {
    uint32_t hdr[4] = { 0, 0, 0, 0 };
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    if (fread(hdr, sizeof hdr, 1, f) != 1) hdr[2] = 0xffffffffu;
    fclose(f);
    return hdr[2];
}

static void check_roundtrip(const char *path, const e2_q8w *src) {
    e2_q8w q;
    memset(&q, 0, sizeof q);
    CHECK(q8c_read_open(path) == 0, "valid cache rejected: %s", q8c_error());
    CHECK(q8c_get_q8w(NAME_Q8W, &q) == 0, "get_q8w: %s", q8c_error());
    CHECK(q.N == TEST_N && q.K == TEST_K, "dims %dx%d", q.N, q.K);
    CHECK(q.nblocks == 1 && q.Npad == ((TEST_N + 7) & ~7), "nblocks %d Npad %d", q.nblocks, q.Npad);
    CHECK(q.borrowed == 1, "not marked borrowed");
    const size_t nw = (size_t)src->Npad * ((TEST_K + 3) / 4) * 4;
    CHECK(!memcmp(q.w, src->w, nw), "weights differ");
    CHECK(!memcmp(q.scale, src->scale, sizeof(float) * (size_t)src->Npad), "scales differ");
    CHECK(!memcmp(q.bsq, src->bsq, sizeof(int32_t) * (size_t)src->nblocks * (size_t)src->Npad),
          "block sums differ");
    const int64_t d[1] = { 3 };
    size_t nb = 0;
    CHECK(q8c_get(NAME_BF16, Q8C_BF16, 1, d, &nb) != NULL && nb == 6, "bf16 blob");
    CHECK(q8c_get(NAME_SCALAR, Q8C_BF16, 0, NULL, &nb) != NULL && nb == 2, "0-d bf16 scalar");
    CHECK(q8c_kind_of(NAME_Q8W) == Q8C_Q8W && q8c_kind_of("nope") == -1, "kind_of");
    CHECK(q8c_get("nope", Q8C_BF16, 1, d, NULL) == NULL, "missing tensor accepted");
    q8c_read_close();
}

/* e2_q8w_load is what the lazy audio path retries as a whole after a failure: a tensor that is
 * already built must not be quantized again (no leak, no second cache record), and a freed one must
 * be rebuilt rather than mistaken for complete. */
static void check_load_idempotent(void) {
    uint16_t *w = malloc(sizeof(uint16_t) * TEST_N * TEST_K);
    for (size_t i = 0; i < (size_t)TEST_N * TEST_K; i++) w[i] = (uint16_t)(0x3c00 + (i * 7) % 64);
    e2_q8w ref, q;
    memset(&ref, 0, sizeof ref);
    memset(&q, 0, sizeof q);
    CHECK(e2_q8w_from_bf16(&ref, w, TEST_N, TEST_K) == 0, "reference quantize failed");
    CHECK(e2_q8w_load(NULL, &q, "idem.q8w", w, TEST_N, TEST_K) == 0, "first load: %s", e2_last_error());
    const int8_t *first = q.w;
    const size_t nw = (size_t)ref.Npad * ((TEST_K + 3) / 4) * 4;
    CHECK(first != NULL && q.N == TEST_N && q.K == TEST_K, "first load produced nothing");
    CHECK(first && !memcmp(q.w, ref.w, nw), "first load quantized wrong");
    CHECK(e2_q8w_load(NULL, &q, "idem.q8w", w, TEST_N, TEST_K) == 0, "second load failed");
    CHECK(q.w == first, "an already built tensor was quantized again");
    e2_q8w_free(&q);
    CHECK(e2_q8w_load(NULL, &q, "idem.q8w", w, TEST_N, TEST_K) == 0, "load after free failed");
    CHECK(q.w != NULL && q.N == TEST_N, "a freed tensor was treated as complete");
    e2_q8w_free(&q);
    e2_q8w_free(&ref);
    free(w);
}

/* ---- corruption: one patched field per case, all of them must be refused by q8c_read_open */
enum {
    C_MAGIC, C_VERSION, C_SHORT, C_MANY_ENTRIES, C_IDX_MISALIGNED, C_IDX_IN_HEADER, C_IDX_AT_EOF,
    C_NAME_UNTERMINATED, C_NAME_EMPTY, C_KIND, C_NDIM, C_NEG_DIM, C_OFF_MISALIGNED, C_OFF_OVERFLOW,
    C_OFF_IN_INDEX, C_NBLOCKS, C_NPAD, C_NBYTES, C_BF16_SIZE, C_BF16_OVERFLOW, C_SCALAR_SIZE, C_COUNT
};
static const char *const g_case_name[C_COUNT] = {
        "bad magic", "bad version", "truncated file", "huge n_entries", "index offset misaligned",
        "index offset in header", "index offset at eof", "name unterminated", "name empty",
        "unknown kind", "bad rank", "negative dim", "blob offset misaligned", "blob overflow",
        "blob inside the index", "nblocks != E2_I8_BLOCK packing", "Npad != round_up(N,8)",
        "blob byte size mismatch", "bf16 blob size != element count", "bf16 element count overflows",
        "0-d bf16 blob is not one element",
};

static void put(const uint8_t *b, size_t at, const void *v, size_t n) { memcpy((void *)(b + at), v, n); }

static uint8_t *slurp(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f);
    *out_n = (size_t)n;
    return b;
}

static void patch(uint8_t *b, size_t size, int which) {
    const uint32_t idx = *(const uint32_t *)(b + 12);
    uint8_t *e0 = b + idx;
    const uint32_t u32_0 = 0, u32_2 = 2, u32_3 = 3, u32_big = 0xffffffffu;
    const int64_t neg = -8, two = 2, npad24 = 24;
    const uint64_t huge = (uint64_t)-16, off_big = (uint64_t)idx;
    switch (which) {
    case C_MAGIC: put(b, 0, &u32_0, 4); break;
    case C_VERSION: put(b, 4, &u32_2, 4); break;
    case C_SHORT: break; /* the caller truncates the file instead */
    case C_MANY_ENTRIES: put(b, 8, &u32_big, 4); break;
    case C_IDX_MISALIGNED: { const uint32_t v = idx + 8; put(b, 12, &v, 4); break; }
    case C_IDX_IN_HEADER: { const uint32_t v = 8; put(b, 12, &v, 4); break; }
    case C_IDX_AT_EOF: { const uint32_t v = (uint32_t)size; put(b, 12, &v, 4); break; }
    case C_NAME_UNTERMINATED: memset(e0 + E_NAME, 'A', E_KIND - E_NAME); break;
    case C_NAME_EMPTY: e0[E_NAME] = 0; break;
    case C_KIND: put(e0, E_KIND, &u32_0, 4); break;
    case C_NDIM: put(e0, E_NDIM, &u32_3, 4); break;
    case C_NEG_DIM: put(e0, E_DIMS, &neg, 8); break;
    case C_OFF_MISALIGNED: { const uint64_t v = E_OFF + 8; put(e0, E_OFF, &v, 8); break; }
    case C_OFF_OVERFLOW: put(e0, E_OFF, &huge, 8); break;
    case C_OFF_IN_INDEX: put(e0, E_OFF, &off_big, 8); break;
    case C_NBLOCKS: put(e0, E_DIMS + 16, &two, 8); break;
    case C_NPAD: put(e0, E_DIMS + 24, &npad24, 8); break;
    case C_NBYTES: { uint64_t v = *(const uint64_t *)(e0 + E_NBYTES); v += 4; put(e0, E_NBYTES, &v, 8); break; }
    case C_BF16_SIZE: { uint8_t *e1 = b + idx + E_SIZE; const uint64_t v = 4; put(e1, E_NBYTES, &v, 8); break; }
    case C_BF16_OVERFLOW: { /* dims {INT64_MAX, 2}: the element count cannot be represented */
        uint8_t *e1 = b + idx + E_SIZE;
        const int64_t big[2] = { INT64_MAX, 2 };
        const uint32_t nd2 = 2;
        put(e1, E_NDIM, &nd2, 4);
        put(e1, E_DIMS, big, sizeof big);
        break;
    }
    case C_SCALAR_SIZE: { uint8_t *e2 = b + idx + 2 * E_SIZE; const uint64_t v = 4; put(e2, E_NBYTES, &v, 8); break; }
    default: break;
    }
}

int main(void) {
    char dir[512];
    snprintf(dir, sizeof dir, "%s/e2q8cXXXXXX", e2_test_tmpdir());
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 2; }
    char valid[544], bad[544]; /* dir is 512: leave room for the appended name */
    snprintf(valid, sizeof valid, "%s/valid.e8cq", dir);
    snprintf(bad, sizeof bad, "%s/bad.e8cq", dir);

    e2_q8w src;
    memset(&src, 0, sizeof src);
    make_q8w(&src);

    /* ---- the writer refuses a second session, and a failed close still releases its state */
    CHECK(write_valid(valid, &src) == 0, "write: %s", q8c_error());
    CHECK(q8c_write_open(bad) == 0, "second writer open");
    CHECK(q8c_write_open(bad) == -1, "two writers accepted");
    CHECK(q8c_write_close() == 0, "close after the refused open");
    CHECK(q8c_write_close() == 0, "double close is not a no-op");
    unlink(bad);

    /* Aborting a replacement must preserve the old artifact, including for mapped readers. */
    CHECK(q8c_read_open(valid) == 0, "map before replacement");
    CHECK(q8c_write_open(valid) == 0, "replacement writer");
    CHECK(q8c_put_q8w(NAME_Q8W, &src) == 0, "partial replacement");
    q8c_write_abort();
    CHECK(q8c_kind_of(NAME_BF16) == Q8C_BF16, "live mapping changed");
    q8c_read_close();
    check_roundtrip(valid, &src);

    /* ---- one record per tensor: a repeated name is refused and the file keeps exactly one */
    snprintf(bad, sizeof bad, "%s/dup.e8cq", dir);
    CHECK(q8c_write_open(bad) == 0, "writer open for the duplicate case");
    CHECK(q8c_put_q8w(NAME_Q8W, &src) == 0, "first record: %s", q8c_error());
    CHECK(q8c_put_q8w(NAME_Q8W, &src) == -1, "duplicate tensor accepted");
    printf("  refused: %-34s %s\n", "duplicate tensor record", q8c_error());
    CHECK(q8c_write_close() == 0, "close after a refused duplicate");
    CHECK(entry_count(bad) == 1, "duplicate record written (%u entries)", entry_count(bad));
    CHECK(q8c_read_open(bad) == 0, "cache after a refused duplicate: %s", q8c_error());
    e2_q8w dupq;
    memset(&dupq, 0, sizeof dupq);
    CHECK(q8c_get_q8w(NAME_Q8W, &dupq) == 0, "record after a refused duplicate");
    q8c_read_close();
    unlink(bad);

    /* ---- the loader's idempotency, then a valid cache round-trips */
    check_load_idempotent();
    check_roundtrip(valid, &src);

    /* ---- every corruption is refused, and does not poison the next open */
    size_t size = 0;
    uint8_t *img = slurp(valid, &size);
    CHECK(img != NULL && size > E_SIZE, "read back the cache");
    for (int c = 0; c < C_COUNT; c++) {
        uint8_t *b = malloc(size);
        memcpy(b, img, size);
        patch(b, size, c);
        const size_t n = (c == C_SHORT) ? 8 : size;
        FILE *f = fopen(bad, "wb");
        if (!f || fwrite(b, 1, n, f) != n) { perror("write bad cache"); free(b); g_fail++; break; }
        fclose(f);
        free(b);
        const int rc = q8c_read_open(bad);
        CHECK(rc == -1, "%s: accepted", g_case_name[c]);
        printf("  refused: %-34s %s\n", g_case_name[c], q8c_error());
        /* the refused open must have closed everything it opened */
        check_roundtrip(valid, &src);
    }
    free(img);

    e2_q8w_free(&src);
    unlink(valid);
    rmdir(dir);
    printf("%s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
