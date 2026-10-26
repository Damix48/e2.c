/* tokenizer.c — EmbeddingGemma 2 text tokenization, loaded from <model_dir>/tokenizer.json.
 *
 * The pipeline is this tokenizer's, hardcoded because it is a property of the shipped file:
 *
 *   1. added/special tokens are split out of the RAW text (longest match at each position wins);
 *      each becomes its own id and never reaches steps 2-4,
 *   2. the runs of text between them are normalized: U+0020 -> U+2581 ("▁"),
 *   3. each Unicode code point of a run becomes one initial symbol: its own vocab id, or - when the
 *      code point is not a token - the <0xNN> byte-fallback tokens of its UTF-8 bytes,
 *   4. BPE over the whole run: repeatedly merge the available pair with the smallest merge rank,
 *   5. wrap the result in the post-processor's <bos> ... <eos> template.
 *
 * Step 4 has no word boundaries to respect: the shipped pre_tokenizer splits on " ", which step 2 has
 * already replaced, so a run is one "word" however long it is. That is why the merge list has 514906
 * entries and why this is a heap BPE and not the textbook O(n^2) scan - a 2000-byte document is a
 * single 2000-symbol word.
 *
 * Not thread-safe: the BPE scratch lives in the tokenizer (one tokenizer per thread), like the rest
 * of the engine's call API.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/e2.h"

void e2_set_error(const char *fmt, ...); /* e2.c: the library's single error slot */

#define TBITS 18      /* vocab size is 1<<18 == 262144 */
#define VTAB_BITS 19  /* 2x the vocab: load factor 0.5 */
#define MTAB_BITS 20
#define MAX_TOKEN 255 /* longest token in the vocab is 93 bytes */
#define MAX_ADDED 64
#define RANK_NONE 0xFFFFFFFFu

typedef struct {
    uint32_t off, hash, id;
    uint8_t len;
} vent; /* 16 bytes: text lives in the shared arena at [off, off+len) */

typedef struct {
    char s[MAX_TOKEN + 1];
    uint32_t len, id;
} addent;

struct e2_tokenizer {
    char *arena;
    uint32_t arena_len, arena_cap;
    vent *vtab;
    uint32_t vmask, n_vocab;
    int32_t ascii[128]; /* single-ASCII-character tokens, -1 when absent */
    uint32_t byte_id[256];

    uint64_t *mtab; /* (a<<18|b)<<20 | rank+1, 0 = empty */
    uint32_t mmask;
    uint32_t *mout; /* rank -> merged token id */
    uint32_t n_merges, mout_cap;

    addent added[MAX_ADDED];
    int n_added;
    uint8_t added_first[32]; /* bitmap of the first byte of each added token */

    uint32_t bos, eos, image_token, boi, eoi;
    uint32_t audio_token, boa, eoa; /* <|audio|> <|audio> <audio|>; 0 when the tokenizer has none */

    /* BPE scratch: a doubly linked list over the symbols plus a min-heap of (rank, pos) validated
     * lazily on pop (an entry is live iff the pair at `pos` still has that rank). */
    int32_t *sym, *prev, *next;
    size_t sym_cap;
    uint64_t *heap;
    size_t heap_n, heap_cap;
};

/* ---------------------------------------------------------------- JSON scanner */
typedef struct {
    const char *p, *end, *base;
} js;

/* Every structural failure in the parser reports where it gave up: a 32 MB file that fails to parse
 * without an offset is not debuggable. */
static int jerr(js *j, const char *what) {
    e2_set_error("tokenizer.json: %s at byte %ld", what, (long)(j->p - j->base));
    return -1;
}

static void js_ws(js *j) {
    while (j->p < j->end && (unsigned char)*j->p <= ' ') j->p++;
}

static int js_lit(js *j, const char *s) {
    size_t n = strlen(s);
    js_ws(j);
    if ((size_t)(j->end - j->p) < n || memcmp(j->p, s, n)) return jerr(j, "js_lit");
    j->p += n;
    return 0;
}

/* Decode the JSON string at the cursor into dst (at most cap-1 bytes + NUL), or -1. Escapes are
 * decoded and non-ASCII escapes are re-encoded as UTF-8; the advance is done per branch because the
 * \uXXXX case consumes five bytes and a surrogate pair seven. */
static long js_str(js *j, char *dst, size_t cap) {
    uint32_t cp;
    size_t n = 0;
    js_ws(j);
    if (j->p >= j->end || *j->p != '"') return jerr(j, "js_str");
    j->p++;
    for (;;) {
        unsigned char c;
        if (j->p >= j->end) return jerr(j, "js_str");
        c = (unsigned char)*j->p;
        if (c == '"') {
            j->p++;
            dst[n] = 0;
            return (long)n;
        }
        if (c != '\\') {
            /* JSON strings are UTF-8 already: a non-ASCII byte is copied as-is, only \uXXXX needs
             * re-encoding, so the two cases do not share a tail. */
            if (n + 1 >= cap) return jerr(j, "js_str");
            dst[n++] = (char)c;
            j->p++;
            continue;
        }
        {
            j->p++;
            if (j->p >= j->end) return jerr(j, "js_str");
            switch (*j->p) {
                case 'n': cp = '\n'; j->p++; break;
                case 't': cp = '\t'; j->p++; break;
                case 'r': cp = '\r'; j->p++; break;
                case 'b': cp = '\b'; j->p++; break;
                case 'f': cp = '\f'; j->p++; break;
                case '/': cp = '/'; j->p++; break;
                case '\\': cp = '\\'; j->p++; break;
                case '"': cp = '"'; j->p++; break;
                case 'u': {
                    unsigned v = 0;
                    j->p++;
                    if ((size_t)(j->end - j->p) < 4) return jerr(j, "js_str");
                    for (int k = 0; k < 4; k++) {
                        char h = j->p[k];
                        v <<= 4;
                        if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                        else return jerr(j, "js_str");
                    }
                    cp = v;
                    j->p += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && (size_t)(j->end - j->p) >= 6 &&
                        j->p[0] == '\\' && j->p[1] == 'u') {
                        unsigned lo = 0;
                        for (int k = 0; k < 4; k++) {
                            char h = j->p[2 + k];
                            lo <<= 4;
                            if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
                            else { lo = 0xFFFFFFFF; break; }
                        }
                        if (lo >= 0xDC00 && lo < 0xE000) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            j->p += 6;
                        }
                    }
                    break;
                }
                default: return jerr(j, "js_str");
            }
        }
        if (cp < 0x80) {
            if (n + 1 >= cap) return jerr(j, "js_str");
            dst[n++] = (char)cp;
        } else if (cp < 0x800) {
            if (n + 2 >= cap) return jerr(j, "js_str");
            dst[n++] = (char)(0xC0 | (cp >> 6));
            dst[n++] = (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            if (n + 3 >= cap) return jerr(j, "js_str");
            dst[n++] = (char)(0xE0 | (cp >> 12));
            dst[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            dst[n++] = (char)(0x80 | (cp & 0x3F));
        } else {
            if (n + 4 >= cap) return jerr(j, "js_str");
            dst[n++] = (char)(0xF0 | (cp >> 18));
            dst[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
            dst[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            dst[n++] = (char)(0x80 | (cp & 0x3F));
        }
    }
}

static int js_int(js *j, long *out) {
    char *end;
    js_ws(j);
    *out = strtol(j->p, &end, 10);
    if (end == j->p) return jerr(j, "js_int");
    j->p = end;
    return 0;
}

static int js_skip(js *j);

static int js_skip_container(js *j, char close) {
    j->p++;
    js_ws(j);
    if (j->p < j->end && *j->p == close) {
        j->p++;
        return 0;
    }
    for (;;) {
        if (js_skip(j)) return jerr(j, "js_skip_container");
        js_ws(j);
        if (j->p >= j->end) return jerr(j, "js_skip_container");
        if (*j->p == ':') { /* object member: the value still has to be skipped */
            j->p++;
            if (js_skip(j)) return jerr(j, "js_skip_container");
            js_ws(j);
            if (j->p >= j->end) return jerr(j, "js_skip_container");
        }
        if (*j->p == ',') {
            j->p++;
            continue;
        }
        if (*j->p == close) {
            j->p++;
            return 0;
        }
        return jerr(j, "js_skip_container");
    }
}

static int js_skip(js *j) {
    char tmp[256];
    long v;
    js_ws(j);
    if (j->p >= j->end) return -1;
    switch (*j->p) {
        case '"': return js_str(j, tmp, sizeof tmp) < 0 ? -1 : 0;
        case '{': return js_skip_container(j, '}');
        case '[': return js_skip_container(j, ']');
        case 't': return js_lit(j, "true");
        case 'f': return js_lit(j, "false");
        case 'n': return js_lit(j, "null");
        default: return js_int(j, &v);
    }
}

/* 1 = the container closed here, 0 = a comma was consumed, -1 = malformed. */
static int js_sep(js *j, char close) {
    js_ws(j);
    if (j->p < j->end && *j->p == close) {
        j->p++;
        return 1;
    }
    if (j->p < j->end && *j->p == ',') {
        j->p++;
        return 0;
    }
    return jerr(j, "js_sep");
}

/* ---------------------------------------------------------------- vocab / merge tables */
static uint32_t hash_bytes(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 16777619u;
    return h;
}

static int vtab_rehash(e2_tokenizer *t, uint32_t nbits) {
    uint32_t n = 1u << nbits;
    vent *nt = calloc(n, sizeof *nt);
    if (!nt) return -1;
    for (uint32_t i = 0; t->vtab && i <= t->vmask; i++) {
        vent e = t->vtab[i];
        uint32_t k;
        if (!e.len) continue;
        k = e.hash & (n - 1);
        while (nt[k].len) k = (k + 1) & (n - 1);
        nt[k] = e;
    }
    free(t->vtab);
    t->vtab = nt;
    t->vmask = n - 1;
    return 0;
}

static int vocab_insert(e2_tokenizer *t, uint32_t off, uint32_t len, uint32_t id) {
    uint32_t h = hash_bytes(t->arena + off, len);
    uint32_t i;
    if (2 * (t->n_vocab + 1) > t->vmask + 1) {
        uint32_t nbits = VTAB_BITS;
        while ((1u << nbits) < 2 * (t->n_vocab + 1)) nbits++;
        if (vtab_rehash(t, nbits)) return -1;
    }
    i = h & t->vmask;
    for (;;) {
        if (!t->vtab[i].len) {
            t->vtab[i].off = off;
            t->vtab[i].len = (uint8_t)len;
            t->vtab[i].hash = h;
            t->vtab[i].id = id;
            t->n_vocab++;
            return 0;
        }
        if (t->vtab[i].hash == h && t->vtab[i].len == len &&
            !memcmp(t->arena + t->vtab[i].off, t->arena + off, len)) {
            t->vtab[i].id = id; /* last wins, like the Rust HashMap */
            return 0;
        }
        i = (i + 1) & t->vmask;
    }
}

static int vocab_find(const e2_tokenizer *t, const char *s, size_t len) {
    uint32_t h, i;
    if (!t->vtab || len == 0 || len > MAX_TOKEN) return -1;
    h = hash_bytes(s, len);
    i = h & t->vmask;
    for (;;) {
        const vent *e = &t->vtab[i];
        if (!e->len) return -1;
        if (e->hash == h && e->len == len && !memcmp(t->arena + e->off, s, len)) return (int)e->id;
        i = (i + 1) & t->vmask;
    }
}

static inline uint32_t merge_rank(const e2_tokenizer *t, int32_t a, int32_t b) {
    uint64_t key;
    uint32_t i;
    if (a < 0 || b < 0) return RANK_NONE;
    key = ((uint64_t)(uint32_t)a << TBITS) | (uint32_t)b;
    i = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> (64 - MTAB_BITS));
    for (;;) {
        uint64_t s = t->mtab[i];
        if (!s) return RANK_NONE;
        if ((s >> 20) == key) return (uint32_t)(s & 0xFFFFF) - 1;
        i = (i + 1) & t->mmask;
    }
}

static void merge_insert(e2_tokenizer *t, uint32_t a, uint32_t b, uint32_t rank) {
    uint64_t key = ((uint64_t)a << TBITS) | b;
    uint64_t val = (key << 20) | (rank + 1);
    uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> (64 - MTAB_BITS));
    for (;;) {
        if (!t->mtab[i] || (t->mtab[i] >> 20) == key) {
            t->mtab[i] = val;
            return;
        }
        i = (i + 1) & t->mmask;
    }
}

static int merge_out_push(e2_tokenizer *t, uint32_t id) {
    if (t->n_merges >= (1u << MTAB_BITS) - 1) {
        /* the rank is packed into 20 bits next to its key, and the table is sized for this file's
         * 514906 merges; a tokenizer with more would need a bigger MTAB_BITS */
        e2_set_error("tokenizer.json: merge list too long");
        return -1;
    }
    if (t->n_merges == t->mout_cap) {
        uint32_t cap = t->mout_cap ? t->mout_cap * 2 : 4096;
        uint32_t *p = realloc(t->mout, sizeof(uint32_t) * cap);
        if (!p) return -1;
        t->mout = p;
        t->mout_cap = cap;
    }
    t->mout[t->n_merges++] = id;
    return 0;
}

/* ---------------------------------------------------------------- loading */
static int parse_vocab(e2_tokenizer *t, js *j) {
    int r;
    if (js_lit(j, "{")) return jerr(j, "parse_vocab");
    for (;;) {
        long v, n;
        js_ws(j);
        if (j->p < j->end && *j->p == '}') {
            j->p++;
            return 0;
        }
        if (t->arena_len + MAX_TOKEN + 1 > t->arena_cap) {
            uint32_t cap = t->arena_cap ? t->arena_cap * 2 : (2u << 20);
            char *p;
            while (cap < t->arena_len + MAX_TOKEN + 1) cap *= 2;
            p = realloc(t->arena, cap);
            if (!p) return jerr(j, "parse_vocab");
            t->arena = p;
            t->arena_cap = cap;
        }
        n = js_str(j, t->arena + t->arena_len, MAX_TOKEN + 1);
        if (n < 0) return jerr(j, "parse_vocab");
        js_ws(j);
        if (j->p >= j->end || *j->p != ':') return jerr(j, "parse_vocab");
        j->p++;
        if (js_int(j, &v) || v < 0 || v > 0xFFFFFF) return jerr(j, "parse_vocab");
        if (vocab_insert(t, t->arena_len, (uint32_t)n, (uint32_t)v)) return jerr(j, "parse_vocab");
        t->arena_len += (uint32_t)n;
        r = js_sep(j, '}');
        if (r < 0) return jerr(j, "parse_vocab");
        if (r) return 0;
    }
}

static int parse_merges(e2_tokenizer *t, js *j) {
    int r = js_lit(j, "[");
    if (r) return jerr(j, "parse_merges");
    js_ws(j);
    if (j->p < j->end && *j->p == ']') { /* not js_sep: an element may follow, not a separator */
        j->p++;
        return 0;
    }
    for (;;) {
        char a[MAX_TOKEN + 1], b[MAX_TOKEN + 1], cat[2 * MAX_TOKEN + 1];
        long na, nb;
        int ia, ib, io;
        js_ws(j);
        if (j->p >= j->end || *j->p != '[') return jerr(j, "parse_merges");
        j->p++;
        if ((na = js_str(j, a, sizeof a)) < 0) return jerr(j, "parse_merges");
        r = js_sep(j, ']');
        if (r < 0 || r) return jerr(j, "parse_merges");
        if ((nb = js_str(j, b, sizeof b)) < 0) return jerr(j, "parse_merges");
        r = js_sep(j, ']'); /* the pair's closing bracket */
        if (r < 0 || !r) return jerr(j, "parse_merges");
        ia = vocab_find(t, a, (size_t)na);
        ib = vocab_find(t, b, (size_t)nb);
        if (ia < 0 || ib < 0) return jerr(j, "parse_merges");
        memcpy(cat, a, (size_t)na);
        memcpy(cat + na, b, (size_t)nb);
        io = vocab_find(t, cat, (size_t)(na + nb));
        if (merge_out_push(t, io < 0 ? 0xFFFFFFFFu : (uint32_t)io)) return jerr(j, "parse_merges");
        if (io >= 0) merge_insert(t, (uint32_t)ia, (uint32_t)ib, t->n_merges - 1);
        r = js_sep(j, ']'); /* the merge list's separator */
        if (r < 0) return jerr(j, "parse_merges");
        if (r) return 0;
    }
}

/* Added tokens are matched at a byte position and the first match wins, so the table is sorted
 * longest-first: <|image|> must not be shadowed by <|image>. */
static int added_cmp(const void *pa, const void *pb) {
    const addent *a = pa, *b = pb;
    return a->len < b->len ? 1 : a->len > b->len ? -1 : 0;
}

static int parse_added(e2_tokenizer *t, js *j) {
    int r = js_lit(j, "[");
    if (r) return jerr(j, "parse_added");
    js_ws(j);
    if (j->p < j->end && *j->p == ']') {
        j->p++;
        return 0;
    }
    for (;;) {
        char key[MAX_TOKEN + 1], content[MAX_TOKEN + 1];
        long id = -1;
        int have_content = 0;
        content[0] = 0;
        if (js_lit(j, "{")) return jerr(j, "parse_added");
        for (;;) {
            js_ws(j);
            if (j->p < j->end && *j->p == '}') {
                j->p++;
                break;
            }
            if (js_str(j, key, sizeof key) < 0) return jerr(j, "parse_added");
            js_ws(j);
            if (j->p >= j->end || *j->p != ':') return jerr(j, "parse_added");
            j->p++;
            if (!strcmp(key, "content")) {
                if (js_str(j, content, sizeof content) < 0) return jerr(j, "parse_added");
                have_content = 1;
            } else if (!strcmp(key, "id")) {
                if (js_int(j, &id)) return jerr(j, "parse_added");
            } else if (js_skip(j)) {
                return jerr(j, "parse_added");
            }
            r = js_sep(j, '}');
            if (r < 0) return jerr(j, "parse_added");
            if (r) break;
        }
        if (id >= 0 && have_content) {
            size_t n = strlen(content);
            if (n > MAX_TOKEN || t->n_added >= MAX_ADDED || n == 0) return jerr(j, "parse_added");
            memcpy(t->added[t->n_added].s, content, n + 1);
            t->added[t->n_added].len = (uint32_t)n;
            t->added[t->n_added].id = (uint32_t)id;
            t->added_first[(unsigned char)content[0] >> 3] |= (uint8_t)(1u << (content[0] & 7));
            t->n_added++;
        }
        r = js_sep(j, ']');
        if (r < 0) return jerr(j, "parse_added");
        if (r) return 0;
    }
}

static int parse_model(e2_tokenizer *t, js *j) {
    int r;
    if (js_lit(j, "{")) return jerr(j, "parse_model");
    for (;;) {
        char key[64];
        js_ws(j);
        if (j->p < j->end && *j->p == '}') {
            j->p++;
            return 0;
        }
        if (js_str(j, key, sizeof key) < 0) return jerr(j, "parse_model");
        js_ws(j);
        if (j->p >= j->end || *j->p != ':') return jerr(j, "parse_model");
        j->p++;
        if (!strcmp(key, "vocab")) {
            if (parse_vocab(t, j)) return jerr(j, "parse_model");
        } else if (!strcmp(key, "merges")) {
            if (parse_merges(t, j)) return jerr(j, "parse_model");
        } else if (!strcmp(key, "type")) {
            char v[16];
            if (js_str(j, v, sizeof v) < 0) return jerr(j, "parse_model");
            if (strcmp(v, "BPE")) {
                e2_set_error("tokenizer.json: model type is %s, expected BPE", v);
                return -1;
            }
        } else if (!strcmp(key, "ignore_merges") || !strcmp(key, "byte_fallback")) {
            /* byte_fallback on, ignore_merges off: the two flags the merge algorithm depends on */
            int want_on = key[0] == 'b';
            int on = js_lit(j, "true") == 0;
            if (!on && js_lit(j, "false")) return jerr(j, "parse_model");
            if (on != want_on) {
                e2_set_error("tokenizer.json: %s is %s, which this tokenizer does not implement", key,
                             on ? "true" : "false");
                return -1;
            }
        } else if (js_skip(j)) {
            return jerr(j, "parse_model");
        }
        r = js_sep(j, '}');
        if (r < 0) return jerr(j, "parse_model");
        if (r) return 0;
    }
}

static int parse_root(e2_tokenizer *t, const char *buf, size_t len) {
    js j = { buf, buf + len, buf };
    int seen_model = 0, r;
    if (js_lit(&j, "{")) return jerr(&j, "parse_root");
    for (;;) {
        char key[64];
        js_ws(&j);
        if (j.p < j.end && *j.p == '}') {
            j.p++;
            break;
        }
        if (js_str(&j, key, sizeof key) < 0) return jerr(&j, "parse_root");
        js_ws(&j);
        if (j.p >= j.end || *j.p != ':') return jerr(&j, "parse_root");
        j.p++;
        if (!strcmp(key, "model")) {
            if (parse_model(t, &j)) return -1;
            seen_model = 1;
        } else if (!strcmp(key, "added_tokens")) {
            if (parse_added(t, &j)) return -1;
        } else if (js_skip(&j)) {
            return jerr(&j, "parse_root");
        }
        r = js_sep(&j, '}');
        if (r < 0) return jerr(&j, "parse_root");
        if (r) break;
    }
    if (!seen_model) {
        e2_set_error("tokenizer.json: no model section");
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------- BPE */
static int scratch_reserve(e2_tokenizer *t, size_t n) {
    size_t cap, i;
    if (n <= t->sym_cap) return 0;
    /* grows by 2x from empty; commit each realloc as it succeeds so a failure leaves the previous
     * (smaller) capacity intact and the next call simply retries */
    for (i = 0; i < 4; i++) {
        void *p;
        cap = t->sym_cap ? t->sym_cap : 256;
        while (cap < n) cap *= 2;
        switch (i) {
            case 0:
                p = realloc(t->sym, cap * sizeof *t->sym);
                if (!p) return -1;
                t->sym = p;
                break;
            case 1:
                p = realloc(t->prev, cap * sizeof *t->prev);
                if (!p) return -1;
                t->prev = p;
                break;
            case 2:
                p = realloc(t->next, cap * sizeof *t->next);
                if (!p) return -1;
                t->next = p;
                break;
            default:
                p = realloc(t->heap, 2 * cap * sizeof *t->heap);
                if (!p) return -1;
                t->heap = p;
                t->heap_cap = 2 * cap;
                break;
        }
    }
    t->sym_cap = cap;
    return 0;
}

static void heap_push(e2_tokenizer *t, uint32_t rank, int32_t pos) {
    uint64_t v = ((uint64_t)rank << 32) | (uint32_t)pos;
    size_t i = t->heap_n++;
    t->heap[i] = v;
    while (i) {
        size_t up = (i - 1) / 2;
        if (t->heap[up] <= t->heap[i]) break;
        uint64_t tmp = t->heap[up];
        t->heap[up] = t->heap[i];
        t->heap[i] = tmp;
        i = up;
    }
}

static int heap_pop(e2_tokenizer *t, uint32_t *rank, int32_t *pos) {
    uint64_t top;
    size_t i = 0;
    if (!t->heap_n) return 0;
    top = t->heap[0];
    t->heap_n--;
    t->heap[0] = t->heap[t->heap_n];
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < t->heap_n && t->heap[l] < t->heap[m]) m = l;
        if (r < t->heap_n && t->heap[r] < t->heap[m]) m = r;
        if (m == i) break;
        uint64_t tmp = t->heap[m];
        t->heap[m] = t->heap[i];
        t->heap[i] = tmp;
        i = m;
    }
    *rank = (uint32_t)(top >> 32);
    *pos = (int32_t)(uint32_t)top;
    return 1;
}

/* Merge the symbol run in ids[0..n) in place; returns the new length or -1. */
static int bpe(e2_tokenizer *t, int32_t *ids, int n) {
    uint32_t rank;
    int32_t pos, i;
    int k;
    if (n < 2) return n;
    if (scratch_reserve(t, (size_t)n)) return -1;
    for (i = 0; i < n; i++) {
        t->sym[i] = ids[i];
        t->prev[i] = i - 1;
        t->next[i] = i + 1;
    }
    t->next[n - 1] = -1;
    t->heap_n = 0;
    for (i = 0; i + 1 < n; i++) {
        rank = merge_rank(t, t->sym[i], t->sym[i + 1]);
        if (rank != RANK_NONE) heap_push(t, rank, i);
    }
    while (heap_pop(t, &rank, &pos)) {
        int32_t nxt = t->next[pos], pv;
        if (nxt < 0 || t->sym[pos] < 0 || t->sym[nxt] < 0) continue;
        if (merge_rank(t, t->sym[pos], t->sym[nxt]) != rank) continue; /* stale heap entry */
        t->sym[pos] = (int32_t)t->mout[rank];
        t->sym[nxt] = -1;
        t->next[pos] = t->next[nxt];
        if (t->next[nxt] >= 0) t->prev[t->next[nxt]] = pos;
        pv = t->prev[pos];
        if (pv >= 0) {
            uint32_t r = merge_rank(t, t->sym[pv], t->sym[pos]);
            if (r != RANK_NONE) heap_push(t, r, pv);
        }
        if (t->next[pos] >= 0) {
            uint32_t r = merge_rank(t, t->sym[pos], t->sym[t->next[pos]]);
            if (r != RANK_NONE) heap_push(t, r, pos);
        }
    }
    k = 0;
    for (i = 0; i >= 0; i = t->next[i]) ids[k++] = t->sym[i];
    return k;
}

/* ---------------------------------------------------------------- encode */
/* Returns 0 for a valid sequence, -1 for a bogus one (which is then passed through byte by byte). */
static int utf8_decode(const char *p, const char *end, uint32_t *cp, int *len) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x80) {
        *cp = c;
        *len = 1;
        return 0;
    }
    if ((c & 0xE0) == 0xC0 && end - p >= 2) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)p[1] & 0x3F);
        *len = 2;
        return *cp >= 0x80 ? 0 : -1;
    }
    if ((c & 0xF0) == 0xE0 && end - p >= 3) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | (((unsigned char)p[1] & 0x3F) << 6) |
              ((unsigned char)p[2] & 0x3F);
        *len = 3;
        return *cp >= 0x800 ? 0 : -1;
    }
    if ((c & 0xF8) == 0xF0 && end - p >= 4) {
        *cp = ((uint32_t)(c & 0x07) << 18) | (((unsigned char)p[1] & 0x3F) << 12) |
              (((unsigned char)p[2] & 0x3F) << 6) | ((unsigned char)p[3] & 0x3F);
        *len = 4;
        return *cp >= 0x10000 && *cp <= 0x10FFFF ? 0 : -1;
    }
    *cp = c;
    *len = 1;
    return -1;
}

static int utf8_encode(uint32_t cp, char *buf) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Normalize + symbolize + BPE one raw run of text; appends to ids and returns the new count. */
static int encode_run(e2_tokenizer *t, const char *p, const char *end, int32_t *ids, int n, int max) {
    int sym = 0;
    while (p < end) {
        uint32_t cp;
        char buf[4];
        int len = 0, k, id;
        if (utf8_decode(p, end, &cp, &len)) { /* bogus byte: same treatment the byte would get */
            if (n + sym + 1 > max) return -1;
            ids[n + sym++] = (int32_t)t->byte_id[(unsigned char)*p];
            p++;
            continue;
        }
        p += len;
        if (cp == ' ') cp = 0x2581; /* the normalizer */
        k = utf8_encode(cp, buf);
        id = cp < 128 ? t->ascii[cp] : vocab_find(t, buf, (size_t)k);
        if (id >= 0) {
            if (n + sym + 1 > max) return -1;
            ids[n + sym++] = id;
        } else {
            for (int i = 0; i < k; i++) {
                if (n + sym + 1 > max) return -1;
                ids[n + sym++] = (int32_t)t->byte_id[(unsigned char)buf[i]];
            }
        }
    }
    if (sym == 0) return n;
    sym = bpe(t, ids + n, sym);
    if (sym < 0) return -1;
    return n + sym;
}

static int emit(int32_t *ids, int n, int max, int32_t id) {
    if (n >= max) return -1;
    ids[n] = id;
    return n + 1;
}

static int encode_core(e2_tokenizer *t, const char *text, const int32_t *n_soft, int n_images,
                       const int32_t *n_audio_soft, int n_audios, int32_t *ids, int max_ids) {
    const char *p = text, *seg = text;
    const char *pend = text + strlen(text);
    int n = 0, img = 0, aud = 0;
    if (max_ids < 2) {
        e2_set_error("id buffer too small for the <bos>/<eos> template");
        return -1;
    }
    n = emit(ids, n, max_ids, (int32_t)t->bos);
    while (p < pend) {
        int best = -1;
        unsigned char c = (unsigned char)*p;
        if ((t->added_first[c >> 3] >> (c & 7)) & 1) {
            for (int i = 0; i < t->n_added; i++) {
                uint32_t l = t->added[i].len;
                if ((size_t)(pend - p) >= l && t->added[i].s[0] == (char)c &&
                    !memcmp(p, t->added[i].s, l)) {
                    best = i; /* the table is longest-first, so the first match is the longest */
                    break;
                }
            }
        }
        if (best < 0) {
            p++;
            continue;
        }
        if ((n = encode_run(t, seg, p, ids, n, max_ids - 1)) < 0) goto too_small;
        seg = p + t->added[best].len;
        p = seg;
        if (t->added[best].id == t->image_token && n_images > 0) {
            if (img >= n_images) {
                e2_set_error("more <|image|> placeholders in the text than images");
                return -1;
            }
            if ((n = emit(ids, n, max_ids - 1, (int32_t)t->boi)) < 0) goto too_small;
            for (int32_t k = 0; k < n_soft[img]; k++)
                if ((n = emit(ids, n, max_ids - 1, (int32_t)t->image_token)) < 0) goto too_small;
            if ((n = emit(ids, n, max_ids - 1, (int32_t)t->eoi)) < 0) goto too_small;
            img++;
        } else if (t->audio_token && t->added[best].id == t->audio_token && n_audios > 0) {
            if (aud >= n_audios) {
                e2_set_error("more <|audio|> placeholders in the text than audio clips");
                return -1;
            }
            if ((n = emit(ids, n, max_ids - 1, (int32_t)t->boa)) < 0) goto too_small;
            for (int32_t k = 0; k < n_audio_soft[aud]; k++)
                if ((n = emit(ids, n, max_ids - 1, (int32_t)t->audio_token)) < 0) goto too_small;
            if ((n = emit(ids, n, max_ids - 1, (int32_t)t->eoa)) < 0) goto too_small;
            aud++;
        } else if ((n = emit(ids, n, max_ids - 1, (int32_t)t->added[best].id)) < 0) {
            goto too_small;
        }
    }
    if ((n = encode_run(t, seg, p, ids, n, max_ids - 1)) < 0) goto too_small;
    if ((n = emit(ids, n, max_ids, (int32_t)t->eos)) < 0) goto too_small;
    return n;
too_small:
    e2_set_error("id buffer too small: 2 + 3 tokens per input byte is the bound");
    return -1;
}

int e2_encode(e2_tokenizer *t, const char *text, int32_t *ids, int max_ids) {
    if (!t || !text || !ids) return -1;
    return encode_core(t, text, NULL, 0, NULL, 0, ids, max_ids);
}

int e2_encode_mm(e2_tokenizer *t, const char *text, const int32_t *n_soft_tokens, int n_images,
                 int32_t *ids, int max_ids) {
    if (!t || !text || !ids) return -1;
    if (n_images < 0 || (n_images > 0 && !n_soft_tokens)) return -1;
    return encode_core(t, text, n_soft_tokens, n_images, NULL, 0, ids, max_ids);
}

int e2_encode_mma(e2_tokenizer *t, const char *text, const int32_t *n_image_soft, int n_images,
                  const int32_t *n_audio_soft, int n_audios, int32_t *ids, int max_ids) {
    if (!t || !text || !ids) return -1;
    if (n_images < 0 || (n_images > 0 && !n_image_soft)) return -1;
    if (n_audios < 0 || (n_audios > 0 && !n_audio_soft)) return -1;
    return encode_core(t, text, n_image_soft, n_images, n_audio_soft, n_audios, ids, max_ids);
}

/* ---------------------------------------------------------------- load / free */
e2_tokenizer *e2_tokenizer_load(const char *model_dir) {
    char path[1024];
    long size;
    char *buf;
    FILE *f;
    e2_tokenizer *t;
    snprintf(path, sizeof path, "%s/tokenizer.json", model_dir);
    f = fopen(path, "rb");
    if (!f) {
        e2_set_error("cannot open %s", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) || (size = ftell(f)) <= 0) {
        fclose(f);
        e2_set_error("cannot size %s", path);
        return NULL;
    }
    rewind(f);
    buf = malloc((size_t)size);
    if (!buf || fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        e2_set_error("cannot read %s", path);
        return NULL;
    }
    fclose(f);

    t = calloc(1, sizeof *t);
    if (!t) {
        free(buf);
        e2_set_error("oom");
        return NULL;
    }
    for (int i = 0; i < 128; i++) t->ascii[i] = -1;
    t->mtab = calloc(1u << MTAB_BITS, sizeof(uint64_t));
    t->mmask = (1u << MTAB_BITS) - 1;
    if (!t->mtab || vtab_rehash(t, VTAB_BITS)) goto fail;

    if (parse_root(t, buf, (size_t)size)) goto fail;
    free(buf);
    buf = NULL;
    if (t->n_merges == 0 || t->n_vocab == 0) {
        e2_set_error("tokenizer.json: empty vocab or merges");
        goto fail;
    }
    {
        char one[2] = { 0, 0 };
        for (int i = 0; i < 128; i++) {
            if (i == ' ') continue;
            one[0] = (char)i;
            t->ascii[i] = vocab_find(t, one, 1);
        }
    }
    for (int i = 0; i < 256; i++) {
        char b[8];
        int id;
        snprintf(b, sizeof b, "<0x%02X>", i);
        id = vocab_find(t, b, 6);
        if (id < 0) {
            e2_set_error("tokenizer.json: no byte token <0x%02X>", i);
            goto fail;
        }
        t->byte_id[i] = (uint32_t)id;
    }
    {
        int bos = vocab_find(t, "<bos>", 5), eos = vocab_find(t, "<eos>", 5);
        int a = vocab_find(t, "<|image|>", 9), b = vocab_find(t, "<|image>", 8),
            c = vocab_find(t, "<image|>", 8);
        if (bos < 0 || eos < 0) {
            e2_set_error("tokenizer.json: no <bos>/<eos>");
            goto fail;
        }
        if (a < 0 || b < 0 || c < 0) {
            e2_set_error("tokenizer.json: no <|image|>/<|image>/<image|>");
            goto fail;
        }
        t->bos = (uint32_t)bos;
        t->eos = (uint32_t)eos;
        t->image_token = (uint32_t)a;
        t->boi = (uint32_t)b;
        t->eoi = (uint32_t)c;
        /* the audio placeholders are optional: a text+image-only tokenizer has none, and then
         * <|audio|> stays an ordinary special token */
        {
            int at = vocab_find(t, "<|audio|>", 9), ab = vocab_find(t, "<|audio>", 8),
                ae = vocab_find(t, "<audio|>", 8);
            if (at >= 0 && ab >= 0 && ae >= 0) {
                t->audio_token = (uint32_t)at;
                t->boa = (uint32_t)ab;
                t->eoa = (uint32_t)ae;
            }
        }
    }
    qsort(t->added, (size_t)t->n_added, sizeof t->added[0], added_cmp);
    return t;

fail:
    free(buf);
    e2_tokenizer_free(t);
    return NULL;
}

void e2_tokenizer_free(e2_tokenizer *t) {
    if (!t) return;
    free(t->arena);
    free(t->vtab);
    free(t->mtab);
    free(t->mout);
    free(t->sym);
    free(t->prev);
    free(t->next);
    free(t->heap);
    free(t);
}
