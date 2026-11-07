/* e2_tokenizer.c - Gemma BPE tokenizer for the packed E2 tokenizer blob. */
#include "e2_internal.h"

#include <stdlib.h>

#define TOK_MAGIC "E2TOK1"

typedef struct {
    uint64_t key;   /* hash for vocab, (l<<32|r) for merges */
    int32_t a, b;
} tok_slot;

struct e2_tok {
    unsigned char *blob;
    size_t blob_size;
    int ntok, nmerge, nspecial;
    uint32_t *off, *len;
    tok_slot *vocab;   int vocab_mask;
    tok_slot *merges;  int merge_mask;
    struct { uint32_t off, len; int id; } *sp;
    int bos, eos;
};

static uint64_t hash64(const void *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char *s = p;
    for (size_t i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ULL; }
    return h ? h : 1;
}

static uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static const unsigned char *tok_str(const e2_tok *t, int id) { return t->blob + t->off[id]; }

static int vocab_find(const e2_tok *t, const char *s, int len) {
    uint64_t h = hash64(s, (size_t)len);
    int i = (int)(h & (uint64_t)t->vocab_mask);
    for (int probe = 0; probe <= t->vocab_mask; probe++) {
        tok_slot *e = &t->vocab[i];
        if (!e->key) return -1;
        if (e->key == h && (int)t->len[e->a] == len &&
            memcmp(tok_str(t, e->a), s, (size_t)len) == 0)
            return e->a;
        i = (i + 1) & t->vocab_mask;
    }
    return -1;
}

static int merge_find(const e2_tok *t, int l, int r) {
    uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    int i = (int)(splitmix(key) & (uint64_t)t->merge_mask);
    for (int probe = 0; probe <= t->merge_mask; probe++) {
        tok_slot *e = &t->merges[i];
        if (!e->key) return -1;
        if (e->key == key) return e->b;
        i = (i + 1) & t->merge_mask;
    }
    return -1;
}

static int merge_rank(const e2_tok *t, int l, int r) {
    uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    int i = (int)(splitmix(key) & (uint64_t)t->merge_mask);
    for (int probe = 0; probe <= t->merge_mask; probe++) {
        tok_slot *e = &t->merges[i];
        if (!e->key) return 0x7fffffff;
        if (e->key == key) return e->a;
        i = (i + 1) & t->merge_mask;
    }
    return 0x7fffffff;
}

static unsigned char *read_all(const char *path, size_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    unsigned char *p = malloc((size_t)n);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f);
    *out = (size_t)n;
    return p;
}

e2_tok *e2_tok_load(const char *path) {
    size_t n;
    unsigned char *b = read_all(path, &n);
    if (!b) { e2_fail("cannot read tokenizer: %s", path); return NULL; }
    if (n < 16 || memcmp(b, TOK_MAGIC, 7) != 0) {
        free(b); e2_fail("bad tokenizer file"); return NULL;
    }
    e2_tok *t = calloc(1, sizeof(*t));
    t->blob = b;
    t->blob_size = n;
    const unsigned char *p = b + 8;
    memcpy(&t->ntok, p, 4); p += 4;
    memcpy(&t->nmerge, p, 4); p += 4;
    t->off = calloc((size_t)t->ntok, sizeof(uint32_t));
    t->len = calloc((size_t)t->ntok, sizeof(uint32_t));
    if (!t->off || !t->len) { e2_tok_free(t); e2_fail("out of memory"); return NULL; }
    int maxlen = 1;
    for (int i = 0; i < t->ntok; i++) {
        uint32_t l;
        memcpy(&l, p, 4); p += 4;
        t->off[i] = (uint32_t)(p - b);
        t->len[i] = l;
        if ((int)l > maxlen) maxlen = (int)l;
        p += l;
    }
    int vcap = 1;
    while (vcap < t->ntok * 2) vcap <<= 1;
    t->vocab_mask = vcap - 1;
    t->vocab = calloc((size_t)vcap, sizeof(tok_slot));
    if (!t->vocab) { e2_tok_free(t); e2_fail("out of memory"); return NULL; }
    for (int i = 0; i < t->ntok; i++) {
        uint64_t h = hash64(tok_str(t, i), t->len[i]);
        if (!h) h = 1;
        int j = (int)(h & (uint64_t)t->vocab_mask);
        while (t->vocab[j].key) j = (j + 1) & t->vocab_mask;
        t->vocab[j].key = h;
        t->vocab[j].a = i;
    }
    t->bos = vocab_find(t, "<bos>", 5);
    t->eos = vocab_find(t, "<eos>", 5);
    if (t->bos < 0) t->bos = 2;
    if (t->eos < 0) t->eos = 1;

    int mcap = 1;
    while (mcap < t->nmerge * 2) mcap <<= 1;
    t->merge_mask = mcap - 1;
    t->merges = calloc((size_t)mcap, sizeof(tok_slot));
    if (!t->merges) { e2_tok_free(t); e2_fail("out of memory"); return NULL; }
    for (int i = 0; i < t->nmerge; i++) {
        uint32_t l, r, rank, merged;
        memcpy(&l, p, 4); memcpy(&r, p + 4, 4);
        memcpy(&rank, p + 8, 4); memcpy(&merged, p + 12, 4);
        p += 16;
        uint64_t key = ((uint64_t)l << 32) | r;
        int j = (int)(splitmix(key) & (uint64_t)t->merge_mask);
        while (t->merges[j].key) j = (j + 1) & t->merge_mask;
        t->merges[j].key = key;
        t->merges[j].a = (int32_t)rank;
        t->merges[j].b = (int32_t)merged;
    }

    memcpy(&t->nspecial, p, 4); p += 4;
    t->sp = calloc((size_t)t->nspecial, sizeof(*t->sp));
    if (!t->sp) { e2_tok_free(t); e2_fail("out of memory"); return NULL; }
    for (int i = 0; i < t->nspecial; i++) {
        uint32_t l, id;
        memcpy(&l, p, 4); memcpy(&id, p + 4, 4);
        p += 8;
        t->sp[i].off = (uint32_t)(p - b);
        t->sp[i].len = l;
        t->sp[i].id = (int)id;
        p += l;
    }
    return t;
}

void e2_tok_free(e2_tok *t) {
    if (!t) return;
    free(t->blob);
    free(t->off);
    free(t->len);
    free(t->vocab);
    free(t->merges);
    free(t->sp);
    free(t);
}

int e2_tok_id(e2_tok *t, const char *token) {
    return vocab_find(t, token, (int)strlen(token));
}

typedef struct { int rank, node; } bpe_ev;

static void heap_push(bpe_ev *h, int *n, int rank, int node) {
    if (rank == 0x7fffffff) return;
    int i = (*n)++;
    h[i].rank = rank; h[i].node = node;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].rank < h[i].rank ||
            (h[p].rank == h[i].rank && h[p].node <= h[i].node)) break;
        bpe_ev tmp = h[p]; h[p] = h[i]; h[i] = tmp;
        i = p;
    }
}

static bpe_ev heap_pop(bpe_ev *h, int *n) {
    bpe_ev top = h[0];
    h[0] = h[--(*n)];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n && (h[c + 1].rank < h[c].rank ||
            (h[c + 1].rank == h[c].rank && h[c + 1].node < h[c].node))) c++;
        if (h[i].rank < h[c].rank ||
            (h[i].rank == h[c].rank && h[i].node <= h[c].node)) break;
        bpe_ev tmp = h[c]; h[c] = h[i]; h[i] = tmp;
        i = c;
    }
    return top;
}

/* Encode one already-normalized byte string into symbol ids (no specials). */
static int encode_piece(e2_tok *t, const unsigned char *s, int n, int *ids, int max) {
    int *sym = malloc((size_t)n * sizeof(int));
    if (!sym) return 0;
    int ns = 0;
    for (int i = 0; i < n;) {
        int clen = 1;
        unsigned char c = s[i];
        if (c >= 0xF0) clen = 4;
        else if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;
        if (i + clen > n) clen = 1;
        int id = vocab_find(t, (const char *)s + i, clen);
        if (id >= 0) {
            sym[ns++] = id;
        } else {
            for (int j = 0; j < clen; j++) {
                char bb[8];
                snprintf(bb, sizeof(bb), "<0x%02X>", s[i + j]);
                int bid = vocab_find(t, bb, 6);
                if (bid >= 0) sym[ns++] = bid;
            }
        }
        i += clen;
    }
    /* BPE with a rank heap and a linked list: O(n log n) instead of the
     * O(n^2) rescans.  A stale entry is detected by rechecking the pair. */
    {
        int *prev = malloc((size_t)n * sizeof(int));
        int *next = malloc((size_t)n * sizeof(int));
        int *alive = malloc((size_t)n * sizeof(int));
        bpe_ev *heap = malloc((size_t)(3 * n + 8) * sizeof(bpe_ev));
        if (!prev || !next || !alive || !heap) {
            free(prev); free(next); free(alive); free(heap);
            free(sym); return -1;
        }
        int hn = 0;
        for (int i = 0; i < ns; i++) { prev[i] = i - 1; next[i] = i + 1; alive[i] = 1; }
        next[ns - 1] = -1;
        for (int i = 0; i + 1 < ns; i++)
            heap_push(heap, &hn, merge_rank(t, sym[i], sym[i + 1]), i);
        while (hn) {
            bpe_ev ev = heap_pop(heap, &hn);
            int i = ev.node;
            if (ev.rank == 0x7fffffff || !alive[i]) continue;
            int j = next[i];
            if (j < 0 || !alive[j]) continue;
            if (merge_rank(t, sym[i], sym[j]) != ev.rank) continue;
            int merged = merge_find(t, sym[i], sym[j]);
            if (merged < 0) continue;
            sym[i] = merged;
            int k = next[j];
            alive[j] = 0;
            next[i] = k;
            if (k >= 0) prev[k] = i;
            if (prev[i] >= 0)
                heap_push(heap, &hn, merge_rank(t, sym[prev[i]], sym[i]), prev[i]);
            if (k >= 0)
                heap_push(heap, &hn, merge_rank(t, sym[i], sym[k]), i);
        }
        int out = 0;
        for (int i = 0; i < ns; i++)
            if (alive[i]) { sym[out++] = sym[i]; }
        ns = out;
        free(prev); free(next); free(alive); free(heap);
    }
    if (ns > max) { free(sym); return -1; }
    for (int i = 0; i < ns; i++) ids[i] = sym[i];
    free(sym);
    return ns;
}

/* Normalize spaces to U+2581 and encode, splitting out added specials. */
int e2_tok_encode(e2_tok *t, const char *text, int *ids, int max_ids) {
    int n = (int)strlen(text);
    unsigned char *norm = malloc((size_t)n * 4 + 4);
    if (!norm) return -1;
    int nn = 0;
    for (int i = 0; i < n; i++) {
        if (text[i] == ' ') {
            norm[nn++] = 0xE2; norm[nn++] = 0x96; norm[nn++] = 0x81;
        } else {
            norm[nn++] = (unsigned char)text[i];
        }
    }
    int out = 0;
    int seg = 0;
    for (int i = 0; i < nn;) {
        int matched = -1, mlen = 0;
        for (int j = 0; norm[i] == '<' && j < t->nspecial; j++) {
            int l = (int)t->sp[j].len;
            if (l > mlen && i + l <= nn &&
                memcmp(norm + i, t->blob + t->sp[j].off, (size_t)l) == 0) {
                matched = j;
                mlen = l;
            }
        }
        if (matched >= 0) {
            if (i > seg) {
                int r = encode_piece(t, norm + seg, i - seg, ids + out, max_ids - out);
                if (r < 0) { free(norm); return -1; }
                out += r;
            }
            if (out >= max_ids) { free(norm); return -1; }
            ids[out++] = t->sp[matched].id;
            i += mlen;
            seg = i;
        } else {
            i++;
        }
    }
    if (nn > seg) {
        int r = encode_piece(t, norm + seg, nn - seg, ids + out, max_ids - out);
        if (r < 0) { free(norm); return -1; }
        out += r;
    }
    free(norm);
    return out;
}
