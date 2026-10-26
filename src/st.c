/* Safetensors: bounded JSON header parser and validated mmap'd tensor views. */
#include "st.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define ST_MAX_TENSORS 4096
#define ST_MAX_DIMS 8
#define ST_MAX_HEADER 100000000

static char g_err[512];
const char *st_error(void) { return g_err; }

typedef struct { const char *p, *end; int depth; } jctx;
static void j_ws(jctx *c) {
    while (c->p < c->end && strchr(" \t\n\r", *c->p) && *c->p) c->p++;
}
static int j_take(jctx *c, char ch) {
    j_ws(c);
    if (c->p == c->end || *c->p != ch) return -1;
    c->p++;
    return 0;
}
static int j_string(jctx *c, char *buf, size_t cap) {
    size_t n = 0;
    if (j_take(c, '"')) return -1;
    while (c->p < c->end) {
        unsigned char ch = (unsigned char)*c->p++;
        if (ch == '"') { if (buf) buf[n] = 0; return 0; }
        if (ch < 32) return -1;
        if (ch == '\\') {
            if (c->p == c->end) return -1;
            ch = (unsigned char)*c->p++;
            switch (ch) {
                case '"': case '\\': case '/': break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                case 'u': {
                    unsigned v = 0;
                    for (int i = 0; i < 4; i++) {
                        if (c->p == c->end) return -1;
                        unsigned char h = (unsigned char)*c->p++;
                        int d = h >= '0' && h <= '9' ? h - '0' :
                                h >= 'a' && h <= 'f' ? h - 'a' + 10 :
                                h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
                        if (d < 0) return -1;
                        v = v * 16 + (unsigned)d;
                    }
                    /* Tensor names and descriptor keys in this model are ASCII. */
                    if (buf && (v == 0 || v > 127)) return -1;
                    ch = (unsigned char)v;
                    break;
                }
                default: return -1;
            }
        }
        if (buf) { if (n + 1 >= cap) return -1; buf[n++] = (char)ch; }
    }
    return -1;
}
/* Shapes and offsets are nonnegative int64, never strtol on an unterminated mapping. */
static int j_number(jctx *c, int64_t *out) {
    j_ws(c);
    if (c->p == c->end || *c->p < '0' || *c->p > '9') return -1;
    const char *start = c->p;
    int zero = *c->p == '0';
    int64_t v = 0;
    do {
        int d = *c->p++ - '0';
        if (v > (INT64_MAX - d) / 10) return -1;
        v = v * 10 + d;
    } while (c->p < c->end && *c->p >= '0' && *c->p <= '9');
    if (zero && c->p - start != 1) return -1;
    *out = v;
    return 0;
}
static int j_skip(jctx *c) {
    j_ws(c);
    if (c->p == c->end || c->depth >= 64) return -1;
    if (*c->p == '"') return j_string(c, NULL, 0);
    if (*c->p == '{' || *c->p == '[') {
        int object = *c->p++ == '{';
        char close = object ? '}' : ']';
        c->depth++;
        j_ws(c);
        if (c->p < c->end && *c->p == close) { c->p++; c->depth--; return 0; }
        for (;;) {
            if (object && (j_string(c, NULL, 0) || j_take(c, ':'))) return -1;
            if (j_skip(c)) return -1;
            j_ws(c);
            if (c->p < c->end && *c->p == close) { c->p++; c->depth--; return 0; }
            if (j_take(c, ',')) return -1;
        }
    }
    const char *lits[] = { "true", "false", "null" };
    for (int i = 0; i < 3; i++) {
        size_t n = strlen(lits[i]);
        if ((size_t)(c->end - c->p) >= n && !memcmp(c->p, lits[i], n)) { c->p += n; return 0; }
    }
    int64_t v;
    return j_number(c, &v);
}
static int j_array(jctx *c, int64_t *values, int cap, int *n) {
    *n = 0;
    if (j_take(c, '[')) return -1;
    j_ws(c);
    if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
    for (;;) {
        if (*n == cap || j_number(c, &values[(*n)++])) return -1;
        j_ws(c);
        if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
        if (j_take(c, ',')) return -1;
    }
}
static int j_tensor(jctx *c, st_tensor *t) {
    unsigned seen = 0;
    if (j_take(c, '{')) return -1;
    for (;;) {
        char key[64];
        if (j_string(c, key, sizeof key) || j_take(c, ':')) return -1;
        unsigned bit = !strcmp(key, "dtype") ? 1 : !strcmp(key, "shape") ? 2 :
                       !strcmp(key, "data_offsets") ? 4 : 0;
        if (bit && (seen & bit)) return -1;
        seen |= bit;
        if (bit == 1) {
            if (j_string(c, t->dtype, sizeof t->dtype)) return -1;
        } else if (bit == 2) {
            if (j_array(c, t->shape, ST_MAX_DIMS, &t->ndim)) return -1;
        } else if (bit == 4) {
            int n;
            int64_t offsets[2];
            if (j_array(c, offsets, 2, &n) || n != 2) return -1;
            t->begin = offsets[0]; t->end = offsets[1];
        } else if (j_skip(c)) return -1;
        j_ws(c);
        if (c->p < c->end && *c->p == '}') { c->p++; return seen == 7 ? 0 : -1; }
        if (j_take(c, ',')) return -1;
    }
}
static int is_bf16(const st_tensor *t) { return !strcmp(t->dtype, "BF16") || !strcmp(t->dtype, "bfloat16"); }
static int is_f32(const st_tensor *t) { return !strcmp(t->dtype, "F32") || !strcmp(t->dtype, "float32"); }
static int tensor_ok(const st_file *f, const st_tensor *t) {
    size_t esz = is_bf16(t) ? 2 : is_f32(t) ? 4 : 0;
    size_t bytes = esz, dataoff = (size_t)(f->data - f->base);
    if (!esz || t->begin < 0 || t->end < t->begin || (uint64_t)t->end > f->size - dataoff ||
        (dataoff + (uint64_t)t->begin) % esz) return 0;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] < 0 || (t->shape[d] && bytes > SIZE_MAX / (uint64_t)t->shape[d])) return 0;
        bytes *= (size_t)t->shape[d];
    }
    return bytes == (uint64_t)(t->end - t->begin);
}

int st_open(st_file *f, const char *path) {
    memset(f, 0, sizeof *f);
    g_err[0] = 0;
    f->fd = open(path, O_RDONLY);
    struct stat sb;
    if (f->fd < 0 || fstat(f->fd, &sb) || sb.st_size < 8 || (uint64_t)sb.st_size > SIZE_MAX) goto bad;
    f->size = (size_t)sb.st_size;
    void *base = mmap(NULL, f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (base == MAP_FAILED) goto bad;
    f->base = base;
    uint64_t hlen;
    memcpy(&hlen, f->base, 8);
    if (hlen > f->size - 8 || hlen > ST_MAX_HEADER) goto bad;
    f->data = f->base + 8 + hlen;
    f->tensors = calloc(ST_MAX_TENSORS, sizeof(st_tensor));
    if (!f->tensors) goto bad;
    jctx c = { (const char *)f->base + 8, (const char *)f->data, 0 };
    if (j_take(&c, '{')) goto bad;
    j_ws(&c);
    if (c.p < c.end && *c.p == '}') c.p++;
    else for (;;) {
        char key[ST_MAX_NAME];
        if (j_string(&c, key, sizeof key) || !key[0] || j_take(&c, ':')) goto bad;
        if (!strcmp(key, "__metadata__")) {
            if (j_skip(&c)) goto bad;
        } else {
            if (f->count == ST_MAX_TENSORS) goto bad;
            st_tensor *t = &f->tensors[f->count];
            memcpy(t->name, key, strlen(key) + 1);
            if (j_tensor(&c, t) || !tensor_ok(f, t)) goto bad;
            for (size_t i = 0; i < f->count; i++) {
                const st_tensor *prev = &f->tensors[i];
                if (!strcmp(prev->name, t->name) ||
                    (prev->begin < t->end && t->begin < prev->end)) goto bad;
            }
            f->count++;
        }
        j_ws(&c);
        if (c.p < c.end && *c.p == '}') { c.p++; break; }
        if (j_take(&c, ',')) goto bad;
    }
    j_ws(&c);
    if (c.p != c.end) goto bad;
    return 0;
bad:
    snprintf(g_err, sizeof g_err, "%s: invalid, truncated or unreadable safetensors file", path);
    st_close(f);
    return -1;
}
void st_close(st_file *f) {
    free(f->tensors);
    if (f->base) munmap((void *)f->base, f->size);
    if (f->fd >= 0) close(f->fd);
    memset(f, 0, sizeof *f);
    f->fd = -1;
}
const st_tensor *st_find(const st_file *f, const char *name) {
    for (size_t i = 0; i < f->count; i++) if (!strcmp(f->tensors[i].name, name)) return &f->tensors[i];
    return NULL;
}
size_t st_numel(const st_tensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= (size_t)t->shape[i];
    return n;
}
const uint16_t *st_bf16(const st_file *f, const st_tensor *t) {
    return t && is_bf16(t) ? (const uint16_t *)(f->data + t->begin) : NULL;
}
const float *st_f32(const st_file *f, const st_tensor *t) {
    return t && is_f32(t) ? (const float *)(f->data + t->begin) : NULL;
}
int st_is_bf16(const st_tensor *t) { return is_bf16(t); }
