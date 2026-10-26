/* Safetensors reader: mmap the checkpoint, parse its JSON header, expose tensors as bf16/f32 views.
 *
 * The header is a small JSON object mapping tensor name -> {dtype, shape, data_offsets}. Rather than
 * pulling in a JSON library, this file contains a minimal parser for exactly that structure (it
 * skips any value it does not need, so extra metadata is harmless).
 */
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

static char g_err[512];

const char *st_error(void) { return g_err; }

static void st_seterr(const char *fmt, const char *a) {
    snprintf(g_err, sizeof g_err, fmt, a ? a : "");
}

/* ---------------------------------------------------------------- JSON mini-parser */
typedef struct {
    const char *p, *end;
    int depth;
} jctx;

static void j_ws(jctx *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++;
}

/* Copy a JSON string (no escapes needed for tensor names) into buf, advancing the cursor. */
static int j_string(jctx *c, char *buf, size_t n) {
    size_t i = 0;
    j_ws(c);
    if (c->p >= c->end || *c->p != '"') return -1;
    c->p++;
    while (c->p < c->end && *c->p != '"') {
        if (*c->p == '\\') { c->p++; if (c->p >= c->end) return -1; }
        if (i + 1 < n) buf[i++] = *c->p;
        c->p++;
    }
    if (c->p >= c->end) return -1;
    c->p++;  /* closing quote */
    buf[i] = 0;
    return 0;
}

static int j_number(jctx *c, long *out) {
    char *end;
    j_ws(c);
    *out = strtol(c->p, &end, 10);
    if (end == c->p) return -1;
    c->p = end;
    return 0;
}

static int j_skip_value(jctx *c);

static int j_skip_container(jctx *c, char open, char close) {
    if (*c->p != open) return -1;
    c->p++;
    for (;;) {
        j_ws(c);
        if (c->p >= c->end) return -1;
        if (*c->p == close) { c->p++; return 0; }
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '"') { char tmp[8]; if (j_string(c, tmp, sizeof tmp)) return -1; continue; }
        j_ws(c);
        if (c->p < c->end && (*c->p == ':' || *c->p == '{' || *c->p == '[')) {
            if (*c->p == ':') { c->p++; continue; }
            if (j_skip_value(c)) return -1;
            continue;
        }
        if (j_skip_value(c)) return -1;
    }
}

static int j_skip_value(jctx *c) {
    j_ws(c);
    if (c->p >= c->end) return -1;
    if (*c->p == '"') { char tmp[8]; return j_string(c, tmp, sizeof tmp); }
    if (*c->p == '{') return j_skip_container(c, '{', '}');
    if (*c->p == '[') return j_skip_container(c, '[', ']');
    long dummy;
    return j_number(c, &dummy);
}

/* Parse one tensor descriptor: {"dtype":"BF16","shape":[...],"data_offsets":[b,e]} */
static int j_tensor(jctx *c, st_tensor *t) {
    if (*c->p != '{') return -1;
    c->p++;
    t->dtype[0] = 0;
    t->ndim = 0;
    t->begin = t->end = 0;
    for (;;) {
        char key[64];
        j_ws(c);
        if (c->p >= c->end) return -1;
        if (*c->p == '}') { c->p++; return t->dtype[0] ? 0 : -1; }
        if (*c->p == ',') { c->p++; continue; }
        if (j_string(c, key, sizeof key)) return -1;
        j_ws(c);
        if (c->p >= c->end || *c->p != ':') return -1;
        c->p++;
        j_ws(c);
        if (!strcmp(key, "dtype")) {
            if (j_string(c, t->dtype, sizeof t->dtype)) return -1;
        } else if (!strcmp(key, "shape")) {
            if (*c->p != '[') return -1;
            c->p++;
            for (;;) {
                j_ws(c);
                if (*c->p == ']') { c->p++; break; }
                if (*c->p == ',') { c->p++; continue; }
                long v;
                if (j_number(c, &v) || t->ndim >= ST_MAX_DIMS) return -1;
                t->shape[t->ndim++] = (int64_t)v;
            }
        } else if (!strcmp(key, "data_offsets")) {
            if (*c->p != '[') return -1;
            c->p++;
            long v;
            for (int i = 0; i < 2; i++) {
                j_ws(c);
                if (*c->p == ',') { c->p++; j_ws(c); }
                if (j_number(c, &v)) return -1;
                if (i == 0) t->begin = (int64_t)v; else t->end = (int64_t)v;
            }
            j_ws(c);
            if (*c->p != ']') return -1;
            c->p++;
        } else {
            if (j_skip_value(c)) return -1;
        }
    }
}

/* ---------------------------------------------------------------- public API */
int st_open(st_file *f, const char *path) {
    memset(f, 0, sizeof *f);
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) { st_seterr("cannot open %s", path); return -1; }
    struct stat sb;
    if (fstat(f->fd, &sb)) { st_seterr("fstat failed on %s", path); return -1; }
    f->size = (size_t)sb.st_size;
    void *base = mmap(NULL, f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (base == MAP_FAILED) { st_seterr("mmap failed on %s", path); return -1; }
    f->base = (const uint8_t *)base;

    if (f->size < 8) { st_seterr("%s too small", path); return -1; }
    uint64_t hlen;
    memcpy(&hlen, f->base, 8);
    if (8 + hlen > f->size) { st_seterr("%s: bad header length", path); return -1; }
    f->data = f->base + 8 + hlen;

    f->tensors = calloc(ST_MAX_TENSORS, sizeof(st_tensor));
    if (!f->tensors) { st_seterr("oom", NULL); return -1; }

    jctx c = { (const char *)(f->base + 8), (const char *)(f->base + 8 + hlen), 0 };
    j_ws(&c);
    if (c.p >= c.end || *c.p != '{') { st_seterr("%s: header is not an object", path); return -1; }
    c.p++;
    for (;;) {
        char key[192];
        j_ws(&c);
        if (c.p >= c.end) { st_seterr("%s: truncated header", path); return -1; }
        if (*c.p == '}') break;
        if (*c.p == ',') { c.p++; continue; }
        if (j_string(&c, key, sizeof key)) { st_seterr("%s: bad header key", path); return -1; }
        j_ws(&c);
        if (c.p >= c.end || *c.p != ':') { st_seterr("%s: bad header", path); return -1; }
        c.p++;
        j_ws(&c);
        if (*c.p != '{') {  /* __metadata__ or anything else: skip */
            if (j_skip_value(&c)) { st_seterr("%s: bad header value", path); return -1; }
            continue;
        }
        if (f->count >= ST_MAX_TENSORS) { st_seterr("too many tensors", NULL); return -1; }
        st_tensor *t = &f->tensors[f->count];
        if (j_tensor(&c, t) == 0) {
            snprintf(t->name, sizeof t->name, "%s", key);
            f->count++;
        }
    }
    return 0;
}

void st_close(st_file *f) {
    if (f->tensors) free(f->tensors);
    if (f->base && f->base != MAP_FAILED) munmap((void *)f->base, f->size);
    if (f->fd >= 0) close(f->fd);
    memset(f, 0, sizeof *f);
    f->fd = -1;
}

const st_tensor *st_find(const st_file *f, const char *name) {
    for (size_t i = 0; i < f->count; i++)
        if (!strcmp(f->tensors[i].name, name)) return &f->tensors[i];
    return NULL;
}

size_t st_numel(const st_tensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= (size_t)t->shape[i];
    return n;
}

static int is_bf16(const st_tensor *t) { return !strcmp(t->dtype, "BF16") || !strcmp(t->dtype, "bfloat16"); }
static int is_f32(const st_tensor *t) { return !strcmp(t->dtype, "F32") || !strcmp(t->dtype, "float32"); }

/* bf16 view of a tensor: bf16 storage is the same byte layout, so this is just a typed pointer.
 * F32 tensors are reported with bf16==NULL; kernels take an explicit dtype. */
const uint16_t *st_bf16(const st_file *f, const st_tensor *t) {
    if (!t || !is_bf16(t)) return NULL;
    return (const uint16_t *)(f->data + t->begin);
}

const float *st_f32(const st_file *f, const st_tensor *t) {
    if (!t) return NULL;
    if (is_f32(t)) return (const float *)(f->data + t->begin);
    return NULL;  /* bf16 tensors are read through st_bf16 + e2_bf16_to_f32 */
}

int st_is_bf16(const st_tensor *t) { return is_bf16(t); }
