/* q8cache.c — packed model cache for the int8 path.
 *
 * Layout (little-endian, no external dependency):
 *
 *   offset 0:  u32 magic "E8CQ", u32 version, u32 n_entries, u32 index_offset
 *   offset 16: the tensor blobs, each padded to 16 bytes
 *   index_offset: n_entries x { char name[64]; u32 kind; u32 ndim; i64 dims[4]; u64 nbytes; u64 off }
 *
 * kind 3 is an e2_q8w payload, concatenated the way the kernel wants it: the quad-interleaved int8
 * weights, then the per-row scales, then the transposed per-block sums. dims carry N, K, nblocks and
 * Npad. Reading maps the file and points the e2_q8w straight into it (no copy), which is why
 * e2_q8w_free has to know a tensor is borrowed.
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "q8cache.h"

#define Q8C_MAGIC 0x51433845u /* "E8CQ" little-endian */
#define Q8C_VERSION 1u
#define Q8C_INDEX_OFF 16

enum { Q8C_NAMELEN = 96, Q8C_MAXDIM = 4 }; /* the longest safetensors name here is 68 chars */

typedef struct {
    char name[Q8C_NAMELEN];
    uint32_t kind, ndim;
    int64_t dims[Q8C_MAXDIM];
    uint64_t nbytes, off;
} q8c_entry;

static char g_err[256];
const char *q8c_error(void) { return g_err; }

/* ---------------------------------------------------------------- writer */
static FILE *w_f;
static char *w_path, *w_tmp;
static uint64_t w_off;
static q8c_entry *w_idx;
static size_t w_n, w_cap;
static q8c_entry *w_cur;

/* deferred blobs: recorded as they are fetched, written at close unless cancelled by q8c_drop */
typedef struct {
    char name[Q8C_NAMELEN];
    int kind, ndim;
    int64_t dims[Q8C_MAXDIM];
    const void *p;
    size_t n;
    int dropped;
} q8c_pend;
static q8c_pend *w_pend;
static size_t w_pend_n, w_pend_cap;

int q8c_write_open(const char *path) {
    if (w_f) { snprintf(g_err, sizeof g_err, "cache already open for writing"); return -1; }
    size_t n = strlen(path) + 1;
    w_path = malloc(n); w_tmp = malloc(n + 16);
    if (w_tmp) w_tmp[0] = 0;
    if (!w_path || !w_tmp) { snprintf(g_err, sizeof g_err, "oom: cache path"); q8c_write_abort(); return -1; }
    memcpy(w_path, path, n);
    snprintf(w_tmp, n + 16, "%s.tmp.XXXXXX", path);
    int fd = mkstemp(w_tmp);
    if (fd < 0 || !(w_f = fdopen(fd, "wb"))) {
        if (fd >= 0) close(fd);
        else w_tmp[0] = 0;
        snprintf(g_err, sizeof g_err, "cannot write %s", path);
        q8c_write_abort(); return -1;
    }
    const uint32_t hdr[4] = { Q8C_MAGIC, Q8C_VERSION, 0, 0 };
    if (fwrite(hdr, sizeof hdr, 1, w_f) != 1) { q8c_write_abort(); return -1; }
    w_off = Q8C_INDEX_OFF;
    return 0;
}

int q8c_begin(const char *name, int kind, int ndim, const int64_t *dims) {
    if (!w_f) return -1;
    for (size_t i = 0; i < w_n; i++) /* one record per tensor: a retry must not append a second */
        if (!strcmp(w_idx[i].name, name)) {
            snprintf(g_err, sizeof g_err, "duplicate tensor in the cache: %s", name);
            return -1;
        }
    if (w_n == w_cap) {
        const size_t cap = w_cap ? w_cap * 2 : 512;
        q8c_entry *n = realloc(w_idx, cap * sizeof *n);
        if (!n) { snprintf(g_err, sizeof g_err, "oom"); return -1; }
        w_idx = n;
        w_cap = cap;
    }
    q8c_entry *e = &w_idx[w_n];
    memset(e, 0, sizeof *e);
    if (snprintf(e->name, sizeof e->name, "%s", name) >= (int)sizeof e->name) {
        snprintf(g_err, sizeof g_err, "tensor name too long for the cache: %s", name);
        return -1;
    }
    e->kind = (uint32_t)kind;
    e->ndim = (uint32_t)ndim;
    for (int i = 0; i < ndim && i < Q8C_MAXDIM; i++) e->dims[i] = dims[i];
    e->off = w_off;
    w_cur = e;
    return 0;
}

int q8c_bytes(const void *p, size_t n) {
    if (!w_f || !w_cur) return -1;
    if (n && fwrite(p, 1, n, w_f) != n) { snprintf(g_err, sizeof g_err, "short write"); return -1; }
    w_cur->nbytes += n;
    w_off += n;
    return 0;
}

int q8c_defer(const char *name, int kind, int ndim, const int64_t *dims, const void *p, size_t n) {
    if (!w_f) return -1;
    if (w_pend_n == w_pend_cap) {
        const size_t cap = w_pend_cap ? w_pend_cap * 2 : 512;
        q8c_pend *q = realloc(w_pend, cap * sizeof *q);
        if (!q) { snprintf(g_err, sizeof g_err, "oom"); return -1; }
        w_pend = q;
        w_pend_cap = cap;
    }
    q8c_pend *e = &w_pend[w_pend_n++];
    if (snprintf(e->name, sizeof e->name, "%s", name) >= (int)sizeof e->name) {
        snprintf(g_err, sizeof g_err, "tensor name too long for the cache: %s", name);
        w_pend_n--;
        return -1;
    }
    e->kind = kind;
    e->ndim = ndim;
    for (int i = 0; i < ndim && i < Q8C_MAXDIM; i++) e->dims[i] = dims[i];
    e->p = p;
    e->n = n;
    e->dropped = 0;
    return 0;
}

int q8c_drop(const char *name) {
    for (size_t i = 0; i < w_pend_n; i++)
        if (!w_pend[i].dropped && !strcmp(w_pend[i].name, name)) { w_pend[i].dropped = 1; return 0; }
    return -1;
}

/* one blob plus its 16-byte alignment padding */
static int w_blob(const char *name, int kind, int ndim, const int64_t *dims, const void *p, size_t n) {
    if (q8c_begin(name, kind, ndim, dims)) return -1;
    if (q8c_bytes(p, n)) return -1;
    w_n++;
    w_cur = NULL;
    const size_t pad = (16 - (size_t)(w_off & 15)) & 15;
    if (pad) {
        static const char zeros[16] = { 0 };
        if (fwrite(zeros, 1, pad, w_f) != pad) return -1;
        w_off += pad;
    }
    return 0;
}

int q8c_put_q8w(const char *name, const e2_q8w *q) {
    const int64_t dims[4] = { q->N, q->K, q->nblocks, q->Npad };
    const size_t kq = (size_t)((q->K + 3) / 4);
    const size_t nw = (size_t)q->Npad * kq * 4;
    const size_t ns = (size_t)q->Npad * sizeof(float);
    const size_t nb = (size_t)q->nblocks * q->Npad * sizeof(int32_t);
    const uint8_t *p8 = (const uint8_t *)q->w;
    if (q8c_begin(name, Q8C_Q8W, 4, dims)) return -1;
    if (q8c_bytes(p8, nw) || q8c_bytes(q->scale, ns) || q8c_bytes(q->bsq, nb)) return -1;
    w_n++;
    w_cur = NULL;
    const size_t pad = (16 - (size_t)(w_off & 15)) & 15;
    if (pad) {
        static const char zeros[16] = { 0 };
        if (fwrite(zeros, 1, pad, w_f) != pad) return -1;
        w_off += pad;
    }
    return 0;
}

int q8c_write_close(void) {
    if (!w_f) return 0;
    int rc = 0;
    for (size_t i = 0; i < w_pend_n; i++) /* the tensors that stayed bf16/f32 */
        if (!w_pend[i].dropped &&
            w_blob(w_pend[i].name, w_pend[i].kind, w_pend[i].ndim, w_pend[i].dims, w_pend[i].p,
                   w_pend[i].n)) {
            rc = -1; /* a partial file is not worth an index, but the writer state is still ours */
            break;
        }
    free(w_pend);
    w_pend = NULL;
    w_pend_n = w_pend_cap = 0;
    if (w_off > UINT32_MAX || w_n > UINT32_MAX) {
        snprintf(g_err, sizeof g_err, "cache exceeds the v1 32-bit index format");
        rc = -1;
    }
    if (!rc) {
        const uint32_t hdr[4] = { Q8C_MAGIC, Q8C_VERSION, (uint32_t)w_n, (uint32_t)w_off };
        if (w_n && fwrite(w_idx, sizeof *w_idx, w_n, w_f) != w_n) rc = -1;
        if (fseek(w_f, 0, SEEK_SET) || fwrite(hdr, sizeof hdr, 1, w_f) != 1) rc = -1;
    }
    if (fflush(w_f) || fsync(fileno(w_f))) rc = -1;
    if (fclose(w_f)) rc = -1;
    w_f = NULL;
    if (!rc && rename(w_tmp, w_path)) rc = -1;
    if (rc) snprintf(g_err, sizeof g_err, "cache write/commit failed");
    q8c_write_abort();
    return rc;
}

void q8c_write_abort(void) {
    if (w_f) fclose(w_f);
    w_f = NULL;
    if (w_tmp && w_tmp[0]) unlink(w_tmp);
    free(w_tmp); free(w_path);
    w_tmp = w_path = NULL;
    free(w_pend); w_pend = NULL; w_pend_n = w_pend_cap = 0;
    free(w_idx); w_idx = NULL; w_n = w_cap = 0; w_cur = NULL;
}

/* ---------------------------------------------------------------- reader */
static int r_fd = -1;
static const uint8_t *r_base;
static size_t r_size;
static const q8c_entry *r_idx;
static uint32_t r_n;

static int r_err(const char *path, const char *name, const char *why) {
    if (name) snprintf(g_err, sizeof g_err, "%s: %.64s: %s", path, name, why);
    else snprintf(g_err, sizeof g_err, "%s: %s", path, why);
    return -1;
}

/* Every entry is checked once, before anything dereferences it: the name has to be terminated inside
 * its field (it is used as a string), the blob has to sit inside the data area before the index and
 * be 16-byte padded so its float/int32 arrays are aligned, and a q8w payload has to be packed exactly
 * the way the compiled kernels expect (E2_I8_BLOCK, Npad rounded to 8). Without the last check a
 * cache written with another block size loads cleanly and computes nonsense. */
static int r_entry_ok(const q8c_entry *e, uint64_t idx_off, const char *path) {
    if (!memchr(e->name, 0, sizeof e->name) || !e->name[0])
        return r_err(path, NULL, "entry with an unterminated or empty name");
    if (e->kind != Q8C_BF16 && e->kind != Q8C_F32 && e->kind != Q8C_Q8W)
        return r_err(path, e->name, "unknown kind");
    if (e->ndim > Q8C_MAXDIM || (e->kind == Q8C_Q8W && e->ndim != 4))
        return r_err(path, e->name, "bad rank");
    for (uint32_t d = 0; d < e->ndim; d++)
        if (e->dims[d] < 0) return r_err(path, e->name, "negative dimension");
    if (e->off < Q8C_INDEX_OFF || (e->off & 15) || e->off > idx_off || e->nbytes > idx_off - e->off)
        return r_err(path, e->name, "blob outside the data area");
    if (e->kind != Q8C_Q8W) {
        /* a bf16/f32 blob has to be exactly its element count (a 0-d tensor is one element): the
         * loader copies/points at nbytes worth of it, so a shorter blob would be read past its end */
        const uint64_t esz = e->kind == Q8C_BF16 ? 2 : 4;
        uint64_t want = esz;
        for (uint32_t d = 0; d < e->ndim; d++) {
            if (e->dims[d] == 0) { want = 0; continue; } /* a zero dimension: an empty tensor */
            if (want > UINT64_MAX / (uint64_t)e->dims[d])
                return r_err(path, e->name, "element count overflows");
            want *= (uint64_t)e->dims[d];
        }
        if (e->nbytes != want) return r_err(path, e->name, "blob size is not its element count");
        return 0;
    }
    const int64_t N = e->dims[0], K = e->dims[1];
    if (N <= 0 || K <= 0 || N > INT_MAX || K > INT_MAX) return r_err(path, e->name, "bad shape");
    const int64_t nblocks = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    const int64_t Npad = (N + 7) & ~(int64_t)7;
    if (Npad > INT_MAX) return r_err(path, e->name, "padded shape exceeds int range");
    if (e->dims[2] != nblocks || e->dims[3] != Npad)
        return r_err(path, e->name, "packed for another E2_I8_BLOCK/Npad");
    /* the three packed arrays, in the order q8c_put_q8w writes them */
    const uint64_t want = (uint64_t)Npad * (uint64_t)((K + 3) / 4) * 4 +
                          (uint64_t)Npad * sizeof(float) +
                          (uint64_t)nblocks * (uint64_t)Npad * sizeof(int32_t);
    if (e->nbytes != want) return r_err(path, e->name, "blob size mismatch");
    return 0;
}

int q8c_read_open(const char *path) {
    if (r_base) return r_err(path, NULL, "already open");
    r_fd = open(path, O_RDONLY);
    if (r_fd < 0) return r_err(path, NULL, "cannot open");
    struct stat st;
    if (fstat(r_fd, &st) || st.st_size < Q8C_INDEX_OFF) { r_err(path, NULL, "too small"); goto fail; }
    r_size = (size_t)st.st_size;
    void *m = mmap(NULL, r_size, PROT_READ, MAP_PRIVATE, r_fd, 0);
    if (m == MAP_FAILED) { r_err(path, NULL, "mmap failed"); goto fail; }
    r_base = m;
    const uint32_t *hdr = (const uint32_t *)r_base;
    if (hdr[0] != Q8C_MAGIC || hdr[1] != Q8C_VERSION) { r_err(path, NULL, "bad magic/version"); goto fail; }
    r_n = hdr[2];
    const uint64_t idx = hdr[3];
    /* the index is a struct array cast out of the mapping, so its offset must be 16-aligned (the
     * writer puts it after the last 16-byte-padded blob) and the whole array has to fit */
    if (idx < Q8C_INDEX_OFF || (idx & 15) || idx > r_size ||
        (uint64_t)r_n > (r_size - idx) / sizeof(q8c_entry)) {
        r_err(path, NULL, "truncated or misaligned index");
        goto fail;
    }
    r_idx = (const q8c_entry *)(r_base + idx);
    /* q8c_get and q8c_get_q8w rely on this pass: after it, every entry is safe to read and its blob
     * lies inside the mapping */
    for (uint32_t i = 0; i < r_n; i++)
        if (r_entry_ok(&r_idx[i], idx, path)) goto fail;
    return 0;
fail:
    q8c_read_close();
    return -1;
}

void q8c_read_close(void) {
    if (r_base) munmap((void *)r_base, r_size);
    r_base = NULL;
    if (r_fd >= 0) close(r_fd);
    r_fd = -1;
    r_idx = NULL;
    r_n = 0;
    r_size = 0;
}

const void *q8c_get(const char *name, int kind, int ndim, const int64_t *dims, size_t *nbytes) {
    for (uint32_t i = 0; i < r_n; i++) {
        const q8c_entry *e = &r_idx[i];
        if (strcmp(e->name, name)) continue;
        if ((int)e->kind != kind || (int)e->ndim != ndim) {
            snprintf(g_err, sizeof g_err, "%s: cache kind/rank mismatch", name);
            return NULL;
        }
        for (int d = 0; d < ndim; d++) {
            if (dims[d] > 0 && (int64_t)e->dims[d] != dims[d]) {
                snprintf(g_err, sizeof g_err, "%s: cache dim %d is %lld", name, d, (long long)e->dims[d]);
                return NULL;
            }
        }
        /* q8c_read_open validated the entry, so this offset and size are inside the mapping */
        if (nbytes) *nbytes = (size_t)e->nbytes;
        return r_base + e->off;
    }
    snprintf(g_err, sizeof g_err, "%s: not in cache", name);
    return NULL;
}

int q8c_kind_of(const char *name) {
    for (uint32_t i = 0; i < r_n; i++)
        if (!strcmp(r_idx[i].name, name)) return (int)r_idx[i].kind;
    return -1;
}

int q8c_get_q8w(const char *name, e2_q8w *q) {
    const int64_t dims[4] = { 0, 0, 0, 0 };
    const uint8_t *p = q8c_get(name, Q8C_Q8W, 4, dims, NULL);
    if (!p) return -1;
    /* re-read the dims this entry carries (q8c_get only validated them); q8c_read_open checked that
     * they match the compiled packing and the blob size, so the three arrays are in bounds */
    for (uint32_t i = 0; i < r_n; i++) {
        if (!strcmp(r_idx[i].name, name)) {
            q->N = (int)r_idx[i].dims[0];
            q->K = (int)r_idx[i].dims[1];
            q->nblocks = (int)r_idx[i].dims[2];
            q->Npad = (int)r_idx[i].dims[3];
            break;
        }
    }
    const size_t kq = (size_t)((q->K + 3) / 4);
    const size_t nw = (size_t)q->Npad * kq * 4;
    const size_t ns = (size_t)q->Npad * sizeof(float);
    q->w = (int8_t *)(uintptr_t)p;
    q->scale = (float *)(uintptr_t)(p + nw);
    q->bsq = (int32_t *)(uintptr_t)(p + nw + ns);
    /* the cache carries the packed arrays, not the saturation bounds (which come from the
     * checkpoint's own 0-d clamp tensors): default to unclamped here, and a clipped linear's loader
     * overwrites them */
    q->out_min = -1e31f;
    q->out_max = 1e31f;
    q->borrowed = 1;
    return 0;
}
