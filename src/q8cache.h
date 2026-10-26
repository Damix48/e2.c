/* q8cache.h — packed model cache for the int8 path (see q8cache.c).
 *
 * `E2_Q8_WRITE=<file>` writes the quantized model the loader builds (the tensors that become int8,
 * plus the ones that stay bf16/f32: the token table, the vision position table and the norm scales).
 * `E2_Q8_READ=<file>` loads from that file instead, so the 1.4 GB bf16 checkpoint is not needed and
 * the load-time quantization never runs. One file, one entry per tensor, keyed by the safetensors
 * name it replaces — the loader itself is what names them, so the two sides cannot drift.
 */
#ifndef E2_Q8CACHE_H
#define E2_Q8CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "int8.h"

/* kinds of payload */
enum { Q8C_BF16 = 1, Q8C_F32 = 2, Q8C_Q8W = 3 };

/* ---- writer (streaming: the index is written last, so there is no size bookkeeping up front) */
int q8c_write_open(const char *path);
int q8c_begin(const char *name, int kind, int ndim, const int64_t *dims);
int q8c_bytes(const void *p, size_t n);
int q8c_write_close(void);
/* Keep a tensor's bytes aside and write them at close time; q8c_drop cancels one by name. The
 * loader fetches every int8 weight in bf16 first (to quantize it), so without this the cache would
 * carry all 1.1 GB of bf16 weights as well as the quantized ones. */
int q8c_defer(const char *name, int kind, int ndim, const int64_t *dims, const void *p, size_t n);
int q8c_drop(const char *name);
int q8c_put_q8w(const char *name, const e2_q8w *q);

/* ---- reader (mmap; the returned pointers live until q8c_read_close) */
int q8c_read_open(const char *path);
void q8c_read_close(void);
const void *q8c_get(const char *name, int kind, int ndim, const int64_t *dims, size_t *nbytes);
/* the kind stored under this name, or -1 (used to tell "quantized weight" from "missing tensor") */
int q8c_kind_of(const char *name);
/* Assigns pointers into the mapping; the e2_q8w is marked borrowed so e2_q8w_free does not free it. */
int q8c_get_q8w(const char *name, e2_q8w *q);
const char *q8c_error(void);

#endif /* E2_Q8CACHE_H */
