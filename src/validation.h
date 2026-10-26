/* validation.h — input-boundary checks shared by the vision and audio entry points.
 *
 * Header-only and static inline because the backends report errors through different sinks
 * (e2_set_error on the CPU, one g_err per GPU backend); the caller stores the returned message.
 */
#ifndef E2_VALIDATION_H
#define E2_VALIDATION_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "model.h"

/* Implemented in validation.c, compiled without fast-math. */
int e2_valid_floats(const float *p, size_t n, float lo, float hi);
int e2_valid_embedding(const float *p);

/* Ids index the embedding table directly, and a batched call gathers its padding rows too, so every
 * id in the block must be a vocabulary entry ("pad with any valid id"). */
static inline int e2_valid_ids(const int32_t *ids, size_t n_ids, char *err, size_t errcap) {
    if (!ids) {
        snprintf(err, errcap, "token ids pointer is NULL");
        return -1;
    }
    for (size_t i = 0; i < n_ids; i++)
        if (ids[i] < 0 || ids[i] >= E2_VOCAB) {
            snprintf(err, errcap, "token id %d at position %zu is out of range [0,%d)", (int)ids[i], i,
                     E2_VOCAB);
            return -1;
        }
    return 0;
}

/* The pooler's grid for a patch layout whose largest position is (max_x, max_y). */
static inline void e2_pool_grid(int max_x, int max_y, int *cols, int *rows) {
    *cols = max_x / E2_V_POOL + 1;
    *rows = max_y / E2_V_POOL + 1;
}

/* Positions are grid cells right-padded with (-1,-1). They bound the axial table the patch embedder
 * reads and the pool grid the pooler writes, before either happens; a pair that is neither a cell
 * nor exactly (-1,-1) is malformed and rejected rather than coerced. cols/rows may be NULL.
 * n_patches must already be range-checked. */
static inline int e2_valid_positions(const int32_t *pos, int n_patches, int max_groups, int *cols,
                                    int *rows, char *err, size_t errcap) {
    if (!pos) {
        snprintf(err, errcap, "position ids pointer is NULL");
        return -1;
    }
    int max_x = -1, max_y = -1;
    for (int i = 0; i < n_patches; i++) {
        const int x = pos[2 * i], y = pos[2 * i + 1];
        if (x == -1 && y == -1) continue; /* padding */
        if (x < 0 || y < 0) {
            snprintf(err, errcap, "patch %d has position (%d,%d): expected a cell or (-1,-1)", i, x, y);
            return -1;
        }
        if (x >= E2_V_POS || y >= E2_V_POS) {
            snprintf(err, errcap, "patch %d has position (%d,%d) outside the %d-wide position table", i,
                     x, y, E2_V_POS);
            return -1;
        }
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
    }
    if (max_x < 0) {
        snprintf(err, errcap, "the image has no valid patches");
        return -1;
    }
    int gc = 0, gr = 0;
    e2_pool_grid(max_x, max_y, &gc, &gr);
    /* Both factors are below E2_V_POS + 1, so the 64-bit product cannot overflow. */
    const int64_t groups = (int64_t)gc * gr;
    if (groups > max_groups) {
        snprintf(err, errcap, "pool grid %dx%d = %lld groups exceeds the %d rows available", gc, gr,
                 (long long)groups, max_groups);
        return -1;
    }
    if (cols) *cols = gc;
    if (rows) *rows = gr;
    return 0;
}

#endif /* E2_VALIDATION_H */
