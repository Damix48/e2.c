/* e2hip.h — EmbeddingGemma 2 on an AMD GPU through ROCm/HIP.
 *
 * The AMD sibling of e2gpu.h (the Intel Level Zero backend): the same int8 recipe, the same model
 * directory, the same fixtures, but the kernels are HIP and the instruction the GEMM is built on is
 * RDNA's V_DOT4 (four signed int8 MACs into an int32 accumulator), not XMX/DPAS.
 *
 *     e2_model *m = e2_load(dir);            // needs the int8 weights (the default)
 *     e2_hip *g = e2_hip_create(m);
 *     e2_hip_text_embed(g, ids, n, out768);
 *
 * The vision tower is here too: e2_hip_image_embed runs the Gemma4 encoder on the device and hands
 * its soft tokens to the same text-tower graph, with the embedding block supplied instead of
 * gathered from token ids (e2_hip_text_image_embed does the <|image|> scatter).
 *
 * Everything the forward pass needs is uploaded once at create time; the whole kernel sequence for
 * a shape is recorded into a hipGraph the first time that shape is used and replayed afterwards, so
 * a call is a pinned-memory copy plus one graph launch.
 *
 * The engine's int8 activations are stored *signed* here (the CPU/Intel paths store q+128 to feed
 * the unsigned-byte dp4a): V_DOT4 is signed x signed, so the +128 bias the other backends have to
 * subtract with the per-block weight sum never exists on this path. Same numbers either way.
 */
#ifndef E2_HIP_H
#define E2_HIP_H

#include <stdint.h>

#include "e2.h"

typedef struct e2_hip e2_hip;

/* Uploads the model to the AMD GPU. `m` must outlive `g` (the weights are read from it). Returns
 * NULL when there is no HIP device or the model has no int8 weights. */
e2_hip *e2_hip_create(e2_model *m);
void e2_hip_destroy(e2_hip *g);

/* Device name, for logging. */
const char *e2_hip_device(e2_hip *g);

/* One sequence of n_ids tokens (n_ids <= 2048) -> one 768-d unit vector. Returns 0 on success. */
int e2_hip_text_embed(e2_hip *g, const int32_t *ids, int n_ids, float *out768);

/* `batch` sequences of `seq_len` tokens (batch x seq_len, row-major); `valid` marks padding rows
 * and may be NULL for fully valid input (any nonzero byte keeps a row). Positions restart per
 * sequence. Writes batch x 768 normalized vectors; all-invalid sequences produce zero. */
int e2_hip_text_embed_batch(e2_hip *g, const int32_t *ids, const uint8_t *valid, int batch,
                            int seq_len, float *out);

/* -------- the vision tower (Gemma4 encoder), same fixture format as e2_image_embed
 *
 * The tower runs on the device and hands its soft tokens to the text tower, which is the same
 * recorded graph the text entry points use, with the embedding block supplied instead of gathered
 * from token ids. The host walks the position ids once to build the valid-patch gather and the
 * pooler's group lists; the tokenizer and the image preprocessing stay on the CPU side. */
int e2_hip_image_embed(e2_hip *g, const float *pixels, const int32_t *position_ids, int n_patches,
                       float *out768);
int e2_hip_text_image_embed(e2_hip *g, const int32_t *ids, int n_ids, const float *pixels,
                            const int32_t *position_ids, int n_patches, float *out768);
/* kernels recorded and wall time of the last vision pass (the soft-token half only) */
int e2_hip_vision_stats(e2_hip *g, long *kernels, double *last_ms);

/* Introspection: kernels recorded for the last shape, calls served, and the time of the last call. */
int e2_hip_stats(e2_hip *g, long *calls, long *kernels_last, double *last_ms);
const char *e2_hip_last_error(void);

#endif /* E2_HIP_H */
