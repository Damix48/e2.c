/* e2gpu.h — EmbeddingGemma 2 on the Intel GPU (Level Zero backend).
 *
 * Projection weights and activation quantization match the CPU engine; GPU attention is additionally
 * quantized, so fidelity can differ. Experimental: strict device checks still expose intermittent
 * wrong outputs (see README). Scope: the text tower (single or padded batch), the vision
 * tower, and the audio Conformer; the tokenizer, image preprocessing and audio front end stay on the
 * CPU side (they are cheap and serial).
 *
 *     e2_model *m = e2_load(dir);            // needs the int8 weights (the default)
 *     e2_gpu *g = e2_gpu_create(m);
 *     e2_gpu_text_embed(g, ids, n, out768);
 *
 * Everything the GPU needs is uploaded once at create time; a call uploads its token ids and reads
 * back one or more 768-d vectors. The whole forward pass is recorded as a command list the first time
 * a shape is used and replayed afterwards, so per-call host work is a copy plus one submission.
 */
#ifndef E2_GPU_H
#define E2_GPU_H

#include <stdint.h>

#include "e2.h"

typedef struct e2_gpu e2_gpu;

/* Uploads the model to the GPU. `m` must outlive `g` (the weights are read from it). Returns NULL if
 * there is no Intel GPU / Level Zero runtime, or the model has no int8 weights. */
e2_gpu *e2_gpu_create(e2_model *m);
void e2_gpu_destroy(e2_gpu *g);

/* Device name, for logging. */
const char *e2_gpu_device(e2_gpu *g);

/* One sequence of n_ids tokens (n_ids <= 2048) -> one 768-d unit vector. Returns 0 on success. */
int e2_gpu_text_embed(e2_gpu *g, const int32_t *ids, int n_ids, float *out768);

/* `batch` sequences of `seq_len` tokens, ids stored as batch x seq_len (pad with any valid id and
 * mark the padding rows in `valid`, which may be NULL for fully valid input). Writes batch x 768
 * normalized vectors (all-invalid sequences produce zero). Positions restart per sequence; any
 * nonzero valid byte keeps its row. Batches exceeding one pass are chunked by whole sequences. */
int e2_gpu_text_embed_batch(e2_gpu *g, const int32_t *ids, const uint8_t *valid, int batch,
                            int seq_len, float *out);

/* -------- the vision tower (Gemma4 encoder), same fixture format as e2_image_embed
 *
 * The tower runs on the device and hands its soft tokens to the text tower, which is the same
 * recorded graph the text entry points use, with the embedding block supplied instead of gathered
 * from token ids. The host still walks the position ids to build the valid-patch gather and the
 * pooler's group lists; the tokenizer and the image preprocessing stay on the CPU side. */
int e2_gpu_image_embed(e2_gpu *g, const float *pixels, const int32_t *position_ids, int n_patches,
                       float *out768);
int e2_gpu_text_image_embed(e2_gpu *g, const int32_t *ids, int n_ids, const float *pixels,
                            const int32_t *position_ids, int n_patches, float *out768);
/* kernels recorded and wall time of the last vision pass (the soft-token half only) */
int e2_gpu_vision_stats(e2_gpu *g, long *kernels, double *last_ms);

/* -------- the audio tower (Gemma4 Conformer). The mel front end and the two subsampling convolutions
 * stay on the CPU (0.4 ms and unchanged from the CPU engine); the twelve Conformer layers run on the
 * device and produce the [n_soft, 512] block the text tower consumes. `soft` must hold
 * E2_A_MAX_TOK * 512 floats; n_soft is one token per 40 ms of input. */
int e2_gpu_audio_soft(e2_gpu *g, const float *pcm, int n_samples, float *soft, int *n_soft);
/* the audio-only embedding: <bos> <|audio> N soft <audio|> <eos> through the GPU text tower */
int e2_gpu_audio_embed(e2_gpu *g, const float *pcm, int n_samples, float *out768);
/* kernels recorded and wall time of the last audio pass (tower only, front end excluded) */
int e2_gpu_audio_stats(e2_gpu *g, long *kernels, double *last_ms);

/* Introspection: kernels recorded for the last shape, calls served, and the time of the last call. */
int e2_gpu_stats(e2_gpu *g, long *calls, long *kernels_last, double *last_ms);
const char *e2_gpu_last_error(void);

#endif /* E2_GPU_H */
