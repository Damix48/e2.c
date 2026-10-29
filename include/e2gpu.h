/* e2gpu.h — EmbeddingGemma 2 on the Intel GPU (Level Zero backend).
 *
 * Projections use the CPU engine's int8 weights and per-128 activation quantization. Text attention
 * selects FP16/INT8 products by sequence length, with FP32 softmax; vision is quantized.
 * Fidelity can differ.
 * Experimental: execution guards contain missing
 * progress/invalid outputs but do not establish the cause of historical intermittent failures.
 * An experimental, default-off build (-DE2_GPU_NO_XMX=ON) targets Intel GPUs without XMX: DP4A
 * projections and an f32 text attention fallback, text only. It is not the default backend and is
 * slower than the CPU on the N100; see PERFORMANCE.md.
 * See PRODUCTION.md for the software gate and its limits. Scope: text (single or padded batch), vision,
 * tower, and the audio Conformer; the tokenizer, image preprocessing and audio front end stay on the
 * CPU side (they are cheap and serial).
 *
 *     e2_model *m = e2_load(dir);            // needs the int8 weights (the default)
 *     e2_gpu *g = e2_gpu_create(m);
 *     e2_gpu_text_embed(g, ids, n, out768);
 *
 * Present towers are uploaded once at create time; a call uploads its token ids and reads
 * back one or more 768-d vectors. The whole forward pass is recorded as a command list the first time
 * a shape is used and replayed afterwards, so per-call host work is a copy plus one submission.
 */
#ifndef E2_GPU_H
#define E2_GPU_H

#include <stdint.h>

#include "e2.h"

typedef struct e2_gpu e2_gpu;

/* Uploads the model's present towers (including text-only packed caches). `m` must outlive `g`.
 * Returns NULL on initialization/upload failure, including no runtime/device or no int8 weights.
 * One model per process; serialize all CPU/GPU/preprocessing calls. Destroy g before freeing m.
 *
 * Execution waits are bounded to 30 s each, not a whole-call deadline or cancellation guarantee.
 * A failed execution/completion/output check permanently retires g: discard the result and restart
 * the worker process, do not retry/recreate contexts in it. Potentially in-flight device resources
 * are intentionally retained until process exit; destroy cannot safely cancel that work.
 * Use a supervised process with an external deadline, since a driver API call itself may block. */
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
 * 750 * 512 floats; n_soft is one token per 40 ms of input. */
int e2_gpu_audio_soft(e2_gpu *g, const float *pcm, int n_samples, float *soft, int *n_soft);
/* the audio-only embedding: <bos> <|audio> N soft <audio|> <eos> through the GPU text tower */
int e2_gpu_audio_embed(e2_gpu *g, const float *pcm, int n_samples, float *out768);
/* kernels recorded and wall time of the last audio pass (tower only, front end excluded) */
int e2_gpu_audio_stats(e2_gpu *g, long *kernels, double *last_ms);

/* Last text graph's kernel count, number of text-graph executions, and last graph's wall time.
 * A batch can execute multiple graphs; these are not whole-batch latency/call counters. */
int e2_gpu_stats(e2_gpu *g, long *calls, long *kernels_last, double *last_ms);
/* Last fallible GPU call's error, cleared on success; free/info/stats leave it unchanged.
 * Input validation failures leave output untouched. On runtime failure discard the whole batch:
 * earlier chunks may already have been written. Nonempty outputs must be finite unit vectors;
 * all-masked rows must be exact zero. Soft-token outputs are checked for finite/fresh writes.
 * These guards cannot detect every finite-but-wrong result: quality acceptance still needs references. */
const char *e2_gpu_last_error(void);

#endif /* E2_GPU_H */
