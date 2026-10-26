/* e2.h — EmbeddingGemma 2 text+image embedding engine (CPU).
 *
 * Scope: text tower + vision tower + audio tower, all producing the model's 768-dim L2-normalized
 * embeddings, and the four input shapes that matter in practice: text, text+image, text+audio and
 * text+image+audio. Video is out of scope. The engine reads the original bf16 safetensors
 * checkpoint directly.
 *
 * Threading: inference uses a persistent worker pool. Calls are not reentrant: do not run load,
 * free, embedding or preprocessing calls concurrently. One live model per process; e2_load rejects
 * a second handle until e2_free releases the first. Destroy GPU/HIP contexts before freeing it.
 */
#ifndef E2_H
#define E2_H

#include <stddef.h>
#include <stdint.h>

typedef struct e2_model e2_model;

/* Load from a directory containing model.safetensors (mmap'd, lazily paged in). */
e2_model *e2_load(const char *model_dir);
void e2_free(e2_model *m);

/* Text: token ids -> 768d unit vector. Returns 0 on success. */
int e2_text_embed(e2_model *m, const int32_t *ids, int n_ids, float *out768);

/* CPU batch: ids/valid are row-major [batch, seq_len], out is [batch, 768].
 * valid may be NULL (all tokens valid); any nonzero byte keeps that token for attention/pooling.
 * Positions restart at zero per sequence, including padding positions (no compaction).
 * seq_len must be 1..2048 and batch positive; all IDs, even masked IDs, must be in vocabulary.
 * All-invalid sequences produce an exact zero vector; other sequences produce unit vectors.
 * Whole sequences are chunked into the existing 2048-row workspace, sharing projection work.
 * Returns 0 on success, -1 on invalid input, without writing output on validation failure.
 * Like single inference, uses model scratch and is not reentrant. */
int e2_text_embed_batch(e2_model *m, const int32_t *ids, const uint8_t *valid,
                        int batch, int seq_len, float *out);

/* Image: pre-patchified pixels -> 768d unit vector.
 * pixels: finite values in [0,1], (n_patches, patch_dim=768) row-major, 16x16x3 patches flattened by the HF image processor;
 * position_ids: (n_patches, 2) int32, (-1,-1) marks padding (stripped, as the model does). */
int e2_image_embed(e2_model *m, const float *pixels, const int32_t *position_ids, int n_patches, float *out768);

/* Audio: 16 kHz mono PCM in [-1,1], n_samples of it -> 768d unit vector. Runs the audio tower, then
 * the text tower over <bos> <|audio> <|audio|>xN <audio|> <eos>. Up to 30 s (750 soft tokens); the
 * sequence is not padded, so n_samples only has to be the real length (at least 161 samples).
 * Longer clips are truncated to the first 30 s, matching the reference processor. */
int e2_audio_embed(e2_model *m, const float *pcm, int n_samples, float *out768);

/* Text + audio: the text's `<|audio|>` placeholders already expanded to <|audio> + N soft slots +
 * <audio|>, exactly what e2_encode_mma builds. */
int e2_text_audio_embed(e2_model *m, const int32_t *ids, int n_ids, const float *pcm, int n_samples,
                        float *out768);

/* Text + image + audio in one sequence, each placeholder spliced in order of appearance. */
int e2_text_image_audio_embed(e2_model *m, const int32_t *ids, int n_ids, const float *pixels,
                              const int32_t *position_ids, int n_patches, const float *pcm,
                              int n_samples, float *out768);

/* Soft tokens a clip of n_samples at 16 kHz becomes, without running anything (one per 40 ms). */
int e2_audio_tokens(int n_samples);

/* Interleaved: image placeholders already expanded into <bos> <|image> <|image|>xN <image|> <eos>. */
int e2_text_image_embed(e2_model *m, const int32_t *ids, int n_ids,
                        const float *pixels, const int32_t *position_ids, int n_patches, float *out768);

/* ---------------------------------------------------------------- audio preprocessing
 * Decode file bytes into caller-owned PCM[E2_AUDIO_MAX_SAMPLES]. Returns the number of 16 kHz
 * mono samples, or -1 with e2_last_error(), leaving PCM untouched on failure. No weights needed.
 * Input: 8..192 kHz, 1..32 channels. Keep the first 30 s, clamp finite channel samples to [-1,1],
 * average channels, then band-limited resample (output length rounded up). NaN/Inf and clips shorter
 * than 161 output samples are rejected. Already-16-kHz mono samples are not filtered.
 * This resampler is not promised bit-identical to external HF/librosa/soxr preprocessing. */
#define E2_AUDIO_SAMPLE_RATE 16000
#define E2_AUDIO_MAX_SAMPLES 480000

/* Prefer libsndfile when compiled in; otherwise use the in-tree RIFF WAV reader. */
int e2_audio_preprocess(const uint8_t *data, size_t len, float *pcm);
/* Always use the in-tree reader: PCM 8/16/24/32-bit and IEEE float 32/64-bit RIFF WAV,
 * including WAVE_FORMAT_EXTENSIBLE. Compressed WAV/RIFX/RF64 need libsndfile. */
int e2_audio_preprocess_wav(const uint8_t *data, size_t len, float *pcm);

/* Common containers advertised by the linked decoder. libsndfile may support additional formats;
 * a container flag does not guarantee every codec/subtype of that container. */
#define E2_AUDIO_WAV 1
#define E2_AUDIO_FLAC 2
#define E2_AUDIO_OGG 4
#define E2_AUDIO_MP3 8
#define E2_AUDIO_AIFF 16
#define E2_AUDIO_AU 32
int e2_audio_supported_formats(void);

/* ---------------------------------------------------------------- tokenizer
 *
 * Reads <model_dir>/tokenizer.json. Owns ~25 MB (the 262144-entry vocab and the 514906-entry merge
 * table); not thread-safe, one tokenizer per thread. Audio/video placeholders are tokenized as
 * ordinary special tokens, not expanded.
 */
typedef struct e2_tokenizer e2_tokenizer;

e2_tokenizer *e2_tokenizer_load(const char *model_dir);
void e2_tokenizer_free(e2_tokenizer *t);

/* Text -> ids with the model's <bos> ... <eos> template. Returns the token count, or -1 (see
 * e2_last_error). max_ids must be at least 2 + 3 per input byte, which is the worst case. */
int e2_encode(e2_tokenizer *t, const char *text, int32_t *ids, int max_ids);

/* Same, but every `<|image|>` in the text is expanded to `<|image>` + n_soft_tokens[k] copies of
 * `<|image|>` + `<image|>`, in order of appearance, which is the layout the processor builds and
 * e2_text_image_embed expects. */
int e2_encode_mm(e2_tokenizer *t, const char *text, const int32_t *n_soft_tokens, int n_images,
                 int32_t *ids, int max_ids);

/* Same for a text that may carry both `<|image|>` and `<|audio|>` placeholders: each is expanded to
 * <|image>/<audio> + n soft slots + <image|>/<audio|>, in order of appearance. Either count array
 * may be NULL with its count 0, which leaves that placeholder as an ordinary token. */
int e2_encode_mma(e2_tokenizer *t, const char *text, const int32_t *n_image_soft, int n_images,
                  const int32_t *n_audio_soft, int n_audios, int32_t *ids, int max_ids);

/* ---------------------------------------------------------------- image preprocessing
 *
 * The HF Gemma4ImageProcessor, bit-for-bit: aspect-ratio-preserving bicubic resize (Pillow's
 * filter, rounded to 8 bits between the two passes), rescale by 1/255, 16x16 patchify, position
 * ids, and right-padding to E2_MAX_PATCHES with a zero patch and (-1,-1).
 *
 * `pixels` holds E2_MAX_PATCHES * 768 floats and `position_ids` E2_MAX_PATCHES * 2 int32, both
 * supplied by the caller. Returns the number of real patches (n_patches / 9 soft tokens), or -1.
 */
#define E2_PATCH_DIM 768
#define E2_MAX_PATCHES 2520 /* max_soft_tokens 280 x pooling 3 x 3 */

/* One entry point for every input the build understands: PNG (in-tree decoder), JPEG and WebP (the
 * system libjpeg-turbo / libwebp when the engine was built with them - check
 * e2_image_supported_formats()). Applies the EXIF orientation before resizing, as the reference
 * does. Returns the real patch count or -1. */
int e2_image_preprocess(const uint8_t *data, size_t len, float *pixels, int32_t *position_ids);

/* Which containers this build can decode. */
#define E2_IMAGE_PNG 1
#define E2_IMAGE_JPEG 2
#define E2_IMAGE_WEBP 4
int e2_image_supported_formats(void);

/* PNG only (in-tree reader, no dependency). */
int e2_image_preprocess_png(const uint8_t *png, size_t len, float *pixels, int32_t *position_ids);

/* Already-decoded, tightly packed 8-bit RGB - the seam for anything else (the caller decodes). */
int e2_image_preprocess_rgb(const uint8_t *rgb, int width, int height, float *pixels,
                            int32_t *position_ids);

int e2_image_max_patches(void);
/* soft tokens a given patch count becomes (the vision tower pools 3x3) */
int e2_image_soft_tokens(int n_patches);

/* Introspection (used by the tests and the benchmark). */
void e2_info(const e2_model *m);

/* Last load/encode/preprocess/embed error, or "" after a successful such call.
 * Free, info and pure size/format queries leave it unchanged. Not thread-safe. */
const char *e2_last_error(void);

#endif /* E2_H */
