/*
 * e2.h - EmbeddingGemma-2 int8 inference library
 *
 * A dependency-free C engine for Google's EmbeddingGemma-2, specialized for
 * this model only. Text, image and audio towers are compiled in; the model
 * file decides which towers are present.
 *
 * Usage:
 *   e2_ctx *ctx = e2_load("e2-int8.bin", "e2-tokenizer.bin");
 *   if (!ctx) { fprintf(stderr, "%s\n", e2_error()); }
 *   float emb[768];
 *   e2_embed_text(ctx, "hello world", emb, 768);
 *   e2_free(ctx);
 */

#ifndef E2_H
#define E2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define E2_EMBED_DIM 768

/* Header ABI version. Bindings must refuse to load a library whose
 * e2_api_version() differs. Bump it on any signature change above. */
#define E2_API_VERSION 1
int e2_api_version(void);

typedef struct e2_ctx e2_ctx;

/* Modalities. A model file may contain only a subset. */
enum {
    E2_MOD_TEXT  = 1,
    E2_MOD_IMAGE = 2,
    E2_MOD_AUDIO = 4,
    E2_MOD_VIDEO = 8
};

/* ------------------------------------------------------------------ */
/* Lifecycle                                                          */
/* ------------------------------------------------------------------ */

e2_ctx *e2_load(const char *model_path, const char *tokenizer_path);
void    e2_free(e2_ctx *ctx);

/* Model file modality mask (E2_MOD_*). */
int  e2_modalities(const e2_ctx *ctx);
/* Runtime thread count, 0 = auto (one per physical core). */
void e2_set_threads(e2_ctx *ctx, int nthreads);
int  e2_threads(const e2_ctx *ctx);

/* Compute device.  CPU is the default and the reference; the GPU backend is
 * opt-in (also selectable at load time with E2_DEVICE=gpu). */
enum { E2_DEVICE_CPU = 0, E2_DEVICE_GPU = 1 };
/* Switch device.  Returns 0 on success, -1 if the GPU is unavailable (the
 * context then stays on the CPU). */
int  e2_set_device(e2_ctx *ctx, int device);
int  e2_device(const e2_ctx *ctx);

/* Last error, never NULL. */
const char *e2_error(void);

/* ------------------------------------------------------------------ */
/* Embeddings                                                         */
/* ------------------------------------------------------------------ */

/*
 * All embed functions write `dim` floats to `out` and L2-normalize the result.
 * dim must be 128, 256, 512 or 768 (Matryoshka truncation; re-normalized).
 * Returns 0 on success, -1 on error.
 */

/* Encode text with the model's default (no task prefix). */
int e2_embed_text(e2_ctx *ctx, const char *text, float *out, int dim);

/* Encode a batch. The whole batch is processed as one sequence set; this is
 * faster than one call per text for short inputs. */
int e2_embed_text_batch(e2_ctx *ctx, const char *const *texts, int n,
                        float *out, int dim);

/* Convenience: prepend a task prefix understood by the model, e.g.
 * "search_query", "search_document", "classification", "clustering",
 * "sentence_similarity", "code_retrieval", "question_answering",
 * "fact_checking". Returns 0 on success. */
int e2_embed_text_task(e2_ctx *ctx, const char *task, const char *text,
                       float *out, int dim);

/* Encode an image file (PNG, JPEG). */
int e2_embed_image(e2_ctx *ctx, const char *path, float *out, int dim);

/* Encode raw interleaved RGB8 pixels. */
int e2_embed_image_rgb(e2_ctx *ctx, const uint8_t *rgb, int width, int height,
                       float *out, int dim);

/* Encode raw mono float32 audio at 16 kHz. */
int e2_embed_audio(e2_ctx *ctx, const float *samples, int nsamples,
                   float *out, int dim);

/*
 * Load an audio file (WAV, FLAC, OGG/Vorbis, Opus, MP3 when built with
 * libsndfile), downmix to mono and resample to 16 kHz. Caller frees the
 * returned buffer with e2_free_buffer(). Returns 0 on success.
 */
int e2_audio_read(const char *path, float **samples, int *nsamples);

/* Free a buffer returned by this library. Never free() it directly: language
 * runtimes link their own allocator on some platforms. */
void e2_free_buffer(void *p);

/* Encode a video as a sequence of frames (image files), sampled as given. */
int e2_embed_video(e2_ctx *ctx, const char *const *frame_paths, int nframes,
                   float *out, int dim);

/*
 * Interleaved multimodal input: `text` may contain <|image|>, <|audio|> and
 * <|video|> placeholders. `images` are consumed left to right, `frames` are
 * consumed one video per placeholder, `audio` per audio placeholder.
 */
typedef struct {
    const char *text;
    const char *const *images;      int n_images;
    const char *const *const *videos; int n_videos; /* each: frame path array */
    const int *video_frames;
    const float *const *audio;      const int *audio_len; int n_audio;
} e2_input;

int e2_embed(e2_ctx *ctx, const e2_input *in, float *out, int dim);

/* ------------------------------------------------------------------ */
/* Utilities                                                          */
/* ------------------------------------------------------------------ */

/* L2-normalize + truncate in place, dim = 128/256/512/768. */
void e2_normalize(float *v, int dim);
float e2_dot(const float *a, const float *b, int n);
const char *e2_model_info(const e2_ctx *ctx);
double e2_last_ms(const e2_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* E2_H */
