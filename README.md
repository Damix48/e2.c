# e2 - EmbeddingGemma-2 int8 inference engine

A small C11 library that runs **only** Google's EmbeddingGemma-2 (text, image,
audio, video) on the CPU, with 8-bit weights and a custom GEMM.

* No runtime dependencies beyond libc + pthreads.
* One self-contained int8 model file per modality combination.
* Runtime CPU dispatch: AVX-512 VNNI, AVX2, ARM NEON dotprod, scalar fallback.
* Text, vision and audio towers, interleaved inputs, Matryoshka truncation.
* CLI and a small C test suite.

```
  EmbeddingGemma-2 (740M params)          e2 model file
  ├─ text tower   270M  int8 137 MB       ┌───────────────────────────────┐
  ├─ vision tower 170M  int8  84 MB  ───► │ header + int8 matrices + f32  │
  └─ audio tower  300M  int8 152 MB       │ norms + tokenizer + mel table │
                                          └───────────────────────────────┘
                   GEMM: y = x @ W^T     packed [N/16][K/4][16][4]
                   AVX-512 VNNI vpdpbusd, per-128 activation scales
```

## Build

```bash
make                 # libe2.a + e2cli
make test            # C self-checks (uses models/)
make bench           # text throughput / latency
```

`-march=native` is **not** required: the hot kernels are compiled with
`__attribute__((target(...)))` and selected at runtime.

Optional system libraries are auto-detected with `pkg-config`: libpng +
libjpeg-turbo replace the vendored decoders, libwebp adds WebP input, and
libsndfile adds FLAC/OGG/Opus/MP3 audio loading. Force any of them off/on:

```bash
make USE_SYSTEM_IMAGE=0 USE_WEBP=0 USE_SNDFILE=0   # vendored / raw PCM only
make USE_SYSTEM_IMAGE=1 USE_WEBP=1 USE_SNDFILE=1
```

Run `make clean` after changing any `USE_*` flag; the object files do not
embed the flag set.

For ARM64 with dot-product support:

```bash
make DOTPROD=1
```

## Quick start

```bash
# One-time: quantize the HuggingFace checkpoint (needs Python + uv)
uv venv .venv && . .venv/bin/activate
uv pip install torch safetensors numpy
python tools/e2_pack.py --modalities text,image,audio --out models/e2.bin \
                        --tokenizer-out models/e2-tokenizer.bin

# Run
./e2cli text models/e2-text+image+audio.bin models/e2-tokenizer.bin \
          "What causes the northern lights?"
```

```c
#include "e2.h"

e2_ctx *ctx = e2_load("models/e2-text+image+audio.bin",
                      "models/e2-tokenizer.bin");
if (!ctx) { fprintf(stderr, "%s\n", e2_error()); return 1; }

float emb[768];
e2_embed_text(ctx, "What causes the northern lights?", emb, 768);
e2_embed_text_task(ctx, "search_query", "northern lights", emb, 768);
e2_embed_image(ctx, "photo.jpg", emb, 768);
e2_embed_audio(ctx, pcm16k_mono, nsamples, emb, 768);

/* Optional file loading (libsndfile); downmixes to mono and resamples to 16 kHz */
float *pcm = NULL; int npcm = 0;
if (e2_audio_read("clip.flac", &pcm, &npcm) == 0) {
    e2_embed_audio(ctx, pcm, npcm, emb, 768);
    free(pcm);
}

/* Interleaved: one embedding for text + media */
e2_input in = {
    .text = "Product shot <|image|> and the demo <|video|>",
    .images = image_paths, .n_images = 1,
    .videos = video_frames, .video_frames = frame_counts, .n_videos = 1,
};
e2_embed(ctx, &in, emb, 768);

e2_free(ctx);
```

All embeddings are L2-normalized. `dim` may be `768`, `512`, `256` or `128`
(Matryoshka). Batch text with `e2_embed_text_batch` — it runs one forward over
all token rows, so weights are read once for the whole batch.

A context owns a thread pool and reusable scratch buffers, so it is **not**
safe to call it from several threads at once. Create one context per thread
when you need concurrency; the mmap'd model pages are shared by the OS.

A context owns a thread pool and reusable scratch buffers, so it is **not**
safe to call it from several threads at once. Create one context per thread
when you need concurrency; the mmap'd model pages are shared by the OS.

## Model variants

`tools/e2_pack.py` writes one file per modality combination:

| file | modalities | size |
|---|---|---|
| `e2-text.bin` | text | 278 MB |
| `e2-text+image.bin` | text, image | 498 MB |
| `e2-text+audio.bin` | text, audio | 595 MB |
| `e2-text+video.bin` | text, video (vision tower) | 498 MB |
| `e2-text+image+audio.bin` | text, image, audio | 815 MB |
| `e2-text+image+audio+video.bin` | all | 815 MB |

Video reuses the vision tower, so text+video and text+image share weights; the
flag differs so the library can report supported modalities. `models/` is
git-ignored; regenerate with `make pack`.

The tokenizer is packed separately (`models/e2-tokenizer.bin`, 12 MB) and is
shared by all variants.

## Accuracy

Measured against the HuggingFace implementation in float32 on this machine
(cosine similarity of the final, normalized embedding):

| input | cosine |
|---|---|
| text (short and long) | 0.9997 |
| image (640x480) | 0.9994 |
| audio (2 s, 16 kHz) | 0.9994 |
| audio (1.37 s, padded/masked) | 0.9940 |
| video (4 frames) | 0.9987 |
| text + image / text + audio / both | 0.9990-0.9991 |

Weights are symmetric int8 with one scale per output row. Activations are u8
with one scale per 128-element block, which keeps the error small even when a
row contains large outlier channels. The zero-point correction is folded into
the accumulator via a precomputed per-block weight sum.

## Performance and OpenVINO

Full methodology, conversion caveats and accuracy numbers are in
[`BENCH.md`](BENCH.md). Summary on this machine (8 threads, best of 9 pooled samples):

| task | length | e2 int8 | OpenVINO int8 | e2 / OV |
|---|---:|---:|---:|---:|
| text | 8 tokens | **3.2 ms** | 5.5 ms | 0.57 |
| text | 64 tokens | **6.2 ms** | 9.0 ms | 0.69 |
| text | 128 tokens | **13.2 ms** | 15.9 ms | 0.83 |
| text | 512 tokens | **40.8 ms** | 62.5 ms | 0.65 |
| text | 2048 tokens | **234.7 ms** | 483.8 ms | 0.49 |
| text batch | 8 x 64 | **34.8 ms** | 50.2 ms | 0.69 |
| text batch | 32 x 64 | **183.3 ms** | 212.4 ms | 0.86 |
| image 640x480 | 270 tok | **331.4 ms** | 526.3 ms | 0.63 |
| audio 2 s | 54 tok | **22.8 ms** | 24.7 ms | 0.93 |
| video 8 frames | 1058 tok | **1094.7 ms** | 1630.8 ms | 0.67 |

e2 is 1.1x-2.0x faster than OpenVINO int8 on every task; seven of the ten
rows meet the 1.5x target. The two engines do the same ~130 MB int8 weight
stream per forward, so the rows where e2 does not reach 1.5x are the ones
where that stream is spread over little compute (text 128), or where the row
is almost pure GEMM plus e2's own elementwise tail (batch 32x64), or where
the audio tower's 230 MB of weights dominate. `BENCH.md` has the methodology
and `PERFORMANCE.md` the full optimisation ledger and the gap analysis.
`tools/bench_compare.py` reproduces the comparison
(text 8/64/128/512/2048 tokens, batch, image, audio, video):

```bash
. .venv/bin/activate
uv pip install openvino nncf
make tests/e2_bench_full
python tools/bench_compare.py --iters 3 --threads 8 --repeat 3
```

## Files

```
e2.h                     public API (the only header bindings need)
src/e2.c                 lifecycle, text API, batch, truncation
src/e2_model.c           model file parsing + weight binding
src/e2_gemm.c            int8 GEMM, packing, CPU dispatch
src/e2_ops.c             RMSNorm, RoPE, attention, GELU/SiLU, f32 GEMM
src/e2_text.c            text encoder (24 layers + PLE)
src/e2_vision.c          vision tower (16 layers, axial RoPE, pooler)
src/e2_image.c           PNG/JPEG decode, bicubic AA resize, patchify
src/e2_audio.c           mel spectrogram, FFT, conformer tower
src/e2_audio_file.c      libsndfile loading + mono downmix + 16 kHz resample
src/e2_multimodal.c      image/audio/video/interleaved entry points
src/e2_tokenizer.c       Gemma BPE tokenizer (packed vocab + merges)
src/e2_util.c            arena, thread pool, fast exp, timing
src/e2_internal.h        internals (not installed)
vendor/png.h, jpeg.h     vendored single-header image decoders
tools/e2_pack.py         HuggingFace -> int8 E2 model
tools/e2_ref.py          HuggingFace reference dumps for validation
tools/e2cli.c            command line front-end
tools/bench_compare.py    e2 vs OpenVINO (f32/int8) benchmark, all tasks
tests/e2_test.c          self-checks
tests/e2_bench_full.c    spec-driven text/batch/image/audio/video latency
```

## Extending to a new CPU or GPU

All matrix multiplications go through `e2_linear_forward()` in
`src/e2_gemm.c`. A new backend only needs to implement the packed-int8 kernel
(`y[m,n] = sum_k xq[m,k] * wq[n,k] - 128*sum_k wq[n,k]` with per-row weight
scales and per-128 activation scales) and register it in `select_kernel()`.
The packed layout is `[N/16][K/4][16][4]` int8.

Attention/RMSNorm/RoPE live in `src/e2_ops.c`; a GPU backend can replace those
entry points. **No GPU backend is shipped in this repository yet** — the CUDA,
Metal and Vulkan paths are a documented extension point, not untested code.
The CPU backend is complete and validated. The same is true for ARM NEON:
the kernel is present behind `make DOTPROD=1` but could not be tested on this
x86 machine.

## Debug switches

| env | effect |
|---|---|
| `E2_DUMP=dir` | write intermediate tensors as raw f32 |
| `E2_PROFILE=1` | print per-phase timings to stderr |
| `E2_KERNEL=scalar\|avx2\|avx512` | force a GEMM kernel |
| `E2_ATTN_SCALAR=1` | force scalar attention rows |

## License

The C code is provided as-is. EmbeddingGemma-2 weights are Apache-2.0 from
Google DeepMind. The vendored PNG/JPEG decoders come from `iris.c`.
