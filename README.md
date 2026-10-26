# e2 - a C inference engine for EmbeddingGemma-2

`e2` is an inference engine for the [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2)
embedding model, implemented entirely in C11. It is written for this model instead of for graphs: the
text, vision and audio towers have their own int8 kernels, their shapes are baked in and checked
against the checkpoint at load, and there is no runtime graph interpreter, no GGUF, no ONNX.

It takes text, image files and audio files (or ready-to-embed 16 kHz PCM), and the four input shapes the model is used with
work end to end: **text**, **text+image**, **text+audio** and **text+image+audio**, each producing the
model's 768-d L2-normalized vector. Video is out of scope.

Nothing beyond libc, pthreads and libm is required. Image input can use libpng, libjpeg-turbo and
libwebp; audio files prefer libsndfile. All codecs are optional: disabling them leaves the in-tree
PNG/DEFLATE and PCM/float WAV readers. Set the four `E2_WITH_PNG/JPEG/WEBP/SNDFILE` options to `OFF`
for dependency-free media input.

## Quick Start

```bash
# Build (add -DE2_WITH_GPU=OFF -DE2_WITH_HIP=OFF for a CPU-only build)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Get the model (~1.4 GB, into models/ which is not tracked by git)
./download_model.sh

# Optional: the packed int8 weight sets (410 MB text-only .. 916 MB all towers), generated locally
# until they are on the Hub: ./download_models.sh all
 tools/export_models.sh models/embeddinggemma-2 models/e2 build

# Run the engine on the frozen fixtures: text, then image and audio
./build/e2_test       models/embeddinggemma-2 tests/fixtures
./build/e2_test_image models/embeddinggemma-2 tests/fixtures
./build/e2_test_audio models/embeddinggemma-2 tests/fixtures
```

With `-DE2_MODEL_DIR=$PWD/models/embeddinggemma-2` at configure time the same checks are registered
with CTest:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DE2_MODEL_DIR=$PWD/models/embeddinggemma-2
cmake --build build -j && ctest --test-dir build --output-on-failure
```

## Example

```c
#include "e2.h"

e2_model *m = e2_load("models/embeddinggemma-2");     /* mmap'd, lazily paged in */
e2_tokenizer *t = e2_tokenizer_load("models/embeddinggemma-2");

int32_t ids[2048];
float v[768];
int n = e2_encode(t, "a fluffy orange cat sitting on a windowsill", ids, 2048);
e2_text_embed(m, ids, n, v);                          /* 768-d unit vector */

e2_text_image_embed(m, ids, n, pixels, position_ids, n_patches, v);
e2_text_audio_embed(m, ids, n, pcm, n_samples, v);

e2_tokenizer_free(t);
e2_free(m);
```

`e2_image_preprocess()` turns PNG/JPEG/WebP bytes into the padded patch tensor;
`e2_audio_preprocess()` turns audio-file bytes into 16 kHz mono PCM. The existing embedding APIs
still accept prepared pixels/PCM directly. `e2_audio_tokens()` says how many soft tokens a clip
becomes (one per 40 ms). `E2_MAX_PATCHES` is 2520 and the sequence limit is 2048 tokens.

## Features

* **Short fixtures for all three towers verified** against the fp32 ONNX/PyTorch reference:
  0.9994-0.9998 cosine. Long steady-tone audio has a lower int8 floor; see
  [PERFORMANCE.md](PERFORMANCE.md#fidelity) and [production acceptance](PRODUCTION.md).
* **int8 weights and activations** in a quad-interleaved layout (`[N/8][K/4][8][4]`) so one 32-byte
  load covers eight output rows, dispatched at runtime to AVX512-VNNI (`vpdpbusd`), AVX-VNNI, AVX2
  (`vpmaddubsw`) or a scalar fallback. The scalar path always compiles.
* **Reads the original bf16 `model.safetensors`** directly, mmap'd. Optional one-shot export of the
  quantized model into a 916 MB file that loads in 0.01 s instead of 0.52 s and peaks at 180 MB RSS
  instead of 884 MB for a text query.
* **A persistent worker pool** where every parallel region claims its work from a shared counter
  instead of taking a fixed slice, and the default size keeps one thread per physical core, dropping
  cores below 75 % of the fastest core's peak frequency on hybrid CPUs.
* **Model front ends in-tree**: the model's BPE tokenizer (`tokenizer.json`, byte fallback, `<bos>`/
  `<eos>` template, `<|image|>`/`<|audio|>` placeholder expansion), and the image preprocessing the
  HF processor does (aspect-preserving bicubic resize, 16x16 patchify, axial position ids, padding).
  Audio's 16 kHz PCM-to-log-mel FFT is in-tree too. Audio-file input uses optional libsndfile,
  with an in-tree WAV reader, mono downmix and band-limited resampling as the dependency-free path.
* **CPU batching**: `e2_text_embed_batch()` shares projection work across sequences in one call.
* **Two GPU backends**: Intel Level Zero (runs all three towers, **experimental**) and AMD ROCm/HIP
  (text and vision, **source-reviewed only, never run on AMD hardware here**).

## Usage

The C API is in [`include/e2.h`](include/e2.h) - load, embed text/image/audio, batch text, the
tokenizer and image/audio preprocessing. `e2_image_supported_formats()` and
`e2_audio_supported_formats()` report common containers supported by the current build.

There is no general-purpose CLI. `tools/embed_cli.c` is a batch text-to-embedding bridge for
evaluation harnesses:

```bash
# TEXTS.bin is [uint32 len][len bytes] repeated; writes n x 768 float32 + the token ids
./build/e2_embed_cli models/embeddinggemma-2 texts.bin out.f32 out.ids
```

`tools/mteb_e2.py` drives it to run MTEB tasks against the engine (and optionally against an OpenVINO
INT8 IR, to compare engines rather than prompts).

The test binaries double as the practical entry points, since they print timings and fidelity per
case: `e2_test`, `e2_test_image`, `e2_test_audio`, `e2_test_pre`, `e2_text_bench`.

### Audio files

With file bytes already in memory and model `m` loaded (include `<stdlib.h>` and `<stdio.h>`):

```c
float *pcm = malloc(E2_AUDIO_MAX_SAMPLES * sizeof *pcm);
if (!pcm) return 1;
int samples = e2_audio_preprocess(audio_bytes, audio_size, pcm);
int rc = samples < 0 ? -1 : e2_audio_embed(m, pcm, samples, v);
if (rc) fprintf(stderr, "%s\n", e2_last_error());
free(pcm);
```

`E2_WITH_SNDFILE=AUTO` prefers libsndfile when its headers/library are installed; `ON` requires it,
and `OFF` forces WAV-only input. libsndfile handles WAV, FLAC, Ogg/Vorbis/Opus, MP3, AIFF, AU and
other formats available in the installed library (container flags do not promise every subtype).
`e2_audio_preprocess_wav()` always uses the in-tree PCM 8/16/24/32-bit / IEEE float 32/64-bit RIFF
WAV reader, including PCM/float WAVE_FORMAT_EXTENSIBLE. Compressed WAV, RF64 and RIFX need libsndfile.

Both paths accept 8..192 kHz and 1..32 channels. They retain the first 30 seconds, clamp finite
channel samples to [-1,1], average channels, and resample to 16 kHz with a normalized windowed-sinc
low-pass filter. Resampling overshoot is clipped too; NaN/Inf and fewer than 161 output samples fail.
The caller owns `E2_AUDIO_MAX_SAMPLES` (480000) floats; the return value is the real sample count,
rounded up after resampling. Output is untouched on error. No weights or worker pool are needed.
Already-16-kHz mono samples are not filtered; the resampler is not a bit-exact HF/librosa/soxr clone.
The existing PCM inference API and frozen reference fixtures are unchanged.

## Building

| option | default | meaning |
| :-- | :-- | :-- |
| `E2_WITH_GPU` | `ON` | Intel Level Zero backend (`src/gpu/`, OpenCL C kernels compiled at build time) |
| `E2_WITH_HIP` | `ON` | AMD ROCm/HIP backend (`src/gpu/hip/`); needs `hipcc` |
| `E2_WITH_PNG` / `E2_WITH_JPEG` / `E2_WITH_WEBP` | `AUTO` | use available system codecs; PNG `OFF` falls back to the in-tree reader |
| `E2_WITH_SNDFILE` | `AUTO` | prefer libsndfile for audio files; `OFF` keeps the in-tree PCM/float WAV reader |
| `E2_MODEL_DIR` | empty | registers the fixture tests in CTest |
| `E2_FIXTURE_DIR` | `tests/fixtures` | where the frozen fixtures live |

Each backend is dropped automatically when its toolchain is missing, so a plain `cmake -B build` on a
CPU-only machine still builds. The int8 path needs AVX2 at runtime; on a CPU without it the engine
falls back to bf16 (`E2_INT8=0` forces that path anywhere). [HARDWARE.md](HARDWARE.md) has the full
CPU/GPU matrix - which parts get which kernel, what a non-x86 port needs - and the compute libraries
that were measured and rejected.

## Testing

```bash
ctest --test-dir build --output-on-failure
```

Two tiers. The first needs no checkpoint and no device, and is what CI can run on any x86-64 machine:

| test | what it proves |
| :-- | :-- |
| `int8` | weight + per-block activation quantization and the VNNI/AVX2 GEMM against a float reference, on outlier-heavy data |
| `metrics` | the cosine/validation helpers every other test trusts |
| `softmax` | the SIMD exponential range reduction (a wrong polynomial passed everything else once) |
| `q8cache` | the packed int8 cache round-trips and refuses a corrupt or incompatible file before anything dereferences it |
| `audio_load` | private loader helpers, without a checkpoint |
| `audioproc` | WAV fallback/libsndfile parity, codec decoding, downmix/resampling, alias rejection, bounds and malformed files |
| `text_attention` | text scheduling: claimed regions, and blocked attention past 513 tokens |
| `gpu_copy` | Level Zero upload/download is byte-exact at unaligned sizes and several MB |
| `gpu_profile_fallback` | a failed timestamp append must not launch a kernel a second time |
| `gpu_attn_recipe` | the text tower's four-kernel int8 attention against an independent implementation, at the sequence lengths where shapes change |

The second tier needs the checkpoint (`-DE2_MODEL_DIR=...`) and compares against frozen fixtures:

| test | what it proves |
| :-- | :-- |
| `preprocess` | the tokenizer and image preprocessing against the HF processor's own outputs, plus a 45-case PNG filter matrix and a codec/orientation matrix (needs `tokenizer.json` only, not the weights) |
| `text` | 15-token query and 196-token document against the fp32 reference |
| `text_batch` | the batch API against the single-sequence path, with masked tokens |
| `text_reference` | a 42-case golden FP32 corpus, including kernel boundaries and retrieval rankings |
| `image` | the vision tower, image-only and image+caption |
| `audio` | the audio tower, text+audio and text+image+audio, with bit-exact PCM from a WAV-file round-trip |
| `validation` | every public input boundary the engine used to trust (token ids, patch positions, NULLs) |
| `lifecycle` | one live model per process, and load/free returns the heap |
| `gpu_text` / `gpu_image` / `gpu_audio` | the device towers against the fp32 reference and against the CPU engine |
| `gpu_text_reference` / `gpu_validation` / `gpu_shapes` | the golden corpus on the device, boundary validation, graph turnover and padded/chunked attention |
| `hip_text` / `hip_image` | the AMD towers; register only when the HIP backend was built |

Assertions and finite-value checks are compiled without `-ffast-math` and with `-UNDEBUG`, even in
Release, so they stay real checks; the production kernels keep fast-math. Device tests skip with
return code 77 when there is no device.

For CPU deployment and release acceptance, see [PRODUCTION.md](PRODUCTION.md). Run
`tools/check_cpu.sh <model_dir>` for the full checkpoint/sanitizer/cache gate. The long steady-tone
audio corpus exposes a lower int8 fidelity floor (~0.987 cosine) than the short fixtures; do not treat
the short-fixture 0.999 score as a universal audio accuracy guarantee.

## Performance

Milliseconds per item, Intel Core Ultra 9 285H (16 cores + Arc 140T iGPU), int8:

| workload | e2 CPU (14 thr) | e2 Intel GPU |
| :-- | --: | --: |
| text query, 15 tokens | **4.7** | 7.1 |
| text document, 196 tokens | 34.7 | **13.5** |
| image, 2520 patches | 720 | **189** |
| audio, 19 soft tokens (0.75 s) | **11.2** | 8.6-9.6 |

The full tables - against OpenVINO INT8 and llama.cpp Q8_0, the audio and front-end breakdowns, the
packed-cache numbers, and what the GPU clock does to a measurement - are in
[PERFORMANCE.md](PERFORMANCE.md).

## Memory

`E2_MAX_SEQ` is 2048 and the workspace is allocated at load: 268 MB for the text tower, ~435 MB with
both text and vision. A text-only process pays virtual address space, not resident memory. From the
packed cache a text query peaks at 180 MB RSS. `e2_load` needs a 64-bit address space.

The audio tower's int8 weights are built on the **first** audio call, not at load, so a text-only or
image-only process never pays for them: that call costs +0.39 s once.

## The packed quantized model

`E2_Q8_WRITE=<file>` dumps the model the loader builds - the int8 weights packed with their per-row
scales and per-block sums, plus the tensors that stay bf16/f32 - into one file. `E2_Q8_READ=<file>`
builds the model from it instead: no checkpoint, no quantization at load. It is the same numbers, not
a re-quantization (two exports are md5-identical, and the embeddings match to the last digit).

The file is this engine's own format (`src/q8cache.c`, documented at the top of that file) and is
**one-way**: there is no converter back to safetensors. `E2_INT8=0` refuses it, because it holds only
int8 weights plus the tensors that never had an int8 form.

`E2_Q8_TOWERS=text|text+image|text+audio|all` (default `all`) chooses which towers are packed, so a
text-only file does not carry 500 MB of vision and audio weights a text process never touches. Reading
takes whatever the file has; asking a text-only cache for an image returns "vision tower not loaded"
rather than a wrong answer.

### The four variants

One weight set, four files, differing only in which towers are packed:

| file | towers | size | input shapes it serves |
| :-- | :-- | --: | :-- |
| `…-int8-text.e8cq` | text | **410 MB** | text |
| `…-int8-text-image.e8cq` | text + vision | 599 MB | text, text+image |
| `…-int8-text-audio.e8cq` | text + audio | 727 MB | text, text+audio |
| `…-int8-text-image-audio.e8cq` | all three | 916 MB | everything, including text+image+audio |

Generate them from a checkpoint - the export is deterministic, two exports of the same checkpoint are
md5-identical, and each file is read back and put through the fixture test for its own shape:

```bash
 tools/export_models.sh <model_dir> models/e2 build     # writes the four files + manifest.sha256 (and legacy manifest.txt)
```

Or fetch the published copies (`HF_E2_REPO` overrides the repository id):

```bash
 ./download_models.sh all        # or: text | text-image | text-audio | text-image-audio
```

**The Hub repository is not published yet**, so `download_models.sh` currently ends with a pointer to
the local export. `tokenizer.json` is still needed from the checkpoint directory: the cache holds
weights, not the tokenizer.

## Known limitations

* **The Intel GPU backend is experimental and not production-ready.** Strict testing exposes
  intermittent wrong or zero outputs; the engine can currently return success with a wrong result.
  One 129-token golden case scores 0.998971, below the 0.999 gate. The cause is not established.
* **The AMD backend has not been validated on AMD hardware here.** HIP cache/RoPE/input fixes are
  source-reviewed only, and its correctness acceptance is outstanding.
* **Measuring the iGPU needs alternating binaries, not repeated runs.** The reported clock does not
  match the clock the GPU runs at, and two runs of the same binary can differ by 1.5-2x.
* The tokenizer implements *this* model's pipeline, not general SentencePiece: it reads
  `tokenizer.json` (no `tokenizer.model`), assumes the normalizer is `" " -> ▁`, and has no decode and
  no truncation.
* Architecture constants are baked in and validated against tensor shapes at load; a different
  Gemma variant would need a config parser. RoPE tables are precomputed to 2048 positions.
* One live model per process and not reentrant: `e2_load` rejects a second handle until `e2_free`.
* `-ffast-math` is on for the kernels, which is part of the ~1e-4 per-tensor deviation and would need
  revisiting for bit-level reproducibility.

## License

MIT, see [LICENSE](LICENSE). No model weights are distributed in this repository.
