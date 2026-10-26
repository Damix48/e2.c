This is a from-scratch C11 inference engine for **EmbeddingGemma-2** (Google DeepMind, Apache-2.0
weights), written for that one model instead of for a graph runtime. The project is called `e2`
(EmbeddingGemma 2). It runs the text, vision and audio towers in int8 and produces the model's 768-d
L2-normalized embedding.

Architecture constants are baked in and **validated against the tensor shapes at load**; a different
Gemma variant would need a config parser, so do not add speculative configurability. See
[README.md](README.md) for the user-facing description and [PERFORMANCE.md](PERFORMANCE.md) for the
measurements.

# Naming Convention

- **`e2_`** prefix for everything public: `e2_load`, `e2_text_embed`, `e2_q8w_load`, `e2_test_*`
- **`E2_`** for macros/constants (`E2_MAX_SEQ`, `E2_V_HEADS`) and environment variables (`E2_THREADS`)
- **`k_`** prefix for device kernels (`k_gemm_dpas2`, `k_attn`, `k_audio_attn`), in both the OpenCL C
  and the HIP sources
- **`g_`** for file-scope mutable state (`g_q8_read`, `g_gemm_ms`); it is process-global and one of
  the reasons the engine is not reentrant
- **`q8`** for the int8 weight format and its cache (`e2_q8w`, `q8c_*`), **`i8`** for int8 kernels
- **`l0_`/`e2_l0_`** for the Level Zero layer, **`hip_`/`e2_hip_`** for the AMD one
- Component-level namespaces stay as they are: `st_*` (safetensors), `e2_png*`, `vf_*` (vision front
  end helpers), `a*` (audio tower locals)

# Supported Modes

Four input shapes, one weight set, four entry points:

- **Text**: ids -> 768d. `e2_text_embed`, plus `e2_text_embed_batch` for CPU batching.
- **Text + image**: `<|image|>` placeholders expanded by `e2_encode_mm`, `e2_text_image_embed`.
- **Text + audio**: `<|audio|>` expanded by `e2_encode_mma`, `e2_text_audio_embed`.
- **Text + image + audio**: both placeholder kinds in one sequence, `e2_text_image_audio_embed`.

Video is out of scope. Sequence limit 2048 ids, image budget `E2_MAX_PATCHES` 2520 (280 soft tokens),
audio up to 30 s (750 soft tokens, one per 40 ms).

# High-Level Capabilities

- int8 weights **and** activations, dispatched at run time: AVX512-VNNI -> AVX-VNNI -> AVX2 -> scalar.
- bf16 path (`E2_INT8=0`): more accurate, 5-9x slower, same blocked attention.
- Persistent worker pool; every region claims work from a shared counter instead of a fixed slice.
- Reads the bf16 `model.safetensors` directly (mmap, demand-paged). Optional packed int8 cache
  (`E2_Q8_WRITE`/`E2_Q8_READ`, 916 MB, 0.01 s load) that replaces the checkpoint entirely.
- Model-specific front ends in-tree: BPE tokenizer from `tokenizer.json`, HF image preprocessing,
  16 kHz PCM -> log-mel with an in-tree FFT. No ONNX or Python in the engine.
- Audio files: optional libsndfile by default, PCM/float RIFF WAV in-tree when it is unavailable;
  `e2_audio_preprocess` downmixes/resamples to 16 kHz mono and caps the result at 30 s.
- Image containers: PNG (in-tree reader, own DEFLATE), JPEG and WebP through the system codecs with
  EXIF orientation applied; each codec is optional at build time.
- Two GPU backends: Intel Level Zero (all three towers, **experimental**) and AMD HIP (text and
  vision, **never validated on AMD hardware**).

# File Structure

```
include/e2.h            - public API: load/free, text/image/audio embed, batch, tokenizer, preprocessing
include/e2gpu.h         - public Intel GPU (Level Zero) entry points
include/e2hip.h         - public AMD GPU (ROCm/HIP) entry points
src/e2.c                - model loading, workspace management, public entry points
src/model.h             - internal model layout: bf16 pointers into the mmap + f32 copies
src/st.c                - safetensors: mmap + JSON header parser + typed tensor views
src/q8cache.c           - packed int8 model cache: writer, reader, mmap'd index
src/int8.c              - weight quantization at load + the scalar int8 GEMM fallback
src/kernels.c           - scalar reference backend + runtime backend dispatch
src/kernels_avx2.c      - AVX2 + FMA backend (fp32)
src/kernels_avx512.c    - AVX-512F/BW/VL/DQ + FMA backend (fp32)
src/kernels_i8_avx2.c   - int8 GEMM for Haswell..Zen3 (vpmaddubsw)
src/kernels_i8_vnni.c   - int8 GEMM on AVX-VNNI / AVX512-VNNI (vpdpbusd)
src/kernels_i8_avx512_vnni.c - int8 GEMM and the packed-operand attention GEMM on AVX512-VNNI
src/threads.c           - persistent worker pool (claim-from-counter, futex sleep)
src/text.c              - text tower: 24 layers, PLE, q/k/v norms, sliding window, fused regions
src/vision.c            - vision tower: patch embedder, axial 2D RoPE, 16 layers, 3x3 pooling
src/audio.c             - audio tower: USM/Conformer, 12 layers, chunked local attention, light conv
src/audioproc.c         - audio files: libsndfile / in-tree WAV, mono downmix + band-limited resampling
src/tokenizer.c         - tokenizer.json -> BPE + byte fallback + placeholder expansion
src/imageproc.c         - the HF Gemma4ImageProcessor in C: resize, patchify, position ids, padding
src/imagedec.c          - JPEG/WebP through the system libraries + EXIF orientation
src/png.c               - in-tree PNG reader with its own DEFLATE (no zlib)
src/validation.h        - input-boundary checks shared by the vision and audio entry points
src/gpu/l0.c            - the Level Zero layer: context, module from SPIR-V, queues, copies
src/gpu/gpu.c           - Intel backend: model upload, graph recording, forward pass
src/gpu/kernels/e2.cl   - the OpenCL C kernels, compiled to SPIR-V at build time
src/gpu/e2_tile.h       - the tiled int8 GEMM's shape, shared by C and OpenCL C
src/gpu/hip/hip.c       - HIP host layer: device selection, memory, graph capture
src/gpu/hip/forward.c   - HIP forward pass: one upload, one recorded graph per shape
src/gpu/hip/kernels.hip - the HIP kernels
tools/embed_cli.c       - batch text -> embedding bridge for MTEB (`e2_embed_cli`)
tools/export_cache.c    - load and write the packed int8 cache, nothing else (`e2_export_cache`)
tools/mteb_e2.py        - runs MTEB tasks through e2_embed_cli
tools/export_models.sh  - writes the four packed models (E2_Q8_TOWERS x E2_Q8_WRITE) + manifest.txt
tools/dump_*.py         - fixture generators (development only, need PyTorch + the checkpoint)
tools/bench/            - bench_compare.py plus the C/HIP benchmark workers
tests/test_*.c          - one file per test, wired into CTest in CMakeLists.txt
tests/fixtures/         - frozen reference fixtures (committed, see its README)
cmake/embed_binary.cmake - SPIR-V blob -> C array
download_model.sh       - fetch model.safetensors + tokenizer.json into models/ (gitignored)
download_models.sh      - fetch the published packed int8 models (placeholder repo id for now)
```

# Build Targets

One `e2` static library carries the CPU engine; the ISA backends are separate objects selected at
run time, the GPU backends are separate libraries (`e2_gpu`, `e2_hip`).

- **generic**: pure C, no dependencies, very slow. Always compiles and is the fallback.
- **AVX2 / AVX-512**: fp32 kernels, chosen by CPUID.
- **AVX2 / AVX-VNNI / AVX512-VNNI**: the int8 path, picked in the same dispatch. Needs AVX2 at
  minimum on the host.
- **Intel GPU**: `src/gpu/`, Level Zero, OpenCL C compiled to SPIR-V by `cmake/embed_binary.cmake`.
  Configure drops the backend when `libze_loader` or the SPIR-V toolchain is missing.
- **AMD GPU**: `src/gpu/hip/`, HIP sources compiled with `hipcc`; dropped when it is missing.

Options: `E2_WITH_GPU`, `E2_WITH_HIP`, `E2_WITH_PNG`, `E2_WITH_JPEG`, `E2_WITH_WEBP`, `E2_WITH_SNDFILE`, `E2_MODEL_DIR`,
`E2_FIXTURE_DIR`. See [README.md](README.md#building).

# Development Rules

- No new dependencies. Acceptable: libc, pthreads, libm, the optional system image codecs and libsndfile,
  Level Zero, ROCm/HIP. No ONNX, no Python, no C++ in the engine.
- The scalar backend must keep compiling and working: it is the reference every SIMD path is checked
  against. If you add a kernel, add its scalar version.
- `-fno-fast-math -UNDEBUG` is forced on every `e2_test*` target by the loop at the end of
  `CMakeLists.txt`. Keep that invariant: assertions and finite-value checks must stay real in Release.
- Always test with `ctest`, and if you can supply a checkpoint, with `-DE2_MODEL_DIR`. A change that
  touches a factory of the arithmetic must move a fidelity number, not just pass.
- Adding a test = new `tests/test_<name>.c` **and** an entry in `CMakeLists.txt` (`foreach(name ...)`
  list or an explicit `add_executable`/`add_test` pair). Nothing is globbed.
- Never commit `build*/`, `benchmark-results/`, `tests/ref_act/` or `*.e8cq` - all ignored, all
  regenerable. `tests/fixtures/*.bin` **is** committed on purpose (~31 MB, ~1.2 MB deflated); do not
  regenerate or "clean" it casually, every published number depends on those bytes.
- Keep code simple and understandable; leave no dead code. Prefer one more measurement over one more
  tuning knob - this tree already has too many `E2_*` A/B switches, so add one only with a test.
- If you optimize one backend, verify the others were not regressed; the CPU path is the reference for
  both GPU backends.
- Stick to standard C11; no compiler-specific tricks outside the ISA-specific translation units.

# How To Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./download_model.sh                                     # -> models/embeddinggemma-2 (1.4 GB, gitignored)
./build/e2_test       models/embeddinggemma-2 tests/fixtures   # text
./build/e2_test_image models/embeddinggemma-2 tests/fixtures   # image, image+caption
./build/e2_test_audio models/embeddinggemma-2 tests/fixtures   # audio, text+audio, text+image+audio
./build/e2_test_pre   models/embeddinggemma-2 tests/fixtures   # tokenizer + preprocessing, no weights
```

`E2_THREADS=n` pins the pool (`1` disables it), `E2_REPS=n` makes `e2_test` report the best of n runs.
The GPU test binaries take the same two arguments and skip with 77 when there is no device.

# Debugging Notes

| variable | effect |
| :-- | :-- |
| `E2_PROFILE=1` | wall-clock breakdown of one embedding per parallel region, plus GEMM parallelism |
| `E2_DUMP=<dir>` | raw f32 intermediates per stage (`x0`, `ple`, `l0_attn_out`, ...) |
| `E2_REF_ACT=<dir>` | compare those stages against `tests/ref_act/*.f32` and report per stage |
| `E2_GPU_PROFILE=1` | bracket every device kernel with timestamp writes |
| `E2_GPU_DUMP=<dir>` / `E2_HIP_DUMP=<dir>` | device-side intermediates |
| `E2_INT8=0` | bf16 path (and: refuses a packed cache, which holds only int8 weights) |
| `E2_Q8_WRITE=<f>` / `E2_Q8_READ=<f>` | write / read the packed int8 model cache |
| `E2_Q8_TOWERS=` | with `E2_Q8_WRITE`: which towers the cache holds (`text`, `text+image`, `text+audio`, `all`) |

`tests/ref_act/` is 218 MB of raw float32 and is **not** committed: the fixture generators write it
when you need to bisect a tower stage by stage. Without it, the fixtures alone still give you the
final embedding comparison.

Both front ends are cheap enough to bisect by hand: the tokenizer and the image preprocessing need no
checkpoint, so `e2_test_pre` runs in about a second.

# Architecture Constants

Baked into `src/model.h` and checked against the checkpoint at load:

| text | vision | audio |
| :-- | :-- | :-- |
| 24 layers, hidden 512, ffn 2048 | 16 layers, hidden 768, 12 heads, head_dim 64 | 12 layers, hidden 1024, 8 heads |
| 4 heads, GQA with 2 (local) / 1 (global) kv heads | patch 16x16x3 -> 768, pooling 3x3 | ffn 4096, output_proj 1536 |
| head_dim 256 local / 512 global, 5:1 local:global | position table 10240, axial 2D RoPE | attention chunk 12, relative window 12 |
| sliding window 1024, RoPE theta 1e4 global / 1e6 local | max patches 2520 (280 soft tokens) | mel 128 bins, FFT 512, hop 160 (10 ms) |
| vocab 262144, context 2048 (8K weights, tables to 2048) | | max 30 s -> 750 soft tokens |

# Known Pitfalls (Historical Bugs)

Each of these cost real debugging time and none of them was caught by an ordinary end-to-end test:

* **Wrong SIMD exponential polynomial.** `1+r*P(r)` instead of `1+r+r*r*P(r)` in the AVX2 and
  AVX-512 range reduction. It passed every e2e fixture (cos 0.998399); only a 42-case golden corpus
  made it visible. `test_softmax` is the regression.
* **Unvalidated patch positions.** Caller-supplied coordinates indexed the axial position table and
  sized the pool grid: one patch at `(300,300)` produced a 31 MB heap-buffer-overflow write. Also the
  reason `validation.h` exists - every public entry point validates ids, positions and pointers.
* **`setjmp`/`longjmp` locals in the JPEG error path.** `buf` and `created` were non-volatile and
  written between `setjmp` and `longjmp`, so cleanup read indeterminate values and leaked the RGB
  buffer plus ~12 KiB of libjpeg state on every malformed file. Fuzzing the decode path found it.
* **PNG row-0 filter state.** A missing zeroed "previous row" for row 0 hid behind every real image
  happening to start with the Sub or None filter; the filter bytes live inside the compressed stream,
  so a mutation fuzzer cannot reach them. The 45-case filter matrix in `png.bin` is the regression.
* **`e2_tensor_exists` must ask both cache kinds.** It probed `Q8C_BF16` only, so the vision tower's
  probe tensor (an int8-quantized weight, stored as `Q8C_Q8W`) looked absent and a full cache loaded
  with no vision tower at all. The export verification catches this: it read every file back and the
  two image shapes failed with "vision tower not loaded" while the text shapes passed.
* **Attention past 513 tokens.** The blocked attention path only existed for short sequences; the
  longer window silently used the wrong shape. `test_text_attention` and the 514/2048 rows of
  [PERFORMANCE.md](PERFORMANCE.md#cpu-single-item) cover it.
* **Duplicate launches under profiling.** Retrying a timestamp append by falling back to a plain
  launch ran the kernel twice. `test_l0_profile_fallback` links with `-Wl,--wrap=zeCommandListAppend*`
  to force each failure mode.
* **Leaked int8 weights across loads.** The text tower's ~570 MB of quantized weights were not freed
  by `e2_free` (three cycles retained 143 -> 285 -> 427 MB). `test_lifecycle` checks the heap returns.
* **The vision softmax and the P quantizer are one kernel by contract.** `e2_softmax_quant_i8` skips
  writing the fp32 probability row because the int8 quantizer is its only consumer; a second consumer
  makes that fusion illegal, not just slower.

# Documentation

`docs/llms/` holds a local, unpublished corpus: the long design/performance write-up, an external
architecture review, the unresolved Intel GPU investigation, and the benchmark runbook. It is ignored
by git and absent from a clone, so do not link to it from committed files. Everything a reader of this
repository needs is in `README.md`, `PERFORMANCE.md`, `HARDWARE.md`, `AGENTS.md`, `tools/README.md` and
`tests/fixtures/README.md`.

# License

MIT (`LICENSE`). No model weights are distributed here; the model is Google DeepMind's
EmbeddingGemma-2 (Apache-2.0), obtained separately.
