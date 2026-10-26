# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

`e2` — a from-scratch C11 inference engine for **EmbeddingGemma-2**, written for that one model
rather than for graphs. It implements the text, vision and audio towers with model-specific int8
kernels, plus the front ends (BPE tokenizer, image preprocessing, log-mel audio), and produces the
model's 768-d L2-normalized embedding for text, text+image, text+audio and text+image+audio. Video is
out of scope. There are also two GPU backends: Intel Level Zero (experimental, device correctness
not accepted) and AMD ROCm/HIP (source-reviewed only).

The weights are **not** in this repository. The engine reads the original bf16 `model.safetensors`
from a model directory, or a prequantized cache exported by the engine itself (`E2_Q8_WRITE`).

## Layout

```
CMakeLists.txt      single build file; targets e2 (static lib), e2_gpu, e2_hip, e2_test*, e2_*_bench
cmake/              embed_binary.cmake (SPIR-V -> C array)
include/            public headers: e2.h (CPU + tokenizer + preprocess), e2gpu.h, e2hip.h
src/                CPU engine; kernels*.c (scalar/AVX2/AVX-512), int8*.c, threads.c, st.c (safetensors
                    mmap), q8cache.c (packed int8 model file), tokenizer.c, imageproc.c, imagedec.c,
                    png.c, vision.c, audio.c, text.c, e2.c (load/dispatch)
src/gpu/            Intel backend: l0.c (driver), gpu.c (graph/commands), kernels/e2.cl (OpenCL C)
src/gpu/hip/        AMD backend: C11 + HIP kernels.hip
tests/              one C file per test, wired into CTest in CMakeLists.txt
tests/fixtures/     frozen reference .bin fixtures (COMMITTED - ~31 MB, compresses to ~1.2 MB)
tools/              Python fixture generators (dump_*.py), embed_cli.c, mteb_e2.py
tools/bench/        benchmark sources: bench_compare.py + the C/HIP workers it builds
docs/llms/          the design/analysis corpus, kept verbatim (see below)
```

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Options: `-DE2_WITH_GPU=OFF` (Level Zero), `-DE2_WITH_HIP=OFF` (ROCm), `-DE2_WITH_PNG/JPEG/WEBP=OFF`
(system codecs; PNG off falls back to the in-tree DEFLATE reader, so a zero-dependency build is one
flag away). int8 needs AVX2 at runtime; AVX-512/AVX-VNNI are picked up when present.

## Test

```sh
ctest --test-dir build --output-on-failure
```

Self-contained checks (no weights) run always. Fixture tests are registered only when the checkpoint
is supplied: `-DE2_MODEL_DIR=/path/to/embeddinggemma-2` (that directory holds `model.safetensors`
and `tokenizer.json`). `-DE2_FIXTURE_DIR` defaults to `tests/fixtures`.

Live knobs: `E2_THREADS`, `E2_INT8=0`, `E2_Q8_READ=<cache.e8cq>`, `E2_Q8_WRITE=<cache.e8cq>`,
`E2_PROFILE=1`, `E2_DUMP=<dir>`, `E2_REF_ACT=<dir>`. Tests that need a device return 77 (skipped)
when it is missing.

## Landmines

* **Never commit** `build*/`, `benchmark-results/`, `tests/ref_act/`, `*.e8cq` — all are ignored and
  all are regenerable. `tests/ref_act/` is 218 MB of raw float32 written by the fixture generators;
  `*.e8cq` is the 916 MB int8 weight cache.
* **`tests/fixtures/*.bin` is committed on purpose.** It is ~31 MB on disk but ~1.2 MB deflated (the
  image fixtures are mostly padding). Do not "clean it up": the fixture tests and every published
  number depend on exactly these bytes. Regenerating them needs the checkpoint plus PyTorch.
* `-fno-fast-math -UNDEBUG` is forced on every `e2_test*` target by a loop at the end of
  `CMakeLists.txt`. Keep that invariant: assertions and finite-value checks must stay real in Release.
* Adding a test = a new `tests/test_<name>.c` **and** an entry in the `foreach(name ...)` list (or an
  explicit `add_executable`/`add_test` pair) in `CMakeLists.txt`; nothing is globbed.
* The engine is **one live model per process and not reentrant** (`e2_load` rejects a second handle;
  `e2_free` then reload). Worker threads share scratch. Do not add "just make it thread-safe" paths.
* `E2_INT8=0` cannot use a `.e8cq` cache — that file contains only int8 weights plus the tensors that
  never had an int8 form. The format is defined in `src/q8cache.c` and is one-way (no exporter to
  safetensors).
* GPU claims in the docs are not correctness acceptance: the Intel path can currently return success
  with a wrong result on some inputs, and the HIP path has not been run on AMD hardware here.

## License

MIT (`LICENSE`, from the repository's initial commit). No model weights are distributed here; the
model the engine runs is Google DeepMind's EmbeddingGemma-2 (Apache-2.0), obtained separately.

## docs/llms/ (local only, not committed)

A documentation corpus — the long design/performance write-up, an external architecture review, the
unresolved Intel GPU investigation, the benchmark runbook and one measured session — lives in
`docs/llms/`. It is **ignored and never published**, so it is present in the author's working tree and
absent from a fresh clone; a reference to it from a committed file must say that it is local (see
`tools/README.md`).

Treat it as the authoritative record of *why* the code looks the way it does, and keep numbers
consistent with it when you touch performance-sensitive paths.

* Paths in those documents predate this layout: they say `engine/tools/...`, `engine/build/...`,
  `engine/tests/fixtures` because this tree used to be the `engine/` subdirectory of the model
  workspace. Read `engine/` as the repository root.
* Some of them also reach outside this repository: `scripts/onnx_common.py` and
  `bench/fixtures/*.npz` live in that model workspace (the HF checkpoint directory), not here. The
  fixture *outputs* are committed, so nothing in the build or the test suite needs them.
