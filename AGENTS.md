# e2 - agent / contributor notes

`e2` is a C11 inference engine for **EmbeddingGemma-2 only**. It is not a
general transformer runtime: the architecture constants, tensor names and
preprocessing are fixed to this checkpoint.

## Ground rules

- C11, no dependencies beyond libc + pthreads.
- Do not add `-march=native` to the default build. Hot kernels use
  `__attribute__((target(...)))` and runtime dispatch. The scalar path must
  always compile and stay correct.
- Keep the public API in `e2.h` small; bindings should only need that header.
- Every change that touches model math must be checked against
  `tools/e2_ref.py` (see below) before committing.
- One source of truth for the quantized format: `tools/e2_pack.py` writes it,
  `src/e2_model.c` and `src/e2_gemm.c` read it. Change all three together.
- Code style: 4 spaces, `e2_` prefix for shared symbols, `snake_case`.

## Build / test

```bash
make                  # libe2.a + e2cli (auto-detects png/jpeg/webp/sndfile)
make USE_SYSTEM_IMAGE=0 USE_WEBP=0 USE_SNDFILE=0   # vendored / PCM-only
make test             # self-checks against models/
make shared           # libe2.so / libe2.dylib for language bindings
make golden           # regenerate the C# SDK reference embeddings
make bench            # text batch throughput
make bench-full       # text/image/audio latency
make clean
```

System libraries are optional: libpng + libjpeg-turbo (faster/robust image
decode), libwebp (WebP input), libsndfile (FLAC/OGG/Opus/MP3 + resampling).
The vendored PNG/JPEG decoders and raw-PCM API are the fallback.

`make test` expects `models/e2-text.bin`, `models/e2-text+image+audio.bin` and
`models/e2-tokenizer.bin`. Regenerate with `make pack` (Python + uv venv).
`make pack` needs a venv at `.venv` with torch, safetensors and numpy.

## Model file format (version 1)

```
"E2MODEL1" (8)
u32 version=1, u32 ntensors, u32 reserved
ntensors * 168-byte entries:
    char name[96]
    u32 dtype (0=f32, 1=int8, 2=int32)
    u32 ndim
    i64 shape[4]
    u64 offset, u64 nbytes, u64 flags, u64 reserved
data (64-byte aligned)
```

Flags: bit 0 = packed int8 matrix. Packed matrices have companion tensors
`<name>.scale` (f32, one per output row) and `<name>.wsum` (i32, one per output
row per 128-wide K block). Unpacked int8 tensors are row-major (embedding
tables). `__meta__` is an i32 config vector, `__params__` f32 scalars,
`__mel_filters__` and `__hann_window__` are preprocessing tables.

Packed layout: `[N/16][K/4][16][4]` int8. For a fixed 16-row output tile the
512 bytes for one K block are contiguous, so a kernel walks K sequentially.

## Quantization

- Weights: symmetric int8, one f32 scale per output row.
- Activations: asymmetric u8 (zero point 128), one f32 scale per 128-element
  block. The correction `-128*sum(w)` is precomputed per block (`wsum`) and
  folded into the integer accumulator.
- In the kernel, the accumulator is int32 per block, then converted to f32 with
  `scale_act_block * scale_weight_row` and accumulated. Do not sum multiple
  blocks in int32 before applying the block scale.

## Validation workflow

1. Produce reference dumps with the HuggingFace implementation:

   ```bash
   . .venv/bin/activate
   python tools/e2_ref.py text  --text "..."   --out /tmp/ref_text
   python tools/e2_ref.py image --image pic.jpg --out /tmp/ref_img
   python tools/e2_ref.py audio --audio clip.wav --out /tmp/ref_aud
   ```

2. Run the C side with `E2_DUMP=/tmp/c` and compare the raw f32 files.
   `E2_DUMP` writes `embeddings`, `ple`, `layer1..24`, `last_hidden_state`,
   `embedding` for text; `vis_*` for vision; `aud_*` for audio.

3. A correct layer should be ~0.9999 cosine against the **weight-quantized**
   PyTorch run (expect larger deltas only from activation quantization and
   from chaotic amplification in the vision tower, which is known).

4. Final normalized embeddings should stay above 0.999 (text), 0.999 (image),
   0.999 (audio) cosine against float32.

The most useful regression test is `tests/e2_test.c` plus the reference-dump
comparison. Keep the test free of external assets.

For a speed comparison against OpenVINO:

```bash
uv pip install openvino nncf
make tests/e2_bench_full
python tools/bench_compare.py --iters 3 --threads 8 --repeat 3
```

`tools/bench_compare.py` generates deterministic assets `/tmp/e2bench`
(image, frames, WAV, texts tokenised to the requested lengths) and runs the
e2 binary from `tests/e2_bench_full.c` plus OpenVINO. It converts the
text/vision/audio towers separately; the audio tower's mask construction does
not trace, so it benchmarks the subsample projection and the layer stack with
a precomputed blocked mask. The text model is re-converted per sequence
length: the sliding-window branch is baked into the trace, so one trace
cannot cover both < 512 and >= 512 tokens.

## Where to change what

| task | file |
|---|---|
| new CPU GEMM kernel | `src/e2_gemm.c` (`select_kernel`, packed layout) |
| attention / norms / rope | `src/e2_ops.c` |
| text math | `src/e2_text.c` |
| vision math | `src/e2_vision.c` |
| image resize / decode | `src/e2_image.c` |
| audio mel / conformer | `src/e2_audio.c` |
| tokenizer | `src/e2_tokenizer.c` |
| audio file loading | `src/e2_audio_file.c` |
| public wrappers | `src/e2_multimodal.c`, `src/e2.c` |
| quantizer / format | `tools/e2_pack.py` |
| C# SDK (P/Invoke) | `sdk/csharp/src/E2/`, tests in `sdk/csharp/tests/` |

`e2.h` is the ABI. Any signature change there must bump `E2_API_VERSION` in
`e2.h` **and** `NativeMethods.ApiVersion` in `sdk/csharp/src/E2/NativeMethods.cs`;
`E2Model.Load` refuses a mismatched library.

## GPU backend

No GPU backend is shipped. The intended extension is:

- Add a device implementation of the packed-int8 linear
  (`e2_linear_forward`) and attention (`e2_attention_full`,
  `e2_attention_text`).
- Put it behind a `#ifdef E2_CUDA` / `E2_METAL` switch in `src/e2_gemm.c` and
  `src/e2_ops.c`; keep the CPU path as fallback.
- Never silently ship an untested GPU kernel. The CPU path is the reference.

## Known limitations

- Text input is capped at 8192 tokens; longer input is truncated by the
  tokenizer.
- Audio is truncated to 480000 samples (30 s), matching the HF
  feature extractor default.
- Video takes already-sampled frame files; no container/codec demuxing.
- Tokenization is O(n^2) in the number of symbols per text (fine for normal
  inputs, slow for very long documents).
- Vision attention is the current performance hotspot.
