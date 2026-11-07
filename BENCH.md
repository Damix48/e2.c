# Benchmarks

Host: Intel Core Ultra 9 285H (6 P + 8 E + 2 LP-E cores, no AVX-512,
AVX-VNNI, Intel Arc 140T iGPU), Debian 13, shared with other workloads
(numbers are min/median of 6, so treat them as indicative). OpenVINO 2026.4.1,
8 inference threads. e2 uses 8 threads, int8 weights, int8 activations.

Reproduce:

```bash
make
make tests/e2_bench_full
. .venv/bin/activate
uv pip install openvino nncf
python tools/bench_compare.py --iters 3 --repeat 3
```

`tools/bench_compare.py` generates deterministic assets in `/tmp/e2bench`
(image, video frames, 16 kHz WAV, texts tokenised to the requested lengths),
runs the e2 binary, and converts the text/vision/audio towers with
`openvino.convert_model`, optionally with NNCF `INT8_SYM` weight compression.
OpenVINO image/video/audio are tower(s) + text encoder at the same sequence
length; e2 includes tokenizer / preprocess and the text encoder. `--repeat`
pools the samples of several passes so a busy host cannot fake a result.

## Speed

| task | length | e2 int8 | OpenVINO int8 | e2 / OV |
|---|---:|---:|---:|---:|
| text | 8 tokens | **2.6 ms** | 6.5 ms | 0.40 |
| text | 64 tokens | **11.2 ms** | 21.5 ms | 0.52 |
| text | 128 tokens | **21.3 ms** | 43.4 ms | 0.49 |
| text | 512 tokens | **83.2 ms** | 207.0 ms | 0.40 |
| text | 2048 tokens | **420.1 ms** | 1374.2 ms | 0.31 |
| text batch | 8 x 64 | **68.8 ms** | 152.2 ms | 0.45 |
| text batch | 32 x 64 | **282.8 ms** | 604.6 ms | 0.47 |
| image 640x480 | 270 tok | **599.5 ms** | 1465.1 ms | 0.41 |
| audio 2 s | 54 tok | **32.0 ms** | 57.0 ms | 0.56 |
| video 8 frames | 1058 tok | **2681.1 ms** | 5020.6 ms | 0.53 |

OpenVINO f32 is slower than its int8 on every row and is omitted. Every row
beats OpenVINO int8 by at least 1.8x (goal was 1.5x).

On the Arc 140T iGPU (`--sources e2,ov-int8-gpu --e2-device gpu`) e2 is
currently 3.5-11x **slower** than OpenVINO GPU: e2 runs one Level Zero
submission per GEMM, which costs ~0.2 ms of dispatch plus ~0.4 ms of host wait
per call, while OpenVINO compiles the whole graph and submits once per
inference. Numbers and the analysis are in PERFORMANCE.md ("GPU (Level Zero)").
The GPU backend is opt-in (`E2_DEVICE=gpu`) and correct (cosine 1.0000 text,
0.9999 image, 0.9988 audio vs the CPU path).
`--repeat 3` pools 9 samples per row; alone, the same-run ratio still moves
+/-10% with host load (see PERFORMANCE.md).

## Accuracy

Cosine similarity of the final normalized embedding against the HuggingFace
float32 implementation:

| input | e2 int8 |
|---|---:|
| text | 0.9997 |
| image (PNG) | 0.9994 |
| image (JPEG q95) | 0.9928 |
| image (WebP lossless) | 0.9994 (identical to PNG) |
| audio 2 s, 16 kHz WAV | 0.9994 |
| audio 2 s, FLAC | identical to WAV |
| audio 2 s, OGG | 0.9959 vs WAV |
| audio 44.1 kHz -> 16 kHz | 0.9965 vs librosa/scipy reference |
| video 4 frames | 0.9987 |
| text+image / text+audio / both | 0.9990-0.9991 |

## OpenVINO conversion notes

- `openvino.convert_model` traces with `torch.jit.trace`. The trace bakes the
  attention masks and the sliding-window branch, so each benchmarked sequence
  length needs its own conversion: reshaping a 2048-token trace down to 8/64/
  512 tokens gives cosine 0.50-0.92 against a native trace.
- Every conversion runs in its own worker process. The PyTorch frontend keeps
  the traced graph alive after the model is dropped (~0.8 GB per conversion,
  ~6 GB at 2048 tokens), which otherwise grows a full run past 20 GB.
- The audio tower's mask construction does not trace. The benchmark converts
  `subsample_conv_projection` and the layer stack separately, with the blocked
  attention mask precomputed in Python. Times are summed.
- Video runs the vision tower once per frame on both sides: the frames are
  independent (their position ids index the frame) and tracing all frames at
  once needs 8x the memory for the same work.
- NNCF weight compression is data-free (`INT8_SYM`) and does not quantize
  activations. A calibrated `nncf.quantize` would likely be faster still.
