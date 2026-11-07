# Benchmarks

Host: AMD Ryzen 7 9700X (8 cores / 16 threads), 54 GB RAM, Debian 13,
shared with other workloads (numbers are median of 3, so treat them as
indicative). OpenVINO 2026.4.1, 8 inference threads. e2 uses 8 threads
(physical cores), int8 weights, int8 activations.

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

| task | length | e2 int8 | OpenVINO f32 | OpenVINO int8 |
|---|---:|---:|---:|---:|
| text | 8 tokens | 7.2 ms | 9.5 ms | **5.5 ms** |
| text | 64 tokens | 43.7 ms | 13.0 ms | **10.7 ms** |
| text | 128 tokens | 91.0 ms | 19.0 ms | **17.2 ms** |
| text | 512 tokens | 502 ms | 71.8 ms | **62.6 ms** |
| text | 2048 tokens | 2821 ms | 524 ms | **501 ms** |
| text batch | 8 x 64 | 350 ms | 68.6 ms | **49.3 ms** |
| text batch | 32 x 64 | 1382 ms | 236 ms | **206 ms** |
| image 640x480 | 270 tok | 3209 ms | 556 ms | **520 ms** |
| audio 2 s | 54 tok | 130 ms | 38.3 ms | **28.5 ms** |
| video 8 frames | 1058 tok | 9551 ms | 1897 ms | **1681 ms** |

Text is competitive at 8 tokens (e2 beats OV f32) and falls behind as the
sequence grows: e2 is ~2-4x slower at 512 tokens and ~5x at 2048, where OV
int8 and f32 converge (the 2048-token encoder is memory-bound). Image, audio
and video are 5-6x behind OpenVINO. The vision attention (bidirectional,
~2.4k patches) is the dominant cost in e2 and is memory-bound in the current
4-query-row kernel; the audio path is dominated by many small-M int8 GEMMs.

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
