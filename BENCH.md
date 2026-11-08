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

## GPU (Intel Arc 140T iGPU, Level Zero)

The GPU backend is opt-in (`E2_DEVICE=gpu`). Every linear, the MLP block, the
attention stack (QK^T, masked softmax, PV) and the norms run on the Xe matrix
engine with int8 DPAS, **one command-list submission per embedding**. Every GEMM
result matches the CPU int8 kernel bit for bit, and the final embedding is
cos 0.9997 against the CPU path.

Round 10 numbers. This box was busy (load average 3-8) and the same binary
measured 64, 73, 87 and 77 ms for text 512 in four consecutive runs, so read
the *ratios*, not the absolute values:

| task | length | round 9.1 | **round 10** | ov-int8 GPU* | e2 / OV |
|---|---:|---:|---:|---:|---:|
| text | 8 | 13.1 | **12.8** | 5.9 | 0.46 |
| text | 64 | 16.4 | **15.4** | 7.3 | 0.47 |
| text | 128 | 21.1 | **20.1** | 8.0 | 0.40 |
| text | 512 | 70.7 | **63.1** | 19.2 | 0.30 |
| text | 2048 | 422.5 | **385** | 134.5 | 0.35 |
| image | 270 tok | 501.5 | **442** | 263 | 0.60 |
| audio | 54 tok | 36.1 | **35.4** | 23.9 | 0.68 |
| video | 8 frames | 2164 | **2107** | 794 | 0.38 |

`*` the OV column is round 9.1's: the HF checkpoint the converter needs is no
longer on this box, so OV could not be re-measured in the same run. The e2
columns are a same-run interleaved A/B of the two e2 binaries (min of 5-6 runs).

Round 10 is **2-12% faster than round 9.1 and 2.8-31x faster than round 8**
(text 512 was 535 ms in round 8, 70.7 in 9.1, 63.1 now). It is still **1.5-3.3x
slower than OpenVINO int8 on the GPU, so the 1.5x goal is not met** — that would
need another 2.2-4.9x.

What round 10 established, all measured (details and the experiment log are in
PERFORMANCE.md "Round 10"):

* **launch count is not the problem.** Halving it (1102 -> 628 kernels per
  forward) changed nothing measurable; a launch costs ~1 us without a barrier
  and ~5 us with one.
* **the f32 activation stream is.** Fusing the MLP's gate/up/gelu into one
  int8-producing kernel (so its 8 MB-per-layer f32 intermediate never exists)
  took the GEMM phase at text 512 from 85 to 49 ms.
* **the GEMM kernel in isolation is fine, the same GEMMs in the real forward are
  not**: `make gpu-bench` with `E2_GB_LAYERS=1` runs the real per-layer sequence
  with cold weights in 7.1 ms at m=512, while the same launches inside the
  forward cost ~49 ms. That 3-7x gap is the top open question.
* **the vision attention is the whole image/video cost**: 240 ms of 442, because
  at 1200 patches the materialised score matrix is 74 MB per layer.

`make gpu-bench` prices the GEMM phase in isolation (`E2_GB_LAYERS=1`,
`E2_GB_YBUF=N`, `E2_GB_RSG/CSG` to match the tile the SPV was built with);
`tools/ab.sh` is the only way to A/B anything on this host; `E2_GPU_TS=1` prints
per-kernel device timestamps, `E2_GPU_SKIP=linear,norm,...` keeps an op group on
the CPU, `E2_GPU_NOLAUNCH=1` keeps only the GEMM launches, `E2_GPU_NOATTN=1`
prices the attention.

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
