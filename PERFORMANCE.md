# e2 - measured performance

Every number below was measured on one machine: **Intel Core Ultra 9 285H, 16 cores + Arc 140T iGPU**,
int8 weights and int8 activations on both sides of every comparison. Reference machine caveats apply
to all of it, and two of them matter more than the tables:

* **This laptop throttles.** `E2_THREADS` pinned, clocks not locked, 1.5 GHz to 4.8 GHz depending on
  power state. A single timing says nothing: the numbers are the best of 7 runs, and comparisons
  between two engines were taken by **alternating the binaries inside one session**, not across
  minutes.
* **The iGPU's reported clock is not the clock it runs at.** Two runs of the same binary can differ
  by 1.5-2x, and in one window about 3 runs in 14 came back ~4x *faster* and wrong. Only numbers from
  the same session should be compared.

## CPU, single item

Best of 7, load time excluded. llama.cpp is the official Q8_0 GGUF (4 threads), OpenVINO is the int8
IR of the same checkpoint.

| | int8, 1 thr | int8, 14 thr | bf16, 1 thr | bf16, 14 thr | llama.cpp Q8_0 (4t) | OpenVINO int8 (16t) |
| :-- | --: | --: | --: | --: | --: | --: |
| 15-token query | 16.5 ms | **4.7 ms** | 137 ms | 19 ms | 20.8 ms | 7.0 ms |
| 196-token document | 203 ms | **34.7 ms** | 1609 ms | 296 ms | 191.7 ms | 48.9 ms |
| image (2520 patches) | - | **720 ms** | - | - | 1300 ms | 1670 ms |
| image + caption | - | **720 ms** | - | - | - | - |

The 15-token query is the tightest case: a 15-token embedding is ~150 us of work spread over ~150
parallel regions, so the pool's per-region cost (a shared-counter claim plus a barrier, ~2.5 us at 14
threads) is a third of the total. Everything else has 15-20 % of headroom.

Against where this started (36.4 ms / 690 ms) the text tower is 7.2x and 19x faster; the image path is
1.6x ahead of llama.cpp's Q8_0 build and 2x ahead of OpenVINO's int8 CPU plugin, both of which pay for
a `(1,2520,2520)` attention bias and materialised `(S,S)` logits that a purpose-built tower does not
need.

The review pass that fixed a SIMD exponential bug, added the batch API and blocked the attention:

| workload | baseline median | candidate median | speedup |
| :-- | --: | --: | --: |
| query, 15 tokens | 4.64 ms | **4.30 ms** | 1.08x |
| document, 196 tokens | 35.96 ms | **35.67 ms** | ~unchanged |
| document, 514 tokens | 145.97 ms | **106.51 ms** | 1.37x |
| document, 2048 tokens | 1063.00 ms | **688.74 ms** | 1.54x |
| batch 8 x 17 | 49.05 ms | **25.04 ms** | 1.96x |
| batch 32 x 196 | 1284.33 ms | **1207.39 ms** | 1.06x |

Where the time goes now (196-token document, `E2_PROFILE=1`, which inflates every region by two clock
reads): ~65 % int8 GEMM, ~20 % attention, ~10 % elementwise/norm/copy/rope, ~2 % quantization proper.

## Fidelity

Cosine against the fp32 ONNX/PyTorch reference:

| case | int8 (default) | bf16 (`E2_INT8=0`) |
| :-- | --: | --: |
| 15-token query | 0.999470 | 0.999760 |
| 196-token document | 0.999421 | 0.999711 |
| image (2520 patches, 256 soft tokens) | 0.999744 | 0.999917 |
| image + caption | 0.999566 | - |
| text + audio (0.75 s, 19 soft tokens) | 0.999457 | 0.999714 |
| text + image + audio | 0.999479 | 0.999798 |

Every tower quantizes **activations**, not just weights, so the error compounds with depth: the vision
tower is 0.99999 at layer 0 and 0.9925 by layer 15, its 512-d soft tokens land at 0.9976, and the text
tower pulls the final embedding back to 0.999744. The audio tower behaves the same way (0.99998 at
layer 0, 0.9971 at layer 11, 0.9962 for the soft tokens, 0.999457 for the final embedding). That is
the cost of the recipe, not a modelling error - the bf16 path on the same input scores 0.999917.

For scale: the weight-only int8 models shipped elsewhere score 0.99983 (OpenVINO) and 0.99990 (GGUF
Q8_0). Quantizing activations too is the more aggressive recipe, and it lands at 0.9994-0.9998.

## The front end

Same machine, `e2_test_pre`, best of 200 / 2000:

| | time | share of the embedding it feeds |
| :-- | --: | --: |
| load `tokenizer.json` (81 ms once, ~25 MB resident) | - | - |
| tokenize a 61-byte, 15-token query | **0.74 us** | 0.015 % of 4.9 ms |
| tokenize an 8000-byte document | **545 us** (14.7 MB/s) | 1.6 % of a 35 ms document |
| expand one `<\|image\|>` + 256 soft tokens | ~3 us | 0.0004 % of 835 ms |
| PNG 512x512 -> 768x768 -> 2304 patches | **0.61 ms** | 0.08 % of 800 ms |
| PNG 1024x256 -> 1584x384 -> 2376 patches | **0.73 ms** | 0.09 % |

A 12 Mpx image, where the decode rather than the resize is the whole story (the resize is capped at
~2520 patches whatever the input, so it is 3.5 ms of the 31 ms below):

| 4000x3000 photo-like content | file | decode + resize + patchify |
| :-- | --: | --: |
| JPEG q90 | 2.4 MB | **31 ms** |
| WebP q90 | 2.8 MB | 128 ms |
| PNG | 22.5 MB | 146 ms |

libjpeg-turbo's IDCT is SIMD and ~4x faster per pixel than libwebp's, which is why a JPEG pipeline is
much cheaper to feed than a WebP one. The system codecs against the in-tree PNG reader (best of 15,
mean in brackets):

| | in-tree | libpng | |
| :-- | --: | --: | :-- |
| 512x512 flat, 6 KB | **0.96 ms** (1.86) | 1.09 ms (2.43) | in-tree faster |
| 900x600 noise, 1.6 MB | 7.39 ms (13.71) | **6.10 ms** (7.16) | libpng 1.2x |
| 4000x3000 photo-like, 22.5 MB | 149 ms (156) | **109 ms** (113) | libpng 1.35x |

The library wins where it matters (a large image is 97 % decode) and loses by ~0.1 ms where nothing
matters. That is the whole case for making it the default.

## The audio tower

Mel precomputed, tower only, int8 on both sides, measured alternately in one session:

| soft tokens | PyTorch fp32 | OV CPU fp32 | OV CPU int8 | **e2 int8 (CPU)** | OV GPU int8 |
| --: | --: | --: | --: | --: | --: |
| 19 (0.75 s) | 48.1 | 14.5 | 14.5 | **11.2** (1.30x) | 8.0 |
| 63 (2.5 s) | 104.5 | 28.6 | 29.0 | **23.2** (1.24x) | 10.8 |
| 125 (5 s) | 155.4 | 41.6 | 42.6 | **39.2** (1.09x) | 13.0 |
| 280 (11.2 s) | 282.2 | 85.9 | 85.4 | **82.9** (1.03x) | 21.5 |

That is 4.3x faster than the PyTorch reference and 1.03-1.30x faster than OpenVINO's int8 on the CPU.
`text+audio` costs 19-21 ms for the 29-token fixture; the audio-only entry point adds the text tower
over the 4 + N ids, so 12 / 23 / 40 / 86 ms of tower becomes 18 / 47 / 85 / 185 ms end to end. Soft
token fidelity is 0.9962 for the 0.75 s fixture and 0.9901 for an 11.2 s clip.

## The packed model cache

| | `model.safetensors` | `E2_Q8_WRITE` cache |
| :-- | --: | --: |
| on disk | 1420 MB | **916 MB** |
| load | 0.52 s | **0.01 s** |
| peak RSS, text-only query | 884 MB | **180 MB** |

## The Intel GPU

All six engines on the same inputs, from one session. The e2 GPU column is why the backend exists.

| workload | e2 CPU (14 thr) | **e2 GPU** | OpenVINO CPU int8 (16 thr) | OpenVINO GPU int8 | llama.cpp CPU Q8_0 | llama.cpp GPU (Vulkan) |
| :-- | --: | --: | --: | --: | --: | --: |
| text query, 15 tok | **4.7** | 7.1 | 7.0 | **6.4** | 21-30 (4 thr) | 31.0 |
| text document, 196 tok | 34.7 | **13.5** | 48.9 | - | 192 (4 thr) | - |
| text batch 8 x 17 | - | **9.4** (854 seq/s) | 29.6 (270/s) | 10.7 (747/s) | - | - |
| text batch 32 x 196 | - | 392 (82/s) | 2005 (16/s) | **280** (114/s) | - | - |
| image, 2520 patches | ~716 | **189** | 1671 | 341 | 1298 (4 thr) | 458 |
| image + caption | ~716 | **190** | 1762 | 345 | 1313 (4 thr) | - |
| fidelity (cos vs fp32) | 0.9994-0.9997 | 0.9994 | 0.9998 | 0.9998 | 0.9999 text / 0.9916 image | 0.99 |

The GPU is the fastest thing measured on a document, a short text batch and an image, and within 11 %
of OpenVINO's GPU plugin on a query. It is 1.40x behind the plugin on the long text batch, where the
cost scales with the key count, because the projections are still send-bound at 7-30 % of the matrix
engine. On the image it is 1.8x faster than the plugin and 3.8x faster than the CPU engine.

**These are timings, not acceptance.** The backend is experimental: intermittent wrong or zero outputs
remain unresolved, and the API can currently return success with a wrong result. Treat the table as
"what the hardware does when it behaves", not as a correctness claim.

## Reproducing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DE2_MODEL_DIR=$PWD/models/embeddinggemma-2
cmake --build build -j

# Structured timings; the affinity mask is for the reference machine. Load time is excluded.
E2_THREADS=14 taskset -c 0-13 ./build/e2_text_bench models/embeddinggemma-2 tests/fixtures cpu 17 8 21

# Best of N, which is the only meaningful figure on a throttling laptop
E2_REPS=7 ./build/e2_test models/embeddinggemma-2 tests/fixtures

# Serial, alternating A/B against OpenVINO INT8 on matched inputs
python3 tools/bench/bench_compare.py --model-dir models/embeddinggemma-2 --fixture-dir tests/fixtures
```

`E2_PROFILE=1` prints a wall-clock breakdown of one embedding per parallel region, `E2_DUMP=<dir>`
writes raw f32 intermediates for bisection, and `E2_GPU_PROFILE=1` brackets every device kernel with
timestamp writes.
