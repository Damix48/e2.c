# e2 vs OpenVINO

`bench_compare.py` runs the same frozen inputs through e2 (CPU and Intel GPU) and through OpenVINO
weight-only INT8 IRs, alternating the engines by round, and checks every call against the same FP32
reference embedding. It writes `results.json`, `REPORT.md` and every raw output into a new
`benchmark-results/<name>/` directory (never overwrites an earlier run).

## Cases

| case | input | batching |
| :-- | :-- | :-- |
| `query` | 15-token query ids | 1 |
| `document` | 196-token document ids | 1 |
| `query_batch8` | 8 x 15-token queries | real batched call (`e2_text_embed_batch`) |
| `document_batch32` | 32 x 196-token documents | real batched call |
| `image` | 2520 padded patches | 1 |
| `image_batch4` | same image | 4 sequential calls in one timed call |
| `audio` | 0.75 s clip (12000 samples, 19 soft tokens) | 1 |
| `audio_batch4` | same clip | 4 sequential calls in one timed call |

The vision and audio towers have no batched entry point in either engine, so their batch cases repeat
the single call and measure throughput, not a batched kernel; the text batch cases are genuine.

## Running

```bash
~/Projects/embeddinggemma-2/.venv/bin/python tools/bench/bench_compare.py \
  --model-dir ~/Projects/embeddinggemma-2 --fixture-dir tests/fixtures --build-dir build-compare \
  --cases query,document,query_batch8,document_batch32,image,image_batch4,audio,audio_batch4 \
  --devices CPU,GPU --runs 21 --warmup 5 --rounds 3 \
  --output benchmark-results/$(date +%F)-e2-vs-ov
```

- `--runs 21 --warmup 5 --rounds 3` is the full setting; expect roughly 25-35 minutes on the
  reference laptop, most of it in `image_batch4` on the CPU.
- `--runs 7 --warmup 3 --rounds 2` finishes in about 6 minutes and is enough for a direction check.
- The first audio case exports the OpenVINO audio IR with PyTorch and NNCF (about a minute) and leaves
  roughly 2 GB of ONNX intermediates in the output directory.
- The audio IR bakes the fixture's tail mask in, so the export must be redone if `audio.bin` changes.

## Reading the result

`REPORT.md` reports the median of the round medians and the worst per-call cosine. Speedup is
`OV / e2`, so values above 1 favor e2. A cell whose accuracy misses the cosine gate keeps its timings
and is marked `ACCURACY FAIL`; the run exits non-zero.

Boundaries: OpenVINO is NNCF `INT8_SYM` weight compression, not calibrated W8A8, so activation and
attention precision differ between the engines. Timings exclude model load and graph compilation.
Numbers are only comparable inside one session; the laptop throttles and the iGPU's reported clock is
not its real clock.
