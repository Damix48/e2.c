# e2 - performance log

Every optimization attempt lands here, accepted or rejected, with numbers. This file is the record of
what was tried and why it stands or was reverted; the tables are the only place a "faster" claim is
allowed to live.

## Guidelines

- **Complexity is only paid for with speed.** Extra branches, a new kernel, a new `E2_*` knob or a new
  allocation must show a measured win in this file. No numbers, no merge.
- **Reject tiny speed gains that add complexity; prefer substantial wins.** A 2 % gain that costs 100
  lines is a regression in everything but wall clock.
- **Between two options that are equally fast, take the smaller diff** - or no change at all.
- **An attempt that failed is worth as much as one that worked.** Write it down with the reason, so it
  is not retried.
- Record the *cause*, not just the effect: "no measurable change" is useless without the explanation
  of why the time was not there to win.

## Testing

- **Quick iteration** (one embedding, no fixtures): `E2_PROFILE=1 E2_THREADS=14 ./build/e2_test <model> tests/fixtures`
- **Before committing**: `ctest --test-dir build` must stay green, and a change touching arithmetic must
  move a fidelity number in `tests/`, not just pass.
- **Load time is excluded** from every figure here; it is measured separately.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DE2_MODEL_DIR=$PWD/models/embeddinggemma-2
cmake --build build -j

# Structured timings; the affinity mask is for the reference machine.
E2_THREADS=14 taskset -c 0-13 ./build/e2_text_bench models/embeddinggemma-2 tests/fixtures cpu 17 8 21

# Best of N, which is the only meaningful figure on a throttling laptop
E2_REPS=7 ./build/e2_test models/embeddinggemma-2 tests/fixtures

# Serial, alternating A/B against OpenVINO INT8 on matched inputs
python3 tools/bench/bench_compare.py --model-dir models/embeddinggemma-2 --fixture-dir tests/fixtures
```

Reference machine: **Intel Core Ultra 9 285H, 16 cores + Arc 140T iGPU**, int8 weights and activations.
Two caveats hold for every number below, and they are why the method is spelled out:

- **The laptop throttles** (1.5 GHz to 4.8 GHz by power state). Numbers are the best of N, and any
  two-engine comparison alternates the binaries inside one session.
- **The iGPU's reported clock is not the clock it runs at.** Two runs of the same binary can differ by
  1.5-2x. Only numbers from the same session are comparable.

## Pipeline

```
text:    tokenizer -> 24 layers (PLE, q/k/v norms, 5:1 sliding/global window) -> mean pool -> 768d L2
vision:  resize/patchify (16x16x3) -> 16 layers, axial 2D RoPE -> 3x3 pool -> 280 soft tokens
audio:   16 kHz mono -> log-mel (128 bins, hop 160) -> 12 layers, chunked local attention -> 750 tokens
loop:    <|image|>/<|audio|> placeholders expanded -> one sequence -> text tower -> 768d L2
```

Four workloads stand for the whole engine in the tables: a **15-token query** (the parallel-overhead
case: ~150 us of work spread over ~150 regions), a **196-token document** (GEMM-bound), an **image**
(2520 patches) and a **text+audio** clip.

## Current Baseline

Fill this in on the machine you are on, before touching anything; an attempt without a baseline is a
guess. Format: one block per reference machine, dated, with the command that produced it.

```
## YYYY-MM-DD / <machine>, <threads> threads, int8
| workload              | best of N | notes |
| :-------------------- | --------: | :---- |
| 15-token query        |       -   |       |
| 196-token document    |       -   |       |
| image (2520 patches)  |       -   |       |
| text+audio (30 s)     |       -   |       |
```

### 2026-10-26 / Core Ultra 9 285H, Arc 140T (0x7d51), 14 CPU threads, int8

Baseline is commit `6edaa1c`; candidate is the Intel attention / small-GEMM change below. Release,
GCC 16.2.1, existing Level Zero runtime and OpenCL C kernels. Three alternating baseline/candidate
rounds, 5 warmups + 21 timed calls each, same frozen inputs. These are **best warm call latencies**,
not load times; batch figures cover the entire batch, including the existing 2048-row chunking.

| workload | baseline ms | candidate ms | speedup |
| :-- | --: | --: | --: |
| 15-token query | 5.217 | 4.613 | 1.13x |
| 196-token document | 28.425 | 12.734 | 2.23x |
| 8 x 15-token queries | 9.041 | 8.202 | 1.10x |
| 32 x 196-token documents | 902.047 | 368.775 | 2.45x |
| image, 2520 patches + text tower | 217.531 | 187.708 | 1.16x |

Reproduce matched input generation / OpenVINO comparison with:

```bash
~/Projects/embeddinggemma-2/.venv/bin/python tools/bench/bench_compare.py \
  --devices GPU --cases query,document,query_batch8,document_batch32,image \
  --build-dir build-gpu-perf --runs 21 --warmup 5 --rounds 3 \
  --output benchmark-results/intel-final
# Build 6edaa1c in a separate directory, then alternate its e2_compare_bench with the candidate:
# e2_compare_bench MODEL gpu benchmark-results/intel-final/CASE.bin 21 5 OUTPUT.f32
```

The generated `ab-results.json`, `*-fidelity.log`, `results.json`, per-worker logs and `REPORT.md`
are in the ignored `benchmark-results/intel-final/`. No fixtures were regenerated.

#### OpenVINO INT8-weight comparison (same session)

OpenVINO **2026.4.1**, NNCF `INT8_SYM` weight compression, GPU latency hint, one stream; this is
**not calibrated W8A8**. Both engines must pass cosine >= 0.999 against the same FP32 fixture on
every call. Inputs, warmups and repetitions as above, engines alternated by round. Median column
is the median of three round medians; best column is the best of all measured calls.

| workload | e2 best ms | OV best ms | e2 median ms | OV median ms |
| :-- | --: | --: | --: | --: |
| query | 4.681 | 4.912 | 4.734 | 5.148 |
| document | 12.959 | 9.733 | 13.112 | 9.955 |
| query batch 8 | 8.411 | 6.929 | 8.495 | 7.182 |
| document batch 32 | 369.487 | 256.363 | 369.986 | 263.427 |
| image | 187.732 | 339.320 | 188.604 | 342.438 |

**The overall OpenVINO target is not met.** e2 wins this query and image comparison, but documents
and text batches remain slower. The query margin is small on a variable-clock laptop; the image
margin is substantial. No OpenVINO audio timing or other Intel GPU was tested in this session.

### 2026-10-27 follow-up / same Arc 140T, 14 CPU threads

Baseline here is the **working tree after Attempts 1–7**, not `6edaa1c`. Its benchmark executable was
saved before this follow-up (SHA256 `19cec3b78a145f232dddb14dc5c1c3f089a924ec9117486c68d78f33f696ec10`).
Three alternating baseline/candidate rounds, 5 warmups + 21 timed calls, Release, identical frozen
inputs and default int8 engine settings. Load/compilation excluded. Best warm call latencies:

| workload | before ms | after ms | speedup |
| :-- | --: | --: | --: |
| query, 15 tokens | 4.573 | 4.235 | 1.08x |
| document, 196 tokens | 12.803 | 10.884 | 1.18x |
| query batch 8 | 8.247 | 7.318 | 1.13x |
| document batch 32 | 369.261 | 263.164 | 1.40x |
| image | 187.539 | 182.949 | 1.03x |

This is **15% lower document latency and 29% lower document-batch latency**, not a claim that every
case now beats OpenVINO. The final same-session OpenVINO 2026.4.1 INT8_SYM comparison is:

| workload | e2 best ms | OV best ms | e2 median ms | OV median ms |
| :-- | --: | --: | --: | --: |
| query | 4.291 | 4.969 | 4.362 | 5.154 |
| document | 10.964 | 9.780 | 11.019 | 9.980 |
| query batch 8 | 7.330 | 6.916 | 7.470 | 7.138 |
| document batch 32 | 263.850 | 256.052 | 264.900 | 263.715 |
| image | 182.991 | 338.921 | 184.015 | 342.141 |

Median is the median of three round medians. The 32-document batch is now approximately tied on
median latency, **not a demonstrated win**; OpenVINO still has the better best call. Single documents
and query batches remain about 10% and 5% slower on median. No frequency/power settings were changed.

Reproduce with the earlier comparison command, changing the output directory to
`benchmark-results/intel-doc-cache`. That ignored directory contains `results.json`, `REPORT.md`,
`ab-results.json`, the saved baseline executable/hash, every checked output and worker log.
`benchmark-results/intel-doc-final` also retains intermediate profiles/compiler metadata and the
pre-cache-cap comparison. Other Intel devices and OpenVINO audio were not benchmarked.

## Already Optimized

What is in the tree and stays unless a measurement says otherwise:

- Runtime ISA dispatch: AVX512-VNNI -> AVX-VNNI -> AVX2 -> scalar, with the scalar path as reference.
- Claim-from-counter worker pool instead of fixed slices, so no region idles at the tail.
- Blocked, packed-operand attention; no materialised `(S,S)` logits or `(1,S,S)` bias.
- Fused regions (softmax+quantizer, PLE, q/k/v norm) - single-consumer kernels by contract.
- Packed int8 cache: 916 MB, ~0.01 s load, replaces the checkpoint entirely.
- In-tree front ends: BPE, image preprocessing, log-mel with an in-tree FFT, PNG/DEFLATE - no ONNX,
  no Python at run time.
- mmap'd bf16 weights, demand-paged; load time excluded from the numbers above.

## Optimization Attempts

One section per attempt. Number them, tag them `ACCEPTED` or `REJECTED`, and give each the numbers that
decided it. Accepted attempts keep their measured delta, so the file also documents where the current
speed came from.

```
### Attempt N: <what was tried> (ACCEPTED|REJECTED)

- What changed, in one or two lines (files, mechanism).
- Baseline -> candidate, per workload, with thread count and reps.
- Result: **<x> ms -> <y> ms (<z> % faster)** or "within noise".
- Why: the cause of the win or of the null result. If rejected, what would make it work instead.
```

### Attempt 1: parallel-key FP32 attention (REJECTED)

- Replaced the serial online-softmax key loop with parallel per-lane key dots, local probabilities,
  and a second context pass; no XMX. Single 196-token document, best of 11, 8 CPU threads:
  **30.082 -> 32.891 ms**. CPU/GPU cosine remained 0.999726.
- Parallelizing keys removed the per-key subgroup reduction but left full head-dimension dots on
  vector ALUs and introduced strided key loads/local-memory traffic. It did not solve matrix
  throughput. Removed, rather than keeping another attention switch.

### Attempt 2: FP16 XMX attention, FP32 accumulation and softmax (ACCEPTED)

- `src/gpu/kernels/e2.cl`: pack K and transposed V once per layer into FP16 DPAS operands; compute
  QK and PV on XMX with FP32 accumulation, stable FP32 softmax, FP16 probabilities. Projection
  weights **and activations remain int8**. Attention itself is not advertised as int8.
- `src/gpu/gpu.c`, `gpu.h`: use this path for sequences >= 64; retain fused FP32 online attention
  for shorter sequences. Independently pad each sequence so odd lengths cannot mix batch members.
  Reuse fixed scratch across layers and graph shapes; no new runtime, dependency or tuning knob.
- Initial unpacked FP16 prototype: **30.082 -> 17.879 ms** on a 196-token document. Converting and
  gathering K/V inside each product still cost ~7.46 ms over 48 launches. Packing once reduced
  the products to ~1.93 ms plus ~0.32 ms packing, and end-to-end to **12.466 ms** (best of 11,
  8 CPU threads). Final alternating end-to-end figures are in the baseline table above.
- Earlier baseline profiling attributed ~55-60% of document GPU time to serial FP32 attention.
  Moving the reductions/products to XMX, rather than only reducing traffic, removes that bottleneck.
- Cost: **104 MiB additional fixed device scratch** (64 MiB scores, 32 MiB probabilities, 8 MiB
  packed K/V), including for small-query-only contexts. Allocation uses the existing checked
  workspace/error path. The worst-case 2048-row/key bound is deliberate; no quadratic per-call
  allocations or cached per-shape buffers.
- `tests/test_gpu_attn_f16.c` adds an independent FP64 oracle for both head dimensions/GQA layouts,
  64/65/129/2048 lengths, odd batches, holes, all-masked rows and the 1024-token sliding window.
  It checks scores, probabilities and every context dimension of sampled query rows. Maximum
  relative context L2 error across these shapes: **0.000647**.
- All **40 CTest tests passed**, checkpoint enabled, including CPU backends, GPU image/audio,
  lifecycle, masking, cache turnover and soak tests. The 42-case frozen FP32 text suite (lengths
  1..2048) changed minimum cosine **0.999372565 -> 0.999365535**; `length_129` improved
  **0.999372565 -> 0.999425075**, with **8/8** retrieval top-1 matches before and after. Document
  fixture cosine **0.999655274 -> 0.999676775**; image **0.999827402 -> 0.999834450**.
  Accuracy remains above 0.999, not bit-identical to the old attention. A separate GPU-disabled
  Release build also passed all **14** checkpoint-free CTest tests.

### Attempt 3: int8 projection token tiles (ACCEPTED for short text; larger tiles REJECTED)

- `e2_gemm_dpas2` now has two compile-time-specialized wrappers sharing the same implementation:
  8 rows for <=128-row text launches, the original 16 rows otherwise. Vision keeps its 16-row
  tile and audio its existing specialized projection path. Preserve the old-GEMM diagnostic switch.
- A universal 32-row tile was slower: document **12.410 -> 16.044 ms**, query batch 8
  **8.746 -> 10.419 ms**, query **~4.4 -> 6.950 ms**. More live accumulators and fewer groups
  outweighed weight reuse; reverted.
- A universal 8-row tile also lost on documents (**12.410 -> 14.366 ms**), but won on queries
  (**~5.43 -> 4.403 ms**) and query batch 8 (**8.746 -> 7.928 ms**). Keep only short-launch
  specialization: less padding and more independent groups, with no arithmetic change. Final
  alternating baseline/candidate improvements are 13%/10% in throughput for these two cases.

### Attempt 4: factor projection weight scales out of the block accumulation (REJECTED)

- Moved the per-output weight scale from every 128-element block to the final store. Alternating
  binaries, best of 21: document **12.396 -> 13.135 ms** (8 CPU threads). CPU/GPU cosine changed
  **0.999756614 -> 0.999713066**, still passing, but slower.
- This changed the floating-point grouping and final epilogue instead of removing the block-scaled
  accumulation dependency. Source-level multiply savings did not translate into a faster compiled
  kernel; the exact instruction-level regression was not established. Reverted completely.

### Attempt 5: launch-group and attention cutoff sweeps (REJECTED)

- With packed FP16 attention, projection group sizes 8/16/32 measured **13.429/12.977/12.407 ms**
  on a document (best of 21, 8 CPU threads). Keep the existing size 32 for this shape; no new knob.
- FP16 attention group size 8 vs 4: **12.478 vs ~12.410 ms**, no win. Keep 4.
- Starting FP16 attention at length 8 instead of 64: query **4.445 vs 4.391 ms**, query batch 8
  **7.910 vs 7.929 ms** (best of 15). Extra launches cancel the faster products for these short
  sequences. Keep the simpler short-query precision/launch behavior at the 64-token cutoff.

### Attempt 6: sixteen-query FP16 attention tile (REJECTED)

- Reused each packed K/V operand for two query blocks per subgroup. Document **12.410 -> 12.257 ms**
  (best of 21), only ~1.2% for extra loops, guards and accumulator state. The attention products
  are already a small fraction of the optimized graph; revert the added complexity.

### Attempt 7: remove tiny-text split-K projection dispatch (ACCEPTED)

- Use the new 8-row DPAS tile for 1..4-row text launches too; delete the split+reduce host branch and
  its now-unused `tiled` parameter. Existing standalone GEMV benchmark kernels stay available.
- Best of 15, 8 CPU threads: 1 token **4.596 -> 4.003 ms**, 4 tokens **4.918 -> 4.029 ms**. Each
  projection now needs one launch instead of two and no 16-token GEMV scratch/reduction. CPU/GPU
  cosine at 4 tokens **0.999547227 -> 0.999619797**; final full CTest and FP32-reference runs pass.

### Attempt 8: identify the remaining gap and remove redundant quantization work (ACCEPTED)

- Baseline document profiling put ~53% of kernel time in int8 projections, ~10% in row operations,
  ~10% in GELU, and ~7% in Q/K/V normalization. A 1960-row batch chunk took ~115 ms: projections
  ~45 ms, row operations ~18 ms, normalization ~9 ms and GELU ~13 ms. Timestamp instrumentation
  and clock changes affect absolute times; use these to locate work, not to compare engines.
- OpenVINO runtime profiling confirmed optimized `jit:gemm:any__i8` projections and FP16 transforms.
  It reported ~3.73 ms of int8 GEMMs for one document and ~65.7 ms for batch 32. Operator fusion and
  activation precision differ, so these are not one-for-one kernel comparisons. `ov-profile.json`
  records all nodes rather than inferring performance from weight storage alone.
- Native compiler metadata exposed 288 bytes of per-thread spill in `k_gemm_dpas2` and 3328 bytes
  in the original generic Q/K/V normalization. The original GELU shader contained over 1000 ISA
  lines, including the general-purpose tanh implementation. This was not primarily host submission
  or tokenization time.
- Fixed an earlier mistake: the FP16 attention's quantize-only call requested an extra RMS norm of
  context. That is not part of model attention, even though the following post-attention norm largely
  cancels its scale. Also skip the first quantization when `k_rowops` will overwrite it after its
  second normalization. Screening: document **12.419 -> 12.390 ms**, too little to be the main fix;
  removing the erroneous normalization is retained for correctness. Attempt 12 removes this extra
  context pass entirely. The attention test checks raw context as well as final embeddings.

### Attempt 9: model-sized normalization and bounded fast GELU (ACCEPTED)

- Specialize the normalization helper on the model's two head sizes, 256/512, rather than carrying
  dynamic head-dimension branches through every element. Keep the original FP32 normalization and
  rotary arithmetic. Separate named kernels for the two sizes added no useful win and were removed.
- Replace the general tanh inside GELU with the equivalent bounded expression
  `copysign((1-exp(-2*abs(u)))/(1+exp(-2*abs(u))), u)`, using `native_exp`. Its exponential input is
  nonpositive, so this does not introduce positive-exponent overflow. The native exponential is also
  used for the nonpositive differences in stable text softmax. No activation quantization granularity
  was relaxed.
- GELU alone screened at **13.066 -> 12.181 ms**, with batch 32 **371.192 -> 355.599 ms**; head-size
  specialization then gave **~12.18 -> 11.69 ms** on the document in that clock state. Native softmax
  by itself was around noise (~11.59 -> 11.49 ms), not credited with a separate end-to-end speedup.
  Final accepted numbers are the alternating table above, not sums of these prototype deltas.
- `tests/test_gpu_ffn.c` independently checks scalar tanh-based GELU, per-128 scales and packed bytes.
  Maximum relative debug-store activation error in its nonzero cases: **2.64e-6**.

### Attempt 10: targeted large register mode (ACCEPTED; blanket mode REJECTED)

- `src/gpu/l0.c` asks Intel's compiler for 256 GRFs only for the `k_gemm_dpas_large` name prefix.
  A separate entrypoint uses exactly the existing 16x16 projection math for launches above 512 rows;
  the fused FFN shares the prefix. The ordinary/short kernels retain 128 GRFs and higher occupancy.
- Compiler metadata confirms 128 GRFs / 288 bytes spill in the normal GEMM, but **256 GRFs and no
  reported spill** in the bulk GEMM and fused FFN. The context fusion stays at 128 GRFs with no
  reported spill. This is a compiler build option, not a new runtime environment switch.
- Before FFN fusion, targeted large mode screened batch 32 at **355.4 -> 328.5 ms**. Forcing large
  mode on every kernel only moved a document **13.08 -> 12.65 ms**, with batch still ~372 ms, and
  auto-large mode gave ~13.03 ms / 370 ms. Lower occupancy elsewhere cancelled spill savings.
- Bulk projection group sizes 8/16/32 measured **347.3/328.8/336.3 ms** on batch 32; keep the
  existing size 16. For query batch 8, sizes 32/16/8 gave **7.026/7.418/7.665 ms**; keep 32.

### Attempt 11: fuse FFN projections, GELU/product and int8 quantization (ACCEPTED)

- `k_gemm_dpas_large_gate` is specific to the model's two 2048x512 FFN projections. Both projections
  share input loads; 16 queries x 128 output features stay in registers/8 KiB of workgroup SLM
  through GELU and per-128 quantization. No FP32 gate/up tensor round trip during normal inference.
- Retain output clamps and debug writes. Existing DP4A/old-GEMM diagnostics take their separate
  kernels. Weights and activation block sizes are unchanged; no wider/coarser quantization was used.
- Screening after targeted GRF selection: batch 32 **328.8 -> 300.1 ms**. The full 32x196 batch
  avoids **4.59 GiB of logical global-memory gate/up writes and reads** across 24 layers. This is
  algorithmic traffic eliminated, not a measured DRAM bandwidth counter. One fewer launch per layer.
- Added `tests/test_gpu_ffn.c` and CTest wiring: independent scalar dot/GELU/quantization oracle,
  zero rows, distinct block scales, both clamps, 1/15/17/32 rows, padding canaries, replay and debug
  stores. All stages must match, not merely the final embedding's unit norm.
- An 8-query fused-FFN variant was slower: query batch 8 **7.020 -> 7.297 ms**, query
  **4.047 -> 4.073 ms**. Additional groups/weight reads outweighed register savings; removed.

### Attempt 12: fuse context GEMM and per-128 quantization (ACCEPTED)

- `k_attn_f16_ctx` owns an entire 128-feature block per workgroup. It stages the FP32 context in
  4 KiB of SLM and emits int8 activations/scales directly, without storing/re-reading FP32 context.
  Debug mode still writes raw context. Padding, GQA, arbitrary masks and per-sequence bounds remain.
- Screening: batch 32 **~300 -> 286.1 ms**; document **10.626 -> 10.470 ms**. Another 24 dependent
  launches disappear. Avoided logical context writes/read/rewrite total **2.01 GiB** for batch 32.
  The graph shrinks from **439 to 391 kernels per chunk** across the two accepted fusions.
- Extended the independent FP64 attention test to execute the fused kernel and compare its
  quantized bytes/scales against a separate quantizer of the checked raw context. It covers both
  head sizes, odd sequences/batches, all-masked rows and the 2048-token/sliding-window boundary.
  Neither fusion adds fixed device workspace; only per-workgroup SLM.

### Attempt 13: alternative projection implementations (REJECTED)

These are same-session exploratory screens, not separate speed claims. Failed prototypes were
removed; the final paired comparison above is the acceptance measurement.

- Fixed-K wrappers (512/1024/2048): document **12.405 -> 12.522 ms**. Constants alone did not fix
  operand/epilogue costs; removed the extra entrypoints and host dispatch arguments.
- SLM-staged input kernel: **11.083 -> 12.950 ms**, batch ~382 ms. Its per-block synchronization
  and staging outweighed input-load reuse. Keep it only in the existing standalone microbenchmark.
- Universal 16x8 and 32x8 tiles: documents **~11.12 -> 12.08/13.11 ms**, batches **~355 ->
  401/417 ms**. More groups/input traffic lost more than narrower accumulators saved.
- A 32x16 projection tile with blanket 256-GRF mode was ~11.68 ms / 352 ms, essentially no
  document improvement; targeting large GRF only to that tile gave ~333 ms on batch versus
  **329 ms** for the simpler 16x16 bulk kernel. Retain 16x16.
- 32-bit element-offset arithmetic: document **11.045 -> 11.176 ms**, no win. Removed.
- Subgroup block-read weight layout: document ~12.19 ms / batch ~361 ms versus ~11.1 ms / ~355 ms
  at that stage. Repacking did not improve the generated kernel; restored the original weight
  layout, with no cache-format changes or extra weight copy left behind.
- Per-row 512-element activation scales plus an integer-dot-only epilogue: document **11.669 ->
  12.526 ms**, batch ~375 ms. It also reduced the 42-case minimum cosine to **0.999008424**.
  A 32-query variant stayed slower (~12.75 ms / 370 ms). Rejected for both speed and accuracy
  margin; no per-row quantization flag or dispatch survives.
- Split-subgroup DPASW was not available in this device/driver's advertised OpenCL extensions or
  compiler builtins. The prototype did not build; no invented throughput number or unsupported
  intrinsic was retained.

### Attempt 14: extra producer fusions and wider FP16 tiles (REJECTED)

- Fusing PLE gate/GELU/quantization by generalizing the FFN kernel gave only **10.758 -> 10.689 ms**
  on a document and ~300 -> 296 ms on batch 32, while query latency regressed **4.078 -> 4.169 ms**.
  Smaller register mode did not fix it. Remove the generalized helper, extra entrypoint and dispatch.
- Packing FP16 K/V during normalization reduced batch **286 -> 276 ms**, but slowed a document
  **10.467 -> 10.654 ms**. Avoiding that regression would require separate kernels/dispatch and
  more padding/write-layout machinery. Not enough return for the added complexity; removed.
- A 16-query x 32-key FP16 attention tile at 128 GRFs gave **10.457 -> 10.492 ms** on a document,
  with batch ~255 ms after the cache cap versus ~262 ms. At 256 GRFs it gave ~10.61 ms / 255 ms.
  The small batch-only gain did not pay for larger accumulators, key padding and quantizer geometry;
  retain the smaller 8x16 tile. These prototypes did not change the accepted arithmetic.

### Attempt 15: runtime ordering and cache overrides (REJECTED)

- In-order command lists without the explicit per-kernel barriers stayed around **11.1 ms**, no
  useful improvement over the existing candidate. Ordering alone did not remove the tensor traffic
  or projection work. Removed rather than adding driver-version fallback/test complexity.
- Explicit `-ze-opt-load-cache-default=4` gave **~11.19 -> 11.21 ms**, no win; removed.
- A cached-allocation hint did not rescue the unsuccessful block-read candidate. It was not
  independently established as a win and was removed with that experiment. No cache-control,
  queue-ordering, device-frequency or power overrides are installed.

### Attempt 16: remeasure the batch working set after fusion (ACCEPTED)

- Before fusion, smaller chunks gave only minor differences: 2/4/8/10 documents measured
  **21.58/42.86/91.13/111.14 ms**. Do not infer the optimal chunk size once and freeze it forever.
- After fusion, 2/3/4/5/6/7/8/9/10 documents measured approximately
  **18.90/25.42/33.88/40.55/50.24/58.99/69.04/80.21/89.05 ms**. Five documents per chunk has
  materially better time per document than ten: active Q/K/V/projection buffers stay smaller.
- Cap the batch working set at **1024 rows**, preserving whole sequences. A single sequence can
  still use the full 2048-row workspace. A 32x196 batch now uses 5/5/5/5/5/5/2 sequences rather
  than 10/10/10/2. Alternating binaries, best of 11: **285.530 -> 261.910 ms**, unchanged cosine.
- No environment knob or larger allocation. Removed the unused mutable `max_rows` field; shape
  tests now use the batch cap and still check 1025/2048-token sequences and mixed/empty masks.
  This wins despite *more* submissions: working-set locality, not fewer submissions, explains it.
- Removed stale hard-coded OpenVINO timings from `test_gpu_text.c`; they are not a matched benchmark.

### Follow-up validation

- **41/41 checkpoint-enabled CTest tests pass**, including CPU backends, GPU image/audio, lifecycle,
  chunk turnover, masks and soak. The GPU-disabled Release build also passes **14/14** tests.
- Default GPU, forced DP4A and old-GEMM paths all pass the 42-case frozen FP32 text suite and **8/8**
  retrieval pairs. Minimum cosine changed **0.999365535 -> 0.999395676**; `length_129`
  **0.999425075 -> 0.999447720**. No fixtures were regenerated.
- All benchmark calls, including every member of batch 32, pass cosine >= 0.999 versus the frozen
  FP32 reference. On the document fixture the cosine changes **0.999676775 -> 0.999613757**;
  improved minimum over the larger corpus does not mean every individual case improves.
- Arithmetic and register selection remain experimental on other Intel GPUs. These timings do not
  establish retrieval quality on a large corpus, long-run stability, or an OpenVINO audio speedup.

### Attempt 17: bounded int8-attention retry (ACCEPTED for sequences >=1024)

The old unmasked stage recipe agrees with its independent FP64 oracle; no indexing defect was
found there. Its probability quantizer nevertheless wastes half the unsigned activation range:
softmax probabilities are nonnegative, so use **0..255 with zero point 0**, not 0..127 biased by
128. The context product must then omit `128 * bsq`; Q and the legacy vision wrapper still need
that correction. Masked queries now explicitly write zero probabilities and block scales.

- Reused the existing int8 K/V packer and attention GEMM. Q is quantized **without another RMS
  normalization** after its learned head norm/RoPE; context also gets quantization only. An ablation
  that normalizes Q again fails the frozen suite at **0.998851267** minimum cosine. This establishes
  an integration hazard, not proof of the cause of the historical length_129 failure.
- With correct quantize-only integration and int8 attention from 64 tokens, the old probability
  recipe now passes all 42 frozen FP32 cases, minimum **0.999048227**. Full-range unsigned P raises
  that minimum to **0.999385973**; both match **8/8** retrieval pairs. Thus the earlier accuracy
  rejection is not a blanket limitation of int8 attention. No fixtures changed.
- Universal dispatch was rejected: at 196 tokens, alternating best-of-11 calls measured
  **11.092 -> 11.514 ms**. Batch 32 had a small best-call gain (**265.203 -> 257.367 ms**), but
  a later pair was **265.905 -> 302.115 ms**. This is not a reliable medium-document win.
- Keep int8 only at **1024..2048 tokens**. FP16 remains at 64..1023; short FP32 attention is
  unchanged. Seven attention launches replace four on long sequences, but the faster products
  outweigh quantization/packing there. No new kernel, environment switch or persistent allocation:
  int8 operands/metadata fit inside the existing K/V scratch, and P/scales inside the probability
  scratch, with explicit capacity checks. Masked context rows are quantized to neutral bytes too.

Matched final timing: Arc 140T, 14 CPU threads, checkpoint-loaded int8 weights, identical repeated
fixture ids, three warmups, best of seven calls in each of two alternating/reversed-order rounds.
Load and graph recording are excluded. Each call checks cosine >=0.999 against the CPU embedding.

| single sequence | previous FP16 path, ms | long-only int8 path, ms | latency reduction |
| :-- | --: | --: | --: |
| 196 tokens (still FP16) | 10.963 | 11.029 | noise / unchanged arithmetic |
| 1024 tokens | 84.511 | 66.090 | 21.8% |
| 2048 tokens | 274.627 | 204.496 | 25.5% |

Both rounds win on long sequences: 1024-token pairs **84.511/66.090** and **91.015/67.904 ms**;
2048-token pairs **274.627/222.640** and **293.037/204.496 ms**. Clocks still vary materially.
The final dispatch passes **41/41** checkpoint-enabled CTests and **14/14** GPU-disabled CTests.
Default, forced DP4A and old-GEMM projection paths pass all 42 frozen FP32 cases and 8/8 retrieval
pairs, with unchanged overall minimum **0.999395676**. The updated independent recipe test covers
24 shapes; an additional equal-logit oracle checks unsigned-P bytes/scales, masked queries/keys,
padding and batch/head/nonzero query-tile offsets.

The optional `e2_gpu_bench --check` passes its attention and three individual GEMM checks, but its
legacy multi-tensor checks fail. The same failures reproduce with the original pre-retry SPIR-V;
that unrelated harness was left unchanged. Logs and timings are in the ignored
`benchmark-results/intel-i8-attention/` directory. Saved baseline benchmark SHA256:
`ea674ac4142600bf98028ec88bd60f7e4967e36a9276eee3defabc548698a420`.
This is a gain over e2's FP16 path, **not a new OpenVINO comparison**; short/medium-document gaps
remain. Other Intel GPUs and broad retrieval quality are not validated.

### Forced INT8 / FP16 / FP32 attention comparison

Follow-up measurement, not another dispatch change. Three benchmark-only copies of `gpu.c` force
one attention implementation at **every** sequence length, including short queries. They link the
same kernel module and CPU library. Projection weights/activations remain int8 and softmax remains
FP32 in all three; this is **not** an all-FP32-model benchmark. FP16/int8 products use XMX while
FP32 uses the online vector kernel, so these compare implementations, not only operand types.

Arc 140T, 14 CPU threads, identical int8 text cache and repeated fixture ids. Each case has three
rotating-order rounds, each with three warmups and seven timed calls; the table is the best timed
call across those rounds, in milliseconds. Load/graph recording are excluded. Every embedding,
including each batch member, passes cosine >=0.999 against the CPU result. All three forced modes
also pass the 42-case frozen FP32 suite and 8/8 retrieval pairs: minimum cosine INT8
**0.999385973**, FP16 **0.999395676**, FP32 **0.999361253**.

| tokens/sequence | batch | forced INT8, ms | forced FP16, ms | forced FP32, ms | INT8 latency vs FP16 |
| --: | --: | --: | --: | --: | --: |
| 15 | 1 | 5.008 | 4.492 | 4.450 | +11.5% |
| 64 | 1 | 7.264 | 6.158 | 7.650 | +18.0% |
| 128 | 1 | 10.067 | 8.850 | 14.072 | +13.7% |
| 196 | 1 | 11.582 | 11.023 | 25.985 | +5.1% |
| 256 | 1 | 14.544 | 14.580 | 38.189 | -0.2% |
| 512 | 1 | 28.171 | 36.591 | 129.560 | -23.0% |
| 1024 | 1 | 71.809 | 82.148 | 409.288 | -12.6% |
| 2048 | 1 | 202.392 | 271.673 | 1121.132 | -25.5% |
| 15 | 8 | 9.582 | 7.144 | 7.427 | +34.1% |
| 196 | 32 | 255.221 | 263.425 | 721.454 | -3.1% |

This corrects the overly broad claim that int8 loses at all lengths below 1024: **512 already wins**
in every round (28.762/36.591, 28.171/36.886, 31.682/36.697 ms, INT8/FP16). At 256 they are tied.
The 1024 production threshold is conservative, not a measured optimum; it was left unchanged for
this measurement request. FP16/FP32 single-query differences are small; the document-batch int8
advantage is modest and not present in every round. Laptop clock variation still matters.

Reproducer, forced host copies, binary hashes, all 90 timing logs and frozen-reference results are
in ignored `benchmark-results/intel-attention-3way/` (`compare.py`, `results.json`). Production
source/build files were not modified. Other GPUs and OpenVINO were not benchmarked in this sweep.

### Attempt 18: fuse int8 attention producers/consumers (ACCEPTED from 512 tokens)

- `k_norm_rope3_i8` quantizes Q per 128 elements and packs K with its existing per-head scale and
  per-128 zero-point sums while normalized/rotated values are in registers. It explicitly zeros
  masked queries/keys and padded K rows. The old FP32/FP16 normalizer uses the same arithmetic with
  compile-time-disabled quantization; raw Q/K stores remain available for dumps.
- `k_attn_i8_ctx` fuses PV with per-128 context quantization through **8 KiB of workgroup SLM**;
  raw context stores are debug-only. This removes three standalone launches per layer: attention
  goes **7 -> 4 launches**, 72 fewer over 24 layers. No persistent memory or environment knob was
  added. Removed the now-unused context branch/argument from the score wrapper; the legacy vision
  explicit-pointer entry point retains its original zero-point behavior.
- Both fusions earn their keep together on larger workloads. Same-session alternating best-of-7
  ablations at 32x196: context-only **250.281 ms**, Q/K-only **241.229 ms**, both **235.238 ms**;
  at 2048 tokens: **197.249 / 193.629 / 189.706 ms**. At a single 196 tokens Q/K-only was worse
  (**11.641 ms**) than context-only (**11.337 ms**); both were **11.414 ms**. These ablations precede
  deletion of the unused score-wrapper branch, so do not mix them with the final timing session.

A batch-dependent precision selector was **rejected**. At `S>=192 && rows>=512` the same masked
196-token input used FP16 as a single and int8 in a batch: their cosine fell to **0.979665686** on
`gpu_shapes`' arbitrary IDs/noncontiguous masks, failing its **0.999999** consistency requirement.
Forcing the same int8 implementation for both passes the unchanged requirement. This was not a
packing/indexing defect, and a frozen natural-text gate is not evidence that arbitrary inputs are
insensitive to arithmetic mode. Do not lower the test threshold or quietly change recipes with
batch size. Production therefore selects **int8 at S>=512**, FP16 at 64..511, FP32 below 64.
Single 196-token documents and their batches retain FP16; the faster forced-int8 batch is not
silently installed. `gpu_shapes` now checks noncontiguous-mask single/batch equality at both 196
and 512 tokens, including int8 chunk turnover.

Final matched session: Arc 140T, int8 text cache, 14 CPU threads, three rotating-order rounds,
three warmups and best of nine calls per round, identical repeated fixture IDs, all outputs checked
against CPU cosine >=0.999. Load/recording excluded; best call across rounds, milliseconds:

| input | before default | after default | reduction |
| :-- | --: | --: | --: |
| 15x1 | 4.031 | 4.019 | noise / unchanged FP32 |
| 196x1 | 10.507 | 10.524 | noise / unchanged FP16 |
| 196x32 | 262.724 | 259.852 | noise / unchanged FP16 |
| 512x1 | 31.677 | 25.689 | 18.9% (FP16 -> fused int8) |
| 1024x1 | 65.501 | 60.725 | 7.3% (int8 -> fused int8) |
| 2048x1 | 201.755 | 185.663 | 8.0% (int8 -> fused int8) |

The forced-int8 comparison isolates the fusions/score cleanup from dispatch: 196x1
**11.057 -> 10.781 ms** (2.5%, still slower than FP16), 196x32 **253.777 -> 230.468 ms** (9.2%),
512 **27.228 -> 25.707 ms** (5.6%), 1024 **65.385 -> 60.662 ms** (7.2%), 2048
**201.560 -> 185.744 ms** (7.8%). This did not make int8 the winner for a short single document.
The probe at 128x4 gave only **23.942 -> 23.360 ms**, 129x4 regressed **22.539 -> 22.713 ms**,
and 64x8 regressed **22.386 -> 23.624 ms**; there is no justification for universal int8 dispatch.

Validation: **42/42 checkpoint-enabled CTests**, **14/14 GPU-disabled CTests**; default, forced
DP4A and old-GEMM projections pass all 42 frozen FP32 text cases and 8/8 retrieval pairs. Overall
minimum cosine improves **0.999395676 -> 0.999410217** with the new cutoff; no fixtures regenerated.
New `gpu_norm_i8` independently checks FP64 RMSNorm/RoPE values, Q/K quantization, scales, block
sums, masks, padded keys, both head sizes, two sequences and both debug-store modes. The attention
recipe test now validates fused context bytes/scales (including zero V heads) as well as its
independent FP64 context oracle.
After the final padding-pointer safeguard, the five affected CTests were rebuilt/rechecked and pass.
Logs, before snapshots, ablation executables, binary hashes and `final_compare.py` are under ignored
`benchmark-results/intel-i8-fusion/`. Other GPUs, broad retrieval quality and a new OpenVINO
comparison remain untested. GPU text FP32 removal was deliberately kept out of this experiment.

### Attempt 19: remove Intel GPU FP32 text attention (REQUESTED SIMPLIFICATION)

Removed the OpenCL `k_attn` kernel and its `E2_ATTN_BODY` macro, the host dispatch branch and
its obsolete microbenchmark/`E2_BENCH_ATTN_GY` knob. Text attention now uses **FP16 at 1..511 tokens**
and **int8 at 512..2048**, independent of batch size. FP32 softmax, accumulation, normalization and
the CPU reference remain; the separate HIP backend is untouched. No replacement tuning knob.

The FP16 stage test now covers 1/2/7/8/15/63-token shapes, both head sizes, tiny GQA batches and
all-masked queries against its independent FP64 oracle. Query samples are clamped to the short
sequence and the all-zero oracle case is checked without dividing by zero. The graph-cache test
requires distinct 1/64-token graph entries with the same FP16 kernel count, rather than assuming
the removed FP32 route has a different count.

**42/42 checkpoint-enabled and 14/14 GPU-disabled CTests pass.** The frozen FP32 suite still
passes all 42 cases and 8/8 retrieval pairs, minimum cosine **0.999410217**. Representative
`length_15` cosine changes **0.999642831 -> 0.999668159**; individual cases can improve or worsen,
so this is not a blanket accuracy claim. No fixtures regenerated.

Same-session alternating binaries, Arc 140T, 14 CPU threads, identical int8 text cache/fixture IDs,
three warmups, best of nine across three rounds, load/recording excluded, every output checked:

| input | before, ms | after, ms | result |
| :-- | --: | --: | :-- |
| 1x1 | 3.599 | 3.929 | +0.330 ms / 9.2% slower |
| 15x1 | 4.075 | 4.041 | essentially unchanged |
| 15x8 | 7.043 | 6.725 | 4.5% lower latency |
| 196x1 | 10.551 | 10.579 | unchanged FP16 / noise |

This is an explicit complexity reduction, **not a universal speedup**: the one-token case pays
for four FP16 attention launches instead of one fused FP32 launch. Kept as requested rather than
adding a new single-token specialization. Before/after executables, hashes, timings and fidelity
logs are in ignored `benchmark-results/intel-fp16-only/`. Other GPUs and OpenVINO were not measured.

## Next Targets

- **15-token query**: the tightest case. Pool claim + barrier (~2.5 us per region at 14 threads) is a
  third of a ~4.7 ms run. Fewer regions, or fusing the small ones, is the only lever left.
- **Intel GPU remaining gap**: document batches now approximately match OpenVINO's median, while
  single documents and query batches still trail. The remaining work is projection microkernel
  throughput and FP32 normalization/activation traffic. Avoid more cache/launch knobs: measured
  producer/consumer fusions and better arithmetic scheduling must pay for any further complexity.
  Short-query wins still need repeated same-session confirmation.
- **Audio tower**: chunked local attention width is a fixed 12; nothing measured above or below it.
- **Load time**: ~0.01 s from the packed cache, 1.4 GB checkpoint read otherwise. Only matters for
  cold-start latency, not for throughput.

## Credits attribution rules

- Ideas, kernels and approaches are only taken from BSD / MIT licensed code.
- If an optimization idea or kernel comes from another project, credits go into the source file that
  uses it.
