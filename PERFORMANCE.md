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

### 2026-10-28 / Oracle Cloud Ampere A1 (Neoverse-N1), 2 vCPU, 11 GB, aarch64, scalar backend

No baseline existed for a non-x86 host; this is the first one. Release, GCC 15.2.0, no GPU backend.
Only the **scalar** backend exists here: `CMakeLists.txt` compiles the AVX2/VNNI/AVX-512 translation
units for x86-64 only, and `e2_cpu_init()` leaves `e2_i8_quantize == NULL`, so `text.c` runs the bf16
GEMM fallback — **no int8 path at all on aarch64**, even though Neoverse-N1 has `asimddp`. The
built-in probe reports **11.8 scalar GFLOPS**.

```
E2_THREADS=2 ./build/e2_text_bench models/embeddinggemma-2 tests/fixtures cpu 15 1 5
E2_THREADS=2 ./build/e2_text_bench models/embeddinggemma-2 tests/fixtures cpu 196 1 3
E2_THREADS=2 ./build/e2_test models/embeddinggemma-2 tests/fixtures
```

| workload | best ms | median ms | notes |
| :-- | --: | --: | :-- |
| 15-token query | 373.141 | 373.632 | best of 5 |
| 196-token document | 5650.346 | 5710.530 | best of 3 |
| 2 x 15-token queries | 811.871 | 815.562 | best of 3, batch 2 |
| 2 x 128-token documents | 7439.877 | 7441.639 | best of 3, batch 2 |

Fidelity is bit-identical to the x86 reference on both fixtures (`cos=1.000000`, `max|Δ|=0.00000`), so
the scalar backend is numerically the reference it claims to be — it is just 81x / 444x slower than
the Intel laptop's int8 + iGPU numbers above (4.613 ms and 12.734 ms), which is what losing both int8
and SIMD costs. Thread scaling is poor: 1 -> 2 threads moves the 128-token pair only 8740.562 -> 7439.877 ms
(1.17x) and the 15-token pair 842.290 -> 811.871 ms (1.04x), i.e. the scalar path is memory-bound
well before it is core-bound.

`E2_PROFILE=1` on the 196-token document puts ~91-96 % of the profiled region time in `attention`
(the fused q/k/v/o + score region) and <5 % in `norm`/`gelu`/`elem`/`rope`. Load time is 0.03 s and
the workspace is 230.2 MB.

These are the **bf16** numbers: aarch64 had no int8 path and no NEON backend when this block was
written. The table stays as the baseline the next two attempts are measured against. Attempt 20 adds
a NEON dot-product int8 GEMM (**30.4 ms** / **1281.7 ms**, 12.3x / 4.4x), Attempt 21 adds a NEON
fp32 GEMM for the attention products (**30.4 ms** / **441.7 ms**, 12.3x / 12.8x) and Attempt 22 moves
text attention itself to int8 (**30.4 ms** / **396.9 ms**, 12.3x / 14.2x).

### 2026-10-29 / AMD Ryzen 7 9700X (Zen 5), 8 threads, int8

x86-64 with **full-width AVX-512** (`avx512f/bw/vl/dq/vnni/bf16/vbmi/...`), GCC 14.2.0, Zen 5 desktop,
54 GB. The pool reports 8 threads on 16 hardware threads on purpose - `e2_nthreads()` drops SMT
siblings, and its comment records that a second logical thread per core measured 1.7x slower here.
Weights come from the packed cache (`E2_Q8_READ`, 0.01 s load); the AVX2-only laptop numbers further
up therefore also ran the same int8 weights, so the gap below is the CPU alone.

```
E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq E2_REPS=7 ./build/e2_test models/e2 tests/fixtures
E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text-image.e8cq ./build/e2_test_image models/e2 tests/fixtures
```

| workload | best ms | notes |
| :-- | --: | :-- |
| 15-token query | **3.951** | best of 7, cos 0.999751 (3.91 on a rerun) |
| 196-token document | **24.918** | best of 7, cos 0.999594 (25.96 on a rerun) |
| image, 2520 patches | **542.830** | best of 3, cos 0.999795 (569.74 on a rerun) |
| image + caption (274 ids) | 542.120 | cos 0.999665 (582.55 on a rerun) |

Rerun spread is 2-5 % on this host, larger on the image than on text; treat these as best-case
latencies, consistent with how the other machines in this file are recorded.

Cross-machine, same packed int8 weights, best of N, load excluded:

| workload | Ampere A1 (2 vCPU, NEON) | Core Ultra 9 285H (AVX2+VNNI) | Ryzen 7 9700X (AVX-512) |
| :-- | --: | --: | --: |
| 15-token query | 30.4 ms | 6.19 ms | **3.95 ms** |
| 196-token document | 380.5 ms | 41.41 ms | **24.92 ms** |
| image, 2520 patches | 8496 ms | 708 ms | **543 ms** |

The laptop comparison is 6 pinned threads against 8 here, so part of the 1.6x is thread count; the
instruction set is the other part (AVX-512-VNNI at 512 bits against AVX-VNNI at 256).

`E2_PROFILE=1` on the image puts attention at 250.2 ms (38 %), **elem at 144.4 ms (22 %)**, mlp 118.3
(18 %), qkv+o 72.1 (11 %), attn_pack 46.1 (7 %). The `elem` share roughly doubles against the Ampere
numbers purely because everything else got faster: those ops are the memory-bound elementwise vtable,
and Attempt 23 already established that GCC vectorizes them, so they become the second-largest item on
a fast machine without being a bug. GEMM regions run at 6.61x parallelism on 8 threads.

The text attention crossover was re-measured here because the fp32 GEMM underneath is much stronger
than on the other two machines. With best of 9 and two rounds: 196 keys fp32 27.61/26.95 vs int8
27.14/26.50 (int8 ~2 % ahead), 256 keys break-even, 384 keys fp32 58.01/56.60 vs int8 51.07/52.01
(+10-12 %). The single-run `reps=7` pass first read 196 keys as 5 % *slower* for int8; that was noise,
which is the same lesson as the earlier x86 correction two attempts up. `E2_ATT_I8_MIN` = 128 is
therefore correct on this machine and needs no per-CPU tuning.

`avx512_fp16` is **not** in this CPU's flags, so even here there is no fp16 arithmetic to exploit and
the fp16-attention idea from the ARM discussion stays untestable on any machine in this set.

**14/14 CTests pass** (no GPU backend: Level Zero and HIP are both absent on this host).

### 2026-10-29 / Intel N100 (Alder Lake-N, 4x Gracemont), 4 threads, int8

Debian, GCC, 4 GB RAM, a fanless 6 W mini-PC on a Proxmox host. `avx2 + avx_vnni + f16c`, **no
AVX-512**, so this takes the same 256-bit VNNI path as the Core Ultra 9 285H. Weights from the packed
cache (0.02 s load). `e2_nthreads()` reports 4 - four physical cores, no SMT.

| workload | best ms | notes |
| :-- | --: | :-- |
| 15-token query | **22.7** | spaced runs, stable to +-1 % |
| 196-token document | **329.4** | spaced runs, stable to +-1 % |
| 512-token document | 957 - 2401 | **not reproducible, see below** |
| image, 2520 patches | not measured | the 600 MB cache was not worth the transfer for this box |

#### This box has to be measured with gaps, and long sequences cannot be measured at all

The first attempt reported 601 then 760 ms for the 196-token document and 22.5 then 63.3 ms for the
query - swings that look like contention. They are not: the VM shows **zero steal time**, no other
process, `performance` governor, and an idle temperature of 49 C. The cause is the power limit. Idle
freq is **3400 MHz**; after about seven seconds of sustained load all four cores sit at **998 MHz**
with the temperature only rising to 51 C. It is a 6 W part, so it is PL1, not thermals.

Re-running with 20 s between invocations gives 329.4 / 331.8 / 330.8 / 329.6 ms and 22.69 / 22.81 /
23.00 ms - stable to about 1 %. So only burst-length work is measurable here. A 512-token document is
~7 s per invocation and came out 957 ms and then 2401 ms for identical work; that number is reported
as a range because it is not a measurement, and anything longer is worse.

Attention A/B at 196 keys, spaced, three alternations: fp32 356.6 / 358.6 / 358.9 ms against int8
342.8 / 332.3 / 389.6 ms. int8 is ahead on two of three but the spread exceeds the difference, so this
machine neither confirms nor contradicts the 128-key threshold; it is not the machine to settle that
question on.

**13/14 CTests pass.** The `artifacts` test fails only because it drives `tools/check_gpu.sh`, which
runs `git rev-parse HEAD` under `set -eu` and git is not installed on this host - the same failure
mode as the arm-server, which had the `.git` directory missing instead.

#### The default Intel GPU backend does not build for this box, and the reason is the hardware

It has an iGPU and a full Level Zero stack: `Alder Lake-N [UHD Graphics]`, device id **0x46D1**,
OpenCL 3.0 NEO, 24 CUs, `intel-level-zero-gpu` 1.3.30872.22 plus `libze_loader`, and `ocloc` present.
`-DE2_WITH_GPU=ON` configures cleanly and auto-detects the target
(`SPIR-V for 0x46d1`), but the ocloc step then fails with three errors:

```
unknown OpenCL extension 'cl_khr_integer_dot_product' - ignoring
no matching function for call to 'dot'
implicit declaration of function 'intel_sub_group_i8_u8_matrix_mad_k32' is invalid in OpenCL
implicit declaration of function 'intel_sub_group_f16_f16_matrix_mad_k16' is invalid in OpenCL
```

Two independent causes, both Xe-LP:

- **No XMX.** `intel_sub_group_{i8_u8,f16_f16}_matrix_mad_*` are the DPAS matrix instructions, and
  Gen12/Xe-LP has no matrix units. The `#ifdef cl_intel_subgroup_matrix_multiply_accumulate` guard in
  `e2.cl` does not save it because ocloc defines that macro for this target while not actually
  providing the intrinsics.
- **Whoever compiles for `0x46d1` does not accept `cl_khr_integer_dot_product`**, so `dot()` - the
  dp4a the int8 GEMM is built on - has no definition. The *device* advertises the extension in
  clinfo; the ocloc target does not. That mismatch is a toolchain question, but it is moot given the
  first point.

So the backend stays Arc-class code and was validated only on an Arc 140T (0x7d51). On this box the
CPU path is the engine, and that is what the numbers above are. The experimental no-XMX build below is
the text-only exception. Nothing was left behind: the trial `build-gpu/` was removed and the CPU build
is untouched; `libze-dev` (1.20.6) stayed installed, which is the only system change made.
`apt purge libze-dev` undoes it.

#### Experimental no-XMX GPU path (2026-10-29)

`-DE2_GPU_NO_XMX=ON` compiles the same kernel source with `-cl-std=CL3.0 -DE2_NO_DPAS`: every DPAS
kernel is excluded, the projections use the existing `k_gemm_tiled` dp4a path, and text attention
falls back to f32 kernels. Text only; the vision and audio kernels are not in the module. The f32
kernels are behind `#ifdef E2_NO_DPAS`, so the default DPAS module does not carry them. The build is
default OFF, and the experimental build disables the non-text `gpu_*` CTest tests.

Measured on the N100 with the text int8 cache, four CPU threads pinned, one text call, spaced:

| workload | CPU, 4 threads | no-XMX GPU | gap |
| :-- | --: | --: | :-- |
| 15-token query | 23.9 ms, 390 % CPU, 0.093 core-s | 41.9 ms, 100 % CPU, 0.042 core-s | 1.75x slower, 2.2x less CPU |
| 196-token document | 337.6 ms, 392 % CPU, 1.323 core-s | 521 ms, 100 % CPU, 0.521 core-s | 1.54x slower, 2.5x less CPU |

The GPU path leaves three cores free and pins one core in the Level Zero wait; a three-hog test left
the document time unchanged. Fidelity: cosine 0.999738 (query) and 0.999643 (document) against fp32.
Document profile: projections 258.5 ms, scores 122.0, context 121.3, softmax 13.1, norm/rowops/gelu/
pool 17.5. The 16x16 SLM scores tile is the accepted tuning (208.6 -> 122.0 ms); a 16x16 context tile
and a four-feature vector context did not beat the scalar context. An int8 DP4A attention path (575
ms) and a fused 16-query attention kernel (740 ms) were tried and rejected - attention is not
compute-bound at these shapes, so neither the precision change nor the fusion paid.

Limits: text only; batch 32x196 exceeds the 30 s GPU wait; the 6 W N100 throttles under repeated long
calls, so the 521 ms figure is a spaced run (sustained repeats can drop to 2-4 s). The test binary
skips the batch section in this build (`E2_GPU_TEST_SINGLE`).

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

- Runtime ISA dispatch: AVX512-VNNI -> AVX-VNNI -> AVX2 -> NEON dot-product -> scalar, with the
  scalar path as reference.
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

### Attempt 20: aarch64 NEON int8 GEMM and activation quantizer (ACCEPTED)

Until this point aarch64 had **no int8 path**: `CMakeLists.txt` compiled the AVX2/VNNI/AVX-512
kernel files only for x86-64 and `e2_cpu_init()` left `e2_i8_quantize NULL`, so `use_int8` was false
and the text tower ran bf16 weights through pure scalar fp32 GEMMs. New `src/kernels_i8_neon.c`
provides `e2_i8_gemm_neon` and `e2_quantize_blocks_neon` behind an `HWCAP_ASIMDDP` check, wired in by
one new branch in the `e2_cpu_init()` chain. No format change: same quantization scheme, same
quad-interleaved weights, same `bsq`, so the packed `*.e8cq` cache is shared with x86 untouched.

`vdotq_s32` (4 int8 products into 4 int32 lanes) is the ARMv8.2 dot product. Two details differ from
the x86 kernels and are worth knowing:

- **`vdotq_u32` is unsigned x unsigned in ACLE**, not the mixed u8 x i8 form `vpdpbusd` has; that is
  `vusdotq_s32`, which needs the i8mm extension. So the kernel flips the stored `a+128` back to
  signed with a scalar `xor 0x80808080` folded into the broadcast, uses `vdotq_s32`, and therefore
  **never reads `bsq`** - the accumulator already is the true `sum_k a_i8*b_i8`, so the 128*sum trick
  the VNNI/AVX2 kernels need disappears instead of costing an extra `vshlq`/`vsubq` per block.
- **MR is 4, not 8.** Each token holds two int32 and two float accumulators (one per 4-row half of
  the 32-byte weight vector), i.e. 4 of the 32 NEON registers; 8 tokens want all 32 before the weight
  and scale vectors get one. Measured below: 6 and 8 are a tie on the query and 1.7 % **slower** on
  the document, 2 is 17 % slower on the query.

Ampere A1 / Neoverse-N1, 2 vCPU, release GCC 15.2, `E2_THREADS=2`, load excluded, best of 5 (query)
and best of 3 (document) unless noted. The before column is the bf16 scalar path from the reference
block above, same session.

| workload | bf16 scalar, ms | NEON int8, ms | speedup |
| :-- | --: | --: | --: |
| 15-token query | 373.141 | 30.431 | **12.3x** |
| 196-token document | 5650.346 | 1281.653 | **4.4x** |

Fidelity stays inside the bar the engine holds itself to: end-to-end cosine **0.999776** (query) and
**0.999670** (document) against the frozen fixtures, `PASS` in `e2_test`. `e2_test_int8` cross-checks
the NEON kernel against the float reference on six shapes and it matches `e2_i8_gemm_scalar` to six
decimals in every one (`neon cos` == `scalar-gemm cos` in the log). The packed-cache load path is
unchanged but cold load grows 0.03 s -> 0.88 s because the weights now actually get quantized.

MR sweep, three alternating rounds, same session, best of 7 (query) / 3 (document):

| MR | query, ms | document, ms |
| --: | --: | --: |
| 2 | 36.069 | 1337.332 |
| **4** | **30.431** | **1281.653** |
| 6 | 32.656 | 1298.482 |
| 8 | 30.471 | 1300.551 |

`E2_PROFILE=1` shows where the remaining time is, and it is no longer the GEMM:

| region | 15-token query | 196-token document |
| :-- | --: | --: |
| gemm | 27.37 ms (84 %) | 292.8 ms (22 %) |
| quantize | 2.55 ms (8 %) | 28.6 ms (2 %) |
| attention | 1.48 ms (5 %) | 983.3 ms (75 %) |

The document case is now dominated by the **fp32 scalar attention**, because `e2_i8_attn_gemm` and
`e2_softmax_quant_i8` are still NULL on ARM - the fused packed-operand attention kernel has no NEON
version yet, so both towers keep their fp32 attention. That is the next win here and it is a much
larger piece of work than this one; the GEMM side has little left (the probe reports 11.9 GFLOPS and
the GEMM region already claims work at 1.9x on 2 threads, i.e. it is memory-bound).

**14/14 CTests pass on aarch64** (`int8` no longer skips: it now has a quantizer to test) and
**21/21 on x86-64**, so no x86 regression. `e2_test_cpu_backends` grew a NEON case and reports
cosine **0.999785** (query) / **0.999680** (document) through the whole tower on that backend.

### Attempt 21: aarch64 NEON fp32 GEMM for the attention products (ACCEPTED)

Attempt 20 left the 196-token document 75 % fp32 scalar attention, because both attention products -
logits `Q K^T` and context `P V` - go through `e2_sgemm_nn_range`, which on aarch64 was
`e2_sgemm_nn_range_scalar`: a naive i-j-k triple loop with a strided B read, i.e. the worst possible
access pattern for the one product the tower spends most of its time in.

New `src/kernels_neon.c` provides `e2_sgemm_nn_range_neon`. NEON is mandatory in the aarch64 baseline,
so this file needs **no `-march` flag and no CPUID probe** - unlike the AVX2/AVX-512 backends it is
always available, and `e2_cpu_init()` selects it unconditionally on that architecture. The structure
is deliberately identical to `kernels_avx2.c`: 64x64x64 blocking with the A and B panels in L1 and an
8x8 register tile walking them, so the two files can be read against each other. The tile holds 16
`float32x4` accumulators - twice the AVX2 count for the same tile, because a NEON register is half as
wide, which the 32-register aarch64 file can afford.

Ampere A1 / Neoverse-N1, 2 vCPU, `E2_THREADS=2`, release GCC 15.2, load excluded, three alternating
rounds, best of 5 (query) / 3 (document):

| workload | bf16 scalar | NEON int8 GEMM | + NEON fp32 GEMM | total speedup |
| :-- | --: | --: | --: | --: |
| 15-token query | 373.141 | 30.431 | **30.367** | **12.3x** |
| 196-token document | 5650.346 | 1281.653 | **441.696** | **12.8x** |

The query is untouched (it was already 85 % int8 GEMM, with only 4 % attention); the document drops
**2.9x**. End-to-end cosine against the frozen fixtures moves 0.999670 -> 0.999628 because the k loop
is now blocked, i.e. the fp32 accumulation order changed; both are far above the 0.999 bar.

#### Rejected: three attempts to cut the int8 kernel's activation-broadcast overhead

The int8 kernel spends, per k-quad, 3 uops (load, `eor 0x80808080`, `dup`) on the activation broadcast
for only 2 `sdot`. Loading 8 or 16 activation bytes at once and taking the broadcasts out of
`vdupq_lane` / `vdupq_laneq` should cut that to 1.5 uops per `sdot`. All three variants were **slower**,
consistently, and are not in the tree:

| variant | query, ms | document, ms |
| :-- | --: | --: |
| baseline (original per-quad broadcast) | **30.6** | **444.2** |
| 4 k-quads hoisted, 8 weight registers | 32.4 | 468.4 |
| 2 k-quads hoisted, weights in an `int8x16_t[2]` | 33.6 | 498.2 |
| 2 k-quads hoisted, weights in named scalars | 33.3 | 478.9 |

The cause is visible in the generated code rather than guessed: `objdump` of the accepted kernel shows
the compiler already emits `ldp q3, q2, [x3], #32` (both weight halves in one post-indexed 32-byte
load), keeps all 8 int32 accumulators and both float pairs in registers, and folds the activation
broadcast into the load where it can. The loop is `sdot`-throughput bound, not issue bound, so removing
uops from it cannot pay - and every hoisting variant spends the registers the accumulator tile needs.
Note also that the vector arrays spill: version 3 was 4 % slower than version 4 purely because
version 3 declared the weights as `int8x16_t wl[2]` instead of two named scalars.

#### Re-measured: MR is still 4 now that the document is GEMM-bound

The Attempt 20 MR sweep was taken when the document was 75 % attention, so it could not resolve the
GEMM's own optimum. Re-run against the GEMM-bound workload (`MR=8` halves the weight traffic, which
now matters): MR=4 **452.5 ms**, MR=6 452.0 ms, MR=8 461.0 ms on the document, queries 30.6 / 32.2 /
30.3. MR=4 confirmed; the register-pressure explanation holds.

#### Where the remaining time is

| region | 15-token query | 196-token document |
| :-- | --: | --: |
| gemm (int8) | 28.1 ms (83 %) | 292.0 ms (66 %) |
| attention (now NEON fp32) | 1.4 ms (4 %) | 115.8 ms (26 %) |
| quantize | 3.1 ms (9 %) | 28.3 ms (6 %) |
| rest | 1.3 ms | 11.6 ms |

Both towers of the remaining time are close to hardware limits on this part: the int8 GEMM runs the
whole document's ~25 GMAC in 292 ms wall with the pool at 1.94x on 2 threads, and the attention's
~2.3 GMAC of fp32 in 116 ms is ~40 GFLOPS against a ~48 GFLOPS fp32 peak. The next real step is the
int8 packed-operand attention (`e2_i8_attn_gemm` + `e2_softmax_quant_i8`), which would move 26 % of
the document from fp32 to int8; it is not attempted here.

**14/14 CTests pass on aarch64, 21/21 and 42/42 on x86-64** (no regression; the x86 files are untouched).

### Attempt 22: int8 attention in the text tower (ACCEPTED on aarch64 only)

The text tower's attention was fp32 on **every** architecture - `grep -c e2_i8_attn_gemm src/text.c`
returned 0, and the comment in `vision.c` claiming the text tower shares that kernel was stale. So
the int8 attention machinery (`e2_i8_attn_gemm` / `e2_softmax_quant_i8`) served vision only.

Rather than port those packed-operand kernels, this reuses what already exists: **both attention
products are the NT product `int8.c` implements**. Logits is `Q[rows,D] x K[keys,D]^T`, and context
is `P[rows,keys] x V^T[D,keys]^T`, so both go through `e2_i8_gemm` with K packed into the ordinary
`e2_q8w` layout. No attention-specific kernel on either ISA, and no change to the packed-cache
format. Per layer, per head: `e2_q8w_from_f32_ld` packs K straight out of the strided projection
output (new; `ld` is its leading dimension), `kt_worker` transposes V row-wise into V^T (it serves V
unchanged) and `e2_q8w_from_f32` packs that. Per query block: quantize Q rows, GEMM, softmax+mask,
quantize the probabilities, GEMM. Both packed operands are then freed.

Accuracy is not the cost here. Same machine, same weights, one alternating variable
(`E2_VATTN_OFF` on x86 vision, which does use the int8 attention kernel):

| workload | int8 attention | fp32 attention | delta |
| :-- | --: | --: | --: |
| image, 2520 patches | 0.999799 | 0.999814 | -0.000015 |
| image + caption | 0.999721 | 0.999698 | **+0.000023** |

+/-2e-5, not even consistently negative. End to end on the text document the cosine moves
0.999624 (fp32) -> 0.999630 (int8) on x86 and 0.999628 -> 0.999606 on aarch64: noise either way, all
far above the 0.999 bar. `e2_test_text_attention` cross-checks the int8 path against the row-wise
fp32 oracle at seven lengths (31..2048, both head sizes) and bounds it at 2e-2, against 2e-5 for the
fp32 path.

#### Where it wins depends entirely on the fp32 GEMM underneath it

`E2_THREADS=2`, batch 1, load excluded, `e2_text_bench`. Ampere A1 / Neoverse-N1:

| keys | fp32, ms | int8, ms | |
| --: | --: | --: | :-- |
| 48 | 85.589 | 129.596 | -51% |
| 64 | 117.345 | 119.410 | -2% |
| 96 | 177.558 | 175.149 | +1% |
| 128 | 248.352 | 235.212 | **+5%** |
| 196 | 438.902 | 380.605 | **+13%** |
| 512 | 1677.673 | 1090.149 | **+35%** |
| 1024 | 4710.493 | 2724.714 | **+42%** |
| 2048 | 12398.677 | 6735.602 | **+46%** |

Core Ultra 9 285H, same method: `E2_THREADS=6`, `taskset -c 0-5`, best of 9, single runs per length.

| keys | fp32, ms | int8, ms | |
| --: | --: | --: | :-- |
| 32 | 8.851 | 10.952 | -24% |
| 48 | 12.769 | 14.172 | -11% |
| 64 | 15.335 | 16.493 | -7% |
| 96 | 21.585 | 22.632 | -5% |
| 128 | 30.893 | 29.800 | **+4%** |
| 196 | 45.976 | 40.652 | **+12%** |
| 256 | 61.350 | 53.009 | **+14%** |
| 384 | 98.766 | 82.874 | **+16%** |
| 512 | 160.662 | 115.233 | **+28%** |
| 768 | 293.048 | 233.182 | **+20%** |
| 1024 | 447.964 | 335.884 | **+25%** |
| 2048 | 1089.161 | 739.517 | **+34%** |

#### The packer was the crossover, and it was serial

The tables above are the *final* state. Before the pack was parallelised both architectures crossed
over far higher (aarch64 64-96 keys, x86-64 ~384), and the reason was measurable in isolation with a
one-thread microbenchmark: `e2_q8w_from_f32` on a 196x256 matrix runs at **1048 M elem/s** and
`e2_i8_gemm` at 196x196x256 at **213 GMAC/s**. Per layer at 196 keys that is ~462k elements of packing
(~0.44 ms) against 78.7 MMAC of int8 GEMM (~0.06 ms across 6 threads) - **~90 % packer, ~10 % GEMM** -
and the pack was a serial `for (h = 0; h < H; h++)` loop while the GEMMs ran on the pool. It is now one
more claim-work region (`pack_worker`), which moved the x86-64 crossover from ~384 keys to ~128 and
flipped 196 keys from -10 % to +12 %.

Both architectures now cross over at the same place, which is why `E2_ATT_I8_MIN` is a single constant
(128) rather than a per-ISA one.

**Correction, because it was got wrong first.** The first x86 measurement of this was unpinned, on a
16-core laptop in `powersave` with a browser running, and read the two paths as inseparable from noise
at both 196 and 2048 keys (fp32 793/829/877 against int8 713/957/932), which led to the path being
shipped disabled on x86-64. Pinning the pool (`taskset`, fixed `E2_THREADS`) and re-running makes the
effect consistent across three alternating rounds at every length, and int8 wins by 13-34 % from 512
keys up. The lesson is the one already in the guidelines - same session, pinned, best of N,
alternating - and the earlier "not separable" verdict was a property of the measurement, not of the
kernel.

Net effect on the reference workloads: the 196-token document goes **441.7 -> 396.9 ms** on the Ampere
A1 (12.8x -> **14.2x** versus the bf16 baseline) and the 15-token query is unchanged at 30.4 ms.

#### Two latent bugs this shook out

- **`e2_i8_gemm` silently required `K % E2_I8_BLOCK == 0`.** Every SIMD backend walks k-quads to the
  end of each 128-element block without consulting K, so a short K reads into the next 8-row group -
  the scalar reference does not, which is what let it hide. The context product is the first caller
  with an arbitrary K (`K = keys`), and it was wrong by 15.5 absolute before this was found. Fixed by
  padding in the caller (zero weight rows, zero activation rows, zero scale blocks) rather than by
  clamping in the kernels: clamping the loop bound turns a constant trip count into a dynamic one and
  measured **5% slower on the whole int8 GEMM** (query 30.6 -> 32.2 ms), so the constant bound stayed
  and the precondition is now documented in `int8.h`.
- **Per-thread scratch slices were offset by `BQ*S` while the rows inside were `Sp`-strided**, so
  workers overwrote each other's probabilities. Caught by the S=31 case, where `Sp/S` is 4.

V^T and the fp32 path's K^T are never alive at the same time, so they share one workspace slot; the
rest of the int8 scratch costs +9.7 MB (230.2 -> 239.9 MB). An earlier version that gave V^T its own
16.8 MB slot regressed the 15-token query by 4% on arena size alone.

**14/14 CTests on aarch64, 21/21 on x86-64.** `e2_test_text_attention` grew the seven int8 cases.

### Attempt 23: NEON packed-operand attention GEMM, which is what vision needed (ACCEPTED)

Attempts 20-22 left the largest single block of time in the ARM run untouched: **8.5 s of a 13.9 s
image was fp32 vision attention**, because `vision.c` gates its int8 path on
`vattn_have_simd() == (e2_i8_attn_gemm != NULL)` and that kernel had no NEON form. New
`e2_i8_attn_gemm_neon` in `src/kernels_i8_neon.c` implements the contract from `int8.h`;
`e2_cpu_init()` now sets it on aarch64.

Only that one kernel was needed. `e2_softmax_quant_i8` stays NULL on ARM deliberately: the callers
fall back to `K->softmax` + `e2_i8_quantize` (`vision.c` says so explicitly), and both are already
vectorized, so fusing them is an optimization rather than a prerequisite.

The kernel reuses the two tricks from the weight GEMM in the same file: the activation is un-biased
on the fly with `xor 0x80` so `vdotq_s32` accumulates the true `sum a_i8*b_i8` and **`bz` is never
read**, and the B operand's 8 output columns are 32 contiguous bytes = two 128-bit halves. Tile is
4 query rows x 8 columns (8 int32 + 8 float accumulators + 2 weight vectors = 19 of 32 registers),
with the row loop split into a fixed 4-row body and a 1-row tail so `i < 4` stays a constant bound -
the same structure, for the same reason, as the weight GEMM. A 1..3 k remainder quad (only possible
when K is not a multiple of 4) is done scalar; `K` is arbitrary here (the context product uses
`K = patches`), so unlike the weight GEMM this contract does not carry the block-multiple
precondition.

Ampere A1 / Neoverse-N1, 2 vCPU, `E2_THREADS=2`, best of 3, three runs:

| vision workload | fp32 attention | int8 attention | |
| :-- | --: | --: | :-- |
| image, 2520 patches | 13882.520 ms | **8487.9 ms** | **1.64x** |
| image + caption (274 ids) | 14082.420 ms | 8694.6 ms | 1.62x |

Fidelity: image cosine **0.999814 -> 0.999769**, caption **0.999677 -> 0.999690**, both `PASS` and
both within 5e-5 of the x86 numbers (0.999768 / 0.999661), so this is the same arithmetic x86 has
been running. Text is unaffected in both directions (`query 30.4 ms / document 380.5 ms`, unchanged).

`E2_PROFILE=1` on the image, before and after:

| region | before | after |
| :-- | --: | --: |
| attention | 8505.1 ms (64 %) | 3098.3 ms (38 %) |
| mlp | 2216.8 ms (17 %) | 2350.8 ms (29 %) |
| elem | 1258.6 ms (9 %) | 1308.8 ms (16 %) |
| qkv+o | 1142.3 ms (9 %) | 1158.8 ms (14 %) |
| attn_pack | 0.0 ms | 170.9 ms |

**Not attempted: the `elem` region, on purpose.** It looked like the obvious next item (1309 ms, 16 %,
and every one of those ops is still the scalar `e2_k()` vtable entry on aarch64), so the generated code
was checked before writing a NEON backend for it:
`objdump` of `scalar_mul` in the arm build shows `ldr q31 / fmul v30.4s / str q30` - GCC already
auto-vectorizes those loops at `-O3 -ffast-math`. They are memory-bound, and a hand-written NEON
vtable entry would add a file and move nothing. The same check is worth doing before anyone else
proposes it.

With attention fixed, ~81 % of the image is now the int8 GEMM (attention 3.1 s + mlp 2.4 s + qkv+o
1.2 s), derived from the profile and the known shapes. The attention GEMM runs at ~42 GMAC/s against
**75-111 GMAC/s for the weight GEMMs in the same run** (qkv+o 87 GMAC / 1159 ms, mlp 261 GMAC /
2351 ms). The gap is not the instruction mix - both kernels issue ~5-6 MAC per uop - it is that vision
attention materialises a `patches x patches` score matrix per head and layer (2304^2 x 12 heads x 16
layers, ~1 G fp32 values per image), writing and re-reading it around the softmax. The next win there
is key-blocking vision attention so the scores stay in cache, as the text tower already does; it is a
restructuring of `vision.c`, not a kernel.

**14/14 CTests on aarch64** (the `int8` test's attention-tail cases K=1/3/9/129 now run on ARM
instead of being skipped), **21/21 on x86-64**.

### Attempt 24: NEON fused softmax + probability quantizer (ACCEPTED, but the win was 2 %, not the predicted 25 %)

Attempt 23 left `e2_softmax_quant_i8` NULL on aarch64, so vision ran `K->softmax` per row and then
`e2_i8_quantize` over the same rows. New `e2_softmax_quant_i8_neon` fuses them, and replaces libm's
vector `expf` (`_ZGVnN4v_expf` in the scalar softmax - confirmed in the arm object file) with the same
range-reduction + degree-6 polynomial `exp_ps` the AVX2 kernel uses, `vrndnq_f32`/`vfmsq_f32` for the
reduction.

The reasoning for doing it was the traffic: done separately, a score row is written by the logits
GEMM, read and rewritten by softmax, then read again by the quantizer. For a 2304-patch image that
matrix is `2304^2 x 12 heads x 16 layers` ~ 1 G values, so removing one full read+write pass should
have been worth ~8 GB and, at the observed bandwidth, a good fraction of the attention region.

**It was worth 2 %.** Ampere A1, `E2_THREADS=2`, best of 3, three runs:

| | before | after |
| :-- | --: | --: |
| image, 2520 patches | 8488-8583 ms | **8324-8387 ms** |
| attention region | 3098.3 ms | 3007.7 ms |
| image + caption (274 ids) | 8694.6 ms | **8521.8 ms** |

So the score-matrix traffic is not the dominant term I assumed, and the earlier estimate that the
attention GEMM runs at ~42 GMAC/s while the weight GEMMs run at 75-111 GMAC/s still stands after this
change: 130.6 GMAC of vision attention in 3008 ms is 43.4 GMAC/s, against qkv+o at 86.9 GMAC / 1135 ms
= 76.6 and mlp at 260.9 GMAC / 2167 ms = 120.4. Whatever is costing the attention region, it is not
the softmax pass.

Accuracy is unchanged within noise: image cosine **0.999769 -> 0.999782**, caption 0.999690 ->
0.999633, against x86's 0.999768 / 0.999661. The fused probabilities agree with
softmax+quantize run separately to **7.6e-10** and produce **identical int8 bytes** at every length
tested, on both backends.

#### This kernel had no test at all, on any backend

`grep e2_softmax_quant_i8 tests/` returned only the dispatch override in `test_cpu_backends.c`, so the
AVX2 and AVX-512 versions have been untested since they were written. `test_int8` now runs
`check_softmax_quant` over five lengths (32, 128, 200, 1300, 2520 - one block, exactly a block,
several, and a patch-sized row) against the reference path the callers fall back to. It checks the
implied probabilities (the fused form leaves the *unnormalised* exponentials in `x` and folds `1/sum`
into the block scales, which is not the same thing as "softmaxed in place"), the block scales
relatively, and the quantized bytes. Both x86 backends and the NEON one pass with 0 byte difference.

**14/14 CTests on aarch64, 21/21 on x86-64.**

### 2026-10-29 / AMD Ryzen 9 7940HS (Zen 4), 8 threads, int8, **native Windows 11**

Windows 11 Pro 26200, clang 21 (LLVM) with the MSVC linker and SDK 10.0.26100 from the Visual Studio
18 Insiders install, CMake 3.31.6, Ninja 1.12.1, 27.8 GB. The pool reports 8 threads on 8C/16T because
`sysconf(_SC_NPROCESSORS_ONLN)` is answered from `GetLogicalProcessorInformationEx(RelationProcessorCore)`
- the same "drop the SMT siblings" policy the Linux path gets from `e2_nthreads()`.

```
set E2_Q8_READ=C:\...\e2-embeddinggemma-2-int8-text.e8cq
set E2_REPS=7
build\e2_test.exe models\e2 tests\fixtures
```

| workload | best ms | notes |
| :-- | --: | :-- |
| 15-token query | **4.49** | best of 7, cos 0.999783 |
| 196-token document | **30.41** | best of 7, cos 0.999596 |
| image, 2520 patches | **609.21** | cos 0.999763 |
| image + caption (274 ids) | 870.24 | cos 0.999690 |

Cross-machine, same packed int8 weights, best of N, load excluded:

| workload | Ampere A1 (NEON) | Intel N100 (AVX2+VNNI) | Core Ultra 9 285H (AVX2+VNNI) | Ryzen 9 7940HS (Win, AVX-512) | Ryzen 7 9700X (AVX-512) |
| :-- | --: | --: | --: | --: | --: |
| 15-token query | 30.4 ms | 22.7 ms | 6.19 ms | 4.49 ms | **3.95 ms** |
| 196-token document | 380.5 ms | 329.4 ms | 41.41 ms | 30.41 ms | **24.92 ms** |
| image, 2520 patches | 8496 ms | n/m | 708 ms | 609 ms | **543 ms** |

Windows lands between the Arrow Lake laptop and the Zen 5 desktop, which is where a Zen 4 mobile part
with a 256-bit AVX-512 datapath belongs. It is ~13 % off the Zen 5 column on text; part of that is the
clock and the narrower vector units, and part may be the park: `threads.c` sleeps on an SRWLock +
`CONDITION_VARIABLE` pair here rather than a futex, and the pool parks ~170 times per embedding. Not
separated, so not claimed.

**12/12 CTests pass** - 12 rather than 14 because `artifacts` is a POSIX shell script (already
`if(UNIX)`) and the two GPU tests need Level Zero/HIP, which are not built on Windows.

### The Windows port itself

The engine was POSIX by construction (`mmap` for the safetensors view and the packed cache, `pthreads`
+ `futex` for the worker pool) and `clang -c src/st.c` failed at `#include <sys/mman.h>`. Rather than
sprinkle `#ifdef _WIN32` through the engine, the port is one directory:

| file | what it is |
| :-- | :-- |
| `src/win32/inc/{unistd.h,sys/mman.h,pthread.h,sched.h}` | the four headers Windows lacks, put on the include path |
| `src/win32/inc/e2win.h` | force-included: `clock_gettime`, `setenv`/`unsetenv`, `mkdtemp`, `truncate`, `sysconf`, `usleep`, `strtok_r`, `rmdir` |
| `src/win32/compat.c` | the implementations: `CreateFileMapping`+`MapViewOfFile`, `_beginthreadex`, `QueryPerformanceCounter`, `_putenv_s`, `_open(_O_CREAT\|_O_EXCL)` for the two mk*temp, `GetLogicalProcessorInformationEx` for the core count |

No engine source has a `_WIN32` branch except three: `threads.c` (the park), and `st.c`/`q8cache.c`
where the file size is read, because UCRT's `struct stat` has a **32-bit `st_size`** and the checkpoint
is 1.4 GB - those two take `_fstat64`. Everything else compiles untouched.

Three toolchain things worth writing down, because each cost a build:

- CMake links with **`-nostdlib`**, so `clang_rt.builtins` is not added - and that library is where
  `__cpu_model` / `__cpu_features2` / `__cpu_indicator_init` live, which is what
  `__builtin_cpu_supports()` lowers to on x86, i.e. the entire runtime backend dispatch. `clang
  --print-libgcc-file-name` answers a bare `libgcc.a` on Windows, so the build globs
  `<llvm>/lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib` from the compiler's own location and
  links it as an INTERFACE dependency of `e2`.
- `WaitOnAddress`/`WakeByAddressAll` are not in every Windows SDK's import library, so the park is an
  `SRWLOCK` + `CONDITION_VARIABLE` pair instead. It has the same "sleep while the word still has the
  expected value" contract, and the callers re-test the generation in a loop, so a spurious wake costs
  nothing.
- `-ffast-math` makes clang warn (`-Wnan-infinity-disabled`) about the `INFINITY` sentinels `int8.h`
  uses as "no clamp" bounds. That is a clang-only warning, so Windows builds pass
  `-Wno-nan-infinity-disabled`; GCC on Linux is unchanged.

Five tests spelled `/tmp/...` inline and now go through `tests/test_tmpdir.h`, which picks `%TEMP%` on
Windows and `/tmp` elsewhere. Linux still passes **21/21** and aarch64 **14/14** with those changes, so
the port costs the other platforms nothing.

### Which AVX width does this engine actually get? (measured on four machines)

Single-thread, pinned, the same 64x2048x512 int8 GEMM, timing the two int8 kernels the engine carries
directly. This exists because the Windows column above sits 13 % behind the Zen 5 column and "AVX-512"
is not one thing across these CPUs.

| CPU | micro-arch | int8 kernel | GMAC/s | notes |
| :-- | :-- | :-- | --: | :-- |
| Ryzen 7 9700X | Zen 5, native 512-bit | `e2_i8_gemm_avx512` (zmm vpdpbusd) | **433.2** | |
| Ryzen 7 9700X | Zen 5 | `e2_i8_gemm_vnni` (ymm vpdpbusd) | 190.2 | same code, 256-bit operands |
| Ryzen 9 7940HS | Zen 4, double-pumped | `e2_i8_gemm_avx512` | 298.3 | 256-bit datapath |
| Core Ultra 9 285H | Arrow Lake, no AVX-512 | `e2_i8_gemm_vnni` | 225.8 | |

Two facts worth keeping:

- On Zen 5 the **same kernel at 512-bit is 2.28x the 256-bit form** (433 vs 190). That is the width
  difference, isolated: identical source, identical arithmetic, only the operand width and the
  32-register AVX-512 file differ.
- Zen 4 reaches 298 GMAC/s, which is ~93 % of what a 2x256-bit-per-cycle design can do at its
  single-core boost - i.e. on Zen 4 a 512-bit instruction costs two uops but keeps the same MAC/cycle
  as two 256-bit ones, so AVX-512 buys instruction density there and nothing else.

So the 13 % end-to-end gap between the Windows Zen 4 box and the Zen 5 box is *not* the 1.45x kernel
gap: the document GEMM runs the pool at 6.6x on 8 threads and is bound by weight bandwidth, not by
per-core int8 throughput. The kernel gap shows up in the single-thread microbenchmark, not in the
embedding.

#### A hazard this found: never call a backend kernel directly

Probing Zen 4 by calling `e2_i8_gemm_vnni` directly **died with STATUS_ILLEGAL_INSTRUCTION
(0xC000001D)**. Zen 4 has AVX512-VNNI but **not AVX-VNNI**, the 256-bit VEX form - that one is Intel's
(Alder Lake and later), and it appears on Zen 5 too but not Zen 4. The engine is correct because
`cpu_has_avxvnni()` asks `__builtin_cpu_supports("avxvnni")` before dispatching anywhere near that
object file, and `kernels_i8_vnni.c` is only ever entered through it. Tests that call a backend
directly (`test_int8`, `test_cpu_backends`) all guard with the matching `cpu_has_*` probe first, and
that is why they pass everywhere. Keep it that way - a direct call is an illegal instruction on a CPU
that does not advertise the feature.

Neither Zen 4 nor Zen 5 has `avx512_fp16` (Intel-only so far, absent from both /proc/cpuinfo dumps), so
the fp16-attention question stays untestable on every machine in this set.

### 2026-10-29 / DISPATCH.md phases 0-5: selection became data, arithmetic untouched

Not an optimization attempt, so there is no speed claim here; it is recorded because the project's
rule is that a refactor touching the attention path must show the other backends did not move.
`src/dispatch.c` now holds the capability bitmask and the per-(backend, tower) rules, `e2_attn_pick()`
is the only selector, and `tests/test_dispatch.c` asserts the whole matrix as data on any host (the
table is inert, so no kernel is ever probed by calling it).

Verified by rebuilding the same tree on four hosts and comparing the printed fidelity before and after:

| host | backend | ctest | dispatch test | fidelity (before -> after) |
| :-- | :-- | --: | :-- | :-- |
| laptop (local) | AVX-VNNI + Level Zero | 43/43 | OK | unchanged |
| nucleus (Zen 5) | AVX512-VNNI | 15/15 | OK | query 0.999751, doc 0.999594, image 0.999795 - all identical |
| n100 (Gracemont) | AVX-VNNI | 14/15 | OK | query 0.999772, doc 0.999580 - identical |
| arm-server (N1) | NEON dotprod | 15/15 | OK | query 0.999776, doc 0.999606, image 0.999782, audio 0.999631 - all identical |
| minipc (Zen 4, Windows) | AVX512-VNNI | 13/13 | OK | new build, cos 0.999783 / 0.999596, PASS |

The n100's one failure is the known `artifacts` test, which needs `git` and that box has none; it is a
pre-existing failure and unrelated. Observed timings moved in both directions within each machine's
run-to-run spread (nucleus query 4.73 -> 3.90 ms, n100 document 317 -> 324 ms, arm document 387 -> 383
ms), which is what a selection-only change should look like: no fidelity number moved, no consistent
timing move.

Phase 3 (the `E2_ATTN_TEXT`/`E2_ATTN_IMAGE` names, old names kept as aliases) was then measured the way
the gate asks: same-model binary pair, alternating, pinned, best-of-N, laptop, `E2_THREADS=6` under
`taskset -c 0-5`. A = phases 0-2, B = phases 0-3:

| workload | A | B | delta |
| :-- | --: | --: | --: |
| query 15 tok (best of 15, 4 rounds) | 5.11 ms | 5.10 ms | -0.2 % |
| document 196 tok (best of 15, 4 rounds) | 38.90 ms | 39.05 ms | +0.4 % |
| image 2520 patches (best of 5, 4 rounds) | 907.66 ms | 907.53 ms | -0.01 % |

Phase 3 re-ran the multi-host fidelity checks with the same binaries rebuilt on each box: nucleus
(0.999751 / 0.999594 / 0.999795), arm-server (0.999776 / 0.999606), n100 (0.999772 / 0.999580) and
minipc (0.999783 / 0.999596) all matched their phase-0-2 values digit for digit.

**Method note, because it nearly produced a false verdict.** The first pass always ran A and then B in
each pair, and B looked 1-2.7 % slower on *every* pair - a plausible regression. It was ordering: the
second binary inherits a warmer machine. Re-running with B first in alternate rounds erased the gap
(image medians changed from A 917 / B 922 to A 911 / B 908). One level down from the Attempt 22
correction: alternating the *binaries* is not enough if their order is fixed.

Phases 4-5 add nothing to a per-embed path. The registry walks five rows once inside `e2_cpu_init()`
(it replaced two else-if ladders, so it is fewer branches at startup, not more), and the `attn path`
line is printed only inside the `E2_PROFILE` blocks, whose `g_prof_on` / `g_vprof_on` flags are false
otherwise. The full ctest matrix and the fidelity numbers were repeated on all four hosts after the
rebuild and did not move.

A final A/B against the phases-0-2 binary, best of 25, eight alternating rounds:
query **5.35 vs 5.35 ms** and document **40.20 vs 40.40 ms** (+0.5 %). The machine had a browser
running this session and was much noisier than during the phase-3 pass, so the *minimum* is the number
to read - and an identical-binary control (the same build copied twice, same protocol) moved the
document median by up to 4 % while its minimum moved by 0.2 %. That control is the reason not to quote
the means here.

Phase 4 also fixed `e2_cpu_name()` on aarch64, which reported `"scalar"` while the NEON int8 kernels
ran; it prints `cpu backend : neon` there now, which is what HARDWARE.md's machine-identification step
expects to see.

**The one backend that had never been built now is.** nucleus received the full ROCm 7.2 stack (clang
22, `libamdhip64`, kernels for `gfx1036`), so `-DE2_WITH_HIP=ON -DE2_HIP_ARCH=gfx1036` builds `e2_hip`
and both HIP tests pass:

| workload | GPU (`gfx1036`, 2 CU) | CPU (8 threads) | cos(gpu, ref) |
| :-- | --: | --: | --: |
| text query 15 tok | 12.3 ms | 6.1 ms | 0.999780 |
| text document 196 tok | 104.3 ms | 32.7 ms | 0.999626 |
| image 2520 patches | 1990 ms | 673 ms | 0.999778 |
| image+caption 274 ids | 1999 ms | 588 ms | 0.999661 |

This is a correctness result, not a performance claim: a 2-CU iGPU loses to an 8-core Zen 5 on every
shape, and there is nothing here worth optimizing. The attention switches were checked against each
other on the same device rather than by assertion — `E2_ATTN_TEXT=f32` and `E2_HIP_ATTN=lds` produce the
same document (180.6 vs 180.8 ms, identical cos), `E2_ATTN_STAGE=legacy` and `E2_HIP_ATTN=old` likewise
(185.8 vs 191.6 ms, identical cos), and `E2_ATTN_TEXT=i8` moves the 15-key query onto int8 (cos 0.999765
vs 0.999780, 604 vs 460 kernels) because the override drops the 64-key gate. With that, HARDWARE.md's
"never validated on AMD hardware" becomes "built and smoke-tested on `gfx1036`, not
performance-validated".

**The registry also bought a measurement tool, and it is the first thing that makes a kernel question
cheap.** `E2_CPU_BACKEND=<name>` downgrades the selected backend (never invents an ISA), and
`e2_text_bench ... backends` sweeps every backend the host can execute in one process, reloading the
model per backend because `use_int8` is decided at load. First use, laptop, 6 pinned threads, packed
cache, 196-token document, best of 5: **vnni 40.05 ms, avx2 117.48 ms** - the `vpmaddubsw` path is
2.9x the `vpdpbusd` one *on the whole engine*, not just in a GEMM microbenchmark, which makes the
AVX2-only row of the dispatch matrix a measured fact rather than a theoretical one. Against the same
checkpoint on a 15-token query the three backends are 7.58 / 11.62 / 124.80 ms (vnni / avx2 /
scalar-bf16, best of 2). The scalar row is bf16 and cannot read a packed int8 cache, so the sweep
skips it with a note when `E2_Q8_READ` is set.

### 2026-10-29 / CPU kernel microbench, and a vision change rejected before it was written

`tools/bench/kernel_bench.c` is the CPU counterpart of `hip_kernel_bench.hip`: per-kernel GMAC/s (or
Melem/s) under `E2_CPU_BACKEND`, so a kernel question no longer has to be inferred from a 40 ms total.
One core, best of 5:

| kernel | laptop vnni | laptop avx2 | nucleus avx512 | nucleus vnni | nucleus avx2 | ARM neon |
| :-- | --: | --: | --: | --: | --: | --: |
| `i8_gemm` 64x512x2048 | **229.4** | 57.2 | **234.5** | 191.4 | 59.5 | 51.2 |
| `attn_gemm` logits 64x2304x64 | 65.8 | - (no kernel) | 123.0 | 52.3 | - | 36.1 |
| `attn_gemm` context 64x64x2304 | 74.7 | - | 151.2 | 65.7 | - | 43.1 |
| `q8w_from_f32` (Melem/s) | 1074 | 1054 | 880 | 882 | 801 | 447 |
| `quantize_blocks` (Melem/s) | 4872 | 4853 | - | - | - | 1664 |
| `sgemm_nn_range` (fp32) | 44.8 | 44.1 | 78.5 | 63.5 | 61.2 | 12.2 |

Two things it found immediately. **Unpinned numbers are meaningless on a hybrid CPU**: the same vnni
GEMM reads 63 GMAC/s when the scheduler puts it on an E-core and 229.4 pinned to a P-core, which is
the whole gap between this table and HARDWARE.md. And the forcing switch had a real bug it exposed:
`E2_CPU_BACKEND=scalar` reset the GEMM and quantizer pointers but not the fused attention ones, so a
forced scalar backend still carried the VNNI attention kernel. Fixed in `apply_backend()` and pinned
by a new assertion in `test_dispatch`.

**The key-blocking idea is rejected, by measurement rather than by opinion.** The premise in the
bullet below was that vision attention re-reads a spilled `patches x patches` score matrix. It does
not spill: a work item is `E2_V_ATT_BQ`(32) x P(2304) x 4 B = 295 KB, which is L2-resident, and moving
the block size in either direction makes the machine with the biggest cache worse and leaves the one
without an L3 flat:

| `E2_V_ATT_BQ` | laptop, attention (6 threads) | ARM, attention (2 threads) |
| --: | --: | --: |
| 16 | 484 ms | - |
| 32 | 440-478 ms | 3127 ms |
| 64 | 513 ms | 3135 ms |

The arithmetic agrees. Two products per image are 2 x 2304^2 x 64 x 12 x 16 = 130 GMAC; at the
measured 65.8 / 74.7 GMAC/s and 6 threads that is 0.31 s of the 0.44-0.48 s attention region, so the
GEMMs are two thirds and the fused softmax+quantize (about 1 G elements per image) most of the rest.
Key-blocking addresses neither. What the table does show is the actual gap: the attention GEMM runs at
65.8 GMAC/s where the weight GEMM runs at 229.4, because the logits product is 16 k-quads deep with a
4-row tile - the accumulator setup is not amortised, exactly the second bullet below.

## Next Targets

- **15-token query**: the tightest case. Pool claim + barrier (~2.5 us per region at 14 threads) is a
  third of a ~4.7 ms run. Fewer regions, or fusing the small ones, is the only lever left.
- **Intel GPU remaining gap**: document batches now approximately match OpenVINO's median, while
  single documents and query batches still trail. The remaining work is projection microkernel
  throughput and FP32 normalization/activation traffic. Avoid more cache/launch knobs: measured
  producer/consumer fusions and better arithmetic scheduling must pay for any further complexity.
  Short-query wins still need repeated same-session confirmation.
- **Audio tower**: chunked local attention width is a fixed 12; nothing measured above or below it.
- **vision attention, measured per kernel**: the region is 50 % of an image (440-478 ms of ~900 at 6
  threads), and the kernel bench shows why. `attn_gemm` runs at 65.8 GMAC/s (logits) and 74.7
  (context) where the weight GEMM runs at 229.4, and the two products are 0.31 s of that region. The
  limiter is the logits product's shape - 16 k-quads deep with a 4-query-row accumulator tile - not
  cache traffic: a 295 KB score block is L2-resident and the block-size sweep above is flat or worse.
  The next win is a wider tile for K=64 (8 or 16 query rows per accumulator), or fusing the softmax
  into the logits kernel so the score row is never written as fp32. Target: >=120 GMAC/s on x86 and
  >=70 on ARM, measured with `e2_kernel_bench`, before touching `vision.c`.
- **text int8 attention below 128 keys**: the pack (still scalar `lrintf` at 1048 M elem/s) and the
  K/V^T transpose are now the whole cost there, and both are bound by element count rather than by
  FLOPs. A SIMD `e2_q8w_from_f32` is the only lever that moves this crossover again.
- **Load time**: ~0.01 s from the packed cache, 1.4 GB checkpoint read otherwise. Only matters for
  cold-start latency, not for throughput.

## Credits attribution rules

- Ideas, kernels and approaches are only taken from BSD / MIT licensed code.
- If an optimization idea or kernel comes from another project, credits go into the source file that
  uses it.
