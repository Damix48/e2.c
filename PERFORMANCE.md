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

No attempts recorded yet.

## Next Targets

- **15-token query**: the tightest case. Pool claim + barrier (~2.5 us per region at 14 threads) is a
  third of a ~4.7 ms run. Fewer regions, or fusing the small ones, is the only lever left.
- **Intel GPU text attention (FP32)**: the open item; INT8 attention is not installed because it does
  not clear the accuracy gate. Kernel throughput was never the bottleneck.
- **Audio tower**: chunked local attention width is a fixed 12; nothing measured above or below it.
- **Load time**: ~0.01 s from the packed cache, 1.4 GB checkpoint read otherwise. Only matters for
  cold-start latency, not for throughput.

## Credits attribution rules

- Ideas, kernels and approaches are only taken from BSD / MIT licensed code.
- If an optimization idea or kernel comes from another project, credits go into the source file that
  uses it.
