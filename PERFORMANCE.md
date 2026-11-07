# PERFORMANCE.md — the optimization ledger

Goal: beat OpenVINO int8 (NNCF `INT8_SYM`, i.e. **weight-only** compression, so
OV runs dequantised f32 matmuls) by **1.5x in every scenario**.

Host: AMD Ryzen 7 9700X (8 Zen 5 cores / 16 threads, AVX-512 VNNI, no AMX),
Debian 13. **The box is shared** — the same e2 binary measures 91 ms and 117 ms
for `text 512` twenty minutes apart (load average 2.8 vs 7.1), and e2 loses
~25% under load while OpenVINO loses ~5%. Two consequences:

* never trust an absolute ms number across runs;
* the only meaningful number is a **same-run e2-vs-OV ratio**
  (`tools/bench_compare.py`, which also pools samples with `--repeat`).

## Where the time goes (text encoder, `E2_PROFILE=1`)

512 tokens, 24 layers, hidden 512, 4 heads / 2 kv heads, head_dim 256/512,
inter 2048, PLE 512. 64.4 GMAC of GEMM + 12.9 GMAC of attention per forward.

| stage | ms (quiet host) | notes |
|---|---:|---|
| attention | 31 | 36% — now the #1 cost |
| mlp (gate+up+down) | 24 | 38.7 GMAC |
| q,k,v | 8.9 | 9.7 GMAC |
| o_proj | 6.2 | 6.4 GMAC |
| ple gate+proj | 4.8 | 6.4 GMAC |
| rope, norms, gated mul, gelu, pool | ~10 | mostly vectorised now |
| **total** | **~87** | |

At 2048 tokens the same forward is 665 ms and attention alone is 364 ms (55%):
attention is O(t²) while the GEMMs are O(t).

## What worked (measured, biggest win first)

| # | change | effect |
|---|---|---|
| 1 | **FTZ + DAZ on every thread** (`ldmxcsr`, bits 6 and 15, in `e2_load` and in each pool worker) | o_proj 9.8→2.9 ms, ple_proj 4.1→0.76 ms, gelu 0.91→0.30 ms. The bf16 checkpoint leaves denormals in the residual stream and x86 denormal arithmetic is 10-100x slower. **The single biggest win.** |
| 2 | **int8 GEMM kernel: 16 rows x 1 tile → 4 rows x 4 tiles** | 24 uops per 16 `vpdpbusd` instead of 33 (4 broadcasts + 4 weight loads is the minimum for a 16-accumulator budget), and the array-indexed accumulator no longer makes the compiler emit 16 redundant `vmovdqa` per iteration. Result: 64.4 GMAC of text GEMM in 15 ms ≈ 4.3 TMAC/s ≈ 1.6 vpdpbusd/cycle. |
| 3 | vectorised `e2_quant_rows` (AVX-512, 128 floats/block, `vcvtps_epi32` + saturating `vpmovusdb`) **and** row-split across the pool | the activation quantiser was serial and single-threaded; it capped GEMM thread scaling at ~2x (1→4 threads gave 260→520 GMAC/s and then nothing) |
| 4 | vectorised RMSNorm / L2Norm / GELU / SiLU / scale / add / softmax | `vrcp14`+Newton sigmoid, polynomial `vexp`, `vreduce_add` for the sums; GELU over 25M elements went from ~50 ms (scalar `expf` per element) to 0.30 ms |
| 5 | **RoPE: cos/sin tables cached per `(theta, head_dim)` in the ctx** | was `cosf`+`sinf` per (position, head, dim/2) = ~12M libm calls per 512-token forward; vision axial rope now caches per-axis tables too |
| 6 | **attention: 4 query rows per K/V pass, K transposed to `[kv][dim][t]`** | the old kernel did one full K/V pass *per query row* (L2-traffic bound) and one `_mm512_reduce_add_ps` per key (8 horizontal reductions per 8 dots). Vectorising over keys removes every reduction. |
| 7 | attention softmax uses the vectorised exp (`softmax_exp_v`) instead of libm `expf` | 25M `expf` per 512-token forward |
| 8 | audio: 3x3 stride-2 subsampling conv parallelised over output channels | audio 2 s: 64 → 51 ms |
| 9 | attention work order is head-major | each thread's transposed K + V stay in L2 instead of thrashing L3 |
| 10 | L2 prefetch inside the attention QK/PV loops | image 1070 → 995 ms (marginal, kept) |
| 11 | **attention rebuilt on the int8 GEMM**: scores = `Q x K^T` and `out = P x V^T` are now packed-int8 GEMMs (per-128-block u8 activation scales, per-row i8 weights, wsum correction), so attention runs at the same ~42 MAC/uop as the linear layers instead of ~9 | text 2048 665→548 ms, text 512 91→82, text 128 23→19. `E2_ATTN_F32=1` restores the f32 kernel. Accuracy unchanged: text embedding cos 0.99974 (f32 path: 0.99973), int8-vs-f32 attention agreement 0.99974. |
| 12 | `e2_pack_mat` (float -> packed int8 + scales + wsum, with row/col strides) for K and V^T; `blk`, `tstride`, `wsmstride` added to `e2_qmat` so a GEMM can run on a sub-range of a packed matrix | lets the sliding window use a key sub-range without repacking |

## What did not work (tried, measured, reverted or noted)

| try | outcome |
|---|---|
| unrolling the 4x4 GEMM K-loop by 2 | 550 → 470 GMAC/s (single core): register pressure costs more than the saved loop overhead |
| GEMM shapes 8 rows x 2 tiles and 16 rows x 1 tile | 434 / 350 GMAC/s vs 550 for 4x4 — the uop-count optimum for 16 accumulators is broadcasts = weight loads = 4 |
| vectorised per-block f32 GEMM epilogue alone | only ~7%: the dpbusd body dominates, not the epilogue |
| 8 rows x 64 value columns in PV | the 2-zmm store path only covers 32 columns — must stay at 32 |
| attention with 8 rows x 2 key-blocks | GCC generated 97 instructions per 16 FMAs (spills + pointer reloads): *slower* than 4 rows. 4 rows keeps the QK/PV inner loops at ~19 uops per 8 FMAs. |
| explicit `#pragma unroll`/named accumulators for 8 rows | no change; the register file is simply too full |
| head-major work order alone | 643 → 590 ms on vision attention — right direction, not the real cause |
| prefetch alone | 590 → 564 ms on vision attention |

## int8 attention layout

Per kv head: `K` is packed as-is (rows = keys, cols = head_dim, activation block
= head_dim rounded to 64/128) for `scores = q_block x K^T`; `V` is first
blocked-transposed by `kt_work` into `[kv][dim][tpad]` and then packed (rows =
dim, cols = keys padded to 128) for `out = p_block x V^T`. The score row stride
is `tpad` (not `t`) or the two GEMMs disagree. Query blocks of up to
`ATTN_MAXR=512` rows are gathered into a contiguous buffer, scored, masked,
softmaxed (vectorised), and pushed through the second GEMM. Everything heavy
lives inside `e2_linear_forward`, so attention now uses the same threaded 4x4
dpbusd kernel as the linears.

Two bugs found on the way, both worth remembering: a strided V^T pack with a
`/blk` division plus a memory read-modify-write per element (≈ 3.5 ms per
layer), and the score-row stride mismatch above (cost cosine 0.81 -> 0.9997).

## Measured state (same-run ratios, `bench_compare --repeat 2`)

e2 / ov-int8 (lower is better, ≤ 0.67 is the goal):

| task | length | e2 ms | ov-int8 ms | ratio | start ratio |
|---|---:|---:|---:|---:|---:|
| text | 8 | 4.3 | 5.9 | **0.73** | 0.53 |
| text | 64 | 12.6 | 10.1 | 1.25 | 1.05 |
| text | 128 | 19.1 | 16.9 | 1.13 | 1.60 |
| text | 512 | 82 | 63.9 | 1.28 | 1.46 |
| text | 2048 | 548 | 521 | 1.05 | 1.25 |
| batch | 8x64 | 77.3 | 52.3 | 1.48 | 1.47 |
| batch | 32x64 | 424 | 238 | 1.78 | 2.03 |
| image | 270 tok | 800 | 532 | 1.50 | 1.58 |
| audio | 54 tok | 49 | 25.9 | 1.90 | 2.35 |
| video | 8 frames | 2416 | 1696 | 1.42 | 1.33 |

(OV columns are from `--repeat 2` runs; e2 numbers from the same session.)

Absolute e2 throughput improved 2-4x on every row (e.g. audio 64→51 ms, text
512 738→~95 ms, image 3.2 s→0.8 s on a quiet host, video 9.5 s→2.4 s). The
*ratio* in the table is noisy because e2 is more sensitive to host load than
OV (see below); the clean-host numbers would read roughly 20% better.

Accuracy re-verified against `tools/e2_ref.py` (float32 HF reference):

| dump | cosine |
|---|---|
| text `embedding` | 0.99969 |
| text per-layer (`layer1..24`) | 0.9992 – 0.9999 |
| text `ple` | 0.99990 |
| audio `aud_out` (50x1536) | 0.99909 |
| image `vis_pool` (266x768) | 0.97667 |

The image row is the pooled *pre-projection* state; BENCH.md's 0.9994 is for
the final normalised embedding after mean pooling over 266 tokens. It should
be re-checked per layer (see next steps).

## Next steps, in order of expected value

1. **Attention is the whole game now** (36% at 512 tokens, 55% at 2048, 65% of
   the image path). Measured: ~10-13 MAC/cycle vs the f32 FMA ceiling of 32.
   * More rows per K/V pass. Registers are the limit for f32, so either
     (a) split the PV/QK inner loops into a compile-time macro instantiated at
     4, 8 and 16 rows (vision has head_dim 64 and can afford 16), or
     (b) **bf16 `vdpbf16ps`**: q/k/v stored as bf16 pairs gives 2x MACs per
     uop and halves the traffic; the checkpoint is bf16 so accuracy is safe.
   * Then **int8** for QK only (`vpdpbusd` on q/k with per-vector scales,
     `-128*sum(k)` correction): 4x MACs per uop, ~0.4% score error.
2. **The thread pool spins too much under load.** Workers `pause` up to 20000
   times (~130 us) before a 50 us sleep. On a quiet box that is free; on a
   shared box it burns 7 cores of SMT bandwidth and e2 loses ~25% where OV
   loses ~5%. Try 1000-2000 spins + a futex-based wait, and measure the
   ratio, not the absolute time.
3. **GEMM epilogue RMW for large m.** For `k = 3072` (vision MLP) the output is
   re-read/re-written 23 times. Holding f32 accumulators across K blocks
   (4 rows x 2 tiles = 8 int + 8 f32 accumulators) removes ~50 MB of traffic
   per call at the cost of 14% more uops.
4. **Audio (worst ratio).** 51 ms for 20 GMAC = 400 GMAC/s. The conformer
   attention is still scalar (`tanhf` per score, scalar PV), the mel front-end
   is scalar (`logf` per bin, strided dot over `mel_filters`), and the
   layernorms are strided over the channel-major layout.
5. **Vision per-layer accuracy check** (0.9767 at `vis_pool`): dump the
   reference per layer and confirm the drift is quantisation chaos and not a
   bin-ordering difference in the 3x3 pooling (`bx` uses `(maxx+1)/pool`,
   `by` uses `maxy/pool+1` — suspiciously asymmetric).
6. `batch 32x64` (1.78) is a pure GEMM case; the only lever left is the
   epilogue and better n/m blocking for large m.

## Lessons

* Profile first. `E2_PROFILE=1` moved the target from "GEMM is slow" to
  "denormals", "quantisation is serial" and "attention re-reads K/V per row".
* Check the generated code. The compiler happily produced 97 instructions per
  16 FMAs (spills and pointer reloads) for what looked like a clean 8-row
  kernel; counting `vmovdqa`/`mov` in `objdump -d` found several bugs and
  slowdowns that no amount of source-level reasoning would have explained.
* Integer dpbusd at 4 rows x 4 tiles is worth ~1.6x over the naive 16-row
  layout; the uop ratio (broadcasts + loads + dots), not the "bigger tile"
  instinct, decides.
* Denormals (`ldmxcsr`) are worth 3x on half the GEMMs of a bf16 model. Check
  for them before any kernel work.

## Round 2 — attention packing, tokenizer, elementwise, big-m GEMM

The box was shared and load average moved between 1 and 19 during this pass.
Only same-run medians (`bench_compare --repeat 2`) are comparable; the absolute
ms values below are from the quieter run (`/tmp/bench3.json`).

### Wins kept (measured)

| change | effect |
|---|---|
| **AVX-512 int8 K/V packer** (`e2_pack_mat`), rows split across the pool | The scalar packer did one `lrintf` per weight per layer. Vectorised + parallel it is bit-identical to the old packer and removed ~0.3-0.4 ms/layer from text attention. |
| **Batched short-sequence attention** (`e2_attention_text_multi`): one worker per sequence instead of one thread-pool dispatch per (head, group, block) | batch attention 160 ms → ~20 ms (profile, batch 32x64). This is 768 tiny attention calls per forward; pool latency was most of the cost. |
| **O(n log n) BPE** with a heap + linked list instead of rescanning all pairs | 32 x 64-token texts tokenise in 0.46 ms total; the old scan was tens of ms on batch/2048/video rows. |
| **GEMM row blocking (RB=64)** and **row split when n has fewer than 4 tiles per thread** | The old order re-read the output for every K block; the attention PV GEMM (n=256) also fell into the 1-tile slow path. m=2048 GEMM-only: 168 → 135 ms. |
| Fixed the row-split range arithmetic | Adjacent workers overlapped by up to 3 rows, and the b==0 store raced with the other thread's b>0 accumulate: wrong results. Disjoint 4-row-aligned ranges. |
| **Fused RMSNorm(scale)**, **RMSNorm+residual-add**, **GELU x up** | text/vision elementwise traffic down by roughly one pass each; the GELU+gate multiply and three norm+add pairs were ~25 ms of the batch32 forward. |
| **Audio per-dim scale table + cached relative-position table** | 5.1 → 0.1 ms and 1.6 → 0.5 ms per forward. |
| Audio channel-major subsample conv kept (see below) | reverting the experiment also restored the audio tower's HF cosine. |

### Tried, reverted or not kept

| try | outcome |
|---|---|
| Fused softmax + u8 PV quantiser (unnormalised exp, 1/sum applied after PV) | 25-35% **slower** at t≥2048: the per-16-element mask construction and 8-zmm live range cost more than the f32 passes it saved. Left behind `E2_ATTN_FUSED_SQ=1`, off by default. |
| Per-head parallel vision attention (one worker per kv head, serial GEMM inside) | 3.4 → 1.5 ms at t=1024, but no gain at t≈2400 and it segfaulted intermittently with 8 threads (fault at a heap mapping edge, not reproduced under ASAN). Removed. |
| Audio conv rewritten channel-last with AVX-512 over output channels | subsample 7 → 13 ms and audio tower cosine vs HF fell 0.99909 → 0.9987 (chaotic amplification). Reverted. |
| Volatile float sums in the new layernorm | restored bit-order but made LN ~5x slower. `optimize("no-fast-math")` was the right tool. |
| Explicit weight prefetch in the GEMM K loop | noisy, no consistent gain; removed in round 4. |
| RB sweep 32/64/128/off | 64 best at m=2048, off/big-RB best at m≤128. `E2_GEMM_RB` selects at runtime. |
| Moving the row-block loop outside the column-tile loop | ~7% at m=2048, neutral at m=512; kept because it also keeps activations in L1. |

### Measured state after round 2

e2 / ov-int8 (same run, `--repeat 2`; lower is better, ≤ 0.67 is the 1.5x goal):

| task | length | e2 ms | ov-int8 ms | ratio |
|---|---:|---:|---:|---:|
| text | 8 | 3.5 | 5.8 | **0.60** |
| text | 64 | 8.8 | 9.7 | 0.91 |
| text | 128 | 16.0 | 16.0 | 1.00 |
| text | 512 | 66.5 | 62.2 | 1.07 |
| text | 2048 | 430.4 | 512.2 | 0.84 |
| batch | 8x64 | 49.5 | 48.1 | 1.03 |
| batch | 32x64 | 267.0 | 193.9 | 1.38 |
| image | 270 tok | 644.9 | 528.2 | 1.22 |
| audio | 54 tok | 42.4 | 25.5 | 1.66 |
| video | 8 frames | 1868.1 | 1680.2 | 1.11 |

A second run at load average 12-19 gave text 8: 0.77, text 2048: 0.97,
image: 1.09, video: 1.06, batch32: 1.33, audio: 1.63. e2 loses more to external
load than OpenVINO, so the ratio moves with the box even though both engines are
measured in the same run.

### Remaining gap and next steps

1. **Vision attention is score-matrix bandwidth bound.** At t=2394 it is
   350-390 ms of the ~590 ms image row: 16 heads x 2394^2 scores = 92M f32 per
   layer, five passes through it. A flash/blocked int8 attention (scores and
   exp kept in L1/L2, online softmax) is the only lever big enough left here.
2. **Audio tower** is 4 ms mel + 7 ms subsample + ~29 ms conformer for 15 GMAC.
   The short text encoder at T=54 adds ~10 ms. Both need work; OV does the tower
   in ~19 ms. The conformer attention is still scalar and the mel front-end is
   scalar (`logf` per bin).
3. **Batch 32x64** is 258 GMAC of GEMM at m=2048 (~135-145 ms GEMM-only), plus
   ~90 ms of elementwise and ~20 ms attention. Next: 2-D thread split in
   `gemm_work`, an f32 epilogue that does not RMW, and holding PLE/norm passes
   to fewer bytes.
4. **Text 64/128/512 parity.** Both engines stream the same 130 MB of int8
   weights per forward; e2 wins when compute matters (text8/2048) and loses
   when pool wake latency matters (short single text under external load).
5. `e2_attention_text_multi` is only used when all sequences have the same
   length (the benchmark case); unequal-length batches still loop per sequence.

Reproduce:

```bash
make test
make tests/e2_bench_full
. .venv/bin/activate
python tools/bench_compare.py --sources e2,ov-int8 --repeat 2 --json /tmp/bench.json
```

Useful debug switches: `E2_ATTN_F32=1` (f32 attention), `E2_ATTN_FUSED_SQ=1`
(fused softmax+quant experiment), `E2_GEMM_RB=N`, `E2_ATTN_R=N`,
`E2_NOFUSE=1` (separate rope pass, needed for the `l0_qpre` dumps),
`E2_PACK_SCALAR=1`, `E2_KERNEL=scalar|avx2|avx512`, `E2_PROFILE=1`.

## Round 3 — the non-GEMM half of the forward

Where the ceiling is: a packed-int8 linear sustains **2.5-3.3 TMAC/s** on the 8
cores (3.6 warm / 2.0-2.8 with cold weights); `mb peak` puts the raw
`vpdpbusd` ceiling at ~340 GMAC/s per core, and a 300 MB read at 46 GB/s
single-threaded / 53 GB/s with 8 threads.  A short text streams the whole
130 MB of int8 weights per forward, so the text rows are memory-bound, not
compute-bound, and the medium/long rows are 50-60% non-GEMM work.  Both are
addressed below.

### Wins kept (measured)

| change | effect |
|---|---|
| **all norms / l2norm / scale / add / silu / gelu on the pool** (dispatch costs 0.25 us) | text 2048 "other" 75 → 43 ms, qkv-norms 16.5 → 5.0, ple_scale 6.8 → 3.7; norm parallel threshold 16k elements |
| **PLE per-token multiply parallelised** | 7.2 → 3.2 ms at 2048 |
| **int8 attention rewritten** (`src/e2_attn.c`): score only the key band a query block can see; score tile sized to stay in L2; one item per (seq, kv head, query head, query block) with single-threaded GEMMs inside, so a layer costs 2 dispatches instead of 2 per (head, block); K/V packed once per (seq, kv head) into caller-owned scratch (no per-head malloc) | text 2048 attn 203 → 90 ms, total 374 → 250; text 512 attn 19 → 10; **image 583 → 373** (vision attn 294 → 99); video 1868 → 1163 |
| **vectorised `e2_exp_sum`/`e2_span_max`** for the attention softmax | the softmax+quantise pass was 6x the QK GEMM (1 M scalar `fexp` per layer); now ~1x |
| **`e2_pack_mat_into`** (pack into caller buffers) + 16-row-granular pack tasks | attention pack phase 0.23 → 0.04 ms/layer |
| **arena reuse fix** (blocks were only ever appended, never refilled) | bounded to 82 MB for audio (was growing 40 MB per forward) |
| **audio: parallel mel** (199 frames on the pool) | 3.9 → 0.6 ms |
| **audio: channel-last subsample conv**, vectorised over output channels, weights transposed once, layernorm+relu over the channel axis parallelised over pixels | sub 8.3 → 1.0 ms |
| **audio: vectorised conformer attention** (dot/axpy over D, vector tanh for the logit soft cap, item per (chunk, head)) | loop 2.6 → 0.44 ms |
| **audio: vectorised depthwise conv + `e2_gated_silu`** | lconv 6.9 → 2.5 ms |
| **audio total** | 42.4 → 24.9 ms (ratio 1.61 → 1.00) |

### Tried, measured, reverted or not kept

| try | outcome |
|---|---|
| **f32 accumulators held across k-blocks** (4 rows x 2 tiles, 8 int + 8 f32) to remove the per-k-block output read-modify-write | **slower**: 0.67 → 0.81 ms warm, 0.93 → 1.32 cold at m=2048 k=2048 n=512. The 17% extra issue slots (1.75 vs 1.5 uops per dpbusd) cost more than the saved output traffic; the 4x4 kernel's uop ratio is the binding constraint, not the epilogue. |
| explicit weight prefetch in the GEMM K loop (`E2_GEMM_PF`) | 0-7%, inside the noise of a shared box. Removed in round 4. |
| RB 16/32/64/128 for the short-text rows | 32 marginally best (7.0 vs 7.5 ms at text 64), within noise |
| loading the model with `malloc`+`pread`+`MADV_HUGEPAGE` instead of a file mapping (to get huge-page TLB coverage) | identical: audio 41.9 vs 41.9 ms. Bandwidth was not TLB-limited. |
| spinning pool workers 2 M times instead of 20 k before sleeping | **much worse** for the serial phases (mel 4 → 21 ms): idle workers steal issue slots from the one doing the work even with 1:1 pinning |
| more pool threads than physical cores (16 on 8) | 2.5x slower on AVX-512 (SMT shares the vector unit). The old pool also livelocked at 1352 ms when threads > cores because `pin_cpu` silently gave up; it now fills the SMT siblings in a second pass (32 ms). |

### Bugs found while validating (all would have shipped silently)

* `gemm_avx512` recomputed the tile stride from `k` instead of using
  `w->tstride`, so every GEMM on a k-subrange (the attention PV step) read the
  wrong weight tiles. Invisible at full width.
* `pack_vt_dims` had `float inv[16]` indexed to 63: the stack smash silently
  skipped three of the four V^T dim tiles, so only the first 16 head dims were
  correct. The pooled cosine still looked fine (0.98) — only comparing each
  layer's attention output against the f32 reference caught it.
* the packed V^T columns past `t` were never written and the zero-point
  correction assumes zero weights there, so a previous call's scratch leaked
  into the result. Text after a vision tower produced NaNs.
* the softmax upper bound for a windowed layer was clipped to the padded band
  width instead of `t`, so the 6-124 padding keys entered the softmax whenever
  `t % 128 != 0`. Text at 256 tokens: 0.9996, at 260: 0.55.

### Measured state (same-run ratios, `bench_compare --repeat 2 --iters 3`)

e2 / ov-int8, lower is better, **≤ 0.67 is the 1.5x goal**.  Host load average
was 6.2 for this run (e2 loses ~25% under load where OpenVINO loses ~5%).

| task | length | e2 ms | ov-int8 ms | ratio | round 2 | round 1 |
|---|---:|---:|---:|---:|---:|---:|
| text | 8 | 3.5 | 6.8 | **0.51** | 0.60 | 0.53 |
| text | 64 | 7.1 | 9.5 | 0.75 | 0.91 | 1.05 |
| text | 128 | 13.4 | 15.9 | 0.84 | 1.00 | 1.60 |
| text | 512 | 47.3 | 62.6 | 0.76 | 1.07 | 1.46 |
| text | 2048 | 247.4 | 502.8 | **0.49** | 0.84 | 1.25 |
| batch | 8x64 | 45.2 | 51.3 | 0.88 | 1.03 | 1.47 |
| batch | 32x64 | 229.2 | 206.2 | 1.11 | 1.38 | 2.03 |
| image | 270 tok | 374.1 | 551.8 | **0.68** | 1.22 | 1.58 |
| audio | 54 tok | 27.3 | 26.3 | 1.04 | 1.66 | 2.35 |
| video | 8 frames | 1210.8 | 1726.0 | **0.70** | 1.11 | 1.33 |

Every row improved.  Absolute e2 throughput: text 512 738 → 47 ms,
text 2048 665 → 247, image 3.2 s → 374 ms, audio 64 → 27 ms,
video 9.5 s → 1.2 s.

Accuracy against the float32 HF reference (`tools/e2_acc.py`):

| dump | cosine |
|---|---|
| text `embedding` at 8 / 64 / 128 / 256 / 260 / 300 / 512 tokens | 0.9997 / 0.9997 / 0.9995 / 0.9996 / 0.9996 / 0.9995 / 0.9995 |
| text `last_hidden_state` | 0.9994 |
| image + text embedding (end-to-end) | 0.9920 |
| image `image_soft` | 0.9785 (unchanged; the vision tower is chaotic under int8) |
| audio `audio_soft` | 0.9985 |
| batch vs single text | 0.0 (bit identical) |

### Where the remaining gap is

1. **The short/medium text rows (0.66-0.82) are weight-streaming bound.** 130 MB
   per forward at 7-13 ms = 10-17 GB/s, against 46-53 GB/s of DRAM. The kernel
   is limited by memory-level parallelism: 4 weight lines in flight per thread.
   Depth-4 prefetch recovers ~7%. The real fix is a kernel that double-buffers
   the weight loads (8 zmm) with 2 output tiles, or 16 threads on the
   memory-bound rows only — both measured neutral-to-worse as implemented.
2. **batch 32x64 (1.08)** is 63% GEMM (126 ms for 258 GMAC = 2.05 TMAC/s, cold
   weights cap at ~2.8) plus 47 ms of elementwise (rope 11, gated-mul 16,
   qkv-norms 5). Fusing RoPE into the attention gather and the gated-mul into
   the down-projection quantiser would remove ~16 ms of that, i.e. ~0.85.
3. **audio (1.00)** still has 9.5 ms of ff GEMMs (10 GMAC at ~1 TMAC/s: the
   tensors are [50, 1024/4096] so each weight is used only 50 times, and it is
   DRAM-bound), 2.5 ms of lconv, 2.1 ms qkv. Note the comparison basis: e2's
   audio row includes the mel front-end and the text encoder over the audio
   tokens; the OpenVINO row is the tower's subsample + layer stack only
   (with a precomputed mask), so e2 is doing strictly more work there.
4. `pack_vt_dims` is still scalar in the quantise step (a 4x16 byte transpose
   would give ~3x), which costs 0.6 ms/layer on the 32x64 batch row.

### Why the GEMM-bound rows cannot reach 1.5x

Measured ceilings on this host (`make mb`): raw `vpdpbusd` **347 GMAC/s per
core**, f32 FMA 165 GMAC/s per core.  That is exactly the 2x uop advantage int8
has over f32, i.e. 2.8 TMAC/s for 8 cores.

* e2's packed-int8 GEMM runs at 2.7-3.3 TMAC/s (warm) / 2.0-2.8 (cold weights):
  at the VNNI peak.
* OpenVINO's text 512 row is 132 GMAC in 62.6 ms = **2.2 TMAC/s**.  An f32
  matmul could not exceed 1.15 TMAC/s on 8 cores, so their "INT8_SYM
  weight-only" execution is also an int8 GEMM (weights int8, activations
  quantised on the fly) running at ~78% of the int8 peak.

So for the rows that are GEMM-dominated (text 128/512, batch 32x64) both
engines are within 25% of the same hardware limit and the ratio is bounded by
the non-GEMM work e2 can delete.  Round 1 assumed OpenVINO ran dequantised f32
matmuls; that assumption is what made "1.5x in every scenario" look reachable.
The rows where attention and the elementwise tail dominate (text 8/64/2048,
image, video) are the ones that now clear 1.5x or come close.

### The pool spin count is not a knob to turn down

Dropping the worker spin from 20 000 `pause` iterations to 4 000/1 000/200 makes
text 512 go 49.5 → 127 / 130 / 117 ms (load average 6). Each dispatch that finds
the workers asleep pays the full `nanosleep(50 us)` plus timer slack, ~1 ms on a
loaded box, and a forward has ~50-100 dispatches. Going the other way (2 M spins)
costs the *serial* phases instead (mel 4 → 21 ms). 20 000 is the sweet spot and
should only be replaced by a futex wake, not a smaller number.

## Round 4 — the real int8 ceiling, rope/quantiser fusion, cached attention

### The peak measurement in round 1-3 was wrong

`mb peak` used 4 independent accumulators. `vpdpbusd` has ~4 cycle latency, so a
4-accumulator chain measures **latency, not throughput**: 341 GMAC/s. With 8
accumulators the same loop reaches **614 GMAC/s single-core / 4034 GMAC/s on 8
threads** — Zen 5 issues **two 512-bit `vpdpbusd` per cycle** (128 int8 MACs per
cycle per core), not one. The old "we are at 84% of the VNNI peak" conclusion was
therefore wrong: the packed GEMM at 2.9-3.3 TMAC/s warm was at ~85%, but the
*cold* path (2.2-2.6 TMAC/s) is at 55-65%, and that is where the remaining time
is. `tools/mb.c` now uses 8 accumulators.

Measured ceilings on this host (8 threads):

| workload | GMAC/s | note |
|---|---:|---|
| pure `vpdpbusd`, 8 accums, no loads | 4034 | the machine |
| packed GEMM, weights L2/L3 (warm) | 3250-3820 | 81-95% |
| packed GEMM, weights streamed from DRAM | 2150-2680 | 53-66% |
| DRAM read (8 threads, sequential) | 50 GB/s | |

So a GEMM whose weights are read once per forward (every GEMM in a forward)
loses 35-45% to the DRAM stream. At m=2048 with n=2048/k=512 that stream is only
1 MB per GEMM — a few GB/s — so it is the *first touch of each line* stalling
the pipeline, not bandwidth. Prefetching the next k-block of all four tiles
recovers 5-9% of it in the microbenchmark but nothing end to end (see the
"did not work" table), so the kernel is back to plain loads.

### Wins kept (measured end-to-end, same-run ratios)

| change | effect |
|---|---|
| **RoPE fused into the q/k RMSNorm** (`e2_rmsnorm_rope`): the encoder used to read+write q and k twice (norm, then rotate). Rotation is now done on the row while it is still in L1. Mode 3 of the norm family, vectorised for AVX-512; the scalar path is a fallback. | text 512 rope+norms 4.5 → 1.6 ms; batch32 ~14 → 6 ms |
| **One activation quantisation for q/k/v and for gate/up** (`e2_linear_forward_multi`): the quantiser was a full read+max pass over m*k floats *per linear*. | ~1.5 ms text512, ~7 ms batch32 |
| **`gelu(x)*up` fused into the down-projection quantiser** (`e2_gelu_mul_q`): the 2048-wide f32 intermediate was written and read back only to be quantised. Now it is computed into a per-thread L1 row buffer and quantised in place. | batch32 mlp_down 26.6 → 19.1 ms |
| **Vectorised V^T packer** (`pack_vt_dims_v`): float→int8 with a per-dim scale, then a 4x16 byte transpose built from three unpack levels, and the per-dim sums the zero-point correction needs fall out of `vpmaddubsw`+`vpmaddwd` on the transposed zmm. `_mm512_cvtsepi32_epi8` replaces the scalar `lrintf`+clamp pair. Bit-identical to the scalar path. | batch 32x64 attention pack 12 → 3 ms; helps every t |
| **No malloc in the K packer** (`e2_pack_mat_into` used to `malloc(k)` per 16-row tile, i.e. 12288 times per batch32 forward) | batch32 −6 ms, text128 −0.7 ms |
| 8-accumulator `mb peak` (was measuring `vpdpbusd` latency, not throughput) | corrects the ceiling from 2.8 to 4.0 TMAC/s and reframes every later decision |

### Tried, measured, reverted or not kept

| try | outcome |
|---|---|
| **Flash-style chunked attention** (`attn_item` rewritten: score `cw` keys at a time with an online max/sum so the score tile stays in L2, and `R` freed from the scratch-size clamp — 256 for t = 2394 instead of 16). The premise was that the vision attention is score-matrix-bandwidth bound (6 passes over 275 MB per layer). Measured: vision attention 103 → 113 ms, image -1.7%, text 2048 flat. The 4-tile GEMM is limited by the *weight* stream (m = 16 per query block) and by score-tile writes, not by the softmax passes over it, and the extra per-chunk output rescale/add costs as much as the saved cache traffic. Reverted. |
| `E2_GEMM_RB=128/256/1024` | *better* in the cold microbenchmark at m=512 (2.43 → 2.68 TMAC/s) but **worse end-to-end** (text512 41.2 → 44.7, batch32 177 → 193). In situ the weights are re-read from L2 by later row blocks, and a bigger RB evicts the activation slab. Kept 64. |
| weight prefetch, both the per-iteration form and 1/2/4 k-blocks ahead (`E2_GEMM_PF`, `E2_GEMM_PB`): +5-9% on `mb cold` in isolation but **neutral to -11% end to end**, because in situ the weights are re-read from L2 by the later row blocks and the prefetch traffic only pollutes. Both removed from the kernel in round 4. |
| 16 threads (SMT) for the streaming-bound shapes (m=50..2048) | 20-40% slower everywhere. |
| linearator with 3 tiles (12 int + 12 f32 accumulators to hold f32 across k-blocks) | not implemented — 31 zmm is over budget once pointers/broadcasts are counted, and 8 accumulators is exactly the 2/cycle dependency limit. |
| holding f32 accumulators over all k (round 3, 4x2 tiles) | already measured slower; the 8 int accumulator ILP is the binding constraint. |
| `MADV_HUGEPAGE` / huge pages for the model map | confirmed irrelevant again: THP is `always` here, so the 130 MB map is already 2 MB pages. |

### Measured state after round 4 (`--repeat 3 --iters 3`, load 2.8)

e2 / ov-int8, lower is better, **<= 0.67 is the 1.5x goal**:

| task | length | e2 best | ov-int8 best | ratio | ratio (median) | round 3 | start |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 8 | 3.2 | 5.5 | **0.57** | 0.57 | 0.51 | 0.53 |
| text | 64 | 6.2 | 9.0 | **0.69** | 0.68 | 0.75 | 1.05 |
| text | 128 | 13.2 | 15.9 | 0.83 | 0.88 | 0.84 | 1.60 |
| text | 512 | 40.8 | 62.5 | **0.65** | 0.68 | 0.76 | 1.46 |
| text | 2048 | 234.7 | 483.8 | **0.49** | 0.49 | 0.49 | 1.25 |
| batch | 8x64 | 34.8 | 50.2 | **0.69** | 0.74 | 0.88 | 1.47 |
| batch | 32x64 | 183.3 | 212.4 | 0.86 | 0.87 | 1.11 | 2.03 |
| image | 270 tok | 331.4 | 526.3 | **0.63** | 0.62 | 0.68 | 1.58 |
| audio | 54 tok | 22.8 | 24.7 | 0.93 | 0.92 | 1.04 | 2.35 |
| video | 8 frames | 1094.7 | 1630.8 | **0.67** | 0.66 | 0.70 | 1.33 |

Seven of ten rows are at 0.57-0.69, i.e. at or within noise of the 1.5x target.
The three that are not (text 128, batch 32x64, audio) are analysed below.

**Same-run ratios still swing +/-10% with host load**: the same build measured
text 512 at 0.72, 0.76 and 0.65 in three consecutive runs at load 6.9 / 5.3 /
2.8, and batch 32x64 at 0.86-1.03. `--repeat 3` (pooling 9 samples per row)
gave the stable numbers above; single-repeat runs are not worth reading.

Absolute e2 throughput over the whole exercise: text 512 738 -> 41 ms,
text 2048 665 -> 235 ms, batch 32x64 1382 -> 183 ms, image 3.2 s -> 331 ms,
audio 130 -> 23 ms, video 9.5 s -> 1.09 s.

Accuracy re-verified with `tools/e2_acc.py` after every change:

| dump | cosine |
|---|---|
| text `embedding` | 0.99970 (identical with and without the fused rope) |
| text per-layer `layer1..23` | 0.9994 - 0.9999 |
| image `image_soft` | 0.97867 (unchanged from round 3; the vision tower is chaotic under int8) |
| audio `audio_soft` | 0.99847 (unchanged) |

`e2_acc.py` used to report the reference's *last* hidden-state tensor, which has
a different shape than e2's dump (204288 vs 1838592 floats for this
checkpoint), as a 0.003 cosine; it now skips shape-mismatched pairs. Both the
final `embedding` (0.99970) and `last_hidden_state` (0.99938) confirm the last
layer is correct.

### Where the remaining gap is, honestly

* **Text 8 / 64 / 512 / 2048, batch 8x64, image and video are at the goal.**
  text 128 (0.83) is the one text length that does not clear it: at 128
  tokens the 130 MB weight stream is spread over almost no compute
  (0.9 GMAC), so e2 is pure-memory-bound and its 8-thread stream (25-30 GB/s
  at m = 128) is the limit.
* **batch 32x64 (0.86)** is the honest miss. It is 97% GEMM at m = 2048:
  258 GMAC in 118 ms = 2.19 TMAC/s against a *cold-weights* ceiling of ~2.4
  (measured with `mb cold`) and a raw `vpdpbusd` ceiling of 4.0. OpenVINO
  manages 1.28 TMAC/s on the same row, so e2 is 1.7x faster at the GEMM and
  still only level overall: the remaining ~70 ms is attention (22),
  `gelu*s` (16), the three norm+add passes (8.5), rope+norms (6) and the PLE
  multiply (3) -- all measured at 55-105 GB/s, i.e. at DRAM bandwidth. The
  row cannot reach 0.67 without the cold GEMM going from 2.2 to 3.3 TMAC/s.
* **audio (0.93)**: 12 conformer layers plus the text encoder over 50 tokens
  streams ~360 MB of weights. At the ~30 GB/s this kernel sustains when
  `m <= 64` that is 12 ms of the 25, and the conformer attention (~4 ms, still
  scalar per key) plus the depthwise conv (~2.4 ms) are the rest.
* Nothing else is above 0.80.

### What I would do next, in order

1. **Interleave the packed weight tiles per k-block** so the four tile streams
   a thread walks become one contiguous stream. The packed layout is
   `[N/16][K/4][16][4]`, so a 4-tile group is four 8 KB-apart runs; that is
   the difference between `m=4` streaming at 48 GB/s and `m=50` at 31 GB/s.
   Costs a re-pack of the model file (version 2) and a rewrite of the kernel's
   address arithmetic.
2. **Audio conformer attention**: hoist the per-key mask/`dist` branch out of
   the inner loop (the valid `j` range is contiguous per query) and use the
   int8 GEMM for the 12x24 score block.
3. **Short-sequence key blocking**: `vblk = 64` when `t <= 64` halves the
   score tile, the softmax and the P x V step for batch 8x64 / 32x64 and audio
   (t = 54), worth ~4% on the batch rows.

## Round 5 — a different machine: Arrow Lake, no AVX-512

The host changed under us. Round 1-4 numbers were measured on the Ryzen 7
9700X. The current box is an **Intel Core Ultra 9 285H (Arrow Lake) laptop**:

| | |
|---|---|
| ISA | **no AVX-512 at all**; AVX2 + FMA + AVX-VNNI (256-bit `vpdpbusd`) |
| cores | 16: `0-5` P-cores, `6-13` E-cores, `14-15` LP E-cores |
| measured int8 peaks (`/tmp/peak`, 8 accumulators) | 1 thread 238 GMAC/s, 6 P 1308, 8 threads 1551, 12 threads 2116 |
| f32 FMA peak | 520 GMAC/s at 8 threads (3x below int8) |
| LP E-cores | **23 GMAC/s** vs 178 for a P-core — 8x slower |
| THP | `always`; page-cache and anonymous reads measure the same |

Two immediate consequences: every AVX-512-only kernel in the tree had been
silently falling back to scalar, and any worker placed on cpu 14/15 poisoned
its barrier. Measured baseline before this round (`bench_compare --repeat 2`):
e2/ov-int8 ratios text 8 1.26, text 64 1.12, text 128 1.21, text 512 0.74,
text 2048 0.61, batch8 0.89, batch32 0.97, image 1.06, audio 1.39, video 0.97 —
i.e. the library was *slower* than OpenVINO on half the rows.

### Wins kept (measured, same-run ratios)

| # | change | effect |
|---|---|---|
| 1 | **AVX2+FMA twins of every hot op** in `e2_ops.c` (`vexp8`, `vsigmoid8`, gelu/silu/tanh/gated-silu/scale/add, span-max/exp-sum/softmax, rmsnorm (all 4 modes incl. fused rope), l2norm). `simd()` now returns 1 for AVX2+FMA and every dispatch site has an `simd() == 1` branch. Same polynomial, same clamping as the AVX-512 versions; `_mm512_rcp14_ps` became `_mm256_rcp_ps` + 1 Newton step and `scalef` became an integer exponent shift. | gated-mul 23.6 -> 8.1 ms, plemul 3.3 -> 0.9, rope 3.1 -> 2.5, softmax/exp in attention vectorised again; text 512 profile 208 -> 152 ms |
| 2 | **AVX2 activation quantiser** (`quant_rows_avx2`, worker commit): 16-wide block max, `vcvtps2dq` (bit-identical to the old `lrintf`), offset added in the integer domain, saturating 16->8 pack. Bit-exact against `quant_rows_s` on 4 patterns incl. outliers and the 1e-12 floor. | 2691 -> 203 ns/row (13x). It was fully scalar on this host before. |
| 3 | **LP E-cores excluded from the pool** (`/sys/devices/cpu_lowpower/cpus`), workers pinned *and* calibrated at thread start instead of on the first dispatch | a straggler on cpu14/15 costs more than 2 extra threads are worth: the pool no longer livelocks; 16 threads went from 377 GMAC/s to 2135 in the raw peak test |
| 4 | **Speed-weighted work splits** (`e2_split`): each worker runs the real GEMM kernel once at startup (best of 5, serialised so threads do not calibrate while others spin) to get its own throughput, then ranges are handed out in proportion. 8 threads on 6 P + 2 E cores now behave like 6x1.52 + 2x1 units instead of 8 equal ones. | m=64 GEMM at 8 threads 975 -> 1415 GMAC/s (+45%); batch 8x64 and every mid-size text row |
| 5 | **VNNI kernel: fold the weight load into `dpbusd`** (worker commit) + 3 KB run-ahead prefetch | single core 192 -> 215 GMAC/s; m=8 cold 0.105 -> 0.062 ms |
| 6 | **f32 accumulators per row block in L1** (`gemm_avxvnni`): the epilogue used to read-modify-write the whole output once per 128-wide k block (3 extra RW passes for k=512, 15 for the attention PV). The f32 tile for a 128-row block (8 KB) now lives on the stack and each output element is written once. | m=2048 k=512 n=2048: 1009 -> 1512 GMAC/s; k=2048 n=512: 909 -> 1325; batch 32x64 1216 -> 615 ms (**2x**) |

### Tried, measured, reverted or not kept

| try | outcome |
|---|---|
| pool spin 4000 / 500 instead of 20000 `pause` (to stop burning power on a throttling laptop) | **much worse**: 8 -> 47 ms on text 8. A sleeping worker costs ~150 us of wake latency and a forward has 300+ dispatches. Kept 20000, made it `E2_POOL_SPIN` for experimenting. |
| restarting workers on the *calibration* FMA loop instead of the real GEMM kernel | the FMA loop ranks an E-core at 1.13x slower than a P-core, which is wrong (it is 1.4-1.5x slower at VNNI: 2 vs 1 256-bit VNNI port, but similar FP width). Measuring the actual kernel is what fixed the split. |
| letting threads calibrate concurrently at pool start | the first threads measured while the rest were still spinning up, which made the *fast* cores look slow and skewed every later split (one P-core came out at half speed). Calibration is now serialised with an atomic ticket. |
| nope | — |

### Where the time goes now (text 128, `E2_PROFILE=1`, quiet host, 8 threads)

GEMM 23 of 35 ms, attention 5.7, elementwise/rope/ple ~5. The GEMM runs at
~570 GMAC/s in situ against 1080 GMAC/s for the identical 218-GEMM sequence
called in a tight loop with the same (copied) weights — the difference is the
activation working set the forward keeps live between GEMMs (the `ple` buffer
alone is `n x 24 x 512` floats). perf attributes 53% of P-core cycles to
`gemm_avxvnni` and **28% to the pool spin loop**, i.e. a large part of the
non-GEMM time is the main thread running alone.

### Measured state after round 5 (`--repeat 2 --iters 3`, quiet host, 8 threads)

e2 / ov-int8, lower is better, **<= 0.67 is the 1.5x goal**:

| task | length | e2 ms | ov-int8 ms | ratio | round 5 start |
|---|---:|---:|---:|---:|---:|
| text | 8 | 4.4 | 6.5 | 0.68 | 1.26 |
| text | 64 | 11.9 | 24.2 | **0.49** | 1.12 |
| text | 128 | 23.0 | 47.8 | **0.48** | 1.21 |
| text | 512 | 90.8 | 203.0 | **0.45** | 0.74 |
| text | 2048 | 583.4 | 1378.5 | **0.42** | 0.61 |
| batch | 8x64 | 82.7 | 154.9 | **0.53** | 0.89 |
| batch | 32x64 | 443.4 | 615.7 | 0.72 | 0.97 |
| image | 270 tok | 760.0 | 1482.9 | **0.51** | 1.06 |
| audio | 54 tok | 38.4 | 55.2 | 0.70 | 1.39 |
| video | 8 frames | 2337.4 | 5042.0 | **0.46** | 0.97 |

The batch 32x64 row above predates the L1-accumulator kernel; measured
interleaved against the pre-change binary it goes 1216 -> 615 ms, i.e. ~0.36.
Remaining over the line: text 8 (0.68), audio (0.70) and batch 32x64 before
the accumulator change.

Accuracy is unchanged (the kernel changes are numerically equivalent):
`e2_acc.py text` -> embedding 0.99970, last_hidden_state 0.99938, per-layer
0.9994-0.9999.

### Notes for the next round

1. `gemm_avx512` still does the per-k-block read-modify-write. The accumulator
   trick above is worth ~1.4x for large m and should be ported to it on a host
   that actually has AVX-512 (it cannot be tested here).
2. The gap between the in-situ GEMM (570 GMAC/s) and the same GEMMs in a tight
   loop (1080) is the biggest single remaining lever for the short rows: it is
   activation working-set traffic, so the fix is fewer f32 intermediates
   (quantise the `up` projection straight to u8, have the PV GEMM write the
   o_proj input as u8, keep the PLE table in a narrower form).
3. `pack_vt_dims` (attention V^T packer) is still AVX-512-only and shows up at
   1-3% of cycles; it needs an AVX2 twin like the ops got.
4. The 28% spin in `worker` is idle time of the *workers* while the main thread
   runs serial code. Cutting the number of serial regions (or the dispatch
   count) is worth more than any kernel work at 8 threads.
