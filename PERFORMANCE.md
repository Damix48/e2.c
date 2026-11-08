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

### Measured state after round 5 (`--repeat 2 --iters 3`, quiet-ish host, 8 threads)

e2 / ov-int8, lower is better, **<= 0.67 is the 1.5x goal**. Same run for both
engines, medians pooled over 6 samples per row:

| task | length | e2 ms | ov-int8 ms | ratio | speedup | round 5 start |
|---|---:|---:|---:|---:|---:|---:|
| text | 8 | 4.6 | 7.0 | **0.66** | 1.52x | 1.26 |
| text | 64 | 11.9 | 22.9 | **0.52** | 1.92x | 1.12 |
| text | 128 | 24.0 | 47.6 | **0.50** | 2.00x | 1.21 |
| text | 512 | 84.4 | 205.8 | **0.41** | 2.44x | 0.74 |
| text | 2048 | 452.0 | 1395.1 | **0.32** | 3.09x | 0.61 |
| batch | 8x64 | 73.5 | 154.9 | **0.47** | 2.11x | 0.89 |
| batch | 32x64 | 305.1 | 625.1 | **0.49** | 2.05x | 0.97 |
| image | 270 tok | 649.7 | 1499.9 | **0.43** | 2.31x | 1.06 |
| audio | 54 tok | 35.5 | 57.4 | **0.62** | 1.62x | 1.39 |
| video | 8 frames | 2271.3 | 5077.4 | **0.45** | 2.24x | 0.97 |

**Every row is at or past the 1.5x target.** Absolute e2 throughput over the
round: text 8 9.3 -> 4.6 ms, text 64 27.9 -> 11.9, text 128 54.8 -> 24.0,
text 512 155.6 -> 84.4, text 2048 879 -> 452, batch 32x64 621 -> 305,
image 1649 -> 650, audio 79 -> 35.5, video 4981 -> 2271.

The two narrowest margins are text 8 (1.52x) and audio (1.62x). text 8 streams
the whole 133 MB of text weights in 4.6 ms = 29 GB/s, which is at this
machine's practical sequential-read limit, so it cannot improve much further
without touching fewer bytes. Audio's margin comes from the conformer tower
(12 layers, T=50: the GEMM there is DRAM-bound because each weight is used
only 50 times).

### Accuracy (vs the float32 HuggingFace reference, `tools/e2_acc.py`)

| dump | cosine | round 4 |
|---|---|---|
| text `embedding` | 0.99970 | 0.99970 |
| text `last_hidden_state` | 0.99938 | 0.99938 |
| text per-layer | 0.9994-0.9999 | same |
| image `image_soft` | 0.97788 | 0.97867 |
| audio `audio_soft` | 0.99361 | 0.99847 |

The audio number moved because the conformer tower amplifies activation
quantisation noise the same way the vision tower does: a 1e-7 change in a
layer-1 intermediate crosses u8 rounding boundaries and the trajectories then
diverge (aud_feat and aud_in are bit-identical, layer 1 differs by 1e-4 rms
and the final audio embedding lands 0.6% away). The vectorised path is not
"wrong" (with the ops forced back to scalar, `E2_OPS_SCALAR=1`, the current
build reproduces the pre-round binary to 0.9995), it is a different sample of
the same chaotic system. Worth re-checking if audio retrieval quality is ever
questioned; the fix would be higher-precision activation quantisation in that
tower, not a different kernel.

### Bugs found while validating (all three would have shipped silently)

* `e2_linear_forward_multi` **truncated its linear list** at the first linear
  that needed an input clip or had a different k: `n = i; break;` skipped every
  linear after it. The audio tower's q/k/v are all clipped linears, so the new
  shared-quantiser call computed q and k and left `v` as whatever was in the
  arena. Caught by comparing staged dumps against the previous binary
  (`aud_l1` cosine 0.90). The fix runs each incompatible linear through the
  single-linear path and keeps the shared quantiser for the rest.
* The audio attention used `-INFINITY` as its mask sentinel and then passed the
  score row through `tanh`. `tanh` maps -inf to -1, so the sentinel was gone;
  worse, the vectorised `tanh` turned it into **NaN** (`rcp(inf)` then
  `r*(2-inf*0)`), which poisoned the whole row. The mask now lives in its own
  `uint8_t ok[]` array and `tanh_v`/`tanh_a2` clamp their input to +-20 first
  (tanh(20) is already exactly 1 in f32), so +-inf can never reach the exp.
* `e2_ops.c`'s AVX2 `simd()` returned 1 for AVX2 without checking FMA, which
  would have run the FMA kernels on a pre-Haswell part.

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

### Debug switches added this round

`E2_KERNEL=scalar|avx2|avxvnni|avx512`, `E2_QUANT=scalar|avx2|avx512`,
`E2_GEMM_RB=N` (row block, 32 default), `E2_GEMM_PF=N` (weight prefetch
distance in 64-byte rows, 48 default), `E2_POOL_SPIN=N` (20000),
`E2_OPS_SCALAR=1` (force the scalar elementwise ops; the audio dump comparison
above uses it), `E2_DEBUG_TW=1` (print the per-thread speed weights),
`E2_GTIME` was used temporarily and is gone.

### Reproduce

```bash
make && make tests/e2_bench_full
. .venv/bin/activate
python tools/bench_compare.py --model ~/Projects/embeddinggemma-2 \
    --sources e2,ov-int8 --repeat 2 --iters 3 --threads 8   # -> ratio table
python tools/e2_acc.py text|image|audio --model ~/Projects/embeddinggemma-2
```

The box is shared and thermally throttled: a single run of the same binary can
read 4.6 or 8.0 ms on text 8 depending on what else is running. Only the
same-run e2-vs-OV ratio means anything, and `--repeat 2` or 3 is required.

## Round 5 audit — replaying round 5 on an AVX-512 host

Round 5 was developed on an Arrow Lake laptop (no AVX-512). Re-running
`master` (7bb7293) against this branch on the Ryzen 7 9700X, interleaved,
min of 4-6 paired runs, showed two regressions, both in code that only runs
on the AVX-512 path:

| row | master | round 5 as merged | + this fix |
|---|---:|---:|---:|
| text 8 | 3.3 | 3.7 (+12%) | 3.3 (+1%) |
| text 64 | 5.8 | 6.7 (+14%) | 6.0 (+3%) |
| text 128 | 11.4 | 12.5 (+10%) | 11.4 (-2%) |
| text 512 | 44.0 | 45.8 (+4%) | 43.0 (+1%) |
| text 2048 | 239 | 257 (+7%) | 242 (+3%) |
| batch 8x64 | 37.0 | 38.2 (+3%) | 36.5 (-1%) |
| batch 32x64 | 179 | 198 (+11%) | 176 (-1%) |
| image | 328 | 345 (+5%) | 334 (-2%) |
| audio | 23.3 | 23.9 (+3%) | 21.9 (-4%) |
| video | 1383 | 1428 (+3%) | 1413 (+1%) |

1. **`e2_split()` truncation.**  Thread weights on a uniform CPU differ by
   ~1% (calibration noise); truncating the cumulative boundary turned that
   into a whole extra work unit (nblk=32 at 8 threads: thread 0 got 3 tiles,
   thread 1 got 5).  That is -25%/+25%, and it cost 10-18% on every GEMM with
   n = 512/768/1024 (`q` +18.6%, `mlp_gate` +14.6%).  *Single-threaded*
   `mb gemm 512 512 512` was faster on this branch (496 vs 465 GMAC/s), which
   is what pointed at the split rather than at the kernel.  Fixed by rounding
   the boundary to the nearest unit; `lo(tid) == hi(tid-1)` still comes from
   one cumulative sum, so ranges stay contiguous, and the proportional
   weighting on a hybrid CPU is unchanged.

2. **`g_rb` 64 -> 32 applied to both kernels.**  32 is right for the new
   AVX-VNNI kernel (it keeps an RB x 16 f32 accumulator tile in L1);
   `gemm_avx512` is byte-identical to round 4 and has no such tile, so 32 cost
   8-17% on the short/mid text rows.  Split into `g_rb` (VNNI, 32) and
   `g_rb512` (AVX-512, 64); `E2_GEMM_RB` still overrides both.

Neither regression touches AVX2-only hosts: the AVX2/FMA kernels are inactive
here (`simd()` returns 2) and the prefetch knob only lives in `gemm_avxvnni`.

Confirmed while auditing: the `e2_linear_forward_multi` skip bug was real —
text embeddings are bit-identical to round 4, the audio embedding changed and
its cosine against the HF reference improved 0.99847 -> 0.99865.

## Round 6 — Arrow Lake host (Core Ultra 9 285H), GPU backend

New host, same code: Intel Core Ultra 9 285H (6 P + 8 E + 2 LP-E, **no
AVX-512**, AVX-VNNI only), Intel Arc 140T iGPU (Xe-LPG, 1024 shaders, no
XMX/DPAS, DP4A only). `e2_cpu_count()` returns 14 (LP cores skipped). All CPU
rows below are 8 threads, matching what the harness gives OpenVINO.

### Baseline before any round-6 change (`--repeat 2 --iters 3`, 8 threads)

| task | length | e2 ms | ov-int8 ms | e2/ov |
|---|---:|---:|---:|---:|
| text | 8 | 4.4 | 7.1 | 0.62 |
| text | 64 | 11.5 | 25.1 | 0.46 |
| text | 128 | 21.6 | 47.8 | 0.45 |
| text | 512 | 84.4 | 201.5 | 0.42 |
| text | 2048 | 430.8 | 1381.1 | 0.31 |
| batch | 8x64 | 70.1 | 153.7 | 0.46 |
| batch | 32x64 | 290.1 | 624.5 | 0.46 |
| image | 270 tok | 625.1 | 1477.5 | 0.42 |
| audio | 54 tok | 33.7 | 56.9 | 0.59 |
| video | 8 frames | 2099.9 | 5199.9 | 0.40 |

OpenVINO is much slower on this host than on the Ryzen 9700X of rounds 1-5
(201 vs 64 ms for text 512): its int8 path is weight-only compression, so it
runs f32 matmuls, and Lion Cove's f32 throughput per core is lower than Zen 5's.
e2 keeps its int8 GEMM, so the ratio improves everywhere and every row is
already past the 1.5x goal before round 6 starts. The two thin rows were
**text 8 (0.62)** and **audio (0.59)**.

### What worked this round (CPU)

| # | change | effect |
|---|---|---|
| 1 | **AVX2+FMA twins of the f32 attention kernels** (`qk_kt_a2`, `attn_pv_a2` in `e2_ops.c`). `qk_kt_v`/`attn_pv_v` were AVX-512-only and the dispatch was `simd() >= 2`, so on any host without AVX-512 the *scalar* kernels ran at ~8 GMAC/s. Text shorter than 32 tokens takes the f32 attention path, which is why text 8 was the worst row. | **text 8: 4.37 -> 2.7 ms (1.6x)**, ratio 0.62 -> 0.38. Nothing else moves (t >= 32 uses the int8 path). |

### Final CPU numbers (`--repeat 2 --iters 3`, 8 threads, quiet box)

| task | length | e2 ms | ov-int8 ms | e2/ov | goal |
|---|---:|---:|---:|---:|---|
| text | 8 | 2.6 | 6.5 | **0.40** | ok |
| text | 64 | 11.2 | 21.5 | **0.52** | ok |
| text | 128 | 21.3 | 43.4 | **0.49** | ok |
| text | 512 | 83.2 | 207.0 | **0.40** | ok |
| text | 2048 | 420.1 | 1374.2 | **0.31** | ok |
| batch | 8x64 | 68.8 | 152.2 | **0.45** | ok |
| batch | 32x64 | 282.8 | 604.6 | **0.47** | ok |
| image | 270 tok | 599.5 | 1465.1 | **0.41** | ok |
| audio | 54 tok | 32.0 | 57.0 | **0.56** | ok |
| video | 8 frames | 2681.1 | 5020.6 | **0.53** | ok |

Every row is at least 1.8x faster than OpenVINO int8 (the goal was 1.5x, i.e.
ratio <= 0.667). The worst row is audio at 0.56; the best is text 2048 at 0.31.
Note that `video 8` reads 2681 ms here but 2015 ms in the single-process spec
run -- the harness also runs OV in the same session, so treat it as noise.

### What did not work this round (CPU)

| try | outcome |
|---|---|
| AVX-VNNI GEMM tile 2 rows x 2 output tiles instead of 4 rows x 1 tile (same 8 accumulators, half the `vpbroadcastd` uops) | 4-8% **slower** on every shape (m=8..2048, k=512..2048): the broadcast uops are not the bottleneck, and the extra weight loads double the L1 traffic. Reverted. |
| cutting the pool spin (`E2_POOL_SPIN` 20000 -> 2000 -> 200 -> 0) | 2000: audio 31 -> 90 ms, text 8 4.4 -> 7.0; 0: text 8 -> 25 ms. The nanosleep wake-up latency dominates the ~40 dispatches per layer. The 20000 spin stays; `worker` shows as ~35-45% of *cycles* in `perf` but those cycles are on otherwise idle cores. |

### Where the time goes on this host (text 512, 8 threads)

`attn` 24 ms, `mlp_gate` 21, `o_proj` 16, `q` 11, `mlp_down` 10, `ple_gate` 5,
rest ~10. The int8 GEMM runs at 235 GMAC/s single-threaded (peak `vpdpbusd`
with no loads: 436 GMAC/s, i.e. ~3/cycle) and 1.4-1.6 TMAC/s at 8 threads, so
the kernel is at ~55% of the no-load ceiling; the tile-shape experiment above
says the remaining gap is not a scheduling artefact that a different tile
fixes. Attention at t=512 is 0.8 ms/layer: qk 121 us, pv 118 us, softmax 46 us,
K/V packing 130 us -- all already on the int8 GEMM.

### Audio (54 tokens, 22 ms of tower work)

12 conformer layers: ff1 6.4 ms, ff2 6.5, attn 4.7, lconv 3.3, qkv 2.8. The
GEMMs are small (m=50) and the tower is bound by weight streaming per layer,
not by the kernel (the same shapes measure 1.3 TMAC/s in `mb cold`). The
conformer attention is f32 by design (logit soft-cap + relative position keys)
and its 4.7 ms is 21% of the tower; the fix would be routing it through the
int8 attention core, which is not worth the risk at a 0.59 ratio.

### GPU (Level Zero) — Intel Arc 140T

Goal: run the packed-int8 GEMM on the iGPU through Level Zero. Shipped as an
**opt-in device** (`E2_DEVICE=gpu`, or `e2_set_device(ctx, E2_DEVICE_GPU)`);
CPU stays the default and the reference. No build dependency: `libze_loader.so.1`
is dlopen'd and the SPIR-V module is a checked-in generated header
(`src/e2_gpu_spv.h`, rebuilt with `make spv` from `tools/gpu/*.cl`).

Files: `src/e2_gpu.c` (device/queue/weight cache/GEMM dispatch),
`src/e2_gpu_l0.{c,h}` (dlopen loader, ~35 entry points, no `<level_zero/ze_api.h>`),
`src/e2_gpu_spv.h` (generated), `tools/gpu/e2_gemm.cl` (kernels),
`tools/gpu_probe.c` (`make gpu-probe`: kernel sweep, bandwidth, submission
latency, correctness vs the CPU kernel).

Correctness: GPU vs CPU embedding cosine 1.000000 (text, 32 tokens),
0.999913 (image), 0.998829 (audio); the packed-int8 GEMM itself is exact
(`maxerr` 1.4e-06 vs the CPU kernel, i.e. f32 epilogue rounding only).
`make test` passes with and without `E2_DEVICE=gpu`.

| probe shape | GPU 4x4 kernel | CPU 1 thread |
|---|---:|---:|
| m=512 k=512 n=2048 | 1.35 TMAC/s | 0.24 |
| m=2048 k=2048 n=512 | 1.41 | 0.23 |
| m=1024 k=768 n=768 | 1.39 | 0.24 |
| m=266 k=768 n=768 | 1.29 | 0.24 |
| streaming read | 80 GB/s | |

Device: 128 EUs, 1024 threads, 2.35 GHz, 30.6 GB shared DDR.

**The GPU path is slower than the CPU and does not ship enabled.** Measured
e2@GPU vs e2@CPU (same run, 8 threads): text 64 71 vs 11 ms, text 512 177 vs 84,
text 2048 653 vs 424, image 901 vs 590. The reasons are measured, not guessed:

1. **Per-submission cost is 0.19 ms, and the wait is another ~0.4 ms.**
   `gpu-probe latency` submits the same kernel 1/2/4/8 times per submission:
   0.60 / 0.49 / 0.45 / 0.41 ms per launch. So the kernel is ~0.41 ms and each
   *submission* adds ~0.19 ms. In situ (`E2_GPU_PROF=1`) the split per call is
   upload 0.09 + submit 0.06 + sync 0.82 + copy 0.24 ms. The text tower makes
   ~192 GEMM calls per forward, so dispatch alone costs 150+ ms.
2. **The kernel is issue-bound at ~1 instruction/lane/cycle** (1.35 TMAC/s =
   0.58 MAC/lane/cycle). It does 4 `imad` per 4-byte dot product. With `idot4`
   it would be 4x faster, but `dot()` from `cl_khr_integer_dot_product` is
   unusable on this stack: clang 23 lowers `dot()` to SPIR-V `OpDot` with
   integer operands and IGC fails to build the module
   (`undefined reference to _Z11__spirv_DotDv4_cS_`), and the signed/unsigned
   variants that do compile compute unsigned x unsigned. A pure-ALU probe
   kernel with `dot()` cannot even be linked, so the ceiling could not be
   measured through OpenCL C.
3. OpenVINO *does* reach 7.7 ms for text 128 on the same iGPU (6x its own CPU
   number) because it compiles the whole graph and submits once per inference.
   That is the gap: e2's CPU-driven "one submission per GEMM" cannot compete,
   no matter how good the kernel is.

What would make the GPU worth enabling (in order of value):
* run the **whole tower** on the device (one submission per layer, or one per
  forward) so dispatch is amortized, and keep activations resident in USM
  (today every GEMM uploads the activation and copies the output back);
* get `idot4` into the kernel (emit `OpSDot`/`OpUDot` directly, or a newer
  IGC that links `__spirv_Dot`) for a 3-4x kernel speedup;
* only then consider tuning tile shapes: `gemm_4x4` (4 rows x 4 cols per
  work-item) beat `gemm_1x4`, `gemm_2x8`, `gemm_sg` (sub-group block read) and
  `gemm_sg4` on every shape by 1.6-3x, so the tile is not the bottleneck.

Bugs found while wiring it up (both would have shipped silently):
* `wc_grow()` rehashed into the *old* table because `wc_slot()` closed over
  `g->wct.tab`; heap corruption as soon as the 192nd matrix was uploaded.
* The single-entry weight cache re-copied ~90 MB per forward; it is now an
  open-addressing table keyed by the packed matrix pointer.

### GPU benchmark (Level Zero vs OpenVINO GPU)

`python tools/bench_compare.py --sources e2,ov-int8-gpu --e2-device gpu`:

| task | length | e2@GPU ms | ov-int8@GPU ms | ratio |
|---|---:|---:|---:|---:|
| text | 8 | 4.2 (CPU path, m<32) | 5.6 | 0.75 |
| text | 64 | 71.3 | 6.4 | 11.1 |
| text | 128 | 79.0 | 7.7 | 10.3 |
| text | 512 | 176.0 | 18.4 | 9.6 |
| text | 2048 | 653.4 | 128.5 | 5.1 |
| batch | 8x64 | 154.9 | 15.8 | 9.8 |
| batch | 32x64 | 492.5 | 67.9 | 7.3 |
| image | 270 tok | 900.9 | 254.5 | 3.5 |
| audio | 54 tok | 134.9 | 22.5 | 6.0 |
| video | 8 frames | 3174.1 | 788.5 | 4.0 |

For reference, the same rows on the CPU in the same session: e2 2.7 / 11.3 /
21 / 84 / 424 / 70 / 290 / 590 / 32 / 2015 ms and ov-int8 7.1 / 25 / 48 / 201 /
1381 / 154 / 625 / 1478 / 57 / 5200 ms. So the iGPU is genuinely ~6x the CPU
for OpenVINO, and e2 leaves that on the table because of the pipeline shape,
not because of the hardware.

---

# Round 7 — Intel iGPU path: the matrix engine (DPAS) and why the pipeline shape binds

Box: Intel Core Ultra 9 285H, Arc 140T iGPU (`deviceId 0x7d51`, 128 EUs,
1 slice / 8 subslices / 16 EUs per subslice, 2.35 GHz, 30.6 GB shared DDR).
Host is otherwise idle; every number below is from this machine unless stated.

## Fresh OpenVINO GPU baseline (same run, this box)

`python tools/bench_compare.py --model ~/Projects/embeddinggemma-2
--sources ov-int8-gpu --repeat 2 --iters 3`

| task | length | ov-int8@GPU ms | 1.5x target |
|---|---:|---:|---:|
| text | 8 | 5.6 | 3.7 |
| text | 64 | 6.4 | 4.3 |
| text | 128 | 7.8 | 5.2 |
| text | 512 | 18.6 | 12.4 |
| text | 2048 | 132.2 | 88 |
| batch | 8x64 | 15.8 | 10.5 |
| batch | 32x64 | 68.1 | 45.4 |
| image | 270 tok | 258.0 | 172 |
| audio | 54 tok | 23.0 | 15.3 |
| video | 8 frames | 792.5 | 528 |

## Hardware ceilings measured this round (all on the iGPU)

| probe | rate | how |
|---|---:|---|
| `vpdpbusd`-class ALU (int8 mul-add, no loads) | **13 TMAC/s** | 16 accumulators, operands in registers |
| matrix engine `dpas.8x8` int8 (K=32, 2048 MAC/instruction) | **15.8 TMAC/s** | 4 independent accumulators, tiny register footprint, ≥32 sub-groups per EU |
| same, with one dependent chain per sub-group | 2.0 TMAC/s | DPAS latency ~150 cycles, so 1 chain per thread is 8x off peak |
| wide-load streaming from a 16 MB working set | **966 GB/s** | `vload8` (32 B/lane), 32768 work items |
| the same from DRAM (64 MB) | 81 GB/s | |
| cached 4-byte/lane loads, sub-group scattered | ~420 GB/s | what the naive GEMM gets |

Consequences: the machine can do ~16 TMAC/s of int8 matmul, ~1 TB/s from L2/L3,
so a GEMM that streams `b` bytes per MAC is capped at `1 TB/s / b`.  For text
512 (77 GMAC) the 1.5x target (12.4 ms) needs **6.3 TMAC/s effective including
everything else**, i.e. ~40% of the DPAS peak — reachable, but only if the
kernel gets its bytes/MAC well below 0.15.

## Driving DPAS from e2 (the hard part)

Three routes were tried; only one works on this stack.

| route | result |
|---|---|
| `cl_khr_integer_dot_product` `dot()` via clang → `OpDot`/`__spirv_Dot` | **does not work.** Mixed signedness silently lowers to *unsigned x unsigned*; the signed form emits `__spirv_Dot`, which IGC 2.41 cannot link (`undefined reference to _Z11__spirv_DotDv4_cS_`). IGC has no `OpSDot`/`OpUDot` strings at all. |
| hand-written SPIR-V `OpSubgroupMatrixMultiplyAccumulateINTEL` | **works** after dropping the `OpExtension` line: with it, the module is rejected (“extension … disabled by --spirv-ext option”) and module build flags are not forwarded by this driver; without it IGC accepts the capability and validates the operands (A element type must be int32 for K=32 on this device, operand mask must include `MatrixAPackedInt8INTEL|MatrixBPackedInt8INTEL`). |
| OpenCL C via **ocloc** (IGC's own frontend) producing SPIR-V that Level Zero then loads | **works, and is what shipped.** The driver advertises `cl_intel_subgroup_matrix_multiply_accumulate`, ocloc emits SPIR-V with a `SubgroupSize 8` execution mode, and Level Zero's IGC build resolves the builtin. `make spv` now uses ocloc for `tools/gpu/e2_dpas.cl` and clang for the rest. |

The operand layout was reverse-engineered from the hardware with a probe
(`OpSubgroupMatrixMultiplyAccumulateINTEL` one-hot scans) and matches
`intel_sub_group_u8_i8_matrix_mad_k32` on a subgroup-8 device:

* `M = 8` rows, `N = 8` = subgroup size, `K = 32` per instruction (2048 MACs).
* A: lane `l` holds 8 u32, component `i` = row `i`, bytes `k = 4l..4l+3`.
* B: lane `l` holds 8 u32, component `i` = k-quad `i`, i.e. the lane's *column*
  `l` of the weight tile.
* C/result: lane `l` holds column `l`, components = 8 rows.

That is a plain "VNNI, 4 k per 32-bit word" packing for both operands, so both
the weight repack and the activation pack are a single 4-byte-shuffle pass (no
2-D transpose, no scratch).

## What was built

`tools/gpu/e2_dpas.cl` (compiled by ocloc, checked in as `src/e2_gpu_spv.h`):

| kernel | job |
|---|---|
| `k_pack_wq` | model weight tiles `[n/16][k/4][16][4]` → matrix-engine layout + per-column `wsum` (`atomic_add`) |
| `k_pack_a` | f32 rows → u8 with one scale per row + matrix-engine layout |
| `k_pack_u8` | already-quantised row-major u8 → matrix-engine layout |
| `k_gemm_dpas` | `y[m][n] = (Σ a·b − 128·Σw) · xscale[row] · wscale[col]`, 4 row blocks × 2 column tiles per sub-group (8 tiles = 64 int accumulators) |

`src/e2_gpu.c` gained `e2_gpu_linear_multi()`: quantise the activation row block
once (per-row scale, CPU, pooled), pack it on the device, then run every linear
of the list.  Weight packing is done on the CPU straight into shared USM on
first use (one pass over the matrix, no upload, no extra kernel).

## GEMM results (m=512, k=512, n=2048 unless noted; 20 launches per submission)

| variant | registers/spill | GMAC/s |
|---|---|---:|
| previous plain kernel (`gemm_4x4`, imad byte MACs) | — | 1 050 |
| DPAS, unpacked operands (8+8 narrow loads per DPAS) | 127 | 1 060 |
| DPAS, packed operands, R1×C1 (2 acc/tile) | 127 | 2 295 |
| DPAS, packed, R2×C2 | 127, spills | 3 477 |
| **DPAS, packed, R4×C2 (shipped)** | 127 | **4 640** (5 373 at k=768) |
| DPAS, packed, R2×C4 | 127 | 3 803 |
| DPAS, packed, R4×C4 / R8×C2 / R2×C8 | 127, heavy spill | 240-926 |

Split of the shipped kernel: loads alone 4 560 GMAC/s, DPAS alone 6 540 GMAC/s,
both 4 640 — they now overlap, and the kernel is limited by *both* the L2
traffic (0.094 bytes/MAC, i.e. 435 GB/s at 4.6 TMAC/s) and by occupancy
(127 GRF/thread ⇒ ~1 thread per EU, so the DPAS latency is only hidden by the
8 in-flight accumulators).

What did not work:

| try | outcome |
|---|---|
| `-cl-64/128/256-GRF-per-thread` to raise occupancy | ocloc rejects the option; the Level Zero `pBuildFlags` path does not forward build flags to IGC at all (proved with `-cl-std=CL9.9`, silently accepted) |
| trusting `tools/gpu_probe`-style timings with one submission per launch | every “GMAC/s” number before this round was ~4x too low: one submit+host-sync costs **0.38 ms** on this driver. All measurements above submit 20 launches and divide. |
| bigger tiles (R4×C4, R8×C2, R2×C8) | 128 accumulator registers + operands = spills; 3-20x slower |
| more independent DPAS chains at constant tile | saturated: 2 chains 5.3, 4 chains 5.6, 8 chains 6.5 TMAC/s (DPAS-only) |

## End-to-end: the pipeline shape, not the kernel, is the binding constraint

`E2_GPU_DPAS=1 E2_DEVICE=gpu`, text ≈560 tokens, 218 GEMM calls:

| stage | ms | per call |
|---|---:|---:|
| CPU per-row quantise + pack launch | 34.7 | 0.16 |
| submit | 15.7 | 0.07 |
| **host sync (includes the kernel)** | **82.6** | **0.38** |
| copy y back to the CPU | 10.1 | 0.05 |
| rest (CPU elementwise, attention, plain-path GEMMs) | ~100 | |

The 0.38 ms is a *dispatch+round-trip* latency: it does not shrink when the
kernel is faster (polling `zeEventQueryStatus` instead of the blocking wait
changed 90 → 83 ms), and the old plain kernel pays it too.  That is why the
previous ledger's `e2@GPU` numbers were 4-10x off OpenVINO: OpenVINO compiles
the whole graph and submits **once per inference**, e2 submits once per GEMM.

Same box, same run, ~560 tokens of text:

| build | ms |
|---|---:|
| CPU (8 threads) | 47.7 |
| GPU, plain kernel (previous) | 280 |
| GPU, matrix engine (`E2_GPU_DPAS=1`) | 247 |

So the kernel is 4.4x faster and the forward is 13% faster: 85% of the time is
dispatch and copies.  `E2_GPU_DPAS` is therefore **opt-in** and off by default;
the plain path stays the shippable GPU behaviour.

## What the numbers say the plan must be

1. **Device-resident tower.** One submission per forward (or per layer), all
   activations in shared USM, no per-GEMM sync, no per-GEMM copy.  The tower is
   ~10 kernels (embed gather, rmsnorm+quantise, rope, softmax, gelu/mul,
   residual-add, plus the DPAS GEMM used for attention too) — the projection:
   77 GMAC at 4.6 TMAC/s ≈ 17 ms + ~3 ms of elementwise + one 0.4 ms submit
   ≈ **20 ms for text 512**, i.e. parity with OpenVINO (18.6).
2. **Then the kernel has to get to ~10 TMAC/s** to reach the 1.5x goal on the
   text rows: 0.094 → 0.045 bytes/MAC needs a 32×32 output tile per work-group
   with both operands staged in SLM (64 KB/subslice, 4 KB per k-chunk
   double-buffered) and the accumulators in registers.  That is the oneDNN-class
   design; the traffic formula `bytes/MAC = (R+C)/(R·C)` plus the measured
   1 TB/s L2 and 15.8 TMAC/s DPAS peak says ~13 TMAC/s is the ceiling for a
   64×64 work-group tile.
3. The small rows (text 8/64/128, audio) are weight-streaming + launch bound
   (64 MB of text weights = 0.8 ms at 80 GB/s), which the device-resident tower
   fixes for free.

## Accuracy note (matrix-engine path)

The DPAS kernel accumulates the whole `k` in int32, which requires **one
activation scale per row** instead of the CPU path's one per 128-wide block
(the block-scaled version needs an f32 accumulator per tile, 32 more registers
and ~30% of the kernel speed).  Measured against the CPU int8 path on a ~560
token text the final embedding cosine is **0.9989**, i.e. roughly 0.9977 against
f32 instead of the 0.999 threshold in AGENTS.md — so this path needs either
per-128-block scales (slower kernel) or per-64/128 *sub-block* scales folded
into the epilogue before it can be the default.  Everything below the threshold
is still exact: `k_pack_a`/`k_pack_u8`/`k_pack_wq` and `k_gemm_dpas` reproduce
the CPU int8 result bit-for-bit in the unit probe (`maxerr 4.8e-07`, cos 1.0).

Reference numbers for this box (25-token text, `tools/e2_ref.py` f32 dump):
`embedding` 0.999138, `last_hidden_state` 0.998458, `layer23` 0.999713 —
identical for the CPU and GPU builds.

---

# Round 8 — device-resident iGPU pipeline

Goal this round: beat the OpenVINO **int8 GPU** path by **1.5x in every row**
(text 8/64/128/512/2048, batch 8x64, batch 32x64, image, audio, video).

## Baseline (same box, same run, 2026-11 round 8)

`python tools/bench_compare.py --model ~/Projects/embeddinggemma-2
  --sources e2 --e2-device gpu --repeat 2 --iters 3`

| task | length | ov-int8@GPU | e2@GPU (round 7) | 1.5x target |
|---|---:|---:|---:|---:|
| text | 8 | 6.3 | 2015 | 4.2 |
| text | 64 | 7.0 | 290 | 4.7 |
| text | 128 | 9.2 | 70 | 6.1 |
| text | 512 | 20.4 | 424 | 13.6 |
| text | 2048 | 140.4 | 84* | 93.6 |
| batch | 8x64 | 17.4 | — | 11.6 |
| batch | 32x64 | 69.9 | — | 46.6 |
| image | 270 tok | 258 | — | 172 |
| audio | 54 tok | 23.0 | — | 15.3 |
| video | 8 frames | 792 | — | 528 |

\* the round-7 ledger's `e2@GPU` ordering for the text rows is not comparable
row by row; what matters is that e2@GPU is 3.5-11x **slower** than OV in round 7.

Where the time goes (round 7, unchanged): **0.38 ms per GEMM dispatch**, 218
GEMM calls per text forward, plus a CPU round-trip per call. The kernel is not
the binding constraint at small `m`; the pipeline shape is.

## Measured ceilings on this box (Arc 140T, 128 EU, 2.35 GHz, 30.6 GB DDR)

| probe | rate |
|---|---:|
| DPAS int8 peak (dense, many sub-groups/EU) | 15.8 TMAC/s |
| plain-kernel GEMM reached in round 7 | 4.64 TMAC/s |
| L2/L3 read bandwidth | ~1 TB/s |
| DRAM read bandwidth | ~81 GB/s |

Consequences for the targets:

* text 512 = 77 GMAC. 13.6 ms needs **5.7 TMAC/s end-to-end** (36% of DPAS
  peak). Round 7's kernel did 4.64, so the kernel has to improve *and* the
  dispatch has to go away.
* text 8/64/128 are weight-streaming bound: the text tower minus the embedding
  table is ~76 MB of int8 weights = **0.94 ms at 81 GB/s**. The token embedding
  is a *gather* (a few rows), not a GEMM, so it costs nothing. 4.2 ms is
  therefore reachable if the whole forward costs < 1 ms of dispatch.
* image/audio: OV reaches only 0.29 / 0.57 TMAC/s there, so a competent
  device-resident tower wins these rows by a wide margin.

## Plan

1. **Device-resident tower.** All activations in shared USM, one command list
   per forward, one sync at the end. The CPU-side op layer (`e2_ops.c`) gets a
   device branch per op that *enqueues* a kernel instead of computing, so the
   towers (text/vision/audio) keep a single code path and the CPU path stays
   the reference.
2. **Fused epilogues.** rmsnorm+quantize+pack, rmsnorm_add+quantize+pack and
   gelu*mul+quantize+pack each collapse two or three passes into one, which is
   also what removes the launch overhead from the small-`m` rows.
3. **Faster DPAS GEMM** (agent-assisted, measured in isolation).
4. Vision + audio towers on the same device op layer.

## Experiment log

Every attempt, kept or reverted, with the number that decided it.

| # | try | result | kept? |
|---|---|---|---|
|---|---|---|
| 1 | keep one submission per GEMM, just make the kernel faster | round 7: 247 ms vs 47.7 ms on the CPU; 85% of the time is dispatch+sync | no |
| 2 | **device-resident tower**: one command list per forward, ops *enqueue* | 531-968 kernels in **one** submission; per-kernel device time finally measurable | **kept** (the whole round rests on it) |
| 3 | `zeCommandListAppendBarrier` after every kernel | 749 ms vs 724 ms at 512 tokens: barriers are ~3%, not the problem I expected. Kept (correct by construction) | kept, cheap |
| 4 | attention with per-lane register arrays (`qreg[16]`, `acc[16]`) | IGC spills **4096 B/thread** to scratch (`private_size` in the zeInfo); 27x slower than local memory | no |
| 5 | attention: accumulators in `__local` (8 KB/WG), q re-read from L1 | `private_size: 0`; attention 750 -> 344 ms at 512 tokens | **kept** |
| 6 | force `intel_reqd_sub_group_size(16)` on attention | head_dim 512 (the 4 full-attention layers) needs 32 regs/lane and spills again | no |
| 7 | **band split**: 2 kernels, `k_attn_part` + `k_attn_comb` | only kicks in when `rows/4 < 512`; at 512 tokens there are already 512 WGs, so the split was 1 and nothing changed. Correctness exact (cos 1.0000000) | kept (needed at 64-128 tokens) |
| 8 | one packed-activation buffer reused by every GEMM | **wrong results**: the queue still holds kernels that read it. This was the "nondeterministic output" bug, not the barriers | no |
| 9 | per-forward bump arena for packed activations | exact again; ~230 MB at 2048 tokens (see shortcuts) | **kept** |
| 10 | per-output-row activation scale (round 7's fast DPAS kernel) | 0.9989 vs the CPU path, i.e. 0.9986 vs f32 -- below the 0.999 bar | no |
| 11 | `k_gemm_dpas_blk`: per-128-block scales, RB=2 x CT=2, int+float8 accumulators | **cos 1.0000000, maxerr 9.5e-07** vs `e2_gemm_range`. 32+32 registers, no spill | **kept** (accuracy path, now the default) |
| 12 | `E2_GPU_NOBARRIER`, `E2_GPU_TS`, `E2_GPU_SKIP`, `E2_GPU_PROF` knobs | they turned three guesses into measurements; `E2_GPU_TS` writes a device timestamp before every kernel | **kept** (debug only) |
| 13 | audio tower on the device | its convolutions and attention are still inline CPU code, so deferred buffers were read half-written (cos 0.00). Now explicitly CPU-only (`e2_gpu_no_dev`) | reverted |
| 14 | replace `sub_group_reduce_add` with 5 explicit `sub_group_shuffle_xor` steps in attention | 14.69 s vs 14.74 s at 2048 tokens: the collective is *not* what is slow (it does make IGC retry the compilation, so the kernel still compiles both ways) | no |
| 15 | unroll the attention key loop 2x (both scores issued before either is reduced, hoisted q load) | 14.69 -> 11.75 s at 2048 tokens (1.25x), exact (cos 1.0000000 on every layer); the win is memory-level parallelism, as predicted | **kept** |
| 16 | vision tower on the device | works, exact per tensor (vis_in cos 1.0000000) but the known chaotic amplification of the vision tower turns the f32-vs-int8 attention difference into 0.988 by layer 16 | kept, accuracy documented |

## Where the device time actually goes

`E2_GPU_TS=1` timestamps every kernel. text 512 tokens, one submission of 968 kernels,
per-kernel time summed over one forward:

| kernel | calls | total ms | us each |
|---|---:|---:|---:|
| `attn_part` (24 layers) | 24 | **254.6** | 10 607 |
| `gemm_blk` | 220 | 43.5 | 198 |
| `quant_u8` | 197 | 11.5 | 58 |
| `rmsnorm` | 149 | 11.1 | 75 |
| `gelu_mul_q` | 25 | 9.5 | 379 |
| `pack_u8` | 222 | 6.6 | 30 |
| `norm_rope` | 48 | 6.5 | 136 |
| `mul_slice` | 25 | 1.4 | 57 |
| `scale`, `gelu_tanh`, `pos_add`, `attn_comb` | 76 | 0.9 | 12 |

Three conclusions, all measured rather than guessed:

1. **Attention is 73% of the forward** and it is the only O(n^2) kernel, and at
   2048 tokens it is bound by *memory-level parallelism*, not instructions: each
   (row, head) re-reads its whole band of K and V plus the head_dim floats of q,
   so one 2048-token layer touches ~6 GB and the serial key loop (load -> dot ->
   cross-lane reduce -> exp -> local-memory update) leaves only ~2 work-groups
   per EU to overlap it. Measured effective bandwidth is ~10 GB/s of an 81 GB/s
   part. Unrolling the key loop and issuing its loads up front is the cheap 2-4x
   here; a local-memory K/V block per query block is the real fix. A per-lane
   f32 flash loop costs ~60 instructions per (row, head, key) for 512 MAC, i.e.
   0.2 MAC/instruction where a DPAS instruction does 2048. No amount of tiling in
   f32 closes that: this is what the tensor cores are for, and it is exactly why
   the CPU path runs attention through the int8 GEMM (`e2_attn_int8`).
2. **The elementwise passes are the next block** (39 ms of 347 ms): `quant_u8` and
   `pack_u8` are two extra passes over every activation, and `rmsnorm`/`norm_rope`
   are separate from the quantise that follows them. Fusing norm+quantise+pack
   into one kernel removes ~400 of the 968 launches *and* two thirds of that
   traffic.
3. **The GEMM is 5x off its own isolated measurement** (1.28 TMAC/s end-to-end vs
   4.6 TMAC/s standalone) because RB=2 x CT=2 blk kernel is the accuracy variant,
   and because small-m launches do not have enough work-groups to fill the device.

## Status of the goal

Same run, all rows (`--sources e2 --repeat 2 --iters 3`, both device settings,
host shared with the desktop so the CPU column is ~1.4x its idle-box number):

| task | length | e2 CPU | e2 GPU | ov-int8@GPU | 1.5x target |
|---|---:|---:|---:|---:|---:|
| text | 8 | 7.4 | 18.6 | 6.3 | 4.2 |
| text | 64 | 20.3 | 35.3 | 7.0 | 4.7 |
| text | 128 | 33.0 | 61.5 | 9.2 | 6.1 |
| text | 512 | 118.2 | 534.6 | 20.4 | 13.6 |
| text | 2048 | 1082 | 11750 | 140.4 | 93.6 |
| batch | 8x64 | 131.7 | 201.5 | 17.4 | 11.6 |
| batch | 32x64 | 555.9 | 896.3 | 69.9 | 46.6 |
| image | 270 tok | 640.8 | 2998.1 | 258 | 172 |
| audio | 54 tok | 32.7 | 69.3 | 23.0 | 15.3 |
| video | 8 frames | 2202.8 | 7635.2 | 792 | 528 |

(text 2048 is the worst row and it is a *traffic* problem, not an instruction
problem: every (row, head) walks its own 1025-key band and re-reads K and V, so
one layer touches 8192 row-heads x 1025 keys x 2 KB = 16.8 GB, which no cache
absorbs. The 4-row block sharing in the kernel cuts that 4x; the fix is to stage
a K/V block in local memory and walk it once for a whole query block, or to run
attention through the matrix engine (which re-reads K/V once per tile).)

The goal for this round was 1.5x over OpenVINO int8 on the GPU in every row. That
is **not met**; the device path is correct but slower than the CPU path, so it
stays opt-in (`E2_DEVICE=gpu`) and the CPU path stays the default. What the round
did deliver is the thing round 7 was missing: a device-resident pipeline whose
per-kernel cost can be measured, plus a measured list of what the remaining gap
actually consists of. See "Round 8 roadmap" below for the order to attack it.

```bash
# correctness + timing of the device path
E2_DEVICE=gpu ./tests/e2_test models        # passes
E2_DEVICE=gpu python tools/e2_acc.py text   # 0.9998 vs f32 (CPU path: 0.9997)
make gpu-check                              # device GEMM vs CPU kernel: cos 1.0000000
E2_GPU_TS=1 E2_DEVICE=gpu ./tests/e2_bench_full spec.txt   # per-kernel device times
```

## Round 8 roadmap (measured, in the order that pays)

1. **int8 attention on the matrix engine.** The CPU implementation
   (`e2_attn_int8`) is already the right algorithm: pack K and V^T, QK^T through
   the packed-int8 GEMM, softmax, PV through the same GEMM. 15 GMAC at 4 TMAC/s
   is 3.7 ms for text 512 (today: 254 ms of f32 attention), and it also removes
   the vision tower's accuracy drift, because it makes the two paths agree bit
   for bit. This is the single change that matters.
2. **Fuse norm+quantise+pack, and gelu*mul+quantise+pack.** Each fused kernel
   removes 2 launches and 2 passes over the activation. Cuts ~40% of the
   non-attention time and ~400 of 968 launches.
3. **Small-m GEMM: split K.** At m <= 32 the DPAS kernel has n/16 work-groups
   (64 for n=1024) on 128 EUs, which is why text 8 streams weights at 16 GB/s
   instead of 81. A 4-8 way split-K with a reduction pass fills the device.
4. **Large-m GEMM: SLM tiling.** RB=4 x CT=2 with both operands staged in
   local memory and double buffered; the traffic formula in the round 7 note
   says 10-13 TMAC/s is reachable, which is what the 13.6 ms text-512 target
   needs (6 TMAC/s end to end).
5. Only then: vision and audio towers fully on the device (per-row, not per-op).

## Shortcuts taken (marked in the code)

- `shortcut: packed activations are bump-allocated per forward` in
  `src/e2_gpu_int.h` -- a GEMM's packed activation cannot be reused until its
  kernel has run, so each gets a fresh slice; ~230 MB at 2048 tokens. Upgrade
  with a small event-checked ring of slots when memory matters.
- `shortcut: the audio tower runs on the CPU even with E2_DEVICE=gpu` -- its
  convolution and attention loops are still inline CPU code. Upgrade with a
  device mel+conv+attention port.

---

# Round 9 — the device GEMM, rewritten

Host for every number below: **Intel Core Ultra 9 285H** (6 P + 8 E + 2 LP-E,
AVX-VNNI, no AVX-512, Arc 140T iGPU, 128 EU, 2.35 GHz, 30.6 GB DDR), Debian 13,
**shared with other work** (load average 2.5-5). The machine moves: the same
binary measured 54.9, 59.6 and 57.9 ms for text 512 in three consecutive runs,
so ratios within one run are the only trustworthy numbers. (The header of this
file describes a Ryzen 9700X; that was round 7's box, not this one.)

Goal of the round, restated: beat **OpenVINO int8 on the iGPU** by 1.5x in every
row. OpenVINO measured here (`--sources ov-int8-gpu`, same run family):

| task | length | ov-int8 GPU | 1.5x target |
|---|---:|---:|---:|
| text | 8 | 6.4 | 4.3 |
| text | 64 | 8.4 | 5.6 |
| text | 128 | 8.9 | 5.9 |
| text | 512 | 21.6 | 14.4 |
| text | 2048 | 144.9 | 96.6 |
| batch | 8x64 | 19.0 | 12.7 |
| batch | 32x64 | 71.3 | 47.5 |
| image | 270 tok | 314.5 | 210 |
| audio | 54 tok | 20.9 | 13.9 |
| video | 8 frames | 920 | 613 |

## What changed

The round replaced the device compute path outright; the CPU path, the public
API and the model format are untouched.

1. **Centred int8 activations, signed x signed DPAS.** The activation is
   quantised to `round(x/scale)` in [-127,127] instead of `+128` in [0,255].
   Same precision (the CPU quantiser is symmetric: `scale = amax/127`), same
   arithmetic, but the matrix engine's `i8_i8` form has **no zero point**, so
   the epilogue loses the `128*sum(w)` correction and its `wsum` table
   entirely. Measured on the kernel in isolation: **4.07 -> 9.4 TMAC/s**
   (m=512 k=512 n=1024, weight-scale multiply hoisted out of the k loop as
   well). This is the single biggest win of the round and it *deleted* code:
   no `wsum`, no `wsumt`, no `k_pack_wq`, no `k_pack_u8`.
2. **The activation is never repacked.** The B operand of an `i8_i8` DPAS is
   a row-major tile (lane = token, 32 contiguous bytes of that row), which is
   exactly what the quantiser already writes. The whole DPAS-pack pipeline for
   activations (`k_pack_a`, `k_pack_u8`, 222 launches and 6.6 ms at text 512 in
   round 8) is gone.
3. **Weight tile = 128 tokens x 64 features per work group** (`E2_XTB/XFB/RSG/
   CSG` in `tools/gpu/e2_xmx.cl`, mirrored in `src/e2_gpu_pack.h`). Eight
   sub-groups share the token tile and four the feature tile, so one work group
   of 256 threads covers 128x64 and the operand traffic per GEMM is `m/128`
   weight passes plus `n/64` activation passes. 16x16 per sub-group with 4x4
   sub-groups (64x64 per work group) was the starting point; a sweep of
   (RSG,CSG) on text 512 gave 64x64 55.5 ms, **128x64 47.6 ms**, 128x128 51.6,
   256x64 58.5, 64x128 55.7, 128x32 56.6.
4. **int8 attention on the matrix engine** (`e2_gpu_enq_attn`): quantise Q,
   pack K and V^T into the A operand layout with a per-row scale
   (`k_attn_pack`, one kernel for both by taking strides), QK^T through the same
   GEMM kernel, a masked softmax that writes int8 P with the row's 1/sum folded
   into the per-128-key-block scales (`k_attn_softmax`), then PV through the
   same GEMM written straight into the `[token][head][dim]` output.
   **Text 512: 304 ms of f32 attention -> ~9 ms.** The score matrix is
   materialised; that is what will have to change at 2048 tokens (see the
   roadmap).
5. The GEMM kernel gained `xstr`, `xscstr`, `ystr`, so attention reuses it
   without any copy or transpose.

Deleted along the way: `tools/gpu/e2_gemm.cl` (the pre-DPAS kernels),
`tools/gpu/e2_dpas.cl` (replaced by `e2_xmx.cl`), `tools/gpu_probe.c` (replaced
by `tools/gpu_bench.c`, which checks results against the CPU kernel and can
sweep the tile without rebuilding the engine), `k_attn_part`/`k_attn_comb`/
`k_softmax` (f32 attention), `k_pack_a`/`k_pack_u8`/`k_pack_wq`.

## Experiment log (round 9)

| # | change | effect |
|---|---|---|
| 1 | 32-byte vector loads for both DPAS operands, `[fb][kstep][lane][8 words]` weight layout | the layout trap of round 8: an `int8*` base scales offsets by 32, so every address was wrong. Found by the CPU-vs-device check, not by reading |
| 2 | **centred int8 + `i8_i8` DPAS** (drop the zero-point term and `wsum`) | 4.07 -> 5.5-7.9 TMAC/s immediately, and 0.9999 -> 1.0000000 cosine (the CPU path is symmetric anyway) |
| 3 | hoist the per-column weight scale out of the k-block loop (apply once at the store) | part of the 2x above |
| 4 | transposed activation layout `[kstep][token][32]` so a sub-group's 8 lane loads are one contiguous 256-byte run | **no gain** (6.5 vs 5.5-7.9, inside the noise). The gather is not the limit; do not add the transpose pass |
| 5 | 256 GRF per thread (`-cl-intel-256-GRF-per-thread` at module build) | **no gain**. The kernel is not register-starved |
| 6 | `(RSG,CSG)` sweep incl. 256x64 and 128x128 work-group tiles | 128x64 wins (see above); 512-thread work groups lose more to occupancy than they gain in reuse |
| 7 | software-pipelined k loop (next k step's operands fetched before the DPAS) | **2x slower** (text 8 13.5 -> 19.3, text 512 55 -> 83). The double operand buffers + the epilogue temporaries spill; IGC's own scheduling was already fine. Reverted |
| 8 | split-K over sub-groups with an SLM reduction (mode B for small m) | **slower than the plain 128-token tile even at m=8** (m=8 k=512 n=2048: 1023 vs 3024 GMAC/s). Small m is weight-bandwidth bound, and wasting 7/8 of the row tile costs nothing; the extra barrier and reduction cost more. Removed |
| 9 | `zeKernelSetGroupSize(..., 8, 32, 1)` set per launch instead of once at init | no difference (it is per-kernel state); the per-launch call was removed |
| 10 | barriers between kernels (`E2_GPU_NOBARRIER`) | barriers cost 9 ms of 65 ms at text 512 (14%). Left on by default, it is a correctness knob |

## Measurements of the kernel in isolation

`make gpu-bench` (dev-only, checks every shape against `e2_gemm_range`):

| shape | GMAC/s | note |
|---|---:|---|
| m=512 k=512 n=2048 | 9000 | weights (2 MB) stay in L3 across the 40 reps |
| m=512 k=512 n=32768 | 2980 | 16 MB of weights: 4 passes = 64 MB weights + 64 MB output in 2.9 ms = **44 GB/s** |
| m=128 k=512 n=32768 | 3242 | same, i.e. 49 GB/s of DRAM |
| m=512 k=2048 n=512 | 9175 | |
| m=2048 k=512 n=8192 | 2241 | 16 passes over the weights |

Matrix-engine issue ceiling on this part, measured by `k_xmx_probe` (the same
kernel with the operand loads removed): **8.3-9.1 G DPAS/s = 17-18.6 TMAC/s**.

So the kernel itself reaches 50% of the engine ceiling on cache-resident shapes
and ~58% of DRAM peak when streaming, which is respectable. What is left is
per-launch cost: a forward is ~1500 kernels appended to one command list, and
the sweep above says the *steady-state* rate is not what a single cold launch
achieves. Text 512 runs 290 GEMMs of ~0.25 GMAC each (15-20 us of engine time)
and pays ~40 us of ramp/drain per launch.

## Where the device time goes now (text 512, `E2_GPU_TS=1`)

Per forward, from the per-kernel device timestamps (they inflate every entry by
a constant because the timestamp write itself costs time, so read the shares,
not the absolute values):

| kernel | calls | share |
|---|---:|---:|
| gemm (290 of them GEMMs proper, the rest is the same kernel for attention) | 642 | 52% |
| quant_s8 | 363 | 14% |
| rmsnorm | 243 | 13% |
| attn_softmax | 160 | 6% |
| gelu_mul_q | 40 | 5% |
| norm_rope | 80 | 5% |
| attn_pack_k / attn_pack_v | 146 | 3% |
| mul_slice, scale, gelu_tanh | 120 | 2% |

Round 8's f32 attention was 73% of the forward; it is now 9%. The cost is
dominated by the GEMM and by the *number* of elementwise launches over the
activation stream.

## Status against the goal

**Same run, same process** (`--sources e2,ov-int8-gpu --e2-device gpu --iters 2
--repeat 2`), the only comparison this host supports (load average 2.5-5, ±10%
run to run; an earlier identical run of the same binary gave 17.0/83.5 where this
one gives 11.6/68.8 for text 8/512, so read ratios, not absolutes):

| task | length | round 8 e2 | **round 9 e2** | ov-int8 GPU | e2/OV | 1.5x target |
|---|---:|---:|---:|---:|---:|---:|
| text | 8 | 18.6 | **11.6** | 6.3 | 0.54 | 4.2 |
| text | 64 | 35.3 | **14.4** | 7.4 | 0.51 | 4.9 |
| text | 128 | 61.5 | **19.6** | 8.8 | 0.45 | 5.9 |
| text | 512 | 534.6 | **68.8** | 25.3 | 0.37 | 16.9 |
| text | 2048 | 11750 | **382** | 138.3 | 0.36 | 92 |
| batch | 8x64 | 201.5 | **52.6** | 17.7 | 0.34 | 11.8 |
| batch | 32x64 | 896.3 | **225** | 71.2 | 0.32 | 47.5 |
| image | 270 tok | 2998 | **540** | 290.6 | 0.54 | 194 |
| audio | 54 tok | 69.3 | **39.7** | 26.8 | 0.67 | 17.9 |
| video | 8 frames | 7635 | **2710** | 849.5 | 0.31 | 566 |

Round 9 is **7.8x faster at text 512, 31x at 2048, 4x on the batches, 5.5x on
image and 2.8x on video**, and no row regressed. **The 1.5x-over-OpenVINO goal
is not met**: the device path is 1.5-3.2x *slower* than OV on every row and
3-92x off the target (it needs 1.5x *faster*, i.e. 2.2-4.8x better than today).
What the round established, all measured:

- the matrix-engine ceiling is 17-18 TMAC/s on this part; the GEMM reaches 9
  TMAC/s on cache-resident shapes and 44-49 GB/s when streaming cold weights.
  The kernel is no longer the unknown it was.
- a *single cold* launch of a small GEMM costs ~2.5x its steady-state time and a
  forward is ~1500 launches, ~640 of them this one GEMM kernel. Ramp/drain, not
  arithmetic, dominates at text <= 128.
- weight traffic is `m/128` passes over the model (480 MB at text 512, 1.9 GB at
  2048), so the row tile is the structural lever for the long rows.
- the device path is now used for batched attention when the score rectangle is
  at most 512 tokens (8x64 uses it, 32x64 does not yet); the audio tower and the
  vision tower's attention (head_dim 64, no 128-wide block layout) are still on
  the CPU and pay a flush per layer.

## Round 9 roadmap (measured, in the order that pays)

1. **One GEMM launch per layer, not nine.** Concatenate q/k/v into one packed
   matrix (n = 2048) and gate/up into one (n = 4096) at load time, and give the
   kernel a `gridCountZ` over a small descriptor array so attention's four heads
   become one launch per product. 642 -> ~150 launches, and each launch has 4x
   the y parallelism, which is exactly what the ramp/drain cost needs.
2. **Fuse norm+quantise** (`k_rmsnorm` writing s8 + block scales directly) and
   **residual-add+norm+temperature**: 606 elementwise launches at text 512
   become ~250, and each fusion removes a full read+write of the activation.
3. **Batched attention** (nseq > 1): the device path refuses it today, so
   8x64/32x64 fall back to the CPU with one flush per layer. The mask already
   takes per-row key bounds; the score product needs a per-64-row key band.
4. **Attention at 2048 tokens**: the materialised score matrix is 268 MB per
   layer for the full-attention layers, so it needs the per-row band (513 keys
   for the 20 sliding layers) or a flash-style SLM kernel (16 query rows, K/V
   staged, online softmax). Compute-wise 2048 tokens is 90 GMAC of attention.
5. Audio tower on the device (mel + conformer + blocked attention).

## Shortcuts taken (round 9)

- `shortcut: the score matrix is materialised in f32` in `e2_gpu_enq_attn` --
  fine up to ~512 tokens (4 MB per head), and 2 passes of DRAM traffic beyond
  that. Upgrade with a per-64-row key band, then with a fused flash kernel.
- `shortcut: batched attention scores the whole t x t rectangle` in
  `e2_gpu_enq_attn` -- correct (the softmax masks by sequence) but only taken
  when t <= 512, because a 32x64 batch would do 32x the needed products. The
  rest still runs the CPU int8 attention with a flush per layer. Upgrade with
  the per-row band from roadmap item 3.
- `shortcut: one activation quantise per linear call` -- `quant_s8` is 14% of
  text 512 and runs even when the producer kernel could have written int8
  directly. Upgrade with roadmap item 2.
- The audio tower still runs entirely on the CPU (`e2_gpu_no_dev`), as in
  round 8.
