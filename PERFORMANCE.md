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
