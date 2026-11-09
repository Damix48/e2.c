# GPU-GAP.md — what is slow in `e2.c-next` and what `~/Projects/e2.c` does instead

Written after a session of measuring both engines on the same host (Intel Core
Ultra 9 285H, Arc 140T iGPU, 128 EU, 2.35 GHz, shared DDR5).  Every number here
is either a same-run interleaved A/B or a reading of the other engine's source;
the two are labelled.  Nothing here is a guess about what "should" be faster.

The purpose of this file is to be the work list: each problem states what we do,
**why it is slow**, and how the older engine solves the same thing, with the
kernel names and the constants so it can be copied rather than reinvented.

---

## 0. Where we stand

Alternating runs of the two engines in one session (3 alternations, min), text
encoder only, both on their own shipped artefacts:

| tokens | `~/Projects/e2.c` | `e2.c-next` | gap |
|---|---:|---:|---:|
| 15 | 4.47 | 9.49 | 2.1x |
| 196 | 11.10 | 23.69 | 2.1x |
| 512 | 26.18 | 55.45 | 2.1x |

A flat 2.1x.  It was 2.5-3.5x before the row-kernel fix described in P1.

Where our 55 ms at 512 tokens goes (per-kernel device timestamps, ~1.5x inflated
by the timestamp write itself — read the shares):

| kernel | launches/forward | share |
|---|---:|---:|
| `gemm` (projections + the two `b2` products + the attention QK/PV) | 233 | 51% |
| `mlp_gu` (fused gate/up/gelu/quantise) | 24 | 15% |
| `attn_softmax` | 44 | 9% |
| `rmsnorm_q` | 145 | 9% |
| `quant_s8` | 49 | 4% |
| `gemm_ple` (fused PLE gate/gelu/slice/quantise) | 24 | 4% |
| `norm_rope` | 48 | 3% |
| `attn_pack_k` / `attn_pack_v` | 48 | 3% |
| `scale`, `rmsnorm` | 25 | 2% |

640 kernels per forward.  A launch costs ~1 us without a barrier and ~5 us with
one, so launches are ~3 ms of the 55 — **launch count is not the problem** and
should not be optimised again.

---

## P1. Row operations — the biggest gap (was 13x, now ~3x)

### What we do

Four separate kernels, 266 launches per forward at 512 tokens:

| kernel | launches | what it does |
|---|---:|---|
| `k_rmsnorm_q` | 145 | RMS-norm a row, write f32 + int8 + per-128 scales |
| `k_quant_s8` | 49 | quantise a row block to int8 + scales |
| `k_norm_rope` | 48 | RMS-norm a row, apply rope, in place |
| `k_scale` | 24 | multiply a tensor by the layer scalar |

### Why it was slow (root cause, measured)

```c
int gs = get_sub_group_size();          /* runtime: 16 on this part */
for (int i = lane; i < dim; i += gs)    /* stride unknown at compile time */
    sum += xp[i] * xp[i];
```

The sub-group stride is a **runtime value**, so the compiler cannot vectorise
this loop.  Each memory instruction moved one float per lane: 16 lanes × 4 B =
**64 bytes per instruction** instead of 256.  Three of these kernels were 30% of
kernel time.

Fixed by giving each lane four consecutive floats and the loop a 4× stride:

```c
const int gs4 = gs * 4;
for (int i = lane * 4; i < dim; i += gs4) {
    float4 v = *(const __global float4 *)(xp + i);
    ...
}
```

Same-run A/B, cumulative over `k_rmsnorm_q` + `k_quant_s8` + `k_norm_rope` +
`k_attn_softmax`: **text 8 −28%, 64 −24%, 128 −23%, 512 −7%, 2048 −3%.**
`rmsnorm_q` fell from 18% of kernel time to 9%.

A second, separate waste: `e2_gpu_enq_mlp` and `e2_gpu_enq_ple` called
`e2_gpu_pack_act` unconditionally, re-quantising the activation that
`k_rmsnorm_q` had *just* quantised — 48 wasted passes per forward.  All consumers
now go through one `act_get()` that uses the one-entry activation cache.

### What the older engine does

**One kernel, ten mode bits.**  `k_rowops` (sub-group size 32) does every row
operation in the graph.  Its header:

> One sub-group per row: lane l holds elements `[b*128 + l*4, +4)` of block b, so
> a block's largest magnitude is a 32-lane butterfly and the quantized bytes of a
> block are one 128-byte contiguous store.

The mode bits (verbatim from the kernel comment):

| bit | meaning |
|---|---|
| 1 | copy the pre-norm source to `res` |
| 2 | quantise the result into `qa`/`asc` |
| 4 | read the source from `src` |
| 8 | add `res` after the norm (residual) |
| 16 | skip the norm and only quantise (the context path) |
| 32 | skip the norm, don't touch the row, **and** write the value as the next residual |
| 128 | clamp the post-norm value (the audio tower's clipped linears) |
| 256 | apply `scale` before the residual, not after (audio FFN) |

`gamma` and `gamma2` are two norm weight vectors in one call, which is what lets
it do **residual add → norm → quantise** in one pass, and `flags & 32` is what
lets it do **o_proj → pre-FFN norm → int8** in one pass.  Those two fusions are
the ones we still do not have.

**`k_norm_rope3` / `k_norm_rope3_i8`** normalise q, **k and v in one launch**,
apply rope, and the `_i8` variant also emits the int8 q and a per-head K with
per-128 block sums (`bsq`, the zero-point correction its DPAS form needs).  So a
whole layer's Q/K/V normalisation + rope + quantisation is **one** launch; we
spend three (`k_norm_rope` × 2 + `k_rmsnorm` for v) plus the attention quantiser.

One hazard they documented, which applies to any rewrite of `k_norm_rope`:

> The rotation pairs element `i` with `i + D/2`.  The reference reads both
> elements of a pair before writing either, and the pairs straddle a lane
> boundary, so the partner is loaded into a register in the same pass — reading
> it back after the store would give the rotated value and a silently wrong
> embedding.

### What to build

One `k_rowops`-shaped kernel with mode bits, covering: residual add, one or two
norm weights, rope (with the per-position table), int8 quantisation, layer scale,
and the clamp the audio tower needs.  Target: **one launch per q/k/v block and
one per residual+norm pair**, i.e. 266 → ~90 launches and no separate
`quant_s8` at all.  Expected ~10-15 ms at 512 tokens if the old engine's 1.6 ms
is reachable.

---

## P2. Attention — 5.5x

### What we do

int8 DPAS with a **materialised f32 score matrix**:

1. `k_quant_s8` — quantise Q (all heads, one launch).
2. `k_attn_pack` — pack K and V^T into the DPAS A layout, one launch each.
3. `k_gemm_z` — scores `[heads][t][rpad]` as f32, one launch for all heads.
4. `k_attn_softmax` — masked softmax, writes int8 `P` with the row's `1/sum`
   folded into the per-128-key-block scales.
5. `k_gemm_z` — context `P·V^T` straight into `[token][head][dim]`.

6 launches per layer, ~18 ms at 512 tokens (attention QK/PV inside `gemm`,
plus `attn_softmax` 9% and the packs 3%).

### Why it is slow

The score matrix is the only large temporary in the forward and it is f32:

| tokens | score matrix per layer |
|---|---:|
| 512 | 4 MB (4 heads) |
| 1200 patches (vision) | 74 MB (12 heads) |

It is written by the score GEMM, read by the softmax, and `P` is written and read
again.  At 1200 patches that is 150 MB of traffic per layer and it explains
essentially all of the image row (240 ms of 431 ms is attention).

Mitigation in place: heads are walked in groups whose score rectangle fits
`E2_GPU_ATT_BUDGET` (8 MB), which recovered 10% on image.  It caps the damage; it
does not remove the matrix.

### What the older engine does

**FP16 on XMX, no f32 score matrix, and the context GEMM owns its quantisation.**
Four kernels:

| kernel | what it does |
|---|---|
| `k_attn_f16_pack` | packs K and V^T once per layer into FP16 DPAS operands, zero-padded to a multiple of 16 keys, so the products need no bounds checks |
| `k_attn_f16` | QK on `intel_sub_group_f16_f16_matrix_mad_k16`, **8 queries × 16 keys per sub-group**, FP32 accumulators, k16 steps |
| `k_attn_f16_softmax` | one sub-group per row, lane-major, `native_exp`; writes **FP16 probabilities** |
| `k_attn_f16_ctx` | PV; **one workgroup owns a 128-d context block**, stages the FP32 context in 4 KiB of SLM and emits int8 activations + per-128 scales directly |

The last one is the important idea: the context product's quantisation needs the
block max over 128 features, and the workgroup *already owns* exactly 128
features, so the max is an SLM reduction over data it has in registers.  No FP32
context round trip, no separate quantiser, 24 fewer dependent launches per
forward.  Their measured effect (Attempt 12): batch 32 ~300 → 286 ms, document
10.63 → 10.47 ms, and the graph shrank 439 → 391 kernels.

Their measured effect of the FP16 move itself (Attempt 2): "earlier baseline
profiling attributed ~55-60% of document GPU time to serial FP32 attention;
moving the reductions/products to XMX, rather than only reducing traffic, removes
that bottleneck."

Rejected there and worth knowing: a **16-query** FP16 tile reusing each packed
K/V operand for two query blocks (Attempt 6) — only 1.2%, reverted.

### Vision (head_dim 64) — they have a separate path, we have none

We refuse the device path for `head_dim % 128 != 0`, so the vision tower's
attention runs on the CPU and forces a flush per layer.  We tried enabling it
with a 64-wide block and measured it **9% slower than the CPU**, so it is gated
off.

They use a different kernel set for this shape, with a different tile and a
*blocked* softmax:

- `k_vattn_packq/packk/packv` — per-row quantisation with a single 64-wide block.
- `k_vattn_scores` — int8 scores with a **32-token × 32-key tile** (`E2_SDM`/
  `E2_SDN` = 32).  Their reasoning, from `e2_tile.h`:
  > the scores product is only two DPAS k-steps deep, so the operand loads, not
  > the matrix engine, set the pace (four 32-byte sends per four DPAS at 16x16).
  > A 32-token by 32-key tile makes that eight sends per sixteen DPAS and halves
  > the weight re-reads on top; the context product's N is the 64-wide head dim,
  > so it takes a 64-token tile (`E2_CDM`/`E2_CDN` = 64×16).
- `k_vattn_psoft` — softmax **in blocks**: per-block max and sum, no full-row
  pass.
- `k_vattn_pasc` — turns the per-block `(max, sum)` into the context's per-block
  scale `asc[row][b] = exp(mb_b − M) / (127·l)`, so P stays unbiased in [0,127]
  and the int8×u8 product needs **no zero-point correction**.
- `k_vattn_ctx` — the context product, consuming those scales.

That is the streaming-softmax structure we lack; it is why they can afford the
vision attention on the device at 1200 patches and we cannot.

### What to build

One fused kernel per tower shape, in this order:

1. **Text, `head_dim` 256/512**: 16 query rows × 128-key blocks, K/V staged in
   SLM, online softmax with FP16 probabilities, P fed back as the DPAS A operand
   (the accumulator layout *is* the A layout, so no shuffle), context quantised
   in the epilogue with the 128-feature block owned by the workgroup.  This
   removes the f32 score matrix, the separate softmax and the separate
   quantiser: 6 launches per layer → 1.
2. **Vision, `head_dim` 64**: the same with the 32×32 score tile and 64×16
   context tile they measured.

---

## P3. The GEMM — 1.7x in situ, and a 2x in-situ penalty

### What we do

One kernel body, two tile variants, `gridCountZ` over a run of GEMMs that share a
shape and differ by a constant stride (the attention heads; `zrep` folds the GQA
mapping in).  Tiles:

| file | workgroup tile | threads | best at |
|---|---|---|---|
| `e2_xmx.cl` | 128 tokens × 64 features | 256 | m ≤ 512 |
| `e2_xmx_big.cl` | 256 tokens × 64 features | 512 | m ≥ 768 |

Activations are **centred int8** (`intel_sub_group_i8_i8_matrix_mad_k32`), which
deletes the `128·sum(w)` correction and its `wsum` table entirely — that change
alone was 4.07 → 9.4 TMAC/s in round 9 and is the one place where our design is
better than theirs, so **do not copy their u8+`bsq` form**.

Measured: **5.1 TMAC/s** isolated at m=512 (`gpu-bench E2_GB_LAYERS`), 2.1-2.6
inside the real forward.  So there is a ~2x penalty for running the same kernel
in the pipeline, and a further 1.7x against their in-situ rate.

### Why the in-situ penalty exists — open

Ruled out, each by measurement: the epilogue store (skipping it changes nothing),
the operand loads (skipping them changes nothing), the tile (full sweep, table in
PERFORMANCE.md), the barrier (~5 us/launch, 3 ms total), launch count, and
device-local vs shared memory (2.36 vs 2.29 ms at m=8 — nothing).

Not ruled out: what their pipeline does differently per launch.  See P3b.

### What the older engine does (P3b)

- **One weight buffer, one activation buffer, integer byte offsets.**  All packed
  weights live in a single `wbuf`, all activations in a single `acts`.  A
  descriptor array of `int32` holds **byte offsets** (16 int32 per tensor,
  `E2_DS_STRIDE`; 9 tensors per layer, `E2_DS_PER_LAYER`), and the kernel selects
  its descriptor with `get_global_id(2)`:

  ```c
  __global const int *dd = desc + t * E2_DS_STRIDE;
  const int N = dd[4], K = dd[5], nb = dd[6];
  __global const uint *wt = (__global const uint *)(wbuf + dd[9]);
  __global const uchar *qa = acts + dd[7];
  ```

  This is how they batch GEMMs in one launch.  It is also the design that works
  where our attempt failed: **a pointer stored in a `__global` struct does not
  survive the IGC SPIR-V path** (every output was wrong, silently).  Offsets do.

- **M is padded to the tile by the caller**, so the kernel's activation loads are
  unguarded and the stores are the only guarded thing.

- **The workgroup tile is 16 tokens × (GY × 16) features**, with `GY = 32` below
  512 rows and `GY = 16` above, chosen per launch.  Note this is the opposite
  geometry to ours: a *small* token tile and a very wide feature tile.  Their
  reason: "`E2_DM` is how many tokens share one weight read... `E2_DN` only
  affects how many tokens share one activation read", and the fp32 accumulator
  (one per output element) is what limits `E2_DM` to 16.

- **256 GRFs only for the bulk GEMM.**  `l0.c` keys off the kernel *name* prefix
  `k_gemm_dpas_large`, so only the bulk projection and the fused FFN get 256 GRFs
  (no reported spill); the ordinary and short kernels keep 128 GRFs and higher
  occupancy.  Their ordinary GEMM spills 288 bytes and they accept it.  Blanket
  mode was measured worse there — and we measured blanket mode as no gain here,
  twice.

- **Fused FFN** (`k_gemm_dpas_large_gate`): both 2048×512 projections share input
  loads, and 16 queries × 128 output features stay in registers plus 8 KiB of
  workgroup SLM through GELU and per-128 quantisation.  Avoided 4.59 GiB of
  logical gate/up traffic for a 32×196 batch.  An 8-query variant was slower.
  Ours (`k_mlp_gu`) is the same idea with a 64-feature tile and an SLM reduction
  for the block max; theirs avoids the reduction by owning 128 features.

### What to build

1. Convert the weight cache and the activation arena to **one buffer each with
   integer offsets**, and batch a layer's independent projections through a
   descriptor array.  This is the only structural difference left on the GEMM and
   the prime suspect for the 2x in-situ penalty.
2. Re-tile the MLP fusion to a 128-feature workgroup tile so the per-128 block
   max needs no cross-sub-group reduction.

---

## P4. The towers that are still on the CPU

| tower | `e2.c-next` | `~/Projects/e2.c` |
|---|---|---|
| audio (mel + conformer) | **entirely CPU** (`e2_gpu_no_dev`) | on the device: `k_audio_dpas4`, `k_audio_reduce_k`, `k_audio_attn`, `k_audio_norm2`, `k_audio_silu_quant`, `k_audio_glu`, `k_audio_dwconv`, `k_audio_add_clamp`, `k_audio_bias_norm` |
| vision attention | CPU (device path measured 9% slower) | device, `k_vattn_*` with a 32×32 tile (see P2) |
| token embedding lookup | serial CPU loop (`e2_text_embed_ids`) | `k_embed` on the device, 0.01 ms |
| pooling | CPU | `k_pool_l2` on the device, 0.09 ms |

Audio is 33 ms of which the tower is most, against 21.5 ms for the old engine.
The embedding lookup and pooling are small but they are the reason their forward
has no host work between submissions.

---

## P5. Measurement traps (so the next person does not lose a round)

Four wrong conclusions in `PERFORMANCE.md` came from the measurement, not the
engine.  All four are now guarded, but the pattern recurs:

1. **A testbed that does not check a result is not a measurement.**  `gpu-bench
   E2_GB_LAYERS` computed its grid from `E2_GB_RSG/CSG` (default 4/8) while the
   checked-in SPIR-V is compiled 8/4, so it stepped `n` by 64 over a grid that
   spanned `n/128`: **half the output columns, 1.8x too good**, silently.  It now
   prints the tile it used and checks the grid covers `m`.
2. **A tile sweep must launch the tile it compiled.**  The first sweep passed the
   tile as `-D` flags but launched with the host's fixed work-group size, so the
   bigger tiles ran a fraction of their tile and reported an impossible
   35 TMAC/s.  `E2_GB_RSG/CSG` exists for this.
3. **A profiling knob must not cache a decision it is filtering on.**
   `E2_GPU_NOLAUNCH` set its own "already parsed" flag inside the filter, so it
   stopped filtering after one kernel and "GEMM only" was really the whole
   pipeline.  An entire phase attribution was fiction.
4. **An A/B must write the buffer it compares.**  Device-local vs shared memory
   showed a phantom 1.6x because the device buffer was never uploaded, so the
   kernel read uninitialised (probably zero-mapped) pages.  Correct answer: no
   difference at all.

And the host itself: this box is shared, load average 1-8, and the same binary
measures 64, 73, 87 and 77 ms for text 512 within minutes.  **The only usable
number is a same-run interleaved min-of-N ratio** (`tools/ab.sh`).  Never compare
an absolute ms across runs, and never compare against an OpenVINO column from a
different session.

---

## Kernel map

| job | `e2.c-next` | `~/Projects/e2.c` |
|---|---|---|
| int8 GEMM | `k_gemm_z`, `k_gemm_b2` (+ `_big` variants) | `k_gemm_dpas2`, `k_gemm_dpas_large`, `k_gemm_dpas_small`, `k_gemm_dpas_slm`, `k_gemm_dpas` (dp4a fallback: `k_gemm_tiled`, `k_gemv_split`+`k_gemv_reduce`) |
| row ops | `k_rmsnorm_q`, `k_quant_s8`, `k_norm_rope`, `k_scale`, `k_mul_slice`, `k_gelu_tanh` | **`k_rowops`** (10 modes), `k_norm_rope3`, `k_norm_rope3_i8` |
| fused FFN | `k_mlp_gu` + `k_gemm_b2` | `k_gemm_dpas_large_gate` |
| fused PLE | `k_gemm_ple` + `k_gemm_b2` | (part of `k_rowops` + the gate kernel) |
| gelu·mul·quantise | inside `k_mlp_gu` / `k_gemm_ple` | `k_gelu_mul_quant` |
| text attention | `k_attn_pack`, `k_gemm_z`, `k_attn_softmax`, `k_gemm_z` | `k_attn_f16_pack`, `k_attn_f16`, `k_attn_f16_softmax`, `k_attn_f16_ctx` (int8 alternative: `k_attn_packw`, `k_attn_gemm`, `k_attn_batch_gemm`, `k_attn_i8_ctx`) |
| vision attention | **CPU** | `k_vattn_packq/packk/packv`, `k_vattn_scores`, `k_vattn_psoft`, `k_vattn_pasc`, `k_vattn_ctx` |
| audio tower | **CPU** | `k_audio_*` (9 kernels) |
| embedding | CPU | `k_embed` |
| pooling | CPU | `k_pool_l2` |
| rope (vision) | `k_rope_vision` | `k_vnorm_rope3` |
| position add | `k_pos_add` | `k_vpos_add` |
| vision patch | (CPU) | `k_vpatch_q` |

---

## Work order

Each item is independent; the payoff estimates are from the measured shares.

1. **P1 — `k_rowops` with mode bits.**  266 → ~90 launches, no separate
   `quant_s8`.  ~10-15 ms at 512 tokens.  Lowest risk, largest measured gap left.
2. **P2 — fused attention, text first.**  6 launches per layer → 1, and the f32
   score matrix disappears.  ~15 ms at 512, most of image/video.
3. **P3 — one `wbuf` + one `acts` + integer-offset descriptors.**  Prime suspect
   for the 2x in-situ GEMM penalty; also removes 267 separate USM allocations.
4. **P2 — vision attention with the 32×32 tile and the blocked softmax.**
   Unlocks image/video.
5. **P4 — audio tower on the device.**  33 → ~20 ms.
6. `k_embed` and pooling on the device.  Small, but removes the last host work
   inside the forward.

---

## How to measure

```bash
make && make tests/e2_test tests/e2_bench_full gpu-bench

# correctness first: the device path must agree with the CPU path
E2_DEVICE=gpu ./tests/e2_test models          # includes a cos>0.995 CPU-vs-GPU check

# interleaved A/B of two binaries - the only trustworthy number on this host
cp tests/e2_bench_full tests/e2_ab_a          # baseline binary
cp tests/e2_bench_full tests/e2_ab_b          # candidate binary
E2_DEVICE=gpu sh tools/ab.sh SPEC 6

# the GEMM phase in isolation (real per-layer shapes, 24 cold weight buffers)
E2_GB_LAYERS=1 E2_GB_REPS=6 ./gpu-bench 512

# where the forward's device time goes
E2_DEVICE=gpu E2_GPU_TS=1 ./tests/e2_bench_full SPEC      # per-kernel timestamps
E2_GPU_NOLAUNCH=1 E2_GPU_NOATTN=1 ...                    # GEMM launches only
E2_GPU_NOATTN=1 ...                                      # price the attention
E2_GPU_SKIP=linear,norm,rope,attn,gelu,ew ...            # keep an op group on the CPU

# accuracy against the float32 reference
python tools/e2_ref.py text --text "..." --out /tmp/ref
E2_DEVICE=gpu E2_DUMP=/tmp/c ./tests/e2_bench_full SPEC  # then compare the dumps
```

Accuracy budget: the path is at **cos 0.9997** against the CPU path and the
target is 0.999.  `tests/e2_test.c` compares the two paths directly, because the
older checks only looked at the embedding's norm and pass on a completely wrong
kernel.
