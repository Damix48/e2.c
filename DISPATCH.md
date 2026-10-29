# Attention path selection, and a plan to standardize it

Status: **implemented**. Phases 0–6 have landed: `src/dispatch.c` owns the capabilities, the
per-(backend, tower) rules and the one query; `src/kernels.c` selects backends from two ordered tables;
`E2_ATTN_TEXT`/`E2_ATTN_IMAGE` override the table with the old names as aliases; `E2_PROFILE` prints
the decision; `tests/test_dispatch.c` asserts the matrix on any host; and §2.5's invariants are in
AGENTS.md. Part 1 below is kept as the record of the state at the time of writing — its §1.5 table of
where each constant lives is historical. Two items are deliberately not done: the HIP staging rename
(no host in the set can compile HIP) and `E2_ATTN_AUDIO` (no selectable audio path); both are explained
in §2.3 and §2.9.

See [PERFORMANCE.md](PERFORMANCE.md) for the measurements, [AGENTS.md](AGENTS.md) for the engine's
conventions.

---

# Part 1 — How it works today

The engine picks a precision in **four** places. Only one of them is "attention", and only one has a
length threshold that varies by more than a constant.

## 1.1 The weight GEMM (W8A8) — no threshold anywhere

Every projection in all three towers — q/k/v/o, gate/up/down, vision patch embed, audio
ff1/ff2/out_proj/embed — runs as int8 weights × int8 activations, 128-element blocks with one fp32
scale per block. This is >90 % of the arithmetic the engine performs.

There is **no sequence-length threshold here**, on any backend. At every shape measured, int8 beats
bf16 by 4–12×, so the only two states are "available" and "unavailable". Which one you get is decided
once at startup by CPUID/HWCAP (`e2_cpu_init`, `src/kernels.c`):

| backend | int8 GEMM instruction | activation quantizer |
|---|---|---|
| AVX512-VNNI (SPR+, Zen 4/5) | `vpdpbusd`, 512-bit | AVX-512 |
| AVX-VNNI (Alder Lake+) | `vpdpbusd`, 256-bit | AVX-VNNI |
| AVX2 (Haswell … Zen 3) | `vpmaddubsw` | AVX2 |
| aarch64 + dotprod (Neoverse-N1+) | `vdotq_s32` | NEON |
| aarch64 without dotprod (Pi 4) | **none** | **none** |
| generic scalar | scalar reference | **none** |

## 1.2 The activation quantizer — the master switch

Converts fp32 activations to int8 plus a per-block scale, before every GEMM. Two things matter:

1. **Its absence disables int8 for the entire engine**, not one region. `src/e2.c:388` sets
   `use_int8 = (e2_i8_quantize != NULL) && !(E2_INT8=0)`, so the scalar fallback and ARMv8.0 fall all
   the way back to bf16 weights. On the ARM box that cost **12.3×** (373.1 → 30.4 ms) until the NEON
   port; the 81× figure in PERFORMANCE.md is the scalar backend against the laptop's int8 + iGPU, a
   different comparison — do not read it as the price of a missing quantizer on ARM.
2. **It is also what the text int8 attention path uses** to quantize Q and the probabilities, so text
   int8 attention inherits that availability rule and needs no threshold of its own.

## 1.3 Attention — the only part with real thresholds

Attention is two products per head per layer:

- **logits** = Q · Kᵀ — cost `S × S × D`
- **context** = P · V — cost `S × D × S`

Its setup cost is **packing**: quantizing K and Vᵀ into the block-major int8 layout, once per head per
layer — cost `S × D`. The ratio is `S` against the packer's constant, so:

> **packing amortises only once the sequence is long enough.** Every threshold below is that single
> trade-off, expressed in whatever units the relevant backend happens to use.

That is why text has a length gate, and why the number differs per backend even though the maths is
identical: the packer's speed and the fp32 product's speed are both machine-specific.

### The matrix — attention path per backend × tower

| backend | **text** attention | **image** attention | **audio** attention |
|---|---|---|---|
| CPU · AVX512-VNNI | int8 ≥ **128** keys, else fp32 | **int8, always** | fp32 |
| CPU · AVX-VNNI | int8 ≥ **128** keys, else fp32 | **int8, always** | fp32 |
| CPU · AVX2 only | int8 ≥ **128** keys, else fp32 | **fp32, always** | fp32 |
| CPU · AVX-512 without VNNI | int8 ≥ **128** keys, else fp32 | **fp32, always** | fp32 |
| CPU · aarch64 + dotprod | int8 ≥ **128** keys, else fp32 | **int8, always** | fp32 |
| CPU · aarch64 no dotprod | fp32 (whole engine bf16) | fp32 | fp32 |
| CPU · generic scalar | fp32 (bf16 weights) | fp32 | fp32 |
| Intel GPU · Level Zero | **fp16 < 512**, int8 ≥ **512** | **int8, always** | fp32 |
| AMD GPU · HIP | **fp32 < 64**, int8 ≥ **64** † | **int8, always** | *not implemented* |

The two odd rows:

- **The image column has no length gate on any backend** — but it has a *capability* gate on CPU.
  Image attention needs the fused packed-operand kernel `e2_i8_attn_gemm`, which exists for
  AVX512-VNNI, AVX-VNNI and aarch64 dotprod (`src/kernels.c:318,323,333`). On AVX2 and on AVX-512
  without VNNI that pointer stays NULL (`src/kernels.c:325-328`), so those CPUs run fp32 attention on
  images. NEON was in that group until Attempt 23 added `e2_i8_attn_gemm_neon`
  (`src/kernels_i8_neon.c:261`), which moved the ARM image run from 13.9 s to 8.5 s — a missing
  kernel, not a tuning miss, and now ported.
- **The audio column is fp32 everywhere, by construction** — see §1.4.

Both GPU rows are thinly corroborated: the Level Zero row is exercised on one host out of the five in
HARDWARE.md (the laptop's Arc 140T — the n100's Xe-LP has no XMX and the backend does not build there),
and the HIP rows were, until this landing, never executed at all. They now are: nucleus built the HIP
backend against ROCm 7.2 and both HIP tests pass on its `gfx1036` iGPU (§2.3), which is a correctness
result, not a performance one — the 2-CU device loses to the 8-core CPU on every shape.

† HIP also drops back to fp32 above the threshold when the K/V pack region does not fit the sequence
(`src/gpu/hip/forward.c:411-413`); the Level Zero equivalent of that overflow is a hard error, not a
fallback (`src/gpu/gpu.c:807-811`). The threshold is therefore necessary but not sufficient on HIP.

### What each attention implementation computes

| implementation | how it works | where |
|---|---|---|
| **fp32 (CPU)** | blocked fp32 GEMM (`e2_sgemm_nn_range`), separate softmax pass | `text.c` `attb_worker`, `vision.c` |
| **int8 (CPU, text)** | two ordinary NT products through the existing weight GEMM; Q and P are the quantized activations, K and Vᵀ packed once per head | `text.c` `attb_i8` |
| **int8 (CPU, image)** | dedicated packed-operand kernel + fused softmax-and-quantize, per head | `e2_i8_attn_gemm` + `e2_softmax_quant_i8` |
| **fp16 (Intel GPU)** | XMX DPAS at half precision, fp32 accumulate, materialised fp16 score matrix | `k_attn_f16*` |
| **int8 (Intel GPU, image)** | packed int8 Q/K/V, softmax and quantize fused into one kernel | `k_vattn_psoft` / `k_vattn_pasc` / `k_vattn_ctx` |
| **fp32 (HIP, short text)** | fused per-head kernel, no packing | `e2k_attn` |
| **int8 (HIP)** | `v_dot4` scores + softmax + context, LDS-staged once S ≥ 512 | `e2k_attn_scores8` etc. |

Note the shape difference: **text int8 attention reuses the weight GEMM** (it is just a matmul), while
**image int8 attention needs its own fused kernel** because that arrangement is what lets softmax feed
the quantizer directly. That is why text got int8 on all five SIMD backends nearly for free, and
image only where the fused kernel exists — VNNI from the start, NEON since Attempt 23.

### Why fp16 exists only on the Intel GPU

The iGPU's XMX units execute fp16 at 2× fp32, so fp16 attention is genuinely cheaper there. **x86 CPUs
cannot do this**: the machine measured has `f16c` but no AVX-512-FP16, and F16C only converts — there
is no fp16 FMA. An fp16 CPU path would pay a conversion per element for the identical multiply count.
The absence of a CPU fp16 row is a result, not an omission.

## 1.4 Why image and audio need no threshold

**Image (vision).** The token count is effectively fixed: `E2_MAX_PATCHES` = 2520 patches, or a fixed
subset from soft-token pooling. There is no short-image case, so the `S²` term dominates the `S·D`
pack from the first call (2520/64 ≈ 39×). Nothing to tune — int8 either exists or it does not.

**Audio.** Structural: **audio attention is not O(S²).** It is a 12-key local window with a
relative-position term (`E2_A_PAST` = 12, `E2_A_CHUNK` = 12, `E2_A_RELPOS` = 13 for the d = 0..11
table, `src/model.h:43-45`):

```
for each past distance d in 0..11:
    score[d] = q · k[p-d]  +  q · relk[RELPOS-1-d]
score = 50 · tanh(score / 50)          # logit cap
softmax over those 12, then ctx = Σ p[d] · v[p-d]
```

So per token it is 24·D multiply-accumulates for the scores (12 distances × two D-long dots) plus
12·D for the context — 36·D, against `2·S·D` for a full attention with S up to 750. Quantizing K/V into
128-wide int8 blocks with per-block scales would cost more than the 12 dots it replaced. **An int8
audio attention would be a loss at any length** — which is why the audio column is fp32 on all nine
backends, both GPUs included, rather than merely unported.

## 1.5 Where each constant lives

| constant / literal | value | tower | backend | file |
|---|---:|---|---|---|
| `E2_ATT_I8_MIN` | 128 | text | all CPU | `src/text.c:748` |
| `512` (hardcoded literal) | 512 | text | Intel GPU | `src/gpu/gpu.c:798` |
| `E2_HIP_ATTN_I8_MIN` | 64 | text | HIP | `src/gpu/hip/hip.h:44` |
| `E2_HIP_A8_LDS_MIN` | 512 | text | HIP | `src/gpu/hip/hip.h:50` |
| `vattn_int8_ok()` → `e2_i8_attn_gemm != NULL` | — | image | all CPU | `src/vision.c:801-803` |

The three precision thresholds differ only because the three packers and fp32 products have different
relative speeds. They are not tuned to each other. The fourth row (`E2_HIP_A8_LDS_MIN`) is a *staging*
threshold — direct-load versus LDS-staged int8 products — not a precision one; §2.3 is about keeping
those two namespaces apart, which is exactly what this table currently fails to do.

**This table is now historical.** Every entry above except `E2_HIP_A8_LDS_MIN` (which is a staging
knob and stays in `hip.h`) moved into the rule table in `src/dispatch.c`, and the two image-audit
predicates were replaced by `e2_attn_pick()`; the staging constant deliberately did not move with the
rest.

## 1.6 Measured crossovers, text only

Text is the only path whose crossover has been measured on more than one machine, so the only one with
real numbers. Positive = int8 faster. Best-of-N, alternating, pinned; see PERFORMANCE.md Attempt 22
and its correction block.

| keys | x86 (Core Ultra 9 285H, 6 threads pinned) | Ampere A1 (Neoverse-N1, 2 threads) |
|---:|---:|---:|
| 32 | −24 % | — |
| 48 | −11 % | −51 % |
| 64 | −7 % | −2 % |
| 96 | −5 % | +1 % |
| **128** | **+4 %** | **+5 %** |
| 196 | +12 % | +13 % |
| 512 | +28 % | +35 % |
| 2048 | +34 % | +46 % |

The two curves agree at **128**, which is why it is one constant rather than a per-architecture split.
The x86 curve only looks like this because the packer was parallelised across heads — with the original
serial pack loop the crossover sat at 384 and 196 keys *lost* 10 %. The remaining cost on both machines
is the scalar `e2_q8w_from_f32` packing loop, not the int8 GEMM.

Two caveats on the single constant. A **third host**, the Zen 5 9700X (full-width AVX-512, the
strongest fp32 GEMM of the set), was re-measured afterwards and its crossover is **196-256** (196 keys
~2 % ahead, 256 break-even, 384 +10-12 %); 128 is therefore conservative rather than exact. It stays
because a few percent of early int8 costs less than a per-CPU constant the project has explicitly
rejected. The N100 could neither confirm nor contradict the threshold — its run-to-run spread exceeded
the difference — so it is not evidence either way. Absolute milliseconds are not comparable across
machines here: the laptop's numbers are pinned to 6 threads by design, and the same session measures
41.4 ms at 6 threads against 77.7 ms at the default 14 (HARDWARE.md); only the int8/fp32 *ratio* on one
machine is the claim.

## 1.7 The switches that overrule all of the above

| variable | effect |
|---|---|
| `E2_INT8=0` | force bf16; refuses a packed `.e8cq` cache (it holds only int8 weights) |
| `E2_ATTN_I8=0` / `=1` | force text attention fp32 / int8 regardless of length |
| `E2_VATTN_OFF=1` | force image attention fp32 |
| `E2_GPU_VATTN_OLD=1` | Intel GPU: pre-fusion three-kernel image attention (fp16 score matrix) |
| `E2_HIP_ATTN_I8_MIN=n` | HIP: move the text threshold |

---

# Part 2 — Standardization plan

## 2.1 What is actually messy

| # | inconsistency | evidence |
|---|---|---|
| 1 | **Three different mechanisms** for the same question | `text.c:748-752` inline `#define` + predicate; `vision.c:801-803` a function that probes a *data pointer*; `audio.c` has no decision at all — fp32 is simply the only code that exists |
| 2 | **Three spellings of the same constant name** | `E2_ATT_I8_MIN` (text, generic), `E2_HIP_ATTN_I8_MIN` (text, HIP), and a bare literal `512` inside `gpu.c:798` with no name |
| 3 | **Five env spellings of "pick the attention path"** | `E2_ATTN_I8` (bool string), `E2_VATTN_OFF` (off-only), `E2_GPU_VATTN_OLD` (old-vs-new), `E2_HIP_ATTN` (`old`/`lds`), `E2_HIP_ATTN_I8` (`direct` — a *staging* switch wearing a *precision* name) |
| 4 | **Capability probe = data pointer** | `e2_i8_attn_gemm != NULL` is both "the fused kernel" and "is int8 image attention available". That is why ARM silently ran fp32 vision for two attempts before Attempt 23 ported the kernel: the predicate has no name that says so |
| 5 | **Backend existence is knowledge spread over three files** | the if/else ladder in `e2_cpu_init()` (`kernels.c:312-364`), plus `gpu.c:256-264`, plus `hip.c:120-131` |
| 6 | **Tower asymmetry** | nothing can answer "which attention for this input", because the answer is assembled in three unrelated blocks |
| 7 | **No observability** | `E2_PROFILE` prints timings but never the *decisions*; every summary like Part 1 has to be reverse-engineered from source |
| 8 | **Tests encode the spellings** | `test_text_attention` flips `E2_ATTN_I8` via `setenv`; `test_gpu_attn_f16` depends on the unnamed `512` |

Items 4 and 6 are not cosmetic: a named capability and one selector would have made "vision falls back
to fp32 on NEON" visible at a glance instead of discovered by profiling.

## 2.2 Target structure: two layers, one function

**Capabilities** (what a backend *has*) × **rules** (when to use each thing), queried through **one
function** that all three towers call.

**Layer 1 — capabilities.** One struct per backend, filled at init, beside the existing kernel table in
`kernels.h` / `gpu.h` / `hip.h`:

```
name   family   caps
```

`caps` is a bitmask: `GEMM_I8`, `QUANT_I8`, `ATTN_I8_GENERIC`, `ATTN_I8_FUSED`, `ATTN_F16`. No numbers
in it. `name` is the string `e2_cpu_name()` already returns, so profile output and the table agree by
construction. This replaces the implicit facts — "text int8 needs gemm+quant", "image int8 needs the
fused kernel", "Intel has fp16" — with declared ones.

**Layer 2 — rules.** A single static table in one new file, indexed `[backend][tower]` →
`{kind, min_keys}`:

```
kind ∈ { F32, F16, I8 },   min_keys = 0 means "no length gate"
```

Designated initializers; anything unlisted is an explicit `{F32, 0}`. **This table is Part 1's matrix**,
and its comments become the "why" documentation that currently lives nowhere:

- text / N1+ → `{I8, 128}` — packer amortises at 128
- image / AVX2 → `{F32, 0}` — no fused kernel on AVX2; capability, not length
- image / VNNI → `{I8, 0}` — fixed 2520 patches, nothing to amortise
- audio / any → `{F32, 0}` — 12-key local window: an int8 pack would cost more than the dots
- text / Level Zero → `{I8, 512}`, fp16 below — XMX fp16 wins under 512

**Layer 3 — the query.**

```
e2_attn_kind e2_attn_pick(e2_backend bk, e2_tower tower, int n_keys);
int          e2_attn_min_keys(e2_backend bk, e2_tower tower);
```

Takes the backend explicitly (see §2.9 item 5: the test has to be able to ask about a backend the host
does not have), looks up the rule, applies the length gate to pick `kind` or `below`, then degrades
I8 → F16 → F32 if the capability the rule asked for is missing. Two call sites in the landed code:
`text.c` `attb_i8`'s predicate and `vision.c` `vattn_int8_ok` (whose pointer probe is deleted). Audio
got no call site after all: a call that always returns F32 would be dead code, and leaving its rows
unlisted already says F32. The GPU backends read the same table where they had literals — `gpu.c`'s
`512` became a `pick()` and `hip.c`'s default threshold became `min_keys()`.

The whole structural change, measured: **+327 / −19 lines** (113 in `dispatch.c`, 65 in `dispatch.h`,
149 in the test, 50 insertions across nine existing files) and the matrix readable on one screen.

## 2.3 Naming standard

One rule: `<prefix>_<region>_<thing>`, and no concept with two names.

| today | proposed | note |
|---|---|---|
| `E2_ATT_I8_MIN` | row in `k_rules` | value stays 128 |
| literal `512` in `gpu.c:798` | `E2_L0_ATTN_TEXT_MIN` in the table | a literal at a call site is the worst of the eight |
| `E2_HIP_ATTN_I8_MIN` | `E2_HIP_ATTN_TEXT_MIN` in the table | same concept, same suffix |
| `e2_i8_attn_gemm` | unchanged, but only *called* | predicate becomes `CAP_ATTN_I8_FUSED` |
| `E2_ATTN_I8`, `E2_VATTN_OFF`, `E2_GPU_VATTN_OLD`, `E2_HIP_ATTN`, `E2_HIP_ATTN_I8` | `E2_ATTN_TEXT`, `E2_ATTN_IMAGE`, `E2_ATTN_AUDIO` = `auto\|f32\|f16\|i8`; `E2_ATTN_STAGE` = `auto\|direct\|lds` | 5 → 4, one grammar |
| `E2_GPU_GEMM_OLD`, `E2_AUDIO_DPAS_OLD`, `E2_HIP_ATTN=old` | `E2_<BACKEND>_KERNEL_<REGION>=old` | see below |

The distinction the current names blur: **precision selection** (`E2_ATTN_*`, decides *what maths*)
versus **kernel-generation A/B** (`*_OLD`, decides *which implementation of the same maths*). Keeping
them in one namespace is why `E2_HIP_ATTN_I8=direct` reads like a precision flag and is not one.

Old env names stay as deprecated aliases parsed in one place, so no test or script breaks in the same
commit as the refactor.

**Landed:** `E2_ATTN_TEXT` and `E2_ATTN_IMAGE` exist (`auto|f32|f16|i8`), `E2_ATTN_STAGE` exists
(`auto|direct|lds|legacy`), both are read once by `e2_cpu_init()`, and the old names — `E2_ATTN_I8=0/1`,
`E2_VATTN_OFF`, `E2_HIP_ATTN=old|lds`, `E2_HIP_ATTN_I8=direct` — are parsed in the same place as
deprecated aliases with lower precedence. `E2_ATTN_AUDIO` is deliberately **not** parsed: audio has no
selectable path (§1.4), so a variable that can never change the answer would be a lie in the interface.

`E2_ATTN_STAGE` is a generation switch, not a precision one: `direct` and `lds` choose between the two
int8 attention kernel families, and `legacy` selects the pre-fusion scalar per-row fp32 kernel — which
is fp32 by construction, and is what `E2_HIP_ATTN=old` used to mean. Only `hip.c` consumes it; Level
Zero keeps its own generation switch (`E2_GPU_VATTN_OLD`) in the `E2_<BACKEND>_KERNEL_<REGION>`
namespace of row six.

This was the one row that could not be verified when it was written, because no machine could build the
HIP backend. That has changed: with the full ROCm 7.2 stack on nucleus, `e2_hip` builds (clang 22,
kernels for `gfx1036`, `libamdhip64`) and both HIP tests pass. On that 2-CU iGPU: text query 12.3 ms /
cos(gpu,ref) 0.999780, document 107.0 ms / 0.999626, image 0.999778, image+caption 0.999661. The
switches were checked against each other rather than by assertion — `E2_ATTN_TEXT=f32` and
`E2_HIP_ATTN=lds` give the same document (180.6 vs 180.8 ms, identical cos), `E2_ATTN_STAGE=legacy` and
`E2_HIP_ATTN=old` likewise (185.8 vs 191.6 ms, identical cos), and `E2_ATTN_TEXT=i8` moves the 15-key
query onto int8 (cos 0.999765 vs 0.999780, 604 vs 460 kernels) because the override drops the 64-key
gate. The earlier root-free compile check (`dpkg-deb -x` of hip-dev/hip-runtime-amd, then `gcc -c` on
the host files) had already found one real bug in this wiring — a wrong `../dispatch.h` depth that
nothing in the tree could see — which is why the full build was worth waiting for.

## 2.4 Phases

Each independently reviewable and measurable.

| phase | work | lines | risk | status |
|---|---|---|---|---|
| **0. Pin the matrix** | `tests/test_dispatch.c`: assert the expected `{kind, min_keys}` per backend per tower per representative length. **Reordered after phase 1** (§2.9 item 5) so it asserts the selector rather than re-implementing it. | +120 | none | **landed** |
| **1. Extract constants** | New `src/dispatch.c` holds the table; existing `#define`s and the `512` literal move into it, call sites read the table. No behaviour change. | +40 / −10 | none | **landed** |
| **2. Unify the three gates** | `text.c` / `vision.c` each call `e2_attn_pick()`; the pointer probes are deleted. Audio needs no call: its row is unlisted and therefore F32. `caps` replaces the implicit probes in `kernels.c`, `gpu.c`, `hip.c`. | +30 / −80 | **the real one** | **landed** |
| **3. Unify env** | New names + one alias parser; tests migrated off `setenv` where possible. | +50 / −40 | low | **landed**: `E2_ATTN_TEXT`/`E2_ATTN_IMAGE`/`E2_ATTN_STAGE` + aliases, no tower reads `getenv`, HIP checked against the old names on gfx1036; `E2_ATTN_AUDIO` deliberately not parsed (§2.3) |
| **4. Backend registry** | `e2_cpu_init()`'s ladder becomes an ordered array of `{probe, install, caps}` entries; two arrays, because AVX-512-without-VNNI needs the AVX-512 fp32 kernels and the AVX2 int8 ones. | +30 / −30 | low | **landed**: also fixes `e2_cpu_name()` on aarch64, which reported `"scalar"` while NEON ran |
| **5. Observability** | One line per selected path per tower under `E2_PROFILE`: `text attn: i8 (S=196, min=128)`. `e2_backend_caps()` is the accessor it will read. | +25 | none | **landed** as `attn path i8 (S=196, min=128)` in the text and vision profile blocks; audio has no decision to report |
| **6. Docs** | Replace the AGENTS.md prose with this table; note the split in PERFORMANCE.md. | ±0 | none | **landed**: AGENTS.md points at `dispatch.c`, lists it, carries §2.5's invariants and the new env names; PERFORMANCE.md records the landing and the A/B |

Phase 0 is what makes phase 2 safe, and is worth doing even if the refactor stops there: nothing in the
test suite currently asserts *which* backend takes *which* path.

**Gate per phase:** `ctest` on both CPU machines (21/21 on x86-64 and 14/14 on aarch64 in the
CPU-only, no-checkpoint configuration those counts came from; the full configuration with
`-DE2_MODEL_DIR` and the GPU backends registers 42 tests), plus — for phases 1–2 only — one
`e2_text_bench` per machine that must not move more than 2 %. Anything above that is reverted, per
the project rule.

## 2.5 Invariants the standard should enforce

**Now in AGENTS.md** (Development Rules, "Attention precision goes through `src/dispatch.c`"), which is
the copy a contributor will actually read. Kept here as the longer form:

1. Every length threshold is a **named constant in `dispatch.c`**, never a literal at a call site.
2. Every precision decision goes through **one function**; no tower file contains a gate.
3. Towers differ by **table data**, not by code structure.
4. No `#if` in tower files — ISA conditionals live in `kernels.c` and the kernel TUs.
5. A capability is a **named flag**, never `ptr != NULL`.
6. Precision switch and kernel-generation switch are **different namespaces**.

## 2.6 What must *not* be standardized

Per the project's "complexity only for speed" rule, these stay per-backend and per-tower:

- **Kernel internals** — blocking factors, `MR` in the NEON GEMM, packed layouts. Standardizing those
  costs speed for zero readability.
- **The two attention implementations stay two.** Text reuses the generic int8 GEMM (it is a matmul);
  image needs its fused packed kernel with softmax feeding the quantizer. The plan unifies *selection*,
  not *implementation*; `ATTN_I8_GENERIC` vs `ATTN_I8_FUSED` exist precisely to keep that difference
  legible.
- **The `K % 128 == 0` precondition** on `e2_i8_gemm` — fixing it in-kernel was measured at 5 % on the
  whole int8 GEMM; keep caller-side padding and keep the documented precondition.
- **No config file, no runtime-loaded constants.** The table is compile-time; env only for A/B. A config
  parser would contradict the baked-in-constants rule.

## 2.7 Payoff, honestly

1. **A whole bug class becomes visible.** Backend×tower degradation is currently discoverable only by
   profiling. With `caps` + `pick()`, `E2_PROFILE` prints it in one line, and phase 0's test asserts it.
2. **Fewer `getenv` calls in a hot path.** `text.c:750` re-reads `E2_ATTN_I8` *per layer* because the
   tests need to flip it. A backend struct resolved at init plus a test setter removes that, and deletes
   the comment explaining the workaround.
3. **Part 1 of this document becomes source, not archaeology.** Answering "what threshold on what
   hardware" currently requires reading five files; after phase 1 it is one table with comments.
4. **A backend can be measured, not just selected.** The registry made `E2_CPU_BACKEND` and
   `e2_text_bench ... backends` cheap to add, and the same binary now reports every backend the host
   can execute. First run, laptop, 196 tokens: vnni 40.05 ms against avx2 117.48 ms — the AVX2-only row
   of the matrix is now a measured 2.9× on the whole engine, not a guess. That is the part of this
   work that helps the *next* kernel, which was the point.

Net size, measured after the landing: **+327 / −19** lines (113 `dispatch.c`, 65 `dispatch.h`, 149
`tests/test_dispatch.c`, 50 insertions and 19 deletions across nine existing files). The original
estimate of +200 / −160 guessed the deletion side too high: the three gates were one-liners, so almost
nothing was removed, and the test grew past its budget. It is still a readability change with no
arithmetic move, not a rewrite — no public header, no `.e8cq` cache format, no ABI change;
`dispatch.c` sits in `src/` beside `kernels.c`.

## 2.8 Risks

- Phase 2 touches the exact code path whose ordering (`pack → gemm → softmax → quantize → gemm`) took
  several debugging rounds. **Phase 0's test is what stops a silent re-introduction of the `Sp`-stride
  and `K`-padding bugs — do not skip it to reach the refactor faster.**
- Phases 1–2 each need one bench run per machine to prove nothing moved; budgeted in the gate, not yet
  spent.
- The env-var work in phase 3 is scoped to the attention-related variables only. The tree holds ~50
  `E2_*` variables in total; a full audit is a separate pass and probably phase 3's real scope.

## 2.9 Open questions, and how the landing answered them

These came out of a source review of this document; each one changes the shape of the code in §2.2 and
should be decided rather than discovered.

1. **Below-threshold target.** `{kind, min_keys}` says what to prefer at or above the threshold, not
   what to use below it. The Level Zero row's "fp16 below 512" is currently produced by the degrade
   chain, not by the rule — a preference encoded by luck. Either make the entry `{kind, min_keys,
   below}` or state explicitly that below `min_keys` walks the same `I8 → F16 → F32` chain.
2. **The fused softmax is its own capability.** `vision.c:790` falls back to `K->softmax` +
   `e2_i8_quantize` when `e2_softmax_quant_i8` is NULL, so `ATTN_I8_FUSED` must not imply it. Add a
   `SOFTMAX_Q_I8` cap — HARDWARE.md already tabulates the softmax+quantizer per ISA, including AVX2,
   which has it and still has no attention GEMM.
3. **Caps are per device on the GPU backends.** On Level Zero the capability set differs between an
   Arc 140T (XMX, fp16 DPAS) and the n100's Xe-LP (no matrix units, and the backend does not build
   there). One struct per backend gives the GPU rows a meaning they do not have; they need a device
   tier, or the window says they are unverified.
4. **The table key must not be `e2_cpu_name()`.** A CPU with AVX-512 but no VNNI names itself
   `avx512+fma` while running AVX2 int8 kernels, and on aarch64 `e2_cpu_name()` returns `"scalar"`
   while NEON int8 and the NEON fp32 GEMM run (`kernels.c:358-363`). That also breaks HARDWARE.md's
   machine-identification checklist, which reads that string. Key the table on a backend enum; fix the
   aarch64 name as a separate bug.
5. **Phase 0 cannot observe what it wants to assert.** The gates are static functions and inline
   pointer expressions, and the only observable today is the pointer being deleted. Land the extract
   (`dispatch.c` + a `pick()` that takes an explicit backend) as phase 0, and the matrix test as
   phase 1; the test then asserts all rows on any host. That ordering also avoids the trap HARDWARE.md
   records: never probe a backend by calling its kernel (AVX-VNNI on the Zen 4 minipc is a
   `STATUS_ILLEGAL_INSTRUCTION`), so `caps` must stay inert data.
6. **Item 8 undercounts the test coupling.** `tests/test_cpu_backends.c` `#include`s `kernels.c` and
   rewrites `e2_cpu_init()` by hand — the largest single "spelling" in the tree, and the one phase 4
   is guaranteed to break. It is not in §2.1's list because it encodes the *ladder*, not a threshold.

**How the landing answered them.** Item 1 became the rule's third field (`below`), so Intel's fp16
below 512 is a preference written in the table rather than a side effect of the degrade order. Item 2
became `E2_CAP_SOFTMAX_Q_I8`, kept separate from `ATTN_I8_FUSED` because vision falls back to plain
softmax when it is missing. Item 4 became the `e2_backend` enum plus `e2_cpu_backend()` — the profile
line and the machine-identification checklist should read that, not `e2_cpu_name()`. Item 5 was resolved
by landing the extract first and giving `e2_attn_pick()` an explicit backend argument, which is also why
the matrix test can assert every row on any host without invoking a kernel. Item 6 was resolved by
having `test_cpu_backends.c` declare the dispatch backend it fakes, in the same `switch` that fakes the
pointers. Item 3 (caps per device) is **closed as not applicable**, with the reason now known: the
Level Zero backend is compiled per device id by `cmake/embed_binary.cmake`, and a device without XMX
does not compile the module at all (the n100 fails at `ocloc`, HARDWARE.md records the two causes). So
"the backend exists" already implies "the device has the kernels", and a per-device caps table would be
speculative configurability with nothing to read. If a non-XMX Level Zero target ever builds, caps must
move from the static table into init — that is the condition to revisit, not a TODO.
