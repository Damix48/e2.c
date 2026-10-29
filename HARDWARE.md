# Hardware matrix

What each machine in this set actually has, what its matrix/vector-dot silicon is, and which part of
`e2` runs on it. Every row was measured on the machine itself (`/proc/cpuinfo`, `clinfo`, `lspci`, and
the engine's own `e2_cpu_name()`), not copied from a spec sheet - two of the entries below contradict
what the marketing name would suggest, and one contradicts what the CPU's own extension flags suggest.
Host addresses are deliberately absent; the aliases are just how the machines are referred to in
`PERFORMANCE.md`.

Timings live in [PERFORMANCE.md](PERFORMANCE.md); this file is about capability. Latencies are repeated
here only so a row can be read on its own.

## The systems

| system | CPU | CPU int8 / matrix ISA | GPU | GPU matrix capability | what e2 runs there | 196-token doc |
| :-- | :-- | :-- | :-- | :-- | :-- | --: |
| laptop (local) | Intel Core Ultra 9 285H, Arrow Lake, 16T | **AVX-VNNI** (256-bit `vpdpbusd`); no AVX-512, no AMX | Intel Arc 140T (0x7d51) | **XMX/DPAS**: `intel_sub_group_i8_u8_matrix_mad_k32`, `u8_i8_k32`, `f16_f16_k16` | CPU: `kernels_i8_vnni.c` int8 GEMM + packed-operand attention GEMM, `kernels_i8_avx2.c` fallback, `kernels_avx2.c` fp32. GPU: Level Zero `src/gpu/` with DPAS kernels | 41.4 ms |
| nucleus | AMD Ryzen 7 9700X, Zen 5, 8C/16T, **full-width 512-bit** | **AVX512-VNNI** (zmm `vpdpbusd`) + AVX-VNNI; no `avx512_fp16` | Radeon iGPU (Granite Ridge, 2 CU, `gfx1036`) | HIP via ROCm 7.2: `v_dot4`, no MFMA/WMMA; built and smoke-tested, not performance-validated | CPU: `kernels_i8_avx512_vnni.c` (512-bit, 16 output rows per weight load, packed-attention GEMM, fused softmax+quantizer), `kernels_avx512.c` fp32. GPU: `src/gpu/hip/` text and vision | **24.9 ms** |
| minipc | AMD Ryzen 9 7940HS, Zen 4, 8C/16T, **256-bit datapath, double-pumped** | **AVX512-VNNI** + AVX512-BF16; **no AVX-VNNI** | Radeon 780M (RDNA3, 12 CU, `gfx1103`) | RDNA3 has **WMMA**; e2's HIP kernel uses **`v_dot4`, not MFMA/WMMA**. Not built here (Windows; see below) | same AVX-512 CPU path as nucleus, plus the native Windows port (`src/win32/`) | 30.4 ms |
| n100 | Intel N100, Alder Lake-N, 4C Gracemont, no SMT | **AVX-VNNI** (256-bit) | Intel UHD Graphics 0x46D1, Xe-LP, 24 EU | **no XMX/DPAS** in silicon | same AVX-VNNI CPU path as the laptop. The default DPAS GPU backend **does not build**; an **experimental text-only no-XMX path** does - see below | 329.4 ms |
| arm-server | Oracle Ampere A1 / Neoverse-N1, 2 vCPU, aarch64 | **ARMv8.2 dotprod** (`SDOT`/`UDOT`); no SVE, no i8mm | none (cloud VM) | - | CPU: `kernels_i8_neon.c` (`vdotq_s32` int8 GEMM, quantizer, packed-attention GEMM, fused softmax+quantizer), `kernels_neon.c` blocked fp32 GEMM | 380.5 ms |

Thread counts differ and they matter: 6 pinned on the laptop, 8 on nucleus and minipc, 4 on the n100, 2
on arm-server. The laptop figure is its pinned-6-thread run; at its default 14 threads the same session
measured 77.7 ms, which is why `PERFORMANCE.md` quotes the pinned number.

## Measured int8 throughput, one thread, same kernel pair

The engine carries both the 256-bit and the 512-bit int8 kernel, so the width question is measurable
rather than a matter of reading spec sheets. Shape 64x2048x512, pinned to one core.

| CPU | kernel | GMAC/s |
| :-- | :-- | --: |
| Ryzen 7 9700X (Zen 5) | `e2_i8_gemm_avx512`, zmm `vpdpbusd` | **433.2** |
| Ryzen 9 7940HS (Zen 4) | `e2_i8_gemm_avx512`, double-pumped | 298.3 |
| Core Ultra 9 285H (Arrow Lake) | `e2_i8_gemm_vnni`, ymm `vpdpbusd` | 225.8 |
| Ryzen 7 9700X, same chip, 256-bit | `e2_i8_gemm_vnni`, ymm `vpdpbusd` | 190.2 |

The last two rows on the 9700X are the width effect isolated: identical source, identical arithmetic,
only operand width and the 32-register AVX-512 file differ, and the 512-bit form is **2.28x** the
256-bit one. Zen 4 reaches ~93 % of what two 256-bit ops per cycle allows, i.e. its AVX-512 buys
instruction density and nothing else. That is the whole reason Zen 5 leads Zen 4 here, and it is why
the 1.45x kernel gap does not translate into a 1.45x embedding gap - the document GEMM is bound by
weight bandwidth, not by per-core int8 width.

## What e2 implements, by ISA

| ISA | files | what is there |
| :-- | :-- | :-- |
| x86-64 AVX512-VNNI | `kernels_i8_avx512_vnni.c`, `kernels_avx512.c` | int8 GEMM (16 output rows per zmm weight load), packed-operand attention GEMM, fused softmax+probability quantizer, fp32 16x16 sgemm tile |
| x86-64 AVX-VNNI | `kernels_i8_vnni.c`, `kernels_avx2.c` | same three int8 kernels at 256-bit, plus the AVX2 fp32 backend |
| x86-64 AVX2 only | `kernels_i8_avx2.c` | int8 GEMM via `vpmaddubsw` + `|a|`/`sign(b)` trick, quantizer, softmax+quantizer. No attention GEMM, so attention stays fp32 |
| x86-64 scalar | `kernels.c`, `int8.c` | the reference every SIMD path is checked against |
| aarch64 NEON + dotprod | `kernels_i8_neon.c`, `kernels_neon.c` | int8 GEMM, quantizer, packed-operand attention GEMM and fused softmax+quantizer via `vdotq_s32`, plus the blocked fp32 GEMM the attention products go through |
| Intel GPU (Level Zero) | `src/gpu/` | OpenCL C compiled to SPIR-V per device at build time, DPAS/XMX for the int8 GEMM and attention |
| AMD GPU (HIP) | `src/gpu/hip/` | text and vision, `v_dot4` only; built and smoke-tested on nucleus's `gfx1036` (both HIP tests pass), not performance-validated |

Dispatch order, for reference: AVX512-VNNI -> AVX-VNNI -> AVX2 -> NEON dotprod -> scalar, selected once
by `e2_cpu_init()`.

## Three things the names and flags get wrong

- **"Zen 4 has AVX-512" is true and misleading.** The 7940HS advertises `avx512f/bw/dq/vl/vnni/bf16`
  and executes every 512-bit instruction as two 256-bit uops. Same MAC/cycle as 256-bit, half the
  instructions. Only Zen 5 (nucleus) has a genuinely wide datapath. Measured above.
- **Zen 4 has AVX512-VNNI but not AVX-VNNI.** The 256-bit VEX form is Intel's (Alder Lake onwards) and
  appears again on Zen 5. Calling `e2_i8_gemm_vnni` directly on the 7940HS dies with
  `STATUS_ILLEGAL_INSTRUCTION` (0xC000001D); the engine never does, because `cpu_has_avxvnni()` asks
  `__builtin_cpu_supports("avxvnni")` first and dispatches to AVX-512 instead. Tests that call a
  backend kernel directly all guard with the matching `cpu_has_*` probe - keep it that way.
- **The n100's OpenCL device advertises `cl_khr_integer_dot_product`, but ocloc for its device id
  rejects it.** Pointing `E2_WITH_GPU=ON` at that box configures cleanly and auto-detects
  `SPIR-V for 0x46d1`, then fails at the ocloc step with `no matching function for call to 'dot'` and
  two `intel_sub_group_*_matrix_mad_*` implicit declarations. Two independent causes: Xe-LP has no
  matrix units, and whoever compiles for that device id does not accept the dot-product extension.
  The `#ifdef cl_intel_subgroup_matrix_multiply_accumulate` guard in `e2.cl` does not help, because
  ocloc defines that macro for this target while not providing the intrinsics.
  The default DPAS backend still does not build, but `-DE2_GPU_NO_XMX=ON` compiles the module with
  `-cl-std=CL3.0 -DE2_NO_DPAS`: dp4a projections plus an f32 text attention fallback, text only. It
  passes the query/document fidelity cases at 42/521 ms against 24/338 ms on the CPU and leaves three
  cores free; see PERFORMANCE.md. Batch 32x196 still exceeds the 30 s GPU wait.

## Matrix hardware none of these use

- **AMX** (Intel tile matmul, Sapphire Rapids and later server): absent from every CPU here, and e2 has
  no AMX path.
- **SVE / SVE2** (Arm): Neoverse-N1 has neither, so nothing is lost today, but e2 has no SVE kernel.
- **i8mm** (Arm, `vusdotq_s32`, the mixed u8 x i8 form that would match `vpdpbusd` directly): the N1
  lacks it, which is why the NEON kernel un-biases the activation with `xor 0x80` and uses
  `vdotq_s32`, and never reads `bsq`. A CPU with i8mm would let that kernel drop the xor and the
  accumulator correction setup entirely.
- **WMMA / MFMA** (RDNA3 and CDNA): the 780M has WMMA and e2's HIP backend does not use it.
- **XMX on Xe-LP** and **`avx512_fp16` on every CPU here**: both absent, which is why the fp16
  attention idea discussed in `PERFORMANCE.md` has no machine in this set to be tested on.

## Adding a machine

Everything in this file is reproducible:

```bash
uname -m; nproc; grep -m1 "model name" /proc/cpuinfo
grep -oE "avx512[a-z0-9_]*|avx_vnni|avx2|asimddp|sve[a-z0-9]*" /proc/cpuinfo | sort -u
clinfo | grep -iE "Device Name|Max compute units|subgroup_matrix|integer_dot_product"
lspci | grep -iE "vga|display|3d"
./build/e2_test models/<dir> tests/fixtures          # prints "cpu backend : <name>, <n> threads"
```

then add a row above and a dated block in `PERFORMANCE.md`. If the CPU is hybrid or has few cores,
pin the pool and leave gaps between runs: on the n100, back-to-back invocations overstate a document
by 1.8-2.3x because it droops from 3400 MHz to 998 MHz under sustained load.
