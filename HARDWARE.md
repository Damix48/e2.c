# e2 - what it runs on

This file is the hardware and dependency contract: which instruction sets the kernels need, which GPU
backends exist and how much to trust them, and which third-party compute libraries were measured and
rejected. Timings below are from the reference machine (Intel Core Ultra 9 285H, 16 cores, no AVX-512)
and are subject to the caveats at the top of [PERFORMANCE.md](PERFORMANCE.md): the box throttles, so
numbers are best-of-N and comparisons were taken by alternating in one session.

The model side is fixed: one checkpoint (EmbeddingGemma-2), text / text+image / text+audio /
text+image+audio, 768-d L2-normalized output, 2048 ids, 2520 patches (280 soft tokens), audio up to
30 s. Video is out of scope.

## CPU

Dispatch is a single CPUID probe in `e2_cpu_init` (`src/kernels.c`). The int8 GEMM and the fp32
kernels are chosen independently, so a machine can run 512-bit float math with 256-bit int8 math.

| backend | needs at run time | source | CPUs | rate, 1 thread |
| :-- | :-- | :-- | :-- | --: |
| int8 AVX512-VNNI | `avx512f` `bw` `vl` `dq` `vnni` | `src/kernels_i8_avx512_vnni.c` | Cascade Lake+ / Ice Lake client / Tiger Lake / Rocket Lake, Sapphire Rapids+, **AMD Zen 4 / Zen 5** | not measured here (this box has no AVX-512) |
| int8 AVX-VNNI | `avx2` `fma` `avxvnni` | `src/kernels_i8_vnni.c` | **Alder Lake 12th gen (2021)**, Raptor Lake 13/14th, Meteor Lake (Core Ultra 100), Arrow Lake (Ultra 200H/S), Lunar Lake (Ultra 200V) | **211 GMAC/s** |
| int8 AVX2 | `avx2` `fma` | `src/kernels_i8_avx2.c` (`vpmaddubsw`) | Haswell (2013) and later, Comet/Coffee Lake, **AMD Zen 1-3**, Skylake-SP (AVX-512 but no VNNI) | **53 GMAC/s** |
| int8 scalar | nothing | `src/int8.c` | the reference every SIMD path is tested against | ~2 GMAC/s |
| fp32 AVX-512 | `avx512f` `bw` `vl` `dq` | `src/kernels_avx512.c` | the VNNI list above plus Skylake-SP | - |
| fp32 AVX2 | `avx2` `fma` | `src/kernels_avx2.c` | Haswell and later | - |
| fp32 scalar | nothing | `src/kernels.c` | always compiles, always correct | - |

Notes that are not obvious from the table:

* **Hybrid Intel is safe.** Every E-core since Gracemont (12th gen) implements AVX-VNNI, so the
  256-bit `vpdpbusd` path cannot hit an illegal instruction when the scheduler migrates a worker
  thread. Client parts from 12th gen onward have no AVX-512 at all, which is why the AVX-VNNI row is
  the interesting one for laptops.
* **Pre-2013 x86 runs, but on the slow path.** `e2_i8_quantize` is NULL without AVX2, so
  `use_int8` is forced off and the engine takes the bf16 path with the *scalar* fp32 GEMM. Measured
  by forcing that backend: **2274 ms** for the 196-token document against 38 ms on the int8 path.
  Correct output, 60x slower - this is the same path an ARM machine would get if the tree built there.
* **The pool sizes itself to physical cores** and drops cores below 75 % of the fastest core's peak
  frequency, which is what makes P/E hybrids behave. `E2_THREADS=n` overrides (1 disables the pool).
* **AVX512-VNNI CPU correctness was validated on Ryzen 7 9700X (Zen 5)** with GCC 14.2.0,
  including the full checkpoint/sanitizer/cache gate and 1/4/8-thread runs. See
  [PRODUCTION.md](PRODUCTION.md) for the tested revision and fidelity. Kernel throughput was not
  benchmarked on that host; this does not validate the separate AMD HIP backend.
* No AMX (Sapphire Rapids' matrix units are unused), no int4/int2 weights, no AVX-512 bf16/fp16: the
  bf16 path is a plain fp32 GEMM over bf16 operands.

## Non-x86: does not build today

`src/kernels.c` calls `__builtin_cpu_init`/`__builtin_cpu_supports` unconditionally and references
`e2_avx2_kernels()`/`e2_avx512_kernels()`, whose translation units CMake only adds for `x86_64`.
Cross-compiling to aarch64 fails at compile time on the builtins and would fail at link time on those
symbols. There is no NEON, SDOT, SVE or RVV code in the tree.

A port needs three things, in this order of size: an `#ifdef` around the dispatch, a quantizer and an
int8 GEMM built on `vdotq_s32` against the existing quad-interleaved layout (`[N/8][K/4][8][4]`, see
`src/int8.h`) - roughly the size of `kernels_i8_vnni.c`, 249 lines - and either an fp32 kernel set or
acceptance of the scalar elementwise ops. The tokenizer, image preprocessing, thread pool, attention
scheduling and pooling are already portable C11.

## GPU

| backend | hardware | towers | state |
| :-- | :-- | :-- | :-- |
| **Intel Level Zero** (`src/gpu/`) - OpenCL C compiled to SPIR-V at build time by `ocloc` | Xe family: Arc A (Alchemist), Arc B (Battlemage), Iris Xe (Tiger Lake+), Core Ultra iGPUs (Meteor/Arrow/Lunar; SPIR-V target defaults to `xe-lpg`, override with `E2_GPU_DEVICE`) | text, vision, audio | **experimental**. Intermittent wrong or zero outputs; one 129-token golden case scores 0.998971, below the 0.999 gate; the API can return success with a wrong result. Cause not established. |
| **AMD HIP** (`src/gpu/hip/`) - built for one ISA from `rocminfo`, else `gfx1036` | RDNA/CDNA with ROCm and `hipcc` | text, vision | **never executed on AMD hardware here.** Source-reviewed only; acceptance outstanding. |

Where the Intel GPU lands (same session, same inputs, [PERFORMANCE.md](PERFORMANCE.md#the-intel-gpu)):

| workload | CPU, 14 thr | Intel GPU | verdict |
| :-- | --: | --: | :-- |
| text query, 15 tok | **4.7 ms** | 7.1 ms | CPU wins |
| text document, 196 tok | 34.7 ms | **13.5 ms** | GPU 2.6x |
| text batch 8 x 17 | - | **9.4 ms** (854 seq/s) | GPU |
| image, 2520 patches | 716 ms | **189 ms** | GPU 3.8x |
| text batch 32 x 196 | 2005 ms (OpenVINO CPU) | **392 ms** | GPU, 1.4x behind OpenVINO's GPU plugin |

There is no CUDA, Metal, Vulkan or OpenCL-ICD path. A new backend is a new `src/gpu/<name>/` plus
entry points mirroring `include/e2gpu.h`.

## Everything else

* **OS**: any POSIX with pthreads and libm - `mmap`, futex-based pool, `clock_gettime`. Linux is what
  is tested; there is no Windows port (`src/st.c` is mmap-based).
* **Build**: CMake only, C11. Optional dependencies, each dropped automatically when missing:
  libpng (an in-tree PNG + DEFLATE reader is always compiled as the fallback), libjpeg-turbo,
  libwebp, the Level Zero loader + `ocloc`, ROCm/`hipcc`.
* **Footprint**: 1420 MB bf16 checkpoint (mmap, demand-paged) or 916 MB packed int8 cache (410 MB
  text-only); load 0.52 s vs 0.01 s; peak RSS for a text query 884 MB vs 180 MB; workspace 233 MB.
* **Process model**: one live model per process, not reentrant, no threads beyond the internal pool.

## Third-party compute libraries: measured, and why not

The custom kernels are the part of this engine most likely to be questioned, so this section records
what was actually tried. All of it was measured head-to-head against the in-tree kernel on the
reference machine - same weights, same activations, same shapes - in throwaway spike branches that
are **not** merged. None of them is a dependency of this tree.

The reason they all lose is structural, not a matter of kernel quality: the weight GEMM is bound by
**weight bytes**, not by FLOPs. int8 is already one byte per weight; every candidate below is an fp32
or fp32-dot library, so it reads 4x the bytes on every architecture and no amount of blocking fixes a
byte count. On the 15-token query the int8 path is already streaming ~130 MB in 4.5 ms, i.e. at memory
bandwidth.

### OpenBLAS (`cblas_sgemm`)

Two separate swaps, measured independently.

| | stock engine | attention sgemm -> BLAS | weight GEMM -> BLAS (fp32 weights) |
| :-- | --: | --: | --: |
| 15-token query | 4.4-4.9 ms | 4.5-4.9 ms (flat) | 19 ms (**4.2x slower**) |
| 196-token document | 38.1-38.9 ms | 36.6-45.3 ms (noisy, neutral) | 110-121 ms (**2.9x slower**) |
| 514-token document | 119-128 ms | 112-124 ms (**5-7 % faster**) | - |
| 2048-token document | 723-769 ms | 644-689 ms (**7-9 % faster**) | - |
| batch 8 x 196 | 274-307 ms | 269-288 ms | 1368 ms (**4.5x slower**) |
| image, 2520 patches | 662 ms | 611 ms (**8 % faster**) | - |
| image + caption | 723 ms | 624 ms (**14 % faster**) | - |

Fidelity is unchanged in every mode (cosine 0.9997-0.9998 against the fixtures), which is expected:
the swap keeps the arithmetic and changes only who runs it.

**Why not for the weight GEMM**: 3-4x slower. OpenBLAS has no int8 GEMM (`u8s8s32`), and an fp32
GEMM must move 4x the weight bytes; also 4x RSS (text tower 130 MB int8 -> 520 MB fp32) and the
packed `.e8cq` cache becomes pointless. **Why not for the attention GEMM either**: it is a real 5-14 %
win, but only on the fallback path, and only at long sequences and on the image; it buys one more
dependency, a nested-threadpool hazard (BLAS must be pinned to one thread because the engine's pool
owns the parallelism) and 4x weight memory for a 0 % change on queries and batches. If this is ever
wanted it belongs behind an `E2_WITH_OPENBLAS=AUTO` option modelled on the image codecs, applied to
`e2_sgemm_nn_range` only. Copying the idea instead - packing `B` in the AVX2 sgemm - reaches the same
place without the dependency, and that is the recommended path.

### Intel oneDNN

Functionally the closest match: `dnnl_gemm_u8s8s32` is this engine's recipe (unsigned activations
with a `+128` offset, signed weights, int32 accumulate, per-output-row weight scales, and a `co[]`
correction array for the zero point). Apache-2.0, C API over C++.

Not adopted, and not measured locally (no `dnnl` on the reference image):

* **The per-128-block activation scale does not fit.** The legacy `dnnl_gemm_*` takes scalar scales,
  so the choice is a per-row activation scale - which the project already measured at 0.9989 cosine,
  below its own 0.999 bar - or K/128 accumulate passes with 4x the output traffic. The matmul
  primitive may express per-K source scales, which would fix this; that is the one API question to
  answer before spending a day on a harness.
* **It brings its own threadpool** (TBB or OpenMP), which collides with the persistent pool exactly as
  OpenBLAS does, and it is strongest at large batched brgemm/AMX work, not at the 8-64 column slivers
  the engine's claim-based pool produces.
* **Its ceiling is visible from here**: OpenVINO's int8 plugin is oneDNN underneath, and this engine
  is 1.03-1.3x faster on the audio tower and ~2x on the image on the same machine.

Verdict: the only candidate worth a microbenchmark rather than an integration. The test is
`dnnl_gemm_u8s8s32` against `e2_i8_gemm_vnni_range` on M = 8/200/2048, K = 512/2048, N = 512/2048/8192,
weights resident, one thread and N threads; if oneDNN is not >= 1.3x on the GEMM alone, no wrapper can
be.

### XNNPACK

Measured, because it is the only candidate with real multi-architecture int8 coverage (AVX-VNNI,
AVX512-VNNI, NEON SDOT, i8mm, SVE, plus WASM/RISC-V) and therefore the only one that could substitute
for writing an ARM kernel. Built from source; `fully_connected_nc_qd8_f32_qc8w` (int8 activations,
int8 weights, per-channel weight scales) produces bit-identical results to the in-tree kernel
(cosine 1.00000 on every shape), so it is a true head-to-head.

| M x K x N | e2, 1 thr | e2, 14 thr | XNNPACK, 1 thr | XNNPACK, 14 thr | XNNPACK f32_qc8w |
| :-- | --: | --: | --: | --: | --: |
| 8 x 512 x 8192 | 0.55 | **0.042** | 0.40 | 0.063 | 1.31 |
| 8 x 512 x 2048 | 0.036 | **0.010** | 0.098 | 0.033 | 0.33 |
| 200 x 512 x 8192 | 3.88 | **0.56** | 10.25 | 1.87 | 26.3 |
| 200 x 512 x 2048 | 0.92 | **0.14** | 2.50 | 0.42 | 6.62 |
| 200 x 2048 x 512 | 0.89 | **0.43** | 2.38 | 0.50 | 6.67 |
| 2048 x 512 x 2048 | 9.99 | **1.65** | 25.3 | 3.63 | 67.7 |
| 2048 x 2048 x 512 | 9.78 | 4.31 | 24.9 | **3.26** | 68.3 |

(ms, best of 7; the engine side uses the real pool, claiming 64-column tiles as `text.c` does.)

* **2-3.3x slower** on every text-tower shape except the deep-K `down_proj` at 2048 tokens, where it
  wins 1.3x - that one shape is the useful finding, and it is about blocking, which can be fixed
  in-tree.
* Single-threaded it sustains **83 GMAC/s** against the engine's 211 (and the engine's own AVX2 int8
  kernel's 53), i.e. it is on an int8 path but a weaker loop than the quad-interleaved
  8-rows-per-32-byte-load form.
* The weight-only variant (`f32_qc8w`, fp32 activations) is 4-7x slower, so "keep int8 weights, feed
  fp32 activations" is not an escape hatch.
* **It costs fidelity.** XNNPACK exposes at best one activation scale per batch row; the per-128-block
  scale that gets this engine to 0.9994 is not expressible, so the recipe would drop to the 0.9989
  configuration.
* **It costs structure.** XNNPACK splits N across its own threadpool, so `gemm_segs`' claim-a-tile
  design goes away, and a per-claim call carries ~3 us of overhead (measured at M=8, K=512, N=64) -
  about 10 ms of pure overhead on a 15-token query.

Verdict: no on x86. On ARM it remains the only zero-kernel-work option and nothing above speaks to it;
the way to settle that is the same bench on real ARM hardware, before any integration.

### FBGEMM

Rejected without measurement: maintenance mode, x86-only (AVX2/AVX512, no ARM/AArch64 target),
C++ with heavy build requirements. It is the one candidate that offers nothing on either axis, since
portability is the reason to look at this class of library in the first place.

### What would actually be adopted from this exercise

1. The ARM int8 kernel (`vdotq_s32`) - the only real answer to "more architectures", one file.
2. Optionally, BLAS for the attention fp32 GEMM only, behind an `AUTO`-detected build option, if the
   5-14 % on long documents and images is worth a dependency to someone. Not for the projections.
3. The blocking lesson from the `2048 x 2048 x 512` row, applied to `kernels_i8_vnni.c` after a
   profile shows the same loss there.
