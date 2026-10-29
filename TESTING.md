# Testing on the five machines

Local runbook, **not committed**. `HARDWARE.md` says what each box can do; this file is only "how do I
run the tests there and what does a green run look like". Everything below was executed in this
session; numbers are from that session.

## The five

| alias | access | repo on the box | weights on disk | build quirks | ctest |
| :-- | :-- | :-- | :-- | :-- | --: |
| laptop (local) | this shell | `/home/damix/Projects/e2.c` | `models/e2` (all four `.e8cq`) + bf16 at `/home/damix/Projects/embeddinggemma-2` | `E2_MODEL_DIR` set, `E2_WITH_GPU=ON` | **43/43** |
| nucleus | ssh | `~/e2.c` | `models/e2` (text, text+image) | `E2_WITH_GPU=ON`, `E2_WITH_HIP=ON`, ROCm 7.2, `E2_HIP_ARCH=gfx1036` | **15/15** |
| n100 | ssh (root) | `/root/e2.c` | `models/e2` (text only) | `E2_WITH_GPU=OFF` (Xe-LP cannot build the kernels), **no git** | 14/15 |
| arm-server | ssh (ubuntu) | `~/e2.c` | `models/embeddinggemma-2` (bf16 checkpoint) | Release only, 2 vCPU | **15/15** |
| minipc | ssh (Windows) | `C:\Users\damix\e2` | `models/e2` (text, text+image) | Ninja + clang + MSVC, no git, no GPU backends | **13/13** |

The counts differ because of configuration, not health: 43 = 15 base + 11 model/fixture tests
(`-DE2_MODEL_DIR`) + 17 Intel GPU tests; 15 = the base suite on a Linux host without a model dir;
13 = base suite minus the shell-script test on Windows. n100's one failure is `artifacts`, which
shells out to `git` and that box has none — pre-existing and unrelated.

## SSH: always disable the agent

The local agent holds FIDO/security keys that these servers accept and then block on, so a plain `ssh`
hangs. Every command needs:

```bash
ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
    -i ~/.ssh/id_ed25519_pi_<host> <host> '<command>'
```

| host | key |
| :-- | :-- |
| nucleus | `~/.ssh/id_ed25519_pi_nucleus` |
| n100 | `~/.ssh/id_ed25519_pi_n100_debian` |
| arm-server | `~/.ssh/id_ed25519_pi_arm_server` |
| minipc | `~/.ssh/id_ed25519_pi_minipc_windows` |

Probe all four at once:

```bash
for h in "nucleus id_ed25519_pi_nucleus" "n100 id_ed25519_pi_n100_debian" \
         "arm-server id_ed25519_pi_arm_server" "minipc id_ed25519_pi_minipc_windows"; do
  set -- $h; echo "=== $1 ==="
  timeout 20 ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
    -i ~/.ssh/$2 $1 'uname -n; nproc' 2>&1 | head -3
done
```

## Syncing the tree

There is no `rsync` on the hosts. Use tar over ssh from the repo root. Exclude `build*` (all of them),
`models/` (host-specific, large), `.git/`, `benchmark-results/` (**2.3 GB here**), `tests/ref_act/`
(218 MB) and `*.e8cq`, or the sync takes minutes and ships nothing useful:

```bash
# Linux hosts
tar czf - --exclude='build*' --exclude=models --exclude=.git \
         --exclude=benchmark-results --exclude=tests/ref_act --exclude='*.e8cq' . \
  | ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
        -i ~/.ssh/id_ed25519_pi_nucleus nucleus 'tar xzf - -C e2.c'

# Windows (note: the checkout is `e2`, not `e2.c`)
tar czf - --exclude='build*' --exclude=models --exclude=.git \
         --exclude=benchmark-results --exclude=tests/ref_act --exclude='*.e8cq' . \
  | ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
        -i ~/.ssh/id_ed25519_pi_minipc_windows minipc 'tar -xzf - -C C:\Users\damix\e2'
```

Check what a sync would send before running it (should be a few MB):

```bash
tar czf - --exclude='build*' --exclude=models --exclude=.git \
         --exclude=benchmark-results --exclude=tests/ref_act --exclude='*.e8cq' . | wc -c
```

Then reconfigure only if `CMakeLists.txt` changed (the build dirs are already configured):

```bash
ssh ... nucleus 'cd e2.c && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release'
```

If a remote tree carries local changes you want to keep, sync individual files instead — the file list
that usually matters is `src/`, `tests/`, `tools/bench/`, `CMakeLists.txt`, `include/`.

## Laptop (AVX-VNNI int8, AVX2 fp32, Intel Arc via Level Zero)

```bash
cd /home/damix/Projects/e2.c
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DE2_MODEL_DIR=/home/damix/Projects/embeddinggemma-2   # needed for the fixture tests
cmake --build build -j"$(nproc)"
(cd build && ctest -j4)                                     # 43/43, ~95 s

# fidelity: int8 via the packed cache, pinned for comparable numbers
export E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq
E2_THREADS=6 taskset -c 0-5 ./build/e2_test models/e2 tests/fixtures
E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text-image.e8cq \
  E2_THREADS=6 taskset -c 0-5 ./build/e2_test_image models/e2 tests/fixtures
E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text-audio.e8cq \
  E2_THREADS=6 taskset -c 0-5 ./build/e2_test_audio models/e2 tests/fixtures

# backend sweep and per-kernel rates (pin: an unpinned run measures an E-core)
E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq \
  E2_THREADS=6 ./build/e2_text_bench models/e2 tests/fixtures backends 196 1 5
taskset -c 0 ./build/e2_kernel_bench backends 5
```

This is the only machine with a working Level Zero device; `ctest` covers it through the `gpu_*`
tests (they skip with 77 when there is no device).

## nucleus (AVX512-VNNI int8, AVX-512 fp32, HIP on gfx1036)

```bash
R="ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
   -i ~/.ssh/id_ed25519_pi_nucleus nucleus"

$R 'cd e2.c && cmake --build build -j8'
$R 'cd e2.c/build && ctest -j8'                             # 15/15, ~1.5 s
$R 'cd e2.c && ./build/e2_test_dispatch'

# fidelity, packed cache (text and text+image are on disk; text+audio is not)
$R 'cd e2.c && E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq \
    E2_THREADS=6 ./build/e2_test models/e2 tests/fixtures'
$R 'cd e2.c && E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text-image.e8cq \
    E2_THREADS=6 ./build/e2_test_image models/e2 tests/fixtures'

# the three CPU backends this box can run, one process
$R 'cd e2.c && E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq \
    E2_THREADS=6 ./build/e2_text_bench models/e2 tests/fixtures backends 196 1 5'
$R 'cd e2.c && taskset -c 0 ./build/e2_kernel_bench backends 5'

# HIP: only if ROCm is present; the backend lives in its own build dir
$R 'cd e2.c && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DE2_WITH_HIP=ON -DE2_HIP_ARCH=gfx1036'
$R 'cd e2.c && E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq \
    ./build/e2_test_hip_text models/e2 tests/fixtures'
$R 'cd e2.c && E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text-image.e8cq \
    ./build/e2_test_hip_image models/e2 tests/fixtures'
```

A green HIP run is cosine ~0.9998 (text query), 0.9996 (document), 0.9998 (image). The 2-CU iGPU is
slower than this CPU on every shape — that is expected, not a regression.

## n100 (AVX-VNNI int8, AVX2 fp32, experimental no-XMX GPU only, no git)

```bash
R="ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
   -i ~/.ssh/id_ed25519_pi_n100_debian n100"

$R 'cd e2.c && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DE2_WITH_GPU=OFF'
$R 'cd e2.c && cmake --build build -j4'
$R 'cd e2.c/build && ctest -j2'                             # 14/15: artifacts fails, no git
$R 'cd e2.c && E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq \
    ./build/e2_test models/e2 tests/fixtures'

# no machine in the set is AVX2-only, so simulate one to exercise the maddubs kernels
$R 'cd e2.c && E2_CPU_BACKEND=avx2 E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq \
    ./build/e2_test models/e2 tests/fixtures'

# experimental text-only no-XMX GPU path (default OFF; DP4A projections + f32 attention)
$R 'cd e2.c && cmake -S . -B build-noxmx -DCMAKE_BUILD_TYPE=Release -DE2_WITH_GPU=ON -DE2_GPU_NO_XMX=ON'
$R 'cd e2.c && cmake --build build-noxmx -j4'
$R 'cd e2.c && E2_Q8_READ=$PWD/models/e2/e2-embeddinggemma-2-int8-text.e8cq \
    ./build-noxmx/e2_test_gpu_text models/e2 tests/fixtures'
```

The default DPAS build must stay off here: ocloc rejects the kernels for this device (HARDWARE.md
records both causes), so a plain `E2_WITH_GPU=ON` build fails at the SPIR-V step. The experimental
`-DE2_GPU_NO_XMX=ON` path is the text-only exception: it compiles with `-cl-std=CL3.0`, drops every
DPAS kernel, and defaults to DP4A projections plus f32 attention. Expect query ~42 ms, document
~521 ms in a spaced run, cos 0.9997 / 0.9996. The test binary skips the batch section in this build
(`E2_GPU_TEST_SINGLE`); the device-readable text cache is enough, no checkpoint is needed.

## arm-server (NEON dotprod, bf16 checkpoint, 2 vCPU)

```bash
R="ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
   -i ~/.ssh/id_ed25519_pi_arm_server arm-server"

$R 'cd e2.c && cmake --build build -j2'
$R 'cd e2.c/build && ctest -j2'                             # 15/15, ~13 s
$R 'cd e2.c && ./build/e2_test_dispatch'
$R 'cd e2.c && E2_THREADS=2 ./build/e2_test models/embeddinggemma-2 tests/fixtures'
$R 'cd e2.c && E2_THREADS=2 ./build/e2_test_image models/embeddinggemma-2 tests/fixtures'  # ~8.5 s/image
$R 'cd e2.c && E2_THREADS=2 ./build/e2_test_audio models/embeddinggemma-2 tests/fixtures'
$R 'cd e2.c && ./build/e2_text_bench models/embeddinggemma-2 tests/fixtures backends 15 1 2'
$R 'cd e2.c && taskset -c 0 ./build/e2_kernel_bench backends 5'
```

This box has no packed int8 cache, so it loads the bf16 checkpoint and quantizes at load (that is
also why the scalar/bf16 row of a backend sweep is runnable here and not elsewhere). `e2_cpu_name()`
prints `neon`; if it prints `scalar` you are running an old binary.

## minipc (Zen 4 AVX512-VNNI, Windows)

PowerShell is the login shell; `tar.exe` is in `C:\WINDOWS\system32`.

```bash
R="ssh -o BatchMode=yes -o IdentityAgent=none -o IdentitiesOnly=yes \
   -i ~/.ssh/id_ed25519_pi_minipc_windows minipc"

$R 'cd C:\Users\damix\e2; cmake -S . -B build; cmake --build build -j8'
$R 'cd C:\Users\damix\e2\build; ctest -j4'                  # 13/13, ~6 s
$R 'cd C:\Users\damix\e2; .\build\e2_test_dispatch.exe'
$R 'cd C:\Users\damix\e2; $env:E2_Q8_READ="$PWD\models\e2\e2-embeddinggemma-2-int8-text.e8cq"; \
    .\build\e2_test.exe models\e2 tests\fixtures'
$R 'cd C:\Users\damix\e2; $env:E2_Q8_READ="$PWD\models\e2\e2-embeddinggemma-2-int8-text.e8cq"; \
    .\build\e2_text_bench.exe models\e2 tests\fixtures backends 196 1 3'
```

Windows has no `taskset`; for a timing-sensitive run pin with
`cmd /c start /affinity 1 /wait .\build\e2_kernel_bench.exe cpu 5`. GPU backends and the shell-script
test are not built here, which is the whole of the 13-vs-15 difference.

## What to actually look at

| command | green means |
| :-- | :-- |
| `ctest` | the counts above; anything else is a real failure except n100 `artifacts` |
| `e2_test_dispatch` | `dispatch matrix: OK` — the whole backend×tower matrix, on any host |
| `e2_test` / `_image` / `_audio` | every line `PASS`, cosine ≥ ~0.9996 text, ~0.9997 image, ~0.9996 audio |
| `e2_test_hip_*` | `PASS`, cosine ≥ 0.9996 |
| `e2_text_bench ... backends` | one JSON line per backend the host can run, `minimum_cosine` ≥ 0.999 |
| `e2_kernel_bench` | one line per kernel, `check:"ok"` for the two GEMMs, rates near the tables in PERFORMANCE.md |

## Pitfalls that cost time

- **Pin every timing run.** On the laptop an unpinned vnni GEMM reads 63 GMAC/s (E-core) where the
  pinned one reads 229.4. Use `taskset -c 0` for the kernel bench and `taskset -c 0-5` +
  `E2_THREADS=6` for engine runs.
- **Reconfigure after `CMakeLists.txt` changes**, otherwise new targets (e.g. `e2_kernel_bench`) are
  simply missing and the test count silently moves.
- **A packed cache is int8-only.** `E2_Q8_READ` plus a model dir containing only `.e8cq` cannot serve
  the scalar/bf16 backend; the backend sweep skips that row with a message. To test bf16, point the
  model dir at a directory with `model.safetensors` and do not set `E2_Q8_READ`.
- **`E2_CPU_BACKEND` can only downgrade.** `avx512` on the laptop is refused (no AVX-512); `vrni` is a
  typo and is refused; both leave the best backend in place and print why.
- **The minipc checkout is `e2`, not `e2.c`**, and it is not a git repository — `ctest`'s `artifacts`
  test is not built there.
- **nucleus's `E2_WITH_HIP=ON` in the cache does not mean HIP is built** on other hosts; check for
  `build/e2_test_hip_text` before running it.
- **First image run includes cold packing** (`attn_pack` 88 ms vs 31 ms warm); compare warm numbers
  or run once to warm up before timing.
