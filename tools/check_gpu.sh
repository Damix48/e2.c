#!/bin/sh
# Intel GPU software gate. No skipped device checks, retries-until-green, or diagnostic overrides.
# Usage: tools/check_gpu.sh MODEL_DIR [BUILD_ROOT] [SOAK_SECONDS]
set -eu
ROOT=$(CDPATH='' cd -- "$(dirname "$0")/.." && pwd)
MODEL=${1:?usage: tools/check_gpu.sh MODEL_DIR [BUILD_ROOT] [SOAK_SECONDS]}
BUILD=${2:-$ROOT/build-gpu-release-check}
SOAK_SECONDS=${3:-300}
case "$SOAK_SECONDS" in ''|0*|*[!0-9]*) echo 'SOAK_SECONDS must be 1..86400' >&2; exit 1 ;; esac
[ "$SOAK_SECONDS" -ge 1 ] && [ "$SOAK_SECONDS" -le 86400 ] || { echo 'SOAK_SECONDS must be 1..86400' >&2; exit 1; }
MODEL=$(CDPATH='' cd -- "$MODEL" && pwd)
[ -f "$MODEL/model.safetensors" ] && [ -f "$MODEL/tokenizer.json" ] || {
    echo 'checkpoint and tokenizer are required' >&2; exit 1;
}
command -v timeout >/dev/null || { echo 'GNU timeout is required for the worker deadline' >&2; exit 1; }
mkdir -p "$BUILD"
BUILD=$(CDPATH='' cd -- "$BUILD" && pwd)
# Preserve only device selection; truncated graphs, alternate kernels, bf16, caches and dumps cannot certify a release.
DEVICE=${E2_GPU:-}
for name in $(env | grep '^E2_' | cut -d= -f1); do unset "$name"; done
if [ -n "$DEVICE" ]; then E2_GPU=$DEVICE; export E2_GPU; fi
JOBS=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
GPU_TESTS='gpu_copy gpu_profile_fallback gpu_l0_runtime gpu_attn_recipe gpu_text gpu_image gpu_audio gpu_text_reference gpu_validation gpu_shapes gpu_runtime gpu_lifecycle gpu_audio_reference gpu_soak'

check() {
    label=$1; dir=$2; shift 2
    rc=0
    ctest --test-dir "$dir" --no-tests=error --timeout 180 --output-on-failure "$@" >"$BUILD/$label.log" 2>&1 || rc=$?
    cat "$BUILD/$label.log"
    if [ -f "$dir/Testing/Temporary/LastTest.log" ]; then
        cp "$dir/Testing/Temporary/LastTest.log" "$BUILD/$label.details.log"
    fi
    [ "$rc" = 0 ] || return "$rc"
    if grep -E 'Skipped|Not Run|No tests were found' "$BUILD/$label.log" >/dev/null; then
        echo 'GPU release gate refuses skipped or missing checks' >&2; return 1;
    fi
}
require_gpu() {
    ctest --test-dir "$1" -N >"$BUILD/registered-tests.log"
    for test in $GPU_TESTS; do
        grep -E ": $test$" "$BUILD/registered-tests.log" >/dev/null || {
            echo "required Intel GPU test missing: $test (backend/toolchain disabled?)" >&2; exit 1;
        }
    done
}
cmake -S "$ROOT" -B "$BUILD/release" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    -DE2_WITH_GPU=ON -DE2_WITH_HIP=OFF -DE2_MODEL_DIR="$MODEL" -DE2_FIXTURE_DIR="$ROOT/tests/fixtures"
cmake --build "$BUILD/release" -j "$JOBS"
require_gpu "$BUILD/release"
{
    date -u
    uname -a
    (cd "$ROOT" && git rev-parse HEAD && git status --short)
    "${CC:-cc}" --version
    cmake --version
    env | LC_ALL=C sort | grep -E '^(E2_GPU=|ZE_|ZET_|NEO_|IGC_|LD_LIBRARY_PATH=|LD_PRELOAD=)' || true
    grep -E '^(CMAKE_(C_COMPILER|C_FLAGS|BUILD_TYPE)|E2_(GPU_DEVICE|OCLOC|ZE_))' "$BUILD/release/CMakeCache.txt" || true
    command -v ocloc >/dev/null && (ocloc --version || true)
    if command -v lscpu >/dev/null; then lscpu; fi
    if command -v lspci >/dev/null; then lspci -nnk; fi
    if command -v pacman >/dev/null; then
        pacman -Q intel-compute-runtime intel-graphics-compiler level-zero-loader || true
    elif command -v dpkg-query >/dev/null; then
        dpkg-query -W '*level-zero*' '*igc*' '*intel-opencl*' '*libze*' 2>/dev/null || true
    fi
    ldd "$BUILD/release/e2_test_gpu_soak"
} >"$BUILD/environment.log" 2>&1
sha256sum "$MODEL/model.safetensors" "$MODEL/tokenizer.json" >"$BUILD/checkpoint.sha256"
(cd "$ROOT" && find include src tests tools cmake .github/workflows -type f -exec sha256sum {} + &&
    sha256sum CMakeLists.txt) >"$BUILD/source.sha256"
for n in 1 4; do E2_THREADS=$n check "release-$n" "$BUILD/release"; done

cmake -S "$ROOT" -B "$BUILD/sanitized" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    -DE2_WITH_GPU=ON -DE2_WITH_HIP=OFF -DE2_MODEL_DIR="$MODEL" -DE2_FIXTURE_DIR="$ROOT/tests/fixtures" \
    -DE2_WITH_PNG=OFF -DE2_WITH_JPEG=OFF -DE2_WITH_WEBP=OFF -DE2_WITH_SNDFILE=OFF \
    -DCMAKE_C_FLAGS='-g -fsanitize=address,undefined -fno-omit-frame-pointer' \
    -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build "$BUILD/sanitized" -j "$JOBS"
require_gpu "$BUILD/sanitized"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 E2_THREADS=4 \
    check sanitized "$BUILD/sanitized"

E2_THREADS=4 E2_FIXTURE_DIR="$ROOT/tests/fixtures" "$ROOT/tools/export_models.sh" "$MODEL" "$BUILD/models" "$BUILD/release"
(cd "$BUILD/models" && sha256sum -c manifest.sha256)
for variant in text text-image text-audio text-image-audio; do
    tests='text|text_reference|lifecycle'
    case "$variant" in *image*) tests="$tests|image" ;; esac
    case "$variant" in *audio*) tests="$tests|audio|audio_reference" ;; esac
    E2_Q8_READ="$BUILD/models/e2-embeddinggemma-2-int8-$variant.e8cq" E2_THREADS=4 \
        check "cache-$variant" "$BUILD/release" -R "^gpu_($tests)$"
done
# A separate long-lived process, not repeated short binaries. External deadline also bounds driver calls.
rc=0
E2_THREADS=4 timeout --signal=TERM --kill-after=10s "$((SOAK_SECONDS + 180))s" \
    "$BUILD/release/e2_test_gpu_soak" "$MODEL" "$ROOT/tests/fixtures" "$SOAK_SECONDS" \
    >"$BUILD/soak.log" 2>&1 || rc=$?
cat "$BUILD/soak.log"
[ "$rc" = 0 ] || exit "$rc"
echo 'Intel GPU software gate passed. This is not certification of other GPUs, drivers, or application quality.'
echo "Evidence: $BUILD (environment, source/checkpoint hashes, test details, caches and soak latency/RSS)."
