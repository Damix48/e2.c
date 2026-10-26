#!/bin/sh
# Complete local CPU release gate. Requires the checkpoint; never silently drops fidelity tests.
# Usage: tools/check_cpu.sh MODEL_DIR [BUILD_ROOT]
set -eu
ROOT=$(CDPATH='' cd -- "$(dirname "$0")/.." && pwd)
MODEL=${1:?usage: tools/check_cpu.sh MODEL_DIR [BUILD_ROOT]}
BUILD=${2:-$ROOT/build-release-check}
MODEL=$(CDPATH='' cd -- "$MODEL" && pwd)
[ -f "$MODEL/model.safetensors" ] && [ -f "$MODEL/tokenizer.json" ] || {
    echo 'checkpoint and tokenizer are required' >&2; exit 1;
}
mkdir -p "$BUILD"
BUILD=$(CDPATH='' cd -- "$BUILD" && pwd)
unset E2_Q8_READ E2_Q8_WRITE E2_Q8_TOWERS E2_INT8 E2_DUMP E2_PROFILE E2_SKIP_VERIFY
JOBS=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
cmake -S "$ROOT" -B "$BUILD/release" -DCMAKE_BUILD_TYPE=Release \
    -DE2_WITH_GPU=OFF -DE2_WITH_HIP=OFF -DBUILD_TESTING=ON -DE2_MODEL_DIR="$MODEL"
cmake --build "$BUILD/release" -j "$JOBS"
for n in 1 4; do E2_THREADS=$n ctest --test-dir "$BUILD/release" --output-on-failure; done
E2_INT8=0 E2_THREADS=4 ctest --test-dir "$BUILD/release" --output-on-failure \
    -R '^(text|text_batch|text_reference|image|audio|audio_reference|validation)$'
cmake -S "$ROOT" -B "$BUILD/sanitized" -DCMAKE_BUILD_TYPE=Release \
    -DE2_WITH_GPU=OFF -DE2_WITH_HIP=OFF -DBUILD_TESTING=ON -DE2_MODEL_DIR="$MODEL" \
    -DE2_WITH_PNG=OFF -DE2_WITH_JPEG=OFF -DE2_WITH_WEBP=OFF \
    -DCMAKE_C_FLAGS='-g -fsanitize=address,undefined -fno-omit-frame-pointer' \
    -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build "$BUILD/sanitized" -j "$JOBS"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 E2_THREADS=4 \
    ctest --test-dir "$BUILD/sanitized" --output-on-failure
E2_THREADS=4 E2_FIXTURE_DIR="$ROOT/tests/fixtures" "$ROOT/tools/export_models.sh" "$MODEL" "$BUILD/models" "$BUILD/release"
(cd "$BUILD/models" && sha256sum -c manifest.sha256)
E2_Q8_READ="$BUILD/models/e2-embeddinggemma-2-int8-text-image-audio.e8cq" E2_THREADS=4 \
    ctest --test-dir "$BUILD/release" --output-on-failure \
    -R '^(text|text_batch|text_reference|image|audio|audio_reference|validation)$'
echo 'CPU software gate passed. Record CPU/ISA coverage and application retrieval acceptance per PRODUCTION.md.'
