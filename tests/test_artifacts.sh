#!/bin/sh
# Offline failure-path tests: no network, checkpoint or Python.
set -eu
ROOT=$1
unset E2_SKIP_VERIFY E2_Q8_READ E2_Q8_WRITE E2_Q8_TOWERS
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT HUP INT TERM
mkdir -p "$WORK/bin" "$WORK/remote" "$WORK/out" "$WORK/model" "$WORK/fixtures"
cat >"$WORK/bin/curl" <<'EOF'
#!/bin/sh
set -eu
out=''
while [ "$#" -gt 0 ]; do
    case "$1" in -o) out=$2; shift 2 ;; *) url=$1; shift ;; esac
done
cp "$E2_TEST_ORIGIN/${url##*/}" "$out"
EOF
chmod +x "$WORK/bin/curl"
export PATH="$WORK/bin:$PATH" E2_TEST_ORIGIN="$WORK/remote"
NAME=e2-embeddinggemma-2-int8-text.e8cq
printf verified >"$WORK/remote/$NAME"
(cd "$WORK/remote" && sha256sum "$NAME") >"$WORK/remote/manifest.sha256"
printf corrupt >"$WORK/remote/$NAME"
printf old >"$WORK/out/$NAME"
if sh "$ROOT/download_models.sh" text "$WORK/out"; then echo 'accepted corrupt cache' >&2; exit 1; fi
[ "$(printf old | sha256sum)" = "$(sha256sum <"$WORK/out/$NAME")" ]
printf verified >"$WORK/remote/$NAME"
sh "$ROOT/download_models.sh" text "$WORK/out"
(cd "$WORK/out" && sha256sum -c manifest.sha256)
rm "$WORK/remote/manifest.sha256"
if sh "$ROOT/download_models.sh" text "$WORK/out"; then echo 'accepted absent manifest' >&2; exit 1; fi
printf bad >"$WORK/remote/model.safetensors"
printf original >"$WORK/model/model.safetensors"
if sh "$ROOT/download_model.sh" "$WORK/model"; then echo 'accepted wrong checkpoint hash' >&2; exit 1; fi
[ "$(printf original | sha256sum)" = "$(sha256sum <"$WORK/model/model.safetensors")" ]
cat >"$WORK/bin/e2_export_cache" <<'EOF'
#!/bin/sh
printf export >"$E2_Q8_WRITE"
EOF
cat >"$WORK/bin/e2_test" <<'EOF'
#!/bin/sh
[ "${E2_TEST_FAIL_VERIFY:-0}" = 0 ]
EOF
chmod +x "$WORK/bin/e2_export_cache" "$WORK/bin/e2_test"
cp "$WORK/bin/e2_test" "$WORK/bin/e2_test_image"
cp "$WORK/bin/e2_test" "$WORK/bin/e2_test_audio"
export E2_FIXTURE_DIR="$WORK/fixtures"
if E2_TEST_FAIL_VERIFY=1 sh "$ROOT/tools/export_models.sh" "$WORK/model" "$WORK/out" "$WORK/bin"; then
    echo 'published failed verification' >&2; exit 1
fi
[ "$(printf verified | sha256sum)" = "$(sha256sum <"$WORK/out/$NAME")" ]
sh "$ROOT/tools/export_models.sh" "$WORK/model" "$WORK/out" "$WORK/bin"
(cd "$WORK/out" && sha256sum -c manifest.sha256)
echo 'artifact integrity and failed replacement preservation OK'

# The GPU release gate must fail closed without weights, a compiled backend, or real device checks.
mkdir -p "$WORK/no-weights"
if sh "$ROOT/tools/check_gpu.sh" "$WORK/no-weights" "$WORK/no-weights-build" 1; then
    echo 'GPU gate accepted missing weights' >&2; exit 1
fi
[ ! -e "$WORK/no-weights-build" ]
printf tokenizer >"$WORK/model/tokenizer.json"
cat >"$WORK/bin/cmake" <<'EOF'
#!/bin/sh
set -eu
while [ "$#" -gt 0 ]; do
    if [ "$1" = -B ]; then
        mkdir -p "$2"
        cp /bin/true "$2/e2_test_gpu_soak"
        break
    fi
    shift
done
EOF
cat >"$WORK/bin/ctest" <<'EOF'
#!/bin/sh
set -eu
[ -z "${E2_GPU_ONLY+x}" ] || { echo 'diagnostic override leaked into the gate' >&2; exit 99; }
[ "${E2_GPU:-}" = "$GPU_GATE_DEVICE" ] || { echo 'device selection lost' >&2; exit 99; }
case " $* " in
  *' -N '*)
    if [ "$GPU_GATE_CASE" = backend ]; then echo 'Total Tests: 0'; exit 0; fi
    for t in gpu_copy gpu_profile_fallback gpu_l0_runtime gpu_attn_recipe gpu_text gpu_image gpu_audio \
             gpu_text_reference gpu_validation gpu_shapes gpu_runtime gpu_lifecycle gpu_audio_reference gpu_soak; do
        echo "Test #1: $t"
    done ;;
  *) if [ "$GPU_GATE_CASE" = skipped ]; then echo '1/1 gpu_copy ***Skipped'; else exit 8; fi ;;
esac
EOF
chmod +x "$WORK/bin/cmake" "$WORK/bin/ctest"
for mode in backend skipped failed; do
    if GPU_GATE_CASE=$mode GPU_GATE_DEVICE=audit-device E2_GPU=audit-device E2_GPU_ONLY=1 sh "$ROOT/tools/check_gpu.sh" "$WORK/model" "$WORK/gate-$mode" 1 \
        >"$WORK/gate-$mode.log" 2>&1; then
        echo "GPU gate accepted $mode checks" >&2; exit 1
    fi
    case "$mode" in
      backend) grep -q 'required Intel GPU test missing' "$WORK/gate-$mode.log" ;;
      skipped) grep -q 'refuses skipped or missing checks' "$WORK/gate-$mode.log" ;;
    esac
    ! grep -E 'diagnostic override leaked|device selection lost' "$WORK/gate-$mode.log"
done
echo 'GPU gate missing-checkpoint/backend, skipped-check and failing-check rejection OK'
