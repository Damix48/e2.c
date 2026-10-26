#!/bin/sh
# export_models.sh - write the four packed int8 models from a checkpoint.
#
#   tools/export_models.sh [MODEL_DIR] [OUT_DIR] [BUILD_DIR]
#       defaults: models/embeddinggemma-2  models/e2  build
#
# Each file is the int8 weight set the loader builds (E2_Q8_WRITE), restricted to the towers the
# corresponding input shape needs (E2_Q8_TOWERS). They are not four different models: it is one weight
# set, and the only difference is which towers are packed into the file, so `text` is smaller than the
# full multimodal one. E2_Q8_READ=<file> then replaces the checkpoint entirely.
#
# Every export is read back and the matching fixture test is run against it, so a truncated or
# mis-selected cache cannot leave this machine silently. Set E2_SKIP_VERIFY=1 to skip that.
set -eu

MODEL=${1:-models/embeddinggemma-2}
OUT=${2:-models/e2}
BUILD=${3:-build}
FIXTURES=${E2_FIXTURE_DIR:-tests/fixtures}

BIN="$BUILD/e2_export_cache"
PREFIX=e2-embeddinggemma-2-int8

die() { echo "export_models.sh: $*" >&2; exit 1; }

[ -x "$BIN" ] || die "$BIN not found (build first: cmake --build $BUILD -j)"
[ -f "$MODEL/model.safetensors" ] || die "$MODEL/model.safetensors not found (./download_model.sh)"

mkdir -p "$OUT"

# combo:slug:test-binary:[test-args]. The slug is the file name; the test is what verifies the
# read-back. A cache without a vision tower needs `noimage` on the audio test, which otherwise runs the
# text+image+audio case too.
COMBOS="text:text:e2_test:
text+image:text-image:e2_test_image:
text+audio:text-audio:e2_test_audio:noimage
all:text-image-audio:e2_test_audio:"

printf '%-38s %10s  %-32s %s\n' FILE SIZE MD5 VERIFY
failed=0
manifest="$OUT/manifest.txt"
stage=$(mktemp -d "$OUT/.export.XXXXXX")
trap 'rm -rf "$stage"' EXIT HUP INT TERM
: >"$stage/manifest.txt"
: >"$stage/manifest.sha256"

for combo in $COMBOS; do
    IFS=':' read -r towers slug test_bin test_args <<EOF
$combo
EOF
    file="$stage/$PREFIX-$slug.e8cq"
    if ! E2_Q8_TOWERS="$towers" E2_Q8_WRITE="$file" "$BIN" "$MODEL" >/dev/null; then
        echo "  $slug: export FAILED" >&2
        failed=1
        continue
    fi
    bytes=$(wc -c <"$file")
    md5=$(md5sum "$file" | cut -d' ' -f1)

    verify="-"
    if [ "${E2_SKIP_VERIFY:-0}" != 1 ]; then
        [ -x "$BUILD/$test_bin" ] && [ -d "$FIXTURES" ] || die "verification requires $BUILD/$test_bin and $FIXTURES"
        # read the cache back and run the fixture test that exercises this shape
        if E2_Q8_READ="$file" "$BUILD/$test_bin" "$MODEL" "$FIXTURES" $test_args >"$OUT/$slug.verify.log" 2>&1; then
            verify="PASS ($test_bin${test_args:+ $test_args})"
        else
            verify="FAIL ($test_bin, see $OUT/$slug.verify.log)"
            failed=1
        fi
    fi

    printf '%-38s %10s  %-32s %s\n' "$(basename "$file")" "$bytes" "$md5" "$verify"
    case "$verify" in FAIL*) continue ;; esac
    printf '%s  %s  %s\n' "$md5" "$bytes" "$(basename "$file")" >>"$stage/manifest.txt"
    (cd "$stage" && sha256sum "$(basename "$file")") >>"$stage/manifest.sha256"
done

[ "$failed" = 0 ] || die "at least one export failed; previous models unchanged"
for file in "$stage"/*.e8cq; do mv -f "$file" "$OUT/"; done
mv -f "$stage/manifest.txt" "$manifest"
mv -f "$stage/manifest.sha256" "$OUT/manifest.sha256"
echo
echo "manifests: $manifest (legacy), $OUT/manifest.sha256 (required for downloads)"
echo "use one with: E2_Q8_READ=$OUT/$PREFIX-text.e8cq $BUILD/e2_test <model_dir> tests/fixtures"
