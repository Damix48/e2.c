#!/bin/sh
# download_model.sh - fetch the EmbeddingGemma-2 checkpoint the engine reads.
#
#   ./download_model.sh [dest-dir]          (default: models/embeddinggemma-2)
#
# The engine needs exactly two files: model.safetensors (bf16, 1.4 GB) and tokenizer.json (31 MB).
# Neither is tracked by git; models/ is ignored, and a packed int8 cache exported with E2_Q8_WRITE
# replaces model.safetensors entirely (it still needs tokenizer.json).
set -eu

REPO=google/embeddinggemma-2
DEST=${1:-models/embeddinggemma-2}
BASE="https://huggingface.co/$REPO/resolve/main"

# name:bytes, checked against the current revision on main. Set E2_SKIP_SIZE_CHECK=1 if upstream
# re-uploads a file and you are sure of what you got.
FILES="model.safetensors:1488915288 tokenizer.json:32170510"

command -v curl >/dev/null 2>&1 || { echo "download_model.sh: curl is required" >&2; exit 1; }
mkdir -p "$DEST"

for spec in $FILES; do
    name=${spec%%:*}
    want=${spec##*:}
    out="$DEST/$name"
    if [ -f "$out" ] && [ "$(wc -c <"$out")" -eq "$want" ]; then
        echo "  $name: already present ($want bytes)"
        continue
    fi
    echo "  $name: downloading $want bytes from $REPO"
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fL --retry 3 -C - -H "Authorization: Bearer $HF_TOKEN" -o "$out" "$BASE/$name"
    else
        curl -fL --retry 3 -C - -o "$out" "$BASE/$name"
    fi
    got=$(wc -c <"$out")
    if [ "$got" -ne "$want" ] && [ "${E2_SKIP_SIZE_CHECK:-0}" != 1 ]; then
        echo "download_model.sh: $name is $got bytes, expected $want" >&2
        echo "  (a truncated download resumes on a rerun; E2_SKIP_SIZE_CHECK=1 ignores this)" >&2
        exit 1
    fi
done

echo
echo "model ready: $DEST"
case "$DEST" in
    /*) ABS=$DEST ;;
    *)  ABS="$PWD/$DEST" ;;
esac
echo "configure the build with:"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DE2_MODEL_DIR=\"$ABS\""
