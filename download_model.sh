#!/bin/sh
# Fetch the exact checkpoint used by the frozen references; stage and verify before publishing.
set -eu
REPO=google/embeddinggemma-2
DEST=${1:-models/embeddinggemma-2}
BASE="https://huggingface.co/$REPO/resolve/main"
# SHA256 pins the bytes even if upstream main changes. Updating these is a deliberate release step.
FILES="model.safetensors:197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79
tokenizer.json:4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"
command -v curl >/dev/null 2>&1 || { echo 'curl is required' >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo 'sha256sum is required' >&2; exit 1; }
mkdir -p "$DEST"
for spec in $FILES; do
    name=${spec%%:*}; want=${spec#*:}; out="$DEST/$name"
    if [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$want" ]; then
        echo "  $name: verified, already present"
        continue
    fi
    part="$out.part"
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fL --retry 3 -C - -H "Authorization: Bearer $HF_TOKEN" -o "$part" "$BASE/$name"
    else
        curl -fL --retry 3 -C - -o "$part" "$BASE/$name"
    fi
    if [ "$(sha256sum "$part" | cut -d' ' -f1)" != "$want" ]; then
        echo "$name: SHA256 mismatch; destination unchanged. Remove $part before retrying." >&2
        exit 1
    fi
    mv -f "$part" "$out"
done
echo "model verified: $DEST"
