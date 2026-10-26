#!/bin/sh
# download_models.sh - fetch the published packed int8 models from Hugging Face.
#
#   ./download_models.sh [which] [dest-dir]
#       which:    all (default) | text | text-image | text-audio | text-image-audio
#       dest-dir: models/e2 (default)
#
# Each file is a packed int8 weight set for one input shape, in this engine's own format - not GGUF,
# not safetensors, not loadable by transformers. Use it with E2_Q8_READ=<file>; tokenizer.json still
# comes from the checkpoint directory (./download_model.sh).
#
# PLACEHOLDER: the model repository is not published yet, so HF_E2_REPO names where it will live and
# every download fails with a pointer to the local export until it does. The files are reproducible:
# tools/export_models.sh writes byte-identical ones from the checkpoint.
HF_REPO=${HF_E2_REPO:-Damix48/e2-embeddinggemma-2-int8}

PREFIX=e2-embeddinggemma-2-int8
WHICH=${1:-all}
DEST=${2:-models/e2}
BASE="https://huggingface.co/$HF_REPO/resolve/main"

die() { echo "download_models.sh: $*" >&2; exit 1; }

case "$WHICH" in
    all)
        FILES="text text-image text-audio text-image-audio" ;;
    text|text-image|text-audio|text-image-audio)
        FILES=$WHICH ;;
    -h|--help)
        awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
        exit 0 ;;
    *)
        die "unknown selection '$WHICH' (all, text, text-image, text-audio, text-image-audio)" ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required"
mkdir -p "$DEST"

AUTH=""
[ -n "${HF_TOKEN:-}" ] && AUTH="Authorization: Bearer $HF_TOKEN"

fetch() { # fetch <url-path> <dest> [quiet]
    if [ -n "$AUTH" ]; then
        curl -fL --retry 2 -sS -H "$AUTH" -o "$2" "$BASE/$1"
    else
        curl -fL --retry 2 -sS -o "$2" "$BASE/$1"
    fi
}

# The manifest carries md5 + size per file; without it (repo not published, or a partial mirror) the
# download is accepted on size alone.
manifest="$DEST/manifest.txt"
if fetch manifest.txt "$manifest" 2>/dev/null; then
    echo "  manifest.txt: up to date"
else
    rm -f "$manifest"
fi

status=0
for slug in $FILES; do
    name="$PREFIX-$slug.e8cq"
    out="$DEST/$name"
    want_md5=""
    if [ -f "$manifest" ]; then
        want_md5=$(awk -v n="$name" '$3 == n { print $1 }' "$manifest")
    fi
    if [ -n "$want_md5" ] && [ -f "$out" ] &&
       [ "$(md5sum "$out" | cut -d' ' -f1)" = "$want_md5" ]; then
        echo "  $name: already present"
        continue
    fi
    printf '  %s: downloading from %s\n' "$name" "$HF_REPO"
    if ! fetch "$name" "$out"; then
        rm -f "$out"
        echo "  $name: not available at $BASE/$name" >&2
        echo "  the repository is not published yet - generate the same file locally with:" >&2
        echo "      tools/export_models.sh <model_dir> $DEST" >&2
        status=1
        continue
    fi
    if [ -n "$want_md5" ]; then
        got=$(md5sum "$out" | cut -d' ' -f1)
        if [ "$got" != "$want_md5" ]; then
            echo "  $name: md5 $got does not match the manifest ($want_md5)" >&2
            rm -f "$out"
            status=1
            continue
        fi
    fi
    echo "  $name: ok ($(wc -c <"$out") bytes)"
done

[ "$status" = 0 ] || die "one or more files could not be downloaded"

echo
echo "ready in $DEST. Use one with:"
first=$(echo $FILES | cut -d' ' -f1)
echo "  E2_Q8_READ=$DEST/$PREFIX-$first.e8cq ./build/e2_test models/embeddinggemma-2 tests/fixtures"
