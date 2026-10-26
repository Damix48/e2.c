#!/bin/sh
# Fetch packed models. A SHA256 manifest is mandatory; unverified downloads are never published.
set -eu
HF_REPO=${HF_E2_REPO:-Damix48/e2-embeddinggemma-2-int8}
PREFIX=e2-embeddinggemma-2-int8
WHICH=${1:-all}
DEST=${2:-models/e2}
BASE="https://huggingface.co/$HF_REPO/resolve/main"
die() { echo "download_models.sh: $*" >&2; exit 1; }
case "$WHICH" in
    all) FILES="text text-image text-audio text-image-audio" ;;
    text|text-image|text-audio|text-image-audio) FILES=$WHICH ;;
    -h|--help) echo "usage: $0 [all|text|text-image|text-audio|text-image-audio] [dest-dir]"; exit 0 ;;
    *) die "unknown selection '$WHICH'" ;;
esac
command -v curl >/dev/null 2>&1 || die 'curl is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
mkdir -p "$DEST"
# Unique staging paths also keep concurrent downloads from sharing a partial file.
stage=$(mktemp -d "$DEST/.download.XXXXXX")
trap 'rm -rf "$stage"' EXIT HUP INT TERM
fetch() {
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fL --retry 2 -sS -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$BASE/$1"
    else
        curl -fL --retry 2 -sS -o "$2" "$BASE/$1"
    fi
}
fetch manifest.sha256 "$stage/manifest.sha256" ||
    die "no SHA256 manifest published; generate locally with tools/export_models.sh <model_dir> $DEST"
for slug in $FILES; do
    name="$PREFIX-$slug.e8cq"; out="$DEST/$name"
    want=$(awk -v n="$name" '$2 == n { print $1 }' "$stage/manifest.sha256")
    [ "${#want}" = 64 ] || die "missing or duplicate SHA256 for $name"
    case "$want" in *[!0-9a-f]*) die "invalid SHA256 for $name" ;; esac
    if [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$want" ]; then
        echo "  $name: verified, already present"
        continue
    fi
    fetch "$name" "$stage/$name" || die "$name unavailable; destination unchanged"
    [ "$(sha256sum "$stage/$name" | cut -d' ' -f1)" = "$want" ] || die "$name: SHA256 mismatch; destination unchanged"
    mv -f "$stage/$name" "$out"
    echo "  $name: verified"
done
mv -f "$stage/manifest.sha256" "$DEST/manifest.sha256"
echo "ready in $DEST (tokenizer.json is still required)"
