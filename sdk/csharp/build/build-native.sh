#!/bin/sh
# Builds libe2 for the current platform and drops it where the C# SDK looks:
# src/E2/runtimes/<rid>/native/ (the nupkg layout).
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
sdk=$(cd "$here/.." && pwd)

make -C "$root" shared "$@"

case "$(uname -s)" in
    Darwin) lib=libe2.dylib; rid=osx-$(uname -m | sed 's/x86_64/x64/;s/arm64/arm64/') ;;
    *)      lib=libe2.so;    rid=linux-$(uname -m | sed 's/x86_64/x64/;s/aarch64/arm64/') ;;
esac

mkdir -p "$sdk/src/E2/runtimes/$rid/native"
cp "$root/$lib" "$sdk/src/E2/runtimes/$rid/native/$lib"
echo "native: $sdk/src/E2/runtimes/$rid/native/$lib"
