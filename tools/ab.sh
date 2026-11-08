#!/bin/sh
# ab.sh SPEC [REPS] - interleaved A/B of two e2_bench_full binaries.
#
# This host is shared (load average 1-5), and the same binary measures 16 and
# 19 ms for text 8 twenty seconds apart.  A single run cannot A/B anything, so
# build the two variants to tests/e2_ab_a and tests/e2_ab_b, then run them
# alternately REPS times and keep the min of each row.  min is the estimator
# that noise can only push up.
#
#   cp tests/e2_bench_full tests/e2_ab_a; <build variant b>; cp ... tests/e2_ab_b
#   sh tools/ab.sh /tmp/spec.tsv 5
set -e
SPEC=${1:?usage: ab.sh SPEC [REPS]}
REPS=${2:-5}
A=tests/e2_ab_a
B=tests/e2_ab_b
[ -x "$A" ] && [ -x "$B" ] || { echo "need $A and $B" >&2; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
: > "$TMP/a"
: > "$TMP/b"
i=0
while [ $i -lt $REPS ]; do
    "$A" "$SPEC" >> "$TMP/a"
    "$B" "$SPEC" >> "$TMP/b"
    i=$((i + 1))
done
awk -F'\t' -v fa="$TMP/a" -v fb="$TMP/b" '
    FNR == NR { if (NF == 3 && ($2 + 0) > 0) { if (!($1 in a) || $2 < a[$1]) a[$1] = $2 } next }
    NF == 3 { if (($2 + 0) > 0 && (!($1 in b) || $2 < b[$1])) b[$1] = $2 }
    END {
        printf "%-16s %9s %9s %8s\n", "row", "A", "B", "B/A"
        for (k in a) if (k in b)
            printf "%-16s %9.3f %9.3f %8.3f\n", k, a[k], b[k], b[k] / a[k]
    }' "$TMP/a" "$TMP/b"
