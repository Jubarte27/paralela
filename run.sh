#!/bin/bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

PARALLEL="$HERE/parallel"
SEQUENTIAL="$HERE/sequential"
COMMON="$HERE/common"
VENV="$HERE/.venv"

case "$1" in
    -s)
        TARGET=$SEQUENTIAL
        ;;
    -p)
        TARGET=$PARALLEL
        ;;
    *)
        TARGET=$PARALLEL
        ;;
esac

SIZE=$2
THREADS=$3

if ! g++ -g -rdynamic -O3 -fopenmp "$TARGET/ga.cpp" -I"$TARGET" -I"$COMMON" -lm -o "$TARGET/ga"; then
    echo "Failed to compile ga.cpp"
    exit 1
fi

source "$VENV/bin/activate" || exit 1
cd "$HERE" || exit 1
"$TARGET/ga" "$SIZE" "$THREADS"