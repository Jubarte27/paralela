#!/bin/bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

PARALLEL="$HERE/parallel"
SEQUENTIAL="$HERE/sequential"
VENV="$HERE/.venv"

TARGET=$PARALLEL

if ! gcc -O1 -fopenmp "$TARGET/ga.c" -I"$TARGET" -lm -o "$TARGET/ga"; then
    echo "Failed to compile ga.c"
    exit 1
fi

source "$VENV/bin/activate" && cd "$TARGET" && ./ga