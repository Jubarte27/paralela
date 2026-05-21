#!/bin/bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

PARALLEL="$HERE/parallel"
SEQUENTIAL="$HERE/sequential"

TARGET=$PARALLEL

gcc -fopenmp "$TARGET/ga.c" -I"$TARGET" -lm -o "$TARGET/ga"

cd "$TARGET" && ./ga