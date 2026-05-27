#!/bin/bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

PARALLEL="$HERE/parallel"
SEQUENTIAL="$HERE/sequential"
COMMON="$HERE/common"
VENV="$HERE/.venv"

TARGET=$PARALLEL

## da pra melhorar isso aqui
FLAGS=(-O3)
while true; do
    case "$1" in
        -s)
            TARGET=$SEQUENTIAL
            shift
            ;;
        -p)
            TARGET=$PARALLEL
            shift
            ;;
        -d)
            FLAGS=(-g -Og) #debug
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [ "$#" -lt "5" ]; then
    echo "Too few arguments"
    exit 1
fi



if ! g++ --std="c++23" "${FLAGS[@]}" -rdynamic -fopenmp "$TARGET/ga.cpp" -I"$TARGET" -I"$COMMON" -lm -o "$TARGET/ga"; then
    echo "Failed to compile ga.cpp"
    exit 1
fi

source "$VENV/bin/activate" || exit 1
cd "$HERE" || exit 1
"$TARGET/ga" "${@:1}"
