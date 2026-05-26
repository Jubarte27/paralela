#!/bin/env bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

BASE_DIR="ga"
mkdir -p "$BASE_DIR"

MAX_THREADS=${SLURM_CPUS_ON_NODE:-$(nproc --ignore=2)}

NUM_GENERATIONS=(10 20);
POP_SIZE=(16 32);
NUM_PARENTS=(3 5);

VERSIONS=("-p" "-s") # parallel, sequential
THREADS=(4 16 "$MAX_THREADS")
INPUT_SIZES=(small full)

EXEC="$HERE/run.sh"
VTUNE_ANALYSIS="hotspots" # performance-snapshot, hotspots, hpc-performance
CSV_FILE="out.csv"

if [ -d "/home/intel/oneapi/vtune/2021.1.1/" ]; then
	source "/home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh"
elif [ -d "/opt/intel/oneapi/vtune/latest/" ]; then
    source "/opt/intel/oneapi/vtune/latest/vtune-vars.sh"
else
    echo "Don't know where vtune profiler is"
    exit 1
fi

# CSV header
echo "Version,InputSize,Threads,TimeSeconds" > $CSV_FILE

for version in "${VERSIONS[@]}"; do
for size in "${INPUT_SIZES[@]}"; do
for threads in "${THREADS[@]}"; do
for gen in "${NUM_GENERATIONS[@]}"; do
for pop in "${POP_SIZE[@]}"; do
for parents in "${NUM_PARENTS[@]}"; do
    DIR="$BASE_DIR/($version)_size($size)_threads($threads)"
    mkdir -p "$DIR"

    EXEC_LOG="$DIR/exec.log"
    VTUNE_LOG="$DIR/vtune.log"

    RES_DIR="$DIR/vtune_v"
    rm -rf "$RES_DIR"

    COMMAND=$EXEC "$version" "$size" "$threads" "$gen" "$pop" "$parents"

    # Intel VTune Profiler
    if ! vtune -collect $VTUNE_ANALYSIS -result-dir "$RES_DIR" -- $COMMAND 2>&1 | tee "$VTUNE_LOG"; then
        exit 1
    fi
    # Just time
    /usr/bin/time -f "%e" $COMMAND 2>&1 | tee "$EXEC_LOG"
    TIME_ELAPSED=$(tail -n 1 "$EXEC_LOG")
    echo "$version,$size,$threads$gen,$pop,$parents,$TIME_ELAPSED" >> $CSV_FILE
done
done
done
done
done
done
