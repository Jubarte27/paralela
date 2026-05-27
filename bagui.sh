#!/bin/env bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

BASE_DIR="ga"
mkdir -p "$BASE_DIR"

EXEC="$HERE/run.sh"
VTUNE_ANALYSIS="hotspots" # performance-snapshot, hotspots, hpc-performance
CSV_IN="doe.csv"
CSV_OUT="out.csv"
CSV_IN_TEL="doe_intel.csv"

if [[ ! -f "$CSV_IN" ]]; then
    echo "Error: Cannot find '$CSV_FILE'"
    exit 1
fi

if [ -d "/home/intel/oneapi/vtune/2021.1.1/" ]; then
	source "/home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh"
elif [ -d "/opt/intel/oneapi/vtune/latest/" ]; then
    source "/opt/intel/oneapi/vtune/latest/vtune-vars.sh"
else
    echo "Don't know where vtune profiler is"
    exit 1
fi

IFS=, read -r -a headers < "$CSV_IN"
echo "$(IFS=','; echo "${headers[*]}"),TIME_ELAPSED" > $CSV_OUT

exp_num=1
tail -n +2 "$CSV_IN" | while IFS=, read -r -a values; do
    cmd=("$EXEC")
    for value in "${values[@]}"; do
        # Maldito Windows
        param_value=$(echo "$value" | tr -d '\r')
        cmd+=("$param_value")
    done
    
    DIR="$BASE_DIR/$exp_num"
    mkdir -p "$DIR"

    EXEC_LOG="$DIR/exec.log"

    echo "Running $exp_num: ${cmd[*]}"
    # Just time
    /usr/bin/time -f "%e" "${cmd[@]}" 2>&1 | tee "$EXEC_LOG"
    TIME_ELAPSED=$(tail -n 1 "$EXEC_LOG")
    echo "$(IFS=','; echo "${value[*]}"),$TIME_ELAPSED" >> $CSV_OUT

    ((exp_num++))
done

exp_num=1
tail -n +2 "$CSV_IN_TEL" | while IFS=, read -r -a values; do
    cmd=("$EXEC")
    for value in "${values[@]}"; do
        # Maldito Windows
        param_value=$(echo "$value" | tr -d '\r')
        cmd+=("$param_value")
    done
    
    DIR="$BASE_DIR/intel/$exp_num"
    mkdir -p "$DIR"

    VTUNE_LOG="$DIR/vtune.log"
    RES_DIR="$DIR/vtune_v"
    rm -rf "$RES_DIR"
    
    echo "Running $exp_num: ${cmd[*]}"
    #Intel VTune Profiler
    if ! vtune -collect $VTUNE_ANALYSIS -result-dir "$RES_DIR" -- "${cmd[@]}" 2>&1 | tee "$VTUNE_LOG"; then
        exit 1
    fi

    ((exp_num++))
done
