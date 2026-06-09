#!/bin/env bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

BASE_DIR="ga"
mkdir -p "$BASE_DIR"

BASE_DIR="$(realpath $BASE_DIR)"

EXEC="$HERE/run.sh"
VTUNE_ANALYSIS="hotspots" # performance-snapshot, hotspots, hpc-performance
CSV_IN="$HERE/doe.csv"
CSV_IN_TEL="$HERE/doe_intel.csv"
CSV_IN_THREADS="$HERE/doe_threads.csv"
CSV_OUT="$BASE_DIR/out.csv"
CSV_OUT_THREADS="$BASE_DIR/out_threads.csv"

while true; do
    case "$1" in
        -t)
            THREAD_ONLY=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

main() {
    in_threads
    in_base
    in_tel
}

in_threads() {
    IFS=, read -r -a headers < "$CSV_IN_THREADS"
    echo "$(IFS=','; echo "${headers[*]}"),TIME_ELAPSED" > "$CSV_OUT_THREADS"

    exp_num=1
    tail -n +2 "$CSV_IN_THREADS" | while IFS=, read -r -a values; do
        cmd=("$EXEC")
        for value in "${values[@]}"; do
            # Maldito Windows
            param_value=$(echo "$value" | tr -d '\r')
            cmd+=("$param_value")
        done
        
        DIR="$BASE_DIR/threads/$exp_num"
        mkdir -p "$DIR"

        EXEC_LOG="$DIR/exec.log"

        echo "Running $exp_num: ${cmd[*]}"
        # Just time
        TIMEFORMAT="%R"
        # feioso
        { time { "${cmd[@]}"; echo; } 2>&1; } 2>&1 | tee "$EXEC_LOG"
        TIME_ELAPSED=$(tail -n 1 "$EXEC_LOG")
        echo "$(IFS=','; echo "${values[*]}"),$TIME_ELAPSED" >> "$CSV_OUT_THREADS"

        ((exp_num++))
    done

    if [ "$THREAD_ONLY" == "true" ]; then
        exit 0
    fi
}

in_base() {
    IFS=, read -r -a headers < "$CSV_IN"
    echo "$(IFS=','; echo "${headers[*]}"),TIME_ELAPSED" > "$CSV_OUT"

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
        TIMEFORMAT="%R"
        # feioso
        { time { "${cmd[@]}"; echo; } 2>&1; } 2>&1 | tee "$EXEC_LOG"
        TIME_ELAPSED=$(tail -n 1 "$EXEC_LOG")
        echo "$(IFS=','; echo "${values[*]}"),$TIME_ELAPSED" >> "$CSV_OUT"

        ((exp_num++))
    done
}


in_tel() {
    if [ -d "/home/intel/oneapi/vtune/2021.1.1/" ]; then
        source "/home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh"
    elif [ -d "/opt/intel/oneapi/vtune/latest/" ]; then
        source "/opt/intel/oneapi/vtune/latest/vtune-vars.sh"
    else
        echo "Don't know where vtune profiler is"
        # exit 1
    fi

    exp_num=1
    tail -n +2 "$CSV_IN_TEL" | while IFS=, read -r -a values; do
        cmd=("$EXEC" "-d")
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
        if ! vtune -collect $VTUNE_ANALYSIS -knob sampling-mode=hw -result-dir "$RES_DIR" -- "${cmd[@]}" 2>&1 | tee "$VTUNE_LOG"; then
            exit 1
        fi

        ((exp_num++))
    done
}

main