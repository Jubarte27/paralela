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

RUNS=()
while true; do
    case "$1" in
        -t)
            RUNS+=(in_threads)
            shift
            ;;
        -b)
            RUNS+=(in_base)
            shift
            ;;
        -i)
            RUNS+=(in_tel)
            shift
            ;;
        *)
            break
            ;;
    esac
done
if (( ${#RUNS[@]} == 0 )); then
    RUNS=(in_threads in_base in_tel)
fi

main() {
    for run in "${RUNS[@]}"; do
        $run
    done
}

run_experiments() {
    local csv_in="$1"; local start_line="$2"; local exp_num="$3"; local sub_dir="$4"; local exec_mode="$5"; local csv_out="$6"

    tail -n +"$start_line" "$csv_in" | while IFS=, read -r -a values; do
        [ -z "${values[0]}" ] && continue

        local cmd
        "${exec_mode}_cmd" cmd

        for value in "${values[@]}"; do
            local param_value
            param_value=$(echo "$value" | tr -d '\r') # Maldito Windows
            cmd+=("$param_value")
        done
        
        # Setup directories and routing path lengths
        local dir="$BASE_DIR/$sub_dir/$exp_num"
        [ "$sub_dir" == "." ] && dir="$BASE_DIR/$exp_num"
        mkdir -p "$dir"

        echo "Running $exp_num: ${cmd[*]}"
        "${exec_mode}_exec" "$dir" cmd values "$csv_out"

        ((exp_num++))
    done
}

get_start_line_and_exp() {
    local mode="$1"; local target="$2"; local source="$3"; local label="$4"

    local completed_exps=0
    local total_exps=$(($(wc -l < "$source") - 1))

    completed_exps=$("${mode}_count" "$target" "$source")

    local start_line=$((2 + completed_exps))
    local exp_num=$((completed_exps + 1))

    if [ "$completed_exps" -ge "$total_exps" ]; then
        echo "Nothing to do (skipping $completed_exps completed runs)" >&2
    elif [ "$completed_exps" -gt 0 ]; then
        echo "Resuming '$label' from experiment $exp_num (skipping $completed_exps completed runs)" >&2
    fi

    echo "$start_line $exp_num"
}

in_threads() {
    read -r start_line exp_num <<< "$(get_start_line_and_exp "csv" "$CSV_OUT_THREADS" "$CSV_IN_THREADS" "in_threads")"

    run_experiments "$CSV_IN_THREADS" "$start_line" "$exp_num" "threads" "time" "$CSV_OUT_THREADS"
}

in_base() {
    read -r start_line exp_num <<< "$(get_start_line_and_exp "csv" "$CSV_OUT" "$CSV_IN" "in_base")"

    run_experiments "$CSV_IN" "$start_line" "$exp_num" "." "time" "$CSV_OUT"
}

in_tel() {
    if [ -d "/home/intel/oneapi/vtune/2021.1.1/" ]; then
        source "/home/intel/oneapi/vtune/2021.1.1/vtune-vars.sh"
    elif [ -d "/opt/intel/oneapi/vtune/latest/" ]; then
        source "/opt/intel/oneapi/vtune/latest/vtune-vars.sh"
    else
        echo "Don't know where vtune profiler is"
        exit 1
    fi

    read -r start_line exp_num <<< "$(get_start_line_and_exp "dir" "$BASE_DIR/intel" "$CSV_IN_TEL" "in_tel")"

    run_experiments "$CSV_IN_TEL" "$start_line" "$exp_num" "intel" "vtune"
}

vtune_cmd() { local -n _cmd="$1"; _cmd=("$EXEC" "-d"); }

vtune_exec() {
    local dir="$1"; local -n _cmd="$2"

    local vtune_log="$dir/vtune.log"
    local res_dir="$dir/vtune_v"
    rm -rf "$res_dir"

    if ! vtune -collect $VTUNE_ANALYSIS -knob sampling-mode=hw -result-dir "$res_dir" -- "${_cmd[@]}" 2>&1 | tee "$vtune_log"; then
        exit 1
    fi
}

time_cmd() { local -n _cmd="$1"; _cmd=("$EXEC"); }

time_exec() {
    local dir="$1"; local -n _cmd="$2"; local -n _values="$3"; local csv_out="$4"

    local exec_log="$dir/exec.log"
    TIMEFORMAT="%R"
    { time { "${_cmd[@]}"; echo; } 2>&1; } 2>&1 | tee "$exec_log"
    
    local time_elapsed
    time_elapsed=$(tail -n 1 "$exec_log" | tr -d '\r')
    echo "$(IFS=','; echo "${_values[*]}"),$time_elapsed" >> "$csv_out"
}

vtune_count() {
    local target="$1"
    local idx=1
    while [ -f "$target/$idx/vtune.log" ]; do ((idx++)); done
    echo $((idx - 1))
}

csv_count() {
    local target="$1"; local source="$2"
    IFS=, read -r -a headers < "$source"
    
    if [ -f "$target" ] && [ "$(wc -l < "$target")" -gt 1 ]; then
        echo $(($(wc -l < "$target") - 1))
    else
        echo "$(IFS=','; echo "${headers[*]}"),TIME_ELAPSED" > "$target"
        echo 0
    fi
}

main