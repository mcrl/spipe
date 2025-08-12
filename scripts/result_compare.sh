#!/bin/bash

expected_file="${SPIPE_ROOT:-.}/results/expected.csv"
actual_file="${SPIPE_ROOT:-.}/results/actual.csv"
output_file="${SPIPE_ROOT:-.}/results/compare.csv"

echo "jobid,name,cluster,model_size,mubs,mbs,nnode,seq,expected_elapsed_ms,actual_elapsed_ms,error_percent" > "$output_file"

declare -A exp_elapsed

# create key-value for expected
while IFS=, read -r jobid name cluster model_size mubs mbs nnode seq tflops elapsed gpu_lat; do
    [[ "$name" == "name" ]] && continue
    key="${name},${cluster},${model_size},${mubs},${mbs},${nnode},${seq}"
    exp_elapsed["$key"]="$elapsed"
done < "$expected_file"

# compare with actual
while IFS=, read -r jobid name cluster model_size mubs mbs nnode seq tflops elapsed gpu_lat; do
    [[ "$name" == "name" ]] && continue
    key="${name},${cluster},${model_size},${mubs},${mbs},${nnode},${seq}"
    exp_val="${exp_elapsed[$key]}"

    [[ -z "$exp_val" ]] && continue

    if [[ "$exp_val" =~ ^[0-9]+([.][0-9]+)?$ && "$elapsed" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        error_percent=$(awk -v e="$exp_val" -v a="$elapsed" 'BEGIN { printf "%.2f", ((a - e) / e) * 100 }')
    else
        error_percent=""
    fi

    echo "$jobid,$name,$cluster,$model_size,$mubs,$mbs,$nnode,$seq,$exp_val,$elapsed,${error_percent}%" >> "$output_file"
done < "$actual_file"
