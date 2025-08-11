#!/bin/bash

expected_file="expected_result.csv"
actual_file="actual_result.csv"
output_file="compare_result.csv"

echo "name,model_size,mubs,mbs,nnode,seq,expected_elapsed_ms,actual_elapsed_ms,error_percent" > "$output_file"

declare -A exp_elapsed

# create key-value for expected
while IFS=, read -r jobid name model_size mubs mbs nnode seq tflops elapsed gpu_lat; do
    [[ "$name" == "name" ]] && continue  # 헤더 스킵
    key="${name},${model_size},${mubs},${mbs},${nnode},${seq}"
    exp_elapsed["$key"]="$elapsed"
done < "$expected_file"

# compare with actual
while IFS=, read -r jobid name model_size mubs mbs nnode seq tflops elapsed gpu_lat; do
    [[ "$name" == "name" ]] && continue  # 헤더 스킵
    key="${name},${model_size},${mubs},${mbs},${nnode},${seq}"
    exp_val="${exp_elapsed[$key]}"

    [[ -z "$exp_val" ]] && continue
    [[ -z "$exp_val" || -z "$elapsed" ]] && continue

    error_percent=$(awk -v e="$exp_val" -v a="$elapsed" 'BEGIN { printf "%.2f", ((a - e) / e) * 100 }')

    echo "$name,$model_size,$mubs,$mbs,$nnode,$seq,$exp_val,$elapsed,${error_percent}%" >> "$output_file"
done < "$actual_file"
