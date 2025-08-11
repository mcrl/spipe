#!/bin/bash

expected_file="expected_result.csv"
actual_file="actual_result.csv"
output_file="compare_result.csv"

echo "name,expected_elapsed_ms,actual_elapsed_ms,error_percent" > "$output_file"

# join based on name
join -t, -1 2 -2 2 \
    <(tail -n +2 "$expected_file" | sort -t, -k2,2) \
    <(tail -n +2 "$actual_file"   | sort -t, -k2,2) \
| while IFS=, read -r jobid_e name_e model_size_e mbs_e gbs_e nnode_e seq_e tflops_e elapsed_e gpu_lat_e \
                         jobid_a name_a model_size_a mbs_a gbs_a nnode_a seq_a tflops_a elapsed_a gpu_lat_a
do
    # skip if elapsed_time is blank
    if [[ -z "$elapsed_e" || -z "$elapsed_a" ]]; then
        continue
    fi

    # clac error percent
    error_percent=$(awk -v e="$elapsed_e" -v a="$elapsed_a" 'BEGIN { printf "%.2f", ((a - e) / e) * 100 }')

    echo "$name_e,$elapsed_e,$elapsed_a,${error_percent}%" >> "$output_file"
done
