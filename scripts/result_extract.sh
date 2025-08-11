#!/bin/bash

output_file=${SPIPE_ROOT:-.}/results/actual_result.csv

echo "jobid,name,cluster,model_size,mubs,mbs,nnode,seq,tflops,elapsed_time_ms,gpu_latency" > "$output_file"

for file in slurm-*.out; do
    # extract config block
    config_block=$(awk '/===========Script Configuration===========/{flag=1;next}/===========================================/{flag=0}flag' "$file")

    jobid=$(echo "$config_block"     | grep -m1 "JOB_ID="         | cut -d'=' -f2)
    name=$(echo "$config_block"      | grep -m1 "SLURM_JOB_NAME=" | cut -d'=' -f2)
    cluster=$(echo "$config_block"   | grep -m1 "CLUSTER="        | cut -d'=' -f2)
    model_size=$(echo "$config_block"| grep -m1 "MODEL_SIZE="     | cut -d'=' -f2)
    nnode=$(echo "$config_block"     | grep -m1 "NNODE="          | cut -d'=' -f2)
    mubs=$(echo "$config_block"      | grep -m1 "MBS="            | cut -d'=' -f2)
    mbs=$(echo "$config_block"       | grep -m1 "GBS="            | cut -d'=' -f2)
    seq=$(echo "$config_block"       | grep -m1 "SEQ="            | cut -d'=' -f2)

    # last iteration line
    last_line=$(grep "iteration   " "$file" | tail -n 1)

    elapsed=$(echo "$last_line"  | grep -oP 'elapsed time per iteration \(ms\): \K[0-9.]+')
    gpu_latency=$(echo "$last_line"  | grep -oP 'max GPU pipeline latency \(ms\): \K[0-9.]+')
    tflops=$(echo "$last_line"   | grep -oP 'throughput per GPU \(TFLOP/s/GPU\): \K[0-9.]+')

    echo "$jobid,$name,$cluster,$model_size,$mubs,$mbs,$nnode,$seq,$tflops,$elapsed,$gpu_latency" >> "$output_file"
done
