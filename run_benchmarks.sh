#!/bin/bash

# Run benchmarks.py for every combination of arguments
# Usage: ./run_benchmarks.sh

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
#pip install --upgrade "jax[cuda12]"

#unset LD_LIBRARY_PATH  # Clear LD_LIBRARY_PATH to avoid conflicts

# Define the arguments
# dataset : either "small", "medium", "large" or "huge"
datasets=("small" "medium" "large") # "huge")

# common_input : either "true" or "false"
common_inputs=("true" "false")

# common_hp : either "true" or "false"
common_hps=("true" "false")

printf "\n\n\nStarting benchmarks...\n\n"

# Iterate over all combinations of arguments
for dataset in "${datasets[@]}"; do
    for common_input in "${common_inputs[@]}"; do
        for common_hp in "${common_hps[@]}"; do
            echo "Running benchmark.py with dataset=$dataset, common_input=$common_input, common_hp=$common_hp"
            PYTHONPATH=. python3 benchmarks/benchmark.py --dataset "$dataset" --common_input "$common_input" --common_hp "$common_hp" > "benchmarks/logs/benchmark_${dataset}_CI${common_input}_CHP${common_hp}.log" 2>&1
        done
    done
done

printf "\nAll benchmarks completed. Check logs/ for output."