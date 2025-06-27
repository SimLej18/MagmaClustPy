#!/bin/bash

# Run benchmarks.py for every combination of arguments
# Usage: ./run_benchmarks.sh

# Install required packages
pip install --upgrade pip
pip install -r "../env/requirements.txt"
#pip install --upgrade "jax[cuda12]"

#unset LD_LIBRARY_PATH  # Clear LD_LIBRARY_PATH to avoid conflicts

# Define the arguments
# dataset : either "small", "medium", "large" or "huge"
datasets=("small" "medium" "large") # "huge")

# shared_input : either "true" or "false"
shared_inputs=("true" "false")

# shared_hp : either "true" or "false"
shared_hps=("true" "false")

printf "\n\n\nStarting benchmarks...\n\n"

# Iterate over all combinations of arguments
for dataset in "${datasets[@]}"; do
    for shared_input in "${shared_inputs[@]}"; do
        for shared_hp in "${shared_hps[@]}"; do
            echo "Running benchmark.py with dataset=$dataset, shared_input=$shared_input, shared_hp=$shared_hp"
            PYTHONPATH=../ python3 benchmark.py --dataset "$dataset" --shared_input "$shared_input" --shared_hp "$shared_hp" > "logs/benchmark_${dataset}_CI${shared_input}_CHP${shared_hp}.log" 2>&1
        done
    done
done

printf "\nAll benchmarks completed. Check logs/ for output."