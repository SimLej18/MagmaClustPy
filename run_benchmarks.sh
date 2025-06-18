#!/bin/bash

# Run benchmarks.py for every combination of arguments
# Usage: ./run_benchmarks.sh

# Install required packages (only if needed)
if [ ! -f .requirements_installed ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt && touch .requirements_installed
fi

# Define the arguments
# dataset : either "small", "medium", "large" or "huge"
datasets=("small" "medium") # "large" "huge")

# common_input : either "true" or "false"
common_inputs=("true" "false")

# common_hp : either "true" or "false"
common_hps=("true" "false")

# Iterate over all combinations of arguments
for dataset in "${datasets[@]}"; do
    for common_input in "${common_inputs[@]}"; do
        for common_hp in "${common_hps[@]}"; do
            echo "Running benchmark.py with dataset=$dataset, common_input=$common_input, common_hp=$common_hp"
            PYTHONPATH=. python3 benchmarks/benchmark.py --dataset "$dataset" --common_input "$common_input" --common_hp "$common_hp" > "benchmarks/logs/benchmark_${dataset}_${common_input}_${common_hp}.log" 2>&1
        done
    done
done

echo "All benchmarks completed. Check logs/ for output."