#!/bin/bash

# Run benchmarks.py for every combination of arguments
# Usage: ./run_benchmarks.sh

# Initialize conda if needed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install conda first."
    exit 1
fi

# Initialize conda for bash
source /opt/anaconda3/etc/profile.d/conda.sh

# If "MagmaClustPy" environment exists, activate it
if conda env list | grep -q "MagmaClustPy"; then
    echo "Activating MagmaClustPy environment...\n\n"
    conda activate MagmaClustPy
else
    echo "MagmaClustPy environment does not exist. Please create it first."
    exit 1
fi

# Install required packages
python -m pip install --upgrade -q pip
python -m pip install -r "env/requirements.txt" -q
#pip install --upgrade "jax[cuda12]"

#unset LD_LIBRARY_PATH  # Clear LD_LIBRARY_PATH to avoid conflicts

# Define the arguments
# dataset : either "small", "medium", "large" or "huge"
datasets=("small" "medium" "large") # "huge")

# shared_input : either "true" or "false"
shared_inputs=("true" "false")

# shared_hp : either "true" or "false"
shared_hps=("true" "false")

printf "Starting benchmarks...\n\n"

# Iterate over all combinations of arguments
for dataset in "${datasets[@]}"; do
    for shared_input in "${shared_inputs[@]}"; do
        for shared_hp in "${shared_hps[@]}"; do
            echo "Running benchmark.py with dataset=$dataset, shared_input=$shared_input, shared_hp=$shared_hp"
            PYTHONPATH=. python benchmarks/benchmark.py --dataset "$dataset" --shared_input "$shared_input" --shared_hp "$shared_hp" > "benchmarks/logs/benchmark_${dataset}_CI${shared_input}_CHP${shared_hp}.log" 2>&1
        done
    done
done

printf "\nAll benchmarks completed. Check logs/ for output."