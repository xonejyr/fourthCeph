#!/bin/bash

# Script to generate LaTeX tables from experiment results

# Make Python script executable
chmod +x ./scripts/utils/generate_latex_table.py

# Check if experiment directory is provided
if [ $# -lt 1 ]; then
    echo "Error: Experiment directory is required"
    echo "Usage: $0 <exp_dir> [exp_id] [-t train_exp_idx] [-e test_exp_idx]"
    exit 1
fi

# Initialize parameters
EXP_DIR="$1"
shift

# Set default EXP_ID to -1 if not provided
EXP_ID=${1:--1}
if [ $# -ge 1 ]; then
    shift
fi

# Set both train and test indices to EXP_ID by default
TRAIN_EXP_IDX=$EXP_ID
TEST_EXP_IDX=$EXP_ID

# Parse additional options
while getopts ":t:e:" opt; do
    case $opt in
        t) TRAIN_EXP_IDX="$OPTARG" ;;
        e) TEST_EXP_IDX="$OPTARG" ;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    esac
done

# Display information about what we're doing
echo "Generating LaTeX table for experiment: $EXP_DIR"
echo "Using train experiment index: $TRAIN_EXP_IDX, test experiment index: $TEST_EXP_IDX"

# Run the Python script
python ./scripts/utils/generate_latex_table.py --exp_dir "$EXP_DIR" --train_exp_idx "$TRAIN_EXP_IDX" --test_exp_idx "$TEST_EXP_IDX"

echo "LaTeX table generation completed" 