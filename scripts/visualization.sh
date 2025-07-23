#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <config_file> <model_path> <experiment_name>"
    exit 1
fi

# Assign input arguments to variables
CONFIG_FILE=$1
MODEL_PATH=$2
EXPERIMENT_NAME=$3

# Define the paths to the three scripts
SCRIPT_JSON="./scripts/visualization_json.sh"
SCRIPT_HEATMAP="./scripts/visualization_pred_heatmap.sh"
SCRIPT_PTS_ON_IMG="./scripts/visualization_pred_pts_on_img.sh"

# Execute the scripts sequentially
echo "Running visualization_json.sh..."
bash "$SCRIPT_JSON" "$CONFIG_FILE" "$MODEL_PATH" "$EXPERIMENT_NAME"

if [ $? -eq 0 ]; then
    echo "visualization_json.sh completed successfully."
else
    echo "visualization_json.sh failed."
    exit 1
fi

echo "Running visualization_pred_heatmap.sh..."
bash "$SCRIPT_HEATMAP" "$CONFIG_FILE" "$MODEL_PATH" "$EXPERIMENT_NAME"

if [ $? -eq 0 ]; then
    echo "visualization_pred_heatmap.sh completed successfully."
else
    echo "visualization_pred_heatmap.sh failed."
    exit 1
fi

echo "Running visualization_pred_pts_on_img.sh..."
bash "$SCRIPT_PTS_ON_IMG" "$CONFIG_FILE" "$MODEL_PATH" "$EXPERIMENT_NAME"

if [ $? -eq 0 ]; then
    echo "visualization_pred_pts_on_img.sh completed successfully."
else
    echo "visualization_pred_pts_on_img.sh failed."
    exit 1
fi

echo "All scripts executed successfully."