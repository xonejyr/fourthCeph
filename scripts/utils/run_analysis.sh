#!/bin/bash

CONFIG=$1   # necessary
EXPID=$2         # optional
NEW_FILE=$3

CONFIG_NAME_WITH_YAML="${CONFIG##*/}" 
CONFIG_NAME="${CONFIG_NAME_WITH_YAML%%.*}"
# Script to run analysis utilities with different configurations
# It will iterate through coordinate and heatmap methods
# For each method, it will use both val and test datasets

# Path to shared params yaml file
SHARED_PARAMS_YAML="./scripts/utils/shared_params.yaml"

# Directory to store temporary YAML files
TEMP_YAML_DIR="./configs/search_yamls"
# Create directory only if it doesn't exist
[ -d "$TEMP_YAML_DIR" ] || mkdir -p "$TEMP_YAML_DIR"

# Function to create and update a new YAML file
create_and_update_yaml() {
    local exp_name=$1
    local dataset=$2
    local method_type=$3
    local new_yaml=$4
    local timestamp=$(date +%Y%m%d%H%M%S)
    local final_yaml

    if [ -z "$new_yaml" ]; then
        # If new_yaml is not provided, create a new file by copying SHARED_PARAMS_YAML
        final_yaml="$TEMP_YAML_DIR/search-${exp_name}-${dataset}-${timestamp}.yaml"
        cp "$SHARED_PARAMS_YAML" "$final_yaml"
        echo "Created new YAML file: $final_yaml" >&2
    else
        # If new_yaml is provided, parse search_space parameters and rename
        # Extract parameter names from search_space section using yq or grep/awk
        local params=()
        if command -v yq >/dev/null 2>&1; then
            # Use yq if available for robust YAML parsing
            mapfile -t params < <(yq e '.search_space | keys' "$new_yaml" | grep -v '^-' | awk '{$1=$1};1')
        else
            # Fallback to grep/awk for basic parsing
            mapfile -t params < <(grep -A100 '^search_space:' "$new_yaml" | grep -E '^[[:space:]]*[A-Z_]+:' | awk -F: '{print $1}' | awk '{$1=$1};1')
        fi

        # Create suffix from parameters (e.g., NUM_BASES_SIGMA)
        local suffix=""
        for param in "${params[@]}"; do
            if [ -n "$param" ]; then
                suffix="${suffix}_${param}"
            fi
        done
        suffix=${suffix#_} # Remove leading underscore

        # Extract original filename without path and extension
        local base_name=$(basename "$new_yaml" .yaml)
        # Create new filename with search_ suffix
        final_yaml="$TEMP_YAML_DIR/${base_name}-search_${suffix}.yaml"

        # Copy the provided YAML to the new name
        cp "$new_yaml" "$final_yaml"
        echo "Renamed provided YAML to: $final_yaml" >&2
    fi

    # Use sed to update the experiment_name and val_dataset in the YAML
    sed -i "s/experiment_name:.*/experiment_name: $exp_name/" "$final_yaml"
    sed -i "s/val_dataset:.*/val_dataset: $dataset/" "$final_yaml"

    echo "Updated $final_yaml with experiment_name: $exp_name, val_dataset: $dataset" >&2

    # Return the path to the YAML file
    echo "$final_yaml"
}

# Function to run analysis commands
run_analysis() {
    local method_type=$1
    local dataset=$2
    local config_file=$3 # Path to the temporary YAML file

    echo "============================================="
    echo "Running analysis for method: $method_type, dataset: $dataset"
    echo "============================================="

    # Plot train history
    echo "Plotting train history..."
    python ./scripts/utils/plot_train_history.py --config_file "$config_file"

    # Get landmark statistics
    echo "Getting landmark statistics..."
    python ./scripts/utils/get_pts_statistics.py --config_file "$config_file"

    # Draw points on images
    echo "Drawing points on images..."
    python ./scripts/utils/draw_pts_img.py --config_file "$config_file"

    echo "Analysis complete for method: $method_type, dataset: $dataset"
    echo "============================================="
}

run_analysis_latexTable() {
    local method_type=$1
    local config_file=$2

    echo "============================================="
    echo "Latex Table generating for method: $method_type"
    echo "============================================="

    EXP_NAME=$(python3 -c "import sys; sys.path.append('./scripts/utils'); from shared_params_manage import ParamManager; pm = ParamManager(config_file='$config_file'); print(pm.get_experiment_name())")

    echo "$EXP_NAME"

    # Get the Latex Table
    echo "Get the Latex Table..."
    ./scripts/utils/generate_results_table.sh "./exp/$EXP_NAME" -1
}

# Main execution
echo "Starting analysis script..."

# Configuration 1: Heatmap method with val dataset
new_yaml=$(create_and_update_yaml "${EXPID}-${CONFIG_NAME}" "val" "heatmap" "${NEW_FILE}")
run_analysis "heatmap" "val" "$new_yaml"
#run_analysis_latexTable "heatmap" "$new_yaml"

# Configuration 2: Heatmap method with test dataset
new_yaml=$(create_and_update_yaml "${EXPID}-${CONFIG_NAME}" "test" "heatmap" "${NEW_FILE}")
run_analysis "heatmap" "test" "$new_yaml"

# get the latexTable for one exp
run_analysis_latexTable "heatmap" "$new_yaml"

# Clean up temporary YAML files (optional, uncomment to enable)
# rm -rf "$TEMP_YAML_DIR"/*

echo "Analysis script completed for all configurations with Latex Done!"