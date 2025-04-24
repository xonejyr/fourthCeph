 #!/bin/bash

CONFIG_NAME=$1   # necessary
EXPID=$2 # optional

# Script to run analysis utilities with different configurations
# It will iterate through coordinate and heatmap methods
# For each method, it will use both val and test datasets

# Path to shared params yaml file
SHARED_PARAMS_YAML="./scripts/utils/shared_params.yaml"

# Store original content to restore at the end
ORIGINAL_CONTENT=$(cat $SHARED_PARAMS_YAML)

# Function to update the shared_params.yaml file
update_yaml() {
    local exp_name=$1
    local dataset=$2
    
    # Use sed to update the experiment_name and val_dataset
    sed -i "s/experiment_name:.*/experiment_name: $exp_name/" $SHARED_PARAMS_YAML
    sed -i "s/val_dataset:.*/val_dataset: $dataset/" $SHARED_PARAMS_YAML
    
    echo "Updated shared_params.yaml with experiment_name: $exp_name, val_dataset: $dataset"
}

# Function to run analysis commands
run_analysis() {
    local method_type=$1
    local dataset=$2
    local config_file=$3 # start from ./configs/
    
    echo "============================================="
    echo "Running analysis for method: $method_type, dataset: $dataset"
    echo "============================================="
    
    # Plot train history
    echo "Plotting train history..."
    python ./scripts/utils/plot_train_history.py --config_file $config_file
    
    # Get landmark statistics
    echo "Getting landmark statistics..."
    python ./scripts/utils/get_pts_statistics.py --config_file $config_file
    
    # Draw points on images
    echo "Drawing points on images..."
    python ./scripts/utils/draw_pts_img.py --config_file $config_file

    
    echo "Analysis complete for method: $method_type, dataset: $dataset"
    echo "============================================="
}

run_analysis_latexTable() {
    local method_type=$1
    local config_file=$2

    echo "============================================="
    echo "Latex Table generating for method: $method_type"
    echo "============================================="

    EXP_NAME=$(python3 -c "import sys; sys.path.append('./scripts/utils'); from shared_params_manage import ParamManager; pm = ParamManager(config_file=${config_file}); print(pm.get_experiment_name())")

    echo $EXP_NAME

    # Get the Latex Table
    echo "Get the Latex Table..."
    ./scripts/utils/generate_results_table.sh "./exp/$EXP_NAME" -1
}

# Main execution
echo "Starting analysis script..."

## Configuration 1: Coordinate method with val dataset
#update_yaml "train_ce_coord_DualUNet-256x256_dualunet_ce_coord_bone" "val"
#run_analysis "coordinate" "val"
#run_analysis_latexTable "coordinate"
#
## Configuration 2: Coordinate method with test dataset
#update_yaml "train_ce_coord_DualUNet-256x256_dualunet_ce_coord_soft" "test"
#run_analysis "coordinate" "test"
#run_analysis_latexTable "coordinate"

# Configuration 3: Heatmap method with val dataset
update_yaml "${EXPID}-${CONFIG_NAME}" "val"
run_analysis "heatmap" "val"
#run_analysis_latexTable "heatmap"

# Configuration 4: Heatmap method with test dataset
update_yaml "${EXPID}-${CONFIG_NAME}" "test"
run_analysis "heatmap" "test"
run_analysis_latexTable "heatmap"

# Restore original content
echo "$ORIGINAL_CONTENT" > $SHARED_PARAMS_YAML
echo "Restored original shared_params.yaml content"

echo "Analysis script completed for all configurations" 