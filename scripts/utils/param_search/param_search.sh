#!/bin/bash
#set -x

CONFIG=$1
EXPID=${2:-"test_unet"}
NUMPARALLELEXPS=${3:-1}
CONFIG_YAML=$4

# Validate inputs
if [ -z "$CONFIG" ]; then
    echo "Error: CONFIG is required" >&2
    exit 1
fi
if [ -z "$CONFIG_YAML" ]; then
    echo "Error: CONFIG_YAML (YAML file path) is required" >&2
    exit 1
fi
if [ ! -f "$CONFIG_YAML" ]; then
    echo "Error: CONFIG_YAML file $CONFIG_YAML does not exist" >&2
    exit 1
fi

CONFIG_NAME_WITH_YAML="${CONFIG##*/}" 
CONFIG_NAME="${CONFIG_NAME_WITH_YAML%%.*}"

RAY_LOG_DIR="./exp/${EXPID}-${CONFIG_NAME}/raySearch_logs"
# Create directory only if it doesn't exist
[ -d "$RAY_LOG_DIR" ] || mkdir -p "$RAY_LOG_DIR"

# Function to read search_space parameters from a YAML file and return a search= suffix
get_search_suffix() {
    local yaml_file=$1

    # Check if the YAML file exists
    if [ ! -f "$yaml_file" ]; then
        echo "Error: YAML file $yaml_file does not exist" >&2
        return 1
    fi

    # Extract parameter names from search_space section using yq or grep/awk
    local params=()
    if command -v yq >/dev/null 2>&1; then
        # Use yq for robust YAML parsing
        mapfile -t params < <(yq e '.search_space | keys' "$yaml_file" | grep -v '^-' | awk '{$1=$1};1')
    else
        # Fallback to grep/awk for basic parsing
        mapfile -t params < <(grep -A100 '^search_space:' "$yaml_file" | grep -E '^[[:space:]]*[A-Z_]+:' | awk -F: '{print $1}' | awk '{$1=$1};1')
    fi

    # Create suffix from parameters (e.g., SIGMA_BETA)
    local suffix=""
    for param in "${params[@]}"; do
        if [ -n "$param" ]; then
            suffix="${suffix}_${param}"
        fi
    done
    suffix=${suffix#_} # Remove leading underscore

    # Return the search= suffix (or empty if no parameters)
    if [ -n "$suffix" ]; then
        echo "search=$suffix"
    else
        echo "search=empty"
    fi
}

# Generate log filename
log_suffix=$(get_search_suffix "$CONFIG_YAML")
if [ $? -ne 0 ]; then
    echo "Failed to generate log suffix" >&2
    exit 1
fi

# 构建 final_yaml
# 提取 CONFIG_YAML 的基本文件名（去掉 .yaml 扩展）
# 假设 CONFIG_YAML 和 log_suffix 已定义
config_dir_config_dir=$(dirname "$CONFIG_YAML")  # 获取目录，例如 ./configs
base_name_config_yaml=$(basename "$CONFIG_YAML" .yaml)
final_yaml="${config_dir_config_dir}/${base_name_config_yaml}-${log_suffix}.yaml"

# 复制文件
if [ -f "$CONFIG_YAML" ]; then
    cp "$CONFIG_YAML" "$final_yaml" && echo "Copied $CONFIG_YAML to $final_yaml"
else
    echo "Error: $CONFIG_YAML does not exist."
    exit 1
fi

# 构建 log file
LOG_FILE="$RAY_LOG_DIR/$log_suffix.log"
echo "Saving output to log file: $LOG_FILE" >&2

# Run Python command and redirect output (stdout and stderr) to log file
echo "Ray search is doing ..."
python ./scripts/utils/param_search/param_search.py \
    --exp-id "${EXPID}" \
    --cfg "${CONFIG}" \
    --seed 2333 \
    --numParallelExps "${NUMPARALLELEXPS}" \
    --config_file "${CONFIG_YAML}" > "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "Python command failed. Check $LOG_FILE for details" >&2
    exit 1
fi

echo "Python command completed successfully. Output saved to $LOG_FILE" >&2