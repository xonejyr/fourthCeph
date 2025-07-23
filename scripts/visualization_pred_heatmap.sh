#!/bin/bash
set -e

# --- 配置 ---
CONFIG_FILE=$1
CHECKPOINT_FILE=$2
EXPERIMENT_ID=${3:-"gaussian_viz_$(date +%Y%m%d_%H%M%S)"} 

# Python脚本路径
PYTHON_SCRIPT_PATH="./scripts/visualization_pred_heatmap.py"

# --- 检查文件 ---
if [ -z "$CONFIG_FILE" ] || [ -z "$CHECKPOINT_FILE" ]; then
    echo "错误: 必须提供配置文件和检查点文件"
    echo "用法: $0 <config.yaml路径> <checkpoint.pth路径> [experiment_id]"
    exit 1
fi

for file in "$CONFIG_FILE" "$CHECKPOINT_FILE" "$PYTHON_SCRIPT_PATH"; do
    if [ ! -f "$file" ]; then
        echo "错误: 文件不存在: $file"
        exit 1
    fi
done

# --- 执行 ---
echo "开始可视化..."
echo "配置文件: $CONFIG_FILE"
echo "检查点文件: $CHECKPOINT_FILE"
echo "实验ID: $EXPERIMENT_ID"

python ${PYTHON_SCRIPT_PATH} \
    --cfg ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --exp-id "${EXPERIMENT_ID}"

echo "可视化完成"