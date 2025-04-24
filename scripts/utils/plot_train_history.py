import pandas as pd
import matplotlib.pyplot as plt
import os

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 路径配置: csv_path, fig_dir
import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description='Set yaml file for plot_train_history')
parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
# Parse arguments
args = parser.parse_args()

# the newst version doesn't need setting path directly
from shared_params_manage import ParamManager
param_manager = ParamManager(config_file=args.config_file)
paths = param_manager.get_paths()
csv_path = paths['csv_path']
fig_dir = paths['root_fig_dir']
#
os.makedirs(fig_dir, exist_ok=True)
#--------------------------------------------------------------------------------80


# for heatmap
#root_path = "/mnt/home_extend/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_ResFPN-256x256_ResFPN_ce_heatmap/params_search-ResNet_SIZE_LR_TYPE/"
#csv_path = os.path.join(root_path, "history_csv/trial_ac2e9084-1cc2-4d22-85b8-cf7e1ae44a0e.csv")
#fig_sub_dir = csv_path.split("/")[-1].split(".")[0]
#fig_dir = os.path.join(root_path, f"history_plot/{fig_sub_dir}")
#os.makedirs(fig_dir, exist_ok=True)
#csv_path="/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_DualUNet-256x256_dualunet_ce_heatmap/train_history.csv"
#fig_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_DualUNet-256x256_dualunet_ce_heatmap/visualizations/"

# for coord
#csv_path="/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_DualUNet-256x256_dualunet_ce_coord/train_history.csv"
#fig_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_DualUNet-256x256_dualunet_ce_coord/visualizations/"
#--------------------------------------------------------------------------------80


import pandas as pd
import matplotlib.pyplot as plt

def plot_dual_axis_metrics_withAnnotation(
    file_path,
    fig_dir,
    loss_col='loss',
    metrics_cols=['mre', 'sd'],
    title=f'Training Loss and Metrics over Epochs for {csv_path.split("/")[-2].split("-")[-1]}',
    figsize=(12, 6),
    loss_color='red',
    metrics_colors=['blue', 'green']
):
    """
    绘制双 Y 轴图表，左侧为 loss，右侧为 metrics，并在图中标识最佳 mre 和 mre/sd 同时最佳点。
    
    参数：
    - file_path: CSV 文件路径
    - loss_col: loss 列名
    - metrics_cols: metrics 列名列表
    - title: 图表标题
    - figsize: 图表尺寸
    - loss_color: loss 曲线颜色
    - metrics_colors: metrics 曲线颜色列表
    """
    # 读取数据
    data = pd.read_csv(file_path)
    epochs = data['epoch']

    # 创建双 Y 轴图表
    fig, ax1 = plt.subplots(figsize=figsize)

    # 绘制 loss（左 Y 轴）
    ax1.plot(epochs, data[loss_col], label='Loss', color=loss_color, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=loss_color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 创建右 Y 轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Metrics (mm)', color='black', fontsize=12)

    # 绘制 metrics（右 Y 轴）
    for i, metric in enumerate(metrics_cols):
        if metric in data.columns:
            mask = data[metric].notna()
            if mask.sum() > 0:
                ax2.plot(
                    epochs[mask],
                    data[metric][mask],
                    label=metric.upper(),
                    color=metrics_colors[i % len(metrics_colors)],
                    marker='o',
                    linestyle='--',
                    linewidth=1.5,
                    markersize=6
                )

    # 处理最佳点
    if 'mre' in data.columns and 'sd' in data.columns:
        # 初始化极大值
        best_mre_only = float('inf')  # 仅基于MRE的最佳值
        best_mre_only_epoch = None
        best_mre_only_sd = None

        best_both_mre = float('inf')  # 基于Both的最佳值
        best_both_sd = float('inf')
        best_both_epoch = None

        # 从epoch=1开始遍历
        for idx, row in data.iterrows():
            curr_mre = row['mre'] if pd.notna(row['mre']) else float('inf')
            curr_sd = row['sd'] if pd.notna(row['sd']) else float('inf')
            curr_epoch = row['epoch']

            # 更新仅基于MRE的最佳值
            if curr_mre < best_mre_only:
                best_mre_only = curr_mre
                best_mre_only_epoch = curr_epoch
                best_mre_only_sd = curr_sd if pd.notna(curr_sd) else None

            # 更新基于Both的最佳值
            if curr_mre < best_both_mre and curr_sd < best_both_sd:
                best_both_mre = curr_mre
                best_both_sd = curr_sd
                best_both_epoch = curr_epoch

        # 获取坐标轴范围
        x_min, x_max = ax2.get_xlim()
        y_min, y_max = ax2.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 检查两个最佳点是否一致
        points_are_same = (best_both_epoch == best_mre_only_epoch and 
                          best_both_mre == best_mre_only and 
                          best_both_sd == best_mre_only_sd and 
                          best_both_epoch is not None)

        if points_are_same and best_both_epoch is not None:
            # 如果一致，只显示一个点和一个注释框（基于Both）
            ax2.plot(
                best_both_epoch, best_both_mre, '*',
                color='#FFD700', markeredgecolor='black', markeredgewidth=1.5,
                markersize=18, label=None
            )
            # 放置在右上角
            text_x = x_max - x_range * 0.1
            text_y = y_max - y_range * 0.1
            ax2.annotate(
                f'Best Metrics (Both)\nEpoch: {int(best_both_epoch)}\nMRE: {best_both_mre:.2f} mm\nSD: {best_both_sd:.2f} mm',
                xy=(best_both_epoch, best_both_mre),
                xytext=(text_x, text_y),
                fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
                ha='right', va='top'
            )
        else:
            # 如果不一致，显示两个点和两个注释框
            # 绘制最佳 MRE 点（紫色星星）
            if best_mre_only_epoch is not None:
                ax2.plot(
                    best_mre_only_epoch, best_mre_only, '*',
                    color='purple', markeredgecolor='black', markeredgewidth=1.5,
                    markersize=18, label=None
                )
                # MRE 注释框放在右上角
                text_x_mre = x_max - x_range * 0.1
                text_y_mre = y_max - y_range * 0.1
                sd_text_mre = f"{best_mre_only_sd:.2f}" if pd.notna(best_mre_only_sd) else "N/A"
                ax2.annotate(
                    f'Best Metrics (MRE)\nEpoch: {int(best_mre_only_epoch)}\nMRE: {best_mre_only:.2f} mm\nSD: {sd_text_mre} mm',
                    xy=(best_mre_only_epoch, best_mre_only),
                    xytext=(text_x_mre, text_y_mre),
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
                    ha='right', va='top'
                )

            # 绘制最佳 Both 点（黄色星星，如果存在）
            if best_both_epoch is not None:
                ax2.plot(
                    best_both_epoch, best_both_mre, '*',
                    color='#FFD700', markeredgecolor='black', markeredgewidth=1.5,
                    markersize=18, label=None
                )
                # Both 注释框放在中部偏右
                text_x_both = x_max - x_range * 0.1
                text_y_both = y_min + y_range * 0.5  # 中部位置
                # 如果与 MRE 注释框太近，调整位置
                if 'text_y_mre' in locals() and abs(text_y_both - text_y_mre) < y_range * 0.2:
                    text_y_both = text_y_mre - y_range * 0.25  # 向下偏移

                ax2.annotate(
                    f'Best Metrics (Both)\nEpoch: {int(best_both_epoch)}\nMRE: {best_both_mre:.2f} mm\nSD: {best_both_sd:.2f} mm',
                    xy=(best_both_epoch, best_both_mre),
                    xytext=(text_x_both, text_y_both),
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
                    ha='right', va='center'
                )

    # 设置右 Y 轴刻度颜色
    ax2.tick_params(axis='y', labelcolor=metrics_colors[0])

    # 设置标题和图例
    plt.title(title, fontsize=14, pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(f"{fig_dir}/train_history.png", dpi=300, bbox_inches='tight')

# 调用函数
plot_dual_axis_metrics_withAnnotation(csv_path, fig_dir)
print(f"- Train history saved to {fig_dir}/train_history.png")