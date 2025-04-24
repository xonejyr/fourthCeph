import pandas as pd
import matplotlib.pyplot as plt
import os

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 路径配置: csv_path, fig_dir

# the newst version doesn't need setting path directly
from shared_params_manage import ParamManager
param_manager = ParamManager()
paths = param_manager.get_paths()
csv_path = paths['csv_path']
fig_dir = paths['root_fig_dir']

os.makedirs(fig_dir, exist_ok=True)
#--------------------------------------------------------------------------------80


# for heatmap
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
        mask_mre_sd = data['mre'].notna() & data['sd'].notna()
        if mask_mre_sd.sum() > 0:
            mre_sd_data = data[mask_mre_sd]

            # 逻辑 1：基于最小 MRE 的最佳点
            best_mre_idx = mre_sd_data['mre'].idxmin()
            best_mre_epoch = mre_sd_data.loc[best_mre_idx, 'epoch']
            best_mre_value = mre_sd_data.loc[best_mre_idx, 'mre']
            best_mre_sd = mre_sd_data.loc[best_mre_idx, 'sd']

            # 逻辑 2：基于 MRE 和 SD 同时变小的最佳点
            best_both_epoch = None
            best_both_mre = None
            best_both_sd = None
            prev_mre = float('inf')
            prev_sd = float('inf')

            for idx, row in mre_sd_data.iterrows():
                curr_mre = row['mre']
                curr_sd = row['sd']
                if curr_mre < prev_mre and curr_sd < prev_sd:
                    best_both_epoch = row['epoch']
                    best_both_mre = curr_mre
                    best_both_sd = curr_sd
                    prev_mre = curr_mre
                    prev_sd = curr_sd

            # 获取坐标轴范围
            x_min, x_max = ax2.get_xlim()
            y_min, y_max = ax2.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min

            # 定义动态偏移函数
            def get_annotation_position(epoch, value, is_left_half, is_lower_half):
                x_offset = x_range * 0.1 if is_left_half else -x_range * 0.1
                y_offset = y_range * 0.1 if is_lower_half else -y_range * 0.1
                text_x = max(x_min + x_range * 0.05, min(epoch + x_offset, x_max - x_range * 0.05))
                text_y = max(y_min + y_range * 0.05, min(value + y_offset, y_max - y_range * 0.05))
                ha = 'left' if is_left_half else 'right'
                va = 'bottom' if is_lower_half else 'top'
                return text_x, text_y, ha, va

            # 检查两个最佳点是否一致
            points_are_same = (best_both_epoch == best_mre_epoch and 
                             best_both_mre == best_mre_value and 
                             best_both_sd == best_mre_sd and 
                             best_both_epoch is not None)

            if points_are_same:
                # 如果一致，只显示一个点和一个注释框（基于MRE）
                ax2.plot(
                    best_mre_epoch, best_mre_value, '*',
                    color='#FFD700', markeredgecolor='black', markeredgewidth=1.5,
                    markersize=18, label=None
                )
                is_left_half = best_mre_epoch < (x_min + x_max) / 2
                is_lower_half = best_mre_value < (y_min + y_max) / 2
                text_x, text_y, ha, va = get_annotation_position(
                    best_mre_epoch, best_mre_value, is_left_half, is_lower_half
                )
                ax2.annotate(
                    f'Best Metrics (MRE)\nEpoch: {int(best_mre_epoch)}\nMRE: {best_mre_value:.2f} mm\nSD: {best_mre_sd:.2f} mm',
                    xy=(best_mre_epoch, best_mre_value),
                    xytext=(text_x, text_y),
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
                    ha=ha, va=va
                )
            else:
                # 如果不一致，显示两个点和两个注释框
                # 绘制最佳 MRE 点（金色星形）
                ax2.plot(
                    best_mre_epoch, best_mre_value, '*',
                    color='#FFD700', markeredgecolor='black', markeredgewidth=1.5,
                    markersize=18, label=None
                )
                is_left_half_mre = best_mre_epoch < (x_min + x_max) / 2
                is_lower_half_mre = best_mre_value < (y_min + y_max) / 2
                text_x_mre, text_y_mre, ha_mre, va_mre = get_annotation_position(
                    best_mre_epoch, best_mre_value, is_left_half_mre, is_lower_half_mre
                )
                ax2.annotate(
                    f'Best Metrics (MRE)\nEpoch: {int(best_mre_epoch)}\nMRE: {best_mre_value:.2f} mm\nSD: {best_mre_sd:.2f} mm',
                    xy=(best_mre_epoch, best_mre_value),
                    xytext=(text_x_mre, text_y_mre),
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
                    ha=ha_mre, va=va_mre
                )

                # 绘制最佳 Both 点（紫色星形，如果存在）
                if best_both_epoch is not None:
                    ax2.plot(
                        best_both_epoch, best_both_mre, '*',
                        color='purple', markeredgecolor='black', markeredgewidth=1.5,
                        markersize=18, label=None
                    )
                    is_left_half_both = best_both_epoch < (x_min + x_max) / 2
                    is_lower_half_both = best_both_mre < (y_min + y_max) / 2
                    text_x_both, text_y_both, ha_both, va_both = get_annotation_position(
                        best_both_epoch, best_both_mre, is_left_half_both, is_lower_half_both
                    )
                    # 避免注释框重叠
                    if abs(text_x_both - text_x_mre) < x_range * 0.1 and abs(text_y_both - text_y_mre) < y_range * 0.1:
                        text_y_both += y_range * 0.1 if is_lower_half_both else -y_range * 0.1

                    ax2.annotate(
                        f'Best Metrics (Both)\nEpoch: {int(best_both_epoch)}\nMRE: {best_both_mre:.2f} mm\nSD: {best_both_sd:.2f} mm',
                        xy=(best_both_epoch, best_both_mre),
                        xytext=(text_x_both, text_y_both),
                        fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                        arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
                        ha=ha_both, va=va_both
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
    

#plot_dual_axis_metrics(csv_path, fig_dir)
plot_dual_axis_metrics_withAnnotation(csv_path, fig_dir)
print(f"- Train history saved to {fig_dir}/train_history.png")