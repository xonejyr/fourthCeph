import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dual_axis_metrics_withAnnotation(
    file_path,
    fig_dir,
    loss_col='loss',
    metrics_cols=['mre', 'sd'],
    title=None,  # 标题将动态生成
    figsize=(12, 6),
    loss_color='red',
    metrics_colors=['blue', 'green']
):
    """
    绘制双 Y 轴图表，左侧为 loss，右侧为 metrics，并在图中标识最佳 mre 和 mre/sd 同时最佳点。
    
    参数：
    - file_path: CSV 文件路径
    - fig_dir: 保存图表的目录
    - loss_col: loss 列名
    - metrics_cols: metrics 列名列表
    - title: 图表标题（若为 None，则动态生成）
    - figsize: 图表尺寸
    - loss_color: loss 曲线颜色
    - metrics_colors: metrics 曲线颜色列表
    """
    # 动态生成标题
    if title is None:
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名（不含扩展名）
        title = f'Training Loss and Metrics over Epochs for {file_name}'

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
        best_mre_only = float('inf')
        best_mre_only_epoch = None
        best_mre_only_sd = None

        best_both_mre = float('inf')
        best_both_sd = float('inf')
        best_both_epoch = None

        for idx, row in data.iterrows():
            curr_mre = row['mre'] if pd.notna(row['mre']) else float('inf')
            curr_sd = row['sd'] if pd.notna(row['sd']) else float('inf')
            curr_epoch = row['epoch']

            if curr_mre < best_mre_only:
                best_mre_only = curr_mre
                best_mre_only_epoch = curr_epoch
                best_mre_only_sd = curr_sd if pd.notna(curr_sd) else None

            if curr_mre < best_both_mre and curr_sd < best_both_sd:
                best_both_mre = curr_mre
                best_both_sd = curr_sd
                best_both_epoch = curr_epoch

        x_min, x_max = ax2.get_xlim()
        y_min, y_max = ax2.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        points_are_same = (best_both_epoch == best_mre_only_epoch and 
                          best_both_mre == best_mre_only and 
                          best_both_sd == best_mre_only_sd and 
                          best_both_epoch is not None)

        if points_are_same and best_both_epoch is not None:
            ax2.plot(
                best_both_epoch, best_both_mre, '*',
                color='#FFD700', markeredgecolor='black', markeredgewidth=1.5,
                markersize=18, label=None
            )
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
            if best_mre_only_epoch is not None:
                ax2.plot(
                    best_mre_only_epoch, best_mre_only, '*',
                    color='purple', markeredgecolor='black', markeredgewidth=1.5,
                    markersize=18, label=None
                )
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

            if best_both_epoch is not None:
                ax2.plot(
                    best_both_epoch, best_both_mre, '*',
                    color='#FFD700', markeredgecolor='black', markeredgewidth=1.5,
                    markersize=18, label=None
                )
                text_x_both = x_max - x_range * 0.1
                text_y_both = y_min + y_range * 0.5
                if 'text_y_mre' in locals() and abs(text_y_both - text_y_mre) < y_range * 0.2:
                    text_y_both = text_y_mre - y_range * 0.25
                ax2.annotate(
                    f'Best Metrics (Both)\nEpoch: {int(best_both_epoch)}\nMRE: {best_both_mre:.2f} mm\nSD: {best_both_sd:.2f} mm',
                    xy=(best_both_epoch, best_both_mre),
                    xytext=(text_x_both, text_y_both),
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, shrinkA=0, shrinkB=5),
                    ha='right', va='center'
                )

    ax2.tick_params(axis='y', labelcolor=metrics_colors[0])

    plt.title(title, fontsize=14, pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()

    # 获取文件名（不含扩展名）作为保存名
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(fig_dir, f"{file_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图表以释放内存
    print(f"- Train history saved to {save_path}")

# 遍历目录并绘图
def process_csv_directory(csv_dir):
    # 获取与 history_csv 同级的 history_png 目录
    parent_dir = os.path.dirname(csv_dir)
    png_dir = os.path.join(parent_dir, 'history_png')
    
    # 如果 history_png 目录不存在，则创建
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    
    # 遍历 history_csv 目录下的所有 CSV 文件
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            csv_path = os.path.join(csv_dir, file_name)
            plot_dual_axis_metrics_withAnnotation(csv_path, png_dir)

# 示例调用
#csv_directory = '/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_HeatmapBasisNFR-512x512_HeatmapBasisNFR_ce_heatmap/params_search/history_csv'  # 替换为你的 history_csv 目录路径
#process_csv_directory(csv_directory)