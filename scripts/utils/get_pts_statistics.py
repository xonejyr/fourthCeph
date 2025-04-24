import json
import os
import torch
import numpy as np

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 路径配置: json_path, output_dir

# the newst version doesn't need setting path directly
from shared_params_manage import ParamManager
param_manager = ParamManager()
paths = param_manager.get_paths()
json_path = paths['json_path']
output_dir = paths['root_fig_dir']

os.makedirs(output_dir, exist_ok=True)
#--------------------------------------------------------------------------------80


#################################################################################80
# path settings
#json_path = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_unet-512x512_unet_ce_heatmap_20250312/test_gt_kpt.json"  # 替换为实际路径
#output_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_unet-512x512_unet_ce_heatmap_20250312/visualizations"  # 替换为输出目录
#--------------------------------------------------------------------------------80


#import json
import torch
import numpy as np
import plotly.express as px
import pandas as pd
import os

# 计算逐标志点的误差（单位：毫米）
def get_distance_of_mm(predictions, ground_truth, spacing=0.1):
    distances = torch.sqrt(torch.sum((predictions - ground_truth) ** 2, axis=-1))
    return distances * spacing  # 返回每个点的误差（毫米）

# 计算 SDR（Success Detection Rate）
def sdr(distances, threshold):
    return (distances < threshold).float()

# 主处理函数
def process_landmark_metrics(json_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取 JSON 文件
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    num_landmarks = len(data[0]["pred_pts"]) // 2  # 计算标志点个数
    all_mre = [[] for _ in range(num_landmarks)]  # 存储每个标志点的 MRE
    all_sdr_2_0mm = [[] for _ in range(num_landmarks)]
    all_sdr_2_5mm = [[] for _ in range(num_landmarks)]
    all_sdr_3_0mm = [[] for _ in range(num_landmarks)]
    all_sdr_4_0mm = [[] for _ in range(num_landmarks)]
    mre_matrix = []  # 存储所有样本的 MRE 用于热力图

    # 计算误差和 SDR
    for sample in data:
        pred_pts = torch.tensor(sample["pred_pts"]).reshape(-1, 2)  # (num_landmarks, 2)
        target_pts = torch.tensor(sample["target_pts"]).reshape(-1, 2)
        distances = get_distance_of_mm(pred_pts, target_pts)  # 计算误差（毫米）
        mre_matrix.append(distances.numpy())  # 保存到矩阵用于热力图

        for i in range(num_landmarks):
            all_mre[i].append(distances[i].item())
            all_sdr_2_0mm[i].append(sdr(distances[i], threshold=2).item())
            all_sdr_2_5mm[i].append(sdr(distances[i], threshold=2.5).item())
            all_sdr_3_0mm[i].append(sdr(distances[i], threshold=3).item())
            all_sdr_4_0mm[i].append(sdr(distances[i], threshold=4).item())

    # 转换为 NumPy 数组
    mre_matrix = np.array(mre_matrix)  # 形状 (num_samples, num_landmarks)
    sd_array = np.std(mre_matrix, axis=0)  # 每个标志点的 SD

    # 判断数据集类型
    dataset_name = os.path.basename(json_path).split("_")[0]
    if dataset_name == 'test':
        num_of_dataset = 2
    elif dataset_name == 'val':
        num_of_dataset = 1
    else:
        raise Exception("Invalid dataset name")

    # 输出统计结果到文本文件
    output_file = os.path.join(output_dir, f'pts_statistics_{dataset_name}.txt')
    with open(output_file, 'w') as f:
        f.write(f"The Localization results by landmark on {dataset_name.capitalize()} {num_of_dataset} dataset\n")
        f.write("Landmark: MRE(mm), SD(mm), SDR(2mm), SDR(2.5mm), SDR(3mm), SDR(4mm)\n")
        
        print(f"The Localization results by landmark on {dataset_name.capitalize()} {num_of_dataset} dataset")
        print("Landmark: MRE(mm), SD(mm), SDR(2mm), SDR(2.5mm), SDR(3mm), SDR(4mm)")
        
        for i in range(num_landmarks):
            mean_mre = np.mean(all_mre[i])
            std_mre = np.std(all_mre[i])
            mean_sdr_2_0mm = np.mean(all_sdr_2_0mm[i]) * 100
            mean_sdr_2_5mm = np.mean(all_sdr_2_5mm[i]) * 100
            mean_sdr_3_0mm = np.mean(all_sdr_3_0mm[i]) * 100
            mean_sdr_4_0mm = np.mean(all_sdr_4_0mm[i]) * 100

            line = f"{i+1}\t {mean_mre:1.2f}\t {std_mre:1.2f}\t {mean_sdr_2_0mm:3.1f}%\t {mean_sdr_2_5mm:.1f}%\t {mean_sdr_3_0mm:.1f}%\t {mean_sdr_4_0mm:.1f}%"
            print(line)
            f.write(line + "\n")

    # 绘制 MRE 热力图
    fig_mre = px.imshow(
        mre_matrix,
        labels={"x": "Landmark Index", "y": "Sample Index", "color": "MRE (mm)"},
        color_continuous_scale="hot",
        aspect="auto",
    )
    fig_mre.update_layout(
        title="Mean Radial Error (MRE) Heatmap",
        width=800,
        height=600,
        font=dict(size=14),
        coloraxis_colorbar=dict(len=0.75, thickness=20),
    )
    fig_mre.write_image(f"{output_dir}/mre_heatmap_{dataset_name}.png", scale=2)

    # 绘制 SD 热力图
    fig_sd = px.imshow(
        sd_array[np.newaxis, :],  # 转换为 (1, num_landmarks) 以适配热力图
        labels={"x": "Landmark Index", "y": "SD", "color": "Standard Deviation (SD)"},
        color_continuous_scale="RdBu",
        aspect="auto",
    )
    fig_sd.update_layout(
        title="Standard Deviation (SD) Heatmap",
        width=800,
        height=300,
        font=dict(size=14),
        coloraxis_colorbar=dict(len=0.75, thickness=20),
    )
    fig_sd.write_image(f"{output_dir}/sd_heatmap_{dataset_name}.png", scale=2)

    # 保存到 Excel
    excel_path = os.path.join(output_dir, f"metrics_results_{dataset_name}.xlsx")
    result_dict = {}
    for item in data:
        image_id = item['image_id']
        pred_pts = torch.tensor(item['pred_pts']).reshape(-1, 2)
        target_pts = torch.tensor(item['target_pts']).reshape(-1, 2)
        distances = get_distance_of_mm(pred_pts, target_pts).tolist()
        result_dict[image_id] = distances

    column_names = [f'Landmark_{i+1}' for i in range(num_landmarks)]
    df = pd.DataFrame.from_dict(result_dict, orient='index', columns=column_names)
    df.index.name = 'Image_ID'
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='Sheet1')

    # 输出保存路径
    print(f"- Metrics by landmarks in Excel saved to {excel_path}")
    print(f"- Metrics by landmarks in txt saved to {output_file}")
    print(f"- MRE heatmap saved to {output_dir}/mre_heatmap_{dataset_name}.png")
    print(f"- SD heatmap saved to {output_dir}/sd_heatmap_{dataset_name}.png")

# 示例调用
#if __name__ == "__main__":
    
process_landmark_metrics(json_path, output_dir)