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


#################################################################################80
# 计算逐标志点的metrics结果
def get_distance_of_mm(predictions, ground_truth, spacing=0.1):
    distances = torch.sqrt(torch.sum((predictions - ground_truth) ** 2, axis=-1))
    return distances * spacing  # 返回每个点的误差

def sdr(distances, threshold=[2, 2.5, 3, 4]):
    return (distances < threshold).float()

# 读取JSON文件
with open(json_path, "r") as f:
    data = json.load(f)

num_landmarks = len(data[0]["pred_pts"]) // 2  # 计算标志点个数
all_mre = [[] for _ in range(num_landmarks)]
all_sd = [[] for _ in range(num_landmarks)]
all_sdr_2_0mm = [[] for _ in range(num_landmarks)]
all_sdr_2_5mm = [[] for _ in range(num_landmarks)]
all_sdr_3_0mm = [[] for _ in range(num_landmarks)]
all_sdr_4_0mm = [[] for _ in range(num_landmarks)]

# 计算误差和 SDR
for sample in data:
    pred_pts = torch.tensor(sample["pred_pts"]).reshape(-1, 2)  # (19,2)
    target_pts = torch.tensor(sample["target_pts"]).reshape(-1, 2)

    distances = get_distance_of_mm(pred_pts, target_pts, target_size=256)  # 计算误差
    
    for i in range(num_landmarks):
        all_mre[i].append(distances[i].item())
        all_sdr_2_0mm[i].append(sdr(distances[i], threshold=2).item())
        all_sdr_2_5mm[i].append(sdr(distances[i], threshold=2.5).item())
        all_sdr_3_0mm[i].append(sdr(distances[i], threshold=3).item())
        all_sdr_4_0mm[i].append(sdr(distances[i], threshold=4).item())


if os.path.basename(json_path).split("_")[0] == 'test':
    num_of_dataset = 2
elif os.path.basename(json_path).split("_")[0] == 'val':
    num_of_dataset = 1 
else:
    raise Exception("Invalid dataset name")

# 构造输出文件路径
output_file = os.path.join(output_dir, f'pts_statistics_{os.path.basename(json_path).split("_")[0]}.txt')
with open(output_file, 'w') as f:
    # 写入标题
    f.write(f"The Localization results by landmark on Test {num_of_dataset} dataset\n")
    f.write("Landmark: MRE(mm), SD(mm), SDR(2mm), SDR(2.5mm), SDR(3mm), SDR(4mm)\n")

    # 计算每个标志点的均值
    print(f"The Localization results by landmark on Test {num_of_dataset} dataset")
    print("Landmark: MRE(mm), SD(mm), SDR(2mm), SDR(2.5mm), SDR(3mm), SDR(4mm)")
    
    for i in range(num_landmarks):
        mean_mre = np.mean(all_mre[i])
        std_mre = np.std(all_mre[i])
        mean_sdr_2_0mm = np.mean(all_sdr_2_0mm[i]) * 100
        mean_sdr_2_5mm = np.mean(all_sdr_2_5mm[i]) * 100
        mean_sdr_3_0mm = np.mean(all_sdr_3_0mm[i]) * 100
        mean_sdr_4_0mm = np.mean(all_sdr_4_0mm[i]) * 100

        print(f"{i+1}\t {mean_mre:1.2f}\t {std_mre:1.2f}\t {mean_sdr_2_0mm:3.1f}%\t {mean_sdr_2_5mm:.1f}%\t {mean_sdr_3_0mm:.1f}%\t {mean_sdr_4_0mm:.1f}%")
        f.write(f"{i+1}\t {mean_mre:1.2f}\t {std_mre:1.2f}\t {mean_sdr_2_0mm:3.1f}%\t {mean_sdr_2_5mm:.1f}%\t {mean_sdr_3_0mm:.1f}%\t {mean_sdr_4_0mm:.1f}%\n")

#--------------------------------------------------------------------------------80

#################################################################################80
# 绘制 MRE 热力图
import json
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

with open(json_path, "r") as f:
    data = json.load(f)

num_landmarks = len(data[0]["pred_pts"]) // 2
mre_matrix = []  # 存储 MRE
sd_matrix = []   # 存储 SD

for sample in data:
    pred_pts = torch.tensor(sample["pred_pts"]).reshape(-1, 2)
    target_pts = torch.tensor(sample["target_pts"]).reshape(-1, 2)
    distances = torch.sqrt(torch.sum((pred_pts - target_pts) ** 2, axis=-1)).numpy()
    mre_matrix.append(distances)

# 转换为 NumPy 数组
mre_matrix = np.array(mre_matrix)
sd_matrix = np.std(mre_matrix, axis=0, keepdims=True)

# 调整宽高比并绘制 MRE 热力图
fig_mre = px.imshow(
    mre_matrix,
    labels={"x": "Landmark Index", "y": "Sample Index", "color": "MRE (mm)"},
    color_continuous_scale="hot",
    aspect="auto",  # 自动调整长宽比，避免过于细长
)
fig_mre.update_layout(
    title="Mean Radial Error (MRE) Heatmap",
    width=800,  # 设置图像宽度
    height=600,  # 设置图像高度
    font=dict(size=14),  # 调整字体大小
    coloraxis_colorbar=dict(
        len=0.75,  # 调整颜色条长度
        thickness=20,  # 调整颜色条厚度
    ),
)
fig_mre.write_image(f"{output_dir}/mre_heatmap_{os.path.basename(json_path).split('_')[0]}.png", scale=2)

# 调整宽高比并绘制 SD 热力图
fig_sd = px.imshow(
    sd_matrix,
    labels={"x": "Landmark Index", "y": "SD", "color": "Standard Deviation (SD)"},
    color_continuous_scale="RdBu",
    aspect="auto",  # 自动调整长宽比
)
fig_sd.update_layout(
    title="Standard Deviation (SD) Heatmap",
    width=800,  # 设置图像宽度
    height=300,  # SD 矩阵高度较小，适当减少高度
    font=dict(size=14),
    coloraxis_colorbar=dict(
        len=0.75,
        thickness=20,
    ),
)
fig_sd.write_image(f"{output_dir}/sd_heatmap_{os.path.basename(json_path).split('_')[0]}.png", scale=2)
#--------------------------------------------------------------------------------80

#################################################################################80
# save to excel
import json
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_distance(pred_pts, target_pts):
    # 确保坐标点数量匹配
    assert len(pred_pts) == len(target_pts), "预测点和目标点数量不匹配"
    
    # 每2个值组成一个(x,y)坐标点
    distances = []
    for i in range(0, len(pred_pts), 2):
        pred_x, pred_y = pred_pts[i], pred_pts[i+1]
        target_x, target_y = target_pts[i], target_pts[i+1]
        
        # 计算欧几里得距离
        dist = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2) * 0.1
        distances.append(dist)
    
    return distances

def process_json_to_excel(json_path, output_excel_path):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_landmarks = len(data[0]["pred_pts"]) // 2
    
    # 创建结果字典
    result_dict = {}
    
    # 处理每个图片的数据
    for item in data:
        image_id = item['image_id']
        pred_pts = item['pred_pts']
        target_pts = item['target_pts']
        
        # 计算所有标志点的距离
        distances = calculate_distance(pred_pts, target_pts)
        result_dict[image_id] = distances
    
    # 创建列名（19个标志点）,可变数量
    column_names = [f'Landmark_{i+1}' for i in range(num_landmarks)]
    
    # 创建DataFrame
    df = pd.DataFrame.from_dict(result_dict, orient='index', columns=column_names)
    
    # 添加图片名称作为第一列
    df.index.name = 'Image_ID'
    
    # 保存到Excel
    with pd.ExcelWriter(output_excel_path) as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    
    #print(f"已保存到 {output_excel_path}")


# 保存数据到 Excel
excel_path = os.path.join(output_dir, f"metrics_results_{os.path.basename(json_path).split('_')[0]}.xlsx")

process_json_to_excel(json_path, excel_path)

print(f"- Metrics by landmarks in Excel saved to {excel_path}")
print(f"- Metrics by landmarks in txt saved to {output_file}")
#--------------------------------------------------------------------------------80