import json
import cv2
import numpy as np
import os

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 路径配置: json_path, image_dir, output_dir

# the newst version doesn't need setting path directly
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
json_path = paths['json_path']
image_dir = paths['image_dir']
output_dir = paths['pts_output_dir']

os.makedirs(output_dir, exist_ok=True)
#--------------------------------------------------------------------------------80


#################################################################################80
# path settings
# coord
#json_path = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_unet-512x512_unet_ce_coord/test_gt_kpt.json"  # 替换为实际路径
#image_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/data/ISBI2015/RawImage/Test2Data"  # 替换为图片目录
#output_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_unet-512x512_unet_ce_coord/pts_img"  # 替换为输出目录
# heatmap
#json_path = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_unet-512x512_unet_ce_heatmap/test_gt_kpt.json"  # 替换为实际路径
#image_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/data/ISBI2015/RawImage/Test2Data"  # 替换为图片目录
#output_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_unet-512x512_unet_ce_heatmap/pts_img"  # 替换为输出目录
#--------------------------------------------------------------------------------80





#################################################################################80
# 绘制landmarks到图片，预测坐标、真实坐标，并在图片上画出
def draw_landmarks(json_path, image_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)#[-10:]
    
    for item in data:
        image_name = item["image_id"]
        target_pts = np.array(item["target_pts"]).reshape(-1, 2)
        pred_pts = np.array(item["pred_pts"]).reshape(-1, 2)
        
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in {image_dir}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_name}")
            continue
        
        # 绘制真实和预测点
        for idx, ((tx, ty), (px, py)) in enumerate(zip(target_pts, pred_pts), start=1):
            cv2.line(image, (int(tx), int(ty)), (int(px), int(py)), (0, 255, 0), 2)  # 连接线蓝色
            cv2.circle(image, (int(tx), int(ty)), 5, (0, 0, 255), -1)  # 真实点红色
            cv2.circle(image, (int(px), int(py)), 5, (255, 0, 0), -1)  # 预测点蓝色

            # 添加数字标注
            cv2.putText(image, str(idx), (int(tx) + 5, int(ty) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image, str(idx), (int(px) + 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 保存图片
        output_path = os.path.join(output_dir, f"gt_vs_pred_{image_name}")
        cv2.imwrite(output_path, image)
        #print(f"Saved {output_path}")


draw_landmarks(json_path, image_dir, output_dir)
print(f"- Visualizations of pts (pred vs gt) on raw image saved to {output_dir}")