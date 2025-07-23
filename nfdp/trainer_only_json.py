import json
import os
import numpy as np
import torch
from nfdp.utils import get_center_scale, transform_preds, get_affine_transform
import cv2
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def compute_normalized_space_distribution(pred_pt, gt_pt, out_sigma, weights, dists, device):
    """Calculate distribution parameters in normalized space [-0.5, 0.5]."""
    k = len(dists)  # Number of joints
    normalized_dists = []
    
    # Ensure all inputs are on the same device
    pred_pt = pred_pt.to(device)
    gt_pt = gt_pt.to(device)
    out_sigma = out_sigma.to(device)
    
    for idx in range(k):
        w = weights[idx]
        dist = dists[idx]
        mu = dist.loc.to(device)  # bar_mu
        sigma = dist.scale.to(device)
        
        # Transform mu to predicted point space: pred_pt = gt_pt + bar_mu * out_sigma
        pred_mu = gt_pt + mu * out_sigma
        pred_sigma = sigma * out_sigma
        
        normalized_dists.append({
            'weight': w.item(),
            'mu': pred_mu.tolist(),
            'sigma': pred_sigma.tolist()
        })
    
    return normalized_dists

def compute_raw_space_distribution(gt_pt, weights, dists, out_sigma, cfg, device):
    """Calculate distribution parameters in raw image space."""
    k = len(dists)
    raw_dists = []
    
    # Ensure inputs are on the same device
    gt_pt = gt_pt.to(device)
    out_sigma = out_sigma.to(device)
    
    # Get size parameters
    RAW_H, RAW_W = cfg.DATASET.PRESET.RAW_IMAGE_SIZE
    HM_H, HM_W = cfg.DATASET.PRESET.HEATMAP_SIZE
    target_ratio_w_h = HM_W / HM_H
    
    # Calculate affine transformation parameters
    center, scale = get_center_scale(RAW_W, RAW_H, target_ratio_w_h, scale_mult=1.25)
    
    for idx in range(k):
        w = weights[idx]
        dist = dists[idx]
        mu = dist.loc.to(device)
        sigma = dist.scale.to(device)
        
        # Map mu from [-0.5, 0.5] to heatmap size
        pred_mu_hm = torch.zeros_like(mu)
        pred_mu_hm[0] = (gt_pt[0] + mu[0] * out_sigma[0] + 0.5) * HM_W
        pred_mu_hm[1] = (gt_pt[1] + mu[1] * out_sigma[1] + 0.5) * HM_H
        
        # Map mu from heatmap size to raw image size
        pred_mu_raw = transform_preds(pred_mu_hm.cpu().numpy(), center, scale, [HM_H, HM_W]) #
        pred_mu_raw = torch.from_numpy(pred_mu_raw).to(device)
        
        # Build covariance matrix in heatmap scale
        sigma_hm = sigma * out_sigma * torch.tensor([HM_W, HM_H], device=device)
        cov_hm = torch.diag(sigma_hm * sigma_hm)  # Diagonal covariance matrix
        
        # Extract linear part of affine transformation
        trans = get_affine_transform(center, scale, 0, [HM_H, HM_W], inv=1)
        linear_trans = torch.tensor(trans[:, :2], dtype=torch.float32, device=device)
        
        # Update covariance matrix
        cov_raw = linear_trans @ cov_hm @ linear_trans.T
        
        # Extract sigma from new covariance matrix
        sigma_raw = torch.sqrt(torch.diag(cov_raw))
        
        raw_dists.append({
            'weight': w.item(),
            'mu': pred_mu_raw.tolist(),
            'sigma': sigma_raw.tolist()
        })
    
    return raw_dists

def save_distribution_data(opt, cfg, vis_loader, model, output_dir):
    """Save mixed Gaussian distribution data."""
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get dataset subset
    subset = vis_loader.dataset.subset
    img_subdir = {
        'train': 'TrainingData',
        'val': 'Test1Data',
        'test': 'Test2Data'
    }.get(subset, 'TrainingData')
    
    # Get device from model
    device = next(model.parameters()).device

    with torch.no_grad():
        # Process one batch
        for idx, batch_data in enumerate(vis_loader):
            inps, target, img_ids = batch_data
            inps = inps.to(device)

            # Get model outputs
            output = model(inps)
            pred_pts = output.pred_pts
            basis_weights = output.basis_weights
            basis_dists = output.basis_dists
            out_sigma = output.sigma
            # Process each image
            for b in range(inps.size(0)):
                img_id = img_ids[b]
                img_id_str = str(img_id.item()) if torch.is_tensor(img_id) else str(img_id)
                # Get ground truth
                gt_pts = target['target_uv'][b].reshape(-1, 2).to(device)  # [-0.5, 0.5]
                # Prepare data to save
                sample_data = {
                    'image_id': img_id_str,
                    'out_sigma': out_sigma.tolist(),
                    'landmarks': []
                }
                # Process each keypoint
                for j in range(model.num_joints):
                    # Get weights and distributions for this keypoint
                    w_j = basis_weights[b, j].to(device)  # [max_bases]
                    k_j = len(basis_dists[j])  # Number of distributions used
                    # Get top-k components
                    top_k_indices = torch.argsort(w_j, descending=True)[:k_j]
                    weights_j = w_j[top_k_indices]
                    dists_j = [basis_dists[j][idx] for idx in range(k_j)]
                    # Get predicted point and sigma
                    pred_pt = pred_pts[b, j].to(device)
                    sigma_j = out_sigma[b, j].to(device)
                    gt_pt = gt_pts[j]
                    # Calculate distribution parameters
                    norm_dists = compute_normalized_space_distribution(
                        pred_pt, gt_pt, sigma_j, weights_j, dists_j, device)
                    raw_dists = compute_raw_space_distribution(
                        gt_pt, weights_j, dists_j, sigma_j, cfg, device)
                    # Collect data for this keypoint
                    landmark_data = {
                        'joint_id': j,
                        'gt_coord': gt_pt.tolist(),
                        'pred_coord': pred_pt.tolist(),
                        'out_sigma': sigma_j.tolist(),
                        'learned_distribution_bar_mu': {
                            'k': k_j,
                            'weights': weights_j.tolist(),
                            'distributions': [
                                {
                                    'mu': dist.loc.to(device).tolist(),
                                    'sigma': dist.scale.to(device).tolist()
                                } for dist in dists_j
                            ]
                        },
                        'pred_pt_distribution_norm05_space': norm_dists,
                        'pred_pt_distribution_raw_space': raw_dists
                    }
                    sample_data['landmarks'].append(landmark_data)
                # Save data for this sample
                save_path = os.path.join(output_dir, f'{img_id_str}_data.json')
                with open(save_path, 'w') as f:
                    json.dump(sample_data, f, indent=4)
                print(f"- Saved {save_path}")

            if idx >= 8:  # Process only the first batch
                break
    return


def visualize_distribution_heatmap(json_path, vis_loader, output_dir, alpha=0.5):
    """
    为每个标志点生成分布热图并叠加到原图上。
    
    Args:
        json_path: 保存的分布数据JSON文件路径
        vis_loader: 原始图像loader
        output_dir: 输出图像保存目录
        alpha: 热图透明度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建局部裁剪保存目录
    #crop_dir = os.path.join(output_dir, 'local_crops')
    #os.makedirs(crop_dir, exist_ok=True)

    subset = vis_loader.dataset.subset
    img_subdir = {
        'train': 'TrainingData',
        'val': 'Test1Data',
        'test': 'Test2Data'
    }.get(subset, 'Test1Data')
    
    # 读取JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_id = data['image_id']
    # 去掉可能存在的.bmp后缀
    image_id_base = image_id.replace('.bmp', '')
    
    # 为每个图像创建独立的局部裁剪文件夹
    image_crop_dir = os.path.join(output_dir, f'{image_id_base}_localcrop')
    os.makedirs(image_crop_dir, exist_ok=True)
    
    # 读取原始图像（假设图像格式为jpg/png）
    image_path = os.path.join(f'/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/data/ISBI2015/RawImage/{img_subdir}', f'{image_id}')
    if not os.path.exists(image_path):
        raise("image_path is not in existence.")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot find image: {image_path}")
    
    height, width = image.shape[:2]
    
    # 创建热图画布
    heatmap = np.zeros((height, width))
    
    # 为每个标志点生成分布热图
    for landmark_idx, landmark in enumerate(data['landmarks']):
        raw_dists = landmark['pred_pt_distribution_raw_space']
        # 获取实际的关键点ID，显示时加1（从1开始显示）
        joint_id = landmark['joint_id'] + 1
        
        # 确保权重归一化
        total_weight = sum(dist['weight'] for dist in raw_dists)
        if len(raw_dists) == 1:
            normalized_weights = [1.0]
        else:
            normalized_weights = [dist['weight'] / total_weight for dist in raw_dists] if total_weight > 0 else [1.0 / len(raw_dists)] * len(raw_dists)
        
        # 创建网格点
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        
        # 累加每个高斯分布的贡献
        landmark_heatmap = np.zeros((height, width))
        pred_mu = None
        for dist_idx, (dist, norm_weight) in enumerate(zip(raw_dists, normalized_weights)):
            mu = dist['mu']
            if dist_idx == 0:  # 保存第一个mu用于标记和裁剪
                pred_mu = mu
            sigma = dist['sigma']
            
            # 创建协方差矩阵（假设分布是独立的）
            cov = np.diag(np.square(sigma))
            
            # 生成高斯分布
            rv = multivariate_normal(mu, cov)
            # 使用归一化后的权重
            landmark_heatmap += norm_weight * rv.pdf(pos)
        
        # 无论热图最大值如何，都进行标记
        pt_x, pt_y = int(pred_mu[0]), int(pred_mu[1])
        cv2.putText(image, f'#{joint_id}', (pt_x-20, pt_y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 归一化并添加到总热图
        if landmark_heatmap.max() > 1e-10:  # 使用更小的阈值
            landmark_heatmap = landmark_heatmap / landmark_heatmap.max()
            heatmap += landmark_heatmap
            
            # 裁剪局部区域
            crop_size = 50  # 裁剪区域大小
            x1 = max(0, pt_x - crop_size//2)
            y1 = max(0, pt_y - crop_size//2)
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)
            
            # 裁剪原图和热图
            crop_img = image[y1:y2, x1:x2].copy()
            crop_heatmap = landmark_heatmap[y1:y2, x1:x2]
            
            # 创建局部热图可视化
            crop_heatmap_colored = cv2.applyColorMap(
                (crop_heatmap * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            crop_overlay = cv2.addWeighted(crop_img, 1-alpha, crop_heatmap_colored, alpha, 0)
            
            # 在局部图上也添加标志点编号（使用joint_id+1）
            cv2.putText(crop_overlay, f'#{joint_id}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存局部裁剪结果到图像特定的文件夹
            crop_path = os.path.join(image_crop_dir, f'landmark_{joint_id}.bmp')
            cv2.imwrite(crop_path, crop_overlay)
    
    # 归一化总热图
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # 创建彩色热图
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 叠加热图到原图
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    # 保存结果为BMP格式
    output_path = os.path.join(output_dir, f'{image_id[:-4]}_heatmap.bmp')
    cv2.imwrite(output_path, overlay)
    print(f"保存可视化结果到: {output_path}")
    #print(f"局部裁剪结果保存在: {crop_dir}")

# 使用示例：
def visualize_all_distributions(distribution_dir, vis_loader, output_dir):
    """
    处理目录中的所有分布数据文件。
    
    Args:
        distribution_dir: 包含JSON文件的目录
        vis_loader: 原始图像loader
        output_dir: 输出图像保存目录
    """
    for filename in os.listdir(distribution_dir):
        if filename.endswith('_data.json'):
            json_path = os.path.join(distribution_dir, filename)
            visualize_distribution_heatmap(json_path, vis_loader, output_dir)