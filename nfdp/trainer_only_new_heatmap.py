import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nfdp.utils import get_affine_transform, get_center_scale

# --- Data Loading ---
def get_image_path(cfg, subset, img_id):
    """Build image file path."""
    img_subdir = {'train': 'TrainingData', 'val': 'Test1Data', 'test': 'Test2Data'}.get(subset, 'val')
    img_id_str = str(img_id.item()) if torch.is_tensor(img_id) else str(img_id)
    image_path = os.path.join(
        f'/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/data/ISBI2015/RawImage/{img_subdir}',
        f'{img_id_str}'
    )
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")
    return image_path

def load_original_image(img_path):
    """Load and validate original image."""
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot load image: {img_path}")
    print(f"Loaded image: {img_path}, Shape: {image.shape}")
    return image

# --- Heatmap Processing ---
def normalize_heatmap(heatmap, norm_type='softmax'):
    """
    Normalize heatmap to [0, 1] over the last two dimensions (hm_h, hm_w).
    
    Args:
        heatmap: NumPy array or PyTorch tensor of shape (batch_size, num_joints, hm_h, hm_w)
        norm_type: 'softmax' or 'minmaxNorm'
    
    Returns:
        Normalized heatmap with the same type and shape as input
    """
    # Determine input type and ensure it's a tensor for processing
    is_numpy = isinstance(heatmap, np.ndarray)
    if is_numpy:
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
    elif not isinstance(heatmap, torch.Tensor):
        raise TypeError("Input must be a NumPy array or PyTorch tensor")
    
    batch_size, num_joints, hm_h, hm_w = heatmap.shape
    
    if norm_type == 'softmax':
        # Reshape to (batch_size * num_joints, hm_h * hm_w) for softmax
        heatmap_flat = heatmap.view(batch_size * num_joints, -1)
        heatmap_norm = F.softmax(heatmap_flat, dim=1)
        heatmap = heatmap_norm.view(batch_size, num_joints, hm_h, hm_w)
    
    elif norm_type == 'minmaxNorm':
        # Reshape to process each heatmap independently
        heatmap_flat = heatmap.view(batch_size * num_joints, hm_h, hm_w)
        heatmap_norm = torch.zeros_like(heatmap_flat)
        
        for i in range(batch_size * num_joints):
            hmap = heatmap_flat[i]
            hmin, hmax = hmap.min(), hmap.max()
            if hmax > hmin:
                heatmap_norm[i] = (hmap - hmin) / (hmax - hmin)
            else:
                print(f"Warning: Invalid heatmap at index {i}, max={hmax}, min={hmin}")
                heatmap_norm[i] = torch.zeros_like(hmap)
        
        heatmap = heatmap_norm.view(batch_size, num_joints, hm_h, hm_w)
    
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")
    
    # Convert back to NumPy if input was NumPy
    if is_numpy:
        heatmap = heatmap.numpy()
    
    return heatmap

import numpy as np
import cv2

def resize_feature_srchw2targethw(feature, src_hw, target_hw):
    """Resize feature from src_hw to target_hw for each [hm_h, hm_w] sub-feature."""
    src_h, src_w = src_hw
    target_h, target_w = target_hw
    target_ratio = target_w / target_h
    center, scale = get_center_scale(src_w, src_h, target_ratio)
    trans = get_affine_transform(center, scale, 0, [target_w, target_h], inv=1)

    # 获取输入特征的形状
    batch_size, num_joints, _, _ = feature.shape
    
    # 初始化输出数组
    feature_resized = np.zeros((batch_size, num_joints, src_h, src_w), dtype=np.float32)

    # 遍历批次和通道，处理每个子特征
    for b in range(batch_size):
        for c in range(num_joints):
            sub_feature = feature[b, c]  # 提取单个热图 [hm_h, hm_w]
            sub_feature = sub_feature.astype(np.float32)  # 确保类型为 float32
            resized_sub_feature = cv2.warpAffine(
                sub_feature,
                trans,
                (int(src_w), int(src_h)),
                flags=cv2.INTER_LINEAR
            )
            feature_resized[b, c] = resized_sub_feature  # 存储调整后的子特征

    return feature_resized # 确保非负值

def heatmaps_add2one(heatmaps, batch_idx):
    """Generate combined heatmap for all joints."""
    batch_size, num_joints, raw_h, raw_w = heatmaps.shape
    total_heatmap = np.zeros((raw_h, raw_w), dtype=np.float32)
    for j in range(num_joints):
        heatmap = heatmaps[batch_idx, j]
        total_heatmap += heatmap
    return total_heatmap

# --- Visualization ---
def color_total_heatmap(total_heatmap, enhance_factor=1.5):
    """Map total heatmap to [0, 255] and apply colormap."""
    
    total_heatmap = total_heatmap * enhance_factor  # Enhance intensity
    total_heatmap = np.clip(total_heatmap, 0, 1)
    heatmap_mapped = (total_heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_mapped, cv2.COLORMAP_JET)
    return heatmap_colored

def add_keypoint_labels(overlay, heatmaps, batch_idx, trans, raw_w, raw_h, num_joints, norm_type='softmax'):
    """Add keypoint labels to the overlay image."""
    for j in range(num_joints):
        heatmap = heatmaps[batch_idx, j].cpu().numpy()
        heatmap = normalize_heatmap(heatmap, norm_type)
        heatmap_resized = resize_heatmap(heatmap, trans, raw_w, raw_h)
        if heatmap_resized.max() > 0:
            max_loc = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
            cv2.putText(
                overlay, f'#{j+1}', (max_loc[1], max_loc[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
    return overlay

def save_image(img, path):
    """Save image and log."""
    cv2.imwrite(path, img)
    print(f"Saved image to: {path}")

# --- Main Visualization Function ---
def visualize_heatmap_distributions(opt, cfg, vis_loader, model, output_dir, norm_type='softmax'):
    """
    Visualize heatmaps overlaid on original images.

    Args:
        opt: Options (not used in this version but kept for compatibility).
        cfg: Configuration object with dataset parameters.
        vis_loader: Data loader for visualization dataset.
        model: Trained model for inference.
        output_dir: Directory to save output images.
        norm_type: Normalization type ('softmax' or 'minmaxNorm').
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Compute inverse affine transformation (HEATMAP_SIZE to RAW_IMAGE_SIZE)
    subset = vis_loader.dataset.subset

    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, batch_data in enumerate(vis_loader):
            if idx >= 8:  # Limit to first 8 batches
                break
            inps, _, img_ids = batch_data
            inps = inps.to(device)

            # Model inference
            output = model(inps)
            heatmaps = output['heatmap'].cpu()
            heatmaps_normalized = np.array(normalize_heatmap(heatmaps, norm_type='softmax'))
            print("- output of heatmap size got.")


            heatmap_resized2rawsize = resize_feature_srchw2targethw(heatmaps_normalized, src_hw=cfg.DATASET.PRESET.RAW_IMAGE_SIZE, target_hw=cfg.DATASET.PRESET.HEATMAP_SIZE)
            print("- output of raw image size got.")


            

            for b in range(inps.size(0)):
                img_id = img_ids[b]
                try:
                    img_path = get_image_path(cfg, subset, img_id)
                    orig_img = load_original_image(img_path)
                except ValueError as e:
                    print(e)
                    continue

                # Generate and process heatmap
                total_heatmap = heatmaps_add2one(heatmap_resized2rawsize, b)
                heatmap_colored = color_total_heatmap(total_heatmap, enhance_factor=2)
                print("- output of coloed raw image size got.")

                # Overlay heatmap on original image
                alpha = 0.4  # Adjusted for better balance
                overlay = cv2.addWeighted(orig_img, 1 - alpha, heatmap_colored, alpha, 0)
                print("- output of final got.")

                ## Add keypoint labels
                #overlay = add_keypoint_labels(
                #    overlay, heatmaps, b, trans, RAW_W, RAW_H, model.num_joints, norm_type
                #)

                # Save results
                img_id_str = str(img_id.item()) if torch.is_tensor(img_id) else str(img_id)
                # save_path = os.path.join(output_dir, f'{img_id_str}_heatmap_with_labels.jpg')
                # save_image(overlay, save_path)

                # Save heatmap only for debugging
                save_path = os.path.join(output_dir, f'{img_id_str}_heatmap_only.jpg')
                save_image(overlay, save_path)

    print(f"Visualization complete, outputs saved in: {output_dir}")
