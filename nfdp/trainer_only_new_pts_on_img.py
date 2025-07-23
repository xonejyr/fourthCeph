import os
import cv2
import numpy as np
import torch

from nfdp.utils import get_affine_transform, get_center_scale, transform_preds

def get_image_path(cfg, subset, img_id):
    """Build image file path with .bmp extension."""
    img_subdir = {'train': 'TrainingData', 'val': 'Test1Data', 'test': 'Test2Data'}.get(subset, 'val')
    img_id_str = str(img_id.item()) if torch.is_tensor(img_id) else str(img_id)
    # Append .bmp extension if not already present
    if not img_id_str.endswith('.bmp'):
        img_id_str = f'{img_id_str}.bmp'
    image_path = os.path.join(
        f'/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/data/ISBI2015/RawImage/{img_subdir}',
        img_id_str
    )
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")
    return image_path

def load_original_image(img_path):
    """Load and validate original image."""
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot load image: {img_path}")
    #print(f"Loaded image: {img_path}, Shape: {image.shape}")
    return image

def map_coords_to_image(coords, src_range, target_size):
    """
    Linearly map coordinates from src_range to target_size.
    coords: [num_joints, 2], src_range: [min, max], target_size: [width, height]
    """
    src_min, src_max = src_range
    target_w, target_h = target_size
    coords_mapped = coords.copy()
    # Map from [src_min, src_max] to [0, target_w] for x and [0, target_h] for y
    coords_mapped[:, 0] = ((coords[:, 0] - src_min) / (src_max - src_min)) * target_w
    coords_mapped[:, 1] = ((coords[:, 1] - src_min) / (src_max - src_min)) * target_h
    return coords_mapped

def convert_coords_srchw2targethw(coords, srchw, targethw):
    """convert the coords from srchw to targethw"""
    src_h, src_w = srchw
    target_h, target_w = targethw 
    target_ratio = target_w / target_h
    center, scale = get_center_scale(src_w, src_h, target_ratio)

    num_joints, _ = coords.shape

    coords_rawImage_SIZE = np.zeros((num_joints, 2))  

    #for i in batch_size:
    for j in range(num_joints):
        coords_rawImage_SIZE[j, :] = transform_preds(coords[j, :], center, scale, [target_w, target_h])
    return coords_rawImage_SIZE

# --- Visualization ---
def draw_keypoints(image, coords, color, label_prefix=''):
    """Draw keypoints on the image with labels."""
    overlay = image.copy()
    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(overlay, (x, y), 5, color, -1)
            cv2.putText(
                overlay, f'{label_prefix}#{i+1}', (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
    return overlay

def save_image(img, path):
    """Save image and log."""
    cv2.imwrite(path, img)
    #print(f"Saved image to: {path}")

# --- Main Visualization Function ---
def draw_keypoints_with_connections(image, gt_coords, pred_coords):
    """
    Draw ground truth and predicted keypoints with connecting lines and labels.

    Args:
        image: Input image (numpy array, BGR format).
        gt_coords: Ground truth coordinates [num_joints, 2] in RAW_IMAGE_SIZE.
        pred_coords: Predicted coordinates [num_joints, 2] in RAW_IMAGE_SIZE.

    Returns:
        Image with drawn keypoints and connections.
    """
    overlay = image.copy()
    for idx, (gt_coord, pred_coord) in enumerate(zip(gt_coords, pred_coords), start=1):
        tx, ty = int(gt_coord[0]), int(gt_coord[1])
        px, py = int(pred_coord[0]), int(pred_coord[1])

        # Check if coordinates are within image bounds
        if (0 <= tx < image.shape[1] and 0 <= ty < image.shape[0] and
            0 <= px < image.shape[1] and 0 <= py < image.shape[0]):
            # Draw connecting line (green)
            cv2.line(overlay, (tx, ty), (px, py), (0, 255, 0), 2)
            # Draw ground truth point (red)
            cv2.circle(overlay, (tx, ty), 5, (0, 0, 255), -1)
            # Draw predicted point (blue)
            cv2.circle(overlay, (px, py), 5, (255, 0, 0), -1)
            # Add labels
            cv2.putText(overlay, str(idx), (tx + 5, ty - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(overlay, str(idx), (px + 5, py - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return overlay

def visualize_keypoints(opt, cfg, vis_loader, model, output_dir):
    """
    Visualize predicted and ground truth keypoints overlaid on original images with connecting lines.

    Args:
        opt: Options (not used in this version but kept for compatibility).
        cfg: Configuration object with dataset parameters.
        vis_loader: Data loader for visualization dataset.
        model: Trained model for inference.
        output_dir: Directory to save output images.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Get image sizes from config
    heatmap_size = cfg.DATASET.PRESET.HEATMAP_SIZE  # [hm_w, hm_h]
    image_size = cfg.DATASET.PRESET.IMAGE_SIZE      # [img_w, img_h]
    raw_image_size = cfg.DATASET.PRESET.RAW_IMAGE_SIZE  # [raw_w, raw_h]

    subset = vis_loader.dataset.subset
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, batch_data in enumerate(vis_loader):
            if idx >= 8:  # Limit to first 8 batches
                break
            inps, labels, img_ids = batch_data
            inps = inps.to(device)

            # Model inference
            output = model(inps)
            pred_pts = output['pred_pts'].cpu().numpy()  # [batch_size, num_joints, 2]
            gt_pts = labels['target_uv'].view(pred_pts.shape).cpu().numpy()      # [batch_size, num_joints, 2]

            for b in range(inps.size(0)):
                img_id = img_ids[b]
                try:
                    img_path = get_image_path(cfg, subset, img_id)
                    orig_img = load_original_image(img_path)
                except ValueError as e:
                    print(e)
                    continue

                # Map predicted points from [-0.5, 0.5] to HEATMAP_SIZE
                pred_pts_mapped = map_coords_to_image(
                    pred_pts[b], src_range=[-0.5, 0.5], target_size=heatmap_size
                )
                # Map ground truth points from [-0.5, 0.5] to IMAGE_SIZE
                gt_pts_mapped = map_coords_to_image(
                    gt_pts[b], src_range=[-0.5, 0.5], target_size=image_size
                )

                # Transform predicted points from HEATMAP_SIZE to RAW_IMAGE_SIZE
                pred_pts_transformed = convert_coords_srchw2targethw(pred_pts_mapped, raw_image_size, heatmap_size)
                # Transform ground truth points from IMAGE_SIZE to RAW_IMAGE_SIZE
                gt_pts_transformed = convert_coords_srchw2targethw(gt_pts_mapped, raw_image_size, image_size)

                # Draw keypoints and connections
                overlay = draw_keypoints_with_connections(orig_img, gt_pts_transformed, pred_pts_transformed)

                # Save results
                img_id_str = str(img_id.item()) if torch.is_tensor(img_id) else str(img_id)
                save_path = os.path.join(output_dir, f'gt_vs_pred_{img_id_str}.jpg')
                cv2.imwrite(save_path, overlay)
                #print(f"Saved {save_path}")

    print(f"Visualization complete, outputs saved in: {output_dir}")