import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSS

def wing_loss(pred, target, w=10.0, epsilon=2.0):
    w = torch.tensor(w, device=pred.device, dtype=pred.dtype)
    epsilon = torch.tensor(epsilon, device=pred.device, dtype=pred.dtype)
    
    diff = torch.abs(pred - target)
    # 修改处1：限制 diff 的最大值，避免溢出
    diff = torch.clamp(diff, max=1e6)  # 防止 diff 过大
    C = w * (1.0 - torch.log(1.0 + w / epsilon))
    # 修改处2：确保 log 的输入 >= 1e-6，防止 log(0) 或负数
    log_term = torch.log(torch.clamp(1.0 + diff / epsilon, min=1e-6))
    loss = torch.where(diff < w, w * log_term, diff - C)
    return loss.mean()

@LOSS.register_module
class AGD2UNetLoss(nn.Module):
    def __init__(self, **cfg):
        super(AGD2UNetLoss, self).__init__()
        
        self.lambda_heatmap = 0.4
        self.lambda_coord = 0.4
        self.lambda_anatomy = 0.2
        self.mse = nn.MSELoss()

    def compute_distance_matrix(self, coords):
        # coords: [B, 19, 2]
        batch_size = coords.size(0)
        num_points = coords.size(1)
        coords_expanded_1 = coords.unsqueeze(2)  # [B, 19, 1, 2]
        coords_expanded_2 = coords.unsqueeze(1)  # [B, 1, 19, 2]
        dist = torch.sqrt(((coords_expanded_1 - coords_expanded_2) ** 2).sum(dim=-1))  # [B, 19, 19]
        return dist

    def forward(self, output, labels):
        H_bone_pred = output['heatmap_bone']
        H_soft_pred = output['heatmap_soft']
        P_pred = output['pred_pts']
        H_bone_gt = labels['target_hm_bone']
        H_soft_gt = labels['target_hm_soft']
        P_gt = labels['target_uv'].reshape(-1, 19, 2)

        # 热图损失
        L_heatmap = self.mse(H_bone_pred, H_bone_gt) + self.mse(H_soft_pred, H_soft_gt)

        # 坐标损失
        L_coord = wing_loss(P_pred, P_gt)

        # 解剖约束损失
        D_pred = self.compute_distance_matrix(P_pred)
        D_gt = self.compute_distance_matrix(P_gt)
        # 修改处：添加平滑项并规范化
        eps = 1e-6
        D_pred = (D_pred + eps) / (D_pred + eps).sum(dim=-1, keepdim=True)  # 归一化为概率分布
        D_gt = (D_gt + eps) / (D_gt + eps).sum(dim=-1, keepdim=True)
        L_anatomy = F.kl_div(D_pred.log(), D_gt, reduction='batchmean')

        # 总损失
        loss = self.lambda_heatmap * L_heatmap + self.lambda_coord * L_coord + self.lambda_anatomy * L_anatomy
        return loss

## 损失函数示例
# loss_fn = AGD2UNetLoss()