import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSS

def contrastive_loss(F_bone, F_soft, tau=0.5):
    batch_size = F_bone.size(0)
    labels = torch.arange(batch_size).to(F_bone.device)
    
    # 计算相似度矩阵
    F_bone_norm = F.normalize(F_bone, dim=-1)
    F_soft_norm = F.normalize(F_soft, dim=-1)
    F_all = torch.cat([F_bone_norm, F_soft_norm], dim=0)
    sim_matrix = torch.mm(F_all, F_all.t()) / tau
    
    # 构造正样本和负样本
    sim_matrix = torch.exp(sim_matrix)
    mask = torch.eye(2 * batch_size, device=F_bone.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, 0)
    
    # 计算对比损失
    pos_sim = torch.diag(sim_matrix[:batch_size, batch_size:])
    neg_sim = sim_matrix.sum(dim=1) - torch.diag(sim_matrix)
    loss = -torch.log(pos_sim / neg_sim).mean()
    return loss

def wing_loss(pred, target, w=10.0, epsilon=2.0):
    diff = torch.abs(pred - target)
    C = w * (1.0 - torch.log(1.0 + w / epsilon))
    loss = torch.where(diff < w, w * torch.log(1.0 + diff / epsilon), diff - C)
    return loss.mean()


@LOSS.register_module
class AICDUNetLoss(nn.Module):
    def __init__(self, **cfg):
        super(AICDUNetLoss, self).__init__()
        self.lambda_contrast=0.3
        self.lambda_heatmap=0.3
        self.lambda_coord=0.2
        self.lambda_causal=0.2

        #self.lambda_contrast = lambda_contrast
        #self.lambda_heatmap = lambda_heatmap
        #self.lambda_coord = lambda_coord
        #self.lambda_causal = lambda_causal
        self.mse = nn.MSELoss()

    def forward(self, output, labels):
        F_bone = output['F_bone']
        F_soft = output['F_soft']
        H_bone_pred = output['heatmap_bone']
        H_soft_pred = output['heatmap_soft']
        P_pred = output['pred_pts']
        H_bone_gt = labels['target_hm_bone']
        H_soft_gt = labels['target_hm_soft']
        P_gt = labels['target_uv'].reshape(-1, 19, 2)

        # 对比损失
        L_contrast = contrastive_loss(F_bone, F_soft)

        # 热图损失
        L_heatmap = self.mse(H_bone_pred, H_bone_gt) + self.mse(H_soft_pred, H_soft_gt)

        # 坐标损失
        L_coord = wing_loss(P_pred, P_gt)

        # 因果约束损失（简单实现为骨组织对软组织的影响）
        P_bone_pred = P_pred[:, :12, :]  # 骨组织
        P_soft_pred = P_pred[:, 12:, :]  # 软组织
        P_soft_gt = P_gt[:, 12:, :]
        L_causal = F.mse_loss(P_soft_pred, P_soft_gt)

        # 总损失
        loss = (self.lambda_contrast * L_contrast +
                self.lambda_heatmap * L_heatmap +
                self.lambda_coord * L_coord +
                self.lambda_causal * L_causal)
        
        return loss

# 损失函数示例
#loss_fn = AICDUNetLoss()