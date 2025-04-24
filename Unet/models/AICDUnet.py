"""
AICDU-Net（Anatomy-Informed Contrastive Dual U-Net with Causal Reasoning）：
核心创新：引入因果推理和对比学习，显式建模骨-软组织标志点的因果关系，并通过对比损失增强特征区分性。
重点：因果性和对比性，强调解剖因果关系和特征分离的动态适应。
理论基础：基于因果推理（Do-Calculus）和对比学习（SimCLR）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 需要安装torch-geometric

from .components.unet_parts import *
from ..utils import Softmax_Integral

from ..builder import MODEL

from easydict import EasyDict


# AICDU-Net模型
@MODEL.register_module
class AICDUNet(nn.Module):
    def __init__(self, bilinear=True, **cfg):
        super(AICDUNet, self).__init__()
        self._preset_cfg = cfg['PRESET']

        self.bone_joints = self._preset_cfg['NUM_BONE_JOINTS']
        self.soft_joints = self._preset_cfg['NUM_SOFT_JOINTS']
        self.hm_height = self._preset_cfg['HEATMAP_SIZE'][0]
        self.hm_width = self._preset_cfg['HEATMAP_SIZE'][1]
        self.total_joints = self.bone_joints + self.soft_joints  # 19
        self.in_channels = cfg['IN_CHANNELS']

        # 因果解剖编码器（简单实现为卷积+全连接）
        self.cae = nn.Sequential(
            DoubleConv(self.in_channels, 64),
            Down(64, 128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.cae_bone = nn.Linear(128 * 4 * 4, 128)
        self.cae_soft = nn.Linear(128 * 4 * 4, 128)

        # 对比特征分离器
        self.proj_bone = nn.Linear(128, 64)
        self.proj_soft = nn.Linear(128, 64)

        # 骨组织分支（深而窄）
        self.bone_down1 = Down(64, 256)
        self.bone_down2 = Down(256, 512)
        self.bone_up1 = Up(512, 256, bilinear)
        self.bone_up2 = Up(256, 64, bilinear)
        self.bone_out = OutConv(64, self.bone_joints)

        # 软组织分支（浅而宽）
        self.soft_down1 = Down(64, 256)
        self.soft_up1 = Up(256, 64, bilinear)
        self.soft_out = OutConv(64, self.soft_joints)

        # 软积分模块
        self.integral = Softmax_Integral(self.total_joints, self.hm_width, self.hm_height)

        # 因果干预模块（简单实现为MLP）
        self.cim_bone_to_soft = nn.Sequential(
            nn.Linear(self.bone_joints * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.soft_joints * 2)
        )

    def forward(self, x):
        # 因果解剖编码器
        feat = self.cae(x)  # [B, 128, 4, 4]
        feat_flat = feat.view(feat.size(0), -1)
        Z_bone = self.cae_bone(feat_flat)  # [B, 128]
        Z_soft = self.cae_soft(feat_flat)  # [B, 128]

        # 对比特征分离器
        F_bone = self.proj_bone(Z_bone)  # [B, 64]
        F_soft = self.proj_soft(Z_soft)  # [B, 64]

        # 为了适配U-Net分支，将特征扩展为空间特征图
        F_bone_expanded = F_bone.view(-1, 64, 1, 1).expand(-1, 64, x.size(2)//4, x.size(3)//4)
        F_soft_expanded = F_soft.view(-1, 64, 1, 1).expand(-1, 64, x.size(2)//4, x.size(3)//4)

        # 骨组织分支
        bone_x1 = F_bone_expanded
        bone_x2 = self.bone_down1(bone_x1)
        bone_x3 = self.bone_down2(bone_x2)
        bone_x = self.bone_up1(bone_x3, bone_x2)
        bone_x = self.bone_up2(bone_x, bone_x1)
        H_bone = self.bone_out(bone_x)

        # 软组织分支
        soft_x1 = F_soft_expanded
        soft_x2 = self.soft_down1(soft_x1)
        soft_x = self.soft_up1(soft_x2, soft_x1)
        H_soft = self.soft_out(soft_x)

        # 合并热图
        H_all = torch.cat([H_bone, H_soft], dim=1)  # [B, 19, H'', W'']

        # 软积分转换为坐标
        P_init = self.integral(H_all)  # [B, 19, 2]

        # 因果干预模块
        P_bone = P_init[:, :self.bone_joints, :]  # [B, 12, 2]
        P_soft = P_init[:, self.bone_joints:, :]  # [B, 7, 2]
        P_bone_flat = P_bone.view(P_bone.size(0), -1)
        P_soft_adjusted = self.cim_bone_to_soft(P_bone_flat).view(P_soft.size())
        P_soft_refined = P_soft + P_soft_adjusted
        P_refined = torch.cat([P_bone, P_soft_refined], dim=1)  # [B, 19, 2]
        
        output=EasyDict(
            pred_pts=P_refined,
            heatmap_bone=H_bone,     # [batch_size, 12, hm_height, hm_width]
            heatmap_soft=H_soft,
            F_bone=F_bone,
            F_soft=F_soft,
            coords=P_init  # [B, 19, 2]  # 因果解��得到的初始坐标，用于干预模块的因果推理
        )
        return output
    