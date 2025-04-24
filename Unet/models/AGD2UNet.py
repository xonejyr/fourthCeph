import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 需要安装torch-geometric

from .components.unet_parts import *
from ..utils import Softmax_Integral

from ..builder import MODEL

from easydict import EasyDict

"""
AGD²U-Net（Anatomy-Guided Dual Differential U-Net）

核心创新：通过差分特征分离器和解剖关系图网络（GNN），利用骨-软组织特征的差异性和解剖学依赖关系，提升标志点检测精度。
重点：差分机制和解剖约束，强调特征分离和几何一致性。
理论基础：基于解剖学先验和信息论（互信息、KL散度）优化。
"""

# AGD²U-Net模型: hm 256x256
@MODEL.register_module
class AGD2UNet(nn.Module):
    def __init__(self, bilinear=True, **cfg):
        super(AGD2UNet, self).__init__()
        self._preset_cfg = cfg['PRESET']

        self.bone_joints = self._preset_cfg['NUM_BONE_JOINTS']  # 15
        self.soft_joints = self._preset_cfg['NUM_SOFT_JOINTS']  # 4
        self.hm_height = self._preset_cfg['HEATMAP_SIZE'][0]  # 256
        self.hm_width = self._preset_cfg['HEATMAP_SIZE'][1]   # 256
        self.total_joints = self.bone_joints + self.soft_joints  # 19
        self.in_channels = cfg['IN_CHANNELS']  # 3

        # 共享特征提取器
        self.shared_extractor = nn.Sequential(
            DoubleConv(self.in_channels, 64),
            Down(64, 128)
        )

        # 差分特征分离器
        self.conv_bone = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_soft = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # 骨组织分支
        self.bone_down1 = Down(128, 256)
        self.bone_down2 = Down(256, 512)
        self.bone_up1 = Up(768, 256, bilinear)
        self.bone_up2 = Up(384, 128, bilinear)
        self.bone_out = OutConv(128, self.bone_joints)

        # 软组织分支
        self.soft_down1 = Down(128, 256)
        self.soft_up1 = Up(384, 128, bilinear)
        self.soft_out = OutConv(128, self.soft_joints)

        # 软积分模块
        self.integral = Softmax_Integral(self.total_joints, self.hm_width, self.hm_height)

        # 解剖关系图网络（GAT）
        self.gat = GATConv(2, 2, heads=4, concat=False)

        # 预定义邻接矩阵
        self.adj_matrix = torch.ones((self.total_joints, self.total_joints)) - torch.eye(self.total_joints)
        self.edge_index = torch.nonzero(self.adj_matrix, as_tuple=False).t().contiguous().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 共享特征提取
        F_shared = self.shared_extractor(x)

        # 差分特征分离
        F_bone = self.conv_bone(F_shared)
        F_soft = self.conv_soft(F_shared)
        F_diff = F_bone - F_soft

        # 骨组织分支
        bone_x1 = F_bone + F_diff
        bone_x2 = self.bone_down1(bone_x1)
        bone_x3 = self.bone_down2(bone_x2)
        bone_x = self.bone_up1(bone_x3, bone_x2)
        bone_x = self.bone_up2(bone_x, bone_x1)
        H_bone = self.bone_out(bone_x)
        # 修改处1：上采样 H_bone 到 256 × 256
        H_bone = F.interpolate(H_bone, size=(self.hm_height, self.hm_width), mode='bilinear', align_corners=False)

        # 软组织分支
        soft_x1 = F_soft + F_diff
        soft_x2 = self.soft_down1(soft_x1)
        soft_x = self.soft_up1(soft_x2, soft_x1)
        H_soft = self.soft_out(soft_x)
        # 修改处2：上采样 H_soft 到 256 × 256
        H_soft = F.interpolate(H_soft, size=(self.hm_height, self.hm_width), mode='bilinear', align_corners=False)

        # 合并热图
        H_all = torch.cat([H_bone, H_soft], dim=1)

        # 软积分转换为坐标
        P_init = self.integral(H_all)  # [B, 19, 2]

        # 解剖关系图网络（GAT）
        batch_size = x.size(0)
        P_flat = P_init.view(-1, 2)
        num_nodes_per_graph = self.total_joints
        edge_index_batch = self.edge_index.clone()
        for i in range(1, batch_size):
            offset = i * num_nodes_per_graph
            edge_index_batch = torch.cat([edge_index_batch, self.edge_index + offset], dim=1)
        edge_index_batch = edge_index_batch.to(P_flat.device)
        P_refined_flat = self.gat(P_flat, edge_index_batch)
        P_refined = P_refined_flat.view(batch_size, self.total_joints, 2)

        output = EasyDict(
            pred_pts=P_refined,
            heatmap_bone=H_bone,
            heatmap_soft=H_soft
        )
        return output