import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from easydict import EasyDict
from .components.unet_parts import Up, OutConv  # 导入你的 UNet 模块

from Unet.utils import Softmax_Integral
import dgl
from dgl.nn import GraphConv
from Unet.builder import MODEL

""" UNet + ResNet + FPN + Hierarchical Graph 骨、软分离  """
@MODEL.register_module
class HybridUResFPNWithHIG(nn.Module):
    def __init__(self, bilinear=False, norm_layer=nn.BatchNorm2d, **cfg):
        super(HybridUResFPNWithHIG, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.bilinear = bilinear
        self._norm_layer = norm_layer

        # 骨组织和软组织区分
        self.bone_num_joints = self._preset_cfg['NUM_JOINTS_BONE']  # 15
        self.soft_num_joints = self._preset_cfg['NUM_JOINTS_SOFT']  # 4
        self.bone_indices = torch.tensor(self._preset_cfg['BONE_INDICES'])
        self.soft_indices = torch.tensor(self._preset_cfg['SOFT_INDICES'])

        # ResNet 配置
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        resnet = eval(f"models.resnet{cfg['NUM_LAYERS']}(pretrained={cfg.get('PRETRAINED_RIGHT', True)})")
        self.feature_channel = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512], 34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048], 101: [256, 512, 1024, 2048], 152: [256, 512, 1024, 2048]
        }[cfg['NUM_LAYERS']]

        # 编码器：ResNet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        # FPN 模块
        self.fpn_conv4 = nn.Conv2d(self.decoder_feature_channel[3], 256, kernel_size=1)  # P4
        self.fpn_conv3 = nn.Conv2d(self.decoder_feature_channel[2], 256, kernel_size=1)  # P3
        self.fpn_conv2 = nn.Conv2d(self.decoder_feature_channel[1], 256, kernel_size=1)  # P2
        self.fpn_smooth = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # 层次图模块
        self.bone_proj = nn.Linear(self.feature_channel, 256)  # 骨组织子图
        self.soft_proj = nn.Linear(self.feature_channel, 256)  # 软组织子图
        self.bone_gcn = GraphConv(256, 256)
        self.soft_gcn = GraphConv(256, 256)

        # 解码器：U-Net 风格
        factor = 2 if bilinear else 1
        self.up1 = Up(self.decoder_feature_channel[3] + 256, 256 // factor, bilinear)  # P4 + C4
        self.up2 = Up(self.decoder_feature_channel[2] + 256 // factor, self.decoder_feature_channel[1] // factor, bilinear)  # P3 + C3
        self.up3_bone = Up(self.decoder_feature_channel[1] + self.decoder_feature_channel[1] // factor, self.decoder_feature_channel[0], bilinear)  # P2 + C2 (骨)
        self.up3_soft = Up(self.decoder_feature_channel[1] + self.decoder_feature_channel[1] // factor, self.decoder_feature_channel[0], bilinear)  # P2 + C2 (软)
        self.bone_out = nn.Conv2d(self.decoder_feature_channel[0], self.bone_num_joints, kernel_size=1)
        self.soft_out = nn.Conv2d(self.decoder_feature_channel[0], self.soft_num_joints, kernel_size=1)

        # Softmax_Integral
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)

    def _build_graph(self, batch_size):
        # 简单全连接图，可根据解剖学关系优化
        bone_graph = dgl.graph((torch.arange(self.bone_num_joints).repeat(self.bone_num_joints),
                               torch.arange(self.bone_num_joints).repeat(self.bone_num_joints, 1).flatten()))
        soft_graph = dgl.graph((torch.arange(self.soft_num_joints).repeat(self.soft_num_joints),
                               torch.arange(self.soft_num_joints).repeat(self.soft_num_joints, 1).flatten()))
        return dgl.batch([bone_graph] * batch_size), dgl.batch([soft_graph] * batch_size)

    def forward(self, x, target_uv=None):
        BATCH_SIZE = x.shape[0]

        # 编码器：ResNet
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)  # [B, 64/256, H/4, W/4]
        c3 = self.layer2(c2)  # [B, 128/512, H/8, W/8]
        c4 = self.layer3(c3)  # [B, 256/1024, H/16, W/16]
        c5 = self.layer4(c4)  # [B, 512/2048, H/32, W/32]

        # FPN：多尺度特征融合
        p4 = self.fpn_conv4(c5)  # [B, 256, H/32, W/32]
        p3 = self.fpn_conv3(c4) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=True)
        p3 = self.fpn_smooth(p3)  # [B, 256, H/16, W/16]
        p2 = self.fpn_conv2(c3) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)
        p2 = self.fpn_smooth(p2)  # [B, 256, H/8, W/8]

        # 层次图：骨/软组织子图
        bone_feats = self.bone_proj(c5.mean(dim=(2, 3))).unsqueeze(1).repeat(1, self.bone_num_joints, 1)  # [B, 15, 256]
        soft_feats = self.soft_proj(c5.mean(dim=(2, 3))).unsqueeze(1).repeat(1, self.soft_num_joints, 1)  # [B, 4, 256]
        bone_graph, soft_graph = self._build_graph(BATCH_SIZE)
        bone_feats = self.bone_gcn(bone_graph, bone_feats.view(-1, 256)).view(BATCH_SIZE, self.bone_num_joints, 256)
        soft_feats = self.soft_gcn(soft_graph, soft_feats.view(-1, 256)).view(BATCH_SIZE, self.soft_num_joints, 256)
        bone_feats = bone_feats.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
        soft_feats = soft_feats.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]

        # 解码器：U-Net 风格上采样 + 层次图特征
        x = self.up  # [B, 256, H/16, W/16]
        x = self.up1(torch.cat([p4, c4], dim=1), c4)  # [B, 256, H/16, W/16]
        x_bone = x + bone_feats  # 融入骨组织特征
        x_soft = x + soft_feats  # 融入软组织特征
        x_bone = self.up2(torch.cat([x_bone, c3], dim=1), c3)  # [B, 128/512, H/8, W/8]
        x_soft = self.up2(torch.cat([x_soft, c3], dim=1), c3)  # [B, 128/512, H/8, W/8]
        x_bone = self.up3_bone(torch.cat([x_bone, c2], dim=1), c2)  # [B, 64/256, H/4, W/4]
        x_soft = self.up3_soft(torch.cat([x_soft, c2], dim=1), c2)  # [B, 64/256, H/4, W/4]
        heatmap_bone = self.bone_out(x_bone)  # [B, 15, hm_height, hm_width]
        heatmap_soft = self.soft_out(x_soft)  # [B, 4, hm_height, hm_width]

        # 组合热图
        heatmap = torch.zeros(BATCH_SIZE, self.num_joints, self.hm_height_dim, self.hm_width_dim, device=x.device)
        heatmap[:, self.bone_indices] = heatmap_bone
        heatmap[:, self.soft_indices] = heatmap_soft

        # 输出坐标
        out_coord = self.integral_hm(heatmap)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        pred_pts_bone = pred_pts[:, self.bone_indices, :]
        pred_pts_soft = pred_pts[:, self.soft_indices, :]

        output = EasyDict(
            pred_pts=pred_pts,
            pred_pts_bone=pred_pts_bone,
            pred_pts_soft=pred_pts_soft,
            heatmap=heatmap,
            heatmap_bone=heatmap_bone,
            heatmap_soft=heatmap_soft
        )
        return output

    def _initialize(self):
        pass