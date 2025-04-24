""" Full assembly of the parts to form the complete network
UNet + FPN + Softmax_Integral
"""

from .components.unet_parts import *  # 假设包含 DoubleConv, Down, Up, OutConv 等组件
from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead  # 复用 FPN 组件
from easydict import EasyDict
from Unet.utils import Softmax_Integral
from Unet.builder import MODEL
import torch
import torch.nn as nn

@MODEL.register_module
class UNetFPN(nn.Module):
    def __init__(self, bilinear=False, norm_layer=nn.BatchNorm2d, **cfg):
        super(UNetFPN, self).__init__()
        
        # 输入通道数和预设配置
        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        
        self.bilinear = bilinear
        self._norm_layer = norm_layer

        # UNet 的编码器部分 (Downsampling)
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 定义特征通道数，与 ResNet+FPN 对齐
        self.decoder_feature_channel = [64, 128, 256, 512]  # 从浅层到深层

        # FPN Neck：融合多尺度特征
        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,  # 输入通道数对应 UNet 的各层输出
            out_channels=self.decoder_feature_channel[0],  # 统一输出通道数为 64
            num_outs=4,  # 输出 4 个尺度
        )

        # FPN Head：生成最终的 heatmap
        self.head = FPNHead(
            feature_strides=(4, 8, 16, 32),  # 特征图 stride，与 UNet 下采样匹配
            in_channels=[self.decoder_feature_channel[0]] * 4,  # 输入通道统一为 64
            channels=512,  # 中间通道数
            num_classes=self.num_joints,  # 输出类别数
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        # Softmax Integral 用于坐标回归
        self.integral_hm = Softmax_Integral(
            num_pts=self.num_joints,
            hm_width=self.hm_width_dim,
            hm_height=self.hm_height_dim
        )

    def _initialize(self):
        pass

    def forward(self, x, target_uv=None):
        BATCH_SIZE = x.shape[0]

        # UNet 编码器：提取多尺度特征
        x1 = self.inc(x)    # [B, 64, H, W]
        x2 = self.down1(x1) # [B, 128, H/2, W/2]
        x3 = self.down2(x2) # [B, 256, H/4, W/4]
        x4 = self.down3(x3) # [B, 512, H/8, W/8]
        x5 = self.down4(x4) # [B, 1024/factor, H/16, W/16]

        # 收集多尺度特征，输入到 FPN
        #feats = [x1, x2, x3, x4]  # 不包括 x5，因为 FPN 通常用 4 个尺度
        feats = [x2, x3, x4, x5] 

        # FPN Neck 融合特征
        fpn_feats = self.neck(feats)  # 输出 4 个尺度的特征图，通道数统一为 64

        # FPN Head 生成 heatmap
        output_hm = self.head(fpn_feats)  # [B, num_joints, hm_height, hm_width]

        # 通过 Softmax Integral 计算坐标
        out_coord = self.integral_hm(output_hm)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        # 计算置信度（参考 ResFPN）
        scores = 1 - pred_pts
        scores = torch.mean(scores, dim=2, keepdim=True)

        # 输出结果
        output = EasyDict(
            pred_pts=pred_pts,  # [batchsize, num_joints, 2]
            heatmap=output_hm,  # [batchsize, num_joints, hm_height, hm_width]
            maxvals=scores.float(),  # [batchsize, num_joints, 1]
        )
        return output

    def use_checkpointing(self):
        # 内存优化，添加 checkpoint
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.neck = torch.utils.checkpoint.checkpoint(self.neck)
        self.head = torch.utils.checkpoint.checkpoint(self.head)