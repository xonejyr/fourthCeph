from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet
from easydict import EasyDict
from Unet.utils import Softmax_Integral
from Unet.builder import MODEL
import torch.nn as nn
import torch
import torch.nn.functional as F
from .components.unet_parts import Up, OutConv  # 导入你的 UNet 模块

# Multi-Scale Feature Enhancement Module
class MSFE(nn.Module):
    """
    [B, in_channels/high_channels, x_high_H, x_high_W], [B, in_channels, x_low_H, x_low_W]
    => [B, out_channels, x_high_H, x_high_W]
    """
    def __init__(self, in_channels, out_channels, high_channels=None):
        super(MSFE, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, 1) # change channels
        self.attn = nn.Sequential( # 脖颈形状，先降通道，再升通道
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        # Use high_channels if provided, else assume in_channels matches high_channels
        fused_channels = (high_channels if high_channels is not None else in_channels) + out_channels
        self.conv_fuse = nn.Conv2d(fused_channels, out_channels, 3, padding=1) # change channels

    def forward(self, x_high, x_low):
        x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=False) # 将x_low的特征进行升维，和x_high一致
        x_low = self.conv_reduce(x_low) # 改变通道, in_channels => out_channels
        attn = self.attn(x_low) # 提取特征，两者一致尺寸， 通道为out_channels
        x_low = x_low * attn # 注意力
        x_fused = torch.cat([x_high, x_low], dim=1) # 对低维特征进行提取，而后与x_high进行拼接
        return self.conv_fuse(x_fused) # 将fused_channels通道数转变为out_channels, 关键是x_high的通道是多少，默认x_high的输入通道为in_channels.in_channels.

# Dynamic Upsampling Module
class DynamicUpsample(nn.Module):
    '''
    [B, in_channels, H, W]
    => [B, out_channels // (scale_factor^2), H * scale_factor, W * scale_factor]

    Dynamically upsamples the input feature map using a convolutional layer and a learned weight generation module.
    '''
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DynamicUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)# 只改变通道
        self.weight_gen = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1), # 只改变通道
            nn.ReLU(),
            nn.Conv2d(16, out_channels // (scale_factor * scale_factor), 1)  # Match x's channels after pixel_shuffle， 只是改变通道
        )

    def forward(self, x):
        # Generate dynamic weights
        weights = self.weight_gen(x)  # [B, in_channels, H, W] => [B, out_channels // scale_factor^2, H, W]
        x = self.conv(x)  # [B, in_channels, H, W] => [B, out_channels, H, W]
        x = F.pixel_shuffle(x, self.scale_factor)  # [B, out_channels // scale_factor^2, H * scale_factor, W * scale_factor], 通道降为1/scale_factor^2, 尺寸升为[H * scale_factor, W * scale_factor]
        # Upsample weights to match x's spatial dimensions, size= [H*scale_factor, W * scale_factor]
        weights = F.interpolate(weights, size=x.shape[2:], mode='bilinear', align_corners=False)
        # weights: [B, out_channels // (scale_factor^2), H * scale_factor, W * scale_factor]
        return x * weights  # Adaptive detail adjustment, element-wise multiplication

@MODEL.register_module
class ResFPN_UNet512(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(ResFPN_UNet512, self).__init__()

        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim  = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]

        self._norm_layer = norm_layer
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Load pretrained ResNet
        import torchvision.models as tm
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained={cfg['PRETRAINED_RIGHT']})")
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in model_state and v.size() == model_state[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # Feature channels
        self.feature_channel = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512],
            34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048],
            101: [256, 512, 1024, 2048],
            152: [256, 512, 1024, 2048],
        }[cfg['NUM_LAYERS']]

        # FPN Neck
        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
        )

        # UNet Decoder
        # 1.输入通道，2.输出通道，3.高级信息尺度
        self.msfe1 = MSFE(2048, 1024, high_channels=128)   # x_high=256 from feats_fpn[0]
        self.msfe2 = MSFE(1024, 512, high_channels=64)   # x_high=32 from upsample2 self-fusion
        self.msfe3 = MSFE(512,  256, high_channels=32)    # x_high=16 from upsample3 self-fusion

        # 1.输入通道，2.输出通道/4，3.尺度x2
        self.upsample1 = DynamicUpsample(1024, 512)     # feats_fpn[1]=512 -> 64
        self.upsample2 = DynamicUpsample(512, 256)     # 256 -> 32
        self.upsample3 = DynamicUpsample(256, 128)      # 128 -> 16

        #self.up = (Up(256, 128))
        self.up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(128, self.num_joints, 1)


        # Softmax Integral
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)
        
    def _initialize(self):
        pass

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        # 获取ResNet特征
        feats = self.preact.forward_feat(x)  # 返回 [C2, C3, C4, C5]
        """
        Feats[0] shape: torch.Size([8, 256, 128, 128])
        Feats[1] shape: torch.Size([8, 512, 64, 64])
        Feats[2] shape: torch.Size([8, 1024, 32, 32])
        Feats[3] shape: torch.Size([8, 2048, 16, 16])
        """

        ## 调试输出
        #print("Feats length:", len(feats))
        #for i, feat in enumerate(feats):
        #    print(f"Feats[{i}] shape:", feat.shape)
        #print("Expected in_channels:", self.decoder_feature_channel)

        # 确保输入数量和通道数匹配
        assert len(feats) == 4, "forward_feat should return exactly 4 feature maps"
        feats_fpn_input = feats  # 直接使用C2到C5

        # FPN Neck
        #feats_fpn = self.neck(feats_fpn_input)  # 输入4个特征图，输出4个特征图
        """
        Feats[0] shape: torch.Size([8, 256, 128, 128])
        Feats[1] shape: torch.Size([8, 256, 64, 64])
        Feats[2] shape: torch.Size([8, 256, 32, 32])
        Feats[3] shape: torch.Size([8, 256, 16, 16])
        """

        ## UNet Decoder（调整为从128x128开始）
        #d4 = feats_fpn_input[0]  # [8, 256, 128, 128]
        ## print(feats_fpn[1].shape)
        #d3 = self.upsample1(feats_fpn_input[1])  # [8, 512, 64, 64] -> [8, 64, 128, 128] 
        #d3 = self.msfe1(d4, d3) # ([8, 256, 128, 128], [8, 64, 128, 128]) => [8, 256, 128, 128]
        #d2 = self.upsample2(d3)  # [8, 256, 128, 128] -> [8, 32, 256, 256]
        ## 移除对feats[0]的依赖，直接上采样
        #d2 = self.msfe2(d2, d2)  # 自融合，避免依赖C1 []
        #d1 = self.upsample3(d2)  # 256x256 -> 512x512
        #d1 = self.msfe3(d1, d1)  # 自融合

        # UNet Decoder 先FPN
        d4 = feats[3]  # [8, 2048, 16, 16]
        # print(feats_fpn[1].shape)
        d3 = self.upsample1(feats[2])  # [8, 1024, 32, 32] => [8, 128, 64, 64]  # 128x128 -> 64x128
        d3 = self.msfe1(d3, d4) # => [8, 1024, 64, 64]
        d2 = self.upsample2(feats[1])  # [8, 512, 64, 64] -> [8, 64, 128, 128]
        # 移除对feats[0]的依赖，直接上采样
        d2 = self.msfe2(d2, d3)  # [8, 512, 128, 128]
        d1 = self.upsample3(feats[0])  # [8, 256, 128, 128] -> [8, 32, 256, 256]
        d1 = self.msfe3(d1, d2)  # [8, 256, 256, 256]
        d_out = self.up(d1) # [8, 256, 256, 256] => [8, 128, 512, 512]


        # 输出热图
        output_hm = self.final_conv(d_out)  # [8, 128, 512, 512] => [8, 19, 512, 512]
        out_coord = self.integral_hm(output_hm)

        # 预测结果
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        #scores = 1 - pred_pts
        #scores = torch.mean(scores, dim=2, keepdim=True)

        output = EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            #maxvals=scores.float(),
        )
        return output