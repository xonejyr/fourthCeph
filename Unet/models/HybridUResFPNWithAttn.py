""" Full assembly of the parts to form the complete network """

from .components.unet_parts import *
from easydict import EasyDict
from Unet.utils import Softmax_Integral

from Unet.builder import MODEL

from torchvision import models

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, in_channels),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        b, c, h, w = x.size()
        avg_pool = self.channel_avg(x).view(b, c)
        channel_weight = self.channel_fc(avg_pool).view(b, c, 1, 1)
        x = x * channel_weight

        # 空间注意力
        spatial_weight = self.spatial_sigmoid(self.spatial_conv(x))
        x = x * spatial_weight
        return x

@MODEL.register_module
class HybridUResFPNWithAttn(nn.Module):
    def __init__(self, bilinear=False, norm_layer=nn.BatchNorm2d, **cfg):
        super(HybridUResFPNWithAttn, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']  # 输入通道数，例如 3
        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']  # 标志点数量，例如 19
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.bilinear = bilinear
        self._norm_layer = norm_layer

        # ResNet 配置
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152], "NUM_LAYERS must be 18, 34, 50, 101, or 152"
        resnet = eval(f"models.resnet{cfg['NUM_LAYERS']}(pretrained={cfg.get('PRETRAINED_RIGHT', True)})")
        self.feature_channel = {
            18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048
        }[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512], 34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048], 101: [256, 512, 1024, 2048], 152: [256, 512, 1024, 2048]
        }[cfg['NUM_LAYERS']]

        # 编码器：ResNet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # C2: 64 or 256
        self.layer2 = resnet.layer2  # C3: 128 or 512
        self.layer3 = resnet.layer3  # C4: 256 or 1024
        self.layer4 = resnet.layer4  # C5: 512 or 2048

        # FPN 模块
        self.fpn_conv4 = nn.Conv2d(self.decoder_feature_channel[3], 256, kernel_size=1)  # P4
        self.fpn_conv3 = nn.Conv2d(self.decoder_feature_channel[2], 256, kernel_size=1)  # P3
        self.fpn_conv2 = nn.Conv2d(self.decoder_feature_channel[1], 256, kernel_size=1)  # P2
        self.fpn_smooth = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # 解码器：U-Net 风格 + 注意力模块
        factor = 2 if bilinear else 1
        self.up1 = Up(self.decoder_feature_channel[3] + 256, 256 // factor, bilinear)  # P4 + C4
        self.up2 = Up(self.decoder_feature_channel[2] + 256 // factor, self.decoder_feature_channel[1] // factor, bilinear)  # P3 + C3
        self.up3 = Up(self.decoder_feature_channel[1] + self.decoder_feature_channel[1] // factor, self.decoder_feature_channel[0], bilinear)  # P2 + C2
        self.attn1 = AttentionModule(256 // factor)
        self.attn2 = AttentionModule(self.decoder_feature_channel[1] // factor)
        self.attn3 = AttentionModule(self.decoder_feature_channel[0])
        self.outc = OutConv(self.decoder_feature_channel[0], self.n_classes)

        # Softmax_Integral
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)

    def forward(self, x, target_uv=None):
        BATCH_SIZE = x.shape[0]

        # 编码器：ResNet
        c1 = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2] or [B, 64, H/2, W/2]
        c1 = self.maxpool(c1)                    # [B, 64, H/4, W/4] or [B, 64, H/4, W/4]
        c2 = self.layer1(c1)                     # [B, 64, H/4, W/4] or [B, 256, H/4, W/4]
        c3 = self.layer2(c2)                     # [B, 128, H/8, W/8] or [B, 512, H/8, W/8]
        c4 = self.layer3(c3)                     # [B, 256, H/16, W/16] or [B, 1024, H/16, W/16]
        c5 = self.layer4(c4)                     # [B, 512, H/32, W/32] or [B, 2048, H/32, W/32]

        # FPN：多尺度特征融合
        p4 = self.fpn_conv4(c5)                  # [B, 256, H/32, W/32]
        p3 = self.fpn_conv3(c4) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, H/16, W/16]
        p3 = self.fpn_smooth(p3)                 # [B, 256, H/16, W/16]
        p2 = self.fpn_conv2(c3) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, H/8, W/8]
        p2 = self.fpn_smooth(p2)                 # [B, 256, H/8, W/8]

        # 解码器：U-Net 风格上采样 + 注意力模块
        x = self.up1(torch.cat([p4, c4], dim=1), c4)  # [B, 256, H/16, W/16]
        x = self.attn1(x)
        x = self.up2(torch.cat([x, c3], dim=1), c3)   # [B, 128 or 512, H/8, W/8]
        x = self.attn2(x)
        x = self.up3(torch.cat([x, c2], dim=1), c2)   # [B, 64 or 256, H/4, W/4]
        x = self.attn3(x)
        logits = self.outc(x)                         # [B, n_classes, hm_height, hm_width]

        # Softmax_Integral 输出坐标
        out_coord = self.integral_hm(logits)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        output = EasyDict(
            pred_pts=pred_pts,  # [batch_size, num_joints, 2]
            heatmap=logits      # [batch_size, num_joints, hm_height, hm_width]
        )
        return output

    def _initialize(self):
        pass

"""
输入图像 [B, 3, H, W]
    ↓
[ResNet18 Encoder]
    ├─── conv1 + bn1 + relu → C1 [B, 64, H/2, W/2]
    ├─── maxpool           → C1' [B, 64, H/4, W/4]
    ├─── layer1            → C2 [B, 64, H/4, W/4]
    ├─── layer2            → C3 [B, 128, H/8, W/8]
    ├─── layer3            → C4 [B, 256, H/16, W/16]
    └─── layer4            → C5 [B, 512, H/32, W/32]
                      ↓    ↓    ↓    ↓
[FPN 特征金字塔]
    ├─── C5 → fpn_conv4 → P4 [B, 256, H/32, W/32]
    ├─── C4 → fpn_conv3 + 上采样P4 → P3 [B, 256, H/16, W/16]
    └─── C3 → fpn_conv2 + 上采样P3 → P2 [B, 256, H/8, W/8]
                      ↓    ↓    ↓
[U-Net 解码器 + 注意力]
    ├─── P4 + C4 → Up1 → D3 [B, 256, H/16, W/16]
    │        ↓
    │   [Attn1: 通道+空间] → D3' [B, 256, H/16, W/16]
    ├─── D3' + C3 → Up2 → D2 [B, 128, H/8, W/8]
    │        ↓
    │   [Attn2: 通道+空间] → D2' [B, 128, H/8, W/8]
    └─── D2' + C2 → Up3 → D1 [B, 64, H/4, W/4]
             ↓
        [Attn3: 通道+空间] → D1' [B, 64, H/4, W/4]
                      ↓
[输出层]
    ├─── D1' → OutConv → 热图 [B, num_joints, hm_height, hm_width]
    └─── Softmax_Integral → 坐标 [B, num_joints, 2]
输出：EasyDict(pred_pts, heatmap)
"""