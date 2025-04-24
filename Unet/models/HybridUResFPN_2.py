""" Full assembly of the parts to form the complete network """
from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead, ConvModule
from .components.NFDP_parts.Resnet import ResNet 


from .components.unet_parts import *
from easydict import EasyDict
from Unet.utils import Softmax_Integral

from torchvision import models

from Unet.builder import MODEL

@MODEL.register_module
class HybridUResFPN(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HybridUResFPN, self).__init__()

        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.img_width_dim = self._preset_cfg['IMAGE_SIZE'][1]
        self.img_height_dim = self._preset_cfg['IMAGE_SIZE'][0]

        self._norm_layer = norm_layer
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained={cfg['PRETRAINED_RIGHT']})")

        self.feature_channel = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
            152: 2048
        }[cfg['NUM_LAYERS']]
        
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512],
            34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048],
            101: [256, 512, 1024, 2048],
            152: [256, 512, 1024, 2048],
        }[cfg['NUM_LAYERS']]

        # UNet 解码器层（使用你的 ConvModule）
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(self.decoder_feature_channel[3], self.decoder_feature_channel[2], 
                             kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.decoder_feature_channel[2], self.decoder_feature_channel[1], 
                             kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.decoder_feature_channel[1], self.decoder_feature_channel[0], 
                             kernel_size=4, stride=2, padding=1)
        ])
        
        self.fusion_convs = nn.ModuleList([
            ConvModule(
                self.decoder_feature_channel[2] * 2, self.decoder_feature_channel[2],
                kernel_size=3, padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')),
            ConvModule(
                self.decoder_feature_channel[1] * 2, self.decoder_feature_channel[1],
                kernel_size=3, padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')),
            ConvModule(
                self.decoder_feature_channel[0] * 2, self.decoder_feature_channel[0],
                kernel_size=3, padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU'))
        ])

        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        
        self.head = FPNHead(
            feature_strides=(4, 8, 16, 32),
            in_channels=[self.decoder_feature_channel[0]] * 4,
            channels=512,
            num_classes=self.num_joints,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        self.hidden_list = cfg['HIDDEN_LIST']
        self.integral_hm = Softmax_Integral(
            num_pts=self.num_joints,
            hm_width=self.hm_width_dim,
            hm_height=self.hm_height_dim
        )

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

    def _initialize(self):
        pass

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        # 获取 ResNet 的特征（编码器部分）
        feats = self.preact.forward_feat(x)  # 返回4个尺度的特征
        encoder_feats = list(feats)  # 转换为列表
        
        # UNet 解码器部分
        x = encoder_feats[-1]  # 取最深层的特征
        
        unet_feats = []
        for i in range(3):
            # 上采样
            x = self.upconvs[i](x)
            
            # 与对应的编码器特征进行跳跃连接
            enc_feat = encoder_feats[2-i]  # 从倒数第二层开始
            if x.shape[2:] != enc_feat.shape[2:]:
                enc_feat = F.interpolate(
                    enc_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            # 融合特征
            x = torch.cat([x, enc_feat], dim=1)
            x = self.fusion_convs[i](x)
            unet_feats.append(x)

        # 组合原始特征和 UNet 处理后的特征
        final_feats = [
            encoder_feats[0],  # 最低层特征保持不变
            unet_feats[0],     # UNet 处理后的特征
            unet_feats[1],
            unet_feats[2]
        ]

        # 通过 FPN neck 和 head
        feats = self.neck(final_feats)
        output_hm = self.head(feats)
        out_coord = self.integral_hm(output_hm)

        # 处理输出
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        scores = 1 - pred_pts
        scores = torch.mean(scores, dim=2, keepdim=True)

        output = EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            maxvals=scores.float(),
        )

        return output

