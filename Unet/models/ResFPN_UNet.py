""" Full assembly of the parts to form the complete network

ResNet + FPN + UNet-like Skip Connections + Softmax_Integral
"""

from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet # for backbone network, used for feature extraction
from easydict import EasyDict
from Unet.utils import Softmax_Integral

from Unet.builder import MODEL

import torch.nn as nn
import torch

@MODEL.register_module
class ResFPN_UNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(ResFPN_UNet, self).__init__()

        self.n_channels = cfg['IN_CHANNELS']

        self._preset_cfg   = cfg['PRESET']
        self.n_classes     = self._preset_cfg['NUM_JOINTS']
        self.num_joints    = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim  = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]

        self._norm_layer = norm_layer
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403
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

        # Define the channels to be used for skip connections
        self.unet_channels = self.decoder_feature_channel[:-1] # Exclude the last layer as FPN starts from there
        fpn_out_channels = self.decoder_feature_channel[0]

        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=fpn_out_channels,
            num_outs=4,
            # Add the channels for skip connections here if needed in FPN_neck_hm
        )

        # Modify the FPNHead to potentially accept more input if the FPN neck's output changes
        self.head = FPNHead(feature_strides=(4, 8, 16, 32),
                            in_channels=[fpn_out_channels] * 4, # Adjust if needed
                            channels=512,
                            num_classes=self.num_joints,
                            norm_cfg=dict(type='BN', requires_grad=True))

        self.hidden_list = cfg['HIDDEN_LIST']
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        # Define layers for fusing skip connections
        self.conv_fuse_1 = nn.Conv2d(self.unet_channels[0] + fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.conv_fuse_2 = nn.Conv2d(self.unet_channels[1] + fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.conv_fuse_3 = nn.Conv2d(self.unet_channels[2] + fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

    def _initialize(self):
        pass

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]
    
        resnet_features = self.preact.forward_feat(x)
        fpn_input_features = resnet_features[-4:] # 前四个
        fpn_outputs = self.neck(fpn_input_features) # 用neck融合前4个
    
        fused_features = list(fpn_outputs)
    
        if len(resnet_features) >= 4 and len(fpn_outputs) >= 4:
            # 假设 resnet_features = [layer1_out, layer2_out, layer3_out, layer4_out]
            # 假设 fpn_outputs = [P2, P3, P4, P5]
    
            # 融合 P2 (fpn_outputs[0]) 和 layer1_out (resnet_features[0])
            fuse_p2 = torch.cat([fpn_outputs[0], resnet_features[0]], dim=1)
            fuse_p2 = self.conv_fuse_1(fuse_p2)
            fused_features[0] = fuse_p2
    
            # 融合 P3 (fpn_outputs[1]) 和 layer2_out (resnet_features[1])
            fuse_p3 = torch.cat([fpn_outputs[1], resnet_features[1]], dim=1)
            fuse_p3 = self.conv_fuse_2(fuse_p3)
            fused_features[1] = fuse_p3
    
            # 融合 P4 (fpn_outputs[2]) 和 layer3_out (resnet_features[2])
            fuse_p4 = torch.cat([fpn_outputs[2], resnet_features[2]], dim=1)
            fuse_p4 = self.conv_fuse_3(fuse_p4)
            fused_features[2] = fuse_p4
    
            # P5 (fpn_outputs[3]) 可以直接使用，或者你也可以考虑与 layer4_out (resnet_features[3]) 进行某种形式的交互，但这取决于你的设计
    
        output_hm = self.head(fused_features)
        out_coord = self.integral_hm(output_hm)
    
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        scores = 1 - pred_pts
        scores = torch.mean(scores, dim=2, keepdim=True)
    
        output = EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            maxvals=scores.float(),
        )
    
        return output