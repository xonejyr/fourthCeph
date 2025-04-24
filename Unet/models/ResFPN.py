""" Full assembly of the parts to form the complete network

ResNet + FPN + Softmax_Integral
"""

from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet # for backbone network, used for feature extraction
from easydict import EasyDict
from Unet.utils import Softmax_Integral

from Unet.builder import MODEL

import torch.nn as nn
import torch

@MODEL.register_module
class ResFPN(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(ResFPN, self).__init__()

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
        #x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")
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

        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
        )
        self.head = FPNHead(feature_strides=(4, 8, 16, 32), # 上采样
                            in_channels=[self.decoder_feature_channel[0]] * 4, # 输入总的特征通道数
                            channels=512, # 中间特征通道数
                            num_classes=self.num_joints, #最终输出特征通道数
                            norm_cfg=dict(type='BN', requires_grad=True)) # 归一化
        # 最终的输出[8, num_joints, 128, 128]

        self.hidden_list = cfg['HIDDEN_LIST']
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)


        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)


    def _initialize(self):
        pass

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feats = self.preact.forward_feat(x)

        feats = self.neck(feats)

        output_hm = self.head(feats)
        out_coord = self.integral_hm(output_hm)


        # (B, N, 2) 
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        # don't know what to do with this
        scores = 1 - pred_pts
        scores = torch.mean(scores, dim=2, keepdim=True)

        output = EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            maxvals=scores.float(),
        )

        return output
