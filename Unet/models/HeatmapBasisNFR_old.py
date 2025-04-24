"""
https://grok.com/share/bGVnYWN5_18c1ef51-7dde-4b14-99c2-bf4037a9aa9b

"""

import torch
import torch.nn as nn
import torch.distributions as distributions
from easydict import EasyDict

# 假设已有模块
from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet
from Unet.utils import Softmax_Integral
from Unet.builder import MODEL


class AdaptiveBasis(nn.Module):
    def __init__(self, num_joints, num_bases=3, dim=2):
        super(AdaptiveBasis, self).__init__()
        self.num_joints = num_joints
        self.num_bases = num_bases
        self.dim = dim

        self.weight_net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, num_bases), nn.Softmax(dim=-1)
        )

        self.basis_net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, num_bases * dim * 2)
        )

    def forward(self, pred_pts):
        B, N, D = pred_pts.shape
        w = self.weight_net(pred_pts)  # [B, N, K]
        params = self.basis_net(pred_pts).view(B, N, self.num_bases, 2 * D)
        mu = params[..., :D]
        log_sigma = params[..., D:]
        sigma = torch.exp(log_sigma)
        basis_dists = [distributions.Normal(mu[:, :, k], sigma[:, :, k]) for k in range(self.num_bases)]
        
        # 调试：检查输出形状
        assert w.shape == (B, N, self.num_bases), f"Expected w shape {(B, N, self.num_bases)}, got {w.shape}"
        assert len(basis_dists) == self.num_bases, f"Expected {self.num_bases} basis_dists, got {len(basis_dists)}"
        
        return w, basis_dists
@MODEL.register_module
class HeatmapBasisNFR(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HeatmapBasisNFR, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.fc_dim = cfg['NUM_FC_FILTERS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        self.feature_channel = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512], 34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048], 101: [256, 512, 1024, 2048], 152: [256, 512, 1024, 2048]
        }[cfg['NUM_LAYERS']]

        self.fcs, out_channel = self._make_fc_layer()
        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
        )
        self.head = FPNHead(
            feature_strides=(4, 8, 16, 32),
            in_channels=[self.decoder_feature_channel[0]] * 4,
            channels=128,
            num_classes=self.num_joints,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.integral_hm = Softmax_Integral(
            num_pts=self.num_joints,
            hm_width=self.hm_width_dim,
            hm_height=self.hm_height_dim
        )

        self.basis_dist = AdaptiveBasis(num_joints=self.num_joints, num_bases=3)
        self.fc_sigma = nn.Linear(self.feature_channel, self.num_joints * 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        import torchvision.models as tm
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in model_state and v.size() == model_state[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())
        return nn.Sequential(*fc_layers), input_channel

    def forward(self, x, labels=None):
        B = x.shape[0]
        feats = self.preact.forward_feat(x)
        feats_for_sigma = self.avg_pool(feats[-1]).reshape(B, -1)
        feats = self.neck(feats)
        output_hm = self.head(feats)

        out_coord = self.integral_hm(output_hm)
        pred_pts = out_coord.reshape(B, self.num_joints, 2)

        w, basis_dists = self.basis_dist(pred_pts)
        out_sigma = self.fc_sigma(feats_for_sigma).reshape(B, self.num_joints, 2).sigmoid()
        scores = 1 - out_sigma.mean(dim=2, keepdim=True)

        basis_loss = None
        if self.training and labels is not None:
            gt_uv = labels['target_uv'].reshape(pred_pts.shape)
            #log_probs = torch.stack([dist.log_prob(gt_uv) for dist in basis_dists], dim=2)
            log_probs = torch.stack([dist.log_prob(gt_uv).sum(dim=-1) for dist in basis_dists], dim=2)
            
            # 调试：检查形状
            #print(f"log_probs shape: {log_probs.shape}")
            #print(f"w shape: {w.shape}")
            #assert log_probs.shape[2] == w.shape[2], f"Dimension mismatch: log_probs {log_probs.shape}, w {w.shape}"
            #
            weighted_log_probs = log_probs + torch.log(w + 1e-9)
            log_p = torch.logsumexp(weighted_log_probs, dim=2)
            basis_loss = -log_p.unsqueeze(2)

        output = EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            sigma=out_sigma,
            maxvals=scores.float(),
            nf_loss=basis_loss
        )
        return output
    
    def _initialize(self):
        pass