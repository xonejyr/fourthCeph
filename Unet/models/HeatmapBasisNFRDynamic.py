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


class AdaptiveBasisVariational(nn.Module):
    def __init__(self, num_joints, max_bases=10, dim=2, kl_weight=0.1):
        super(AdaptiveBasisVariational, self).__init__()
        self.num_joints = num_joints
        self.max_bases = max_bases
        self.dim = dim
        self.kl_weight = kl_weight

        # 预测变分权重
        self.weight_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_bases),
            nn.Softmax(dim=-1)  # 变分权重 w_ik
        )

        # 预测 GMM 参数
        self.basis_net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_bases * dim * 2)  # mu 和 log_sigma
        )

        # Dirichlet 先验用于正则化权重
        self.prior_alpha = nn.Parameter(torch.ones(max_bases) * 1.0)  # 可学习的先验参数

    def forward(self, pred_pts):
        B, N, D = pred_pts.shape

        # 预测变分权重
        w = self.weight_net(pred_pts)  # [B, N, K_max]

        # 预测 GMM 参数
        params = self.basis_net(pred_pts).view(B, N, self.max_bases, 2 * D)
        mu = params[..., :D]  # [B, N, K_max, D]
        log_sigma = params[..., D:]  # [B, N, K_max, D]
        sigma = torch.exp(log_sigma.clamp(min=-10, max=10))  # 防止数值不稳定

        # 生成高斯分布
        basis_dists = [
            distributions.Normal(mu[:, :, k], sigma[:, :, k])
            for k in range(self.max_bases)
        ]

        # 计算 KL 散度（Dirichlet 先验 vs 变分权重）
        prior_dist = distributions.Dirichlet(self.prior_alpha)
        var_dist = distributions.Dirichlet(w.view(-1, self.max_bases))
        kl_div = distributions.kl_divergence(var_dist, prior_dist).view(B, N).sum()

        # 调试：检查形状
        assert w.shape == (B, N, self.max_bases), f"Expected w shape {(B, N, self.max_bases)}, got {w.shape}"
        assert len(basis_dists) == self.max_bases, f"Expected {self.max_bases} basis_dists, got {len(basis_dists)}"

        return w, basis_dists, self.kl_weight * kl_div
    
@MODEL.register_module
class HeatmapBasisNFRDynamic(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HeatmapBasisNFRDynamic, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.fc_dim = cfg['NUM_FC_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        self.max_bases = cfg['NUM_BASES']  # 最大基分布数量
        self.kl_weight =cfg['KL_WEIGHT']

        # ImageNet 预训练模型加载（保持不变）
        import torchvision.models as tm
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        self.feature_channel = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512], 34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048], 101: [256, 512, 1024, 2048], 152: [256, 512, 1024, 2048]
        }[cfg['NUM_LAYERS']]

        self.fcs, out_channel = self._make_fc_layer()
        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4
        )
        self.head = FPNHead(
            feature_strides=(4, 8, 16, 32),
            in_channels=[self.decoder_feature_channel[0]] * 4,
            channels=128,
            num_classes=self.num_joints,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.hidden_list = cfg['HIDDEN_LIST']
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 替换 AdaptiveBasis 为变分版本
        self.basis_dist = AdaptiveBasisVariational(num_joints=self.num_joints, max_bases=self.max_bases, kl_weight=self.kl_weight)
        self.fc_sigma = Linear(out_channel, self.num_joints * 2, norm=False)
        self.integral_hm = Softmax_Integral(
            num_pts=self.num_joints,
            hm_width=self.hm_width_dim,
            hm_height=self.hm_height_dim
        )

        self.fc_layers = [self.fc_sigma]
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
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

    def _initialize(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feats = self.preact.forward_feat(x)
        feats_for_sigma = self.avg_pool(feats[-1]).reshape(BATCH_SIZE, -1)
        feats = self.neck(feats)

        output_hm = self.head(feats)
        out_coord = self.integral_hm(output_hm)

        out_sigma = self.fc_sigma(feats_for_sigma).reshape(BATCH_SIZE, self.num_joints, -1)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        # 使用变分 GMM 预测基分布
        basis_dists_weights, basis_dists, kl_div = self.basis_dist(pred_pts)

        # 保存结果
        self.basis_dists_weights = basis_dists_weights
        self.basis_dists = basis_dists

        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid()
        scores = 1 - sigma
        scores = torch.mean(scores, dim=2, keepdim=True)

        basis_loss = None
        if self.training and labels is not None:
            gt_uv = labels['target_uv'].reshape(pred_pts.shape)
            log_probs = torch.stack(
                [dist.log_prob(gt_uv).sum(dim=-1) for dist in basis_dists],
                dim=2
            )  # [B, N, K_max]
            weighted_log_probs = log_probs + torch.log(basis_dists_weights + 1e-9)
            log_p = torch.logsumexp(weighted_log_probs, dim=2)
            basis_loss = -log_p.unsqueeze(2) + kl_div  # 添加 KL 正则化

        output = EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            sigma=sigma,
            maxvals=scores.float(),
            basis_dists_weights=basis_dists_weights,
            basis_dists=basis_dists,
            nf_loss=basis_loss
        )
        return output

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())
        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm
        if self.bias:
            y = y + self.linear.bias
        return y