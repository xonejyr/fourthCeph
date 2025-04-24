""" Full assembly of the parts to form the complete network """

from .components.unet_parts import *
from easydict import EasyDict
from Unet.utils import Softmax_Integral

import dgl
from dgl.nn import GraphConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models

from Unet.builder import MODEL

"""
Hybrid Graph-Transformer with Interpretable Attention（混合图-Transformer带可解释注意力）。这个模型结合了图结构、Transformer和可解释性机制，旨在：

快速收敛：利用Transformer的全局建模能力。
低mre：通过图结构和分组优化捕捉组织特性。
可解释性：引入注意力可视化，揭示骨/软组织点的学习差异。
"""
class AttentionDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionDecoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4)

    def forward(self, x, context):
        x = self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h * w).permute(2, 0, 1)  # [H*W, B, C]
        context_flat = context.view(b, c, -1).permute(2, 0, 1)  # [context_len, B, C]
        attn_output, attn_weights = self.attn(x_flat, context_flat, context_flat)
        x = attn_output.permute(1, 2, 0).view(b, c, h, w)
        return x, attn_weights

@MODEL.register_module
class HybridGraphTransformer(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HybridGraphTransformer, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']  # 例如 3
        self._preset_cfg = cfg['PRESET']
        self.num_joints = self._preset_cfg['NUM_JOINTS']  # 总标志点数，例如 19
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self._norm_layer = norm_layer

        # 骨组织和软组织区分
        self.bone_num_joints = self._preset_cfg['NUM_JOINTS_BONE']  # 15
        self.soft_num_joints = self._preset_cfg['NUM_JOINTS_SOFT']  # 4
        self.bone_indices = torch.tensor(self._preset_cfg['BONE_INDICES'])  # [0, 1, ..., 18]
        self.soft_indices = torch.tensor(self._preset_cfg['SOFT_INDICES'])  # [12, 13, 14, 15]

        # ResNet 配置
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152], "NUM_LAYERS must be 18, 34, 50, 101, or 152"
        resnet_model = eval(f"models.resnet{cfg['NUM_LAYERS']}(pretrained={cfg.get('PRETRAINED_RIGHT', True)})")
        self.feature_channel = {
            18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048
        }[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512], 34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048], 101: [256, 512, 1024, 2048], 152: [256, 512, 1024, 2048]
        }[cfg['NUM_LAYERS']]

        # CNN Encoder: ResNet
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1  # C2
        self.layer2 = resnet_model.layer2  # C3
        self.layer3 = resnet_model.layer3  # C4
        self.layer4 = resnet_model.layer4  # C5

        # Graph Construction
        self.global_proj = nn.Linear(self.feature_channel, 512)  # 全局节点
        self.bone_proj = nn.Linear(self.feature_channel, 256)    # 骨组织子图
        self.soft_proj = nn.Linear(self.feature_channel, 256)    # 软组织子图

        # Graph-Transformer
        self.global_transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=512, nhead=8), num_layers=1)
        self.bone_gcn = GraphConv(256, 256)
        self.soft_gcn = GraphConv(256, 256)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        # Decoder with Interpretable Attention
        self.bone_decoder1 = AttentionDecoder(256, self.decoder_feature_channel[1])  # 128 or 512
        self.bone_decoder2 = AttentionDecoder(self.decoder_feature_channel[1], self.decoder_feature_channel[0])  # 64 or 256
        self.bone_out = nn.Conv2d(self.decoder_feature_channel[0], self.bone_num_joints, kernel_size=1)
        self.soft_decoder1 = AttentionDecoder(256, self.decoder_feature_channel[1])
        self.soft_decoder2 = AttentionDecoder(self.decoder_feature_channel[1], self.decoder_feature_channel[0])
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

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        # CNN Encoder
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)  # [B, 64/256, H/4, W/4]
        c3 = self.layer2(c2)  # [B, 128/512, H/8, W/8]
        c4 = self.layer3(c3)  # [B, 256/1024, H/16, W/16]
        c5 = self.layer4(c4)  # [B, 512/2048, H/32, W/32]

        # Graph Construction
        g = self.global_proj(F.avg_pool2d(c5, c5.size()[2:])).squeeze(-1).squeeze(-1)  # [B, 512]
        bone_feats = self.bone_proj(c5.mean(dim=(2, 3))).unsqueeze(1).repeat(1, self.bone_num_joints, 1)  # [B, 15, 256]
        soft_feats = self.soft_proj(c5.mean(dim=(2, 3))).unsqueeze(1).repeat(1, self.soft_num_joints, 1)  # [B, 4, 256]

        # Graph-Transformer
        g = self.global_transformer(g.unsqueeze(0)).squeeze(0)  # [B, 512]
        bone_graph, soft_graph = self._build_graph(BATCH_SIZE)
        bone_feats = self.bone_gcn(bone_graph, bone_feats.view(-1, 256)).view(BATCH_SIZE, self.bone_num_joints, 256)
        soft_feats = self.soft_gcn(soft_graph, soft_feats.view(-1, 256)).view(BATCH_SIZE, self.soft_num_joints, 256)
        bone_feats, _ = self.cross_attn(bone_feats.transpose(0, 1), g.unsqueeze(0), g.unsqueeze(0))
        soft_feats, _ = self.cross_attn(soft_feats.transpose(0, 1), g.unsqueeze(0), g.unsqueeze(0))
        bone_feats = bone_feats.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
        soft_feats = soft_feats.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]

        # Decoder with Interpretable Attention
        bone_x, bone_attn1 = self.bone_decoder1(bone_feats, c4)
        bone_x, bone_attn2 = self.bone_decoder2(bone_x, c3)
        heatmap_bone = self.bone_out(bone_x)  # [B, 15, hm_height, hm_width]
        soft_x, soft_attn1 = self.soft_decoder1(soft_feats, c4)
        soft_x, soft_attn2 = self.soft_decoder2(soft_x, c3)
        heatmap_soft = self.soft_out(soft_x)  # [B, 4, hm_height, hm_width]

        # Combine heatmaps with correct indices
        heatmap = torch.zeros(BATCH_SIZE, self.num_joints, self.hm_height_dim, self.hm_width_dim, device=x.device)
        heatmap[:, self.bone_indices] = heatmap_bone
        heatmap[:, self.soft_indices] = heatmap_soft

        # Output coordinates
        out_coord = self.integral_hm(heatmap)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        pred_pts_bone = pred_pts[:, self.bone_indices, :]
        pred_pts_soft = pred_pts[:, self.soft_indices, :]

        # 模仿 ResFPN 的 scores
        scores = 1 - pred_pts
        scores = torch.mean(scores, dim=2, keepdim=True)

        output = EasyDict(
            pred_pts=pred_pts,
            pred_pts_bone=pred_pts_bone,
            pred_pts_soft=pred_pts_soft,
            heatmap=heatmap,
            heatmap_bone=heatmap_bone,
            heatmap_soft=heatmap_soft,
            maxvals=scores.float()
        )
        return output

    def _initialize(self):
        pass





"""
输入图像 [B, 3, H, W]
    ↓
[CNN Encoder (ResNet18)]
    ├─── conv1 + bn1 + relu → C1 [B, 64, H/2, W/2]
    ├─── maxpool           → C1' [B, 64, H/4, W/4]
    ├─── layer1            → C2 [B, 64, H/4, W/4]
    ├─── layer2            → C3 [B, 128, H/8, W/8]
    ├─── layer3            → C4 [B, 256, H/16, W/16]
    └─── layer4            → C5 [B, 512, H/32, W/32]
                      ↓
[Graph Construction]
    ├─── 全局节点 G ← AvgPool(C5) [B, 512]
    ├─── 骨组织子图 B ← {B1, B2, ..., Bn} ← Proj(C5) + PosEncoding
    └─── 软组织子图 S ← {S1, S2, ..., Sm} ← Proj(C5) + PosEncoding
                      ↓
[Graph-Transformer]
    ├─── G → Self-Attn → G' [全局特征更新]
    ├─── B → GCN + Cross-Attn(G') → B' [骨组织子图优化]
    └─── S → GCN + Cross-Attn(G') → S' [软组织子图优化]
                      ↓
[Decoder with Interpretable Attention]
    ├─── B' → UpSample + Attn(C4, C3) → Heatmap_B [骨组织热图]
    └─── S' → UpSample + Attn(C4, C3) → Heatmap_S [软组织热图]
                      ↓
[输出层]
    ├─── Heatmap_B → Softmax_Integral → 骨组织坐标 [B, n, 2]
    └─── Heatmap_S → Softmax_Integral → 软组织坐标 [B, m, 2]
输出：EasyDict(pred_pts_bone, pred_pts_soft, heatmap_bone, heatmap_soft)
"""