import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torchvision.models import resnet50, ResNet50_Weights
from torch_geometric.data import Data, Batch
from easydict import EasyDict

from DHDN.builder import MODEL

#################################################################################80
## DHDN_Bigger
# 1.引入pattern_dim
# 2.CNN: 升级为 ResNet50 + FPN 
# 3.更多层的hyperGraph, 替换attention为transformer
# 4.VAE增强
# 5.refiner增强

@MODEL.register_module
class DHDN_Bigger(nn.Module):
    def __init__(self, **cfg):
        super(DHDN_Bigger, self).__init__()

        self._preset_cfg = cfg['PRESET']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.image_size_h = self._preset_cfg['HEATMAP_SIZE'][0]
        self.image_size_w = self._preset_cfg['HEATMAP_SIZE'][1]

        self.num_nodes_per_hyperedge = cfg['NUM_NODES_PER_HYPEREDGE']
        self.num_patterns = cfg.get('NUM_PATTERNS', 10)  # 增加到 10
        self.pattern_dim = cfg.get('PATTERN_DIM', 2)    # 每个模式 2 维
        self.num_features = cfg['NUM_FEATURES']
        self.hidden_dim = cfg.get('HIDDEN_DIM', 128)    # 增加到 128

        # CNN: 升级为 ResNet50 + FPN
        self.cnn = resnet50()
        self.cnn.fc = nn.Identity()  # 移除全连接层
        self.fpn = nn.Sequential(
            nn.Conv2d(2048, self.hidden_dim, kernel_size=1),  # 从 ResNet50 的 2048 通道降维
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局池化
        )
        self.cnn_proj = nn.Linear(self.hidden_dim, self.hidden_dim)  # 投影到目标维度

        # DHE: 多层超图卷积 + Transformer
        self.hyperedge_init = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )
        self.hyper_gnn = nn.ModuleList([
            pyg_nn.HypergraphConv(self.hidden_dim, self.hidden_dim) for _ in range(3)  # 3 层超图卷积
        ])
        self.hyper_gnn_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(3)])
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        # DPE: 增强 VAE
        self.encoder = nn.Sequential(
            nn.Linear(self.num_joints * self.hidden_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.num_patterns * self.pattern_dim * 2)  # mu + log_var
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.num_patterns * self.pattern_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.num_joints * self.hidden_dim)
        )

        # GRM: 深度 Refiner
        self.refiner = nn.Sequential(
            nn.Linear(self.num_joints * self.hidden_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.num_joints * 2)
        )
    
    def _initialize(self):
        pass

    def extract_image_features(self, images, points):
        batch_size = images.size(0)
        patches = []
        for b in range(batch_size):
            for p in points[b]:
                x = int((p[0] + 0.5) * self.image_size_w)
                y = int((p[1] + 0.5) * self.image_size_h)
                patch_size_half = self.image_size_w // 32
                x_min, x_max = max(0, x - patch_size_half), min(self.image_size_w, x + patch_size_half)
                y_min, y_max = max(0, y - patch_size_half), min(self.image_size_h, y + patch_size_half)
                if x_max <= x_min: x_min = x_max - 1
                if y_max <= y_min: y_min = y_max - 1
                patch = images[b:b+1, :, y_min:y_max, x_min:x_max]
                patch = F.interpolate(patch, size=(224, 224), mode='bilinear')
                patches.append(patch)
        patches = torch.cat(patches, dim=0)  # [batch * 19, 3, 224, 224]
        cnn_features = self.cnn(patches)     # [batch * 19, 2048]
        features = self.fpn(cnn_features.view(-1, 2048, 1, 1))  # [batch * 19, hidden_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch * 19, hidden_dim]
        features = self.cnn_proj(features)   # [batch * 19, hidden_dim]
        return features.view(batch_size, self.num_joints, -1)  # [batch, 19, hidden_dim]

    def build_hyperedges_batchnew(self, points, img_features):
        batch_size = points.size(0)
        device = points.device
        edge_indices = []
        for b in range(batch_size):
            coord_dist = torch.cdist(points[b:b+1], points[b:b+1]).squeeze(0)
            feat_dist = torch.cdist(img_features[b:b+1], img_features[b:b+1]).squeeze(0)
            combined_dist = 0.5 * coord_dist + 0.5 * feat_dist
            _, indices = combined_dist.topk(k=self.num_nodes_per_hyperedge, dim=-1, largest=False)
            edge_index_b = torch.cat([
                torch.arange(self.num_joints, device=device).repeat_interleave(self.num_nodes_per_hyperedge).unsqueeze(0),
                indices.view(-1).unsqueeze(0)
            ], dim=0)
            edge_indices.append(edge_index_b)
        return edge_indices  # List of [2, 19 * num_nodes_per_hyperedge]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, target_uv):
        batch_size = img.size(0)
        points = target_uv.view(batch_size, self.num_joints, 2)

        # Extract image features
        img_features = self.extract_image_features(img, points)  # [batch, 19, hidden_dim]

        # DHE: Dynamic Hypergraph Evolution
        h = self.hyperedge_init(points)  # [batch, 19, hidden_dim]
        edge_indices = self.build_hyperedges_batchnew(points, img_features)
        data_list = [Data(x=h[b] + img_features[b], edge_index=edge_indices[b]) for b in range(batch_size)]  # 融合坐标和图像特征
        batch_data = Batch.from_data_list(data_list)
        h = batch_data.x
        for gnn, norm in zip(self.hyper_gnn, self.hyper_gnn_norms):
            h = gnn(h, batch_data.edge_index)
            h = norm(F.relu(h))  # 添加非线性
        h = h.view(batch_size, self.num_joints, -1)  # [batch, 19, hidden_dim]
        h = self.transformer(h)  # [batch, 19, hidden_dim]

        # DPE: Disentangled Pattern Extractor
        h_flat = h.view(batch_size, -1)  # [batch, 19 * hidden_dim]
        z_params = self.encoder(h_flat)  # [batch, 2 * num_patterns * pattern_dim]
        z_params = z_params.view(batch_size, self.num_patterns, self.pattern_dim * 2)
        mu = z_params[:, :, :self.pattern_dim]  # [batch, num_patterns, pattern_dim]
        log_var = z_params[:, :, self.pattern_dim:]  # [batch, num_patterns, pattern_dim]
        z = self.reparameterize(mu, log_var)  # [batch, num_patterns, pattern_dim]
        z_flat = z.view(batch_size, -1)  # [batch, num_patterns * pattern_dim]

        # GRM: Generative Refinement
        h_recon = self.decoder(z_flat)  # [batch, 19 * hidden_dim]
        h_recon = h_recon.view(batch_size, self.num_joints, -1)  # [batch, 19, hidden_dim]
        points_refined = self.refiner(h_recon.view(batch_size, -1))  # [batch, 19 * 2]
        points_refined = points_refined.view(batch_size, self.num_joints, 2)  # [batch, 19, 2]

        output = EasyDict(
            pred_pts=points_refined,
            mu=mu,
            log_var=log_var,
            z=z,
            heatmap=None
        )
        return output

# 初始化示例
# cfg = {
#     'PRESET': {'NUM_JOINTS': 19, 'HEATMAP_SIZE': [256, 256]},
#     'NUM_NODES_PER_HYPEREDGE': 3,
#     'NUM_PATTERNS': 10,
#     'PATTERN_DIM': 2,
#     'NUM_FEATURES': 2,
#     'HIDDEN_DIM': 128
# }
# model = DHDN(**cfg).cuda()