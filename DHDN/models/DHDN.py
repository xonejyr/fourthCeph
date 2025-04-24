import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torchvision.models import resnet18,  ResNet18_Weights
from torch_geometric.data import Data, Batch

from easydict import EasyDict

from DHDN.builder import MODEL

@MODEL.register_module
class DHDN(nn.Module):
    def __init__(self, **cfg):
        # num_joints=19, num_features=2, num_patterns=5, hidden_dim=64, image_size_h=256, image_size_w=256):
        super(DHDN, self).__init__()

        self._preset_cfg   = cfg['PRESET']
        self.num_joints     = self._preset_cfg['NUM_JOINTS']
        self.image_size_h = self._preset_cfg['HEATMAP_SIZE'][0]
        self.image_size_w = self._preset_cfg['HEATMAP_SIZE'][1]
        self.num_patterns = self._preset_cfg['HEATMAP_SIZE'][1]

        self.num_nodes_per_hyperedge = cfg['NUM_NODES_PER_HYPEREDGE']
        self.num_patterns = cfg['NUM_PATTERNS'] # the number of latent patterns
        self.num_features = cfg['NUM_FEATURES'] # 输入的坐标是二维的x,y,故此为2
        self.hidden_dim = cfg['HIDDEN_DIM'] # hidden_dim 是模型中隐藏层特征的维度，表示每个节点（标志点）或超边的特征向量长度。它是超图特征H 和图像特征的维度，也是网络内部表示的宽度。

        # CNN for image features (适配 3 通道输入)
        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Linear(512, self.hidden_dim)  # 输出 hidden_dim

        # DHE: Dynamic Hypergraph Evolution
        self.hyperedge_init = nn.Linear(self.num_features, self.hidden_dim)  # 输入坐标 [x, y] 映射到高维空间
        self.hyper_gnn = pyg_nn.HypergraphConv(self.hidden_dim, self.hidden_dim) # 超图卷积，只是更新，而不改变维度，完全地线性映射, 需要手动添加激活层
        self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=4) # 多头注意力机制

        # DPE: Disentangled Pattern Extractor (VAE)
        # 相当于构建了一个潜在的模式空间
        self.encoder = nn.Sequential(
            nn.Linear(self.num_joints * self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_patterns * 2)  # mu + log_var, 自制的VAE，输出的是mu和log_var，
        )
        self.decoder = nn.Linear(self.num_patterns, self.num_joints * self.hidden_dim) # 输入是采样的Z

        # GRM: Generative Refinement Module
        self.refiner = nn.Sequential(
            nn.Linear(self.num_joints * self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_joints * 2)  # 从高纬空间将特征映射到 [19 * 2]
        )

    def extract_image_features(self, images, points):
        # images: [batch, 3, H, W], points: [batch, 19, 2]
        batch_size = images.size(0)
        height, width = images.size(2), images.size(3)  # H, W
        patches = []
        for b in range(batch_size):
            for p in points[b]:  # p 是归一化坐标 [-0.5, 0.5]
                # 转换到图像坐标 [0, W] 和 [0, H]
                x = int((p[0] + 0.5) * self.image_size_w)
                y = int((p[1] + 0.5) * self.image_size_h)
                # 计算裁剪边界并限制在图像范围内
                patch_size_half = self.image_size_w//32
                x_min = max(0, x - patch_size_half)
                x_max = min(width, x + patch_size_half)
                y_min = max(0, y - patch_size_half)
                y_max = min(height, y + patch_size_half)
                # 确保最小尺寸
                if x_max <= x_min:
                    x_min = x_max - 1
                if y_max <= y_min:
                    y_min = y_max - 1
                patch = images[b:b+1, :, y_min:y_max, x_min:x_max]
                # 调试：打印 patch 尺寸
                # print(f"Patch shape: {patch.shape}, x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")
                patch = F.interpolate(patch, size=(224, 224), mode='bilinear')
                patches.append(patch)
        patches = torch.cat(patches, dim=0)  # [batch * 19, 3, 224, 224]
        features = self.cnn(patches)  # [batch * 19, hidden_dim]
        return features.view(batch_size, self.num_joints, -1)  # [batch_size, num_joints, hidden_dim]
    
    def extract_image_features_1(self, images, points):
        # images: [batch, 3, H, W], points: [batch, 19, 2]
        batch_size = images.size(0)
        patches = []
        for b in range(batch_size):
            for p in points[b]:  # p 是归一化坐标 [-0.5, 0.5]
                # 转换到图像坐标 [0, W] 和 [0, H]
                x = int((p[0] + 0.5) * self.image_size_w)
                y = int((p[1] + 0.5) * self.image_size_h)
                patch = images[b:b+1, :, max(0, y-16):y+16, max(0, x-16):x+16] # 在对应的图里提取16x16 patch

                patch = F.interpolate(patch, size=(224, 224), mode='bilinear')  # ResNet 需要 224x224
                patches.append(patch)
        patches = torch.cat(patches, dim=0)  # [batch * 19, 3, 224, 224]
        features = self.cnn(patches)  # [batch * 19, hidden_dim], 将 CNN 得到的 patch 提取特征为hidden_dim
        return features.view(batch_size, self.num_joints, -1) # [batch_size, num_joints, hidden_dim]

    def build_hyperedges(self, points, img_features):
        # points: [batch, 19, 2], img_features: [batch, 19, hidden_dim]
        coord_dist = torch.cdist(points, points)  # [batch, 19, 19], 计算两个张量之间的成对距离（pairwise distance）, 通过设置p来确定距离类型。默认为欧氏距离，包括一个节点与自身的距离
        feat_dist = torch.cdist(img_features, img_features)  # [batch, 19, 19] 
        combined_dist = 0.5 * coord_dist + 0.5 * feat_dist # [batch, 19, 19] 
        _, indices = combined_dist.topk(k=self.num_nodes_per_hyperedge, dim=-1, largest=False) # 返回指定维度上最小（或最大）的 k 个值和它们的索引, 最近邻获取
        edge_index = torch.cat([torch.arange(self.num_joints).repeat_interleave(self.num_nodes_per_hyperedge).unsqueeze(0),
                                indices.view(-1).unsqueeze(0)], dim=0)
        # 第一行：[1, batch * 19 * 3]（中心节点）。[0, 0, 0, 1, 1, 1, ..., 18, 18, 18]（57 个元素）。
        # 第二行：[1, batch * 19 * 3]（邻居节点）。[2, 5, 7, 0, 3, 4, ..., ...]（57 个元素）。
        # 在 edge_index 中，超边被拆分为多条二元边（例如 [0, 2], [0, 5], [0, 7]），但 HypergraphConv 会将其视为一个整体超边。
        return edge_index # [2, num_joints * self.num_nodes_per_hyperedge]
    
    def build_hyperedges_batch(self, points, img_features):
        batch_size = points.size(0)
        edge_indices = []
        for b in range(batch_size):
            # 每个样本独立计算距离
            coord_dist = torch.cdist(points[b:b+1], points[b:b+1]).squeeze(0)  # [19, 19]
            feat_dist = torch.cdist(img_features[b:b+1], img_features[b:b+1]).squeeze(0)  # [19, 19]
            combined_dist = 0.5 * coord_dist + 0.5 * feat_dist  # [19, 19]
            _, indices = combined_dist.topk(k=self.num_nodes_per_hyperedge, dim=-1, largest=False)  # [19, 3]
            edge_index_b = torch.cat([torch.arange(self.num_joints).repeat_interleave(self.num_nodes_per_hyperedge).unsqueeze(0),
                                      indices.view(-1).unsqueeze(0)], dim=0)  # [2, 19 * 3]
            edge_indices.append(edge_index_b)
        # 按批量偏移索引
        edge_index = torch.cat([e + b * self.num_joints for b, e in enumerate(edge_indices)], dim=1) #将索引值加上偏移，确保每个样本的点索引是独立的。
        return edge_index  # [2, batch * 19 * 3]
    
    def build_hyperedges_batchnew(self, points, img_features):
        """ deal with the batch problem for hypergraph """
        batch_size = points.size(0)
        device = points.device
        edge_indices = []
        for b in range(batch_size):
            coord_dist = torch.cdist(points[b:b+1], points[b:b+1]).squeeze(0)  # [19, 19]
            feat_dist = torch.cdist(img_features[b:b+1], img_features[b:b+1]).squeeze(0)  # [19, 19]
            combined_dist = 0.5 * coord_dist + 0.5 * feat_dist
            _, indices = combined_dist.topk(k=self.num_nodes_per_hyperedge, dim=-1, largest=False)  # [19, 3]
            edge_index_b = torch.cat([torch.arange(self.num_joints, device=device).repeat_interleave(self.num_nodes_per_hyperedge).unsqueeze(0),
                                      indices.view(-1).unsqueeze(0)], dim=0)  # [2, 19 * 3]
            edge_indices.append(edge_index_b)
        return edge_indices  # List of [2, 19 * 3]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std # 随机采样 z
    
    def _initialize(self):
        pass

    def forward(self, img, target_uv):
        # img: [batch, 3, H, W], target_uv: [batch, 19 * 2]
        batch_size = img.size(0)
        points = target_uv.view(batch_size, self.num_joints, 2)  # [batch, 19, 2]

        # Extract image features
        img_features = self.extract_image_features(img, points) # [batch_size, num_joints, hidden_dim], 提取并映射为 hidden_dim

        # DHE: Dynamic Hypergraph Evolution (Batch)
        ## h = self.hyperedge_init(points)  # [batch, 19, hidden_dim], 关于节点的超图
        ## edge_index = self.build_hyperedges_batch(points, img_features) # [2, batch * 19 * 3]
        ## h = self.hyper_gnn(h.view(-1, h.size(-1)), edge_index)  # 将h的维度重组为2维，第一个维度计算，第二个维度为h.size(-1)，即 hidden_dim，得到[batch * 19, hidden_dim]

        h = self.hyperedge_init(points)  # [batch, 19, hidden_dim]
        # 逐样本构建超图
        edge_indices = self.build_hyperedges_batchnew(points, img_features)  # List of [2, 19 * 3]
        data_list = [Data(x=h[b], edge_index=edge_indices[b]) for b in range(batch_size)]
        batch_data = Batch.from_data_list(data_list)  # 自动处理批量
        # 超图卷积
        h = self.hyper_gnn(batch_data.x, batch_data.edge_index)  # [batch * 19, hidden_dim]
        h = h.view(batch_size, self.num_joints, -1) # [batch, 19, hidden_dim]
        h, _ = self.attention(h, h, h) # self.attetnion


        # DPE: Disentangled Pattern Extractor
        h_flat = h.view(batch_size, -1)  # [batch, 19 * hidden_dim]
        z_params = self.encoder(h_flat) # [batch,  2 * self.num_patterns] (mus and log_vars)
        mu, log_var = z_params[:, :self.num_patterns], z_params[:, self.num_patterns:] # 读取，前几个是mu，后几个是log_vars
        z = self.reparameterize(mu, log_var)  # 根据mu和log_var重采样z [batch, num_patterns]

        # GRM: Generative Refinement
        h_recon = self.decoder(z)  # [batch, 19 * hidden_dim], reconstructed hidden representation from a pattern z
        h_recon = h_recon.view(batch_size, self.num_joints, -1) # [batch, 19, hidden_dim]
        points_refined = self.refiner(h_recon.view(batch_size, -1))  # map from hidden space to x, y [batch, 19 * 2]
        points_refined = points_refined.view(batch_size, self.num_joints, 2) # [batchsize, num_joints, 2], of course in [-0.5, 0.5]

        output = EasyDict(
            pred_pts=points_refined, # [batchsize, num_joints, 2]
            mu=mu, # [batch, num_patterns] 
            log_var=log_var, # mu, log_var 含义：VAE 的均值和对数方差，用于生成𝑍, [batch, num_patterns]（例如 [batch, num_patterns]）。
            z=z, # [batchsize, num_patterns]
            heatmap=None # [batchsize, num_joints, hm_height, hm_width]
        )

        return output
        # points_refined [batchsize, num_joints, 2]



# 如何使用模型：初始化模型
# model = DHDN(num_joints=19, num_features=2, num_patterns=5, hidden_dim=64,
#            image_size_h=256, image_size_w=256).cuda()