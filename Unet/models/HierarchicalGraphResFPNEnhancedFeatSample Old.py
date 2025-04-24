from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet
from easydict import EasyDict
from Unet.utils import Softmax_Integral
from Unet.builder import MODEL
import torch.nn as nn
import torch
import torch.nn.functional as F

class RefinementStep(nn.Module):
    def __init__(self, graph_channels, num_joints, hm_height_dim, hm_width_dim, 
                 bone_indices, soft_indices, gcn_layers=2, adj_type='cross', 
                 multi_scale_channels=None):
        super(RefinementStep, self).__init__()
        self.graph_channels = graph_channels
        self.num_joints = num_joints
        self.hm_height_dim = hm_height_dim
        self.hm_width_dim = hm_width_dim
        self.bone_indices = bone_indices
        self.soft_indices = soft_indices
        self.adj_type = adj_type

        # Graph attention layers
        self.bone_gat = MultiGraphAttentionLayer(graph_channels, graph_channels, num_layers=gcn_layers)
        self.soft_gat = MultiGraphAttentionLayer(graph_channels, graph_channels, num_layers=gcn_layers)
        self.cross_gat = MultiGraphAttentionLayer(graph_channels * 2, graph_channels, num_layers=gcn_layers)

        # Multi-scale feature projection
        self.multi_scale_channels = multi_scale_channels or [256, 256, 256, 256]
        self.scale_projections = nn.ModuleList([
            nn.Linear(ch, graph_channels) for ch in self.multi_scale_channels
        ])

        # F_beta extractor for cropped features
        self.f_beta = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.crop_proj = nn.Linear(512, graph_channels)

        # Adaptive fusion for cropped and multi-scale features
        self.fusion_weights = nn.Sequential(
            nn.Linear(graph_channels * 2, 128),  # Concat cropped and multi-scale
            nn.ReLU(),
            nn.Linear(128, 2),  # Weights for 2 feature types
            nn.Softmax(dim=-1)
        )

        # Adjacency construction using coordinates and multi-scale features
        self.adj_proj = nn.Linear(graph_channels + 2, graph_channels)  # Combine multi-scale feats and coords
        self.adj_attention = nn.Linear(graph_channels * 2, 1)  # Attention for adjacency

        # Final convolution to generate heatmap
        self.refine_head = nn.Conv2d(graph_channels, num_joints, 1)

    def _sample_multi_scale_features(self, multi_scale_feats, coords):
        B = coords.shape[0]
        sampled_feats = []
        for feat, proj in zip(multi_scale_feats, self.scale_projections):
            H, W = feat.shape[2], feat.shape[3]
            norm_coords = coords / torch.tensor([self.hm_width_dim, self.hm_height_dim], device=coords.device)
            norm_coords = 2 * norm_coords - 1
            grid = norm_coords.view(B, -1, 1, 2)
            sampled = F.grid_sample(feat, grid, mode='bilinear', align_corners=True)
            sampled = sampled.squeeze(-1).permute(0, 2, 1)
            sampled = proj(sampled)
            sampled_feats.append(sampled)
        return torch.mean(torch.stack(sampled_feats, dim=0), dim=0)

    def _process_cropped_features(self, cropped, step_idx):
        if step_idx >= 3:
            raise ValueError(f"step_idx {step_idx} exceeds maximum allowed refinement steps (2)")
        crop = cropped[step_idx]  # [B, N, H, W, C]
        crop_sizes = [64, 32, 16]
        B, N = crop.shape[0], crop.shape[1]
        crop = crop.view(B * N, crop_sizes[step_idx], crop_sizes[step_idx], 3).permute(0, 3, 1, 2)
        crop = F.interpolate(crop, size=(32, 32), mode='bilinear', align_corners=True)
        crop_feats = self.f_beta(crop)
        crop_feats = crop_feats.view(B, N, 512)
        return self.crop_proj(crop_feats)

    def _build_adjacency(self, coords, multi_scale_feats):
        B, N = coords.shape[0], coords.shape[1]
        multi_feats = self._sample_multi_scale_features(multi_scale_feats, coords)  # [B, N, graph_channels]
        combined = torch.cat([multi_feats, coords], dim=-1)  # [B, N, graph_channels + 2]
        feat = self.adj_proj(combined)  # [B, N, graph_channels]
        
        feat_i = feat.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, graph_channels]
        feat_j = feat.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, graph_channels]
        attn_input = torch.cat([feat_i, feat_j], dim=-1)  # [B, N, N, graph_channels * 2]
        adj = self.adj_attention(attn_input).squeeze(-1)  # [B, N, N]
        adj = torch.sigmoid(adj)  # Normalize to [0, 1]
        return adj

    def forward(self, coords, multi_scale_feats, cropped, step_idx):
        B = coords.shape[0]
        
        bone_coords = coords[:, self.bone_indices]
        soft_coords = coords[:, self.soft_indices]

        # Build adjacency matrices using coords and multi-scale features
        bone_adj = self._build_adjacency(bone_coords, multi_scale_feats)
        soft_adj = self._build_adjacency(soft_coords, multi_scale_feats)

        # Get multi-scale and cropped features
        bone_multi_scale = self._sample_multi_scale_features(multi_scale_feats, bone_coords)
        soft_multi_scale = self._sample_multi_scale_features(multi_scale_feats, soft_coords)
        bone_cropped = self._process_cropped_features(cropped, step_idx)[:, self.bone_indices]
        soft_cropped = self._process_cropped_features(cropped, step_idx)[:, self.soft_indices]

        # Adaptive fusion
        bone_fusion_input = torch.cat([bone_multi_scale, bone_cropped], dim=-1)
        soft_fusion_input = torch.cat([soft_multi_scale, soft_cropped], dim=-1)
        bone_weights = self.fusion_weights(bone_fusion_input)  # [B, N_bone, 2]
        soft_weights = self.fusion_weights(soft_fusion_input)  # [B, N_soft, 2]
        
        bone_feats = (bone_weights[..., 0:1] * bone_multi_scale + 
                      bone_weights[..., 1:2] * bone_cropped)
        soft_feats = (soft_weights[..., 0:1] * soft_multi_scale + 
                      soft_weights[..., 1:2] * soft_cropped)

        # Graph attention
        bone_gat_feats = self.bone_gat(bone_feats, bone_adj)
        soft_gat_feats = self.soft_gat(soft_feats, soft_adj)

        # Cross-tissue interaction
        bone_mean = bone_gat_feats.mean(dim=-1).unsqueeze(2)  # [B, N_bone, 1]
        soft_mean = soft_gat_feats.mean(dim=-1).unsqueeze(1)  # [B, 1, N_soft]
        cross_diff = bone_mean - soft_mean  # [B, N_bone, N_soft]
        cross_adj = torch.softmax(-cross_diff.abs(), dim=-1)
        bone_cross = torch.bmm(cross_adj, soft_gat_feats)
        soft_cross = torch.bmm(cross_adj.transpose(1, 2), bone_gat_feats)

        bone_cross_in = torch.cat([bone_gat_feats, bone_cross], dim=-1)
        soft_cross_in = torch.cat([soft_gat_feats, soft_cross], dim=-1)
        bone_enhanced = self.cross_gat(bone_cross_in, bone_adj)
        soft_enhanced = self.cross_gat(soft_cross_in, soft_adj)

        # Reconstruct full feature map
        enhanced_feats = torch.zeros(B, self.num_joints, self.graph_channels, device=coords.device)
        enhanced_feats[:, self.bone_indices] = bone_enhanced
        enhanced_feats[:, self.soft_indices] = soft_enhanced

        # Generate heatmap
        enhanced_feats = enhanced_feats.view(B, self.num_joints * self.graph_channels, 1, 1)
        enhanced_feats = enhanced_feats.expand(-1, -1, self.hm_height_dim, self.hm_width_dim)
        enhanced_feats = enhanced_feats.reshape(B, self.graph_channels, self.num_joints * self.hm_height_dim, self.hm_width_dim)
        refined_hm = self.refine_head(enhanced_feats)
        refined_hm = refined_hm.view(B, self.num_joints, self.num_joints, self.hm_height_dim, self.hm_width_dim).mean(dim=2)

        return refined_hm

class CrossAttentionAdj(nn.Module):
    def __init__(self, coord_dim, feat_dim, hidden_dim=128):
        super(CrossAttentionAdj, self).__init__()
        self.query = nn.Linear(coord_dim, hidden_dim)
        self.key = nn.Linear(feat_dim, hidden_dim)
        self.value = nn.Linear(feat_dim, hidden_dim)
    
    def forward(self, coords, feat):
        Q = self.query(coords)
        K = self.key(feat)
        V = self.value(feat)
        attn = torch.bmm(Q, K.transpose(1, 2)) / (K.shape[-1] ** 0.5)
        adj = torch.softmax(attn, dim=-1)
        return adj + torch.eye(coords.shape[1]).unsqueeze(0).to(coords.device)

class MultiGraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, num_layers=2, dropout=0.1):
        super(MultiGraphAttentionLayer, self).__init__()
        self.heads = heads
        self.out_channels = out_channels // heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        self.layers.append(
            nn.ModuleDict({
                'W': nn.Linear(in_channels, out_channels, bias=False),
                'attn': nn.Linear(out_channels * 2, heads, bias=False)
            })
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                nn.ModuleDict({
                    'W': nn.Linear(out_channels, out_channels, bias=False),
                    'attn': nn.Linear(out_channels * 2, heads, bias=False)
                })
            )
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = x
        for i in range(self.num_layers):
            h = self.layers[i]['W'](h)
            h = h.view(-1, h.shape[1], self.heads, self.out_channels)
            h1 = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1, -1)
            h2 = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1, -1)
            attn_input = torch.cat([h1, h2], dim=-1)
            attn = self.layers[i]['attn'](attn_input.view(h.shape[0], h.shape[1], h.shape[1], -1))
            attn = self.leaky_relu(attn)
            attn = F.softmax(attn, dim=2)
            attn = self.dropout(attn)
            h = torch.einsum('bnnh,bnhc->bnhc', attn, h)
            h = h.view(h.shape[0], h.shape[1], -1)
            if i > 0:
                h = h + x
            x = h
        return h

@MODEL.register_module
class HierarchicalGraphResFPNEnhancedFeatSample(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HierarchicalGraphResFPNEnhancedFeatSample, self).__init__()

        self.n_channels = cfg['IN_CHANNELS']
        self.gcn_layers = cfg['GCN_LAYERS']
        self.adj_type = cfg['ADJ_TYPE']
        self.num_refine_steps = cfg['NUM_REFINE_STEPS']
        if not (0 <= self.num_refine_steps <= 3):
            raise ValueError(f"num_refine_steps must be between 0 and 3, got {self.num_refine_steps}")

        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        import torchvision.models as tm
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained={cfg['PRETRAINED_RIGHT']})")

        self.feature_channel = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}[cfg['NUM_LAYERS']]
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
        self.head = FPNHead(
            feature_strides=(4, 8, 16, 32),
            in_channels=[self.decoder_feature_channel[0]] * 4,
            channels=512,
            num_classes=self.num_joints,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.refine_steps = nn.ModuleList([
            RefinementStep(
                graph_channels=256,
                num_joints=self.num_joints,
                hm_height_dim=self.hm_height_dim,
                hm_width_dim=self.hm_width_dim,
                bone_indices=self._preset_cfg['BONE_INDICES'],
                soft_indices=self._preset_cfg['SOFT_INDICES'],
                gcn_layers=self.gcn_layers,
                adj_type=self.adj_type,
                multi_scale_channels=[self.decoder_feature_channel[0]] * 4
            ) for _ in range(self.num_refine_steps)
        ])

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

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]
        all_heatmaps = []
        all_coords = []

        feats = self.preact.forward_feat(x)
        multi_scale_feats = self.neck(feats)
        init_hm = self.head(multi_scale_feats)
        all_heatmaps.append(init_hm)

        coords = self.integral_hm(init_hm)
        coords = coords.view(BATCH_SIZE, self.num_joints, 2)
        all_coords.append(coords)

        cropped = None
        if labels is not None and all(k in labels for k in ['cropped_1', 'cropped_2', 'cropped_3']):
            cropped = [labels['cropped_1'], labels['cropped_2'], labels['cropped_3']]
        else:
            raise ValueError("Cropped features are required for refinement")

        current_hm = init_hm
        for i, step in enumerate(self.refine_steps):
            current_hm = step(coords, multi_scale_feats, cropped, step_idx=i)
            all_heatmaps.append(current_hm)
            coords = self.integral_hm(current_hm).view(BATCH_SIZE, self.num_joints, 2)
            all_coords.append(coords)

        output_hm = current_hm if self.num_refine_steps > 0 else init_hm
        out_coord = coords
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        bone_pts = pred_pts[:, self._preset_cfg['BONE_INDICES']]
        soft_pts = pred_pts[:, self._preset_cfg['SOFT_INDICES']]
        scores = torch.mean(1 - pred_pts, dim=2, keepdim=True)

        return EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            all_heatmaps=all_heatmaps,
            all_coords=all_coords,
            maxvals=scores.float(),
            bone_struct=EasyDict(pred_pts=bone_pts, heatmap=output_hm[:, self._preset_cfg['BONE_INDICES']]),
            soft_struct=EasyDict(pred_pts=soft_pts, heatmap=output_hm[:, self._preset_cfg['SOFT_INDICES']])
        )