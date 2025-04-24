from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet
from easydict import EasyDict
from Unet.utils import Softmax_Integral
from Unet.builder import MODEL
import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossAttentionAdj(nn.Module):
    def __init__(self, coord_dim, feat_dim, hidden_dim=128):
        super(CrossAttentionAdj, self).__init__()
        self.query = nn.Linear(coord_dim, hidden_dim)
        self.key = nn.Linear(feat_dim, hidden_dim)
        self.value = nn.Linear(feat_dim, hidden_dim)
    
    def forward(self, coords, feat):
        Q = self.query(coords)  # [B, N, hidden_dim]
        K = self.key(feat)      # [B, N, hidden_dim]
        V = self.value(feat)    # [B, N, hidden_dim]
        attn = torch.bmm(Q, K.transpose(1, 2)) / (K.shape[-1] ** 0.5)  # [B, N, N]
        adj = torch.softmax(attn, dim=-1)
        return adj + torch.eye(coords.shape[1]).unsqueeze(0).to(coords.device)

class MultiGraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, num_layers=2, dropout=0.1):
        super(MultiGraphAttentionLayer, self).__init__()
        self.heads = heads
        self.out_channels = out_channels // heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            nn.ModuleDict({
                'W': nn.Linear(in_channels, out_channels, bias=False),
                'attn': nn.Linear(out_channels * 2, heads, bias=False)
            })
        )
        # Subsequent layers
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
        h = x  # [B, N, C_in]
        for i in range(self.num_layers):
            # Linear transformation
            h = self.layers[i]['W'](h)  # [B, N, C_out]
            h = h.view(-1, h.shape[1], self.heads, self.out_channels)  # [B, N, heads, C_out/heads]
            
            # Attention mechanism
            h1 = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1, -1)  # [B, N, N, heads, C_out/heads]
            h2 = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1, -1)  # [B, N, N, heads, C_out/heads]
            attn_input = torch.cat([h1, h2], dim=-1)  # [B, N, N, heads, 2*C_out/heads]
            attn = self.layers[i]['attn'](attn_input.view(h.shape[0], h.shape[1], h.shape[1], -1))  # [B, N, N, heads]
            attn = self.leaky_relu(attn)
            attn = F.softmax(attn, dim=2)  # [B, N, N, heads]
            attn = self.dropout(attn)
            
            # Apply attention
            h = torch.einsum('bnnh,bnhc->bnhc', attn, h)  # [B, N, heads, C_out/heads]
            h = h.view(h.shape[0], h.shape[1], -1)  # [B, N, C_out]
            
            # Residual connection (if not the first layer)
            if i > 0:
                h = h + x
            x = h  # Update input for the next layer
        
        return h

@MODEL.register_module
class HierarchicalGraphResFPNMultichannel(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HierarchicalGraphResFPNMultichannel, self).__init__()

        self.n_channels = cfg['IN_CHANNELS']
        self.gcn_layers = cfg['GCN_LAYERS']
        self.adj_type = cfg['ADJ_TYPE']
        
        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]

        # 其余配置保持不变
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

        self.bone_num_joints = self._preset_cfg['NUM_JOINTS_BONE']
        self.soft_num_joints = self._preset_cfg['NUM_JOINTS_SOFT']
        self.bone_indices = self._preset_cfg['BONE_INDICES']
        self.soft_indices = self._preset_cfg['SOFT_INDICES']


        # Bone-specific feature extractor
        self.bone_preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        self.bone_neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
        )
        self.bone_head = FPNHead(
            feature_strides=(4, 8, 16, 32),
            in_channels=[self.decoder_feature_channel[0]] * 4,
            channels=512,
            num_classes=self._preset_cfg['NUM_JOINTS_BONE'],
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        # Soft-specific feature extractor
        self.soft_preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        self.soft_neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
        )
        self.soft_head = FPNHead(
            feature_strides=(4, 8, 16, 32),
            in_channels=[self.decoder_feature_channel[0]] * 4,
            channels=512,
            num_classes=self._preset_cfg['NUM_JOINTS_SOFT'],
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        
        # Graph enhancement (保持不变)
        self.spatial_dim = self.hm_height_dim * self.hm_width_dim
        self.graph_channels = 256
        self.reduce = nn.Linear(self.spatial_dim, self.graph_channels)
        
        self.bone_gat = MultiGraphAttentionLayer(self.graph_channels, self.graph_channels, num_layers=self.gcn_layers)
        self.soft_gat = MultiGraphAttentionLayer(self.graph_channels, self.graph_channels, num_layers=self.gcn_layers)
        self.cross_gat = MultiGraphAttentionLayer(self.graph_channels * 2, self.graph_channels, num_layers=self.gcn_layers)

        self.cross_adj = CrossAttentionAdj(coord_dim=2, feat_dim=self.hm_height_dim * self.hm_width_dim)

        self.final_head = nn.Conv2d(self.graph_channels, self.num_joints, 1)
        self.integral_hm_bone = Softmax_Integral(num_pts=self.bone_num_joints,
                                          hm_width=self.hm_width_dim,
                                          hm_height=self.hm_height_dim)
        self.integral_hm_soft = Softmax_Integral(num_pts=self.soft_num_joints,
                                          hm_width=self.hm_width_dim,
                                          hm_height=self.hm_height_dim)
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                          hm_width=self.hm_width_dim,
                                          hm_height=self.hm_height_dim)

        # 初始化两个backbone的权重
        model_state = self.bone_preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                if k in model_state and v.size() == model_state[k].size()}
        model_state.update(state)
        self.bone_preact.load_state_dict(model_state)
        self.soft_preact.load_state_dict(model_state)

    def _build_dynamic_adj(self, coords, heatmap):
        """利用目标坐标和heatmap动态构建邻接矩阵"""
        B, N = heatmap.shape[0], heatmap.shape[1]
        # Normalize coordinates
        coords = coords.view(B,  N, 2)  # [B, N, 2]
        coords = coords / torch.tensor([self.hm_width_dim, self.hm_height_dim], device=coords.device)
        
        # Distance-based adjacency
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
        dist = torch.norm(diff, dim=-1)  # [B, N, N]
        adj = torch.exp(-dist / dist.mean())
        
        # Feature-based adjacency
        feat = heatmap.view(B, N, -1)
        feat_adj = torch.bmm(feat, feat.transpose(1, 2))
        feat_adj = torch.softmax(feat_adj, dim=-1)
        
        return 0.5 * adj + 0.5 * feat_adj  # Combine spatial and feature similarity
    
    def _build_dynamic_adj_learnableAddCombine(self, coords, heatmap):
        B, N = heatmap.shape[0], heatmap.shape[1]
        # Normalize coordinates
        coords = coords.view(B, N, 2)  # [B, N, 2]
        coords = coords / torch.tensor([self.hm_width_dim, self.hm_height_dim], device=coords.device)

        # Distance-based adjacency
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
        dist = torch.norm(diff, dim=-1)  # [B, N, N]
        adj = torch.exp(-dist / dist.mean())  # [B, N, N]

        # Feature-based adjacency
        feat = heatmap.view(B, N, -1)
        feat_adj = torch.bmm(feat, feat.transpose(1, 2))
        feat_adj = torch.softmax(feat_adj, dim=-1)  # [B, N, N]

        # Learnable weighting
        gate = torch.sigmoid(nn.Parameter(torch.ones(1, device=coords.device)))  # Learnable scalar
        combined_adj = gate * adj + (1 - gate) * feat_adj

        # Optional: Add self-loops to preserve node identity
        combined_adj = combined_adj + torch.eye(N, device=coords.device).unsqueeze(0)

        return combined_adj
    
    def _build_dynamic_adj_cross(self, coords, heatmap):
        #print("the device of coords is: ", coords.device)
        #print("the device of heatmap is: ", heatmap.device)
        B, N = heatmap.shape[0], heatmap.shape[1]
        coords = coords.view(B, N, 2) / torch.tensor([self.hm_width_dim, self.hm_height_dim], device=coords.device)
        feat = heatmap.view(B, N, -1)

        #cross_adj = 
        #print("the device of coords is: ", coords.device)
        #print("the device of feat is: ", feat.device)
        adj = self.cross_adj(coords, feat)
        return adj
    
    def _heatmap_to_features(self, heatmap):
        B, N, H, W = heatmap.shape
        feat = heatmap.view(B, N, -1)
        return self.reduce(feat)  # [B, N, graph_channels]

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        # Bone feature extraction
        bone_feats = self.bone_preact.forward_feat(x)
        bone_feats = self.bone_neck(bone_feats)
        bone_hm = self.bone_head(bone_feats)  # [B, bone_num_joints, H, W]

        # Soft feature extraction
        soft_feats = self.soft_preact.forward_feat(x)
        soft_feats = self.soft_neck(soft_feats)
        soft_hm = self.soft_head(soft_feats)  # [B, soft_num_joints, H, W]

        # Get initial coordinates
        bone_coords = self.integral_hm_bone(bone_hm).view(BATCH_SIZE, self.bone_num_joints, 2)
        soft_coords = self.integral_hm_soft(soft_hm).view(BATCH_SIZE, self.soft_num_joints, 2)

        # Build adjacency matrices
        if self.adj_type == 'learnableAddCombine':
            bone_adj = self._build_dynamic_adj_learnableAddCombine(bone_coords, bone_hm)  # [B, bone_num_joints, bone_num_joints]
            soft_adj = self._build_dynamic_adj_learnableAddCombine(soft_coords, soft_hm)  # [B, soft_num_joints, soft_num_joints]
        elif self.adj_type == 'cross':
            #print('the dvice of bone_coords is: ', bone_coords.device)
            #print('the device of bone_hm is: ', bone_hm.device)
            bone_adj = self._build_dynamic_adj_cross(bone_coords, bone_hm)  # [B, bone_num_joints, bone_num_joints]
            soft_adj = self._build_dynamic_adj_cross(soft_coords, soft_hm)  # [B, soft_num_joints, soft_num_joints]
        elif self.adj_type == 'const':
            bone_adj = self._build_dynamic_adj(bone_coords, bone_hm)  # [B, bone_num_joints, bone_num_joints]
            soft_adj = self._build_dynamic_adj(soft_coords, soft_hm)  # [B, soft_num_joints, soft_num_joints]
        else:
            raise ValueError('Invalid adjacency type.')

        # Graph enhancement
        bone_feats = self._heatmap_to_features(bone_hm)
        soft_feats = self._heatmap_to_features(soft_hm)

        bone_gat_feats = self.bone_gat(bone_feats, bone_adj)
        soft_gat_feats = self.soft_gat(soft_feats, soft_adj)

        # Cross-tissue interaction
        cross_diff = bone_hm.mean(dim=(2, 3)).unsqueeze(2) - soft_hm.mean(dim=(2, 3)).unsqueeze(1)
        cross_adj = torch.softmax(-cross_diff.abs(), dim=-1)

        bone_cross = torch.bmm(cross_adj, soft_gat_feats)
        soft_cross = torch.bmm(cross_adj.transpose(1, 2), bone_gat_feats)

        bone_cross_in = torch.cat([bone_gat_feats, bone_cross], dim=-1)
        soft_cross_in = torch.cat([soft_gat_feats, soft_cross], dim=-1)

        bone_enhanced = self.cross_gat(bone_cross_in, bone_adj)
        soft_enhanced = self.cross_gat(soft_cross_in, soft_adj)

        # Reconstruct full feature map
        enhanced_feats = torch.zeros(BATCH_SIZE, self.num_joints, self.graph_channels, 
                                   device=x.device)
        enhanced_feats[:, self.bone_indices] = bone_enhanced
        enhanced_feats[:, self.soft_indices] = soft_enhanced

        # Final processing (保持不变)
        enhanced_feats = enhanced_feats.view(BATCH_SIZE, self.num_joints * self.graph_channels, 1, 1)
        enhanced_feats = enhanced_feats.expand(-1, -1, self.hm_height_dim, self.hm_width_dim)
        enhanced_feats = enhanced_feats.reshape(BATCH_SIZE, self.graph_channels, self.num_joints * self.hm_height_dim, self.hm_width_dim)

        output_hm = self.final_head(enhanced_feats)
        output_hm = output_hm.view(BATCH_SIZE, self.num_joints, self.num_joints, self.hm_height_dim, self.hm_width_dim)
        output_hm = output_hm.mean(dim=2)

        # Add residual connection with initial heatmaps
        init_hm = torch.zeros_like(output_hm)
        init_hm[:, self.bone_indices] = bone_hm
        init_hm[:, self.soft_indices] = soft_hm
        output_hm = output_hm + init_hm

        out_coord = self.integral_hm(output_hm)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        bone_pts = pred_pts[:, self.bone_indices]
        soft_pts = pred_pts[:, self.soft_indices]

        scores = 1 - pred_pts
        scores = torch.mean(scores, dim=2, keepdim=True)

        return EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            maxvals=scores.float(),
            bone_struct=EasyDict(pred_pts=bone_pts, heatmap=output_hm[:, self.bone_indices]),
            soft_struct=EasyDict(pred_pts=soft_pts, heatmap=output_hm[:, self.soft_indices])
        )
    
    def _initialize(self):
        pass