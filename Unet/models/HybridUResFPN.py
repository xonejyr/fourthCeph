""" Full assembly of the parts to form the complete network """
from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet 


from .components.unet_parts import *
from easydict import EasyDict
from Unet.utils import Softmax_Integral

from torchvision import models

from Unet.builder import MODEL

@MODEL.register_module
class HybridUResFPN(nn.Module):
    def __init__(self, bilinear=False, norm_layer=nn.BatchNorm2d, **cfg):
        super(HybridUResFPN, self).__init__()

        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.n_classes = self._preset_cfg['NUM_JOINTS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]  # 128
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]  # 128
        self.bilinear = bilinear
        self._norm_layer = norm_layer

        # ResNet 主干
        import torchvision.models as tm  # noqa: F401,F403
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained={cfg.get('PRETRAINED_RIGHT', True)})")

        self.feature_channel = {
            18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048
        }[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = {
            18: [64, 128, 256, 512], 34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048], 101: [256, 512, 1024, 2048], 152: [256, 512, 1024, 2048]
        }[cfg['NUM_LAYERS']]

        fpn_out_channels = self.decoder_feature_channel[0]

        # FPN 颈部
        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=fpn_out_channels,  # 256 for ResNet-50
            num_outs=4,
        )

        # U-Net 解码器
        #factor = 2 if bilinear else 1
        #factor = 1
        self.up1 = Up(self.decoder_feature_channel[2] , fpn_out_channels, bilinear)  # 
        self.up2 = Up(self.decoder_feature_channel[1] , fpn_out_channels, bilinear)  # 
        #self.up3 = Up(self.decoder_feature_channel[0] , fpn_out_channels, bilinear)  # 

        self.final_conv = nn.Conv2d(self.decoder_feature_channel[0], self.n_classes, kernel_size=3, padding=1)
        self.outc = OutConv(self.n_classes, self.n_classes)

        # Softmax_Integral
        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)

        # 加载预训练权重
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        # 编码器：提取 ResNet 特征
        feats = self.preact.forward_feat(x)  # 
        c2, c3, c4, c5 = feats  # [B, 256, 128, 128], [B, 512, 64, 64], [B, 1024, 32, 32], [B, 2048, 16, 16], 4, 8 ,16, 32 分之一
        print("===========================================================")
        print(f"the dimensions are \n c2: {c2.shape}, \n c3: {c3.shape}, \n c4: {c4.shape}, \n c5: {c5.shape}")

        # FPN 颈部：多尺度特征融合
        p_feats = self.neck(feats)  # [p2, p3, p4, p5]
        p2, p3, p4, p5 = p_feats  # 
        print(f"the dimensions are \n p2: {p2.shape}, \n p3: {p3.shape}, \n p4: {p4.shape}, \n p5: {p5.shape}")

        y = torch.cat([p4, c4], dim=1)
        print(y.shape)
        #print(self.bilinear)

        # U-Net 解码器：上采样
        x1 = self.up1(p4, c3)  # 
        x2 = self.up2(p3, c2)  # 
        #x3 = self.up3(p2, p2)  # 

        # 调整到目标热图尺寸
        x = self.final_conv(x2)                        # 
        x = F.interpolate(x, size=(self.hm_height_dim, self.hm_width_dim), mode='bilinear', align_corners=True)  #
        logits = self.outc(x)                          # 

        # Softmax_Integral 输出坐标
        out_coord = self.integral_hm(logits)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        # 兼容原 ResFPN 的输出格式
        scores = torch.ones_like(pred_pts[..., :1])  # 伪造 maxvals，保持输出一致性
        output = EasyDict(
            pred_pts=pred_pts,    # [batch_size, num_joints, 2]
            heatmap=logits,       #
            maxvals=scores.float()  # [batch_size, num_joints, 1]
        )
        return output

    def _initialize(self):
        pass

# 测试代码
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义配置参数
    cfg = {
        'IN_CHANNELS': 3,
        'NUM_LAYERS': 18,  # 使用 ResNet-18 测试
        'PRESET': {
            'NUM_JOINTS': 17,  # 假设检测 17 个关键点
            'HEATMAP_SIZE': [128, 128]  # 热图尺寸
        },
        'PRETRAINED_RIGHT': True
    }
    
    # 初始化模型
    model = HybridUResFPN(bilinear=True, norm_layer=nn.BatchNorm2d, **cfg)
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 创建测试输入 [B, C, H, W] = [8, 3, 512, 512]
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # 设置调试模式
    torch.autograd.set_detect_anomaly(True)
    
    print("Starting dimension verification...")
    print(f"Input shape: {input_tensor.shape}")
    
    # 前向传播并逐步验证维度
    with torch.no_grad():
        # 1. ResNet 编码器
        feats = model.preact.forward_feat(input_tensor)
        c2, c3, c4, c5 = feats
        print("\nAfter ResNet Encoder:")
        print(f"c2 shape: {c2.shape}")  # 预期 [8, 64, 128, 128]
        print(f"c3 shape: {c3.shape}")  # 预期 [8, 128, 64, 64]
        print(f"c4 shape: {c4.shape}")  # 预期 [8, 256, 32, 32]
        print(f"c5 shape: {c5.shape}")  # 预期 [8, 512, 16, 16]
        
        # 2. FPN 颈部
        p_feats = model.neck(feats)
        p2, p3, p4, p5 = p_feats
        print("\nAfter FPN Neck:")
        print(f"p2 shape: {p2.shape}")  # 预期 [8, 64, 128, 128]
        print(f"p3 shape: {p3.shape}")  # 预期 [8, 64, 64, 64]
        print(f"p4 shape: {p4.shape}")  # 预期 [8, 64, 32, 32]
        print(f"p5 shape: {p5.shape}")  # 预期 [8, 64, 16, 16]
        
        # 3. U-Net 解码器第一步
        y = torch.cat([p4, c4], dim=1)
        print(f"\nAfter concat p4 and c4: {y.shape}")  # 预期 [8, 64+256, 32, 32]
        
        x1 = model.up1(y, c3)
        print(f"After up1: {x1.shape}")  # 预期 [8, 64, 64, 64]
        
        # 4. U-Net 解码器第二步
        x2_input = torch.cat([x1, c3], dim=1)
        print(f"After concat x1 and c3: {x2_input.shape}")  # 预期 [8, 64+128, 64, 64]
        
        x2 = model.up2(x2_input, c2)
        print(f"After up2: {x2.shape}")  # 预期 [8, 64, 128, 128]
        
        # 5. U-Net 解码器第三步
        x3_input = torch.cat([x2, c2], dim=1)
        print(f"After concat x2 and c2: {x3_input.shape}")  # 预期 [8, 64+64, 128, 128]
        
        x3 = model.up3(x3_input, p2)
        print(f"After up3: {x3.shape}")  # 预期 [8, 64, 128, 128]
        
        # 6. 输出层
        x = model.final_conv(x3)
        print(f"After final_conv: {x.shape}")  # 预期 [8, 17, 128, 128]
        
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        print(f"After interpolate: {x.shape}")  # 预期 [8, 17, 128, 128]
        
        logits = model.outc(x)
        print(f"After outc: {logits.shape}")  # 预期 [8, 17, 128, 128]
        
        # 7. Softmax_Integral
        out_coord = model.integral_hm(logits)
        print(f"After integral_hm: {out_coord.shape}")  # 预期 [8, 17*2]
        
        pred_pts = out_coord.reshape(batch_size, model.num_joints, 2)
        print(f"Final pred_pts: {pred_pts.shape}")  # 预期 [8, 17, 2]
        
        # 8. 完整输出
        output = model(input_tensor)
        print("\nFinal output shapes:")
        print(f"pred_pts: {output.pred_pts.shape}")  # 预期 [8, 17, 2]
        print(f"heatmap: {output.heatmap.shape}")  # 预期 [8, 17, 128, 128]
        print(f"maxvals: {output.maxvals.shape}")  # 预期 [8, 17, 1]

    print("\nDimension verification completed!")

"""
输入图像 [B, 3, H, W]
    ↓
[ResNet18 Encoder]
    ├─── conv1 + bn1 + relu → C1 [B, 64, H/2, W/2]
    ├─── maxpool           → C1' [B, 64, H/4, W/4]
    ├─── layer1            → C2 [B, 64, H/4, W/4]
    ├─── layer2            → C3 [B, 128, H/8, W/8]
    ├─── layer3            → C4 [B, 256, H/16, W/16]
    └─── layer4            → C5 [B, 512, H/32, W/32]
                      ↓    ↓    ↓    ↓
[FPN 特征金字塔]
    ├─── C5 → fpn_conv4 → P4 [B, 256, H/32, W/32]
    ├─── C4 → fpn_conv3 + 上采样P4 → P3 [B, 256, H/16, W/16]
    └─── C3 → fpn_conv2 + 上采样P3 → P2 [B, 256, H/8, W/8]
                      ↓    ↓    ↓
[U-Net 解码器]
    ├─── P4 + C4 → Up1 → D3 [B, 256, H/16, W/16]
    ├─── D3 + C3 → Up2 → D2 [B, 128, H/8, W/8]
    └─── D2 + C2 → Up3 → D1 [B, 64, H/4, W/4]
                      ↓
[输出层]
    ├─── D1 → OutConv → 热图 [B, num_joints, hm_height, hm_width]
    └─── Softmax_Integral → 坐标 [B, num_joints, 2]
输出：EasyDict(pred_pts, heatmap)
"""