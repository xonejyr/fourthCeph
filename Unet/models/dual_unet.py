""" Full assembly of the parts to form the complete network """

from .components.unet_parts import *
from easydict import EasyDict
from Unet.utils import Softmax_Integral

from Unet.builder import MODEL

# 双U-Net模型
@MODEL.register_module
class DualUNet(nn.Module):
    def __init__(self, bilinear=False, **cfg):
        super(DualUNet, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.bilinear = bilinear

        # 骨组织U-Net（15个标志点）
        self.bone_num_joints = self._preset_cfg['NUM_JOINTS_BONE']
        self.bone_unet = self._build_unet(self.bone_num_joints)

        # 软组织U-Net（4个标志点）
        self.soft_num_joints = self._preset_cfg['NUM_JOINTS_SOFT']
        self.soft_unet = self._build_unet(self.soft_num_joints)

        self.bone_indices = self._preset_cfg['BONE_INDICES']
        self.soft_indices = self._preset_cfg['SOFT_INDICES']

    def _build_unet(self, num_joints):
        inc = DoubleConv(self.n_channels, 64)
        down1 = Down(64, 128)
        down2 = Down(128, 256)
        down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        down4 = Down(512, 1024 // factor)
        up1 = Up(1024, 512 // factor, self.bilinear)
        up2 = Up(512, 256 // factor, self.bilinear)
        up3 = Up(256, 128 // factor, self.bilinear)
        up4 = Up(128, 64, self.bilinear)
        outc = OutConv(64, num_joints)
        
        integral_hm = Softmax_Integral(num_pts=num_joints,
                                      hm_width=self.hm_width_dim,
                                      hm_height=self.hm_height_dim)
        
        return nn.ModuleDict({
            'inc': inc, 'down1': down1, 'down2': down2, 'down3': down3, 'down4': down4,
            'up1': up1, 'up2': up2, 'up3': up3, 'up4': up4, 'outc': outc,
            'integral_hm': integral_hm
        })
    
    def _initialize(self):
        pass

    def forward(self, x, target_uv_bone=None, target_uv_soft=None):
        batch_size = x.shape[0]

        # 骨组织U-Net前向传播
        x1_b = self.bone_unet['inc'](x)
        x2_b = self.bone_unet['down1'](x1_b)
        x3_b = self.bone_unet['down2'](x2_b)
        x4_b = self.bone_unet['down3'](x3_b)
        x5_b = self.bone_unet['down4'](x4_b)
        x_b = self.bone_unet['up1'](x5_b, x4_b)
        x_b = self.bone_unet['up2'](x_b, x3_b)
        x_b = self.bone_unet['up3'](x_b, x2_b)
        x_b = self.bone_unet['up4'](x_b, x1_b)
        logits_bone = self.bone_unet['outc'](x_b)
        out_coord_bone = self.bone_unet['integral_hm'](logits_bone)
        pred_pts_bone = out_coord_bone.reshape(batch_size, self.bone_num_joints, 2)

        # 软组织U-Net前向传播
        x1_s = self.soft_unet['inc'](x)
        x2_s = self.soft_unet['down1'](x1_s)
        x3_s = self.soft_unet['down2'](x2_s)
        x4_s = self.soft_unet['down3'](x3_s)
        x5_s = self.soft_unet['down4'](x4_s)
        x_s = self.soft_unet['up1'](x5_s, x4_s)
        x_s = self.soft_unet['up2'](x_s, x3_s)
        x_s = self.soft_unet['up3'](x_s, x2_s)
        x_s = self.soft_unet['up4'](x_s, x1_s)
        logits_soft = self.soft_unet['outc'](x_s)
        out_coord_soft = self.soft_unet['integral_hm'](logits_soft)
        pred_pts_soft = out_coord_soft.reshape(batch_size, self.soft_num_joints, 2)


        # 拼接骨组织和软组织预测点为 [batch_size, 19, 2]
        # 初始化 pred_pts 为全零张量，形状为 [batch_size, 19, 2]
        pred_pts = torch.zeros(batch_size, 19, 2, device=pred_pts_bone.device)
        heatmap = torch.zeros(batch_size, 19, self.hm_height_dim, self.hm_width_dim, device=pred_pts_bone.device)

        # 将骨组织预测点放回对应位置
        for idx, bone_idx in enumerate(self.bone_indices):
            pred_pts[:, bone_idx, :] = pred_pts_bone[:, idx, :]
            heatmap[:, bone_idx, :, :] = logits_bone[:, idx, :, :]

        # 将软组织预测点放回对应位置
        for idx, soft_idx in enumerate(self.soft_indices):
            pred_pts[:, soft_idx, :] = pred_pts_soft[:, idx, :]
            heatmap[:, soft_idx, :, :] = logits_soft[:, idx, :, :]
            

        # 输出结果
        output = EasyDict(
            pred_pts=pred_pts, # [batchsize, num_joints, 2]
            pred_pts_bone=pred_pts_bone,  # [batch_size, 12, 2]
            pred_pts_soft=pred_pts_soft,  # [batch_size, 7, 2]
            heatmap_bone=logits_bone,     # [batch_size, 12, hm_height, hm_width]
            heatmap_soft=logits_soft,      # [batch_size, 7, hm_height, hm_width]
            heatmap=heatmap      # [batch_size, 19, hm_height, hm_width]  # 整体热图，用于干预模块的因果推理
        )
        return output
    
@MODEL.register_module
class DualUNet_onlyBone(nn.Module):
    def __init__(self, bilinear=False, **cfg):
        super(DualUNet_onlyBone, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.bilinear = bilinear

        # 骨组织U-Net（15个标志点）
        self.bone_num_joints = self._preset_cfg['NUM_JOINTS_BONE']
        self.bone_unet = self._build_unet(self.bone_num_joints)

        ## 软组织U-Net（4个标志点）
        #self.soft_num_joints = self._preset_cfg['NUM_SOFT_JOINTS']
        #self.soft_unet = self._build_unet(self.soft_num_joints)

        self.bone_indices = self._preset_cfg['BONE_INDICES']
        #self.soft_indices = self._preset_cfg['SOFT_INDICES']

    def _build_unet(self, num_joints):
        inc = DoubleConv(self.n_channels, 64)
        down1 = Down(64, 128)
        down2 = Down(128, 256)
        down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        down4 = Down(512, 1024 // factor)
        up1 = Up(1024, 512 // factor, self.bilinear)
        up2 = Up(512, 256 // factor, self.bilinear)
        up3 = Up(256, 128 // factor, self.bilinear)
        up4 = Up(128, 64, self.bilinear)
        outc = OutConv(64, num_joints)
        
        integral_hm = Softmax_Integral(num_pts=num_joints,
                                      hm_width=self.hm_width_dim,
                                      hm_height=self.hm_height_dim)
        
        return nn.ModuleDict({
            'inc': inc, 'down1': down1, 'down2': down2, 'down3': down3, 'down4': down4,
            'up1': up1, 'up2': up2, 'up3': up3, 'up4': up4, 'outc': outc,
            'integral_hm': integral_hm
        })
    
    def _initialize(self):
        pass

    def forward(self, x, target_uv_bone=None, target_uv_soft=None):
        batch_size = x.shape[0]

        # 骨组织U-Net前向传播
        x1_b = self.bone_unet['inc'](x)
        x2_b = self.bone_unet['down1'](x1_b)
        x3_b = self.bone_unet['down2'](x2_b)
        x4_b = self.bone_unet['down3'](x3_b)
        x5_b = self.bone_unet['down4'](x4_b)
        x_b = self.bone_unet['up1'](x5_b, x4_b)
        x_b = self.bone_unet['up2'](x_b, x3_b)
        x_b = self.bone_unet['up3'](x_b, x2_b)
        x_b = self.bone_unet['up4'](x_b, x1_b)
        logits_bone = self.bone_unet['outc'](x_b)
        out_coord_bone = self.bone_unet['integral_hm'](logits_bone)
        pred_pts_bone = out_coord_bone.reshape(batch_size, self.bone_num_joints, 2)

        ## 软组织U-Net前向传播
        #x1_s = self.soft_unet['inc'](x)
        #x2_s = self.soft_unet['down1'](x1_s)
        #x3_s = self.soft_unet['down2'](x2_s)
        #x4_s = self.soft_unet['down3'](x3_s)
        #x5_s = self.soft_unet['down4'](x4_s)
        #x_s = self.soft_unet['up1'](x5_s, x4_s)
        #x_s = self.soft_unet['up2'](x_s, x3_s)
        #x_s = self.soft_unet['up3'](x_s, x2_s)
        #x_s = self.soft_unet['up4'](x_s, x1_s)
        #logits_soft = self.soft_unet['outc'](x_s)
        #out_coord_soft = self.soft_unet['integral_hm'](logits_soft)
        #pred_pts_soft = out_coord_soft.reshape(batch_size, self.soft_num_joints, 2)
#

        ## 拼接骨组织和软组织预测点为 [batch_size, 19, 2]
        ## 初始化 pred_pts 为全零张量，形状为 [batch_size, 19, 2]
        #pred_pts = torch.zeros(batch_size, 19, 2, device=pred_pts_bone.device)
        #heatmap = torch.zeros(batch_size, 19, self.hm_height_dim, self.hm_width_dim, device=pred_pts_bone.device)

        ## 将骨组织预测点放回对应位置
        #for idx, bone_idx in enumerate(self.bone_indices):
        #    pred_pts[:, bone_idx, :] = pred_pts_bone[:, idx, :]
        #    heatmap[:, bone_idx, :, :] = logits_bone[:, idx, :, :]
#
        ## 将软组织预测点放回对应位置
        #for idx, soft_idx in enumerate(self.soft_indices):
        #    pred_pts[:, soft_idx, :] = pred_pts_soft[:, idx, :]
        #    heatmap[:, soft_idx, :, :] = logits_soft[:, idx, :, :]
        #    

        # 输出结果
        output = EasyDict(# [batchsize, num_joints, 2]
            pred_pts_bone=pred_pts_bone,  # [batch_size, 12, 2]
            heatmap_bone=logits_bone,     # [batch_size, 12, hm_height, hm_width]hm_height, hm_width]
        )
        return output
    

@MODEL.register_module
class DualUNet_onlySoft(nn.Module):
    def __init__(self, bilinear=False, **cfg):
        super(DualUNet_onlySoft, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']
        self._preset_cfg = cfg['PRESET']
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.bilinear = bilinear

        ## 骨组织U-Net（15个标志点）
        #self.bone_num_joints = self._preset_cfg['NUM_BONE_JOINTS']
        #self.bone_unet = self._build_unet(self.bone_num_joints)

        # 软组织U-Net（4个标志点）
        self.soft_num_joints = self._preset_cfg['NUM_JOINTS_SOFT']
        self.soft_unet = self._build_unet(self.soft_num_joints)

        #self.bone_indices = self._preset_cfg['BONE_INDICES']
        self.soft_indices = self._preset_cfg['SOFT_INDICES']

    def _build_unet(self, num_joints):
        inc = DoubleConv(self.n_channels, 64)
        down1 = Down(64, 128)
        down2 = Down(128, 256)
        down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        down4 = Down(512, 1024 // factor)
        up1 = Up(1024, 512 // factor, self.bilinear)
        up2 = Up(512, 256 // factor, self.bilinear)
        up3 = Up(256, 128 // factor, self.bilinear)
        up4 = Up(128, 64, self.bilinear)
        outc = OutConv(64, num_joints)
        
        integral_hm = Softmax_Integral(num_pts=num_joints,
                                      hm_width=self.hm_width_dim,
                                      hm_height=self.hm_height_dim)
        
        return nn.ModuleDict({
            'inc': inc, 'down1': down1, 'down2': down2, 'down3': down3, 'down4': down4,
            'up1': up1, 'up2': up2, 'up3': up3, 'up4': up4, 'outc': outc,
            'integral_hm': integral_hm
        })
    
    def _initialize(self):
        pass

    def forward(self, x, target_uv_bone=None, target_uv_soft=None):
        batch_size = x.shape[0]

        ## 骨组织U-Net前向传播
        #x1_b = self.bone_unet['inc'](x)
        #x2_b = self.bone_unet['down1'](x1_b)
        #x3_b = self.bone_unet['down2'](x2_b)
        #x4_b = self.bone_unet['down3'](x3_b)
        #x5_b = self.bone_unet['down4'](x4_b)
        #x_b = self.bone_unet['up1'](x5_b, x4_b)
        #x_b = self.bone_unet['up2'](x_b, x3_b)
        #x_b = self.bone_unet['up3'](x_b, x2_b)
        #x_b = self.bone_unet['up4'](x_b, x1_b)
        #logits_bone = self.bone_unet['outc'](x_b)
        #out_coord_bone = self.bone_unet['integral_hm'](logits_bone)
        #pred_pts_bone = out_coord_bone.reshape(batch_size, self.bone_num_joints, 2)

        # 软组织U-Net前向传播
        x1_s = self.soft_unet['inc'](x)
        x2_s = self.soft_unet['down1'](x1_s)
        x3_s = self.soft_unet['down2'](x2_s)
        x4_s = self.soft_unet['down3'](x3_s)
        x5_s = self.soft_unet['down4'](x4_s)
        x_s = self.soft_unet['up1'](x5_s, x4_s)
        x_s = self.soft_unet['up2'](x_s, x3_s)
        x_s = self.soft_unet['up3'](x_s, x2_s)
        x_s = self.soft_unet['up4'](x_s, x1_s)
        logits_soft = self.soft_unet['outc'](x_s)
        out_coord_soft = self.soft_unet['integral_hm'](logits_soft)
        pred_pts_soft = out_coord_soft.reshape(batch_size, self.soft_num_joints, 2)


        ## 拼接骨组织和软组织预测点为 [batch_size, 19, 2]
        ## 初始化 pred_pts 为全零张量，形状为 [batch_size, 19, 2]
        #pred_pts = torch.zeros(batch_size, 19, 2, device=pred_pts_bone.device)
        #heatmap = torch.zeros(batch_size, 19, self.hm_height_dim, self.hm_width_dim, device=pred_pts_bone.device)
#
        ## 将骨组织预测点放回对应位置
        #for idx, bone_idx in enumerate(self.bone_indices):
        #    pred_pts[:, bone_idx, :] = pred_pts_bone[:, idx, :]
        #    heatmap[:, bone_idx, :, :] = logits_bone[:, idx, :, :]
#
        ## 将软组织预测点放回对应位置
        #for idx, soft_idx in enumerate(self.soft_indices):
        #    pred_pts[:, soft_idx, :] = pred_pts_soft[:, idx, :]
        #    heatmap[:, soft_idx, :, :] = logits_soft[:, idx, :, :]
        #    

        # 输出结果
        output = EasyDict(
            pred_pts_soft=pred_pts_soft,  # [batch_size, 7, 2]
            heatmap_soft=logits_soft,      # [batch_size, 7, hm_height, hm_width]
        )
        return output