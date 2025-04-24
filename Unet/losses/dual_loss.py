import torch.nn as nn
import torch

from Unet.builder import LOSS
from .utils import get_loss_by_const_mask, get_loss_by_pow_mask

#################################################################################80
# doualMSELoss for Dual U-Net (Bone and Soft Tissue)
"""
# 假设输入和标签
output = {
    'heatmap_bone': torch.randn(2, 12, 64, 64),  # 骨组织热图
    'heatmap_soft': torch.randn(2, 7, 64, 64),   # 软组织热图
    'pred_pts_bone': torch.randn(2, 12, 2),      # 骨组织坐标
    'pred_pts_soft': torch.randn(2, 7, 2)        # 软组织坐标
}
labels = {
    'target_hm_bone': torch.randn(2, 12, 64, 64),
    'target_hm_weight_bone': torch.ones(2, 12, 64, 64),
    'target_hm_soft': torch.randn(2, 7, 64, 64),
    'target_hm_weight_soft': torch.ones(2, 7, 64, 64),
    'target_uv_bone': torch.randn(2, 12 * 2),
    'target_uv_weight_bone': torch.ones(2, 12 * 2),
    'target_uv_soft': torch.randn(2, 7 * 2),
    'target_uv_weight_soft': torch.ones(2, 7 * 2)
}
"""

@LOSS.register_module
class DualMSELoss_softmax(nn.Module):
    ''' MSE Loss for Dual U-Net (Bone and Soft Tissue)
    '''
    def __init__(self, **cfg):
        super(DualMSELoss_softmax, self).__init__()
        self.criterion = nn.MSELoss()
        self._type = cfg['PRESET']['METHOD_TYPE']  # 'heatmap' 或 'coord'
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        if self._type == 'heatmap':
            # 骨组织热图损失
            batchsize, bone_joints, hm_height, hm_width = output['heatmap_bone'].size()
            gt_hm_bone = labels['target_hm_bone']
            gt_hm_weight_bone = labels['target_hm_weight_bone']
            pred_hm_bone = torch.softmax(output['heatmap_bone'].view(batchsize, bone_joints, -1), dim=-1).view_as(output['heatmap_bone'])  # softmax

            if self._mask_type == 'const':
                bone_loss = get_loss_by_const_mask(pred_hm_bone, gt_hm_bone, self.hm_criterion, self._basenumber)
            elif self._mask_type == 'pow':
                bone_loss = get_loss_by_pow_mask(pred_hm_bone, gt_hm_bone, gt_hm_weight_bone, self.hm_criterion, self._basenumber)
            else:
                raise ValueError("Unsupported mask type, you should choose either 'const' or 'pow'")

            # 软组织热图损失
            batchsize, soft_joints, hm_height, hm_width = output['heatmap_soft'].size()
            gt_hm_soft = labels['target_hm_soft']
            gt_hm_weight_soft = labels['target_hm_weight_soft']
            pred_hm_soft = torch.softmax(output['heatmap_soft'].view(batchsize, soft_joints, -1), dim=-1).view_as(output['heatmap_soft'])  # softmax

            if self._mask_type == 'const':
                soft_loss = get_loss_by_const_mask(pred_hm_soft, gt_hm_soft, self.hm_criterion, self._basenumber)
            elif self._mask_type == 'pow':
                soft_loss = get_loss_by_pow_mask(pred_hm_soft, gt_hm_soft, gt_hm_weight_soft, self.hm_criterion, self._basenumber)
            else:
                raise ValueError("Unsupported mask type, you should choose either 'const' or 'pow'")

            # 总损失：骨组织和软组织损失的平均
            loss = 0.5 * (bone_loss + soft_loss)

        elif self._type == 'coord':
            # 骨组织坐标损失
            pred_pts_bone = output['pred_pts_bone'].reshape(labels['target_uv_bone'].shape)
            gt_uv_bone = labels['target_uv_bone']
            gt_uv_weight_bone = labels['target_uv_weight_bone']
            bone_loss = 0.5 * self.uv_criterion(pred_pts_bone.mul(gt_uv_weight_bone), gt_uv_bone.mul(gt_uv_weight_bone))

            # 软组织坐标损失
            pred_pts_soft = output['pred_pts_soft'].reshape(labels['target_uv_soft'].shape)
            gt_uv_soft = labels['target_uv_soft']
            gt_uv_weight_soft = labels['target_uv_weight_soft']
            soft_loss = 0.5 * self.uv_criterion(pred_pts_soft.mul(gt_uv_weight_soft), gt_uv_soft.mul(gt_uv_weight_soft))

            # 总损失：骨组织和软组织损失的平均
            loss = 0.5 * (bone_loss + soft_loss)

        else:
            raise ValueError("Unsupported loss type, you should choose either 'heatmap' or 'coord'")
        
        return loss
    
@LOSS.register_module
class DualMSELoss_softmax_onlyBone(nn.Module):
    ''' MSE Loss for Dual U-Net (Bone and Soft Tissue)
    '''
    def __init__(self, **cfg):
        super(DualMSELoss_softmax_onlyBone, self).__init__()
        self.criterion = nn.MSELoss()
        self._type = cfg['PRESET']['METHOD_TYPE']  # 'heatmap' 或 'coord'
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        if self._type == 'heatmap':
            # 骨组织热图损失
            batchsize, bone_joints, hm_height, hm_width = output['heatmap_bone'].size()
            gt_hm_bone = labels['target_hm_bone']
            gt_hm_weight_bone = labels['target_hm_weight_bone']
            pred_hm_bone = torch.softmax(output['heatmap_bone'].view(batchsize, bone_joints, -1), dim=-1).view_as(output['heatmap_bone'])  # softmax

            if self._mask_type == 'const':
                bone_loss = get_loss_by_const_mask(pred_hm_bone, gt_hm_bone, self.hm_criterion, self._basenumber)
            elif self._mask_type == 'pow':
                bone_loss = get_loss_by_pow_mask(pred_hm_bone, gt_hm_bone, gt_hm_weight_bone, self.hm_criterion, self._basenumber)
            else:
                raise ValueError("Unsupported mask type, you should choose either 'const' or 'pow'")

        elif self._type == 'coord':
            # 骨组织坐标损失
            pred_pts_bone = output['pred_pts_bone'].reshape(labels['target_uv_bone'].shape)
            gt_uv_bone = labels['target_uv_bone']
            gt_uv_weight_bone = labels['target_uv_weight_bone']
            bone_loss = 0.5 * self.uv_criterion(pred_pts_bone.mul(gt_uv_weight_bone), gt_uv_bone.mul(gt_uv_weight_bone))


        else:
            raise ValueError("Unsupported loss type, you should choose either 'heatmap' or 'coord'")
        
        return bone_loss
    
@LOSS.register_module
class DualMSELoss_softmax_onlySoft(nn.Module):
    ''' MSE Loss for Dual U-Net (Bone and Soft Tissue)
    '''
    def __init__(self, **cfg):
        super(DualMSELoss_softmax_onlySoft, self).__init__()
        self.criterion = nn.MSELoss()
        self._type = cfg['PRESET']['METHOD_TYPE']  # 'heatmap' 或 'coord'
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        if self._type == 'heatmap':

            # 软组织热图损失
            batchsize, soft_joints, hm_height, hm_width = output['heatmap_soft'].size()
            gt_hm_soft = labels['target_hm_soft']
            gt_hm_weight_soft = labels['target_hm_weight_soft']
            pred_hm_soft = torch.softmax(output['heatmap_soft'].view(batchsize, soft_joints, -1), dim=-1).view_as(output['heatmap_soft'])  # softmax

            if self._mask_type == 'const':
                soft_loss = get_loss_by_const_mask(pred_hm_soft, gt_hm_soft, self.hm_criterion, self._basenumber)
            elif self._mask_type == 'pow':
                soft_loss = get_loss_by_pow_mask(pred_hm_soft, gt_hm_soft, gt_hm_weight_soft, self.hm_criterion, self._basenumber)
            else:
                raise ValueError("Unsupported mask type, you should choose either 'const' or 'pow'")


        elif self._type == 'coord':

            # 软组织坐标损失
            pred_pts_soft = output['pred_pts_soft'].reshape(labels['target_uv_soft'].shape)
            gt_uv_soft = labels['target_uv_soft']
            gt_uv_weight_soft = labels['target_uv_weight_soft']
            soft_loss = 0.5 * self.uv_criterion(pred_pts_soft.mul(gt_uv_weight_soft), gt_uv_soft.mul(gt_uv_weight_soft))

        else:
            raise ValueError("Unsupported loss type, you should choose either 'heatmap' or 'coord'")
        
        return soft_loss