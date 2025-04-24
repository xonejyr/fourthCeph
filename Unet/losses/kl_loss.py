import torch.nn as nn
import torch

from Unet.builder import LOSS

from .utils import get_loss_by_const_mask, get_loss_by_pow_mask

@LOSS.register_module
class KLLoss(nn.Module):
    ''' KL Divergence Loss for Heatmap Regression
    '''
    def __init__(self, **cfg):
        super(KLLoss, self).__init__()
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self.hm_criterion = nn.KLDivLoss(reduction='none')  # 使用 KLDivLoss，reduction='none' 以便逐元素计算
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        if self._type == 'heatmap':
            batchsize, num_joints, hm_height, hm_width = output['heatmap'].size()
            gt_hm = labels['target_hm']
            gt_hm_weight = labels['target_hm_weight']

            # Softmax 处理预测热图，确保是概率分布
            pred_hm = torch.softmax(output['heatmap'].view(batchsize, num_joints, -1), dim=-1).view_as(output['heatmap'])

            # 对 gt_hm 进行归一化，使其和为 1（可选，取决于你的 gt_hm 是否已归一化）
            gt_hm_normalized = gt_hm / (gt_hm.sum(dim=[2, 3], keepdim=True) + 1e-8)  # 每个热图归一化

            # 避免 log(0) 的问题
            pred_hm = pred_hm.clamp(min=1e-8)  # 防止 log(0)
            gt_hm_normalized = gt_hm_normalized.clamp(min=1e-8)

            # 计算 KL 散度（pred_hm 需要 log）
            loss = self.hm_criterion(pred_hm.log(), gt_hm_normalized)

            # 通过调用函数加权
            if self._mask_type == 'const':
                loss = get_loss_by_const_mask(pred_hm, gt_hm_normalized, self.hm_criterion, self._basenumber)
            elif self._mask_type == 'pow':
                loss = get_loss_by_pow_mask(pred_hm, gt_hm_normalized, gt_hm_weight, self.hm_criterion, self._basenumber)
            else:
                raise ValueError("Unsupported mask type, you should choose either 'const' or 'pow'")

        elif self._type == 'coord':
            pred_pts = output['pred_pts'].reshape(labels['target_uv'].shape)
            gt_uv = labels['target_uv']
            gt_uv_weight = labels['target_uv_weight']
            loss = 0.5 * self.uv_criterion(pred_pts.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))
        else:
            raise ValueError("Unsupported loss type, you should choose either 'heatmap' or 'coord'")

        return loss