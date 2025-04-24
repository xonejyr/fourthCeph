import torch.nn as nn
import torch

from Unet.builder import LOSS

from .utils import get_loss_by_const_mask, get_loss_by_pow_mask

@LOSS.register_module
class FocalLoss(nn.Module):
    ''' Focal Loss for Heatmap Regression
    '''
    def __init__(self, **cfg):
        super(FocalLoss, self).__init__()
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self._gamma = cfg.get('GAMMA', 2.0)  # Focal Loss 的 gamma 参数，默认 2.0
        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        if self._type == 'heatmap':
            batchsize, num_joints, hm_height, hm_width = output['heatmap'].size()
            gt_hm = labels['target_hm']
            gt_hm_weight = labels['target_hm_weight']

            # Softmax 处理预测热图
            pred_hm = torch.softmax(output['heatmap'].view(batchsize, num_joints, -1), dim=-1).view_as(output['heatmap'])

            # 计算基础 MSE Loss
            loss = self.hm_criterion(pred_hm, gt_hm)

            # Focal Loss 调制因子：(1 - pred_hm)^gamma
            modulating_factor = torch.pow(1 - pred_hm, self._gamma)
            loss = loss * modulating_factor

            # 通过调用函数加权
            if self._mask_type == 'const':
                loss = get_loss_by_const_mask(pred_hm, gt_hm, self.hm_criterion, self._basenumber)
            elif self._mask_type == 'pow':
                loss = get_loss_by_pow_mask(pred_hm, gt_hm, gt_hm_weight, self.hm_criterion, self._basenumber)
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