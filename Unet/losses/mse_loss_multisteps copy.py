import torch.nn as nn
import torch

from Unet.builder import LOSS
from .utils import get_loss_by_const_mask, get_loss_by_pow_mask


@LOSS.register_module
class MSELoss_softmax(nn.Module):
    ''' MSE Loss
    '''
    def __init__(self, **cfg):
        super(MSELoss_softmax, self).__init__()
        self.criterion = nn.MSELoss()
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        if self._type == 'heatmap':
            batchsize, num_joints, hm_height, hm_width = output['heatmap'].size()
            gt_hm = labels['target_hm']
            gt_hm_weight = labels['target_hm_weight']

            pred_hm = torch.softmax(output['heatmap'].view(batchsize, num_joints, -1), dim=-1).view_as(output['heatmap'])  # softmax

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


@LOSS.register_module
class MSELoss_softmax_multiStepSample(nn.Module):
    ''' MSE Loss for multi-step with constraints
    '''
    def __init__(self, **cfg):
        super(MSELoss_softmax_multiStepSample, self).__init__()
        self.single_loss = MSELoss_softmax(**cfg)
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self.step_weights = cfg.get('STEP_WEIGHTS', None)  # 每步权重
        self.consistency_weight = cfg.get('CONSISTENCY_WEIGHT', 0.1)  # 一致性约束权重
        self.monotonic_weight = cfg.get('MONOTONIC_WEIGHT', 0.1)  # 单调改进约束权重

    def forward(self, output, labels):
        """
        计算多步损失，并添加约束：
        - 基础损失：每步与真实标签的 MSE 损失
        - 一致性损失：相邻步骤之间的差异
        - 单调改进损失：惩罚后续步骤损失大于前一步的情况
        """
        final_hm = output['heatmap']
        final_pts = output['pred_pts']
        all_heatmaps = output.get('all_heatmaps', [final_hm])
        all_coords = output.get('all_coords', [final_pts])

        if not isinstance(all_heatmaps, list):
            all_heatmaps = [all_heatmaps]
        if not isinstance(all_coords, list):
            all_coords = [all_coords]

        num_steps = len(all_heatmaps)
        if self.step_weights is None:
            step_weights = [1.0 / num_steps] * num_steps
        else:
            step_weights = self.step_weights + [1.0] * (num_steps - len(self.step_weights))
            step_weights = step_weights[:num_steps]

        total_loss = 0.0
        consistency_loss = 0.0
        monotonic_loss = 0.0
        step_losses = []  # 记录每步的单独损失

        if self._type == 'heatmap':
            gt_hm = labels['target_hm']
            for step_idx, pred_hm in enumerate(all_heatmaps):
                step_output = {'heatmap': pred_hm}
                step_loss = self.single_loss(step_output, labels)
                total_loss += step_weights[step_idx] * step_loss
                step_losses.append(step_loss)

                # 一致性损失：相邻步骤热图的差异
                if step_idx > 0:
                    diff = (pred_hm - all_heatmaps[step_idx - 1]).pow(2).mean()
                    consistency_loss += diff

            # 单调改进损失：惩罚后续步骤损失大于前一步
            for step_idx in range(1, num_steps):
                if step_losses[step_idx] > step_losses[step_idx - 1]:
                    monotonic_loss += (step_losses[step_idx] - step_losses[step_idx - 1]).clamp(min=0)

        elif self._type == 'coord':
            gt_uv = labels['target_uv']
            gt_uv_weight = labels['target_uv_weight']
            for step_idx, pred_pts in enumerate(all_coords):
                step_output = {'pred_pts': pred_pts}
                step_loss = self.single_loss(step_output, labels)
                total_loss += step_weights[step_idx] * step_loss
                step_losses.append(step_loss)

                # 一致性损失：相邻步骤坐标的差异
                if step_idx > 0:
                    diff = (pred_pts - all_coords[step_idx - 1]).pow(2).mean()
                    consistency_loss += diff

            # 单调改进损失：惩罚后续步骤损失大于前一步
            for step_idx in range(1, num_steps):
                if step_losses[step_idx] > step_losses[step_idx - 1]:
                    monotonic_loss += (step_losses[step_idx] - step_losses[step_idx - 1]).clamp(min=0)

        else:
            raise ValueError("Unsupported loss type, you should choose either 'heatmap' or 'coord'")

        # 总损失 = 基础损失 + 一致性损失 + 单调改进损失
        total_loss = total_loss + self.consistency_weight * consistency_loss + self.monotonic_weight * monotonic_loss

        return total_loss