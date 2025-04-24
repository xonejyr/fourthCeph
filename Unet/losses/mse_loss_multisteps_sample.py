import torch.nn as nn
import torch

from Unet.builder import LOSS
from .utils import get_loss_by_const_mask, get_loss_by_pow_mask


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
        self.single_loss = MSELoss_softmax(**cfg)  # 假设已有单步 MSE 损失函数
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        
        # 每步权重，强调最终结果，同时提升初始质量
        self.step_weights = cfg.get('STEP_WEIGHTS', [0.4, 0.3, 0.3])  # init_hm 高权重，后续递增
        self.init_weight = cfg.get('INIT_WEIGHT', 0.5)  # 额外权重给 init_hm
        self.consistency_weight = cfg.get('CONSISTENCY_WEIGHT', 0.1)  # 一致性约束权重
        self.monotonic_weight = cfg.get('MONOTONIC_WEIGHT', 0.2)  # 单调改进约束权重
        self.entropy_weight = cfg.get('ENTROPY_WEIGHT', 0.1)  # init_hm 熵正则权重

    def forward(self, output, labels):
        """
        计算多步损失，包含以下目标：
        - 最终热图接近真实热图：高权重监督 final_hm
        - 每步精化改进：单调性约束 + 递增权重
        - init_hm 质量高：额外权重 + 熵正则
        
        Args:
            output (EasyDict): 包含 heatmap, pred_pts, all_heatmaps, all_coords
            labels (dict): 包含 target_hm, target_uv, target_uv_weight
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
        # 调整权重长度，确保与步骤数匹配
        step_weights = self.step_weights + [1.0] * (num_steps - len(self.step_weights))
        step_weights = step_weights[:num_steps]
        # 归一化权重总和为 1（可选）
        step_weights = [w / sum(step_weights) for w in step_weights]

        total_loss = 0.0
        consistency_loss = 0.0
        monotonic_loss = 0.0
        entropy_loss = 0.0
        step_losses = []  # 记录每步的单独损失

        if self._type == 'heatmap':
            gt_hm = labels['target_hm']
            
            # 计算每步的基础 MSE 损失
            for step_idx, pred_hm in enumerate(all_heatmaps):
                step_output = {'heatmap': pred_hm}
                step_loss = self.single_loss(step_output, labels)
                weight = step_weights[step_idx]
                if step_idx == 0:  # init_hm 额外权重
                    weight += self.init_weight
                elif step_idx == num_steps - 1:  # final_hm 强调最终结果
                    weight *= 1.5  # 可调整倍数
                total_loss += weight * step_loss
                step_losses.append(step_loss)

                # 一致性损失：相邻步骤热图差异
                if step_idx > 0:
                    diff = (pred_hm - all_heatmaps[step_idx - 1]).pow(2).mean()
                    consistency_loss += diff
                
                # init_hm 熵正则：鼓励集中分布
                if step_idx == 0:
                    prob = torch.softmax(pred_hm, dim=1)  # 假设 pred_hm 是 [B, num_joints, H, W]
                    entropy = - (prob * torch.log(prob + 1e-10)).mean()
                    entropy_loss += entropy

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
                weight = step_weights[step_idx]
                if step_idx == 0:  # init_coords 额外权重
                    weight += self.init_weight
                elif step_idx == num_steps - 1:  # final_pts 强调最终结果
                    weight *= 1.5
                total_loss += weight * step_loss
                step_losses.append(step_loss)

                # 一致性损失：相邻步骤坐标差异
                if step_idx > 0:
                    diff = (pred_pts - all_coords[step_idx - 1]).pow(2).mean()
                    consistency_loss += diff

            # 单调改进损失
            for step_idx in range(1, num_steps):
                if step_losses[step_idx] > step_losses[step_idx - 1]:
                    monotonic_loss += (step_losses[step_idx] - step_losses[step_idx - 1]).clamp(min=0)

        else:
            raise ValueError("Unsupported loss type, choose 'heatmap' or 'coord'")

        # 总损失 = 基础损失 + 一致性损失 + 单调改进损失 + init_hm 熵正则
        total_loss = (total_loss +
                      self.consistency_weight * consistency_loss +
                      self.monotonic_weight * monotonic_loss +
                      self.entropy_weight * entropy_loss)

        return total_loss