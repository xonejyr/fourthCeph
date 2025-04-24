import math
import torch
import torch.nn as nn
from Unet.builder import LOSS

@LOSS.register_module
class RLELossWithBasis_withOrth(nn.Module):
    def __init__(self, size_average=True, residual=True, use_basis=True, use_div=False, use_reg=False, 
                 alpha=0.1, beta=0.01, **cfg):
        super(RLELossWithBasis_withOrth, self).__init__()
        self.size_average = size_average
        self.residual = cfg['RESIDUAL']  # 是否启用 Q_logprob
        self.use_basis = cfg['USE_BASIS']  # 是否启用 basis_loss
        self.use_div = cfg['USE_DIV']  # 是否启用 div_loss
        self.use_reg = cfg['USE_REG']  # 是否启用 reg_loss
        self.alpha = cfg['ALPHA']  # div_loss 权重
        self.beta = cfg['BETA']  # reg_loss 权重
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels):
        pred_jts = output.pred_pts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        total_loss = torch.tensor(0.0, device=pred_jts.device)
        loss_dict = {}  # 用于记录各部分损失

        if output.losses is not None:
            # 基分布损失
            if self.use_basis and output.losses.basis_loss is not None:
                loss_dict['basis_loss'] = output.losses.basis_loss.squeeze(-1).sum(-1).mean()

                #print(f"the size of biasis loss is {output.losses.basis_loss.shape}")
                total_loss += loss_dict['basis_loss']

            # 多样性正则化
            if self.use_div and output.losses.div_loss is not None:
                loss_dict['div_loss'] = self.alpha * output.losses.div_loss
                #print(f"the size of div_loss is {loss_dict['div_loss'].shape}")
                total_loss += loss_dict['div_loss'][0]

            # 参数正则化
            if self.use_reg and output.losses.reg_loss is not None:
                loss_dict['reg_loss'] = self.beta * output.losses.reg_loss
                total_loss += loss_dict['reg_loss']

        # 残差项
        if self.residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss_dict['Q_logprob'] = Q_logprob.mean()
            #print(f"the size of Q_logprob is {Q_logprob.shape}")
            total_loss += loss_dict['Q_logprob']

        # 归一化
        if self.size_average and gt_uv_weight.sum() > 0:
            for key in loss_dict:
                loss_dict[key] = loss_dict[key].sum() / gt_uv_weight.sum()
            total_loss = total_loss.sum() / gt_uv_weight.sum()
        else:
            for key in loss_dict:
                loss_dict[key] = loss_dict[key].sum()
            total_loss = total_loss.sum()

        return total_loss #, loss_dict  # 返回总损失和各部分损失