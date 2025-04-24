import math
import torch
import torch.nn as nn

from Unet.builder import LOSS


@LOSS.register_module
class RLELoss_noBase(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, size_average=True, **cfg):
        super(RLELoss_noBase, self).__init__()
        self.residual = cfg['RESIDUAL']
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def forward(self, output, labels):
        #nf_loss = output.nf_loss
        pred_jts = output.pred_pts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        #nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        if self.residual:
            loss = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            # = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()

@LOSS.register_module
class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''
    def __init__(self, size_average=True, **cfg):
        super(RLELoss, self).__init__()
        self.residual = cfg['RESIDUAL']
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_pts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        if self.residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            #loss =  Q_logprob + nf_loss
            loss = nf_loss + Q_logprob
        else:
            loss = nf_loss

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()

class RLELoss1(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, size_average=True, **cfg):
        super(RLELoss, self).__init__()
        self.residual = cfg['RESIDUAL']
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_pts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        if self.residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob
        else:
            loss = nf_loss

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()