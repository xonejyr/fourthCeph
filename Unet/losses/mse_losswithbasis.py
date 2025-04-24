import math
import torch
import torch.nn as nn

from Unet.builder import LOSS

from .utils import get_loss_by_const_mask, get_loss_by_pow_mask


#@LOSS.register_module
class MSELoss_softmax_withBasis(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, size_average=True, **cfg):
        super(MSELoss_softmax_withBasis, self).__init__()
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

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()


@LOSS.register_module
class MSELoss_softmax_withBasis(nn.Module):
    ''' MSE Loss

    '''
    def __init__(self, **cfg):
        super(MSELoss_softmax_withBasis, self).__init__()
        self.criterion = nn.MSELoss()
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']
        self.residual = cfg['RESIDUAL']
        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        #print(f"size of nf_loss is {nf_loss.shape}")
        if self._type == 'heatmap':
            batchsize, num_joints, hm_height, hm_width = output['heatmap'].size()
            gt_hm = labels['target_hm']
            gt_hm_weight = labels['target_hm_weight']

            #print(f"size of nf_loss is {nf_loss.shape}")
            #print(f"size of labels['target_hm_weight'] is {labels['target_hm_weight'].shape}")
            nf_loss = (nf_loss.squeeze(-1) * gt_hm_weight[:, :, :1].squeeze(-1).squeeze(-1)).sum(-1).mean()
            

            ####### softmax of the output heatmap
            pred_hm = torch.softmax(output['heatmap'].view(batchsize, num_joints, -1), dim=-1).view_as(output['heatmap']) # softmax

            if self._mask_type == 'const':
                loss = get_loss_by_const_mask(pred_hm, gt_hm, self.hm_criterion, self._basenumber)
            elif self._mask_type == 'pow':
                loss = get_loss_by_pow_mask(pred_hm, gt_hm, gt_hm_weight, self.hm_criterion, self._basenumber)
            else:
                raise ValueError("Unsupported mask type, you should choose either 'const' or 'pow'")

            loss = loss / (hm_height * hm_height)
            

        elif self._type == 'coord':
            pred_pts = output['pred_pts'].reshape(labels['target_uv'].shape)
            gt_uv = labels['target_uv']
            gt_uv_weight = labels['target_uv_weight']
            print(f"the size of nf_loss is {nf_loss.shape}")
            print(f"size of labels['target_uv_weight'] is {labels['target_uv_weight'].shape}")
            nf_loss = nf_loss.squeeze(-1).sum(-1).mean()
            loss = 0.5 * self.uv_criterion(pred_pts.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))
        else:
            raise ValueError("Unsupported loss type, you should choose either 'heatmap' or 'coord'")
        
        if self.residual:
            loss = nf_loss + loss if nf_loss is not None else loss
        else:
            loss = nf_loss if nf_loss is not None else torch.tensor(0.0, device=output.device)
        
        return loss