import torch.nn as nn
import torch

from Unet.builder import LOSS
from .utils import get_loss_by_const_mask, get_loss_by_pow_mask

@LOSS.register_module
class MSELoss(nn.Module):
    ''' MSE Loss

    '''
    def __init__(self, **cfg):
        super(MSELoss, self).__init__()
        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()
        
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']


    def forward(self, output, labels):
        if self._type == 'heatmap':
            batchsize, num_joints, hm_height, hm_width = output['heatmap'].size()
            gt_hm = labels['target_hm']
            #pred_hm = torch.softmax(output['heatmap'].view(-1), dim=0).view_as(output['heatmap'])
            ####### softmax of the output heatmap
            #pred_hm = torch.softmax(output['heatmap'].view(batchsize, num_joints, -1), dim=-1).view_as(output['heatmap']) # softmax
            
            ##### 对每个 [hm_height, hm_width] 展平后进行 Min-Max 归一化
            heatmap_flat = output['heatmap'].view(batchsize, num_joints, -1)  # 形状: [batchsize, num_joints, hm_height * hm_width]
            min_vals = heatmap_flat.min(dim=-1, keepdim=True)[0]   # 每个热图的最小值
            max_vals = heatmap_flat.max(dim=-1, keepdim=True)[0]   # 每个热图的最大值
            pred_hm_flat = (heatmap_flat - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除以 0
            pred_hm = pred_hm_flat.view_as(output['heatmap'])  # 恢复形状
            
            gt_hm_weight = labels['target_hm_weight']
            loss = 0.5 * self.hm_criterion(pred_hm.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))
            # 加权训练，让1被放大，让0被缩小，40^x
            ratio = torch.pow(self._basenumber, gt_hm)
            loss = torch.mul(loss, ratio)
            loss = torch.mean(loss)

            #weights = gt_hm > 0.5  # 前景掩码
            #loss = nn.MSELoss(reduction='none')(pred_hm, gt_hm)
            #loss = (loss * (weights.float() * 10 + 1)).mean() # 前景为10， 背景为1
        elif self._type == 'coord':
            pred_pts = output['pred_pts'].reshape(labels['target_uv'].shape)
            gt_uv = labels['target_uv']
            gt_uv_weight = labels['target_uv_weight']
            loss = 0.5 * self.uv_criterion(pred_pts.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))
        else:
            raise ValueError("Unsupported loss type, you should choose either 'heatmap' or 'coord'")
        
        return loss

@LOSS.register_module
class MSELoss_minmaxNorm(nn.Module):
    ''' MSE Loss

    '''
    def __init__(self, **cfg):
        super(MSELoss_minmaxNorm, self).__init__()      
        self._type = cfg['PRESET']['METHOD_TYPE']
        self._basenumber = cfg['BASENUMBER']
        self._mask_type = cfg['MASK_TYPE']

        self.hm_criterion = nn.MSELoss(reduction='none')
        self.uv_criterion = nn.MSELoss()

    def forward(self, output, labels):
        if self._type == 'heatmap':
            #print(f"the size of the output['heatmap'] is {output['heatmap'].shape}")
            batchsize, num_joints, hm_height, hm_width = output['heatmap'].size()
            gt_hm = labels['target_hm']
            gt_hm_weight = labels['target_hm_weight']

            ##### 对每个 [hm_height, hm_width] 展平后进行 Min-Max 归一化
            heatmap_flat = output['heatmap'].view(batchsize, num_joints, -1)  # 形状: [batchsize, num_joints, hm_height * hm_width]
            min_vals = heatmap_flat.min(dim=-1, keepdim=True)[0]   # 每个热图的最小值
            max_vals = heatmap_flat.max(dim=-1, keepdim=True)[0]   # 每个热图的最大值
            pred_hm_flat = (heatmap_flat - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除以 0
            pred_hm = pred_hm_flat.view_as(output['heatmap'])  # 恢复形状

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

            ####### softmax of the output heatmap
            pred_hm = torch.softmax(output['heatmap'].view(batchsize, num_joints, -1), dim=-1).view_as(output['heatmap']) # softmax

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
class MSELoss_doublesoftmax(nn.Module):
    ''' MSE Loss

    '''
    def __init__(self, **cfg):
        super(MSELoss_doublesoftmax, self).__init__()
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
            gt_hm = torch.softmax(gt_hm.view(batchsize, num_joints, -1), dim=-1).view_as(gt_hm) # softmax
            gt_hm_weight = labels['target_hm_weight']

            ####### softmax of the output heatmap
            pred_hm = torch.softmax(output['heatmap'].view(batchsize, num_joints, -1), dim=-1).view_as(output['heatmap']) # softmax
            

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