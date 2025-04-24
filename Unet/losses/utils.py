import torch

def get_loss_by_const_mask(pred_hm, gt_hm, criterion, basenumber):
    weights = gt_hm > 0.5  # 前景掩码
    loss = criterion(pred_hm, gt_hm)
    loss = (loss * (weights.float() * basenumber + 1)).mean() # 前景为10， 背景为1
    return loss

def get_loss_by_pow_mask(pred_hm, gt_hm, gt_hm_weight, criterion, basenumber):
    loss = 0.5 * criterion(pred_hm.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))
    # 加权训练，让1被放大，让0被缩小，40^x
    ratio = torch.pow(basenumber, gt_hm)
    loss = torch.mul(loss, ratio)
    loss = torch.mean(loss)
    return loss