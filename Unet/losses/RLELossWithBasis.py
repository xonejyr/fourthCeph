class RLELossWithBasis(nn.Module):
    def __init__(self, size_average=True, **cfg):
        super(RLELossWithBasis, self).__init__()
        self.residual = cfg['RESIDUAL']
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / ((math.sqrt(2) * sigma) + 1e-9)

    def forward(self, output, labels):
        pred_jts = output.pred_pts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        # 使用基分布生成的损失
        nf_loss = output.nf_loss  # 现在是 basis_loss
        if nf_loss is not None:
            nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        # 残差项
        if self.residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss =  nf_loss + Q_logprob if nf_loss is not None else Q_logprob
        else:
            loss = nf_loss if nf_loss is not None else torch.tensor(0.0, device=pred_jts.device)
#
        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / gt_uv_weight.sum()
        else:
            return loss.sum()