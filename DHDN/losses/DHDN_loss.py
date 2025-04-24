import torch
import torch.nn as nn
import torch.nn.functional as F

from DHDN.builder import LOSS

@LOSS.register_module
class DHDNLoss_Bigger(nn.Module):
    def __init__(self, **cfg):
        super(DHDNLoss_Bigger, self).__init__()
        self.beta = cfg.get('BETA', 0.1)
        self.gamma = cfg.get('GAMMA', 0.5)
        self.lambda_ = cfg.get('LAMBDA', 0.0)

    def forward(self, output, labels):
        pred_pts = output['pred_pts'].reshape(labels['target_uv'].shape)  # [batch, num_joints * 2]
        gt_uv = labels['target_uv']  # [batch, num_joints * 2]
        gt_uv_weight = labels['target_uv_weight']  # [batch, num_joints * 2]

        mu = output['mu']  # [batch, num_patterns, pattern_dim]
        log_var = output['log_var']  # [batch, num_patterns, pattern_dim]

        batch_size = gt_uv.size(0)
        num_patterns = mu.size(1)  # 模式数量
        pattern_dim = mu.size(2)   # 每个模式的维度

        # L_recon: Reconstruction Loss
        recon_loss = F.mse_loss(pred_pts.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))

        # L_KL: KL Divergence Loss
        # 对所有维度（batch, num_patterns, pattern_dim）求和，并根据总元素数归一化
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * num_patterns * pattern_dim)

        # L_disentangle: Disentanglement Loss
        # 将 mu 展平为 [batch, num_patterns * pattern_dim] 以计算协方差
        mu_flat = mu.reshape(batch_size, -1)  # [batch, num_patterns * pattern_dim]
        cov = torch.cov(mu_flat.T)  # [num_patterns * pattern_dim, num_patterns * pattern_dim]
        off_diag = cov - torch.diag(cov.diagonal())
        disentangle_loss = off_diag.abs().sum() / off_diag.numel()

        # L_dynamics: Placeholder (保持不变)
        # dynamics_loss = torch.tensor(0.0, device=gt_uv.device)

        # Total Loss
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * disentangle_loss

        return total_loss
    
@LOSS.register_module
class DHDNLoss(nn.Module):
    def __init__(self, **cfg):
        super(DHDNLoss, self).__init__()
        self.beta = cfg.get('BETA', 0.1)    # 默认超参数 beta
        self.gamma = cfg.get('GAMMA', 0.5)  # 默认超参数 gamma
        self.lambda_ = cfg.get('LAMBDA', 0.0)  # 默认超参数 lambda_

    def forward(self, output, labels):
        pred_pts = output['pred_pts'].reshape(labels['target_uv'].shape) # [batch_size, num_points * 2]
        gt_uv = labels['target_uv'] # [batch_size, num_points * 2]
        gt_uv_weight = labels['target_uv_weight'] # [batch_size, num_points * 2]

        log_var = output['log_var'] # [batch_size, num_patterns]， log(sigma^2)
        mu = output['mu'] # [batch_size, num_patterns]


        batch_size = gt_uv.size(0)
        num_patterns= mu.size(1)

        # L_recon: Reconstruction Loss, defualt reduction='mean'
        recon_loss = F.mse_loss(pred_pts.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))
        # L_recon = (1/N) * Σ ||P'_i - P_i||^2

        # L_KL: KL Divergence Loss, 
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / ( num_patterns * batch_size)
        # L_KL = (-1/2K) * Σ (1 + log(σ_k^2) - μ_k^2 - σ_k^2)

        # L_disentangle: Disentanglement Loss
        cov = torch.cov(mu.T)  # [num_patterns, num_patterns]， 协方差矩阵。
        off_diag = cov - torch.diag(cov.diagonal()) # 提取非对角元素，方法是从协方差矩阵中减去对角线元素。
        disentangle_loss = off_diag.abs().sum() / off_diag.numel() # L_disentangle = {k!=m} |Cov(z_k, z_m)| / 非对角元素个数
        # L_disentangle = Σ_{k≠m} |Cov(z_k, z_m)|

        # L_dynamics: Placeholder (set to 0 for now)
        # dynamics_loss = torch.tensor(0.0, device=gt_uv.device)
        # L_dynamics = (1/T) * Σ ||H^(t+1) - H^(t)||^2 (待实现)

        # Total Loss


        total_loss = recon_loss + self.beta * kl_loss + self.gamma * disentangle_loss 

        #total_loss = recon_loss + beta * kl_loss + gamma * disentangle_loss + lambda_ * dynamics_loss

        return total_loss
        #{
        #    'recon': recon_loss.item(),
        #    'kl': kl_loss.item(),
        #    'disentangle': disentangle_loss.item()
        #    'dynamics': dynamics_loss.item()
        #}


@LOSS.register_module
class DHDNLoss_Dynamic(nn.Module):
    def __init__(self, **cfg):
        super(DHDNLoss_Dynamic, self).__init__()
        self.beta = cfg.get('BETA', 0.1)    # 默认超参数 beta
        self.gamma = cfg.get('GAMMA', 0.5)  # 默认超参数 gamma
        self.lambda_ = cfg.get('LAMBDA', 0.0)  # 默认超参数 lambda_

    def forward(self, output, labels):
        pred_pts = output['pred_pts'].reshape(labels['target_uv'].shape) # [batch_size, num_points * 2]
        gt_uv = labels['target_uv'] # [batch_size, num_points * 2]
        gt_uv_weight = labels['target_uv_weight'] # [batch_size, num_points * 2]

        log_var = output['log_var'] # [batch_size, num_patterns]， log(sigma^2)
        mu = output['mu'] # [batch_size, num_patterns]


        batch_size = gt_uv.size(0)
        num_patterns= mu.size(1)

        # L_recon: Reconstruction Loss, defualt reduction='mean'
        recon_loss = F.mse_loss(pred_pts.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))
        # L_recon = (1/N) * Σ ||P'_i - P_i||^2

        # L_KL: KL Divergence Loss, 
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / ( num_patterns * batch_size)
        # L_KL = (-1/2K) * Σ (1 + log(σ_k^2) - μ_k^2 - σ_k^2)

        # L_disentangle: Disentanglement Loss
        cov = torch.cov(mu.T)  # [num_patterns, num_patterns]， 协方差矩阵。
        off_diag = cov - torch.diag(cov.diagonal()) # 提取非对角元素，方法是从协方差矩阵中减去对角线元素。
        disentangle_loss = off_diag.abs().sum() / off_diag.numel() # L_disentangle = {k!=m} |Cov(z_k, z_m)| / 非对角元素个数
        # L_disentangle = Σ_{k≠m} |Cov(z_k, z_m)|

        # L_dynamics: Placeholder (set to 0 for now)
        dynamics_loss = 0.0
        num_layers = len(output['h_layers']) - 1
        for t in range(num_layers):
            diff = (output['h_layers'][t + 1] - output['h_layers'][t]).pow(2).mean()
            dynamics_loss += diff
        dynamics_loss /= num_layers
        # dynamics_loss = torch.tensor(0.0, device=gt_uv.device)
        # L_dynamics = (1/T) * Σ ||H^(t+1) - H^(t)||^2 (待实现)

        # Total Loss


        total_loss = recon_loss + self.beta * kl_loss + self.gamma * disentangle_loss + self.lambda_ * dynamics_loss

        #total_loss = recon_loss + beta * kl_loss + gamma * disentangle_loss + lambda_ * dynamics_loss

        return total_loss
        #{
        #    'recon': recon_loss.item(),
        #    'kl': kl_loss.item(),
        #    'disentangle': disentangle_loss.item()
        #    'dynamics': dynamics_loss.item()
        #}


def chamfer_distance(pred, gt):
    dist = torch.cdist(pred, gt)  # [batch, 19, 19]
    loss1 = dist.min(dim=2)[0].mean()  # pred 到 gt 的最近距离
    loss2 = dist.min(dim=1)[0].mean()  # gt 到 pred 的最近距离
    return loss1 + loss2

@LOSS.register_module
class DHDNLoss_Dynamic_1_1(nn.Module):
    def __init__(self, **cfg):
        super(DHDNLoss_Dynamic, self).__init__()
        self.beta = cfg.get('BETA', 0.1)    # 默认超参数 beta
        self.gamma = cfg.get('GAMMA', 0.5)  # 默认超参数 gamma
        self.lambda_ = cfg.get('LAMBDA', 0.0)  # 默认超参数 lambda_

    def forward(self, output, labels):
        pred_pts = output['pred_pts'].reshape(labels['target_uv'].shape) # [batch_size, num_points * 2]
        gt_uv = labels['target_uv'] # [batch_size, num_points * 2]
        gt_uv_weight = labels['target_uv_weight'] # [batch_size, num_points * 2]

        log_var = output['log_var'] # [batch_size, num_patterns]， log(sigma^2)
        mu = output['mu'] # [batch_size, num_patterns]


        batch_size = gt_uv.size(0)
        num_patterns= mu.size(1)

        # L_recon: Reconstruction Loss, defualt reduction='mean'
        recon_loss = F.mse_loss(pred_pts.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))
        # L_recon = (1/N) * Σ ||P'_i - P_i||^2

        # L_KL: KL Divergence Loss, 
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / ( num_patterns * batch_size)
        # L_KL = (-1/2K) * Σ (1 + log(σ_k^2) - μ_k^2 - σ_k^2)

        # L_disentangle: Disentanglement Loss
        cov = torch.cov(mu.T)  # [num_patterns, num_patterns]， 协方差矩阵。
        off_diag = cov - torch.diag(cov.diagonal()) # 提取非对角元素，方法是从协方差矩阵中减去对角线元素。
        disentangle_loss = off_diag.abs().sum() / off_diag.numel() # L_disentangle = {k!=m} |Cov(z_k, z_m)| / 非对角元素个数
        # L_disentangle = Σ_{k≠m} |Cov(z_k, z_m)|

        # L_dynamics: Placeholder (set to 0 for now)
        dynamics_loss = 0.0
        num_layers = len(output['h_layers']) - 1
        for t in range(num_layers):
            diff = (output['h_layers'][t + 1] - output['h_layers'][t]).pow(2).mean()
            dynamics_loss += diff
        dynamics_loss /= num_layers
        # dynamics_loss = torch.tensor(0.0, device=gt_uv.device)
        # L_dynamics = (1/T) * Σ ||H^(t+1) - H^(t)||^2 (待实现)

        # Total Loss
        total_loss = recon_loss + self.beta * kl_loss + self.gamma * disentangle_loss + self.lambda_ * dynamics_loss

        chamfer_loss = chamfer_distance(pred_pts, gt_uv.view(batch_size, self.num_joints, 2))

        total_loss += 0.5 * chamfer_loss  # 加权系数可调

        #total_loss = recon_loss + beta * kl_loss + gamma * disentangle_loss + lambda_ * dynamics_loss

        return total_loss
        #{
        #    'recon': recon_loss.item(),
        #    'kl': kl_loss.item(),
        #    'disentangle': disentangle_loss.item()
        #    'dynamics': dynamics_loss.item()
        #}