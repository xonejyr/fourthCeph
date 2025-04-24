import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F
from easydict import EasyDict as edict
import cv2
import os

import matplotlib.pyplot as plt
import inspect

#################################################################################80
## DNDN可视化
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_points(points, points_refined, title="Points Comparison"):
    points, points_refined = points.cpu().numpy(), points_refined.cpu().numpy()
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Original')
    plt.scatter(points_refined[:, 0], points_refined[:, 1], c='red', label='Refined')
    plt.legend()
    plt.title(title)
    plt.show()

def visualize_pattern_effect(model, points, images, k, title=f"Effect of z_k"):
    model.eval()
    with torch.no_grad():
        points_refined, mu, log_var, z = model(points.unsqueeze(0).cuda(), images.unsqueeze(0).cuda())
        z_perturbed = z.clone()
        z_perturbed[0, k] += 1.0  # 扰动 z_k
        h_recon = model.decoder(z_perturbed)
        points_perturbed = model.refiner(h_recon.view(1, -1)).view(1, model.num_points, 2)
        diff = (points_perturbed - points_refined).squeeze(0).cpu().numpy()
    plt.imshow(np.linalg.norm(diff, axis=-1).reshape(1, -1), cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.show()

def visualize_z_distribution(z, title="Z Distribution"):
    z = z.cpu().detach().numpy()
    z_embedded = TSNE(n_components=2).fit_transform(z)
    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=range(len(z)))
    plt.title(title)
    plt.show()


#--------------------------------------------------------------------------------80


#################################################################################80
# 注册与加载
# this is a module registry class, which is used to manage the modules adaptively
class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property # make the method to be read-only
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        # type check
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__

        # check if the module name is already registered in the registry,
        # if so, raise an error to avoid overwriting the existing module.
        if module_name in self._module_dict:
            #raise KeyError('{} is already registered in {}'.format(module_name, self.name))
            print(f'{module_name} is already registered in {self.name}, will not register again. The old one remained ...')
        else:
            self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls

# build the object from the config dict
def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from. has a name and is actally a dict with {classname: class, ...}
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    # check the input
    assert isinstance(cfg, dict) and 'TYPE' in cfg
    # each cfg has 'TYPE'
    assert isinstance(default_args, dict) or default_args is None
    
    args = cfg.copy()
    obj_type = args.pop('TYPE')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type) # @register_module, then can get by 'TYPE' key as the class name to get the class
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type # if it is a class, just pass it directly to obj_cls
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
        # using default_args, usually cfg['preset'] in a class to set the default value of the args (the remained args after pop 'TYPE'), else, doesn't change, use original in args
    return obj_cls(**args) # pass the default_args to the class find by 'TYPE' key in the registry object


def retrieve_from_cfg(cfg, registry):
    """Retrieve a module class from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        class: The class.
    """
    assert isinstance(cfg, dict) and 'TYPE' in cfg
    args = cfg.copy()
    obj_type = args.pop('TYPE')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))

    return obj_cls

#--------------------------------------------------------------------------------80


#################################################################################80
# 热图可视化
def visualize_heatmaps(opt, images, target_hm, pred_hm, img_ids, target_uv=None, pred_uv=None, type='heatmap', show_norm='minmaxNorm'):
    """
    可视化热图和原图，帮助诊断问题，对整个 batch 保存到指定路径。
    
    Args:
        opt: 训练选项，包含 exp_id 和 cfg
        images: 原图，[batchsize, 3, H, W]，RGB图像
        target_hm: 真实热图，[batchsize, num_joints, hm_height, hm_width]
        pred_hm: 预测热图（logits），[batchsize, num_joints, hm_height, hm_width]
        img_ids: 图像 ID 元组，例如 ('301.bmp', '302.bmp', ...)
        target_size: 原图尺寸 [rawImage_H, rawImage_W]
    """
    batch_size, num_joints, hm_height, hm_width = target_hm.shape
    img_height, img_width = images.shape[2:]  # 512, 512
    cfg_file_name = os.path.basename(opt.cfg).split('.')[0]
    base_path = f"./exp/{opt.exp_id}-{cfg_file_name}/hm_visualizations_{show_norm}/"

    # change to IMAGE_SIZE
    target_uv = scale_to_targetsize(src_coord=target_uv.cpu(), target_size=(img_height, img_width))
    pred_uv = scale_to_targetsize(src_coord=pred_uv.cpu(), target_size=(img_height, img_width))
    # change to IMAGE_SIZE

    # 确保基础目录存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 遍历 batch 中的每张图像
    for i in range(batch_size):
        img_id = img_ids[i]  # 例如 '301.bmp'
        img_dir = os.path.join(base_path, img_id)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # 转换为 CPU 和 NumPy
        img = images[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        img = img + 0.5 # [0, 1]
        np.clip(img, a_min=0., a_max=255.)

        # 遍历每个 joint
        for joint_idx in range(num_joints):
            target = target_hm[i, joint_idx].cpu().numpy()  # [hm_height, hm_width]
            pred_logits = pred_hm[i, joint_idx].cpu()  # [hm_height, hm_width]
            
            if show_norm =='minmaxNorm':
                # 对预测热图应用 minmaxNorm 
                heatmap_flat = pred_logits.view(-1)  # 形状: [hm_height * hm_width]
                min_vals = heatmap_flat.min()
                max_vals = heatmap_flat.max()
                pred_prob_flat = (heatmap_flat - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除以 0
                pred_prob = pred_prob_flat.view_as(pred_logits).numpy() 
            elif show_norm == 'softmax':
                # 对预测热图应用 softmax
                pred_prob = torch.softmax(pred_logits.view(-1), dim=0).view_as(pred_logits).numpy()
            else: 
                raise ValueError("Unsupported show_norm: {}".format(show_norm))

            if type == 'heatmap':
                # 从热图提取坐标
                def get_coords(heatmap, target_size, hm_size):
                    h, w = hm_size
                    orig_h, orig_w = target_size
                    flat = heatmap.reshape(-1)
                    idx = flat.argmax()
                    y = idx // w
                    x = idx % w
                    # 映射到原图尺寸
                    x_orig = (x / w) * orig_w
                    y_orig = (y / h) * orig_h
                    return x_orig, y_orig

                pred_x, pred_y = get_coords(pred_prob, (img_height, img_width), (hm_height, hm_width))


            elif type == 'coord':
                # 直接使用 target_uv 作为 ground truth 坐标

                pred_x, pred_y = pred_uv[i, joint_idx]
            else:
                raise ValueError("Unsupported heat map type: {}".format(type))

            
            target_x, target_y = target_uv[i, joint_idx]

            # 可视化
            plt.figure(figsize=(15, 5))

            # 子图 1：原图 + 标志点
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.scatter(target_x, target_y, c='green', label='Target', s=50)
            plt.scatter(pred_x, pred_y, c='red', label='Pred', s=50)
            plt.legend()
            plt.title(f"Image {img_id} - Landmark {joint_idx}")

            # 子图 2：真实热图
            plt.subplot(1, 3, 2)
            plt.imshow(target, cmap='hot', interpolation='nearest')
            plt.title("Ground Truth Heatmap")
            plt.colorbar()

            # 子图 3：预测热图（softmax）
            plt.subplot(1, 3, 3)
            plt.imshow(pred_prob, cmap='hot', interpolation='nearest')
            if show_norm =='minmaxNorm':
                plt.title("Predicted Heatmap (max-min Normalization)")
            elif show_norm =='softmax':
                plt.title("Predicted Heatmap (Softmax Normalization)")
            else:
                raise ValueError("Unsupported show_norm: {}".format(show_norm))
            
            plt.colorbar()

            # 下方标题
            plt.suptitle(f"Heatmap Visualization for Landmark {joint_idx} of Image {img_id} using {opt.exp_id}", fontsize=16, y=0.05)
            # 调整布局，为底部标题留出足够空间
            plt.tight_layout(pad=1.0)
            plt.subplots_adjust(bottom=0.15)  # 增加底部边距

            
            # 保存路径
            save_path = os.path.join(img_dir, f"hm_visual_Landmark-{img_id.split('.')[0]}_{joint_idx:02}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形，释放内存
#--------------------------------------------------------------------------------80

#################################################################################80
# 提取坐标
#   get_coord → 调用 heatmap_to_coord 或 heatmap_to_coord_medical
#   heatmap_to_coord_medical → 调用 get_max_pred → 调用 transform_preds
#   transform_preds → 调用 get_affine_transform 和 affine_transform

class get_coord(object):
    def __init__(self, cfg, hm_size):
        self.type = cfg.VALIDATE.get('HEATMAP2COORD')
        self.target_size = cfg.DATASET.PRESET.RAW_IMAGE_SIZE
        self.hm_size = hm_size # heatmap_size

    def __call__(self, output, idx):
        if self.type == 'coord':
            pred_jts = output.pred_pts[idx]
            return heatmap_to_coord(pred_jts, self.hm_size)
        elif self.type == 'heatmap':
            pred_hms = output.heatmap[idx]
            # print('need to correct')
            return heatmap_to_coord_medical(pred_hms)
        else:
            raise NotImplementedError
def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals # of heatmap size

def heatmap_to_coord(pred_jts, hm_size):
    """
    heatmap to coord, 
    of heatmap size, 
    float

    args:
        pred_jts (torch.Tensor): [N, num_joints, 2] of heatmap size
        
        hm_shape: [1,1] as [hm_h, hm_w]

    Return:
        preds: [N, num_joints, 2] of rawImage size
    """
    hm_h, hm_w = hm_size

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 2 or 3"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(float)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_w
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_h

    return coords

def heatmap_to_coord_medical(hms, **kwargs):
    """
    heatmap to coordinate with medical standard, 
    of heatmap size, 
    float

    args:
        hms (numpy.ndarray): heatmap, [num_joints, hm_h, hm_w]
        
    Return: of heatmap_size
        preds (numpy.ndarray): coordinate, [None, num_joints, 2]
        maxvals (numpy.ndarray): max value of each joint, [None, num_joints]
    """
    if not isinstance(hms, np.ndarray):
        hms = hms.cpu().data.numpy()
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing, 利用梯度信息对峰值坐标进行细化（如亚像素精度）。
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                             hm[py + 1][px] - hm[py - 1][px]))
            coords[p] += np.sign(diff) * .25

    return coords[None, :, :], maxvals[None, :, :] # None是为了添加新维度

def transform_preds(coords, center, scale, output_size):
    """
    output_size => (center, scale)
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords

#################################################################################80
# 仿射变换 
# 1.如何使用（顺变化）：origin => target
# center, scale = get_center_scale(w_origin, h_origin, target_ratio_w_h, scale_mult=1.25)
# trans = get_affine_transform(center, scale, 0, (w_target, target_h), inv=0)
# img_target = cv2.warpAffine(img_src, trans, (int(w_target), int(target_h)), flags=cv2.INTER_LINEAR)
# coords_target[0:2] = affine_transform(coords_origin[0:2], trans)
#
# 如何使用（逆变化）：origin => target
# center, scale = get_center_scale(w_origin, h_origin, target_ratio_w_h, scale_mult=1.25)
# trans = get_affine_transform(center, scale, 0, (w_target, target_h), inv=1)
# img_origin = cv2.warpAffine(img_target, trans, (int(w_origin), int(h_origin)), flags=cv2.INTER_LINEAR)
# coords_origin[0:2] = affine_transform(coords_target[0:2], trans)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    # 沿着逆时针方向
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return np.array(src_result)

# 检测区域生成辅助，用于transform_data.py
def get_center_scale(w, h, aspect_ratio=1.0, scale_mult=1.25):
    '''
    计算输入图像的中心点坐标和经过比例调整后的缩放尺度，
    用于生成以图像为中心、
    符合目标宽高比并带有扩展边界的检测区域。
    '''
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = w * 0.5
    center[1] = h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale
#--------------------------------------------------------------------------------80

#################################################################################80
# transform heatmap to coord by softmax_integral
class Softmax_Integral(nn.Module):
    def __init__(self, num_pts, hm_width, hm_height):
        super(Softmax_Integral, self).__init__()
        self.num_pts = num_pts
        self.hm_width = hm_width
        self.hm_height = hm_height

    def forward(self, pred_hms):
        pred_hms = pred_hms.reshape((pred_hms.shape[0], self.num_pts, -1))
        pred_hms = F.softmax(pred_hms, 2)

        x, y = generate_2d_integral_preds_tensor(pred_hms, self.num_pts, self.hm_width, self.hm_height)
        x = x / float(self.hm_width) - 0.5
        y = y / float(self.hm_height) - 0.5
        preds = torch.cat((x, y), dim=2)
        preds = preds.reshape((pred_hms.shape[0], self.num_pts * 2))
        return preds

def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)

    accu_x = accu_x * torch.arange(float(x_dim)).to(accu_x.device)
    accu_y = accu_y * torch.arange(float(y_dim)).to(accu_y.device)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    return accu_x, accu_y
#--------------------------------------------------------------------------------80

#################################################################################80
# 数据记录，用于在train过程中动态跟踪,添加了追踪字典类型
class DataLogger(object):
    """Average data logger."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0  # 初始为标量，适配原有逻辑
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        if isinstance(value, dict):
            # 如果是字典，动态转换为字典模式
            if not isinstance(self.sum, dict):
                self.sum = {}  # 初始化为空字典
            for key, val in value.items():
                self.sum[key] = self.sum.get(key, 0) + val * n
        else:
            # 如果是标量，保持原有逻辑
            if isinstance(self.sum, dict):
                raise ValueError("Cannot mix scalar and dict updates after initialization")
            self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        if isinstance(self.sum, dict):
            self.avg = {key: val / self.cnt for key, val in self.sum.items()}
        else:
            self.avg = self.sum / self.cnt if self.cnt > 0 else 0

class DataLogger_old(object):
    """Average data logger."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt
#--------------------------------------------------------------------------------80

#################################################################################80
# 加载config并转换为yaml，yaml可以用config.keys来访问
def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config
#--------------------------------------------------------------------------------80


#################################################################################80
# （可选）采用乘除法的形式，直接将结果映射到原图尺寸
import torch
def scale_srcsize_to_targetsize(src_coord, src_size, target_size):
    """ 
    Change src_coord from [-0.5, 0.5] to be in target size.

    Args:
        src_coord (torch.Tensor): Input coordinates, shape [batchsize, num_joints * 2]
        src_size (list or tuple): Target size [src_h, src_w]
        target_size (list or tuple): Target size [height, width]

    Returns:
        torch.Tensor: Scaled coordinates, shape [batchsize, num_joints * 2]
    """
    # 确保输入是张量
    if not isinstance(src_coord, torch.Tensor):
        src_coord = torch.tensor(src_coord, dtype=torch.float32)
    
    # 提取目标高度和宽度
    height, width = target_size
    src_h, src_w = src_size
    
    # 将坐标从 [-0.5, 0.5] 映射到 [0, 1]
    scaled_coord = src_coord.clone()  # 复制，避免修改原始数据
    scaled_coord = scaled_coord + 0.5  # 从 [-0.5, 0.5] 到 [0, 1]
    
    # 分离 x 和 y 坐标 (偶数索引是 x，奇数索引是 y)
    x_coords = scaled_coord[:, 0::2]  # 取第 0, 2, 4, ... 个元素 (x 坐标)
    y_coords = scaled_coord[:, 1::2]  # 取第 1, 3, 5, ... 个元素 (y 坐标)
    
    # 按目标尺寸缩放
    x_coords = x_coords * width / src_w  # 从 [0, 1] 到 [0, width]
    y_coords = y_coords * height / src_h # 从 [0, 1] 到 [0, height]
    
    # 重构输出坐标
    scaled_coord[:, 0::2] = x_coords  # 将 x 坐标放回偶数位置
    scaled_coord[:, 1::2] = y_coords  # 将 y 坐标放回奇数位置
    
    return scaled_coord

def scale_to_targetsize(src_coord, target_size):
    """ 
    Change src_coord from [-0.5, 0.5] to be in target size.

    Args:
        src_coord (torch.Tensor): Input coordinates, shape [batchsize, num_joints * 2]
        target_size (list or tuple): Target size [height, width]

    Returns:
        torch.Tensor: Scaled coordinates, shape [batchsize, num_joints * 2]
    """
    # 确保输入是张量
    if not isinstance(src_coord, torch.Tensor):
        src_coord = torch.tensor(src_coord, dtype=torch.float32)
    
    # 提取目标高度和宽度
    height, width = target_size
    
    # 将坐标从 [-0.5, 0.5] 映射到 [0, 1]
    scaled_coord = src_coord.clone()  # 复制，避免修改原始数据
    scaled_coord = scaled_coord + 0.5  # 从 [-0.5, 0.5] 到 [0, 1]
    
    # 分离 x 和 y 坐标 (偶数索引是 x，奇数索引是 y)
    x_coords = scaled_coord[:, 0::2]  # 取第 0, 2, 4, ... 个元素 (x 坐标)
    y_coords = scaled_coord[:, 1::2]  # 取第 1, 3, 5, ... 个元素 (y 坐标)
    
    # 按目标尺寸缩放
    x_coords = x_coords * width   # 从 [0, 1] 到 [0, width]
    y_coords = y_coords * height  # 从 [0, 1] 到 [0, height]
    
    # 重构输出坐标
    scaled_coord[:, 0::2] = x_coords  # 将 x 坐标放回偶数位置
    scaled_coord[:, 1::2] = y_coords  # 将 y 坐标放回奇数位置
    
    return scaled_coord
#--------------------------------------------------------------------------------80

#################################################################################80
# stimulation of file writer for blank
class NullWriter(object):
    def write(self, arg):
        pass

    def flush(self):
        pass
#--------------------------------------------------------------------------------80