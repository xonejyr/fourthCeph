import random
import cv2
import numpy as np
import torch
from ..utils import get_center_scale, split_targets, split_heatmaps
from .transforms import affine_transform, get_affine_transform, im_to_torch, Compose, \
    ConvertImgFloat, PhotometricDistort, RandomSampleCrop


class Preprocessing(object):
    def __init__(self):
        self.data_aug = Compose([ConvertImgFloat(),
                                 RandomSampleCrop(min_win=0.9),
                                 PhotometricDistort()])

    def __call__(self, img, pts):
        img_out, pts = self.data_aug(img.copy(), pts.copy())
        return img_out, pts



class Transform(object):
    """Generation of cropped input and heatmaps from SimplePose.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor,
                 input_size, output_size, rot, sigma,
                 train, loss_type='heatmap', shift=(0, 0), bone_indices=None, soft_indices=None):
        self._scale_factor = scale_factor
        self._rot = rot
        self.shift = shift
        self._input_size = input_size  # preset input size, not the size of the exact image
        self._heatmap_size = output_size
        # self.shift =
        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)
        self.imgwidth = input_size[1]
        self.imght = input_size[0]
        self.pixel_std = 1
        self.align_coord = True
        self.process = Preprocessing()
        self.bone_indices = bone_indices
        self.soft_indices = soft_indices

    def test_transform(self, src):

        center, scale = get_center_scale(
            self.imgwidth, self.imght, self._aspect_ratio, scale_mult=1.25)

        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        img = im_to_torch(img)
        img = img.add_(-0.5)

        return img

    def _target_generator(self, joints_ed, num_joints):
        """
        build the heatmap of HEATMAP_SIZE from the joints_ed of IMAGE_SIZE
        sigma is of HEATMAP_SIZE

        args:
            joints_ed: the joints corresponding to the IMAGE_SIZE
            num_joints: number of joints
        outputs:
            target torch.Size([num_joints, hm_h, hm_w])
            joints_hmSize torch.Size([num_joints, 2]) the joints of HEATMAP_SIZE
        """ 
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_ed[:, 0, 1]
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]),
                          dtype=np.float32)
        tmp_size = self._sigma * 3

        # build the joints of heatmapsize
        joints_hmSize = np.zeros((num_joints, 2), dtype=np.float32)

        for i in range(num_joints):
            mu_x = int(joints_ed[i, 0, 0] / self._feat_stride[0] + 0.5) #+0.5, 四舍五入, IMAGE_SIZE to HEATMAP_SIZE
            mu_y = int(joints_ed[i, 1, 0] / self._feat_stride[1] + 0.5)

            joints_hmSize[i, 0] = mu_x
            joints_hmSize[i, 1] = mu_y

            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1), joints_hmSize

    def _integral_target_generator(self, joints_ed, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_ed[:, 0, 1]
        target_weight[:, 1] = joints_ed[:, 0, 1]

        target_visible = np.ones((num_joints, 1), dtype=np.float32)
        target_visible[:, 0] = target_weight[:, 0]

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_ed[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_ed[:, 1, 0] / patch_height - 0.5

        target_visible[target[:, 0] > 0.5] = 0
        target_visible[target[:, 0] < -0.5] = 0
        target_visible[target[:, 1] > 0.5] = 0
        target_visible[target[:, 1] < -0.5] = 0

        target_visible_weight = target_weight[:, :1].copy()

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight, target_visible, target_visible_weight

    def crop_joint_regions(self, src, joints):
        """Crop regions around joints from the image."""
        cropped_1 = np.zeros((self.num_joints, 64, 64, src.shape[2]), dtype=np.float32)
        cropped_2 = np.zeros((self.num_joints, 32, 32, src.shape[2]), dtype=np.float32)
        cropped_3 = np.zeros((self.num_joints, 16, 16, src.shape[2]), dtype=np.float32)

        for i in range(self.num_joints):
            x, y = joints[i, 0:2, 0]
            x, y = int(x), int(y)
            for size, half_size, cropped in [(64, 32, cropped_1), 
                                             (32, 16, cropped_2), 
                                             (16, 8, cropped_3)]:
                y_start = max(0, y - half_size)
                y_end = min(src.shape[0], y + half_size)
                x_start = max(0, x - half_size)
                x_end = min(src.shape[1], x + half_size)
                crop = src[y_start:y_end, x_start:x_end, :]
                pad_top = half_size - (y - y_start) if y - half_size < 0 else 0
                pad_bottom = half_size - (y_end - y) if y + half_size > src.shape[0] else 0
                pad_left = half_size - (x - x_start) if x - half_size < 0 else 0
                pad_right = half_size - (x_end - x) if x + half_size > src.shape[1] else 0
                crop_padded = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                crop_padded = crop_padded[:size, :size, :]
                cropped[i] = crop_padded

        return cropped_1, cropped_2, cropped_3

    def __call__(self, src, label):
        gt_joints = label['joints']
        if self._train:
            src, gt_joints = self.process(src, gt_joints)

        src = np.clip(src, a_min=0., a_max=255.)

        label['width'], label['height'] = src.shape[1], src.shape[0] # rawImage_size after composed
        imgwidth, imght = label['width'], label['height']
        assert imgwidth == src.shape[1] and imght == src.shape[0]
        self.num_joints = gt_joints.shape[0]

        joints_vis = np.zeros((self.num_joints, 1), dtype=np.float32)
        joints_vis[:, 0] = gt_joints[:, 0, 1]

        joints = gt_joints
        cropped_1, cropped_2, cropped_3 = self.crop_joint_regions(src, joints)

        input_size = self._input_size

        center, scale = get_center_scale(imgwidth, imght, self._aspect_ratio, ) # rawImage_size
        # rescale
        if self._train:
            sf = self._scale_factor
            scale = scale * random.uniform(0.75, 1 + sf)
        else:
            scale = scale * 1.0

        # rotation
        if self._train:
            rf = self._rot
            r = random.uniform(-rf, rf) if random.random() <= 0.5 else 0
        else:
            r = 0
        # shift
        if self._train:
            sft_x = random.uniform(-self.shift[0], self.shift[0]) if random.random() <= 0.5 else 0
            sft_y = random.uniform(-self.shift[1], self.shift[1]) if random.random() <= 0.5 else 0
            sft = np.array([sft_x, sft_y], dtype=np.float32)
        else:
            sft = np.array([0, 0], dtype=np.float32)

        

        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h], shift=sft)# IMAGE_SIZE
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR) # to IMAGE_SIZE
        # print(img.shape)
        # deal with landmark visibility this part contains problem
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        # generate training targets
        target_hm, target_hm_weight, joints_hmSize = self._target_generator(joints.copy(), self.num_joints)
        target_uv, target_uv_weight, target_visible, target_visible_weight = self._integral_target_generator(
            joints.copy(), self.num_joints, inp_h, inp_w)

        # img = np.clip(img, a_min=0., a_max=255.)
        img = im_to_torch(img)
        img.add_(-0.5)

        # split bone and soft related
        target_uv_bone, target_uv_soft, target_uv_weight_bone, target_uv_weight_soft = split_targets(target_uv, target_uv_weight, self.bone_indices, self.soft_indices)
        target_hm_bone, target_hm_soft, target_hm_weight_bone, target_hm_weight_soft = split_heatmaps(target_hm, target_hm_weight, self.bone_indices, self.soft_indices)


        output = {
            'type': '2d_data',
            'image': img,            
            'cropped_1': torch.from_numpy(cropped_1).float(),  # [num_joints, 64, 64, channels]
            'cropped_2': torch.from_numpy(cropped_2).float(),  # [num_joints, 32, 32, channels]
            'cropped_3': torch.from_numpy(cropped_3).float(),  # [num_joints, 16, 16, channels],


            'target_hm': torch.from_numpy(target_hm).float(),
            'target_hm_weight': torch.from_numpy(target_hm_weight).float(),

            'target_hm_bone': torch.from_numpy(target_hm_bone).float(),
            'target_hm_weight_bone': torch.from_numpy(target_hm_weight_bone).float(),
            'target_hm_soft': torch.from_numpy(target_hm_soft).float(),
            'target_hm_weight_soft': torch.from_numpy(target_hm_weight_soft).float(),



            'target_uv': torch.from_numpy(target_uv).float(),
            'target_uv_weight': torch.from_numpy(target_uv_weight).float(),
            'target_uv_of_hmSize': torch.from_numpy(joints_hmSize).float(), # [num_joints, 2]

            'target_uv_bone': torch.from_numpy(target_uv_bone).float(),
            'target_uv_soft': torch.from_numpy(target_uv_soft).float(),
            'target_uv_weight_bone': torch.from_numpy(target_uv_weight_bone).float(),
            'target_uv_weight_soft': torch.from_numpy(target_uv_weight_soft).float()
        }
        return output

'''
type: This is a string key that simply indicates the type of the data. Its value is '2d_data'.

image: This is the transformed image tensor. Its dimensions are [3, H, W], where H and W are the height and width of the transformed image (the dimensions of the input image after scaling, rotation, and shifting).

target_hm: This is the target heatmap for the joints. Its dimensions are [num_joints, output_height, output_width], where num_joints is the number of joints in the dataset, and output_height and output_width are the dimensions of the heatmap.

target_hm_weight: This is the weight associated with the heatmap target. Its dimensions are [num_joints, 1]. It typically contains a value of 1 for visible joints and 0 for invisible joints.

target_uv: This is the target for the 2D coordinates of the joints. Its dimensions are [num_joints, 2], where num_joints is the number of joints and each joint has 2 values corresponding to its x and y coordinates.

target_uv_weight: This is the weight associated with the 2D joint coordinates. Its dimensions are [num_joints, 1]. It contains the weight for each joint, indicating if the joint is visible or not (similar to target_hm_weight).

'''