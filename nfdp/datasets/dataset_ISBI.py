# for path
import os
# data loading in PyTorch
from matplotlib import pyplot as plt
import torch.utils.data as data     
# YAML parser, covert to dict
import yaml                 
from easydict import EasyDict as edict  
# image processing, opencv
import cv2
import numpy as np
import torch

# custom dataset and dataloader
from .transform_data import Transform
import torch
from ..utils import scale_to_targetsize

from ..builder import DATASET
from .. import builder

import seaborn as sns

# process poins to (x, y, z) format, where z is 1 for all points to refer visible.
# usually for cephalometrics dataset, z for all landmarks are 1
def pts_process(pts):
    joints_3d = np.zeros((len(pts), 2, 2), dtype=np.float32)
    for i in range(len(pts)):
        joints_3d[i, 0, 0] = pts[i][0]
        joints_3d[i, 1, 0] = pts[i][1]
        joints_3d[i, :2, 1] = 1
    return joints_3d


@DATASET.register_module
class Dataset_ISBI(data.Dataset):
    CLASSES = ['ISBI 2015']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18]
    num_joints = 19
    joints_name = [
        'L0', 'L1', 'L2', 'L3',
        'L4', 'L5', 'L6', 'L7',
        'L8', 'L9', 'L10', 'L11',
        'L12', 'L13', 'L14', 'L15',
        'L16', 'L17', 'L18'
    ]

    def __init__(self,
                 **cfg):
        # parser the config file
        # path setting, including root, img_prefix, ann_file for dataset
        # augmentation parameters
        # image id load, the name of the image
        self._preset_cfg = cfg['PRESET']
        self.subset = cfg['subset'] # pass by builder default_args = {'subset':'train'}

        self._root = self._preset_cfg['ROOT']
        self.bone_indices = self._preset_cfg['BONE_INDICES']
        self.soft_indices = self._preset_cfg['SOFT_INDICES']
        
        if self.subset == 'train':
            self._img_prefix = self._preset_cfg['TRAIN']['IMG_PREFIX']
        elif self.subset == 'val':
            self._img_prefix = self._preset_cfg['VAL']['IMG_PREFIX']
        elif self.subset == 'test':
            self._img_prefix = self._preset_cfg['TEST']['IMG_PREFIX']
        else:
            raise ValueError(f'Invalid subset: {self.subset}')


        self._ann_file = os.path.join(self._root, self._preset_cfg['ANN'][0])
        self._ann_file2 = os.path.join(self._root, self._preset_cfg['ANN'][1])
        self.img_dir = os.path.join(self._root, self._img_prefix)

        if 'AUG' in cfg.keys() and self.subset=='train':
            self._scale_factor = self._preset_cfg['TRAIN']['AUG']['SCALE_FACTOR']
            self._rot = self._preset_cfg['TRAIN']['AUG']['ROT_FACTOR']
            self._shift = self._preset_cfg['TRAIN']['AUG']['SHIFT_FACTOR']
        else:
            self._scale_factor = 0
            self._rot = 0
            self._shift = (0, 0)

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']
        self._sigma = self._preset_cfg['SIGMA']

        self._train = self.subset == 'train'

        self.transformation = Transform(
            self, scale_factor=self._scale_factor,
            input_size=self._input_size, # IMAGE_SIZE
            output_size=self._output_size, # HEATMAP_SIZE
            rot=self._rot, sigma=self._sigma,
            train=self._train, shift=self._shift, bone_indices=self.bone_indices, soft_indices=self.soft_indices)

        self.img_ids = sorted(os.listdir(self.img_dir))


    # load the image of index
    def load_image(self, index):
        image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        # usually in shape of [height, width, channels], here the height, width are the original size
        return image
    
    def load_image_PseudoMultimodal(self, index):
        # 读取灰度图
        path = os.path.join(self.img_dir, self.img_ids[index])
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H, W)

        # 边缘信息 (常用值: 50, 150)
        edges = cv2.Canny(gray, 50, 150)  # (H, W)

        # 局部对比度增强 (常用值: clipLimit=2.0, tileGridSize=(8, 8))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)  # (H, W)

        # 组合成 3 通道
        image = np.stack([gray, edges, contrast], axis=2)  # (H, W, 3)
        return image

    # load the annotation file of image_id, actually two annotation files
    def load_annoFolder(self, img_id):
        return os.path.join(self._ann_file, img_id[:-4] + '.txt'), os.path.join(self._ann_file2, img_id[:-4] + '.txt')

    # load the two annotation files of image_id, return the average of the two as annotations, and convert to 3D format
    def load_annotation(self, index):

        img_id = self.img_ids[index]
        annoFolder1, annoFolder2 = self.load_annoFolder(img_id)
        pts1 = []
        pts2 = []
        with open(annoFolder1, 'r') as f:
            lines = f.readlines()
            for i in range(self.num_joints):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts1.append(coordinates_int)
        with open(annoFolder2, 'r') as f:
            lines = f.readlines()
            for i in range(self.num_joints):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts2.append(coordinates_int)
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        pts = (pts1 + pts2) / 2
        pts_ed = pts_process(pts)
        return pts_ed # 3D结构，带有valid标签 # [num_joints, 2, 2]

    def __getitem__(self, index):

        '''
        Parameters
        ----------
        index

        Returns augmented image, augmented joints, image id
        -------

        '''

        img_id = self.img_ids[index]
        img = self.load_image(index) # [rawImage_height, rawImage_width, 3]
        joints = self.load_annotation(index) # [num_joints, 2, 2]

        label = dict(joints=joints) # key 为 'joints', value is the 3D structure, 没有归一化
        label['width'] = img.shape[1] # 新增项
        label['height'] = img.shape[0]  # 新增项
        # target['label']

        # this returns a dictionary containing augmented image and labels
        target = self.transformation(img, label) 


        img = target.pop('image') # so, there would be no 'image' in the target anymore
        # print(img.shape)
        return img, target, img_id
        # img [C, H, W] here the H, W are resized
        # target a dict {type, target_hm, target_hm_weights, target_uv, target_uv_weights}
        '''
        img [3,IMAGE_SIZE_H, IMAGE_SIZE_W]
        type 2d_data

        #===========================================================
        target_hm torch.Size([num_joints, hm_h, hm_w])
        target_hm_weight torch.Size([num_joints, 1, 1])

        target_hm_bone torch.Size([num_joints_bones, hm_h, hm_w]) # num_joints_bones = 13
        target_hm_bone_weight torch.Size([num_joints_bones, 1, 1])
        target_hm_soft torch.Size([num_joints_soft, hm_h, hm_w]) # num_joints_soft = 6
        target_hm_soft_weight torch.Size([num_joints_soft, 1, 1])

        #===========================================================
        target_uv torch.Size([num_joints * 2])
        target_uv_weight torch.Size([num_joints * 2]) # num_joints = 19

        target_uv_bone torch.Size([num_joints_bones * 2]) # num_joints_bones = 13
        target_uv_bone_weight torch.Size([num_joints_bones * 2])
        target_uv_soft torch.Size([num_joints_soft * 2]) # num_joints_soft = 6
        target_uv_soft_weight torch.Size([num_joints_soft * 2])
        '''
        # img_id "xxx.bmp"

    def __len__(self):
        return len(self.img_ids)
    
@DATASET.register_module
class Dataset_ISBI_PseudoMultimodal(data.Dataset):
    CLASSES = ['ISBI 2015']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18]
    num_joints = 19
    joints_name = [
        'L0', 'L1', 'L2', 'L3',
        'L4', 'L5', 'L6', 'L7',
        'L8', 'L9', 'L10', 'L11',
        'L12', 'L13', 'L14', 'L15',
        'L16', 'L17', 'L18'
    ]

    def __init__(self,
                 **cfg):
        # parser the config file
        # path setting, including root, img_prefix, ann_file for dataset
        # augmentation parameters
        # image id load, the name of the image
        self._preset_cfg = cfg['PRESET']
        self.subset = cfg['subset'] # pass by builder default_args = {'subset':'train'}

        self._root = self._preset_cfg['ROOT']
        self.bone_indices = self._preset_cfg['BONE_INDICES']
        self.soft_indices = self._preset_cfg['SOFT_INDICES']
        
        if self.subset == 'train':
            self._img_prefix = self._preset_cfg['TRAIN']['IMG_PREFIX']
        elif self.subset == 'val':
            self._img_prefix = self._preset_cfg['VAL']['IMG_PREFIX']
        elif self.subset == 'test':
            self._img_prefix = self._preset_cfg['TEST']['IMG_PREFIX']
        else:
            raise ValueError(f'Invalid subset: {self.subset}')


        self._ann_file = os.path.join(self._root, self._preset_cfg['ANN'][0])
        self._ann_file2 = os.path.join(self._root, self._preset_cfg['ANN'][1])
        self.img_dir = os.path.join(self._root, self._img_prefix)

        if 'AUG' in cfg.keys() and self.subset=='train':
            self._scale_factor = self._preset_cfg['TRAIN']['AUG']['SCALE_FACTOR']
            self._rot = self._preset_cfg['TRAIN']['AUG']['ROT_FACTOR']
            self._shift = self._preset_cfg['TRAIN']['AUG']['SHIFT_FACTOR']
        else:
            self._scale_factor = 0
            self._rot = 0
            self._shift = (0, 0)

        self._input_size = self._preset_cfg['IMAGE_SIZE']
        self._output_size = self._preset_cfg['HEATMAP_SIZE']
        self._sigma = self._preset_cfg['SIGMA']

        self._train = self.subset == 'train'

        self.transformation = Transform(
            self, scale_factor=self._scale_factor,
            input_size=self._input_size, # IMAGE_SIZE
            output_size=self._output_size, # HEATMAP_SIZE
            rot=self._rot, sigma=self._sigma,
            train=self._train, shift=self._shift, bone_indices=self.bone_indices, soft_indices=self.soft_indices)

        self.img_ids = sorted(os.listdir(self.img_dir))


    # load the image of index
    def load_image(self, index):
        image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        # usually in shape of [height, width, channels], here the height, width are the original size
        return image
    
    def load_image_PseudoMultimodal(self, index):
        # 读取灰度图
        path = os.path.join(self.img_dir, self.img_ids[index])
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H, W)

        # 边缘信息 (常用值: 50, 150)
        edges = cv2.Canny(gray, 50, 150)  # (H, W)

        # 局部对比度增强 (常用值: clipLimit=2.0, tileGridSize=(8, 8))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)  # (H, W)

        # 组合成 3 通道
        image = np.stack([gray, edges, contrast], axis=2)  # (H, W, 3)
        return image

    # load the annotation file of image_id, actually two annotation files
    def load_annoFolder(self, img_id):
        return os.path.join(self._ann_file, img_id[:-4] + '.txt'), os.path.join(self._ann_file2, img_id[:-4] + '.txt')

    # load the two annotation files of image_id, return the average of the two as annotations, and convert to 3D format
    def load_annotation(self, index):

        img_id = self.img_ids[index]
        annoFolder1, annoFolder2 = self.load_annoFolder(img_id)
        pts1 = []
        pts2 = []
        with open(annoFolder1, 'r') as f:
            lines = f.readlines()
            for i in range(self.num_joints):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts1.append(coordinates_int)
        with open(annoFolder2, 'r') as f:
            lines = f.readlines()
            for i in range(self.num_joints):
                coordinates = lines[i].replace('\n', '').split(',')
                coordinates_int = [int(i) for i in coordinates]
                pts2.append(coordinates_int)
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        pts = (pts1 + pts2) / 2
        pts_ed = pts_process(pts)
        return pts_ed # 3D结构，带有valid标签 # [num_joints, 2, 2]

    def __getitem__(self, index):

        '''
        Parameters
        ----------
        index

        Returns augmented image, augmented joints, image id
        -------

        '''

        img_id = self.img_ids[index]
        #img = self.load_image(index) # [rawImage_height, rawImage_width, 3]
        img = self.load_image_PseudoMultimodal(index)
        joints = self.load_annotation(index) # [num_joints, 2, 2]

        label = dict(joints=joints) # key 为 'joints', value is the 3D structure, 没有归一化
        label['width'] = img.shape[1] # 新增项
        label['height'] = img.shape[0]  # 新增项
        # target['label']

        # this returns a dictionary containing augmented image and labels
        target = self.transformation(img, label) 


        img = target.pop('image') # so, there would be no 'image' in the target anymore
        # print(img.shape)
        return img, target, img_id
        # img [C, H, W] here the H, W are resized
        # target a dict {type, target_hm, target_hm_weights, target_uv, target_uv_weights}
        '''
        img [3,IMAGE_SIZE_H, IMAGE_SIZE_W]
        type 2d_data

        #===========================================================
        target_hm torch.Size([num_joints, hm_h, hm_w])
        target_hm_weight torch.Size([num_joints, 1, 1])

        target_hm_bone torch.Size([num_joints_bones, hm_h, hm_w]) # num_joints_bones = 13
        target_hm_bone_weight torch.Size([num_joints_bones, 1, 1])
        target_hm_soft torch.Size([num_joints_soft, hm_h, hm_w]) # num_joints_soft = 6
        target_hm_soft_weight torch.Size([num_joints_soft, 1, 1])

        #===========================================================
        target_uv torch.Size([num_joints * 2])
        target_uv_weight torch.Size([num_joints * 2]) # num_joints = 19

        target_uv_bone torch.Size([num_joints_bones * 2]) # num_joints_bones = 13
        target_uv_bone_weight torch.Size([num_joints_bones * 2])
        target_uv_soft torch.Size([num_joints_soft * 2]) # num_joints_soft = 6
        target_uv_soft_weight torch.Size([num_joints_soft * 2])
        '''
        # img_id "xxx.bmp"

    def __len__(self):
        return len(self.img_ids)


    
if __name__ == '__main__':
    # parser the config file
    with open('/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/configs/512x512_unet_ce_heatmap.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = edict(cfg) # to make a dict be get by .key other than ['key']

    # instantiate the dataset
    print("==========================================================")

    dataset = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset='train')

    rawImage = dataset.load_image(0)
    print(f"shape of the rawImage {rawImage.shape}")
    print(f"how many items are there in this dataset (dataloader): {len(dataset)}")
    print(f"the class of the dataset {dataset.CLASSES}")
    print("==========================================================")
    print(f"Here is an example of an data item after being transformed:")
    print(f"its name is \t\t {dataset.img_ids[0]}")
    img, label, img_id = dataset[0]
    print(f"its shape is \t\t {img.shape}")
    print(f"its label infomation including things about target_uv and target_hm")
    for key, value in label.items():
        if key == 'type':
            pass
        else:
            print(key, value.shape)
    

    def get_coords(heatmap, hm_size):
                h, w = hm_size
                flat = heatmap.reshape(-1)
                idx = flat.argmax()
                y = idx // w
                x = idx % w
                ## 映射到原图尺寸
                #x_orig = (x / w) * orig_w
                #y_orig = (y / h) * orig_h
                return x, y
    print("==========================================================")
    print("test for heatmap to uv")
    print(f"the shape of the target_hm is {label['target_hm'].shape}")
    print(f"the shape of the target_uv is {label['target_uv'].shape}")
    num_joints, hm_h, hm_w = label['target_hm'].shape
    returnedCoords = []
    for i in range(num_joints):
        returnedCoord = get_coords(label['target_hm'][i], (hm_h, hm_w))
        returnedCoords.append(returnedCoord)
    returnedCoords = [item for tuple in returnedCoords for item in tuple]
    returnedCoords = torch.stack(returnedCoords)
    print(f"The coordinates of the joints from hm by get_coords: \n{returnedCoords}")
    target_uv = scale_to_targetsize(label['target_uv'].unsqueeze(0), (hm_h,hm_w))
    print(f"the coordinates from target_uv are \n{target_uv}")
    print("==========================================================")
    for i in range(num_joints):
        print(f"the sum of the heatmap for Landmark {i} is: \t{torch.sum(label['target_hm'][i])}")
        print(f"the max of the heatmap for Landmark {i} is: \t{torch.max(label['target_hm'][i])}")

    
