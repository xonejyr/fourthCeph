
import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from .transforms_WebCeph2k import fliplr_joints, crop, generate_target, transform_pixel

from ..builder import DATASET
from .. import builder

@DATASET.register_module
class Dataset_WebCeph2k(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):

        if is_train == 'train':
            self.csv_file = cfg.DATASET.TRAINSET
        elif is_train == 'test':
            self.csv_file = cfg.DATASET.TESTSET
        else:
            self.csv_file = cfg.DATASET.VALIDSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0].split('/')[-1]) # xxx.jpg
        
        ################################################
        # 这都是啥？
        scale = self.landmarks_frame.iloc[idx, 1]
        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])
        pixels = self.landmarks_frame.iloc[idx, 4] # 0.34


        pts = self.landmarks_frame.iloc[idx, 5:].values # idx为5及以后
        pts = pts.astype('float').reshape(-1, 3) # [num_joints, 3]
        vis = pts[:,0] # 0 or 1
        pts = pts[:,1:] # [num_joints, 2], the real pts
        

        scale *= 1.25 # scale ~0.3
        nparts = pts.shape[0] # num_joints
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32) # [H, W, 3]

        r = 0
        if self.is_train == 'train':
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='WebCeph2k')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1])) # [num_joints, H, W]
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                if tpts[i, 0] < 0 or tpts[i, 1] > self.input_size[0] / 4:
                    vis[i] = 0
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std # 归一化
        img = img.transpose([2, 0, 1]) # [C, rawImage_H, rawImage_W]
        target = torch.Tensor(target) # [num_joints, H, W]
        tpts = torch.Tensor(tpts) # [num_joints, 2]
        tpts4 = torch.Tensor(tpts * 4 / self.input_size[0]) # 放大4倍，缩小input_size,
        vis = torch.Tensor(vis)

        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'tpts4': tpts4, 
                'image_path':image_path, 'vis': vis, 'pixels':pixels}

        return img, target, meta


if __name__ == '__main__':

    pass
