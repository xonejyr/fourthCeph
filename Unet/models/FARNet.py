from collections import OrderedDict
from easydict import EasyDict

import torch
import torch.nn as nn
import torchvision.models as models

from .components.FARNet_parts import wblock

from Unet.builder import MODEL

@MODEL.register_module
class FARNet(nn.Module):
    def __init__(self, **cfg):
        # with input dimenson and output dimension as 
        super(FARNet, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']

        self._preset_cfg   = cfg['PRESET']
        self.n_classes     = self._preset_cfg['NUM_JOINTS']
        self.num_joints    = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim  = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]



        #self.features = models.densenet121(pretrained=True).features
        model = models.densenet121(weights=None)  # 创建空模型
        self.features = model.features  # 只取 features 部分
        state_dict = torch.load('/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/Unet/models/pretrained_weights/densenet121-a639ec97.pth')
        self.features.load_state_dict(state_dict, strict=False)  # 只加载匹配的键

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear')

        self.wblock1 = wblock(1, 512, 512, 256)
        self.wblock2 = wblock(1, 1024, 1024, 256)
        self.wblock3 = wblock(3, 2048, 1024, 256)

        self.w1_conv11_0 = nn.Sequential(OrderedDict([
            ('conv11_0', nn.Conv2d(self.n_channels, 32, kernel_size=1)),
            ('norm11_0', nn.BatchNorm2d(32)),
            ('relu11_0', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv33_01 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(512)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(256)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))

        self.w1_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_2', nn.Conv2d(1280, 256, kernel_size=1)),
            ('norm11_2', nn.BatchNorm2d(256)),
            ('relu11_2', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_3', nn.Conv2d(768, 256, kernel_size=1)),
            ('norm11_3', nn.BatchNorm2d(256)),
            ('relu11_3', nn.ReLU(inplace=True)),
        ]))

        self.mid_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(256)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))

        self.w2_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(1024, 256, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(256)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_2', nn.Conv2d(1280, 256, kernel_size=1)),
            ('norm11_2', nn.BatchNorm2d(256)),
            ('relu11_2', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_3', nn.Conv2d(768, 256, kernel_size=1)),
            ('norm11_3', nn.BatchNorm2d(256)),
            ('relu11_3', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_4 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512, 128, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(128)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.w2_conv11_5 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(192, 64, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(64)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride1 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(512)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride2 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(1024)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv33_stride3 = nn.Sequential(OrderedDict([
            ('conv33_0', nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)),
            ('norm33_0', nn.BatchNorm2d(2048)),
            ('relu33_0', nn.ReLU(inplace=True)),
        ]))
        self.conv_33_refine1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_33_refine2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv_11_refine = nn.Conv2d(64, 19, kernel_size=1)
        #self.conv_11_refine = nn.Conv2d(64, 45, kernel_size=1)
        
        self.conv_33_last1 = nn.Conv2d(115, 115, kernel_size=3, stride=1, padding=1)
        #self.conv_33_last1 = nn.Conv2d(141, 141, kernel_size=3, stride=1, padding=1)
        
        self.conv_33_last2 = nn.Conv2d(115, 115, kernel_size=3, stride=1, padding=1)
        
        self.conv_11_last = nn.Conv2d(115, self.num_joints, kernel_size=1)
        #self.conv_11_last = nn.Conv2d(141, 45, kernel_size=1)
        
        
    def forward(self, x):
        #print(x.size())
        w1_f0 = self.w1_conv11_0(x)
        x = self.features[0](x)
        w1_f1 = x
        for i in range(1, 5):
            x = self.features[i](x)
        w1_f2 = x
        for i in range(5, 7):
            x = self.features[i](x)
        w1_f3 = x
        for i in range(7, 9):
            x = self.features[i](x)
        w1_f4 = x
        for i in range(9, 12):
            x = self.features[i](x)
        # first upsample and concat
        x = self.w1_conv33_01(x)
        x = self.w1_conv11_1(x)
        w2_f5 = x
        x = self.upsample2(x)
        x = torch.cat((x, w1_f4), 1)
        # second upsample and concat
        x = self.w1_conv11_2(x)
        w2_f4 = x
        x = self.upsample2(x)
        x = torch.cat((x, w1_f3), 1)
        # third upsample and concat
        x = self.w1_conv11_3(x)
        w2_f3 = x
        x = self.upsample2(x)
        x = torch.cat((x, w1_f2), 1)

        x = self.mid_conv11_1(x)
        w3_f2 = x

        x = self.conv33_stride1(x)
        x = self.wblock1(x, w2_f3)
        w3_f3 = x
        x = self.conv33_stride2(x)
        x = self.wblock2(x, w2_f4)
        w3_f4 = x
        x = self.conv33_stride3(x)
        x = self.wblock3(x, w2_f5)
        x = self.w2_conv11_1(x)
        x = self.upsample2(x)
        x = torch.cat((x, w3_f4), 1)
        x = self.w2_conv11_2(x)
        x = self.upsample2(x)
        x = torch.cat((x, w3_f3), 1)
        x = self.w2_conv11_3(x)
        x = self.upsample2(x)
        x = torch.cat((x, w3_f2), 1)
        x = self.w2_conv11_4(x)
        x = self.upsample2(x)
        x = torch.cat((x, w1_f1), 1)
        x = self.w2_conv11_5(x)

        refine_hp = self.conv_33_refine1(x)
        refine_hp = self.conv_11_refine(refine_hp) #[1, 45, 400, 320]
        x = self.upsample2(x) #[1, 64, 800, 640]
        refine1_up = self.upsample2(refine_hp) #[1, 45, 800, 640]
        x = torch.cat((x, w1_f0, refine1_up), 1) #[1, 141, 800, 640]
        
        # output
        hp = self.conv_33_last1(x)
        hp = self.conv_11_last(hp)

        output = EasyDict(
            pred_pts=None, # [batchsize, num_joints, 2]
            heatmap=hp # [batchsize, num_joints, hm_height, hm_width]
        )
        #return hp, refine_hp  #(B,C,H,W) (1, 19, 800, 640)

        return output # for consistency, the refine_hp would not be returned
    # 现在已经调整为19！！！
    # 如果直接使用代码则输出为 
    # hp  [batchsize, 45, IMAGE_H, IMAGE_W]
    # refine_hp [batchsize, 45, IMAGE_H/2, IMAGE_W/2]
    # 如果使用注释则代码输出为
    # hp  [batchsize, 19, IMAGE_H, IMAGE_W]
    # refine_hp [batchsize, 19, IMAGE_H/2, IMAGE_W/2]

