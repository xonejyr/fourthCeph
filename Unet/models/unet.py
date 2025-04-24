""" Full assembly of the parts to form the complete network """

from .components.unet_parts import *
from easydict import EasyDict
from Unet.utils import Softmax_Integral

from Unet.builder import MODEL

@MODEL.register_module
class UNet(nn.Module):
    def __init__(self, bilinear=False, **cfg):
        # in cepha dataset, self.n_channels = 3, n_classes = 19
        super(UNet, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']

        self._preset_cfg   = cfg['PRESET']
        self.n_classes     = self._preset_cfg['NUM_JOINTS']
        self.num_joints    = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim  = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        

        self.bilinear = bilinear

        self.inc = (DoubleConv(self.n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, self.n_classes))

        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)
        
    def _initialize(self):
        pass

    def forward(self, x, target_uv=None):
        BATCH_SIZE = x.shape[0]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)# x5 up, 然后与x4拼接
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        out_coord = self.integral_hm(logits)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        output = EasyDict(
            pred_pts=pred_pts, # [batchsize, num_joints, 2]
            heatmap=logits # [batchsize, num_joints, hm_height, hm_width]
        )
        return output

    def use_checkpointing(self): #内存优化（面向深度网络），只保留必要的计算图（保留权重、梯度，放弃中间特征），降低内存占用
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)