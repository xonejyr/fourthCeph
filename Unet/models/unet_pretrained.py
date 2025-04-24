import torch
import torch.nn as nn
from easydict import EasyDict
import segmentation_models_pytorch as smp
from Unet.utils import Softmax_Integral

from Unet.builder import MODEL



@MODEL.register_module
class UNetPretrained(nn.Module):
    def __init__(self, bilinear=False, **cfg):
        super(UNetPretrained, self).__init__()
        self.n_channels = cfg['IN_CHANNELS']

        self._preset_cfg   = cfg['PRESET']
        self.n_classes     = self._preset_cfg['NUM_JOINTS']
        self.num_joints    = self._preset_cfg['NUM_JOINTS']
        self.hm_width_dim  = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=self.n_channels,
            classes=self.n_classes
        )

        # 加载本地权重，设置 weights_only=False
        pretrained_path = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/Unet/models/pretrained_weights/resnet34-333f7ec4.pth"
        resnet34_state_dict = torch.load(pretrained_path, weights_only=False)

        # 调整键名
        encoder_state_dict = {f"encoder.{k}": v for k, v in resnet34_state_dict.items()}
        self.model.encoder.load_state_dict(encoder_state_dict, strict=False)

        self.integral_hm = Softmax_Integral(num_pts=self.num_joints,
                                            hm_width=self.hm_width_dim,
                                            hm_height=self.hm_height_dim)
    
    def _initialize(self):
        pass

    def forward(self, x, target_uv=None):
        BATCH_SIZE = x.shape[0]
        output = self.model(x)  # [batchsize, 19, H, W], the input and output have the same size, better be those times 32

        out_coord = self.integral_hm(output)
        pred_pts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)

        output = EasyDict(
            pred_pts=pred_pts, # [batchsize, num_joints, 2]
            heatmap=output # [batchsize, num_joints, hm_height, hm_width]
        )
        return output

if __name__ == '__main__':
    pass
    ## 初始化
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = Farnet().to(device)
    #
    ## 测试输入
    #x = torch.randn(1, 3, 800, 640).to(device)
    #hp, refine_hp = model(x)
    #print("hp shape:", hp.shape)  # [1, 19, 800, 640]