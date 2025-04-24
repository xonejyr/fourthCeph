#import torch
#import torch.onnx
#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#
#
#from Unet.models.dual_unet import DualUNet
#from easydict import EasyDict
#
#
## 配置
#cfg = EasyDict({
#    'IN_CHANNELS': 3,
#    'PRESET': EasyDict({
#        'HEATMAP_SIZE': [256, 256],
#        'NUM_BONE_JOINTS': 15,
#        'NUM_SOFT_JOINTS': 4,
#        'BONE_INDICES': list(range(15)),
#        'SOFT_INDICES': list(range(15, 19))
#    })
#})
#
## 初始化模型
#model = DualUNet(bilinear=False, **cfg)
#model.eval()
#
## 输入张量
#x = torch.randn(1, cfg['IN_CHANNELS'], cfg['PRESET']['HEATMAP_SIZE'][0], cfg['PRESET']['HEATMAP_SIZE'][1])
#
## 固定路径
#onnx_path = "./Unet/models/visualizations/DualUNet_v1.onnx"
#os.makedirs(os.path.dirname(onnx_path), exist_ok=True)  # 确保目录存在
#
## 导出 ONNX
#torch.onnx.export(
#    model,
#    x,
#    onnx_path,
#    input_names=["input"],
#    output_names=["output"],
#    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#    opset_version=11
#)
#
#print(f"模型已导出到 {onnx_path}")

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 路径配置: model_visualization_dir

# the newst version doesn't need setting path directly
from shared_params_manage import ParamManager
param_manager = ParamManager()
paths = param_manager.get_paths()
model_visualization_dir = paths['model_visualization_dir']

os.makedirs(model_visualization_dir, exist_ok=True)
#--------------------------------------------------------------------------------80


import os
from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='DualUNet Flow', format='png')

# 设置全局样式（横向布局，大胆现代风格）
dot.attr(rankdir='LR', splines='spline', bgcolor='white', pad='0.7', nodesep='0.6', ranksep='1.0')
dot.node_attr.update(shape='box', style='filled,rounded', fontname='Helvetica', fontsize='16', margin='0.2', penwidth='1.0')
dot.edge_attr.update(color='#424242', arrowsize='0.8', penwidth='1.5', splines='spline')

#dot.attr(rankdir='LR', splines='ortho', bgcolor='white', pad='0.7', nodesep='0.6', ranksep='1.2')
#dot.node_attr.update(shape='box', style='filled,rounded', fontname='Helvetica', fontsize='16', margin='0.2', penwidth='0.5')
#dot.edge_attr.update(color='#555555', arrowsize='0.8', penwidth='1.2')

# 输入张量
dot.node('input', 'Input Tensor\n[1, 3, 256, 256]', fillcolor='#CFD8DC', fontcolor='#000000')

# 输入张量分成两份
dot.node('bone_tensor', 'bone_tensor\n[1, 15, 256, 256]', fillcolor='#BBDEFB', fontcolor='#000000')
dot.node('soft_tensor', 'soft_tensor\n[1, 4, 256, 256]', fillcolor='#C8E6C9', fontcolor='#000000')

# bone_unet 和 soft_unet
dot.node('bone_unet', 'bone_unet\n(Down -> Up)\n(15 joints)', fillcolor='#1E88E5', fontcolor='#FFFFFF')
dot.node('soft_unet', 'soft_unet\n(Down -> Up)\n(4 joints)', fillcolor='#2E7D32', fontcolor='#FFFFFF')

# 输出热图
dot.node('bone_hm', 'Bone Heatmap\n[1, 15, 256, 256]', fillcolor='#64B5F6', fontcolor='#000000')
dot.node('soft_hm', 'Soft Heatmap\n[1, 4, 256, 256]', fillcolor='#81C784', fontcolor='#000000')

# Softmax_Integral
dot.node('bone_softmax', 'Softmax_Integral', fillcolor='#64B5F6', fontcolor='#000000')
dot.node('soft_softmax', 'Softmax_Integral', fillcolor='#81C784', fontcolor='#000000')

# 坐标
dot.node('bone_coords', 'Bone Coords\n[1, 15, 2]', fillcolor='#BBDEFB', fontcolor='#000000')
dot.node('soft_coords', 'Soft Coords\n[1, 4, 2]', fillcolor='#C8E6C9', fontcolor='#000000')

# 预测点
dot.node('pred_pts', 'Predicted Points\n[1, 19, 2]', fillcolor='#CFD8DC', fontcolor='#000000')

# 连接
dot.edge('input', 'bone_tensor', constraint='true')
dot.edge('input', 'soft_tensor', constraint='true')
dot.edge('bone_tensor', 'bone_unet')
dot.edge('soft_tensor', 'soft_unet')
dot.edge('bone_unet', 'bone_hm')
dot.edge('soft_unet', 'soft_hm')
dot.edge('bone_hm', 'bone_softmax')
dot.edge('soft_hm', 'soft_softmax')
dot.edge('bone_softmax', 'bone_coords')
dot.edge('soft_softmax', 'soft_coords')
dot.edge('bone_coords', 'pred_pts')
dot.edge('soft_coords', 'pred_pts')

# 指定保存路径和名称
model_visualization_dir = 'Unet/models/visualizations/'
# 保存为 PNG
dot.render(f'{model_visualization_dir}/dual_unet_flow', cleanup=True)

print(f"- Model visualization saved to {model_visualization_dir}/dual_unet_flow.png")

