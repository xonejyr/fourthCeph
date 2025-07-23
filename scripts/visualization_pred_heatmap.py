"""visualization_pred_heatmap.py: 用于生成和可视化预测热图的脚本"""
import logging
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import importlib

# 导入所需模块
MODULE_NAME = "nfdp"
try:
    opt_module = importlib.import_module(f"{MODULE_NAME}.opt")
    trainer_module = importlib.import_module(f"{MODULE_NAME}.trainer_only_new_heatmap")
    builder_module = importlib.import_module(f"{MODULE_NAME}.builder")
    visualize_distribution_heatmap = trainer_module.visualize_heatmap_distributions
except ImportError as e:
    print(f"导入{MODULE_NAME}模块时出错: {e}")
    sys.exit(1)

# 从导入的模块分配变量
opt = opt_module.opt
cfg = opt_module.cfg
logger = opt_module.logger
builder = builder_module

def setup_logging(opt, output_vis_dir):
    """配置日志记录"""
    if opt.log:
        os.makedirs(os.path.dirname(output_vis_dir), exist_ok=True)
        log_file_name = os.path.join(output_vis_dir, 'visualization_output.log')
        
        # 清除现有的处理程序
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        file_handler = logging.FileHandler(log_file_name)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(logging.NullHandler())

def setup_output_directory(opt, cfg):
    """设置可视化数据的输出目录"""
    checkpoint_dir = os.path.dirname(opt.checkpoint)
    #max_bases = cfg['MODEL']['NUM_BASES']
    # 创建基础输出目录
    output_vis_dir = os.path.join(checkpoint_dir, f'results_analysis')
    # 分别创建json和热图的子目录
    json_dir = os.path.join(output_vis_dir, 'json_data')
    heatmap_dir = os.path.join(output_vis_dir, 'heatmap_distributions')
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    
    return output_vis_dir, json_dir, heatmap_dir

def setup_dataloader(cfg, opt):
    """创建用于可视化的DataLoader"""
    vis_subset_name = 'val'  # 固定使用'val'子集
    logger.info(f"使用数据集子集: '{vis_subset_name}' 进行可视化")
    
    try:
        dataset = builder.build_dataset(
            cfg.DATASET,
            cfg.DATASET.PRESET,
            subset=vis_subset_name,
            is_train=False
        )
    except Exception as e:
        logger.error(f"创建数据集时出错: {e}")
        raise
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True if opt.gpu is not None else False
    )

def setup_model(opt, cfg):
    """加载和准备模型"""
    model = builder.build_model(cfg.MODEL, preset_cfg=cfg.DATASET.PRESET)
    load_location = torch.device('cpu') if opt.gpu is None else torch.device(f'cuda:{opt.gpu}')
    
    if not os.path.exists(opt.checkpoint):
        logger.error(f"找不到检查点文件: {opt.checkpoint}")
        raise FileNotFoundError(f"找不到检查点文件: {opt.checkpoint}")
    
    logger.info(f'从检查点加载模型: {opt.checkpoint}...')
    try:
        model.load_state_dict(torch.load(opt.checkpoint, map_location=load_location))
    except RuntimeError as e:
        logger.warning(f"加载state_dict时出错 (strict=True): {e}")
        logger.info("尝试使用strict=False加载...")
        try:
            model.load_state_dict(torch.load(opt.checkpoint, map_location=load_location), strict=False)
            logger.info("使用strict=False成功加载。")
        except Exception as e2:
            logger.error(f"即使使用strict=False也加载失败: {e2}")
            raise
    
    return model.to(load_location)

def main():
    """主函数，用于协调可视化过程"""
    # 确定设备
    opt.gpu = 0 if torch.cuda.is_available() else None
    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        logger.info(f"使用GPU: {opt.gpu}")
    else:
        logger.info("在CPU上运行。")

    torch.backends.cudnn.benchmark = True

    # 设置输出目录
    output_vis_dir, json_dir, heatmap_dir = setup_output_directory(opt, cfg)
    
    # 设置日志记录
    setup_logging(opt, output_vis_dir)
    
    logger.info('******************************')
    logger.info("开始生成预测热图可视化")
    logger.info(f"选项: {opt}")
    logger.info(f"输出目录: {output_vis_dir}")
    logger.info('******************************')
    
    # 设置DataLoader
    vis_loader = setup_dataloader(cfg, opt)
    
    # 设置模型
    model = setup_model(opt, cfg)

    
    # 生成热图可视化
    logger.info("正在生成热图可视化...")
    
    # 处理所有生成的json文件
    #for filename in os.listdir(json_dir):
    #if filename.endswith('_data.json'):
    try:
        visualize_distribution_heatmap(
            opt=opt,
            cfg=cfg,
            vis_loader=vis_loader,
            model=model,
            output_dir=heatmap_dir
        )
        logger.info(f"已生成热图: ")
    except Exception as e:
        logger.error(f"为生成热图时出错: {e}")
    
    logger.info("可视化脚本完成。")
    logger.info(f"热图可视化保存在: {heatmap_dir}")

if __name__ == "__main__":
    main()