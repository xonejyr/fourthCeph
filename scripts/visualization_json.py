"""visualization.py: Script for visualizing learned Gaussian distributions."""
from datetime import datetime
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import importlib

# Import required modules
MODULE_NAME = "nfdp"
try:
    opt_module = importlib.import_module(f"{MODULE_NAME}.opt")
    trainer_module = importlib.import_module(f"{MODULE_NAME}.trainer_only_json")
    builder_module = importlib.import_module(f"{MODULE_NAME}.builder")
    # 修改这里的导入语句
    visualize_distribution_heatmap = trainer_module.visualize_distribution_heatmap
except ImportError as e:
    print(f"Error importing modules from {MODULE_NAME}: {e}")
    print("Ensure MODULE_NAME is correct and PYTHONPATH is set.")
    sys.exit(1)

# Assign variables from imported modules
opt = opt_module.opt
cfg = opt_module.cfg
logger = opt_module.logger
save_distribution_data_fn = trainer_module.save_distribution_data
builder = builder_module

def setup_logging(opt, output_vis_dir):
    """Configure logging for the visualization script."""
    if opt.log:
        os.makedirs(os.path.dirname(output_vis_dir), exist_ok=True)
        log_file_name = os.path.join(output_vis_dir, 'visualization_output.log')
        
        # Clear existing handlers
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
    """Set up the output directory for visualization data."""
    checkpoint_dir = os.path.dirname(opt.checkpoint)
    #max_bases = cfg['MODEL']['NUM_BASES']
    # 创建基础输出目录
    output_vis_dir = os.path.join(checkpoint_dir, f'results_analysis')
    # 分别创建json和热图的子目录
    json_dir = os.path.join(output_vis_dir, 'json_data')
    heatmap_dir = os.path.join(output_vis_dir, 'heatmaps')
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    
    return output_vis_dir, json_dir, heatmap_dir

def setup_dataloader(cfg, opt):
    """Create DataLoader for visualization."""
    vis_subset_name = 'val'  # Fixed to use 'val' subset
    logger.info(f"Using dataset subset: '{vis_subset_name}' for visualization")
    
    try:
        dataset = builder.build_dataset(
            cfg.DATASET,
            cfg.DATASET.PRESET,
            subset=vis_subset_name,
            is_train=False
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True if opt.gpu is not None else False
    )

def setup_model(opt, cfg):
    """Load and prepare the model."""
    model = builder.build_model(cfg.MODEL, preset_cfg=cfg.DATASET.PRESET)
    load_location = torch.device('cpu') if opt.gpu is None else torch.device(f'cuda:{opt.gpu}')
    
    if not os.path.exists(opt.checkpoint):
        logger.error(f"Checkpoint file not found: {opt.checkpoint}")
        raise FileNotFoundError(f"Checkpoint file not found: {opt.checkpoint}")
    
    logger.info(f'Loading model from checkpoint: {opt.checkpoint}...')
    try:
        model.load_state_dict(torch.load(opt.checkpoint, map_location=load_location))
    except RuntimeError as e:
        logger.warning(f"Error loading state_dict (strict=True): {e}")
        logger.info("Attempting to load with strict=False...")
        try:
            model.load_state_dict(torch.load(opt.checkpoint, map_location=load_location), strict=False)
            logger.info("Successfully loaded with strict=False.")
        except Exception as e2:
            logger.error(f"Failed to load even with strict=False: {e2}")
            raise
    
    return model.to(load_location)

def main():
    """Main function to orchestrate visualization."""
    # Determine device
    opt.gpu = 0 if torch.cuda.is_available() else None
    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        logger.info(f"Using GPU: {opt.gpu}")
    else:
        logger.info("Running on CPU.")

    torch.backends.cudnn.benchmark = True

    # Setup output directories
    output_vis_dir, json_dir, heatmap_dir = setup_output_directory(opt, cfg)
    
    # Setup logging
    setup_logging(opt, output_vis_dir)
    
    logger.info('******************************')
    logger.info("Starting Gaussian distribution visualization")
    logger.info(f"Options: {opt}")
    logger.info(f"Output directory: {output_vis_dir}")
    logger.info('******************************')
    
    # Setup DataLoader
    vis_loader = setup_dataloader(cfg, opt)
    
    # Setup model
    model = setup_model(opt, cfg)
    
    # Save distribution data to json directory
    save_distribution_data_fn(opt, cfg, vis_loader, model, json_dir)
    
    # Generate heatmap visualizations
    logger.info("Generating heatmap visualizations...")
    dataset = vis_loader.dataset
    image_dir = dataset.img_dir  # 假设数据集对象有 img_dir 属性指向原始图像目录
    
    # 处理所有生成的json文件
    for filename in os.listdir(json_dir):
        if filename.endswith('_data.json'):
            json_path = os.path.join(json_dir, filename)
            try:
                visualize_distribution_heatmap(
                    json_path=json_path,
                    vis_loader=vis_loader,
                    output_dir=heatmap_dir,
                    alpha=0.5
                )
                logger.info(f"Generated heatmap for {filename}")
            except Exception as e:
                logger.error(f"Error generating heatmap for {filename}: {e}")
    
    logger.info("Visualization script completed.")
    logger.info(f"JSON files saved in: {json_dir}")
    logger.info(f"Heatmap visualizations saved in: {heatmap_dir}")

if __name__ == "__main__":
    main()