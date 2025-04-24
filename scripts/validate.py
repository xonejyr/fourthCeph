"""Validation script."""
from datetime import datetime
import logging
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


from Unet.opt import opt, cfg, logger
from Unet.trainer_only import validate
from Unet import builder
from Unet.utils import NullWriter
from Unet.datasets.transforms import get_coord

def main():
    main_worker(None, opt, cfg)


def main_worker(gpu, opt, cfg):
    if gpu is not None:
        opt.gpu = gpu
    else:
        # 手动设置为 GPU 0（如果你只有一个 GPU）
        opt.gpu = 0

    if opt.log:
        cfg_file_name = os.path.basename(opt.cfg).split('.')[0]
        filehandler = logging.FileHandler(
            '{}/test.log'.format(os.path.dirname(opt.checkpoint), cfg_file_name))
        #log_dir = f'./exp/{opt.exp_id}-{cfg_file_name}'
        ##os.makedirs(log_dir, exist_ok=True)
        #filehandler = logging.FileHandler(f'{log_dir}/test.log')
        streamhandler = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
    else:
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')
    now = datetime.now()
    logger.info(f"Start Time is {now.strftime('%Y-%m-%d %H:%M:%S')}")

    torch.backends.cudnn.benchmark = True # PyTorch 中用来优化卷积神经网络（CNN）性能的一个设置。

    test_dataset = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    m = builder.build_model(cfg.MODEL,preset_cfg=cfg.DATASET.PRESET)

    heatmap_to_coord = get_coord(cfg, cfg.DATASET.PRESET.HEATMAP_SIZE)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=True)

    m.cuda(opt.gpu)

    with torch.no_grad():

        mre, sd, sdr = validate(opt, cfg, test_loader, m, heatmap_to_coord, show_hm=True)
        logger.info(f"############# Test Result #############")
        logger.info(f'MRE:\t\t{mre:.4f}mm, \n\rSD:\t\t{sd:.4f}mm\n\r')
        for radius, rate in sdr.items():
            logger.info(f'SDR ({radius}mm):\t{rate:.4f}%')


if __name__ == "__main__":

    main()
