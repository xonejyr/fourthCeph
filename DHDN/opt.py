import argparse  # to parse command line arguments
import logging
import os
from types import MethodType # bound a new method for an object in dynamic
import torch

from DHDN.utils import update_config


# description, usually the first setting for a parser, necessary for parser generation
parser = argparse.ArgumentParser(description='Normalizing Flow-based Distribution Prior')

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--exp-id', default='default', type=str,
                    help='Experiment ID')
#parser.add_argument('--gpu-id', default='0', type=int,
#                    help='GPU ID')
parser.add_argument('--model-dir', default=None, type=str,
                    help='dir to save the pth')

"----------------------------- Training options -----------------------------"
parser.add_argument('--seed', default=42, type=int,
                    help='random seed')
parser.add_argument('--snapshot', default=50, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

"----------------------------- Log options -----------------------------"
parser.add_argument('--valid-batch',
                    help='validation batch size',
                    type=int)
# used for loading the pretrained model
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    type=str)
parser.add_argument('--log', default=True, help='check if log is needed')

# the configs load from parser, the total configs
opt = parser.parse_args()
cfg_file_name = os.path.basename(opt.cfg).split('.')[0]
# split from opt.cfg to get updated configs from the YAML file
cfg = update_config(opt.cfg)

cfg['FILE_NAME'] = cfg_file_name

# here the exp_id refers to what experiment we are doing, while cfg_file_name refers to the name of the YAML filelog
# actually I feel cfg_file_name[:-5] without '.yaml' will be better
opt.work_dir = './exp/{}-{}/'.format(opt.exp_id, cfg_file_name)
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists("./exp/{}-{}".format(opt.exp_id, cfg_file_name)):
    os.makedirs("./exp/{}-{}".format(opt.exp_id, cfg_file_name), exist_ok=True)

# get the root logger
logger = logging.getLogger('')

# custom log way: the loss and accuracy of each epoch will be printed in the console
def epochInfo(self, set, idx, loss):
    self.info('{set}-{idx:d} epoch | loss:{loss:.8f}'.format(
        set=set,
        idx=idx,
        loss=loss
    ))

# bound the custom log way to the root logger
logger.epochInfo = MethodType(epochInfo, logger)
