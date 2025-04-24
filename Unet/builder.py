from torch import nn

from Unet.utils import Registry, build_from_cfg, retrieve_from_cfg


MODEL = Registry('model')
LOSS = Registry('loss')
DATASET = Registry('dataset')


# general function to build a module from config, 
# if cfg is a list, it will build a sequential module
# if cfg is a dict, it will build a module directly
def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        """
        the cfg here is from yaml, so it is a dict, 
        so the following is always used
        """
        return build_from_cfg(cfg, registry, default_args)
        # cfg, whose 'TYPE' key represents the class name registered in the registry, 
        # registry is the regitry object, which has all same-kind modules, like all models, losses, datasets
        # the default_args is used to make the settings of the class module find by cfg.TYPE key in the registry object

# build NFDP module from config, here preset_cfg is a general parameters while **kwargs are specific parameters for framework
def build_model(cfg, preset_cfg, **kwargs):
    """
    using cfg, preset_cfg, and **kwargs as settings to build a DATASET
    """
    exec(f'from .models import {cfg.TYPE}')
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    # actally the preset_cfg is changed to pass to a sub-dict whose key is 'PRESET' and the value is preset_cfg, if else in **kwargs, it will be directly passed to the build function as dict items, which means {key1: value1, key2: value2,....}
    return build(cfg, MODEL, default_args=default_args)

# build LOSS module from config
def build_loss(cfg, preset_cfg, **kwargs):
    """
    using cfg as settings to build a DATASET
    """
    exec(f'from .losses import {cfg.TYPE}')
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, LOSS, default_args=default_args)

# build DATASET module from config
# example usage: 
# cfg = {'TYPE': 'MyDataset', 'path': '/data/train', 'transform': transforms}
# preset_cfg = {'batch_size': 32}
# dataset = build_dataset(cfg, preset_cfg)
# what is exec?
def build_dataset(cfg, preset_cfg, **kwargs):
    """
    using cfg, preset_cfg, and **kwargs as settings to build a DATASET
    """
    exec(f'from .datasets import {cfg.TYPE}')
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    # 新增项目
    # default_args = {'PRESET':preset_cfg, key1: value1, key2: value2,...}
    return build(cfg, DATASET, default_args=default_args)

# retrieve the dataset marked in config
# difference with build_dataset?
def retrieve_dataset(cfg):
    exec(f'from .datasets import {cfg.TYPE}')
    return retrieve_from_cfg(cfg, DATASET)
