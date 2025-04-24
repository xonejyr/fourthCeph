import os
import random
import shutil
import subprocess
import sys
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# 以项目根目录为默认目录，太重要了！它使得代码的撰写可以使用相对路径
from pathlib import Path

import yaml
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ray import tune, train
# Ray Tune 是一个用于分布式超参数调优的库，能够自动化地搜索最佳的超参数组合。
from ray.tune import CLIReporter
# CLIReporter 是一个用于在命令行界面（CLI）中实时报告训练进度和结果的工具。它会输出诸如当前试验的配置、训练指标（如损失、准确率等）以及试验状态（如运行中、完成、失败等）的信息。
from ray.tune.schedulers import ASHAScheduler
# 这行代码从 ray.tune.schedulers 模块中导入了 ASHAScheduler 类。ASHAScheduler 是一种基于异步连续减半算法（ASHA）的调度器，用于提前终止表现不佳的试验，从而加速超参数搜索过程。

parser = argparse.ArgumentParser(description="Hyperparameter search for UNet")
parser.add_argument('--exp-id', type=str, default="test_unet", help='Base experiment ID')
parser.add_argument('--cfg', type=str, required=True, help='Base config file path')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--numParallelExps', type=int, default=1, help='Number of sampler runs')
parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
    #parser.add_argument('--numSamples', type=int, default=None, help='Number of sampler runs')
args = parser.parse_args()
#################################################################################80
# 若要使用，有以下基础要修改
# 1. search_space 是超参数的搜索空间
# 2. train_with_config 中的config
# 3. reporter.parameter_columns
from scripts.utils.shared_params_manage import ParamManager
param_manager = ParamManager(config_file=args.config_file)
# the newst version doesn't need setting path directly
#from shared_params_manage import ParamManager

#from scripts.utils.shared_params_manage import ParamManager
#param_manager = ParamManager()



#sys.path.append("/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/scripts/utils")  # 替换为你的实际路径
#from shared_params_manage import ParamManager
#param_manager = ParamManager(config_file="/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/scripts/utils/shared_params.yaml")
#search_space = param_manager.get_search_space()

#--------------------------------------------------------------------------------80


def train_with_config(config, args, checkpoint_dir=None):
    # current working path is a tmp directory which is absolutely not the root directory of the project.
    # so the abs path is better for use

    # 在每个 worker 中添加路径
    #ParamManager = import_module("ParamManager", package="/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/scripts/utils")
    #param_manager = ParamManager("/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/scripts/utils/shared_params.yaml")
    ## 其余训练逻辑
    
    # 打开配置文件
    #args.cfg = os.path.abspath(args.cfg)
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f) #原始的cfg文件加载为字典

    #print("===========================================")
    #print(f"the args.cfg: {args.cfg}")

    # 更新超参数
    # config是超参数配置字典。它包含了当前试验的超参数值
    """
    "lr": tune.loguniform(1e-4, 1e-1), 
        "loss_type": tune.choice(['MSELoss', 'MSELoss_minmaxNorm', 'MSELoss_softmax']),
        "loss_basenumber": tune.choice([1, 10, 20, 40, 60]),
        "loss_masktype": tune.choice(['const', 'pow']),

        'sigma': tune.choice([3, 10, 15]),
        'IMAGE_SIZE': tune.choice([256, 256],[128, 128]),
        'model': tune.choice(['UNetPretrained', 'UNet']),
        'lr': tune.choice([1e-4, 1e-3]),
    """
    # 将search_space里的值赋给修改cfg字典
    #cfg['LOSS']['TYPE'] = config["loss_type"]
    #cfg['LOSS']['MASK_TYPE'] = config["loss_masktype"]
    #cfg['LOSS']['BASENUMBER'] = config["loss_basenumber"]

    #cfg['DATASET']['PRESET']['SIGMA'] = config["sigma"]
    #cfg['DATASET']['PRESET']['IMAGE_SIZE'] = config["IMAGE_SIZE"]
    #cfg['DATASET']['PRESET']['HEATMAP_SIZE'] = config["IMAGE_SIZE"] # 重设IMAGE_SIZE
    #cfg['MODEL']['TYPE'] = config["model"]
    #cfg['TRAIN']['LR'] = config["lr"]
    #cfg['DATASET']['PRESET']['METHOD_TYPE'] = config["METHOD_TYPE"]
    #cfg['TRAIN']['LR_SCHEDULE']['TYPE'] = config["SCHEDULE_TYPE"]
    #cfg['LOSS']['BETA'] = config["BETA"]
    #cfg['LOSS']['GAMMA'] = config["GAMMA"]
    ##cfg['TRAIN']['OPTIMIZER'] = config["optim"]
    cfg = param_manager.update_train_config(config, cfg)
    report_dict = {}
    report_dict.update(config)


    # 生成实验ID和目录
    if "_experiment_id" not in config:  # 使用 "_" 开头，避免 Tune 误认为是超参数
        config["_experiment_id"] = str(uuid.uuid4())
    trial_id = config["_experiment_id"]
    #print("================================================")
    #print(f"Inside train_with_config - Trial ID: {trial_id}")  # 调试输出
    #print(f"Config: {config}")  # 调试输出，确保 _experiment_id 在 config 中

    exp_dir = str(Path(args.cfg).parents[1] / "exp")
    cfg_file_name = os.path.basename(args.cfg).split('.')[0]

    param_names_suffix = param_manager.get_parameter_columns()
    # 拼接参数名，用 _ 连接
    result_suffix = '_'.join(param_names_suffix) if param_names_suffix else ''
    
    params_search_dir = f"{exp_dir}/{args.exp_id}-{cfg_file_name}/params_search={result_suffix}" # params_search_dir: 存放超参数搜索结果的目录
    os.makedirs(params_search_dir, exist_ok=True) # 如果没有则创建（递归创建），如果重复则不创建

    # 保存单实验配置文件: config_{trial_id}.yaml
    tmp_cfg_dir = f"{params_search_dir}/configs"
    os.makedirs(tmp_cfg_dir, exist_ok=True)
    tmp_cfg_path = f"{tmp_cfg_dir}/config_{trial_id}.yaml"
    with open(tmp_cfg_path, 'w') as f:
        yaml.dump(cfg, f)

    # 单实验日志文件路径: trial_{trial_id}.log
    #log_file = f"{params_search_dir}/logs/trial_{trial_id}.log"
    log_file_dir = f"{params_search_dir}/logs"
    os.makedirs(log_file_dir, exist_ok=True)
    log_file = f"{log_file_dir}/trial_{trial_id}.log"

    csv_file_dir = f"{params_search_dir}/history_csv"
    os.makedirs(csv_file_dir, exist_ok=True)
    csv_file = f"{csv_file_dir}/trial_{trial_id}.csv"

    # 检查点路径（用于临时保存模型）
    if checkpoint_dir:
        model_dir = os.path.join(checkpoint_dir)
    else:
        model_dir = f"{params_search_dir}/models/model_{trial_id}"
        os.makedirs(model_dir, exist_ok=True)
    

    scripts_dir = str(Path(args.cfg).parents[1] / "scripts")
    #print("===========================================")
    #print(f"the scripts_dir: {scripts_dir}")
    # 构造train.py命令
    cmd = [
        "python", f"{scripts_dir}/train_1_1.py",
        "--exp-id", args.exp_id,
        "--cfg", tmp_cfg_path,
        "--seed", str(args.seed) if args.seed else "2333",
        "--log", log_file,
        "--model-dir", model_dir,
        "--csv", csv_file
    ]

    # 准备直接采用train的返回来进行
    #opt = argparse.Namespace(
    #    exp_id=args.exp_id,
    #    cfg=tmp_cfg_path,
    #    seed="2333",
    #    log=log_file
    #)
    #best_mre, best_sd = train(None, opt=opt, cfg=cfg)


    # 执行训练并捕获输出
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    # 保存原始日志
    with open(log_file, 'w') as f:
        f.write(stdout)
        f.write(stderr)

    with open(log_file, 'r') as f:
        log_content = f.read()

    # 解析输出
    mre, sd = parse_output(log_content)
    if mre == 999 or sd == 999:
        print(f"Trial {trial_id} failed to produce valid metrics!")
        # 可选择抛出异常让Ray Tune重试
        raise tune.TuneError("Invalid metrics, retrying...")
    ##print("=====================================================")
    #print(f"the parsed best mre and sd are: {mre}, {sd}")

    
    report_dict.update({
        "best_mre": mre, 
        "best_sd": sd,
        "log_file": log_file,
        "config_file": tmp_cfg_path,
        "model_dir": model_dir,
        "csv_file": csv_file
        })

    # 针对全部寻优参数的值的获取
    
    
    ### 只针对非独立参数（动态采样参数）从 cfg 中提取值
    #for param in param_manager.search_space.keys():
    #    # 检查参数是否是 tune.grid_search 或 tune.choice 的结果
    #    search_space_value = param_manager.search_space[param]
    #    is_static = isinstance(search_space_value, (list, tuple)) or hasattr(search_space_value, 'categories')
    #    if not is_static:
    #        # 如果不是静态参数（如 tune.sample_from），从 cfg 中提取实际值
    #        if param in param_manager.cfg_paths:
    #            path = param_manager.cfg_paths[param].split('.')
    #            current = cfg
    #            for key in path:
    #                current = current[key]
    #            report_dict[param] = current
    
    #print("===================================================")
    #print(report_dict)
    
    tune.report(report_dict) # update用于两个字典的拼接

def parse_output(output):
    best_mre, best_sd = 999, 999
    lines = output.splitlines()
    for line in reversed(lines):
        # 去除前缀并清理
        line = line.split(')', 1)[-1].strip() if ')' in line else line.strip()
        if "best mean:" in line:
            try:
                # 分割出 mre 和 sd 部分
                parts = line.split('|')
                if len(parts) != 2:
                    print(f"Unexpected format in line: {line}")
                    continue
                
                # 提取 best mean
                mre_part = parts[0].split('best mean:')[-1].strip()
                # 提取 best sd
                sd_part = parts[1].split('best sd:')[-1].strip().replace('#', '')
                
                best_mre = float(mre_part)
                best_sd = float(sd_part)
                break  # 找到最新值后退出
            except (IndexError, ValueError) as e:
                print(f"Error parsing line '{line}': {e}")
                continue
    return best_mre, best_sd

def main(args):
    args.cfg = os.path.abspath(args.cfg) 
    # for this is a extra-file, it is better for abs path,
    # while in later the tune.run(), tmp directory may be used
    # print("=============================================")
    # print(f"Absolute config path: {args.cfg}") # 所有的外部文件都最好通过绝对路径获取

    # 在每个 worker 中添加路径
    #sys.path.append("/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/scripts/utils")
    #from ..shared_params_manage import ParamManager
    #param_manager = ParamManager()
    # 其余训练逻辑

    # 定义超参数搜索空间
    search_space = param_manager.get_search_space()
    """{
        # 生成一个在指定范围内（1e-4 到 1e-1）的随机值，但这些值在对数尺度上是均匀分布的。
        #"lr": tune.loguniform(1e-4, 1e-1), 
        #"loss_type": tune.choice(['MSELoss_softmax', 'MSELoss_doublesoftmax', 'FocalLoss', 'KLLoss']),
        #"loss_basenumber": tune.choice([1, 20, 40]),
        #"loss_masktype": tune.choice(['const', 'pow']),

        #'sigma': tune.choice([3, 10, 15]),
        #'IMAGE_SIZE': tune.choice([[256, 256],[128, 128]]),
        #'HIDDEN_DIM': tune.choice([32, 64, 128, 256]),
        #'NUM_PATTERNS': tune.choice([3, 5, 10, 15]),
        #'BETA': tune.choice([0.001, 0.01, 0.1, 0.5]),
        #'GAMMA': tune.choice([0.001, 0.01, 0.1, 0.5]),
        #'model': tune.choice(['UNetPretrained', 'UNet']),
        'lr': tune.choice([1e-5, 1e-4, 1e-3, 1e-2]),
        'METHOD_TYPE': tune.choice(['coord', 'heatmap']),
        'SCHEDULE_TYPE': tune.choice(['linearLR', 'plateauLR','cosineAnnealingLR']),
        #'optim': tune.choice(['adam', 'sgd']),
        #"trial_id": tune.sample_from(lambda _: str(random.randint(1000, 9999)))
        # 每次调用 tune.sample_from(lambda _: str(random.randint(1000, 9999))) 时，都会生成一个新的 4 位随机数字字符串。
        # 这个字符串可以用于唯一标识实验、设置随机种子等。但是用了tune.get_trial_id() 就不必再用这个了
        #"trial_id": tune.sample_from(lambda _: str(random.randint(1000, 9999)))
        #"trial_id": tune.sample_from(lambda spec: spec.trial_id) #将 trial_id 传入 config, 无法工作
    }
    """

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # 配置Tune调度器，
    # ASHAScheduler 是 Ray Tune 中的一种调度器，用于实现 异步连续减半算法（ASHA）。它的作用是动态地停止表现不佳的试验（Trials），从而加速超参数搜索。
    scheduler = ASHAScheduler(
        metric="best_mre", # 指定用于评估试验表现的指标。Ray Tune 需要你显式地调用 tune.report() 来提交这些指标。
        mode="min", # 指定优化目标的方向。metric to min here, 也可以是max
        max_t=cfg['TRAIN']['END_EPOCH'],  # 指定每个试验的最大训练时长（如 epoch 数）。 可以指定为cfg.TRAIN.END_EPOCH
        grace_period=2, # 指定每个试验的最短训练时长（如 epoch 数）。
        reduction_factor=2 # 指定每次减半的比例，2表示每次保留实验结果较好的50%
    )

    # 配置结果报告
    # CLIReporter 是 Ray Tune 中的一种报告器，用于在命令行界面（CLI）中实时显示试验的进度和结果。
    reporter = CLIReporter(
        metric_columns=["best_mre","best_sd"], # 指定要在报告中显示的指标列。Ray Tune 需要你显式地调用 tune.report() 来提交这些指标。
        #param_manager.get_parameter_columns() + 
        #parameter_columns=["loss_type", "loss_basenumber", "loss_masktype", "sigma", "IMAGE_SIZE", "model", "lr"] # 指定要在报告中显示的超参数列。
        #parameter_columns=["METHOD_TYPE", "SCHEDULE_TYPE", "lr"] # 指定要在报告中显示的超参数列。
        parameter_columns=param_manager.get_parameter_columns() # 指定要在报告中显示的超参数列。

    )
    # 运行Tune实验，
    # tune.run 是 Ray Tune 的核心函数，用于运行超参数搜索实验。
    # 根据指定的搜索空间、调度器和报告器运行超参数搜索。
    # 返回一个 Analysis 对象，用于分析试验结果。
    cfg_file_name = os.path.basename(args.cfg).split('.')[0]

    param_names_suffix = param_manager.get_parameter_columns()
    # 拼接参数名，用 _ 连接
    result_suffix = '_'.join(param_names_suffix) if param_names_suffix else ''
    storage_path = os.path.abspath(f"./exp/{args.exp_id}-{cfg_file_name}/params_search={result_suffix}") 
    # "+"="+f"


    # tune会在storage_path/name下保存所有的元数据

    # 自定义试验目录名
    def custom_trial_dirname(trial):
        #print(f"====================================================")
        #print(f"Full config: {trial.config}")  # 打印完整配置
        #
        #print("=====================================================")
        #trial_id = trial.config.get('trial_id', trial.trial_id)
        trial_id = trial.config.get('_experiment_id', trial.trial_id)
        #params = ",".join(f"{k}={v}" for k, v in trial.config.items() if k != "trial_id")
        #return f"trial_{trial_id}_{params}"
        # 使用 trial.evaluated_params 获取实际采样值
        params = trial.evaluated_params if hasattr(trial, 'evaluated_params') else trial.config
        params_str = ",".join(f"{k}={v}" for k, v in params.items() if k not in ["_experiment_id"])
        return f"trial_{trial_id}_{params_str}"
        
    #def custom_trial_dirname(trial):
    #    trial_id = trial.config['trial_id']
    #    params = ",".join(f"{k}={v}" for k, v in trial.config.items() if k != "trial_id")  # 排除 trial_id
    #    return f"trial_{trial_id}_{params}"

    # tune为分布式部署而设置，如果你要运行在集群环境（多机分布式）或需要自定义 Ray 配置（如调整资源分配），则需要在 tune.run(...) 之前手动调用 ray.init(...)。
    analysis = tune.run(
        #train_with_config, # 指定训练函数。
        lambda config: train_with_config(config, args), # 显式传递args
        resources_per_trial={"cpu": 1, "gpu": float(1/args.numParallelExps)}, # 指定每个试验所需的计算资源。
        config=search_space, # 指定超参数搜索空间。
        progress_reporter=reporter, # 指定报告器
        #num_samples=args.numSamples, # 指定试验的总次数，search_space 定义了超参数的范围或选项，但是不必遍历，这里配置ray.tune直接从参数空间中随机采样num_samples不同的组合来进行实验, 
        #num_samples=-1：这个设置确保 Tune 不对组合进行采样，而是完整运行所有实验。
        scheduler=scheduler, # 指定调度器。。
        storage_path=storage_path,  # Tune内部临时存储
        name="metadata_for_search", # 指定实验的名称，可以通过参数导入
        trial_dirname_creator=custom_trial_dirname, # 自定义文件命名
        raise_on_failed_trial=False,        # 失败不中断
        max_failures=99999999,                    # 无限重试直到成功
    )
    """
    storage保存的文件
    exp/
    └── tune_results/
        ├── experiment_state-20231010_123456.json
        ├── trial_1/ # 实验1
        │   ├── params.json # 实验1的参数配置
        │   ├── result.json # 实验1验的结果
        │   ├── checkpoint_1/ # 实验1的ckpt，检查点的保存是由你的 train_with_config 函数中的代码决定的：
            tune.load_checkpoint(), 
            with tune.checkpoint_dir(epoch) as checkpoint_dir: 
                checkpoint_path = **, 
                toch.save(...) 
            tune.report(val_loss=val_loss)
        │   └── log.txt       # 实验1的记录
        ├── trial_2/
        │   ├── params.json
        │   ├── result.json
        │   └── log.txt
        ...
        └── best_checkpoint/ # 整个调优过程中的最优模型
            ├── model.pth
            └── config.json

    analysis对象包含：
        1.试验结果数据：analysis.dataframe; 每行对应一个试验（Trial）。列包括超参数、指标值、状态、训练时长等。
        2.每次实验结果：analysis.results; 一个字典，包含每个试验的结果。键是试验 ID，值是一个字典，包含该试验的指标值、超参数、检查点路径等。
        3.最佳实验信息：analysis.[best_trial, best_config, best_logdir, best_checkpoint]
        4.实验元数据：
            experiment_state; 实验的全局状态信息,包括所有试验的状态、配置、结果等。
            stats;实验的统计信息, 包括总试验数、完成试验数、失败试验数等
    """

    # 保存和可视化结果
    #save_and_visualize_results(analysis)

    """
    理解 Ray Tune 的架构
        Ray Tune 的核心架构包括：
            Scheduler：管理试验的执行顺序和资源分配（如 ASHA、HyperBand）。
            Search Algorithm：定义如何从搜索空间中采样超参数（如随机搜索、贝叶斯优化）。
            Trainable：定义训练逻辑（如 train_with_config 函数）。
            你可以通过实现自定义的 Search Algorithm 或 Scheduler 来设计新的寻优算法。
        自定义搜索算法（如进化算法）：继承 SearchAlgorithm 基类，
        自定义Scheduler：继承 TrialScheduler
    """

    # 输出最佳配置
    best_trial = analysis.get_best_trial("best_mre", "min", "last")
    #print("=========================================")
    #print(f"Best trial config: {best_trial.config}")  # 调试输出
    # 使用 get 避免 KeyError，回退到 trial.trial_id

    # the result is for best_mre, ...
    # the configs are config/lr, ...
    print("Best trial final mre:\t", best_trial.last_result["best_mre"])
    print("Best trial final sd:\t", best_trial.last_result["best_sd"])
    print("Best trial config file:\t", best_trial.last_result["config_file"])
    print("Best trial log_file:\t", best_trial.last_result["log_file"])
    print("Best trial model_dir:\t", best_trial.last_result["model_dir"])

    # 将最佳检查点中的模型复制到最终路径
    best_trial_id = best_trial.trial_id
    base_dir = f"./exp/{args.exp_id}-{cfg_file_name}"
    params_search_dir = f"{base_dir}/params_search={result_suffix}"
    best_model_dir = best_trial.last_result["model_dir"]
    best_model_src = f"{best_model_dir}/best.pth"  # 假设 train.py 保存最佳模型到这里，实际上不是，检测是否序号对应
    best_model_dst_dir = f"{params_search_dir}/best_pths"
    os.makedirs(best_model_dst_dir, exist_ok=True)  # 创建 best_pths 目录
    best_model_dst = f"{best_model_dst_dir}/best_model_{best_trial_id}.pth"
    if os.path.exists(best_model_src):
        subprocess.run(["cp", best_model_src, best_model_dst])
        print(f"Saved best model of trial {best_trial_id} to {best_model_dst}")
    else:
        print(f"Warning: Best model file {best_model_src} not found!")
    
    # 可选：清理其他检查点
    for trial in analysis.trials:
        if trial.trial_id != best_trial.trial_id:
            checkpoint_dir = trial.last_result["model_dir"]
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
                #print(f"Deleted checkpoint for trial {trial.trial_id}")

    
    #print(f"Received args.cfg: {args.cfg}")
     # the newst version doesn't need setting path directly
    
    
main(args)
    #print(param_manager.get_parameter_columns())