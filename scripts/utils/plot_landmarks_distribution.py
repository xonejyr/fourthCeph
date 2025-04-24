import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))# 此处为进入到对应的下列Unet的父目录

from Unet.opt import cfg
from Unet import builder
#from ...Unet.opt import cfg
#from ...Unet import builder

from scipy.spatial.distance import cdist  # 用于计算相似度（可选）

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 路径配置: fig_dir
import os
# the newst version doesn't need setting path directly
import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description='Set yaml file for plot_landmarks_distribution')
parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
# Parse arguments
args = parser.parse_args()

# the newst version doesn't need setting path directly
from shared_params_manage import ParamManager
param_manager = ParamManager(config_file=args.config_file)
paths = param_manager.get_paths()
fig_dir = paths['root_fig_dir']
fig_dir = os.path.join(fig_dir, "distribution_statistics")

os.makedirs(fig_dir, exist_ok=True)
#--------------------------------------------------------------------------------80

#################################################################################80
# 若要使用，有以下基础要修改
#fig_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_unet-512x512_unet_ce_heatmap/visualizations/"
#os.makedirs(fig_dir, exist_ok=True)
#--------------------------------------------------------------------------------80

def plot_gaussian_distribution(all_points, mus, sigmas, fig_dir, subset):
    num_joints = len(all_points)
    colors = plt.cm.get_cmap("tab10", num_joints)  # 生成不同颜色
    #colors = plt.colormaps.get_cmap("tab10", num_joints)
    
    plt.figure(figsize=(20, 8))
    
    for j in range(num_joints):
        points = np.array(all_points[j])  # 获取所有样本的该标志点坐标
        mu_j = mus[j].copy()
        sigma_j = sigmas[j]
        
        # 绘制所有样本的散点
        plt.scatter(points[:, 0], points[:, 1], color=colors(j), alpha=0.5, label=f"Point {j+1:02} | {sigmas[j]:.0f}")
        
        # 绘制均值
        plt.scatter(mu_j[0], mu_j[1], color=colors(j), alpha=1, marker='X', edgecolors='black', s=300, linewidths=2)
        
        # 绘制高斯核密度估
        #sns.kdeplot(x=points[:, 0], y=points[:, 1], levels=3, color=colors(j))
        # 绘制3sigma范围的圆
        #circle = plt.Circle(mu_j, 3 * sigma_j, color=colors(j), fill=False, linestyle="dashed")
        #plt.gca().add_patch(circle)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Landmark Distribution for '{subset}' Dataset ")
    plt.grid()
    plt.gca().invert_yaxis()
    plt.savefig(fig_dir + f'/landmark_distribution_{subset}.png')


def compute_gaussian_params(all_points):
    """
    计算每个标志点的均值 mu 和标准差 sigma（单一标量）。
    
    参数:
        all_points (list): num_joints 个子列表，每个子列表包含 N 个 [x, y] 坐标对
    
    返回:
        mus (np.array): num_joints 个 [x, y] 均值
        sigmas (np.array): num_joints 个 sigma 标量
    """
    num_joints = len(all_points)
    mus = np.zeros((num_joints, 2))  # 存放均值
    sigmas = np.zeros(num_joints)  # 存放 sigma（标量）

    for j in range(num_joints):
        points = np.array(all_points[j])  # 形状: [N, 2]
        mu_j = np.mean(points, axis=0)  # 均值 (x, y)
        sigma_j = np.sqrt(np.mean(np.sum((points - mu_j) ** 2, axis=1)))  # 欧几里得标准差（标量）

        mus[j] = mu_j
        sigmas[j] = sigma_j

    return mus, sigmas



# 绘制一个数据集上的所有标志点坐标分布，并打印坐标
def plot_landmarks_of_dataset(data_loader, fig_dir):
    num_joints = None
    all_points = []
    
    # 遍历数据集
    for inps, labels, img_ids in data_loader:
        keypoints = labels['target_uv'].reshape(inps.shape[0], -1, 2).numpy() # [batchsize, num_joints, 2]
        keypoints[..., 0] = (keypoints[..., 0] + 0.5) * cfg.DATASET.PRESET.IMAGE_SIZE[0]
        keypoints[..., 1] = (keypoints[..., 1] + 0.5) * cfg.DATASET.PRESET.IMAGE_SIZE[1]
        if num_joints is None:
            num_joints = keypoints.shape[1]
            all_points = [[] for _ in range(num_joints)] # initialize a list of blank lists (number of joints)
        
        for j in range(num_joints):
            all_points[j].extend(keypoints[:, j].tolist())  # 确保转换为列表,keypoints[:, j] [batchsize, 2]
 # # 扩展整个 batch 的 j 号点
        # [num_joints, batchsize, 2]
    
    # all_points dimension of [num_joints, len(dataset), 2]
    
    all_points = [np.array(points) for points in all_points]
    
    # 计算均值和标准差
    mus, sigmas = compute_gaussian_params(all_points)
    #mus = np.array([points.mean(axis=0) for points in all_points]) # x, y in mus
    #sigma = np.array([points.std(axis=0) for points in all_points]) # 
    
    plot_gaussian_distribution(all_points, mus, sigmas, fig_dir, data_loader.dataset.subset)

    distribution_file = f"distribution_statistics_{data_loader.dataset.subset}.txt"
    file_path = os.path.join(fig_dir, distribution_file)
    
    # 准备要写入的内容
    lines = []
    lines.append(f"The Gaussian distribution for each landmark point in '{data_loader.dataset.subset}' dataset:")
    lines.append("\t\tmu\t\tsigma")
    for j in range(num_joints):
        lines.append(f'Point {j+1:02}: [{mus[j, 0]:.3f}, {mus[j, 1]:.3f}], \t{sigmas[j]:.3f}')
    
    # 打印到控制台
    for line in lines:
        print(line)
    
    # 写入文件
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
        f.close()

    return mus, sigmas

def compare_landmark_differences(mus_train, mus_val, mus_test, sigmas_train, sigmas_val, sigmas_test,
                                 fig_dir,
                                subset_names=["train", "val", "test"]):
    """
    比较三个数据集的标志点分布差异。
    
    参数：
    - mus_train, mus_val, mus_test: 均值数组，形状 [num_joints, 2]
    - sigmas_train, sigmas_val, sigmas_test: 标准差数组，形状 [num_joints]
    - subset_names: 数据集名称列表
    - fig_dir: 保存图片的目录
    """
    num_joints = mus_train.shape[0]
    joint_indices = np.arange(num_joints) + 1 # 标志点索引
    
    # 创建保存目录
    fig_dir=os.path.join(fig_dir, "comparison_figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. 均值 X 坐标对比图
    plt.figure(figsize=(12, 6))
    plt.plot(joint_indices, mus_train[:, 0], 'b-', label=f"{subset_names[0]} X Mean")
    plt.plot(joint_indices, mus_val[:, 0], 'g-', label=f"{subset_names[1]} X Mean")
    plt.plot(joint_indices, mus_test[:, 0], 'r-', label=f"{subset_names[2]} X Mean")
    plt.xticks(joint_indices)
    plt.title("Mu-X Coordinate Across Datasets")
    plt.xlabel("Joint Index")
    plt.ylabel("Mu-X")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "mean_x_comparison.png"))
    plt.close()

    # 2. 均值 Y 坐标对比图
    plt.figure(figsize=(12, 6))
    plt.plot(joint_indices, mus_train[:, 1], 'b-', label=f"{subset_names[0]} Y Mean")
    plt.plot(joint_indices, mus_val[:, 1], 'g-', label=f"{subset_names[1]} Y Mean")
    plt.plot(joint_indices, mus_test[:, 1], 'r-', label=f"{subset_names[2]} Y Mean")
    plt.xticks(joint_indices)
    plt.title("Mu-Y Across Datasets")
    plt.xlabel("Joint Index")
    plt.ylabel("Mu-Y")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "mean_y_comparison.png"))
    plt.close()

    # 3. 标准差对比图
    plt.figure(figsize=(12, 6))
    plt.plot(joint_indices, sigmas_train, 'b-', label=f"{subset_names[0]} Sigma")
    plt.plot(joint_indices, sigmas_val, 'g-', label=f"{subset_names[1]} Sigma")
    plt.plot(joint_indices, sigmas_test, 'r-', label=f"{subset_names[2]} Sigma")
    plt.xticks(joint_indices)
    plt.title("Sigma Across Datasets")
    plt.xlabel("Joint Index")
    plt.ylabel("Sigma")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "sigma_comparison.png"))
    plt.close()

    ## 4. 计算并打印均值之间的欧几里得距离
    #dist_train_val = np.mean(cdist(mus_train, mus_val))  # 默认欧几里得距离
    #dist_train_test = np.mean(cdist(mus_train, mus_test))
    #dist_val_test = np.mean(cdist(mus_val, mus_test))
    #
    #print("Cross-Dataset Mean Differences (Euclidean Distance):")
    #print(f"{subset_names[0]} vs {subset_names[1]}: {dist_train_val:.4f}")
    #print(f"{subset_names[0]} vs {subset_names[2]}: {dist_train_test:.4f}")
    #print(f"{subset_names[1]} vs {subset_names[2]}: {dist_val_test:.4f}")

    ## 5. 保存差异统计到文件
    #stats_file = os.path.join(fig_dir, "cross_dataset_differences.txt")
    #with open(stats_file, 'w') as f:
    #    f.write("Cross-Dataset Mean Differences (Euclidean Distance):\n")
    #    f.write(f"{subset_names[0]} vs {subset_names[1]}: {dist_train_val:.4f}\n")
    #    f.write(f"{subset_names[0]} vs {subset_names[2]}: {dist_train_test:.4f}\n")
    #    f.write(f"{subset_names[1]} vs {subset_names[2]}: {dist_val_test:.4f}\n")
    #print(f"- Differences saved to: {stats_file}")

# if the subset is 'train', the dataset would be augmented, so not choose it, other the results would change every time
dataset_train = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset='train')
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
dataset_val = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset='val')
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
dataset_test = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset='test')
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

mus_train, sigmas_train = plot_landmarks_of_dataset(data_loader_train, fig_dir)
mus_val, sigmas_val = plot_landmarks_of_dataset(data_loader_val, fig_dir)
mus_test, sigmas_test = plot_landmarks_of_dataset(data_loader_test, fig_dir)
print(f"- Distribution of landmarks saved to {fig_dir}")

compare_landmark_differences(
    mus_train, mus_val, mus_test,
    sigmas_train, sigmas_val, sigmas_test, fig_dir,
    subset_names=["train", "val", "test"]
)

print(f" Comparison of sigmas val vs test")
for i in range(len(sigmas_val)):
    print(f"Point {i+1:02}: Train: {sigmas_val[i]:02.2f}, Val: {sigmas_test[i]:02.2f} (diff: {np.abs(sigmas_val[i] - sigmas_train[i]):02.2f})")
    #print(np.abs(sigmas_val - sigmas_train))

#################################################################################80
# 打印分布差异
def load_keypoints(cfg, data_loader):
    #dataset = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset=subset)
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    all_keypoints = []
    for inps, labels, img_ids in  data_loader:
        keypoints = labels['target_uv'].reshape(inps.shape[0], -1, 2).numpy() # [batchsize, num_joints, 2]
        # 获取目标图像尺寸
        height, width = cfg.DATASET.PRESET.IMAGE_SIZE

        # 转换坐标
        #keypoints_transformed = keypoints.copy()  # 创建副本避免修改原始数据
        keypoints[..., 0] = (keypoints[..., 0] + 0.5) * width  # x 坐标
        keypoints[..., 1] = (keypoints[..., 1] + 0.5) * height  # y 坐标

        all_keypoints.append(keypoints)
    return np.concatenate(all_keypoints, axis=0)  # [n_samples, num_joints, 2]

# 可视化函数（只针对指定标志点）
def plot_distribution(kp_train, kp_val, kp_test, subset_train="train", subset_val="val", subset_test="test", joint_indices=None, fig_dir="figures"):
    """
    可视化三个数据集（train, val, test）的标志点分布，并分析相似度和差异。
    
    参数：
    - kp_train, kp_val, kp_test: 三个数据集的标志点，形状 [n_samples, num_joints, 2]
    - subset_train, subset_val, subset_test: 数据集名称
    - joint_indices: 要分析的标志点索引列表，若为 None 则分析所有点
    - fig_dir: 保存图片的目录
    """
    # 如果未指定 joint_indices，则使用所有标志点
    if joint_indices is None:
        joint_indices = range(kp_train.shape[1])

    # 过滤出指定标志点的数据
    kp_train_selected = kp_train[:, joint_indices, :]  # [n_samples_train, len(joint_indices), 2]
    kp_val_selected = kp_val[:, joint_indices, :]      # [n_samples_val, len(joint_indices), 2]
    kp_test_selected = kp_test[:, joint_indices, :]    # [n_samples_test, len(joint_indices), 2]

    # 为每个选定的标志点绘制散点图和箱线图
    for i, joint_idx in enumerate(joint_indices):
        # 散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(kp_train_selected[:, i, 0], kp_train_selected[:, i, 1], c='blue', alpha=0.5, label=subset_train)
        plt.scatter(kp_val_selected[:, i, 0], kp_val_selected[:, i, 1], c='green', alpha=0.5, label=subset_val)
        plt.scatter(kp_test_selected[:, i, 0], kp_test_selected[:, i, 1], c='red', alpha=0.5, label=subset_test)
        plt.title(f"Joint {joint_idx} Distribution")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.savefig(os.path.join(fig_dir, f"scatter_joint_{joint_idx}_{subset_train}_vs_{subset_val}_vs_{subset_test}.png"))
        plt.close()

        # 箱线图（X坐标为例）
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[kp_train_selected[:, i, 0], kp_val_selected[:, i, 0], kp_test_selected[:, i, 0]], 
                    palette=['blue', 'green', 'red'])
        plt.xticks([0, 1, 2], [subset_train, subset_val, subset_test])
        plt.title(f"Joint {joint_idx} X Coordinate Distribution")
        plt.ylabel("X Coordinate")
        plt.savefig(os.path.join(fig_dir, f"boxplot_joint_{joint_idx}_x_{subset_train}_vs_{subset_val}_vs_{subset_test}.png"))
        plt.close()

    ## 统计指标（均值对比）
    #mean_train = np.mean(kp_train_selected, axis=0)  # [len(joint_indices), 2]
    #mean_val = np.mean(kp_val_selected, axis=0)      # [len(joint_indices), 2]
    #mean_test = np.mean(kp_test_selected, axis=0)    # [len(joint_indices), 2]
    #
    #plt.figure(figsize=(12, 6))
    #plt.plot(joint_indices, mean_train[:, 0], 'b-', label=f"{subset_train} X Mean")
    #plt.plot(joint_indices, mean_val[:, 0], 'g-', label=f"{subset_val} X Mean")
    #plt.plot(joint_indices, mean_test[:, 0], 'r-', label=f"{subset_test} X Mean")
    #plt.title("Mean X Coordinate Across Selected Joints")
    #plt.xlabel("Joint Index")
    #plt.ylabel("Mean X")
    #plt.legend()
    #plt.savefig(os.path.join(fig_dir, f"mean_x_{subset_train}_vs_{subset_val}_vs_{subset_test}_selected.png"))
    #plt.close()

    # 可选：计算相似度（例如欧几里得距离）
    #print("Similarity Analysis (Euclidean Distance between means):")
    #dist_train_val = np.mean(cdist(mean_train, mean_val, metric='euclidean'))
    #dist_train_test = np.mean(cdist(mean_train, mean_test, metric='euclidean'))
    #dist_val_test = np.mean(cdist(mean_val, mean_test, metric='euclidean'))
    #print(f"{subset_train} vs {subset_val}: {dist_train_val:.4f}")
    #print(f"{subset_train} vs {subset_test}: {dist_train_test:.4f}")
    #print(f"{subset_val} vs {subset_test}: {dist_val_test:.4f}")

# 加载两个数据集的标志点
subset1, subset2 = "val", "test"  # 可调整为其他组合，如"val" vs "test"
kp_train = load_keypoints(cfg, data_loader_train)  # [n_samples1, num_joints, 2]
kp_val = load_keypoints(cfg, data_loader_val)  # [n_samples2, num_joints, 2]
kp_test = load_keypoints(cfg, data_loader_test)  # [n_samples2, num_joints, 2]
num_joints = kp_train.shape[1]

# 指定要分析的标志点索引
joint_indices = [5, 15]  # 示例：只选择第0、5、10个标志点，可根据需要修改

# 执行可视化
plot_distribution(kp_train, kp_val, kp_test, "train", "val", "test", joint_indices, fig_dir)