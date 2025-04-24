import ray
from ray.tune.analysis import ExperimentAnalysis
import os

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 路径设置: experiment_path, fig_dir
# 2. ray初始化: config_cols, just copy that in importance_analysis.py
# 3. 箱线图: params, just copy parameter_columns in param_search.py
# 4. 散点图: px.scatter() size, shape, color
# Note: to change, copy that in param_search.py to do is OK
#       parameter_columns=["NUM_PATTERNS", "HIDDEN_DIM", "lr", "BETA", "GAMMA"]

# the newst version doesn't need setting path directly
from shared_params_manage import ParamManager
param_manager = ParamManager()
paths = param_manager.get_paths()
experiment_path = paths['experiment_path']
fig_dir = paths['experiment_fig_dir']

os.makedirs(fig_dir, exist_ok=True)
#--------------------------------------------------------------------------------80


#################################################################################80
# path settings
# coord
#experiment_path = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_unet-512x512_unet_ce_coord/params_search/metadata_for_search"  # 替换为你的实验路径
#fig_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_unet-512x512_unet_ce_coord/params_search/visualizations"
# heatmap
#experiment_path = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_DHDN-256x256_DHDN_ce_coord/params_search/metadata_for_search"  # 替换为你的实验路径
#fig_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_DHDN-256x256_DHDN_ce_coord/params_search/visualizations"
#os.makedirs(fig_dir, exist_ok=True)
#--------------------------------------------------------------------------------80


#################################################################################80
# 数据加载与处理
# 初始化 Ray（如果未初始化）
ray.init(ignore_reinit_error=True)

# 使用 ExperimentAnalysis 加载实验数据
analysis = ExperimentAnalysis(experiment_path)

# 将结果转为 DataFrame
df = analysis.dataframe()

# 提取配置参数和指标（假设指标名为 'best_mre' 和 'best_sd'）
# 如果指标名称不同，请根据 df.columns 查看并替换
#config_cols = ['config/loss_type', 'config/loss_basenumber', 'config/loss_masktype', 'config/sigma', 'config/IMAGE_SIZE', 'config/model', 'config/lr']
config_cols = [f"config/{col}" for col in param_manager.get_parameter_columns()]
#['config/NUM_PATTERNS', 'config/HIDDEN_DIM', 'config/lr', 'config/BETA', 'config/GAMMA']
metric_cols = ['best_mre', 'best_sd']  # 替换为你的实际指标名

# 确保只保留需要的列
df = df[config_cols + metric_cols]

# 清理列名，移除 'config/' 前缀
df.columns = [col.replace('config/', '') for col in df.columns]
# 预处理 IMAGE_SIZE，将列表转为字符串
# 检查并转换可能的列表类型列
for col in df.columns:
    # 如果该列的第一个元素是列表，则转换为字符串
    if isinstance(df[col].iloc[0], list):
        df[col] = df[col].apply(lambda x: 'x'.join(map(str, x)) if isinstance(x, list) else x)
    # 将 None 转换为字符串 'None'
    df[col] = df[col].apply(lambda x: 'None' if x is None else x)
    df[col] = df[col].apply(lambda x: 'True' if x is True else x)

df1 = df.sort_values(by='best_mre', ascending=True, inplace=False)
#df = df.dropna()  # 直接去除所有含有 NaN 的行
print(df1.head(50))  # 检查数据
#--------------------------------------------------------------------------------80

#################################################################################80
# 可视化
# 箱线图 - 参数独立影响

## simple version
#import seaborn as sns
#import matplotlib.pyplot as plt
#
## 设置绘图风格
#sns.set(style="whitegrid")
#
## 绘制每个参数对 best_mre 和 best_sd 的箱线图
#params = ['loss_type', 'loss_basenumber', 'loss_masktype', 'sigma', 'IMAGE_SIZE', 'model', 'lr']
#fig, axes = plt.subplots(len(params), 2, figsize=(15, 5 * len(params)))
#
#for i, param in enumerate(params):
#    # best_mre
#    sns.boxplot(x=param, y='best_mre', data=df, ax=axes[i, 0])
#    axes[i, 0].set_title(f'{param} vs best_mre')
#    axes[i, 0].tick_params(axis='x', rotation=45)
#    
#    # best_sd
#    sns.boxplot(x=param, y='best_sd', data=df, ax=axes[i, 1])
#    axes[i, 1].set_title(f'{param} vs best_sd')
#    axes[i, 1].tick_params(axis='x', rotation=45)
#
#plt.tight_layout()
#
#plt.savefig(f"{fig_dir}/params_search_analysis.png")

# a beautiful one
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置全局绘图风格
sns.set(style="whitegrid", font_scale=1.2)  # 增大字体比例

# 定义参数列表（假设与之前一致）
#params = ['loss_type', 'loss_basenumber', 'loss_masktype', 'sigma', 'IMAGE_SIZE', 'model', 'lr']
#params = ['optim', 'lr', 'loss_type']
#params = ["NUM_PATTERNS", "HIDDEN_DIM", "lr", "BETA", "GAMMA"]
params = param_manager.parameter_columns

# 创建子图
# a single picture for all params
#fig, axes = plt.subplots(len(params), 2, figsize=(18, 6 * len(params)), sharey=False)
#axes = axes.reshape(len(params), 2)

# 使用更鲜艳的调色板
palette = "Set2"  # 可选: "husl", "deep", "muted", "bright"

for i, param in enumerate(params):
    # a picture for each param
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
    # best_mre 箱线图
    sns.boxplot(
        x=param, 
        y='best_mre', 
        data=df, 
        ax=axes[0], 
        #ax=axes[i, 0], 
        #palette=palette,  # 应用调色板
        hue=param,
        legend=False,
        width=0.6,        # 调整箱体宽度
        linewidth=1.5     # 增加线条粗细
    )
    # 添加散点图，展示数据分布
    sns.stripplot(
        x=param, 
        y='best_mre', 
        data=df, 
        ax=axes[0], 
        #ax=axes[i, 0], 
        color='black', 
        size=3, 
        alpha=0.3, 
        jitter=True
    )
    axes[0].set_title(f'{param} vs Best MRE', fontsize=14, weight='bold')
    axes[0].set_xlabel('')  # 移除 x 标签，依赖标题
    axes[0].set_ylabel('Best MRE', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    #axes[i, 0].set_title(f'{param} vs Best MRE', fontsize=14, weight='bold')
    #axes[i, 0].set_xlabel('')  # 移除 x 标签，依赖标题
    #axes[i, 0].set_ylabel('Best MRE', fontsize=12)
    #axes[i, 0].tick_params(axis='x', rotation=45, labelsize=10)

    # best_sd 箱线图
    sns.boxplot(
        x=param, 
        y='best_sd', 
        data=df, 
        ax=axes[1], 
        #ax=axes[i,1], 
        #palette=palette, 
        hue=param,
        legend=False,
        width=0.6, 
        linewidth=1.5
    )
    # 添加散点图
    sns.stripplot(
        x=param, 
        y='best_sd', 
        data=df, 
        ax=axes[1], 
        #ax=axes[i,1], 
        color='black', 
        size=3, 
        alpha=0.3, 
        jitter=True
    )
    axes[1].set_title(f'{param} vs Best SD', fontsize=14, weight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Best SD', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    # a single picture for all params
    #axes[i, 1].set_title(f'{param} vs Best SD', fontsize=14, weight='bold')
    #axes[i, 1].set_xlabel('')
    #axes[i, 1].set_ylabel('Best SD', fontsize=12)
    #axes[i, 1].tick_params(axis='x', rotation=45, labelsize=10)
    
# 调整布局，增加间距
    plt.tight_layout(pad=3.0)
    plt.savefig(f"{fig_dir}/params_search_analysis_{param}.png", dpi=300, bbox_inches='tight')

# 保存为高质量图片（可选）
#plt.savefig(f"{fig_dir}/params_search_analysis.png", dpi=300, bbox_inches='tight')
#--------------------------------------------------------------------------------80

#################################################################################80
# 可视化
# 交互式散点图 - 参数组合关系
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# 确保 param_manager.sort_by_importance 已定义
if not hasattr(param_manager, 'sort_by_importance'):
    raise AttributeError("param_manager 缺少 sort_by_importance 属性")

# 获取 size_col
if len(param_manager.sort_by_importance) > 1:
    size_col = param_manager.sort_by_importance[1]  # 取第二个元素，例如 "lr" 或 "loss_type"
elif len(param_manager.sort_by_importance) == 1:
    size_col = param_manager.sort_by_importance[0]  # 使用唯一元素作为默认值
    print(f"提示：sort_by_importance 只有一个元素，长度为 {len(param_manager.sort_by_importance)}，size_col 设置为 {size_col}")
else:
    raise ValueError("sort_by_importance 列表为空，无法获取 size_col")

# 检测类型并处理
if df[size_col].dtype == 'object':  # 如果是字符类型
    le = LabelEncoder()
    df['size_encoded'] = le.fit_transform(df[size_col])
    # 归一化到合理范围（可选，避免点太小或太大）
    df['size_encoded'] = 10 + 90 * (df['size_encoded'] - df['size_encoded'].min()) / (df['size_encoded'].max() - df['size_encoded'].min())
    size_col_to_use = 'size_encoded'  # 使用编码后的列
else:
    size_col_to_use = size_col  # 直接使用原列（数值型）


# 使用散点图展示 best_mre 和 best_sd 的关系
if len(param_manager.sort_by_importance) == 1:
    fig = px.scatter(
        df,
        x='best_mre',
        y='best_sd',
        color=param_manager.sort_by_importance[0],           # 用颜色表示 model
        size=size_col_to_use,              # 用点的大小表示 lr
        #symbol=param_manager.sort_by_importance[2],     # 用符号表示 loss_type
        #hover_data=[size_col].append(param_manager.sort_by_importance[3:]),#['loss_basenumber', 'loss_masktype', 'lr', 'IMAGE_SIZE'],  # 悬停显示其他参数
        title='best_mre vs best_sd with Parameter Variations'
        )
elif len(param_manager.sort_by_importance) == 2:
    fig = px.scatter(
        df,
        x='best_mre',
        y='best_sd',
        color=param_manager.sort_by_importance[0],           # 用颜色表示 model
        size=size_col_to_use,              # 用点的大小表示 lr
        #symbol=param_manager.sort_by_importance[2],     # 用符号表示 loss_type
        #hover_data=[size_col].append(param_manager.sort_by_importance[3:]),#['loss_basenumber', 'loss_masktype', 'lr', 'IMAGE_SIZE'],  # 悬停显示其他参数
        title='best_mre vs best_sd with Parameter Variations'
        )
elif len(param_manager.sort_by_importance) > 2:
    fig = px.scatter(
        df,
        x='best_mre',
        y='best_sd',
        color=param_manager.sort_by_importance[0],           # 用颜色表示 model
        size=size_col_to_use,              # 用点的大小表示 lr
        symbol=param_manager.sort_by_importance[2],     # 用符号表示 loss_type
        hover_data=[size_col].append(param_manager.sort_by_importance[3:]),#['loss_basenumber', 'loss_masktype', 'lr', 'IMAGE_SIZE'],  # 悬停显示其他参数
        title='best_mre vs best_sd with Parameter Variations'
        )
else:
    raise ValueError("sort_by_importance 列表元素数量不正确")


# 添加映射关系到标题或注释，如果是数值，则添加 size 说明到左上方
if df[size_col].dtype == 'object':
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    mapping_str = " < ".join([f"{k}: {v}" for k, v in mapping.items()])
    annotation_text = f"Size ({size_col}): {mapping_str}"
else:
    annotation_text = f"Size: {size_col}"

fig.update_layout(
    annotations=[
        dict(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0, y=1.05,  # 图表左上方
            showarrow=False,
            font=dict(size=15)  # 小字
        )
    ]
)

fig.update_layout(width=1000, height=600, hovermode='closest')
# 方法 1：保存为静态图片（如 PNG）
fig.write_image(f"{fig_dir}/scatter_plot.png", scale=2)  # scale=2 提高分辨率

# 方法 2：保存为交互式 HTML 文件
fig.write_html(f"{fig_dir}/scatter_plot.html")


#--------------------------------------------------------------------------------80

#################################################################################80
# 可视化
# 平行坐标图 - 全局视角
# a simple one
#from plotly import graph_objects as go
#import pandas as pd
#
## 为分类变量编码为数值（平行坐标图需要数值）
#df_encoded = df.copy()
#for col in ['loss_type', 'optim']:
#    df_encoded[col] = pd.factorize(df[col])[0]
#
## 绘制平行坐标图
#fig = go.Figure(data=go.Parcoords(
#    line=dict(color=df['best_mre'], colorscale='Viridis', showscale=True),
#    dimensions=[
#        dict(label=col, values=df_encoded[col]) for col in df_encoded.columns
#    ]
#))
#fig.update_layout(
#    title='Parallel Coordinates Plot of Parameters and Metrics',
#    width=1200,
#    height=600
#)
#
## 方法 2：保存为交互式 HTML 文件
#fig.write_html(f"{fig_dir}/parallel_coord.html")
#fig.write_image(f"{fig_dir}/parallel_coord.png", scale=2)  # scale=2 提高分辨率
#


#--------------------------------------------------------------------------------80
#################################################################################80
# 绘制散点矩阵图
#import plotly.express as px
#
#fig = px.scatter_matrix(
#    df,
#    dimensions=['loss_basenumber', 'sigma', 'lr', 'best_mre', 'best_sd'],
#    color='model',
#    symbol='loss_type',
#    hover_data=['loss_type', 'loss_masktype', 'IMAGE_SIZE'],
#    title='Scatter Matrix of Parameters and Metrics'
#)
#fig.write_html(f"{fig_dir}/scatter_matrix.html")


#################################################################################80
# 获取最佳试验（按 best_mre 最小化）
best_trial = analysis.get_best_trial(metric="best_mre", mode="min")
best_config = analysis.get_best_config(metric="best_mre", mode="min")
print(f"Best trial config: \t{best_config}")
print(f"Best trial best_mre: \t{best_trial.last_result['best_mre']:.3f} mm")
print(f"Best trial best_sd: \t{best_trial.last_result['best_sd']:.3f} mm")
print(f"the trial id of the best trial: \t{best_trial.trial_id}")
#--------------------------------------------------------------------------------80

#################################################################################80
# ray.results信息获取
def print_top_n_trials(experiment_path, n=5):
    # 初始化 Ray
    if not ray.is_initialized():
        ray.init()
    
    # 加载 Analysis 对象
    analysis = ExperimentAnalysis(experiment_path)
    
    # 获取所有试验，按 best_mre 排序
    trials = sorted(analysis.trials, key=lambda t: t.last_result.get("best_mre", float('inf')))
    
    # 打印前 N 个试验
    print(f"Top {n} trials (sorted by best_mre):")
    for i, trial in enumerate(trials[:n], 1):
        print(f"\nTrial {i}:")
        print(f"  Trial ID: {trial.trial_id}")
        print(f"  Config: {trial.config}")
        print(f"  Results: {trial.last_result}")
        print(f"  Directory: {trial.local_path}")

#print_top_n_trials(experiment_path)  # 打印前 1 项
#--------------------------------------------------------------------------------80

# 可选：关闭 Ray
ray.shutdown()

print(f"- Scatter & Box analysis results saved to {fig_dir}")
