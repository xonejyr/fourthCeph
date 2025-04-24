import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import ray
from ray.tune.analysis import ExperimentAnalysis
import os

import yaml

#################################################################################80
# 若要使用，有以下基础要修改
# 1. 绝对路径设置: experiment_path, fig_dir
# 2. 搜索参数列表: config_cols
# 3. 类别/数值变量: cat_columns, num_columns
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
#experiment_path = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_coord_DHDN-256x256_DHDN_ce_coord/params_search/metadata_for_search"  # 替换为你的实验路径
##print_top_n_trials(experiment_path)  # 打印前 1 项
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
# ["NUM_PATTERNS", "HIDDEN_DIM", "lr", "BETA", "GAMMA"]
#config_cols = ['config/NUM_PATTERNS', 'config/HIDDEN_DIM', 'config/lr', 'config/BETA', 'config/GAMMA']
config_cols = [f"config/{col}" for col in param_manager.get_parameter_columns()]
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

df1 = df.sort_values(by='SIGMA', ascending=True, inplace=False)
print(df1.head(50))  # 检查数据
#--------------------------------------------------------------------------------80

#################################################################################80
# 分离自变量和因变量并可视化
X = df.drop(columns=['best_mre', 'best_sd'])  # 自变量
y_mre = df['best_mre']  # 因变量 1
y_sd = df['best_sd']    # 因变量 2

# 在训练模型前清理数据
#mask = ~X.isna().any(axis=1) & ~y_mre.isna()  # 确保 X 和 y_mre 都没有 NaN
#X_clean = X.loc[mask]
#y_mre_clean = y_mre.loc[mask]
#y_sd_clean = y_sd.loc[mask]

# 对分类变量进行编码
#cat_columns = []  # 分类变量列
#num_columns = ["NUM_PATTERNS", "HIDDEN_DIM", "lr", "BETA", "GAMMA"] # 数值变量列
cat_columns, num_columns = param_manager.infer_types()
#print(cat_columns, num_columns)

# Label Encoding（适用于随机森林，简单高效）
le = LabelEncoder()
for col in cat_columns:
    X[col] = le.fit_transform(X[col])

# 训练随机森林模型并提取特征重要性
def get_feature_importance(X, y, feature_names):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = rf.feature_importances_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# 获取两个因变量的特征重要性
feature_names = X.columns
imp_mre = get_feature_importance(X, y_mre, feature_names)
imp_sd = get_feature_importance(X, y_sd, feature_names)


# 按重要性排序
imp_mre = imp_mre.sort_values(by='Importance', ascending=False)
imp_sd = imp_sd.sort_values(by='Importance', ascending=False)

# 可视化
plt.figure(figsize=(12, 6))

# best_mre 的特征重要性图
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=imp_mre, palette='viridis')
plt.title('Feature Importance for best_mre', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# best_sd 的特征重要性图
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=imp_sd, palette='viridis')
plt.title('Feature Importance for best_sd', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)

plt.tight_layout()
plt.savefig(f"{fig_dir}/feature_importance.png")



# 打印重要性排序
print("Feature Importance for best_mre:")
print(imp_mre)
print("\nFeature Importance for best_sd:")
print(imp_sd)


# 保存排序后的键到配置文件
sorted_features = imp_mre.sort_values(by="Importance", ascending=False)["Feature"].tolist()
# 获取类型信息
cat_columns, num_columns = param_manager.infer_types()
num_features = [col for col in num_columns]

# 修改处 1：检查 num_imp 是否为空
num_imp = imp_mre[imp_mre["Feature"].isin(num_features)].sort_values(by="Importance", ascending=False)
if not num_imp.empty:  # 如果有数值型特征
    max_num_feature = num_imp.iloc[0]["Feature"]
    # 修改处 2：调整第二个位置为 max_num_feature
    if len(sorted_features) > 1:  # 只有当长度大于1时才调整
        if max_num_feature in sorted_features:
            if sorted_features[1] != max_num_feature:
                sorted_features.remove(max_num_feature)
                sorted_features.insert(1, max_num_feature)
        else:
            print(f"警告：max_num_feature '{max_num_feature}' 不在 sorted_features 中，跳过调整")
    # 长度为1时无需调整，不输出警告
else:
    print("警告：没有检测到数值型特征，跳过 max_num_feature 调整")
    max_num_feature = None


config_file="./scripts/utils/shared_params.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['sort_by_importance'] = sorted_features

# 写回 YAML 文件
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Sorted features saved to {config_file}")

f.close()
print(f"- Importance analysis results saved to {fig_dir}")
#--------------------------------------------------------------------------------80