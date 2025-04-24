import os
import shutil
import random

# 设置原始和目标路径
source_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/data/ISBI2015/RawImage"
target_dir = "/home/jingyu/Projects/python/vscode/Jingyu/Landmark/secondCeph/data/ISBI2015/RawImageNew"

# 获取所有子目录中的图片文件
subdirs = ['Test1Data', 'Test2Data', 'TrainingData']
all_images = {}

for subdir in subdirs:
    path = os.path.join(source_dir, subdir)
    images = [f for f in os.listdir(path) if f.endswith('.bmp')]
    all_images[subdir] = images

# 计算每个子集的原始样本数量
original_counts = {subdir: len(images) for subdir, images in all_images.items()}

# 将所有图片合并到一个列表中
all_image_files = []
for subdir in subdirs:
    for image in all_images[subdir]:
        all_image_files.append((subdir, image))

# 随机打乱所有图片
random.shuffle(all_image_files)

# 按照原始数量重新分配
new_split = {
    'Test1Data': [],
    'Test2Data': [],
    'TrainingData': []
}

# 分配图片到新的子集
start_idx = 0
for subdir in subdirs:
    count = original_counts[subdir]
    new_split[subdir] = all_image_files[start_idx:start_idx + count]
    start_idx += count

# 创建新的目录结构并复制文件
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for subdir in subdirs:
    target_subdir = os.path.join(target_dir, subdir)
    if not os.path.exists(target_subdir):
        os.makedirs(target_subdir)
    
    # 复制文件到新目录
    for orig_subdir, image_file in new_split[subdir]:
        source_path = os.path.join(source_dir, orig_subdir, image_file)
        target_path = os.path.join(target_dir, subdir, image_file)
        shutil.copy2(source_path, target_path)

print("数据集重新划分完成！")
print(f"原始样本数量: {original_counts}")
print(f"新的样本数量: {{'Test1Data': {len(new_split['Test1Data'])}, 'Test2Data': {len(new_split['Test2Data'])}, 'TrainingData': {len(new_split['TrainingData'])}}}")