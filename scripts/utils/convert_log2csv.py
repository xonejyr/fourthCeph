import re
import os
import pandas as pd

def log_to_csv(log_path, csv_path):
    # 初始化列表来存储数据
    data = {
        'epoch': [],
        'loss': [],
        'mre': [],
        'sd': []
    }
    
    # 读取log文件
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # 提取每个epoch的训练loss
    train_loss_matches = re.finditer(r'Train-(\d+) epoch \| loss:([\d.]+)', log_content)
    for match in train_loss_matches:
        epoch = int(match.group(1))
        loss = float(match.group(2))
        data['epoch'].append(epoch)
        data['loss'].append(loss)
        # 先填充空的mre和sd
        data['mre'].append('')
        data['sd'].append('')
    
    # 提取验证结果中的MRE和SD
    validation_matches = re.finditer(r'############# Validation Result Epoch (\d+) #############.*?'
                                   r'MRE:\s*([\d.]+)mm,.*?'
                                   r'SD:\s*([\d.]+)mm', 
                                   log_content, re.DOTALL)
    
    for match in validation_matches:
        epoch = int(match.group(1))
        mre = float(match.group(2))
        sd = float(match.group(3))
        
        # 找到对应的epoch索引
        if epoch in data['epoch']:
            idx = data['epoch'].index(epoch)
            data['mre'][idx] = mre
            data['sd'][idx] = sd
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 保存到CSV文件，不包含sdr列
    df.to_csv(csv_path, index=False)
    print(f"CSV file has been saved to: {csv_path}")

def convert_file(log_path, csv_path=None):
    # 如果没有指定csv_path，使用与log文件相同的文件名但更改扩展名
    if csv_path is None:
        csv_path = os.path.splitext(log_path)[0] + '.csv'
    
    log_to_csv(log_path, csv_path)

def convert_directory(log_dir, output_dir=None):
    """
    转换指定目录下所有的log文件到CSV
    
    Args:
        log_dir (str): 包含log文件的输入目录
        output_dir (str): 输出CSV文件的目录，如果未指定则在log_dir下创建history_plot文件夹
    """
    # 如果未指定输出目录，默认在输入目录下创建history_plot文件夹
    if output_dir is None:
        output_dir = os.path.join(log_dir, 'history_plot')
    
    # 确保输入目录存在
    if not os.path.exists(log_dir):
        raise ValueError(f"Input directory does not exist: {log_dir}")
    
    # 遍历目录中的所有文件
    for filename in os.listdir(log_dir):
        # 只处理.log文件
        if filename.lower().endswith('.log'):
            log_path = os.path.join(log_dir, filename)
            # 生成输出文件名，保持原文件名但更改扩展名为.csv
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            
            try:
                log_to_csv(log_path, csv_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# 使用示例
if __name__ == "__main__":
    ## 你可以这样调用：
    #log_file = "path/to/your/logfile.log"
    #csv_file = "path/to/your/output.csv"
    #
    ## 指定两个路径
    #convert_file(log_file, csv_file)

    # 使用示例1：指定输入和输出目录
    root_dir = "/mnt/home_extend/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_ResFPN-256x256_ResFPN_ce_heatmap/params_search-ResNet_SIZE_LR_TYPE/"
    log_directory = os.path.join(root_dir, "logs")
    output_directory =  os.path.join(root_dir, "history_csv")
    convert_directory(log_directory, output_directory)
    
    # 或者只指定log路径，csv会自动生成同名文件
    # convert_file(log_file)