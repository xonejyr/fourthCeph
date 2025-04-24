#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path

def extract_results_from_train(log_file, exp_idx=-1):
    """Extract MRE, SD, and SDR values from the best block in a specific experiment in train.log."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        if not content.strip():
            return None
        
        experiment_start_patterns = [r'Note: NumExpr detected 32 cores but'
            #r'Namespace\(',
            #r'Start Time is \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        ]
        
        start_indices = []
        for pattern in experiment_start_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                start_indices.append(match.start())
        
        start_indices = sorted(set(start_indices))
        if not start_indices:
            experiment_content = content
        else:
            actual_idx = exp_idx if exp_idx >= 0 else len(start_indices) + exp_idx
            if actual_idx < 0 or actual_idx >= len(start_indices):
                return None
            
            exp_start = start_indices[actual_idx]
            exp_end = start_indices[actual_idx + 1] if actual_idx + 1 < len(start_indices) else len(content)
            experiment_content = content[exp_start:exp_end]
        
        if len(experiment_content) < 200:
            return None
        
        validation_blocks = re.findall(
            r'############# Validation Result Epoch (\d+) #############\s*(.*?)(?=(?:#############|\Z))',
            experiment_content, re.DOTALL
        )
        
        if not validation_blocks:
            return None
        
        # 初始化最佳值（使用 float 的最大值）
        best_mre = float('inf')
        best_sd = float('inf')
        best_block = None
        
        # 遍历所有 validation blocks 找出最佳 block
        for epoch, block in validation_blocks:
            mre = re.search(r'MRE:\s*([\d.]+)mm', block)
            sd = re.search(r'SD:\s*([\d.]+)mm', block)
            sdr_2mm = re.search(r'SDR \(2\.0mm\):\s*([\d.]+)%', block)
            sdr_2_5mm = re.search(r'SDR \(2\.5mm\):\s*([\d.]+)%', block)
            sdr_3mm = re.search(r'SDR \(3\.0mm\):\s*([\d.]+)%', block)
            sdr_4mm = re.search(r'SDR \(4\.0mm\):\s*([\d.]+)%', block)
            
            # 如果某个值缺失，跳过这个 block
            if not all([mre, sd, sdr_2mm, sdr_2_5mm, sdr_3mm, sdr_4mm]):
                continue
            
            current_mre = float(mre.group(1))
            current_sd = float(sd.group(1))
            
            # 如果当前 MRE 和 SD 同时小于 best_mre 和 best_sd，更新最佳 block
            if current_mre < best_mre and current_sd < best_sd:
                best_mre = current_mre
                best_sd = current_sd
                best_block = {
                    'epoch': epoch,
                    'mre': current_mre,
                    'sd': current_sd,
                    'sdr_2mm': float(sdr_2mm.group(1)),
                    'sdr_2_5mm': float(sdr_2_5mm.group(1)),
                    'sdr_3mm': float(sdr_3mm.group(1)),
                    'sdr_4mm': float(sdr_4mm.group(1))
                }
        
        # 如果没有找到有效的最佳 block，返回 None
        if best_block is None:
            return None
        
        # 可选：从 block 中提取 best_mean 和 best_sd（如果日志中有这些字段）
        best_mean_match = re.search(r'best mean: ([\d.]+)', validation_blocks[-1][1])
        best_sd_match = re.search(r'best sd: ([\d.]+)', validation_blocks[-1][1])
        if best_mean_match:
            best_block['best_mean'] = float(best_mean_match.group(1))
        if best_sd_match:
            best_block['best_sd'] = float(best_sd_match.group(1))
        
        return best_block
    
    except Exception:
        return None

def extract_results_from_test(log_file, exp_idx=-1):
    """Extract MRE, SD, and SDR values from a specific experiment in test.log."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        if not content.strip():
            print("Log file is empty.")
            return None
        
        # 使用 'Start Time is' 作为实验分段标志
        experiment_start_pattern = r'Start Time is \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        start_indices = [match.start() for match in re.finditer(experiment_start_pattern, content)]
        
        if not start_indices:
            print("No experiment sections found.")
            experiment_content = content  # 如果没有分段，使用整个内容
        else:
            actual_idx = exp_idx if exp_idx >= 0 else len(start_indices) + exp_idx
            if actual_idx < 0 or actual_idx >= len(start_indices):
                print(f"Invalid exp_idx: {exp_idx}, total experiments: {len(start_indices)}")
                return None
            
            exp_start = start_indices[actual_idx]
            exp_end = start_indices[actual_idx + 1] if actual_idx + 1 < len(start_indices) else len(content)
            experiment_content = content[exp_start:exp_end].strip()
        
        # 调试：输出提取的实验内容
        print(f"Extracted experiment content (length={len(experiment_content)}):\n{experiment_content[:200]}...")

        # 提取测试结果块
        test_block = re.search(
            r'############# Test Result #############\s*(.*?)(?=(?:#############|\Z))',
            experiment_content, re.DOTALL
        )
        
        if not test_block:
            print("No test result block found in the extracted content.")
            return None
        
        block = test_block.group(1)
        print(f"Test result block:\n{block}")

        # 优化正则表达式，适应可能的逗号和空格
        mre = re.search(r'MRE:\s*([\d.]+)\s*mm', block)
        sd = re.search(r'SD:\s*([\d.]+)\s*mm', block)
        sdr_2mm = re.search(r'SDR \(2\.0mm\):\s*([\d.]+)%', block)
        sdr_2_5mm = re.search(r'SDR \(2\.5mm\):\s*([\d.]+)%', block)
        sdr_3mm = re.search(r'SDR \(3\.0mm\):\s*([\d.]+)%', block)
        sdr_4mm = re.search(r'SDR \(4\.0mm\):\s*([\d.]+)%', block)
        
        if not all([mre, sd, sdr_2mm, sdr_2_5mm, sdr_3mm, sdr_4mm]):
            print("Failed to match all required fields:")
            print(f"MRE: {mre}, SD: {sd}, SDR 2mm: {sdr_2mm}, SDR 2.5mm: {sdr_2_5mm}, SDR 3mm: {sdr_3mm}, SDR 4mm: {sdr_4mm}")
            return None
        
        return {
            'mre': float(mre.group(1)),
            'sd': float(sd.group(1)),
            'sdr_2mm': float(sdr_2mm.group(1)),
            'sdr_2_5mm': float(sdr_2_5mm.group(1)),
            'sdr_3mm': float(sdr_3mm.group(1)),
            'sdr_4mm': float(sdr_4mm.group(1))
        }
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def generate_latex_table(exp_dir, train_exp_idx=-1, test_exp_idx=-1):
    """Generate LaTeX table from experiment results and display/save formatted results."""
    exp_name = os.path.basename(exp_dir)
    print(f"\n>> Processing experiment directory: {exp_name}")
    
    train_log = os.path.join(exp_dir, 'train.log')
    test_log = os.path.join(exp_dir, 'test.log')
    
    train_results = extract_results_from_train(train_log, train_exp_idx) if os.path.exists(train_log) else None
    test_results = extract_results_from_test(test_log, test_exp_idx) if os.path.exists(test_log) else None
    
    # Default values if no results found
    if not train_results:
        train_results = {'mre': 0.0, 'sd': 0.0, 'sdr_2mm': 0.0, 'sdr_2_5mm': 0.0, 'sdr_3mm': 0.0, 'sdr_4mm': 0.0}
    if not test_results:
        test_results = {'mre': 0.0, 'sd': 0.0, 'sdr_2mm': 0.0, 'sdr_2_5mm': 0.0, 'sdr_3mm': 0.0, 'sdr_4mm': 0.0}
    
    # Generate LaTeX table
    latex_table = r"""\begin{table}[ht]
	\centering
	\caption{Localization results of MRE$\pm$SD and SDR (\percent) on ISBI 2015 dataset}
	\resizebox{\textwidth}{!}{
		\begin{tabular}{@{}lcc|cccccccc@{}}
			\toprule
			\multirow{2}{*}{Exp} & \multicolumn{2}{c|}{MRE$\pm$SD (mm)} & \multicolumn{2}{c}{2 mm} & \multicolumn{2}{c}{2.5 mm} & \multicolumn{2}{c}{3 mm} & \multicolumn{2}{c}{4 mm} \\ 
			\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}
			& Test 1         & Test 2         & Test 1   & Test 2   & Test 1   & Test 2   & Test 1   & Test 2   & Test 1   & Test 2   \\ \midrule
			%s               & %.2f$\pm$%.2f   & %.2f$\pm$%.2f & %.2f    & %.2f    & %.2f    & %.2f    & %.2f    & %.2f   & %.2f    & %.2f    \\
			\bottomrule
		\end{tabular}
	}
\end{table}""" % (
        exp_name,
        train_results['mre'], train_results['sd'],
        test_results['mre'], test_results['sd'],
        train_results['sdr_2mm'], test_results['sdr_2mm'],
        train_results['sdr_2_5mm'], test_results['sdr_2_5mm'],
        train_results['sdr_3mm'], test_results['sdr_3mm'],
        train_results['sdr_4mm'], test_results['sdr_4mm']
    )

    # Create output directory
    output_dir = os.path.join(exp_dir, 'latex_table')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LaTeX table
    output_file = os.path.join(output_dir, 'results_table.tex')
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    # Format and display results
    result_text = f"""Experiment: {exp_name}

    Train experiment index: {train_exp_idx}, Test experiment index: {test_exp_idx}

    Train Results:
    Epoch: {train_results.get('epoch', 'N/A')}
    MRE: {train_results['mre']:.4f} mm
    SD: {train_results['sd']:.4f} mm
    SDR (2.0mm): {train_results['sdr_2mm']:.2f}%
    SDR (2.5mm): {train_results['sdr_2_5mm']:.2f}%
    SDR (3.0mm): {train_results['sdr_3mm']:.2f}%
    SDR (4.0mm): {train_results['sdr_4mm']:.2f}%
    Best Mean: {train_results.get('best_mean', 'N/A')}
    Best SD: {train_results.get('best_sd', 'N/A')}

    Test Results:
    MRE: {test_results['mre']:.4f} mm
    SD: {test_results['sd']:.4f} mm
    SDR (2.0mm): {test_results['sdr_2mm']:.2f}%
    SDR (2.5mm): {test_results['sdr_2_5mm']:.2f}%
    SDR (3.0mm): {test_results['sdr_3mm']:.2f}%
    SDR (4.0mm): {test_results['sdr_4mm']:.2f}%
    """
    print(result_text)
    
    # Save results to file
    details_file = os.path.join(output_dir, 'results_details.txt')
    with open(details_file, 'w') as f:
        f.write(result_text)

    print(f"- Results details saved to {details_file}")
    print(f"- LaTeX table saved to {output_file}")
    print(f">> Completed processing {exp_name}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX table from experiment results')
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--train_exp_idx', type=int, default=-1, help='Train experiment index (-1 for last)')
    parser.add_argument('--test_exp_idx', type=int, default=-1, help='Test experiment index (-1 for last)')
    args = parser.parse_args()
    
    generate_latex_table(args.exp_dir, args.train_exp_idx, args.test_exp_idx)

if __name__ == '__main__':
    main()