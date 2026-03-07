#!/usr/bin/env python3
import os
import multiprocessing
import subprocess
from tqdm import tqdm  # 如果没有安装tqdm，可以注释掉相关行，或者 pip install tqdm

def merge_and_rename_fastas(input_dir: str, merged_output: str):
    """
    合并文件夹下所有 .fasta 文件。
    关键操作：将 FASTA 标题重命名为文件名（Phage ID），以便追踪。
    例如：/path/MT127620.fasta -> 这里的序列标题会变为 >MT127620
    """
    # 获取所有 fasta 文件
    files = [f for f in os.listdir(input_dir) if f.endswith(".fasta") or f.endswith(".fa") or f.endswith(".fna")]
    files.sort()
    
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 未找到任何 .fasta/.fa/.fna 文件。")

    print(f"[INFO] 检测到 {len(files)} 个噬菌体序列文件，开始合并...")
    print(f"[INFO] 正在合并并重命名 Header 为 Phage ID...")

    with open(merged_output, "w") as out_f:
        # 使用 tqdm 显示进度条
        for filename in tqdm(files, desc="Merging"):
            filepath = os.path.join(input_dir, filename)
            # 提取文件名作为 ID (去除后缀)
            phage_id = os.path.splitext(filename)[0]
            
            with open(filepath, "r") as in_f:
                lines = in_f.readlines()
                if not lines:
                    continue
                
                # 写入新的 Header (>PhageID)
                out_f.write(f">{phage_id}\n")
                
                # 写入序列 (跳过原文件中的 Header 行，只保留序列行)
                for line in lines:
                    if not line.startswith(">"):
                        out_f.write(line)
                
                # 确保每个文件结尾有换行符，防止序列粘连
                if lines and not lines[-1].endswith("\n"):
                    out_f.write("\n")

    print(f"[INFO] 合并完成，输出文件：{merged_output}")

def run_cdhit_est(input_fasta: str, output_prefix: str, identity=0.95, memory=64000):
    """
    运行 cd-hit-est (用于核苷酸序列聚类)
    """
    # 自动检测CPU核心数 (预留1个核心)
    cpu_count = max(1, multiprocessing.cpu_count() - 2)
    print(f"[INFO] 检测到CPU核心数: {multiprocessing.cpu_count()}，将使用 {cpu_count} 线程运行 CD-HIT-EST。")

    # 根据相似度自动调整 word size (-n)
    # cd-hit-est 官方推荐: 0.95->10, 0.9->8, 0.85->6, 0.8->5
    if identity >= 0.95:
        word_size = 10
    elif identity >= 0.9:
        word_size = 8
    else:
        word_size = 5

    cmd = [
        "cd-hit-est",        # ⚠️ 注意：核苷酸序列必须用 cd-hit-est，不能用 cd-hit
        "-i", input_fasta,
        "-o", output_prefix,
        "-c", str(identity), # 相似度阈值
        "-n", str(word_size),# word size
        "-M", str(memory),   # 最大内存(MB)
        "-T", str(cpu_count),# 线程数
        "-d", "0",           # 描述长度，0表示保留完整
        "-aS", "0.9",        # (可选) 较短序列的对齐覆盖度，防止局部匹配导致的错误聚类
        "-g", "1"            # (可选) 将序列聚类到最相似的簇，更准确但稍慢
    ]

    print(f"[INFO] 开始运行 CD-HIT-EST 聚类 (Identity={identity}, WordSize={word_size})...")
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[✅ 完成] 聚类成功！")
    except FileNotFoundError:
        print("[❌ 错误] 未找到 'cd-hit-est' 命令。请确保已安装 CD-HIT 并添加到环境变量。")
        print("安装命令参考: conda install -c bioconda cd-hit")
    except subprocess.CalledProcessError as e:
        print(f"[❌ 错误] CD-HIT 运行失败: {e}")

def count_sequences(fasta_file: str):
    """统计合并后的序列数量"""
    count = 0
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count

if __name__ == "__main__":
    # === 用户配置区域 ===
    # 您的输入文件夹
    INPUT_DIR = "/home/wangjingyuan/wys/phage_fasta_final"
    
    # 输出目录 (建议放在 processed 文件夹下)
    OUTPUT_DIR = "clustering_result_jiqun"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 合并后的文件名
    MERGED_FASTA = os.path.join(OUTPUT_DIR, "all_phages_merged.fasta")
    
    # 聚类输出前缀
    OUTPUT_PREFIX = os.path.join(OUTPUT_DIR, "phages_clustered_0.95")
    
    # 聚类参数
    IDENTITY_THRESHOLD = 0.95  # ⚠️ 顶刊去冗余通常设为 0.95 (95% ANI)
    MAX_MEMORY = 100000        # 100GB 内存限制
    
    # === 执行流程 ===
    # 1. 合并并重命名
    merge_and_rename_fastas(INPUT_DIR, MERGED_FASTA)

    # 2. 统计数量
    seq_count = count_sequences(MERGED_FASTA)
    print(f"[统计] 共有 {seq_count} 条噬菌体序列参与聚类。")

    # 3. 运行聚类
    run_cdhit_est(MERGED_FASTA, OUTPUT_PREFIX, IDENTITY_THRESHOLD, MAX_MEMORY)

    # 4. 结果指引
    print("\n" + "="*30)
    print("📊 结果文件说明：")
    print(f"1. 代表性序列 (非冗余集): {OUTPUT_PREFIX}")
    print(f"2. 聚类详细列表 (.clstr):  {OUTPUT_PREFIX}.clstr")
    print("="*30)
    print("下一步建议：请解析 .clstr 文件，将同一簇内的噬菌体划分到同一个数据集(Train/Test)中。")