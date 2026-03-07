#!/usr/bin/env python3
import os
import multiprocessing
import subprocess

def clean_fasta_header(line: str) -> str:
    """
    只保留FASTA标题中第一个空格前的部分。
    例：
      >NC_002516.2_1 # 483 # 2027 ... → >NC_002516.2_1
      >AB626963_CDS_[437..559] [note=score:-2.216389E+01] → >AB626963_CDS_[437..559]
    """
    if line.startswith(">"):
        return line.split(" ")[0].strip() + "\n"
    return line

def merge_and_clean_faa(input_dir: str, merged_faa: str):
    """合并所有faa文件，并清理标题格式"""
    faa_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".faa")])
    if not faa_files:
        raise FileNotFoundError("未找到任何 .faa 文件，请检查输入目录。")

    print(f"[INFO] 检测到 {len(faa_files)} 个 .faa 文件，开始合并与清理标题...")
    with open(merged_faa, "w") as out:
        for f in faa_files:
            with open(f, "r") as infile:
                for line in infile:
                    out.write(clean_fasta_header(line))

    print(f"[INFO] 合并完成：{merged_faa}")

def run_cdhit(input_faa: str, output_prefix: str, identity=0.9, memory=64000):
    """运行CD-HIT聚类"""
    # 自动检测CPU核心数
    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    print(f"[INFO] 检测到CPU核心数: {multiprocessing.cpu_count()}，将使用 {cpu_count} 线程运行CD-HIT。")

    cmd = [
        "cd-hit",
        "-i", input_faa,
        "-o", output_prefix,
        "-c", str(identity),     # 相似度阈值
        "-n", "5",               # word size, 对应c=0.9
        "-M", str(memory),       # 最大内存(MB)
        "-T", str(cpu_count),     # 线程数
        "-d","0"
    ]

    print("[INFO] 开始运行CD-HIT聚类...")
    print("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[✅ 完成] CD-HIT 聚类完成。")

def count_sequences(faa_file: str):
    """统计序列数量"""
    count = 0
    with open(faa_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count

if __name__ == "__main__":
    # === 用户配置区域 ===
    input_dir = "/home/wangjingyuan/wys/WYSPHP/annotation_out/phage_final"         # ⚠️ 修改为你的faa目录
    merged_faa = "all_proteins_phage_cleaned.faa"  # 合并输出
    output_prefix = "all_proteins_phage_nr80"      # CD-HIT输出前缀
    identity_threshold = 0.8                 # 聚类相似度
    max_memory = 150000                       # 最大内存(MB)

    # === 执行流程 ===
    merge_and_clean_faa(input_dir, merged_faa)

    seq_count = count_sequences(merged_faa)
    print(f"[统计] 合并后共包含 {seq_count} 条蛋白序列。")

    run_cdhit(merged_faa, output_prefix, identity_threshold, max_memory)

    print(f"[结果] 非冗余结果文件：{output_prefix}.faa")
    print(f"[结果] 聚类信息文件：{output_prefix}.clstr")
