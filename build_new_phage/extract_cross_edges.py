#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count

def get_basename(f):
    return os.path.splitext(os.path.basename(f))[0]

def run_cmd(cmd):
    """运行 shell 命令并检查错误"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}\n{result.stderr}")
    return result

def ensure_index(sig_dir, index_path):
    """检测或构建索引库 (.sbt.zip)"""
    if os.path.exists(index_path):
        print(f"✅ 检测到索引库: {index_path}")
        return
    sig_files = list(Path(sig_dir).glob("*.sig"))
    if not sig_files:
        raise SystemExit(f"❌ {sig_dir} 中未找到 .sig 文件")
    print(f"⚙️ 正在构建索引库: {index_path} ...")
    cmd = ["sourmash", "index", index_path] + [str(f) for f in sig_files]
    run_cmd(cmd)
    print(f"✅ 索引库构建完成: {index_path}")

def parse_sourmash_output(lines, threshold):
    """解析 sourmash search 输出，返回符合阈值的 (target_name, sim) 列表"""
    hits = []
    float_re = re.compile(r'^\s*(\d+(\.\d+)?)%?\s+(.+)$')
    for line in lines:
        line = line.strip()
        if not line or "matches above threshold" in line:
            continue
        if line.startswith("similarity") or line.startswith("----"):
            continue
        m = float_re.match(line)
        if m:
            sim = float(m.group(1)) / 100.0 if '%' in m.group(0) else float(m.group(1))
            if sim >= threshold:
                name = os.path.basename(m.group(3))
                name = name.replace("cherry_", "")  # 去掉 cherry_ 前缀
                hits.append((name, sim))
    return hits

def sourmash_search(query_sig, db_sig, threshold=0.8):
    cmd = ["sourmash", "search", query_sig, db_sig, "--containment", "--ignore-abundance"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ 警告: sourmash search 失败 {query_sig} vs {db_sig}")
        return []
    lines = result.stdout.strip().splitlines()
    return parse_sourmash_output(lines, threshold)

def process_pair(args):
    """处理单个 (query_sig, db_sig, threshold) 对"""
    q_path, db_path, threshold = args
    q_name = get_basename(q_path).replace("cherry_", "")
    results = []

    hits = sourmash_search(str(q_path), str(db_path), threshold)
    for target, sim in hits:
        dst_name = get_basename(target)  # 去掉 .fasta
        dst_name = dst_name.replace("cherry_", "")
        if q_name == dst_name:
            continue  # 去掉自比
        results.append((q_name, dst_name, "phage-phage", f"{sim:.6f}"))
    return results

def main():
    parser = argparse.ArgumentParser(
        description="⚡ 多进程提取高相似度边（自动索引）：query vs input + query vs query"
    )
    parser.add_argument("--input_sigs", required=True, help="参考 signatures 目录")
    parser.add_argument("--query_sigs", required=True, help="query signatures 目录")
    parser.add_argument("--threshold", required=True, type=float, help="相似度阈值")
    parser.add_argument("--output_tsv", required=True, help="输出 tsv 文件路径")
    parser.add_argument("--threads", type=int, default=0, help="使用线程数（默认自动检测）")
    args = parser.parse_args()

    n_cpus = args.threads or cpu_count()
    print(f"🧠 检测到 {cpu_count()} 核心，使用 {n_cpus} 进程并行计算")

    # ===== 构建索引 =====
    input_index = str(Path(args.input_sigs).resolve() / "input_index.sbt.zip")
    query_index = str(Path(args.query_sigs).resolve() / "query_index.sbt.zip")
    ensure_index(args.input_sigs, input_index)
    ensure_index(args.query_sigs, query_index)

    query_sigs = list(Path(args.query_sigs).glob("*.sig"))
    if not query_sigs:
        raise SystemExit("❌ 未找到 query .sig 文件")
    print(f"🔍 Query 数量: {len(query_sigs)}")

    # ===== 构建任务列表 =====
    all_tasks = []
    for q in query_sigs:
        all_tasks.append((q, input_index, args.threshold))   # query vs input
        all_tasks.append((q, query_index, args.threshold))   # query vs query

    print(f"📦 共 {len(all_tasks)} 个任务待执行")
    print("🚀 开始并行搜索...")

    # ===== 并行计算 =====
    with Pool(processes=n_cpus) as pool, open(args.output_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["src_id", "dst_id", "edge_type", "weight"])

        for i, results in enumerate(pool.imap_unordered(process_pair, all_tasks), 1):
            for row in results:
                writer.writerow(row)
            if i % 10 == 0 or i == len(all_tasks):
                print(f"进度: {i}/{len(all_tasks)} ({i/len(all_tasks)*100:.1f}%)")

    print(f"✅ 结果输出完成: {args.output_tsv}")

if __name__ == "__main__":
    main()
