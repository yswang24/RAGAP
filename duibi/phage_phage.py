#!/usr/bin/env python3
"""
convert_phage_phage_edges.py

将形如：
source,target,weight
/path/A.fasta,/path/B.fasta,1.0
的表格转换为：
src_id	dst_id	edge_type	weight
A	B	phage-phage	1
"""

import pandas as pd
import os

# === 输入输出路径 ===
input_file = "/home/wangjingyuan/wys/duibi/work-21-PP-sequence-na/graph_edges.csv"
output_file = "/home/wangjingyuan/wys/duibi/edges_na/phage_phage_edges_na.tsv"

# === 读取输入 ===
df = pd.read_csv(input_file, sep=",", dtype=str)

# === 检查列名 ===
required_cols = ["source", "target", "weight"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ 缺少列: {missing}")

# === 提取 fasta 文件名（去掉路径和扩展名） ===
def extract_id(path):
    """提取文件名去除扩展名"""
    base = os.path.basename(str(path).strip())
    return os.path.splitext(base)[0]

df["src_id"] = df["source"].apply(extract_id)
df["dst_id"] = df["target"].apply(extract_id)

# === 统一 weight 为整数 1 ===
df["weight"] = 1

# === 添加 edge_type ===
df["edge_type"] = "phage-phage"

# === 重新排序列 ===
out_df = df[["src_id", "dst_id", "edge_type", "weight"]]

# === 保存为 TSV ===
out_df.to_csv(output_file, sep="\t", index=False)

print(f"✅ 已保存到: {output_file}")
print(f"共 {len(out_df)} 条边。")
