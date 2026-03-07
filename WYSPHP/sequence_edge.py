#!/usr/bin/env python3
import pandas as pd
import os

# === 输入输出文件路径 ===
INPUT_CSV = "/home/wangjingyuan/wys/create_edge/work-31-HH-sequence/graph_edges.csv"     # 你的输入文件
OUTPUT_TSV = "sequence_sequence_edges.tsv"    # 目标输出文件

# 读取 CSV
df = pd.read_csv(INPUT_CSV)

# 解析 source 和 target -> 去掉路径和扩展名
df["src_id"] = df["source"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
df["dst_id"] = df["target"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

# 固定 edge_type 为 phage-phage
df["edge_type"] = "sequence-sequence"

# weight 保留原始值
df["weight"] = df["weight"]

# 选择列并保存为 TSV
df_out = df[["src_id", "dst_id", "edge_type", "weight"]]
df_out.to_csv(OUTPUT_TSV, sep="\t", index=False)

print(f"✅ 已输出 {OUTPUT_TSV}")
