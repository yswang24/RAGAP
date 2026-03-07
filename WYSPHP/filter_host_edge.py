
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_host_edge.py

用法:
    python filter_host_edge.py \
        virus_host_taxid.tsv \
        /home/wangjingyuan/wys/create_edge/work-31-HH/graph_edges.csv \
        filtered_host_edges_0.85.csv \
        0.85

参数:
    1) virus_host.tsv      包含 GCF_id 列的 tsv 文件
    2) host_edges.csv      包含 source,target,weight 三列的 csv 文件
    3) filtered_edges.csv  输出文件路径
    4) [可选] 阈值，默认 0.8
"""

import sys
import os
import csv

if len(sys.argv) < 4:
    print("用法: python filter_host_edge_simple.py <tsv_path> <edge_csv_path> <out_csv_path> [threshold]", file=sys.stderr)
    sys.exit(1)

tsv_path      = sys.argv[1]
edge_csv_path = sys.argv[2]
out_path      = sys.argv[3]
threshold     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

# --- 1. 读取 GCF_id 集合 ---
gcf_ids = set()
with open(tsv_path, newline='') as fh:
    reader = csv.DictReader(fh, delimiter='\t')
    if 'GCF_id' not in reader.fieldnames:
        raise KeyError("TSV 文件中没有找到 GCF_id 列")
    for row in reader:
        gcf_ids.add(row['GCF_id'].strip())

# 辅助函数：从文件路径提取 GCF_XXXXXX
def extract_gcf_id(path):
    base = os.path.basename(path)
    if base.startswith("GCF_") and base.endswith(".fasta"):
        return base[:-6]
    return ""

# --- 2. 逐行筛边 ---
with open(edge_csv_path, newline='') as fe, \
     open(out_path, 'w', newline='') as fo:

    reader = csv.DictReader(fe)
    # 保留原始列顺序
    writer = csv.DictWriter(fo, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        # 解析 weight
        try:
            w = float(row['weight'])
        except (KeyError, ValueError):
            # 如果没有 weight 列或转换失败，就跳过
            continue

        if w < threshold:
            continue

        src_id = extract_gcf_id(row.get('source', ''))
        tgt_id = extract_gcf_id(row.get('target', ''))

        # 只要 source 或 target 在我们的 GCF_id 集合里，就保留
        if src_id in gcf_ids or tgt_id in gcf_ids:
            writer.writerow(row)

print(f"✅ 完成筛选，共写入 {out_path}")
