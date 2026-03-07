#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
move_by_gcf.py

根据 TSV 文件中第四列的 GCF_id，从源目录中将对应的 .faa 文件移动到目标目录。

用法：
    python3 move_by_gcf.py \
      --tsv_file /home/wangjingyuan/wys/WYSPHP/virus_host_taxid.tsv \
      --src_dir /home/wangjingyuan/wys/WYSPHP/annotation_out/host \
      --dst_dir /home/wangjingyuan/wys/WYSPHP/annotation_out/host_cluster_phage

参数：
    --tsv_file  输入的 TSV 文件，必须包含列名 'GCF_id'
    --src_dir   源目录，包含 {GCF_id}.faa 文件
    --dst_dir   目标目录，脚本会自动创建
"""

import argparse
import os
import shutil
import sys
import csv

def parse_args():
    p = argparse.ArgumentParser(description="根据 TSV 中的 GCF_id 移动对应的 .faa 文件，已存在则跳过")
    p.add_argument('--tsv_file', required=True,
                   help='输入 TSV 文件，示例字段包括 virus_taxid, refseq_id, host_taxid, GCF_id')
    p.add_argument('--src_dir', required=True,
                   help='源目录，包含 .faa 文件')
    p.add_argument('--dst_dir', required=True,
                   help='目标目录，用于存放移动后的 .faa 文件')
    return p.parse_args()

def main():
    args = parse_args()

    # 1. 读取 TSV 中的 GCF_id 列
    gcf_ids = set()
    with open(args.tsv_file, newline='') as tsvf:
        reader = csv.DictReader(tsvf, delimiter='\t')
        if 'GCF_id' not in reader.fieldnames:
            print("Error: 在 TSV 文件中未找到 'GCF_id' 列，请检查字段名。", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            gid = row['GCF_id'].strip()
            if gid:
                gcf_ids.add(gid)

    if not gcf_ids:
        print("警告：未从 TSV 文件中读取到任何 GCF_id，退出。", file=sys.stderr)
        return

    # 2. 创建目标目录
    os.makedirs(args.dst_dir, exist_ok=True)

    # 3. 移动对应的 .faa 文件（已存在则跳过）
    moved = 0
    skipped = 0
    missing = 0

    for gid in gcf_ids:
        src_file = os.path.join(args.src_dir, f"{gid}.faa")
        dst_file = os.path.join(args.dst_dir, f"{gid}.faa")

        if os.path.exists(dst_file):
            # 如果目标已经有同名文件，跳过
            print(f"Skip: 目标已存在 {dst_file}")
            skipped += 1
            continue

        if os.path.isfile(src_file):
            shutil.move(src_file, dst_file)
            print(f"Moved: {src_file} → {dst_file}")
            moved += 1
        else:
            print(f"Warning: 源文件不存在 {src_file}", file=sys.stderr)
            missing += 1

    print(f"\n✅ 操作完成：")
    print(f"  成功移动: {moved}")
    print(f"  已跳过（已存在）: {skipped}")
    print(f"  源文件缺失: {missing}")

if __name__ == '__main__':
    main()