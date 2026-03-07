#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
copy_selected_faa.py

根据 ID 列表，复制源文件夹中对应的 .faa 文件到新文件夹。

用法：
    python3 copy_selected_faa.py \
      --id_list /home/wangjingyuan/wys/WYSPHP/host_cluster.txt \
      --src_dir /home/wangjingyuan/wys/WYSPHP/annotation_out/host \
      --dst_dir /home/wangjingyuan/wys/WYSPHP/annotation_out/host_cluster

参数：
    --id_list   TXT 文件，每行一个 ID (如 GCF_000730245)
    --src_dir   源文件夹，里面有 {ID}.faa 文件
    --dst_dir   目标文件夹，脚本会自动创建
"""

import argparse
import os
import shutil
import sys

def parse_args():
    p = argparse.ArgumentParser(description="根据 ID 列表复制 .faa 文件")
    p.add_argument('--id_list', required=True,
                   help='每行一个 ID 的 TXT 文件 (不带后缀)')
    p.add_argument('--src_dir', required=True,
                   help='源目录，包含 .faa 文件')
    p.add_argument('--dst_dir', required=True,
                   help='目标目录，将复制匹配的 .faa 文件到此')
    return p.parse_args()

def main():
    args = parse_args()

    # 1. 读取 ID 列表
    with open(args.id_list) as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        print(f"Error: 在 {args.id_list} 中未读取到任何 ID", file=sys.stderr)
        sys.exit(1)

    # 2. 创建目标文件夹（如果不存在）
    os.makedirs(args.dst_dir, exist_ok=True)

    # 3. 复制文件
    copied = 0
    for id_ in ids:
        src_path = os.path.join(args.src_dir, f"{id_}.faa")
        if os.path.isfile(src_path):
            dst_path = os.path.join(args.dst_dir, f"{id_}.faa")
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            print(f"Warning: 未找到文件 {src_path}", file=sys.stderr)

    print(f"✅ 复制完成：共复制 {copied}/{len(ids)} 个文件 到 {args.dst_dir}")

if __name__ == "__main__":
    main()
