#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_gcf_ids.py

从一个 TXT 文件（每行一个 FASTA 路径）中提取所有 GCF_* ID（去掉路径和 .fasta 后缀），
并写入新的 TXT 文件，一行一个 ID。

用法：
    python3 extract.py \
      --input /home/wangjingyuan/wys/create_edge/host_representatives_0.95.txt \
      --output host_cluster.txt
"""

import argparse
import os

def parse_args():
    p = argparse.ArgumentParser(
        description="从路径列表中提取 GCF_* ID（basename，无 .fasta）"
    )
    p.add_argument('--input',  '-i', required=True,
                   help='输入 TXT，每行一个 FASTA 文件的完整路径')
    p.add_argument('--output', '-o', required=True,
                   help='输出 TXT，每行一个提取出的 ID（如 GCF_000730245）')
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            path = line.strip()
            if not path:
                continue
            fn = os.path.basename(path)
            # 支持 .fasta 或 .fa 后缀
            for ext in ('.fasta', '.fa'):
                if fn.endswith(ext):
                    fn = fn[:-len(ext)]
                    break
            # 如果想只保留 GCF_ 开头的 ID，可加以下过滤：
            # if not fn.startswith('GCF_'):
            #     continue
            fout.write(fn + "\n")

    print(f"✅ 已从 `{args.input}` 中提取 ID，共写入 `{args.output}`。")

if __name__ == '__main__':
    main()
