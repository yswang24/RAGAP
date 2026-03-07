#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def extract_gcf_ids(series):
    """
    从host_gcf列中提取GCF编号，支持:
    - 纯GCF: GCF_016728825
    - 含前后缀: GCF_016728825.1、GCF_016728825_something
    - 多值情况: 用逗号/分号/空格/竖线分隔
    """
    ids = set()
    pat = re.compile(r"GCF_\d+")
    for v in series.dropna().astype(str):
        # 尝试按常见分隔符拆分
        parts = re.split(r"[,\s;|]+", v)
        for p in parts:
            m = pat.search(p)
            if m:
                ids.add(m.group(0))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser(
        description="根据 TSV 的 host_gcf 列，从来源目录复制对应的 parquet 到目标目录"
    )
    ap.add_argument("--tsv", required=True, help="输入的TSV文件路径")
    ap.add_argument("--src", required=True, help="来源目录（存放parquet）")
    ap.add_argument("--dst", required=True, help="目标目录（复制到这里）")
    ap.add_argument(
        "--pattern",
        default="{gcf}*.parquet",
        help="匹配文件模式（相对来源目录）；可用变量 {gcf}。默认: {gcf}*.parquet",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将要复制的文件，不实际复制",
    )
    args = ap.parse_args()

    tsv_path = Path(args.tsv)
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not tsv_path.exists():
        print(f"[错误] TSV不存在: {tsv_path}", file=sys.stderr)
        sys.exit(1)
    if not src_dir.exists():
        print(f"[错误] 来源目录不存在: {src_dir}", file=sys.stderr)
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 读取TSV
    try:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    except Exception as e:
        print(f"[错误] 读取TSV失败: {e}", file=sys.stderr)
        sys.exit(1)

    if "host_gcf" not in df.columns:
        print("[错误] TSV不含列: host_gcf", file=sys.stderr)
        sys.exit(1)

    gcf_ids = extract_gcf_ids(df["host_gcf"])
    if not gcf_ids:
        print("[警告] 未在host_gcf列中提取到任何GCF编号。")
        sys.exit(0)

    print(f"[信息] 去重后的GCF数量: {len(gcf_ids)}")
    total_found = 0
    total_copied = 0
    missing = []
    planned = []

    for gcf in gcf_ids:
        # 允许像 GCF_016728825.parquet / GCF_016728825_anything.parquet / GCF_016728825.1.parquet
        pattern = args.pattern.format(gcf=gcf)
        matches = list(src_dir.glob(pattern))

        if not matches:
            # 再兜底：精确名 GCF_xxx.parquet
            fallback = src_dir / f"{gcf}.parquet"
            if fallback.exists():
                matches = [fallback]

        if not matches:
            missing.append(gcf)
            continue

        total_found += len(matches)
        for src_file in matches:
            dest_file = dst_dir / src_file.name
            planned.append((src_file, dest_file))
            if not args.dry_run:
                try:
                    shutil.copy2(src_file, dest_file)
                    total_copied += 1
                except Exception as e:
                    print(f"[错误] 复制失败: {src_file} -> {dest_file} | {e}", file=sys.stderr)

    # 报告
    print("\n========== 任务报告 ==========")
    print(f"GCF编号总数: {len(gcf_ids)}")
    print(f"匹配到的parquet文件数: {total_found}")
    if args.dry_run:
        print(f"[Dry-Run] 计划复制文件数: {len(planned)}（未实际复制）")
    else:
        print(f"成功复制文件数: {total_copied}")

    if planned:
        print("\n已匹配的文件（来源 -> 目标）:")
        for s, d in planned:
            print(f"- {s}  ->  {d}")

    if missing:
        print("\n未找到对应parquet的GCF编号（请检查文件是否存在或命名是否匹配）:")
        for g in missing:
            print(f"- {g}")

    print("\n完成。")


if __name__ == "__main__":
    main()

'''
# 基本用法：从 /data/parquets/ 复制到 /data/selected/
python copy_gcf_parquets.py \
  --tsv /path/to/edges.tsv \
  --src /data/parquets \
  --dst /data/selected

# 如果你的文件命名是严格的 “GCF_xxx.parquet”，可以改匹配模式，提高精确度
python extract_host_parquet.py \
  --tsv /home/wangjingyuan/wys/duibi/pairs_all.tsv \
  --src /home/wangjingyuan/wys/WYSPHP/dnabert4_host_embeddings_final \
  --dst /home/wangjingyuan/wys/duibi/dnabert4_host_embeddings_final_cherry \
  --pattern "{gcf}.parquet"

# 只看将会复制哪些文件（不真正复制）
python copy_gcf_parquets.py \
  --tsv /path/to/edges.tsv \
  --src /data/parquets \
  --dst /data/selected \
  --dry-run
'''