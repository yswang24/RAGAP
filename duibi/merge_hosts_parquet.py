#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys

TARGET_COLS = ["host_gcf", "sequence_id", "host_species_taxid", "host_dna_emb", "tangent_emb"]

def ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        # 先尝试 JSON 风格
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                return json.loads(s.replace("(", "[").replace(")", "]"))
            except Exception:
                pass
        # 兜底按逗号切
        parts = [p for p in s.strip("[]() ").split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except Exception:
            return parts
    if pd.isna(x):
        return []  # 空设为 []
    return [x]

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # 统一列
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan if col != "tangent_emb" else [[] for _ in range(len(df))]
    # 只保留目标列并排序
    df = df[TARGET_COLS].copy()

    # 基础类型
    df["host_gcf"] = df["host_gcf"].astype(str).str.strip()
    df["sequence_id"] = df["sequence_id"].astype(str).str.strip()
    # taxid 用字符串存（避免大整型溢出/缺失混乱）
    df["host_species_taxid"] = df["host_species_taxid"].astype(str)
    df.loc[df["host_species_taxid"].isin(["nan", "None", ""]), "host_species_taxid"] = np.nan

    # 嵌入列统一为 list
    df["host_dna_emb"] = df["host_dna_emb"].map(ensure_list)
    df["tangent_emb"] = df["tangent_emb"].map(ensure_list)

    return df

def read_many(paths):
    parts = []
    for p in paths:
        df = pd.read_parquet(p)
        df = coerce_schema(df)
        parts.append(df)
    return parts

def main():
    ap = argparse.ArgumentParser(description="合并多个目标格式 parquet（含一个已存在的大文件）为一个总 parquet。")
    ap.add_argument("--dir", required=True, help="目录：内含若干 GCF_*.parquet（目标列格式）")
    ap.add_argument("--big_file", required=True, help="已存在的“大文件”（目标列格式 parquet）")
    ap.add_argument("--out", required=True, help="合并输出 parquet 路径")
    ap.add_argument("--dedup", default="host_gcf,sequence_id", help="去重键，逗号分隔（默认 host_gcf,sequence_id）")
    ap.add_argument("--sortby", default="host_gcf,sequence_id", help="可选：按列排序输出（逗号分隔）")
    args = ap.parse_args()

    dir_path = Path(args.dir).expanduser().resolve()
    big_path = Path(args.big_file).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    # 收集文件
    small_files = sorted([p for p in dir_path.glob("*.parquet")])
    if not small_files:
        print(f"[WARN] 目录中未找到 parquet：{dir_path}", file=sys.stderr)

    # 读取并规范
    parts = []
    if big_path.exists():
        parts += read_many([big_path])
    else:
        print(f"[WARN] big_file 不存在：{big_path}", file=sys.stderr)

    if small_files:
        parts += read_many(small_files)

    if not parts:
        print("[ERROR] 没有可合并的数据。", file=sys.stderr)
        sys.exit(1)

    df_all = pd.concat(parts, ignore_index=True)

    # 去重
    if args.dedup.strip():
        keys = [k.strip() for k in args.dedup.split(",") if k.strip()]
        df_all = df_all.drop_duplicates(subset=keys, keep="first")

    # 排序
    if args.sortby.strip():
        sort_cols = [c.strip() for c in args.sortby.split(",") if c.strip()]
        sort_cols = [c for c in sort_cols if c in df_all.columns]
        if sort_cols:
            df_all = df_all.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_path, index=False)
    print(f"[OK] 合并完成：{out_path}  行数={len(df_all)}  去重键=({args.dedup})")

if __name__ == "__main__":
    main()
'''
python merge_hosts_parquet.py \
  --dir /home/wangjingyuan/wys/duibi/host_parquets0 \
  --big_file /home/wangjingyuan/wys/duibi/host_catalog0.parquet \
  --out /home/wangjingyuan/wys/duibi/host_catalog.parquet \
  --dedup host_gcf,sequence_id \
  --sortby host_gcf,sequence_id
'''