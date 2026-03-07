#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path

def load_tsv(path: Path, key_col: str = "host_gcf") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if key_col not in df.columns:
        raise ValueError(f"{path} 缺少必要列：{key_col}")
    # 规范化 key 列（去空白、统一为字符串、去缺失）
    df[key_col] = (
        df[key_col]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "NA": pd.NA, "NaN": pd.NA})
    )
    df = df.dropna(subset=[key_col])
    return df

def main():
    ap = argparse.ArgumentParser(
        description="基于 host_gcf 求两个TSV交集，并各自输出共有行"
    )
    ap.add_argument("--tsv1", required=True, help="第一个 TSV 路径")
    ap.add_argument("--tsv2", required=True, help="第二个 TSV 路径")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--key", default="host_gcf", help="用于匹配的列名(默认 host_gcf)")
    args = ap.parse_args()

    p1, p2, outdir = Path(args.tsv1), Path(args.tsv2), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df1 = load_tsv(p1, args.key)
    df2 = load_tsv(p2, args.key)

    # 求交集
    s1 = set(df1[args.key])
    s2 = set(df2[args.key])
    common = s1 & s2

    # 过滤并保持原列顺序
    df1_common = df1[df1[args.key].isin(common)].copy()
    df2_common = df2[df2[args.key].isin(common)].copy()

    # 输出文件名
    out1 = outdir / f"{p1.stem}_common_by_{args.key}.tsv"
    out2 = outdir / f"{p2.stem}_common_by_{args.key}.tsv"
    out_ids = outdir / f"common_{args.key}.txt"

    # 保存
    df1_common.to_csv(out1, sep="\t", index=False)
    df2_common.to_csv(out2, sep="\t", index=False)
    # 方便复用：把交集 ID 也写一份
    pd.Series(sorted(common)).to_csv(out_ids, index=False, header=False)

    print(f"文件1原始行数: {len(df1)} | 共有行数: {len(df1_common)} -> {out1}")
    print(f"文件2原始行数: {len(df2)} | 共有行数: {len(df2_common)} -> {out2}")
    print(f"共有 {len(common)} 个唯一 {args.key}，已输出: {out_ids}")

if __name__ == "__main__":
    main()



'''
python extrac_common.py \
  --tsv1 /home/wangjingyuan/wys/duibi/pairs_train.tsv \
  --tsv2 /home/wangjingyuan/wys/duibi/pairs_test.tsv \
  --outdir out_common \
  --key host_gcf
'''