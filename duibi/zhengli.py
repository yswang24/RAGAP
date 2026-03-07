#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import pandas as pd

def normalize_str(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    return s if s.lower() not in {"", "na", "nan", "none", "null"} else None

def collect_existing_gcf(prok_dir):
    """
    在给定目录中识别存在的 GCF 号：
    - 目录名形如 GCF_XXXXXXXXX
    - 或存在以 GCF_XXXX 开头的 fasta 文件（.fa/.fna/.fasta）
    """
    if not prok_dir:
        return None

    if not os.path.isdir(prok_dir):
        print(f"[WARN] prokaryote 目录不存在：{prok_dir}，忽略存在性校验。", file=sys.stderr)
        return None

    pat = re.compile(r"^(GCF_\d+)")
    valid = set()

    # 1) 目录名
    for name in os.listdir(prok_dir):
        m = pat.match(name)
        if m:
            valid.add(m.group(1))

    # 2) fasta 文件
    for root, _, files in os.walk(prok_dir):
        for fn in files:
            m = pat.match(fn)
            if not m:
                continue
            if fn.endswith((".fa", ".fna", ".fasta")):
                valid.add(m.group(1))
    return valid

def main():
    ap = argparse.ArgumentParser(
        description="将 GCF/taxid TSV 转为边表：src_id dst_id edge_type weight"
    )
    ap.add_argument("--in_tsv", required=True, help="输入 TSV，列包含：GCF, taxid（以及其它列）")
    ap.add_argument("--out_tsv", required=True, help="输出边表 TSV：src_id\tdst_id\tedge_type\tweight")
    ap.add_argument("--prokaryote_dir", default=None,
                    help="可选：若提供，仅保留在该目录中能找到的 GCF（目录名或fasta文件名匹配）")
    ap.add_argument("--edge_type", default="host-taxonomy", help="边类型（默认 host-taxonomy）")
    ap.add_argument("--weight", default="1", help="边权重（默认 1，字符串写回）")
    args = ap.parse_args()

    # 读取
    try:
        df = pd.read_csv(args.in_tsv, sep="\t", dtype=str)
    except Exception as e:
        sys.exit(f"[ERROR] 读取输入失败：{e}")

    # 列名自检与容错
    cols = {c.lower(): c for c in df.columns}
    need = ["gcf", "taxid"]
    for k in need:
        if k not in cols:
            sys.exit(f"[ERROR] 输入缺少列：{k}；现有列：{list(df.columns)}")
    col_gcf = cols["gcf"]
    col_taxid = cols["taxid"]

    # 规范化
    df[col_gcf] = df[col_gcf].apply(normalize_str)
    df[col_taxid] = df[col_taxid].apply(normalize_str)

    # 只保留两列
    df2 = df[[col_gcf, col_taxid]].rename(columns={col_gcf: "src_id", col_taxid: "dst_id"})

    # 过滤空值
    before = len(df2)
    df2 = df2.dropna(subset=["src_id", "dst_id"])
    df2 = df2[(df2["src_id"] != "") & (df2["dst_id"] != "")]
    after_basic = len(df2)

    # 可选：只保留 prokaryote_dir 中存在的 GCF
    valid_gcf = collect_existing_gcf(args.prokaryote_dir)
    if valid_gcf is not None:
        df2 = df2[df2["src_id"].isin(valid_gcf)]
    after_exist = len(df2)

    # 去重
    df2 = df2.drop_duplicates(subset=["src_id", "dst_id"])

    # 填写 edge_type / weight
    df2["edge_type"] = args.edge_type
    df2["weight"] = str(args.weight)

    # 输出
    df2 = df2[["src_id", "dst_id", "edge_type", "weight"]]
    try:
        df2.to_csv(args.out_tsv, sep="\t", index=False)
    except Exception as e:
        sys.exit(f"[ERROR] 写出失败：{e}")

    # 日志
    print(f"[OK] 输入行数: {before}")
    print(f"[OK] 基本清洗后: {after_basic}")
    if valid_gcf is not None:
        print(f"[OK] 存在性校验后: {after_exist}（目录：{args.prokaryote_dir}）")
    print(f"[OK] 输出行数: {len(df2)}")
    if len(df2) == 0:
        print("[WARN] 输出为空，请检查 taxid/GCF 是否缺失或筛选过严。")

if __name__ == "__main__":
    main()
'''
python zhengli.py \
  --in_tsv /home/wangjingyuan/wys/duibi/gcf_check_placeable_but_missing.tsv \
  --out_tsv host_tax_edges_na.tsv

'''