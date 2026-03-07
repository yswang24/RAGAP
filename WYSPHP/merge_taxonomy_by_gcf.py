#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
from pathlib import Path

def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).rename(columns=str.strip)

def main(file1, file2, out_path, multiple_sep=";"):
    # 读取
    df1 = load_tsv(file1)  # virus_taxid, refseq_id, host_taxid, Extracted_GCFs
    df2 = load_tsv(file2)  # domain..species, src_id, taxid

    # 标准化列名（容错）
    colmap1 = {c.lower(): c for c in df1.columns}
    colmap2 = {c.lower(): c for c in df2.columns}

    need1 = ["virus_taxid", "refseq_id", "host_taxid", "extracted_gcfs"]
    need2 = ["domain", "phylum", "class", "order", "family", "genus", "species", "src_id", "taxid"]
    for k in need1:
        if k not in colmap1:
            raise KeyError(f"文件1缺少列: {k}")
    for k in need2:
        if k not in colmap2:
            raise KeyError(f"文件2缺少列: {k}")

    df1 = df1.rename(columns={colmap1["extracted_gcfs"]: "Extracted_GCFs"})
    df2 = df2.rename(columns={colmap2["src_id"]: "src_id"})

    # 拆分 Extracted_GCFs（支持多 GCF 用分号分隔）
    df1["Extracted_GCFs"] = df1["Extracted_GCFs"].fillna("").str.strip()
    # 如果有空行，保留以便后面能看见 unmatched
    df1_exp = df1.assign(Extracted_GCFs=df1["Extracted_GCFs"].str.split(multiple_sep)).explode("Extracted_GCFs")
    df1_exp["Extracted_GCFs"] = df1_exp["Extracted_GCFs"].str.strip()
    # 去掉完全空的 GCF
    df1_exp = df1_exp[df1_exp["Extracted_GCFs"] != ""].copy()

    # 选择并去重 df2 的关键列
    tax_cols = ["domain", "phylum", "class", "order", "family", "genus", "species", "taxid"]
    tax_cols_mapped = [colmap2[c] if c in colmap2 else c for c in tax_cols]
    # 将分类列标准化为小写键对应的原名
    df2 = df2.rename(columns={colmap2.get(c, c): c for c in tax_cols})
    keep_cols = ["src_id"] + tax_cols
    df2 = df2[keep_cols].drop_duplicates()

    # 合并（左连接，保留文件1所有行）
    merged = df1_exp.merge(df2, how="left", left_on="Extracted_GCFs", right_on="src_id")

    # 输出列顺序（你可以按需调整）
    out_cols = [
        colmap1["virus_taxid"],  # 原文件1列名
        colmap1["refseq_id"],
        colmap1["host_taxid"],
        "Extracted_GCFs",
        "domain", "phylum", "class", "order", "family", "genus", "species",
        "taxid"
    ]
    # 某些原列名可能大小写不同，这里确保存在
    out_cols = [c for c in out_cols if c in merged.columns]

    merged[out_cols].to_csv(out_path, sep="\t", index=False)

    # 额外信息：未匹配统计
    unmatched = merged["domain"].isna().sum()
    total = len(merged)
    print(f"完成：输出 {total} 行到 {out_path}")
    if unmatched:
        print(f"注意：有 {unmatched} 行未能在文件2中找到匹配的 src_id（检查 Extracted_GCFs 与 src_id 是否一致、是否有前后空格/版本号差异）。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件2的分类信息按 Extracted_GCFs/src_id 对应整合到文件1。")
    parser.add_argument("--file1", required=True, help="文件1 TSV 路径（含 Extracted_GCFs）")
    parser.add_argument("--file2", required=True, help="文件2 TSV 路径（含 src_id 与分类学列）")
    parser.add_argument("--out", required=True, help="输出 TSV 路径")
    parser.add_argument("--sep", default=";", help="文件1中多 GCF 的分隔符（默认 ';'）")
    args = parser.parse_args()

    main(args.file1, args.file2, args.out, multiple_sep=args.sep)



'''
python merge_taxonomy_by_gcf.py \
  --file1 /home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF.tsv \
  --file2 /home/wangjingyuan/wys/WYSPHP/host_taxonomy_lineage.tsv \
  --out /home/wangjingyuan/wys/WYSPHP/phage_host.tsv
'''