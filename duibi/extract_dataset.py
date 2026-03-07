# import pandas as pd

# # === 输入输出路径 ===
# input_file = "/home/wangjingyuan/wys/duibi/CHERRY/Interactiondata/TEST_PAIR_TAX.tsv"
# output_file = "TEST_PAIR_TAX_filter.tsv"

# # === 读取 TSV 文件 ===
# df = pd.read_csv(input_file, sep="\t")

# # === 删除第一行（accession 那行表头重复） ===
# df = df.iloc[1:, :]

# # === 去除多余空白 ===
# df.columns = [c.strip() for c in df.columns]

# # === 指定需要的列 ===
# accession_col = "Unnamed: 0"  # accession 对应列
# host_cols = [
#     "hostTaxonomy",
#     "Unnamed: 10",  # superkingdom
#     "Unnamed: 11",  # phylum
#     "Unnamed: 12",  # class
#     "Unnamed: 13",  # order
#     "Unnamed: 14",  # family
#     "Unnamed: 15",  # genus
#    # species（注意比原来多一列，因为前一列是 genus）
# ]

# # 检查列是否存在
# missing = [c for c in [accession_col] + host_cols if c not in df.columns]
# if missing:
#     raise ValueError(f"❌ 缺少列: {missing}")

# # === 只保留需要的列 ===
# df = df[[accession_col] + host_cols]

# # === 重命名列 ===
# df.columns = ["accession", "superkingdom", "phylum", "class", "order", "family", "genus", "species"]

# # === 删除 species 为空的行 ===
# df = df[df["species"].notna() & (df["species"].astype(str).str.strip() != "")]

# # === 输出为 TSV ===
# df.to_csv(output_file, sep="\t", index=False)

# print(f"✅ 已保存筛选后的文件到：{output_file}")
# print(f"保留 {len(df)} 行。")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一个目录中收集所有 .fasta 文件名（去掉扩展名），
在给定 TSV（含 accession 列）中筛选这些 accession 对应的行，输出到新 TSV。
"""

import os
import glob
import argparse
import pandas as pd

def collect_accessions_from_folder(folder, exts=(".fasta", ".fa", ".fna")):
    accs = set()
    for ext in exts:
        for path in glob.glob(os.path.join(folder, f"*{ext}")):
            base = os.path.basename(path)
            name, _ = os.path.splitext(base)  # 例如 NC_021190.fasta -> NC_021190
            if name:
                accs.add(name)
    return accs

def main():
    ap = argparse.ArgumentParser(description="根据文件夹内 FASTA 文件名筛选 TSV 中对应的 accession 行")
    ap.add_argument("--tsv", required=True, help="输入 TSV（需包含 accession 列）")
    ap.add_argument("--fasta_dir", required=True, help="包含若干 .fasta/.fa/.fna 的目录")
    ap.add_argument("--out", default="filtered_by_fasta.tsv", help="输出 TSV 文件名（默认：filtered_by_fasta.tsv）")
    ap.add_argument("--report_missing", action="store_true", help="额外输出未在TSV中找到的 fasta 基名列表")
    args = ap.parse_args()

    # 1) 收集文件夹中的 accession 名称（基名）
    accs_from_files = collect_accessions_from_folder(args.fasta_dir)
    if not accs_from_files:
        raise SystemExit(f"在 {args.fasta_dir} 未找到任何 .fasta/.fa/.fna 文件。")

    # 2) 读取 TSV 并筛选
    df = pd.read_csv(args.tsv, sep="\t", dtype=str, keep_default_na=False)
    if "accession" not in df.columns:
        raise SystemExit(f"输入 TSV 缺少 'accession' 列，实际列：{list(df.columns)}")

    # 统一去空白
    df["accession"] = df["accession"].astype(str).str.strip()

    # 筛选（保留原行顺序）
    mask = df["accession"].isin(accs_from_files)
    df_out = df.loc[mask].copy()

    # 3) 写出
    df_out.to_csv(args.out, sep="\t", index=False)
    print(f"[完成] 已从 {args.tsv} 中筛选出 {len(df_out)} 行，写入 {args.out}")

    # 4) 可选：报告哪些 fasta 基名未在 TSV 中出现
    if args.report_missing:
        accs_in_tsv = set(df["accession"])
        missing = sorted(accs_from_files - accs_in_tsv)
        miss_path = os.path.splitext(args.out)[0] + ".missing.txt"
        with open(miss_path, "w") as f:
            for m in missing:
                f.write(m + "\n")
        print(f"[报告] 未在 TSV 中找到的 {len(missing)} 个 accession 已写入 {miss_path}")

if __name__ == "__main__":
    main()
'''
python extract_dataset.py \
  --tsv /home/wangjingyuan/wys/duibi/VHM_PAIR_TAX_filter_new.tsv \
  --fasta_dir /home/wangjingyuan/wys/duibi/cherry_train_fasta \
  --out /home/wangjingyuan/wys/duibi/VHM_PAIR_TAX_filter_new_na.tsv \
  --report_missing

'''