#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GCF 编号追加到物种表的最后一列：
- 输入1：物种信息 TSV，包含列：accession, superkingdom, phylum, class, order, family, genus, species
- 输入2：分类学 TSV，包含列：taxid, parent_taxid, name, rank, alias
- 逻辑：在分类学表中筛选 rank == 'species' 且 name == 物种名，
       从 alias 中提取所有形如 GCF_XXXXXXXXX 的编号，并写回到输入1的最后一列（列名 GCF_ids）
"""

import re
import argparse
import pandas as pd

GCF_RE = re.compile(r'(GCF_\d{9,})')  # 匹配 GCF_ 后接 9位以上数字（兼容 9-12位等）

def extract_gcf_ids(alias_val):
    """从 alias 字段提取所有 GCF 编号；无则返回空列表。"""
    if pd.isna(alias_val) or str(alias_val).strip() == "":
        return []
    return GCF_RE.findall(str(alias_val))

def main():
    ap = argparse.ArgumentParser(description="根据 species 在 taxonomy TSV 中找到 GCF 编号并追加为最后一列")
    ap.add_argument("--species_tsv", required=True, help="含 species 列的 TSV（将被追加 GCF_ids 列）")
    ap.add_argument("--taxonomy_tsv", required=True, help="含 taxid,parent_taxid,name,rank,alias 的 TSV")
    ap.add_argument("--out", default="with_gcf.tsv", help="输出 TSV 文件名（默认 with_gcf.tsv）")
    ap.add_argument("--case_sensitive", action="store_true",
                    help="物种名匹配是否区分大小写（默认不区分）")
    args = ap.parse_args()

    # 读取数据
    df_sp = pd.read_csv(args.species_tsv, sep="\t", dtype=str, keep_default_na=False)
    df_tx = pd.read_csv(args.taxonomy_tsv, sep="\t", dtype=str, keep_default_na=False)

    # 基本校验
    if "species" not in df_sp.columns:
        raise SystemExit(f"输入 {args.species_tsv} 缺少 'species' 列，实际列：{list(df_sp.columns)}")
    for col in ("name", "rank", "alias"):
        if col not in df_tx.columns:
            raise SystemExit(f"输入 {args.taxonomy_tsv} 缺少 '{col}' 列，实际列：{list(df_tx.columns)}")

    # 仅保留 taxonomy 中的 species 行，并准备一个 name -> GCFs 的映射
    tx_species = df_tx[df_tx["rank"].str.strip().str.lower() == "species"].copy()

    # 统一规范：去首尾空格
    tx_species["name_norm"] = tx_species["name"].str.strip()
    df_sp["species_norm"] = df_sp["species"].str.strip()

    # 大小写处理
    if not args.case_sensitive:
        tx_species["name_key"] = tx_species["name_norm"].str.lower()
        df_sp["species_key"] = df_sp["species_norm"].str.lower()
    else:
        tx_species["name_key"] = tx_species["name_norm"]
        df_sp["species_key"] = df_sp["species_norm"]

    # 从 alias 提取 GCF 列表，并按物种名聚合（如果 taxonomy 里同名 species 多行，合并去重）
    tx_species["gcf_list"] = tx_species["alias"].apply(extract_gcf_ids)
    gcf_map = (
        tx_species.groupby("name_key")["gcf_list"]
        .apply(lambda lists: sorted(set(x for lst in lists for x in lst)))
        .to_dict()
    )

    # 映射回物种表
    def map_species_to_gcf(name_key):
        gcfs = gcf_map.get(name_key, [])
        return ";".join(gcfs) if gcfs else "NA"

    df_sp["GCF_ids"] = df_sp["species_key"].apply(map_species_to_gcf)

    # 清理辅助列，输出
    df_sp.drop(columns=["species_norm", "species_key"], inplace=True, errors="ignore")
    df_sp.to_csv(args.out, sep="\t", index=False)
    print(f"[完成] 已将 GCF_ids 追加到最后一列，共 {len(df_sp)} 行。输出：{args.out}")

if __name__ == "__main__":
    main()
'''
python add_gcf_by_species.py \
  --species_tsv /home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new.tsv \
  --taxonomy_tsv /home/wangjingyuan/wys/duibi/taxonomy_with_alias.tsv \
  --out /home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF.tsv
'''