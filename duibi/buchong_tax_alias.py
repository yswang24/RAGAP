# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import pandas as pd
# from collections import defaultdict

# LEVELS = [
#     "species",
#     "genus",
#     "family",
#     "order",
#     "class",
#     "phylum",
#     "superkingdom",
# ]

# def split_gcf_cell(x):
#     if pd.isna(x):
#         return []
#     s = str(x).strip()
#     if not s:
#         return []
#     # 兼容多种分隔符：逗号/分号/空白
#     parts = []
#     for token in s.replace(";", " ").replace(",", " ").split():
#         t = token.strip()
#         if t:
#             parts.append(t)
#     return parts

# def normalize_name(x):
#     # 轻度规范化：去两端空格；不改大小写，以免影响学名大小写
#     return str(x).strip() if pd.notna(x) else ""

# def build_level_maps(mapping_df):
#     """
#     将 mapping_df 中每一层级的 name -> set(GCF) 建索引。
#     例如：species_map['Achromobacter xylosoxidans'] = {'GCF_016728825', ...}
#     """
#     level_maps = {lvl: defaultdict(set) for lvl in LEVELS}

#     # 规范化各列名（常见拼写）
#     col_rename = {
#         "superkingdom": "superkingdom",
#         "phylum": "phylum",
#         "class": "class",
#         "order": "order",
#         "family": "family",
#         "genus": "genus",
#         "species": "species",
#         "GCF_ids": "GCF_ids",
#     }
#     mapping_df = mapping_df.rename(columns=col_rename)

#     # 必要列检查
#     need_cols = ["GCF_ids"] + LEVELS
#     for c in need_cols:
#         if c not in mapping_df.columns:
#             raise ValueError(f"源映射文件缺少必要列: {c}")

#     # 逐行收集
#     for _, row in mapping_df.iterrows():
#         gcfs = set(split_gcf_cell(row["GCF_ids"]))
#         if not gcfs:
#             continue
#         for lvl in LEVELS:
#             name = normalize_name(row[lvl])
#             if name:
#                 level_maps[lvl][name].update(gcfs)

#     return level_maps

# def update_alias(existing_alias_cell, gcfs_to_add):
#     # 读取现有 alias；去重；合并新增；再用分号拼接
#     existed = set(split_gcf_cell(existing_alias_cell))
#     before = len(existed)
#     existed.update(gcfs_to_add)
#     after = len(existed)
#     changed = after > before
#     return (";".join(sorted(existed)), changed)

# def main():
#     parser = argparse.ArgumentParser(
#         description="根据物种→GCF 映射，按物种优先、逐级回退写入 taxonomy.tsv 的 alias 列。"
#     )
#     parser.add_argument(
#         "--map_tsv",
#         required=True,
#         help="源映射 TSV，包含列：species、genus、family、order、class、phylum、superkingdom、GCF_ids",
#     )
#     parser.add_argument(
#         "--taxonomy_tsv",
#         required=True,
#         help="待更新的 taxonomy TSV，包含列：taxid, parent_taxid, name, rank[, alias]",
#     )
#     parser.add_argument(
#         "--out_tsv",
#         default=None,
#         help="输出文件路径（默认：在 taxonomy 同目录，文件名加 _with_gcf.tsv）",
#     )
#     args = parser.parse_args()

#     # 读取映射与构建索引
#     map_df = pd.read_csv(args.map_tsv, sep="\t", dtype=str).fillna("")
#     level_maps = build_level_maps(map_df)

#     # 读取 taxonomy
#     tax_df = pd.read_csv(args.taxonomy_tsv, sep="\t", dtype=str).fillna("")
#     for col in ["taxid", "parent_taxid", "name", "rank"]:
#         if col not in tax_df.columns:
#             raise ValueError(f"taxonomy 文件缺少必要列: {col}")

#     if "alias" not in tax_df.columns:
#         tax_df["alias"] = ""

#     updated_rows = 0
#     added_pairs = 0  # 统计新增的 (row, GCF) 数
#     no_hit_counts = 0

#     # 逐行处理 taxonomy
#     for i, row in tax_df.iterrows():
#         target_name = normalize_name(row["name"])
#         if not target_name:
#             continue

#         # 规则：总是先按 species 名称匹配；若无，再按 genus→...→superkingdom
#         gcfs_found = set()
#         for lvl in LEVELS:
#             # 用 taxonomy 的 name 去对应层级的 name 表找
#             gcfs_found = level_maps[lvl].get(target_name, set())
#             if gcfs_found:
#                 break

#         if not gcfs_found:
#             no_hit_counts += 1
#             continue

#         new_alias, changed = update_alias(row.get("alias", ""), gcfs_found)
#         if changed:
#             added_pairs += len(set(split_gcf_cell(new_alias)) - set(split_gcf_cell(row.get("alias", ""))))
#             tax_df.at[i, "alias"] = new_alias
#             updated_rows += 1

#     # 输出
#     if args.out_tsv:
#         out_path = args.out_tsv
#     else:
#         if args.taxonomy_tsv.lower().endswith(".tsv"):
#             out_path = args.taxonomy_tsv[:-4] + "_with_gcf.tsv"
#         else:
#             out_path = args.taxonomy_tsv + "_with_gcf.tsv"

#     tax_df.to_csv(out_path, sep="\t", index=False)

#     print("=== 完成 ===")
#     print(f"输入映射文件: {args.map_tsv}")
#     print(f"输入 taxonomy: {args.taxonomy_tsv}")
#     print(f"输出文件:     {out_path}")
#     print(f"更新行数:     {updated_rows}")
#     print(f"新增 GCF 数:  {added_pairs}")
#     print(f"未命中行数:   {no_hit_counts}")

# if __name__ == "__main__":
#     main()
# '''
# python buchong_tax_alias.py \
#   --map_tsv /home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF_all_updated.tsv \
#   --taxonomy_tsv /home/wangjingyuan/wys/duibi/taxonomy_with_alias.tsv \
#   --out_tsv /home/wangjingyuan/wys/duibi/taxonomy_with_alias_na.tsv

# '''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

LEVELS = ["species","genus","family","order","class","phylum","superkingdom"]

def split_list_cell(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if not s: return []
    parts = []
    for t in s.replace(",", " ").replace(";", " ").split():
        t = t.strip()
        if t:
            parts.append(t)
    return parts

def norm(x):
    return str(x).strip() if pd.notna(x) else ""

def main():
    ap = argparse.ArgumentParser(description="检查 map_tsv 中的 GCF 是否在 taxonomy_tsv 有对应位置（逐级按 name 匹配）。")
    ap.add_argument("--map_tsv", required=True,
                    help="包含列：species, genus, family, order, class, phylum, superkingdom, GCF_ids")
    ap.add_argument("--taxonomy_tsv", required=True,
                    help="包含列：taxid, parent_taxid, name, rank[, alias]")
    ap.add_argument("--out_prefix", default="check_gcf",
                    help="输出文件前缀（默认：check_gcf）")
    args = ap.parse_args()

    # 读入
    map_df = pd.read_csv(args.map_tsv, sep="\t", dtype=str).fillna("")
    tax_df = pd.read_csv(args.taxonomy_tsv, sep="\t", dtype=str).fillna("")
    if "alias" not in tax_df.columns:
        tax_df["alias"] = ""
    if "taxid" not in tax_df.columns:
        raise ValueError("taxonomy 文件缺少必要列: taxid")
    if "name" not in tax_df.columns:
        raise ValueError("taxonomy 文件缺少必要列: name")

    # 构建 name -> 行索引
    name_to_idx = {}
    for i, row in tax_df.iterrows():
        name = norm(row["name"])
        if name and name not in name_to_idx:
            name_to_idx[name] = i

    # alias 预处理为集合
    alias_sets = [set(split_list_cell(a)) for a in tax_df["alias"]]

    unplaceable_rows = []           # 完全找不到位置（含 taxid 空列）
    placeable_but_missing_rows = [] # 有位置但 alias 未包含（含 taxid）

    # 遍历 map_tsv
    for _, r in map_df.iterrows():
        gcfs = split_list_cell(r.get("GCF_ids", ""))
        if not gcfs:
            continue

        best_hit_idx = None
        best_level = None
        best_name = None
        for lvl in LEVELS:
            nm = norm(r.get(lvl, ""))
            if not nm:
                continue
            if nm in name_to_idx:
                best_hit_idx = name_to_idx[nm]
                best_level = lvl
                best_name = nm
                break

        if best_hit_idx is None:
            # 无匹配位置：taxid 留空
            for g in gcfs:
                unplaceable_rows.append({
                    "GCF": g,
                    "reason": "no_matching_taxon_name",
                    "species": norm(r.get("species","")),
                    "genus": norm(r.get("genus","")),
                    "family": norm(r.get("family","")),
                    "order": norm(r.get("order","")),
                    "class": norm(r.get("class","")),
                    "phylum": norm(r.get("phylum","")),
                    "superkingdom": norm(r.get("superkingdom","")),
                    "taxid": "",  # 统一保留列
                })
        else:
            existing = alias_sets[best_hit_idx]
            matched_taxid = str(tax_df.at[best_hit_idx, "taxid"])
            for g in gcfs:
                if g not in existing:
                    placeable_but_missing_rows.append({
                        "GCF": g,
                        "target_taxonomy_name": best_name,
                        "matched_level": best_level,
                        "taxonomy_row_index": best_hit_idx,
                        "taxid": matched_taxid,
                    })

    # 导出（均保证含 taxid 列）
    # 1) unplaceable
    un_cols = ["GCF","reason"] + LEVELS + ["taxid"]
    if unplaceable_rows:
        df_un = pd.DataFrame(unplaceable_rows).drop_duplicates()
        # 补齐缺列（以防万一）
        for c in un_cols:
            if c not in df_un.columns:
                df_un[c] = ""
        df_un = df_un[un_cols]
    else:
        df_un = pd.DataFrame(columns=un_cols)
    df_un.to_csv(f"{args.out_prefix}_unplaceable_gcf.tsv", sep="\t", index=False)

    # 2) placeable_but_missing
    pm_cols = ["GCF","target_taxonomy_name","matched_level","taxonomy_row_index","taxid"]
    if placeable_but_missing_rows:
        df_pm = pd.DataFrame(placeable_but_missing_rows).drop_duplicates()
        for c in pm_cols:
            if c not in df_pm.columns:
                df_pm[c] = ""
        df_pm = df_pm[pm_cols]
    else:
        df_pm = pd.DataFrame(columns=pm_cols)
    df_pm.to_csv(f"{args.out_prefix}_placeable_but_missing.tsv", sep="\t", index=False)

    # 控制台汇总
    print("=== 完成 ===")
    print(f"map_tsv:      {args.map_tsv}")
    print(f"taxonomy_tsv: {args.taxonomy_tsv}")
    print(f"无对应位置 GCF 数: {len(unplaceable_rows)}  -> {args.out_prefix}_unplaceable_gcf.tsv")
    print(f"可定位但未写入alias GCF 数: {len(placeable_but_missing_rows)} -> {args.out_prefix}_placeable_but_missing.tsv")

if __name__ == "__main__":
    main()

'''
python buchong_tax_alias.py \
  --map_tsv /home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF_all_updated.tsv \
  --taxonomy_tsv /home/wangjingyuan/wys/duibi/taxonomy_with_alias.tsv \
  --out_prefix gcf_check

'''