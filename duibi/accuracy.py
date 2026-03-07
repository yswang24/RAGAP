# import pandas as pd

# # 读取文件
# #pred_df = pd.read_csv("phage_prediction_results_test_topk.tsv", sep="\t")
# #pred_df = pd.read_csv("/home/wangjingyuan/wys/wys_shiyan/phage_prediction_results_epoch_2000.tsv", sep="\t")
# pred_df = pd.read_csv("/home/wangjingyuan/wys/wys_shiyan/best_now/phage_prediction_results_test_topk.tsv", sep="\t")


# true_df = pd.read_csv("phage.tsv", sep="\t")


# # 只保留 rank=1 的预测
# top1_df = pred_df[pred_df['rank'] == 1]

# # 合并预测和真实标签
# merged = pd.merge(top1_df, true_df[['phage_id', 'species']], on='phage_id', how='inner')

# # 判断预测是否正确
# merged['correct'] = merged['host_species'] == merged['species']

# # 计算hit@1准确率
# accuracy = merged['correct'].mean()
# print(f"Hit@1 accuracy: {accuracy:.4f} ({merged['correct'].sum()}/{len(merged)})")

# # 打印预测错误的 phage_id 及其预测/真实结果
# wrong = merged[~merged['correct']]
# if not wrong.empty:
#     print("\n预测错误的 phage 列表:")
#     print(wrong[['phage_id', 'host_species', 'species']].to_string(index=False))
# else:
#     print("\n所有预测都正确 ✅")









#!/usr/bin/env python3
"""
compute_hitk.py

计算 species & genus 级别的 Hit@k。

Usage:
    python accuracy.py --pred /home/wangjingyuan/wys/wys_shiyan/output-train-GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg70_evl20_drop0.2_1e-5_cos_new_40000_True_noleak_study2/predictions/phage_prediction_results_test_topk.tsv --taxonomy /home/wangjingyuan/wys/wys_shiyan/taxonomy_with_alias.tsv --truth phage_host.tsv --k 1 3 5 10 20\
        --out_prefix accuracy_result

输出:
    result_hit_results.tsv   # 每个 phage 的命中明细
    result_hit_summary.tsv   # hit@k 汇总（species & genus）

"""

# import argparse
# import pandas as pd
# import sys
# from collections import defaultdict
# from typing import Optional, Dict, Tuple, List


# def read_prediction(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path, sep=None, engine='python', dtype=str)  # 自动检测分隔符
#     # 强制列名小写便于处理
#     df.columns = [c.strip() for c in df.columns]
#     return df


# def read_taxonomy(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path, sep=None, engine='python', dtype=str)
#     df.columns = [c.strip() for c in df.columns]
#     # Ensure required columns exist
#     for col in ['taxid', 'parent_taxid', 'name', 'rank', 'alias']:
#         if col not in df.columns:
#             raise ValueError(f"taxonomy file must contain column '{col}' (found: {df.columns.tolist()})")
#     return df


# def read_truth(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path, sep=None, engine='python', dtype=str)
#     df.columns = [c.strip() for c in df.columns]
#     # Expect refseq_id and genus/species columns
#     for col in ['refseq_id', 'genus', 'species']:
#         if col not in df.columns:
#             raise ValueError(f"truth file must contain column '{col}' (found: {df.columns.tolist()})")
#     return df


# def build_taxonomy_maps(tax_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
#     """
#     Build:
#       - alias_to_taxid: map alias (e.g. GCF_006539325) -> taxid (string)
#       - taxid_info: map taxid -> {'parent_taxid':..., 'name':..., 'rank':...}
#     """
#     alias_to_taxid: Dict[str, str] = {}
#     taxid_info: Dict[str, Dict[str, str]] = {}

#     for _, row in tax_df.iterrows():
#         taxid = str(row['taxid']).strip()
#         parent = str(row['parent_taxid']).strip() if pd.notna(row['parent_taxid']) else ''
#         name = str(row['name']).strip() if pd.notna(row['name']) else ''
#         rank = str(row['rank']).strip() if pd.notna(row['rank']) else ''
#         alias = str(row['alias']).strip() if pd.notna(row['alias']) else ''

#         taxid_info[taxid] = {'parent_taxid': parent, 'name': name, 'rank': rank}
#         if alias and alias != 'nan':
#             # Some alias fields might contain multiple aliases separated by ';' or ',' - split conservatively
#             for a in [x.strip() for x in alias.replace(';', ',').split(',') if x.strip()]:
#                 alias_to_taxid[a] = taxid

#     return alias_to_taxid, taxid_info


# def find_genus_from_taxid(taxid: str, taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
#     """
#     Walk up the taxonomy tree from taxid to find the first node with rank == 'genus'.
#     Returns genus name or None if not found.
#     """
#     visited = set()
#     current = taxid
#     while current and current not in visited:
#         visited.add(current)
#         info = taxid_info.get(current)
#         if not info:
#             return None
#         if info.get('rank', '').lower() == 'genus':
#             name = info.get('name', '').strip()
#             return name if name else None
#         # move up
#         parent = info.get('parent_taxid', '')
#         if parent == '' or parent == current:
#             return None
#         current = parent
#     return None


# def map_hostid_to_genus(host_id: str, alias_to_taxid: Dict[str, str], taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
#     """
#     Given host_id (e.g., 'GCF_006539325'), try to find genus via taxonomy maps.
#     If not found, return None.
#     """
#     if pd.isna(host_id) or host_id == '':
#         return None
#     host_id = str(host_id).strip()
#     # direct alias lookup
#     taxid = alias_to_taxid.get(host_id)
#     if taxid:
#         genus = find_genus_from_taxid(taxid, taxid_info)
#         if genus:
#             return genus
#     # sometimes alias might be embedded (e.g., multiple GCFs in a cell) -> try splitting
#     for sep in [';', ',', '|', ' ']:
#         if sep in host_id:
#             parts = [p.strip() for p in host_id.split(sep) if p.strip()]
#             for p in parts:
#                 t = alias_to_taxid.get(p)
#                 if t:
#                     g = find_genus_from_taxid(t, taxid_info)
#                     if g:
#                         return g
#     return None


# def extract_genus_from_species_str(species_str: str) -> Optional[str]:
#     """Fallback: from a species string like 'Streptococcus pyogenes' extract 'Streptococcus'."""
#     if pd.isna(species_str) or species_str == '':
#         return None
#     parts = str(species_str).strip().split()
#     if len(parts) >= 1:
#         return parts[0]
#     return None


# def prepare_prediction(pred_df: pd.DataFrame) -> pd.DataFrame:
#     # Normalize column names for robust access (lowercase)
#     cols = {c.lower(): c for c in pred_df.columns}
#     # required columns: phage_id, host_id OR host_species; rank OR score
#     if 'phage_id' not in cols:
#         raise ValueError("prediction file must contain 'phage_id' column")
#     # Keep original casing but provide convenient column names
#     pred_df = pred_df.copy()
#     pred_df['phage_id'] = pred_df[cols['phage_id']].astype(str).str.strip()
#     # host_id may be missing in some datasets but try
#     host_id_col = cols.get('host_id')
#     host_species_col = cols.get('host_species') or cols.get('host')
#     if host_id_col:
#         pred_df['host_id_raw'] = pred_df[host_id_col].astype(str).str.strip()
#     else:
#         pred_df['host_id_raw'] = pd.NA

#     if host_species_col:
#         pred_df['host_species_raw'] = pred_df[host_species_col].astype(str).str.strip()
#     else:
#         pred_df['host_species_raw'] = pd.NA

#     # ranking: if there's 'rank' column numeric, use it; else if 'score' exists, we sort by score desc
#     if 'rank' in cols:
#         # ensure rank numeric for sorting
#         try:
#             pred_df['rank_val'] = pd.to_numeric(pred_df[cols['rank']], errors='coerce')
#         except Exception:
#             pred_df['rank_val'] = pd.NA
#     else:
#         pred_df['rank_val'] = pd.NA

#     if 'score' in cols:
#         pred_df['score_val'] = pd.to_numeric(pred_df[cols['score']], errors='coerce')
#     else:
#         pred_df['score_val'] = pd.NA

#     return pred_df


# def compute_hitk_for_phage(preds: List[Dict], true_species: Optional[str], true_genus: Optional[str], ks: List[int]) -> Dict[str, int]:
#     """
#     preds: ordered list of predictions (each is dict with keys 'pred_species', 'pred_genus')
#     returns dict like {'hit_species@1':0/1, 'hit_species@3':0/1, ... , 'hit_genus@1':0/1, ...}
#     """
#     results = {}
#     total = len(preds)
#     # build lists
#     pred_species_list = [p.get('pred_species') for p in preds]
#     pred_genus_list = [p.get('pred_genus') for p in preds]

#     for k in ks:
#         topk_s = pred_species_list[:k]
#         topk_g = pred_genus_list[:k]

#         if true_species is None or true_species == '' or pd.isna(true_species):
#             # if true species missing -> define as not hit (0)
#             results[f'hit_species@{k}'] = 0
#         else:
#             # exact string match (case-insensitive)
#             hit_s = any((ts := (str(true_species).strip())).lower() == str(x).strip().lower() for x in topk_s if pd.notna(x))
#             results[f'hit_species@{k}'] = 1 if hit_s else 0

#         if true_genus is None or true_genus == '' or pd.isna(true_genus):
#             results[f'hit_genus@{k}'] = 0
#         else:
#             hit_g = any((tg := (str(true_genus).strip())).lower() == str(x).strip().lower() for x in topk_g if pd.notna(x))
#             results[f'hit_genus@{k}'] = 1 if hit_g else 0

#     return results


# def main(args):
#     print("Reading files...")
#     pred_df_raw = read_prediction(args.pred)
#     tax_df = read_taxonomy(args.taxonomy)
#     truth_df = read_truth(args.truth)

#     # normalize inputs
#     pred_df = prepare_prediction(pred_df_raw)
#     alias_to_taxid, taxid_info = build_taxonomy_maps(tax_df)

#     # Build truth map: refseq_id -> (genus, species)
#     truth_map: Dict[str, Tuple[str, str]] = {}
#     for _, r in truth_df.iterrows():
#         ref = str(r['refseq_id']).strip()
#         g = str(r['genus']).strip() if pd.notna(r['genus']) else ''
#         s = str(r['species']).strip() if pd.notna(r['species']) else ''
#         truth_map[ref] = (g, s)

#     # Determine ordering in predictions: prefer 'rank_val' ascending; if not available use 'score_val' desc
#     use_rank = pred_df['rank_val'].notna().any()
#     use_score = pred_df['score_val'].notna().any()

#     if use_rank:
#         # ensure numerical rank; if missing for some rows, coerce to large number
#         pred_df['rank_val'] = pd.to_numeric(pred_df['rank_val'], errors='coerce')
#         pred_df['rank_val_filled'] = pred_df['rank_val'].fillna(pred_df['rank_val'].max() + 1000 if pred_df['rank_val'].notna().any() else 1e9)
#         sort_ascending = True
#         sort_keys = ['phage_id', 'rank_val_filled']
#     elif use_score:
#         pred_df['score_val'] = pd.to_numeric(pred_df['score_val'], errors='coerce')
#         # larger score == better
#         pred_df['score_val_filled'] = pred_df['score_val'].fillna(-1e9)
#         sort_ascending = False
#         # we'll group by phage and sort per group by score descending
#         sort_keys = None
#     else:
#         # no rank or score -> keep file order (grouped)
#         sort_keys = ['phage_id']
#         sort_ascending = True

#     # Create per-row predicted species/genus:
#     # pred_species: prefer host_species_raw if present; else try to resolve from host_id via taxonomy (by looking up taxid->name when rank=species)
#     # pred_genus: try taxonomy lookup via host_id alias -> genus; fallback to parsing host_species_raw first token
#     pred_aug_rows = []
#     # If we want to preserve all columns, we can copy them; for clarity we build a new df
#     for _, row in pred_df.iterrows():
#         host_id_raw = row.get('host_id_raw') if 'host_id_raw' in row else None
#         host_species_raw = row.get('host_species_raw') if 'host_species_raw' in row else None

#         # predicted species: host_species_raw if present and not 'nan'
#         pred_species = None
#         if pd.notna(host_species_raw) and str(host_species_raw).strip() and str(host_species_raw).strip().lower() != 'nan':
#             pred_species = str(host_species_raw).strip()

#         # predicted genus: try mapping host_id -> genus via taxonomy first
#         pred_genus = None
#         if pd.notna(host_id_raw) and str(host_id_raw).strip() and str(host_id_raw).strip().lower() != 'nan':
#             pred_genus = map_hostid_to_genus(str(host_id_raw).strip(), alias_to_taxid, taxid_info)

#         # fallback: if genus still None, try extract from host_species_raw
#         if pred_genus is None and pred_species:
#             pred_genus = extract_genus_from_species_str(pred_species)

#         # also fallback: if pred_species None but host_id maps to a taxid that has rank==species, we can use that name:
#         if pred_species is None and pd.notna(host_id_raw) and str(host_id_raw).strip():
#             taxid = alias_to_taxid.get(str(host_id_raw).strip())
#             if taxid:
#                 info = taxid_info.get(taxid)
#                 if info and info.get('rank', '').lower() == 'species' and info.get('name'):
#                     pred_species = info.get('name')

#         # keep original row fields for later
#         new_row = dict(row)
#         new_row['pred_species'] = pred_species
#         new_row['pred_genus'] = pred_genus
#         new_row['host_id_raw'] = host_id_raw
#         new_row['host_species_raw'] = host_species_raw
#         pred_aug_rows.append(new_row)

#     pred_aug_df = pd.DataFrame(pred_aug_rows)

#     # For grouping and ordering, build groups
#     if sort_keys:
#         pred_aug_df = pred_aug_df.sort_values(by=sort_keys, ascending=sort_ascending, kind='mergesort')
#     else:
#         # sort by phage_id and score_val desc
#         pred_aug_df = pred_aug_df.sort_values(by=['phage_id', 'score_val_filled'], ascending=[True, False], kind='mergesort')

#     # Build per-phage ordered list of predictions
#     phage_groups = pred_aug_df.groupby('phage_id', sort=True)

#     ks = sorted([int(x) for x in args.k])
#     ks = [int(x) for x in ks if int(x) >= 1]

#     detailed_rows = []
#     n_phages_with_label = 0
#     # Counters for summary
#     summary_counts = defaultdict(int)  # e.g., 'hit_species@1' -> sum
#     phage_total = 0

#     for phage_id, group in phage_groups:
#         phage_total += 1
#         # produce ordered list of preds
#         preds_ordered = []
#         # order preserving already due to sorting; take rows in order
#         for _, r in group.iterrows():
#             preds_ordered.append({'pred_species': r.get('pred_species'), 'pred_genus': r.get('pred_genus'),
#                                   'host_id_raw': r.get('host_id_raw'), 'host_species_raw': r.get('host_species_raw')})

#         # find true labels from truth_map using phage_id -> refseq_id
#         true_genus = None
#         true_species = None
#         if phage_id in truth_map:
#             true_genus, true_species = truth_map[phage_id]
#             n_phages_with_label += 1
#         else:
#             # maybe phage_id equals refseq id with slight differences (strip version suffix); try some fallbacks
#             # fallback 1: if phage_id contains '|' or whitespace and truth_map has one of parts
#             found = False
#             if isinstance(phage_id, str):
#                 for sep in ['|', ' ', ';', ',']:
#                     if sep in phage_id:
#                         for part in phage_id.split(sep):
#                             p = part.strip()
#                             if p in truth_map:
#                                 true_genus, true_species = truth_map[p]
#                                 found = True
#                                 break
#                         if found:
#                             break
#             # else leave None

#         # compute hit flags
#         hit_flags = compute_hitk_for_phage(preds_ordered, true_species, true_genus, ks)

#         # accumulate summary counts only for phages that have a true label
#         if true_genus is not None or true_species is not None:
#             for k in ks:
#                 summary_counts[f'hit_species@{k}'] += hit_flags[f'hit_species@{k}']
#                 summary_counts[f'hit_genus@{k}'] += hit_flags[f'hit_genus@{k}']
#         # else: do not count towards denominators of Hit@k -- we'll only compute Hit@k over phages with known true labels

#         row = {
#             'phage_id': phage_id,
#             'true_genus': true_genus,
#             'true_species': true_species,
#             'n_predictions': len(preds_ordered)
#         }
#         # add top-k predicted lists as strings for reference
#         max_k = max(ks)
#         topk_species = [str(x['pred_species']) if pd.notna(x['pred_species']) else '' for x in preds_ordered[:max_k]]
#         topk_genus = [str(x['pred_genus']) if pd.notna(x['pred_genus']) else '' for x in preds_ordered[:max_k]]
#         # store comma-separated
#         row['top{}_species'.format(max_k)] = ';'.join(topk_species)
#         row['top{}_genus'.format(max_k)] = ';'.join(topk_genus)

#         # add hit flags
#         for k in ks:
#             row[f'hit_species@{k}'] = hit_flags[f'hit_species@{k}']
#             row[f'hit_genus@{k}'] = hit_flags[f'hit_genus@{k}']

#         detailed_rows.append(row)

#     detailed_df = pd.DataFrame(detailed_rows)

#     # compute summary metrics (only over phages with known true labels)
#     denom = sum(1 for pid in pred_aug_df['phage_id'].unique() if pid in truth_map)
#     # If some phage ids were not found in truth_map but we tried fallback, denom computed above might be smaller than n_phages_with_label.
#     # Use n_phages_with_label instead because we incremented that above when matched.
#     denom = n_phages_with_label if n_phages_with_label > 0 else 0

#     summary_rows = []
#     if denom == 0:
#         print("Warning: no phage with known true labels found in truth file. Summary will be zeros.")
#     for level in ['species', 'genus']:
#         row = {'level': level}
#         for k in ks:
#             key = f'hit_{level}@{k}'
#             count = summary_counts.get(key, 0)
#             rate = count / denom if denom > 0 else 0.0
#             row[f'Hit@{k}'] = rate
#             row[f'Count@{k}'] = count
#         summary_rows.append(row)

#     summary_df = pd.DataFrame(summary_rows)

#     # Outputs
#     out_prefix = args.out_prefix or 'result'
#     out_details = f"{out_prefix}_hit_results.tsv"
#     out_summary = f"{out_prefix}_hit_summary.tsv"

#     detailed_df.to_csv(out_details, sep='\t', index=False)
#     summary_df.to_csv(out_summary, sep='\t', index=False)

#     # Print summary to stdout
#     print(f"\nTotal phage predictions processed (unique phage_id): {phage_total}")
#     print(f"Phages with known true label used for Hit@k (denominator): {denom}")
#     print("Hit@k summary (rates):")
#     print(summary_df.to_string(index=False))

#     print(f"\nWrote detailed results to: {out_details}")
#     print(f"Wrote summary to: {out_summary}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Compute Hit@k (species & genus) from prediction + taxonomy + truth files.")
#     parser.add_argument('--pred', required=True, help="Prediction TSV file (must contain phage_id; preferably host_id, host_species, rank or score).")
#     parser.add_argument('--taxonomy', required=True, help="taxonomy TSV (columns: taxid,parent_taxid,name,rank,alias).")
#     parser.add_argument('--truth', required=True, help="Truth TSV (must contain refseq_id, genus, species).")
#     parser.add_argument('--k', required=True, nargs='+', help="List of k values (e.g. --k 1 3 5)", metavar='K')
#     parser.add_argument('--out_prefix', default='result', help="Prefix for output files (default: result).")
#     args = parser.parse_args()
#     try:
#         main(args)
#     except Exception as e:
#         print("Error:", e)
#         sys.exit(1)
#!/usr/bin/env python
import argparse
import pandas as pd
import sys
from collections import defaultdict
from typing import Optional, Dict, Tuple, List


def read_prediction(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)  # 自动检测分隔符
    df.columns = [c.strip() for c in df.columns]
    return df


def read_taxonomy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # Ensure required columns exist
    for col in ['taxid', 'parent_taxid', 'name', 'rank', 'alias']:
        if col not in df.columns:
            raise ValueError(f"taxonomy file must contain column '{col}' (found: {df.columns.tolist()})")
    return df


def read_truth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    # Expect refseq_id and genus/species columns
    for col in ['refseq_id', 'genus', 'species']:
        if col not in df.columns:
            raise ValueError(f"truth file must contain column '{col}' (found: {df.columns.tolist()})")
    return df


# NEW: 读取 pairs_all.tsv，构建 host_gcf -> host_species_taxid 映射
def read_pairs_map(path: str) -> Dict[str, str]:
    """
    从 /home/.../pairs_all.tsv 读取 host_gcf 与 host_species_taxid 的映射
    期望列: phage_id, host_gcf, host_species_taxid, label, source
    实际只用 host_gcf 和 host_species_taxid
    """
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in ['host_gcf', 'host_species_taxid']:
        if col not in df.columns:
            raise ValueError(f"pairs file must contain column '{col}' (found: {df.columns.tolist()})")
    gcf_to_taxid: Dict[str, str] = {}
    for _, r in df.iterrows():
        gcf = str(r['host_gcf']).strip() if pd.notna(r['host_gcf']) else ''
        tid = str(r['host_species_taxid']).strip() if pd.notna(r['host_species_taxid']) else ''
        if gcf and tid:
            # 如果同一个 gcf 对应多个 taxid，这里最后一个会覆盖前面
            gcf_to_taxid[gcf] = tid
    return gcf_to_taxid


def build_taxonomy_maps(tax_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Build:
      - alias_to_taxid: map alias (e.g. GCF_006539325) -> taxid (string)
      - taxid_info: map taxid -> {'parent_taxid':..., 'name':..., 'rank':...}
    """
    alias_to_taxid: Dict[str, str] = {}
    taxid_info: Dict[str, Dict[str, str]] = {}

    for _, row in tax_df.iterrows():
        taxid = str(row['taxid']).strip()
        parent = str(row['parent_taxid']).strip() if pd.notna(row['parent_taxid']) else ''
        name = str(row['name']).strip() if pd.notna(row['name']) else ''
        rank = str(row['rank']).strip() if pd.notna(row['rank']) else ''
        alias = str(row['alias']).strip() if pd.notna(row['alias']) else ''

        taxid_info[taxid] = {'parent_taxid': parent, 'name': name, 'rank': rank}
        if alias and alias != 'nan':
            for a in [x.strip() for x in alias.replace(';', ',').split(',') if x.strip()]:
                alias_to_taxid[a] = taxid

    return alias_to_taxid, taxid_info


def find_genus_from_taxid(taxid: str, taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Walk up the taxonomy tree from taxid to find the first node with rank == 'genus'.
    Returns genus name or None if not found.
    """
    visited = set()
    current = taxid
    while current and current not in visited:
        visited.add(current)
        info = taxid_info.get(current)
        if not info:
            return None
        if info.get('rank', '').lower() == 'genus':
            name = info.get('name', '').strip()
            return name if name else None
        parent = info.get('parent_taxid', '')
        if parent == '' or parent == current:
            return None
        current = parent
    return None


# NEW: 用安全方式判断标签是否缺失
def is_missing_label(x) -> bool:
    """安全判断标签是否缺失：None / 空串 / 'nan' / NaN 都算缺失"""
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip().lower()
        return s == '' or s == 'nan'
    return pd.isna(x)


# 修改：增加 gcf_to_taxid 参数
def map_hostid_to_genus(
    host_id: str,
    alias_to_taxid: Dict[str, str],
    taxid_info: Dict[str, Dict[str, str]],
    gcf_to_taxid: Optional[Dict[str, str]] = None       # NEW
) -> Optional[str]:
    """
    Given host_id (e.g., 'GCF_016728825'), try to find genus via:
      1) pairs_all.tsv 提供的 host_gcf -> host_species_taxid -> genus
      2) taxonomy_with_alias.tsv 中 alias -> taxid -> genus (旧逻辑)
    """
    if pd.isna(host_id) or host_id == '':
        return None
    host_id = str(host_id).strip()

    # --- 1) 优先用 GCF -> species_taxid -> genus (来自 pairs_all.tsv) ---
    if gcf_to_taxid is not None:
        taxid = gcf_to_taxid.get(host_id)
        if taxid:
            genus = find_genus_from_taxid(taxid, taxid_info)
            if genus:
                return genus

    # --- 2) 旧逻辑：alias -> taxid -> genus ---
    taxid = alias_to_taxid.get(host_id)
    if taxid:
        genus = find_genus_from_taxid(taxid, taxid_info)
        if genus:
            return genus

    # 3) host_id 中可能有多个 GCF，用分隔符拆开再用 alias_to_taxid
    for sep in [';', ',', '|', ' ']:
        if sep in host_id:
            parts = [p.strip() for p in host_id.split(sep) if p.strip()]
            for p in parts:
                t = alias_to_taxid.get(p)
                if t:
                    g = find_genus_from_taxid(t, taxid_info)
                    if g:
                        return g
    return None


def extract_genus_from_species_str(species_str: str) -> Optional[str]:
    """Fallback: from a species string like 'Streptococcus pyogenes' extract 'Streptococcus'."""
    if pd.isna(species_str) or species_str == '':
        return None
    parts = str(species_str).strip().split()
    if len(parts) >= 1:
        return parts[0]
    return None


def prepare_prediction(pred_df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in pred_df.columns}
    if 'phage_id' not in cols:
        raise ValueError("prediction file must contain 'phage_id' column")
    pred_df = pred_df.copy()
    pred_df['phage_id'] = pred_df[cols['phage_id']].astype(str).str.strip()

    host_id_col = cols.get('host_id')
    host_species_col = cols.get('host_species') or cols.get('host')
    if host_id_col:
        pred_df['host_id_raw'] = pred_df[host_id_col].astype(str).str.strip()
    else:
        pred_df['host_id_raw'] = pd.NA

    if host_species_col:
        pred_df['host_species_raw'] = pred_df[host_species_col].astype(str).str.strip()
    else:
        pred_df['host_species_raw'] = pd.NA

    if 'rank' in cols:
        try:
            pred_df['rank_val'] = pd.to_numeric(pred_df[cols['rank']], errors='coerce')
        except Exception:
            pred_df['rank_val'] = pd.NA
    else:
        pred_df['rank_val'] = pd.NA

    if 'score' in cols:
        pred_df['score_val'] = pd.to_numeric(pred_df[cols['score']], errors='coerce')
    else:
        pred_df['score_val'] = pd.NA

    return pred_df


def compute_hitk_for_phage(preds: List[Dict], true_species: Optional[str], true_genus: Optional[str], ks: List[int]) -> Dict[str, int]:
    results = {}
    pred_species_list = [p.get('pred_species') for p in preds]
    pred_genus_list = [p.get('pred_genus') for p in preds]

    for k in ks:
        topk_s = pred_species_list[:k]
        topk_g = pred_genus_list[:k]

        if true_species is None or true_species == '' or pd.isna(true_species):
            results[f'hit_species@{k}'] = 0
        else:
            ts = str(true_species).strip().lower()
            hit_s = any(ts == str(x).strip().lower() for x in topk_s if pd.notna(x))
            results[f'hit_species@{k}'] = 1 if hit_s else 0

        if true_genus is None or true_genus == '' or pd.isna(true_genus):
            results[f'hit_genus@{k}'] = 0
        else:
            tg = str(true_genus).strip().lower()
            hit_g = any(tg == str(x).strip().lower() for x in topk_g if pd.notna(x))
            results[f'hit_genus@{k}'] = 1 if hit_g else 0

    return results


def main(args):
    print("Reading files...")
    pred_df_raw = read_prediction(args.pred)
    tax_df = read_taxonomy(args.taxonomy)
    truth_df = read_truth(args.truth)
    # NEW: 读取 pairs_all 映射
    gcf_to_taxid = read_pairs_map(args.pairs) if args.pairs else {}

    pred_df = prepare_prediction(pred_df_raw)
    alias_to_taxid, taxid_info = build_taxonomy_maps(tax_df)

    truth_map: Dict[str, Tuple[str, str]] = {}
    for _, r in truth_df.iterrows():
        ref = str(r['refseq_id']).strip()
        g = str(r['genus']).strip() if pd.notna(r['genus']) else ''
        s = str(r['species']).strip() if pd.notna(r['species']) else ''
        truth_map[ref] = (g, s)

    use_rank = pred_df['rank_val'].notna().any()
    use_score = pred_df['score_val'].notna().any()

    if use_rank:
        pred_df['rank_val'] = pd.to_numeric(pred_df['rank_val'], errors='coerce')
        pred_df['rank_val_filled'] = pred_df['rank_val'].fillna(
            pred_df['rank_val'].max() + 1000 if pred_df['rank_val'].notna().any() else 1e9
        )
        sort_keys = ['phage_id', 'rank_val_filled']
        sort_ascending = True
    elif use_score:
        pred_df['score_val'] = pd.to_numeric(pred_df['score_val'], errors='coerce')
        pred_df['score_val_filled'] = pred_df['score_val'].fillna(-1e9)
        sort_keys = None
        sort_ascending = False
    else:
        sort_keys = ['phage_id']
        sort_ascending = True

    pred_aug_rows = []
    for _, row in pred_df.iterrows():
        host_id_raw = row.get('host_id_raw') if 'host_id_raw' in row else None
        host_species_raw = row.get('host_species_raw') if 'host_species_raw' in row else None

        pred_species = None
        if pd.notna(host_species_raw) and str(host_species_raw).strip() and str(host_species_raw).strip().lower() != 'nan':
            pred_species = str(host_species_raw).strip()

        pred_genus = None
        if pd.notna(host_id_raw) and str(host_id_raw).strip() and str(host_id_raw).strip().lower() != 'nan':
            pred_genus = map_hostid_to_genus(
                str(host_id_raw).strip(),
                alias_to_taxid,
                taxid_info,
                gcf_to_taxid=gcf_to_taxid  # NEW: 传入 pairs_all 映射
            )

        if pred_genus is None and pred_species:
            pred_genus = extract_genus_from_species_str(pred_species)

        if pred_species is None and pd.notna(host_id_raw) and str(host_id_raw).strip():
            # 如果 host_id 本身就在 alias_to_taxid 里，且 rank=species，可以用它做 species 名
            host_id_str = str(host_id_raw).strip()
            taxid = gcf_to_taxid.get(host_id_str) or alias_to_taxid.get(host_id_str)
            if taxid:
                info = taxid_info.get(taxid)
                if info and info.get('rank', '').lower() == 'species' and info.get('name'):
                    pred_species = info.get('name')

        new_row = dict(row)
        new_row['pred_species'] = pred_species
        new_row['pred_genus'] = pred_genus
        new_row['host_id_raw'] = host_id_raw
        new_row['host_species_raw'] = host_species_raw
        pred_aug_rows.append(new_row)

    pred_aug_df = pd.DataFrame(pred_aug_rows)

    if sort_keys:
        pred_aug_df = pred_aug_df.sort_values(by=sort_keys, ascending=sort_ascending, kind='mergesort')
    else:
        pred_aug_df = pred_aug_df.sort_values(
            by=['phage_id', 'score_val_filled'],
            ascending=[True, False],
            kind='mergesort'
        )

    phage_groups = pred_aug_df.groupby('phage_id', sort=True)

    ks = sorted([int(x) for x in args.k])
    ks = [int(x) for x in ks if int(x) >= 1]

    detailed_rows = []
    error_rows = []  # Top1 错误样本
    n_phages_with_label = 0
    summary_counts = defaultdict(int)
    phage_total = 0

    for phage_id, group in phage_groups:
        phage_total += 1
        preds_ordered = []
        for _, r in group.iterrows():
            preds_ordered.append({
                'pred_species': r.get('pred_species'),
                'pred_genus': r.get('pred_genus'),
                'host_id_raw': r.get('host_id_raw'),
                'host_species_raw': r.get('host_species_raw')
            })

        true_genus = None
        true_species = None
        if phage_id in truth_map:
            true_genus, true_species = truth_map[phage_id]
            n_phages_with_label += 1
        else:
            found = False
            if isinstance(phage_id, str):
                for sep in ['|', ' ', ';', ',']:
                    if sep in phage_id:
                        for part in phage_id.split(sep):
                            p = part.strip()
                            if p in truth_map:
                                true_genus, true_species = truth_map[p]
                                found = True
                                break
                        if found:
                            break

        hit_flags = compute_hitk_for_phage(preds_ordered, true_species, true_genus, ks)

        if true_genus is not None or true_species is not None:
            for k in ks:
                summary_counts[f'hit_species@{k}'] += hit_flags[f'hit_species@{k}']
                summary_counts[f'hit_genus@{k}'] += hit_flags[f'hit_genus@{k}']

        # -------- Top1 错误样本收集 --------
        if preds_ordered:
            top1 = preds_ordered[0]
            pred_species_top1 = top1.get('pred_species')
            pred_genus_top1 = top1.get('pred_genus')
            host_id_top1 = top1.get('host_id_raw')
            host_species_top1 = top1.get('host_species_raw')

            has_true_species = not is_missing_label(true_species)
            has_true_genus = not is_missing_label(true_genus)

            wrong_species = (
                has_true_species
                and 'hit_species@1' in hit_flags
                and hit_flags['hit_species@1'] == 0
            )
            wrong_genus = (
                has_true_genus
                and 'hit_genus@1' in hit_flags
                and hit_flags['hit_genus@1'] == 0
            )

            if wrong_species or wrong_genus:
                error_rows.append({
                    'phage_id': phage_id,
                    'true_species': true_species,
                    'true_genus': true_genus,
                    'pred_species_top1': pred_species_top1,
                    'pred_genus_top1': pred_genus_top1,
                    'host_id_top1': host_id_top1,
                    'host_species_top1': host_species_top1,
                    'wrong_species': int(wrong_species),
                    'wrong_genus': int(wrong_genus),
                })
        # -------- Top1 错误样本结束 --------

        row = {
            'phage_id': phage_id,
            'true_genus': true_genus,
            'true_species': true_species,
            'n_predictions': len(preds_ordered)
        }
        max_k = max(ks)
        topk_species = [
            str(x['pred_species']) if pd.notna(x['pred_species']) else ''
            for x in preds_ordered[:max_k]
        ]
        topk_genus = [
            str(x['pred_genus']) if pd.notna(x['pred_genus']) else ''
            for x in preds_ordered[:max_k]
        ]
        row[f'top{max_k}_species'] = ';'.join(topk_species)
        row[f'top{max_k}_genus'] = ';'.join(topk_genus)

        for k in ks:
            row[f'hit_species@{k}'] = hit_flags[f'hit_species@{k}']
            row[f'hit_genus@{k}'] = hit_flags[f'hit_genus@{k}']

        detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)

    denom = n_phages_with_label if n_phages_with_label > 0 else 0
    summary_k_rows = []
    if denom == 0:
        print("Warning: no phage with known true labels found in truth file. Summary will be zeros.")
    for k in ks:
        species_count = summary_counts.get(f'hit_species@{k}', 0)
        genus_count = summary_counts.get(f'hit_genus@{k}', 0)
        species_rate = species_count / denom if denom > 0 else 0.0
        genus_rate = genus_count / denom if denom > 0 else 0.0
        summary_k_rows.append({
            'k': k,
            'Hit_species': species_rate,
            'Count_species': species_count,
            'Hit_genus': genus_rate,
            'Count_genus': genus_count
        })

    summary_df = pd.DataFrame(summary_k_rows)

    out_prefix = args.out_prefix or 'result'
    out_details = f"{out_prefix}_hit_results.tsv"
    out_summary = f"{out_prefix}_hit_summary.tsv"
    out_errors = f"{out_prefix}_top1_errors.tsv"

    detailed_df.to_csv(out_details, sep='\t', index=False)
    summary_df.to_csv(out_summary, sep='\t', index=False)

    if error_rows:
        errors_df = pd.DataFrame(error_rows)
        errors_df.to_csv(out_errors, sep='\t', index=False)
    else:
        out_errors = "(no_top1_errors_generated)"
        print("No Top1 error rows detected; not writing top1 error file.")

    print(f"\nTotal phage predictions processed (unique phage_id): {phage_total}")
    print(f"Phages with known true label used for Hit@k (denominator): {denom}")
    print("Hit@k summary (per-k rows):")
    if not summary_df.empty:
        print(summary_df.to_string(index=False, float_format='{:0.4f}'.format))
    else:
        print("(empty)")

    print(f"\nWrote detailed results to: {out_details}")
    print(f"Wrote summary to: {out_summary}")
    print(f"Wrote Top1 error details to: {out_errors}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute Hit@k (species & genus) from prediction + taxonomy + truth files."
    )
    parser.add_argument('--pred', required=True,
                        help="Prediction TSV file (must contain phage_id; preferably host_id, host_species, rank or score).")
    parser.add_argument('--taxonomy', required=True,
                        help="taxonomy TSV (columns: taxid,parent_taxid,name,rank,alias).")
    parser.add_argument('--truth', required=True,
                        help="Truth TSV (must contain refseq_id, genus, species).")
    parser.add_argument('--pairs', required=False, default=None,   # NEW
                        help="Pairs TSV (phage_id,host_gcf,host_species_taxid,...) to map host_gcf -> species taxid.")
    parser.add_argument('--k', required=True, nargs='+', metavar='K',
                        help="List of k values (e.g. --k 1 3 5)")
    parser.add_argument('--out_prefix', default='result',
                        help="Prefix for output files (default: result).")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

"""
compute_hitk.py

计算 species & genus 级别的 Hit@k。

Usage:
    python accuracy.py --taxonomy /home/wangjingyuan/wys/wys_shiyan/taxonomy_with_alias.tsv\
        --truth /home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF_all_updated.tsv --k 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30\
        --out_prefix accuracy_result3\
        --pairs  /home/wangjingyuan/wys/duibi/pairs_train_copy.tsv\
        --pred /home/wangjingyuan/wys/duibi/output-train-GAT_softce_cluster_hid512_2layer_4heads_20.10_512_neg20_evl10_drop0.2_1e-5_cos_new_30000_common_new3/predictions/phage_prediction_results_final.tsv
        
        
        
        输出:
    result_hit_results.tsv   # 每个 phage 的命中明细
    result_hit_summary.tsv   # hit@k 汇总（species & genus）

"""