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
#     # Expect refseq_id and genus/species columns; family is optional
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
#             for a in [x.strip() for x in alias.replace(';', ',').split(',') if x.strip()]:
#                 alias_to_taxid[a] = taxid

#     return alias_to_taxid, taxid_info


# def find_genus_from_taxid(taxid: str, taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
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
#         parent = info.get('parent_taxid', '')
#         if parent == '' or parent == current:
#             return None
#         current = parent
#     return None


# def find_family_from_taxid(taxid: str, taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
#     visited = set()
#     current = taxid
#     while current and current not in visited:
#         visited.add(current)
#         info = taxid_info.get(current)
#         if not info:
#             return None
#         if info.get('rank', '').lower() == 'family':
#             name = info.get('name', '').strip()
#             return name if name else None
#         parent = info.get('parent_taxid', '')
#         if parent == '' or parent == current:
#             return None
#         current = parent
#     return None


# def map_hostid_to_genus(host_id: str, alias_to_taxid: Dict[str, str], taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
#     if pd.isna(host_id) or host_id == '':
#         return None
#     host_id = str(host_id).strip()
#     taxid = alias_to_taxid.get(host_id)
#     if taxid:
#         genus = find_genus_from_taxid(taxid, taxid_info)
#         if genus:
#             return genus
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


# def map_hostid_to_family(host_id: str, alias_to_taxid: Dict[str, str], taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
#     if pd.isna(host_id) or host_id == '':
#         return None
#     host_id = str(host_id).strip()
#     taxid = alias_to_taxid.get(host_id)
#     if taxid:
#         family = find_family_from_taxid(taxid, taxid_info)
#         if family:
#             return family
#     for sep in [';', ',', '|', ' ']:
#         if sep in host_id:
#             parts = [p.strip() for p in host_id.split(sep) if p.strip()]
#             for p in parts:
#                 t = alias_to_taxid.get(p)
#                 if t:
#                     f = find_family_from_taxid(t, taxid_info)
#                     if f:
#                         return f
#     return None


# def extract_genus_from_species_str(species_str: str) -> Optional[str]:
#     if pd.isna(species_str) or species_str == '':
#         return None
#     parts = str(species_str).strip().split()
#     if len(parts) >= 1:
#         return parts[0]
#     return None


# def prepare_prediction(pred_df: pd.DataFrame) -> pd.DataFrame:
#     cols = {c.lower(): c for c in pred_df.columns}
#     if 'phage_id' not in cols:
#         raise ValueError("prediction file must contain 'phage_id' column")
#     pred_df = pred_df.copy()
#     pred_df['phage_id'] = pred_df[cols['phage_id']].astype(str).str.strip()

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

#     if 'rank' in cols:
#         pred_df['rank_val'] = pd.to_numeric(pred_df[cols['rank']], errors='coerce')
#     else:
#         pred_df['rank_val'] = pd.NA

#     if 'score' in cols:
#         pred_df['score_val'] = pd.to_numeric(pred_df[cols['score']], errors='coerce')
#     else:
#         pred_df['score_val'] = pd.NA

#     return pred_df


# def compute_hitk_for_phage(
#     preds: List[Dict],
#     true_species: Optional[str],
#     true_genus: Optional[str],
#     true_family: Optional[str],
#     ks: List[int]
# ) -> Dict[str, int]:
#     results = {}
#     pred_species_list = [p.get('pred_species') for p in preds]
#     pred_genus_list = [p.get('pred_genus') for p in preds]
#     pred_family_list = [p.get('pred_family') for p in preds]

#     for k in ks:
#         topk_s = pred_species_list[:k]
#         topk_g = pred_genus_list[:k]
#         topk_f = pred_family_list[:k]

#         # species
#         if true_species is None or true_species == '' or pd.isna(true_species):
#             results[f'hit_species@{k}'] = 0
#         else:
#             hit_s = any((ts := str(true_species).strip()).lower() == str(x).strip().lower() for x in topk_s if pd.notna(x))
#             results[f'hit_species@{k}'] = 1 if hit_s else 0

#         # genus
#         if true_genus is None or true_genus == '' or pd.isna(true_genus):
#             results[f'hit_genus@{k}'] = 0
#         else:
#             hit_g = any((tg := str(true_genus).strip()).lower() == str(x).strip().lower() for x in topk_g if pd.notna(x))
#             results[f'hit_genus@{k}'] = 1 if hit_g else 0

#         # family
#         if true_family is None or true_family == '' or pd.isna(true_family):
#             results[f'hit_family@{k}'] = 0
#         else:
#             hit_f = any((tf := str(true_family).strip()).lower() == str(x).strip().lower() for x in topk_f if pd.notna(x))
#             results[f'hit_family@{k}'] = 1 if hit_f else 0

#     return results


# def main(args):
#     print("Reading files...")
#     pred_df_raw = read_prediction(args.pred)
#     tax_df = read_taxonomy(args.taxonomy)
#     truth_df = read_truth(args.truth)

#     pred_df = prepare_prediction(pred_df_raw)
#     alias_to_taxid, taxid_info = build_taxonomy_maps(tax_df)

#     # Build truth map: refseq_id -> (genus, species, family)
#     has_family_col = 'family' in truth_df.columns
#     truth_map: Dict[str, Tuple[str, str, str]] = {}
#     for _, r in truth_df.iterrows():
#         ref = str(r['refseq_id']).strip()
#         g = str(r['genus']).strip() if pd.notna(r['genus']) else ''
#         s = str(r['species']).strip() if pd.notna(r['species']) else ''
#         f = str(r['family']).strip() if has_family_col and pd.notna(r['family']) else ''
#         truth_map[ref] = (g, s, f)

#     # Determine ordering
#     use_rank = pred_df['rank_val'].notna().any()
#     use_score = pred_df['score_val'].notna().any()

#     if use_rank:
#         pred_df['rank_val'] = pd.to_numeric(pred_df['rank_val'], errors='coerce')
#         pred_df['rank_val_filled'] = pred_df['rank_val'].fillna(pred_df['rank_val'].max() + 1000 if pred_df['rank_val'].notna().any() else 1e9)
#         sort_keys = ['phage_id', 'rank_val_filled']
#         sort_ascending = True
#     elif use_score:
#         pred_df['score_val'] = pd.to_numeric(pred_df['score_val'], errors='coerce')
#         pred_df['score_val_filled'] = pred_df['score_val'].fillna(-1e9)
#         sort_keys = None
#     else:
#         sort_keys = ['phage_id']
#         sort_ascending = True

#     # Augment predictions with predicted taxa
#     pred_aug_rows = []
#     for _, row in pred_df.iterrows():
#         host_id_raw = row.get('host_id_raw')
#         host_species_raw = row.get('host_species_raw')

#         pred_species = None
#         if pd.notna(host_species_raw) and str(host_species_raw).strip().lower() != 'nan':
#             pred_species = str(host_species_raw).strip()

#         pred_genus = None
#         if pd.notna(host_id_raw) and str(host_id_raw).strip().lower() != 'nan':
#             pred_genus = map_hostid_to_genus(str(host_id_raw).strip(), alias_to_taxid, taxid_info)
#         if pred_genus is None and pred_species:
#             pred_genus = extract_genus_from_species_str(pred_species)

#         pred_family = None
#         if pd.notna(host_id_raw) and str(host_id_raw).strip().lower() != 'nan':
#             pred_family = map_hostid_to_family(str(host_id_raw).strip(), alias_to_taxid, taxid_info)

#         # Fallback: if pred_species is None but host_id maps to species-rank taxon
#         if pred_species is None and pd.notna(host_id_raw) and str(host_id_raw).strip():
#             taxid = alias_to_taxid.get(str(host_id_raw).strip())
#             if taxid:
#                 info = taxid_info.get(taxid)
#                 if info and info.get('rank', '').lower() == 'species' and info.get('name'):
#                     pred_species = info.get('name')

#         new_row = dict(row)
#         new_row['pred_species'] = pred_species
#         new_row['pred_genus'] = pred_genus
#         new_row['pred_family'] = pred_family
#         pred_aug_rows.append(new_row)

#     pred_aug_df = pd.DataFrame(pred_aug_rows)

#     if sort_keys:
#         pred_aug_df = pred_aug_df.sort_values(by=sort_keys, ascending=sort_ascending, kind='mergesort')
#     else:
#         pred_aug_df = pred_aug_df.sort_values(by=['phage_id', 'score_val_filled'], ascending=[True, False], kind='mergesort')

#     phage_groups = pred_aug_df.groupby('phage_id', sort=True)

#     ks = sorted([int(x) for x in args.k])
#     ks = [k for k in ks if k >= 1]
#     max_k = max(ks) if ks else 1

#     detailed_rows = []
#     n_phages_with_label = 0
#     summary_counts = defaultdict(int)
#     phage_total = 0

#     for phage_id, group in phage_groups:
#         phage_total += 1
#         preds_ordered = []
#         for _, r in group.iterrows():
#             preds_ordered.append({
#                 'pred_species': r.get('pred_species'),
#                 'pred_genus': r.get('pred_genus'),
#                 'pred_family': r.get('pred_family')
#             })

#         true_genus = true_species = true_family = None
#         if phage_id in truth_map:
#             true_genus, true_species, true_family = truth_map[phage_id]
#             n_phages_with_label += 1
#         else:
#             found = False
#             if isinstance(phage_id, str):
#                 for sep in ['|', ' ', ';', ',']:
#                     if sep in phage_id:
#                         for part in phage_id.split(sep):
#                             p = part.strip()
#                             if p in truth_map:
#                                 true_genus, true_species, true_family = truth_map[p]
#                                 found = True
#                                 break
#                         if found:
#                             break

#         hit_flags = compute_hitk_for_phage(preds_ordered, true_species, true_genus, true_family, ks)

#         if true_genus is not None or true_species is not None or true_family is not None:
#             for k in ks:
#                 summary_counts[f'hit_species@{k}'] += hit_flags[f'hit_species@{k}']
#                 summary_counts[f'hit_genus@{k}'] += hit_flags[f'hit_genus@{k}']
#                 summary_counts[f'hit_family@{k}'] += hit_flags[f'hit_family@{k}']

#         row = {
#             'phage_id': phage_id,
#             'true_genus': true_genus,
#             'true_species': true_species,
#             'true_family': true_family,
#             'n_predictions': len(preds_ordered)
#         }

#         topk_species = [str(x['pred_species']) if pd.notna(x['pred_species']) else '' for x in preds_ordered[:max_k]]
#         topk_genus = [str(x['pred_genus']) if pd.notna(x['pred_genus']) else '' for x in preds_ordered[:max_k]]
#         topk_family = [str(x['pred_family']) if pd.notna(x['pred_family']) else '' for x in preds_ordered[:max_k]]

#         row[f'top{max_k}_species'] = ';'.join(topk_species)
#         row[f'top{max_k}_genus'] = ';'.join(topk_genus)
#         row[f'top{max_k}_family'] = ';'.join(topk_family)

#         for k in ks:
#             row[f'hit_species@{k}'] = hit_flags[f'hit_species@{k}']
#             row[f'hit_genus@{k}'] = hit_flags[f'hit_genus@{k}']
#             row[f'hit_family@{k}'] = hit_flags[f'hit_family@{k}']

#         detailed_rows.append(row)

#     detailed_df = pd.DataFrame(detailed_rows)

#     denom = n_phages_with_label if n_phages_with_label > 0 else 0

#     summary_k_rows = []
#     if denom == 0:
#         print("Warning: no phage with known true labels found in truth file. Summary will be zeros.")
#     for k in ks:
#         species_count = summary_counts.get(f'hit_species@{k}', 0)
#         genus_count = summary_counts.get(f'hit_genus@{k}', 0)
#         family_count = summary_counts.get(f'hit_family@{k}', 0)
#         species_rate = species_count / denom if denom > 0 else 0.0
#         genus_rate = genus_count / denom if denom > 0 else 0.0
#         family_rate = family_count / denom if denom > 0 else 0.0
#         summary_k_rows.append({
#             'k': k,
#             'Hit_species': species_rate,
#             'Count_species': species_count,
#             'Hit_genus': genus_rate,
#             'Count_genus': genus_count,
#             'Hit_family': family_rate,
#             'Count_family': family_count
#         })

#     summary_df = pd.DataFrame(summary_k_rows)

#     out_prefix = args.out_prefix or 'result'
#     out_details = f"{out_prefix}_hit_results.tsv"
#     out_summary = f"{out_prefix}_hit_summary.tsv"

#     detailed_df.to_csv(out_details, sep='\t', index=False)
#     summary_df.to_csv(out_summary, sep='\t', index=False)

#     print(f"\nTotal phage predictions processed (unique phage_id): {phage_total}")
#     print(f"Phages with known true label used for Hit@k (denominator): {denom}")
#     print("Hit@k summary (per-k rows):")
#     if not summary_df.empty:
#         print(summary_df.to_string(index=False, float_format='{:0.4f}'.format))
#     else:
#         print("(empty)")

#     print(f"\nWrote detailed results to: {out_details}")
#     print(f"Wrote summary to: {out_summary}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Compute Hit@k (species, genus, family) from prediction + taxonomy + truth files.")
#     parser.add_argument('--pred', required=True, help="Prediction TSV file (must contain phage_id; preferably host_id, host_species, rank or score).")
#     parser.add_argument('--taxonomy', required=True, help="taxonomy TSV (columns: taxid,parent_taxid,name,rank,alias).")
#     parser.add_argument('--truth', required=True, help="Truth TSV (must contain refseq_id, genus, species; optionally family).")
#     parser.add_argument('--k', required=True, nargs='+', help="List of k values (e.g. --k 1 3 5)", metavar='K')
#     parser.add_argument('--out_prefix', default='result', help="Prefix for output files (default: result).")
#     args = parser.parse_args()
#     try:
#         main(args)
#     except Exception as e:
#         print("Error:", e)
#         sys.exit(1)
# '''
#     python acc.py --pred /home/wangjingyuan/wys/wys_shiyan/output-train-GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg70_evl-1_drop0.2_1e-5_cos_new_30000_True_noleak_noptax1_new_data/predictions/phage_prediction_results_test_topk.tsv\
#           --taxonomy /home/wangjingyuan/wys/wys_shiyan/taxonomy_with_alias.tsv\
#             --truth phage_host.tsv\
#                   --k 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30\
#         --out_prefix accuracy_result_data1
# '''


import argparse
import pandas as pd
import sys
from collections import defaultdict
from typing import Optional, Dict, Tuple, List


def read_prediction(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)  # 自动检测分隔符
    # 强制列名小写便于处理
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
    # Expect refseq_id and genus/species columns; family is optional
    for col in ['refseq_id', 'genus', 'species']:
        if col not in df.columns:
            raise ValueError(f"truth file must contain column '{col}' (found: {df.columns.tolist()})")
    return df


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


def find_family_from_taxid(taxid: str, taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
    visited = set()
    current = taxid
    while current and current not in visited:
        visited.add(current)
        info = taxid_info.get(current)
        if not info:
            return None
        if info.get('rank', '').lower() == 'family':
            name = info.get('name', '').strip()
            return name if name else None
        parent = info.get('parent_taxid', '')
        if parent == '' or parent == current:
            return None
        current = parent
    return None


def map_hostid_to_genus(host_id: str, alias_to_taxid: Dict[str, str], taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
    if pd.isna(host_id) or host_id == '':
        return None
    host_id = str(host_id).strip()
    taxid = alias_to_taxid.get(host_id)
    if taxid:
        genus = find_genus_from_taxid(taxid, taxid_info)
        if genus:
            return genus
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


def map_hostid_to_family(host_id: str, alias_to_taxid: Dict[str, str], taxid_info: Dict[str, Dict[str, str]]) -> Optional[str]:
    if pd.isna(host_id) or host_id == '':
        return None
    host_id = str(host_id).strip()
    taxid = alias_to_taxid.get(host_id)
    if taxid:
        family = find_family_from_taxid(taxid, taxid_info)
        if family:
            return family
    for sep in [';', ',', '|', ' ']:
        if sep in host_id:
            parts = [p.strip() for p in host_id.split(sep) if p.strip()]
            for p in parts:
                t = alias_to_taxid.get(p)
                if t:
                    f = find_family_from_taxid(t, taxid_info)
                    if f:
                        return f
    return None


def extract_genus_from_species_str(species_str: str) -> Optional[str]:
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
        pred_df['rank_val'] = pd.to_numeric(pred_df[cols['rank']], errors='coerce')
    else:
        pred_df['rank_val'] = pd.NA

    if 'score' in cols:
        pred_df['score_val'] = pd.to_numeric(pred_df[cols['score']], errors='coerce')
    else:
        pred_df['score_val'] = pd.NA

    return pred_df


def compute_hitk_for_phage(
    preds: List[Dict],
    true_species: Optional[str],
    true_genus: Optional[str],
    true_family: Optional[str],
    ks: List[int]
) -> Dict[str, int]:
    results = {}
    pred_species_list = [p.get('pred_species') for p in preds]
    pred_genus_list = [p.get('pred_genus') for p in preds]
    pred_family_list = [p.get('pred_family') for p in preds]

    for k in ks:
        topk_s = pred_species_list[:k]
        topk_g = pred_genus_list[:k]
        topk_f = pred_family_list[:k]

        # species
        if true_species is None or true_species == '' or pd.isna(true_species):
            results[f'hit_species@{k}'] = 0
        else:
            hit_s = any((ts := str(true_species).strip()).lower() == str(x).strip().lower() for x in topk_s if pd.notna(x))
            results[f'hit_species@{k}'] = 1 if hit_s else 0

        # genus
        if true_genus is None or true_genus == '' or pd.isna(true_genus):
            results[f'hit_genus@{k}'] = 0
        else:
            hit_g = any((tg := str(true_genus).strip()).lower() == str(x).strip().lower() for x in topk_g if pd.notna(x))
            results[f'hit_genus@{k}'] = 1 if hit_g else 0

        # family
        if true_family is None or true_family == '' or pd.isna(true_family):
            results[f'hit_family@{k}'] = 0
        else:
            hit_f = any((tf := str(true_family).strip()).lower() == str(x).strip().lower() for x in topk_f if pd.notna(x))
            results[f'hit_family@{k}'] = 1 if hit_f else 0

    return results


def main(args):
    print("Reading files...")
    pred_df_raw = read_prediction(args.pred)
    tax_df = read_taxonomy(args.taxonomy)
    truth_df = read_truth(args.truth)

    pred_df = prepare_prediction(pred_df_raw)
    alias_to_taxid, taxid_info = build_taxonomy_maps(tax_df)

    # Build truth map: refseq_id -> (genus, species, family)
    has_family_col = 'family' in truth_df.columns
    truth_map: Dict[str, Tuple[str, str, str]] = {}
    for _, r in truth_df.iterrows():
        ref = str(r['refseq_id']).strip()
        g = str(r['genus']).strip() if pd.notna(r['genus']) else ''
        s = str(r['species']).strip() if pd.notna(r['species']) else ''
        f = str(r['family']).strip() if has_family_col and pd.notna(r['family']) else ''
        truth_map[ref] = (g, s, f)

    # Determine ordering
    use_rank = pred_df['rank_val'].notna().any()
    use_score = pred_df['score_val'].notna().any()

    if use_rank:
        pred_df['rank_val'] = pd.to_numeric(pred_df['rank_val'], errors='coerce')
        pred_df['rank_val_filled'] = pred_df['rank_val'].fillna(pred_df['rank_val'].max() + 1000 if pred_df['rank_val'].notna().any() else 1e9)
        sort_keys = ['phage_id', 'rank_val_filled']
        sort_ascending = True
    elif use_score:
        pred_df['score_val'] = pd.to_numeric(pred_df['score_val'], errors='coerce')
        pred_df['score_val_filled'] = pred_df['score_val'].fillna(-1e9)
        sort_keys = None
    else:
        sort_keys = ['phage_id']
        sort_ascending = True

    # Augment predictions with predicted taxa
    pred_aug_rows = []
    for _, row in pred_df.iterrows():
        host_id_raw = row.get('host_id_raw')
        host_species_raw = row.get('host_species_raw')

        pred_species = None
        if pd.notna(host_species_raw) and str(host_species_raw).strip().lower() != 'nan':
            pred_species = str(host_species_raw).strip()

        pred_genus = None
        if pd.notna(host_id_raw) and str(host_id_raw).strip().lower() != 'nan':
            pred_genus = map_hostid_to_genus(str(host_id_raw).strip(), alias_to_taxid, taxid_info)
        if pred_genus is None and pred_species:
            pred_genus = extract_genus_from_species_str(pred_species)

        pred_family = None
        if pd.notna(host_id_raw) and str(host_id_raw).strip().lower() != 'nan':
            pred_family = map_hostid_to_family(str(host_id_raw).strip(), alias_to_taxid, taxid_info)

        # Fallback: if pred_species is None but host_id maps to species-rank taxon
        if pred_species is None and pd.notna(host_id_raw) and str(host_id_raw).strip():
            taxid = alias_to_taxid.get(str(host_id_raw).strip())
            if taxid:
                info = taxid_info.get(taxid)
                if info and info.get('rank', '').lower() == 'species' and info.get('name'):
                    pred_species = info.get('name')

        new_row = dict(row)
        new_row['pred_species'] = pred_species
        new_row['pred_genus'] = pred_genus
        new_row['pred_family'] = pred_family
        pred_aug_rows.append(new_row)

    pred_aug_df = pd.DataFrame(pred_aug_rows)

    if sort_keys:
        pred_aug_df = pred_aug_df.sort_values(by=sort_keys, ascending=sort_ascending, kind='mergesort')
    else:
        pred_aug_df = pred_aug_df.sort_values(by=['phage_id', 'score_val_filled'], ascending=[True, False], kind='mergesort')

    phage_groups = pred_aug_df.groupby('phage_id', sort=True)

    ks = sorted([int(x) for x in args.k])
    ks = [k for k in ks if k >= 1]
    max_k = max(ks) if ks else 1

    detailed_rows = []
    comparison_rows = []  # 新增：用于存储对比信息的行
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
                'pred_family': r.get('pred_family')
            })

        true_genus = true_species = true_family = None
        if phage_id in truth_map:
            true_genus, true_species, true_family = truth_map[phage_id]
            n_phages_with_label += 1
        else:
            found = False
            if isinstance(phage_id, str):
                for sep in ['|', ' ', ';', ',']:
                    if sep in phage_id:
                        for part in phage_id.split(sep):
                            p = part.strip()
                            if p in truth_map:
                                true_genus, true_species, true_family = truth_map[p]
                                found = True
                                break
                        if found:
                            break

        hit_flags = compute_hitk_for_phage(preds_ordered, true_species, true_genus, true_family, ks)

        if true_genus is not None or true_species is not None or true_family is not None:
            for k in ks:
                summary_counts[f'hit_species@{k}'] += hit_flags[f'hit_species@{k}']
                summary_counts[f'hit_genus@{k}'] += hit_flags[f'hit_genus@{k}']
                summary_counts[f'hit_family@{k}'] += hit_flags[f'hit_family@{k}']

        # 原详细结果行
        row = {
            'phage_id': phage_id,
            'true_genus': true_genus,
            'true_species': true_species,
            'true_family': true_family,
            'n_predictions': len(preds_ordered)
        }

        topk_species = [str(x['pred_species']) if pd.notna(x['pred_species']) else '' for x in preds_ordered[:max_k]]
        topk_genus = [str(x['pred_genus']) if pd.notna(x['pred_genus']) else '' for x in preds_ordered[:max_k]]
        topk_family = [str(x['pred_family']) if pd.notna(x['pred_family']) else '' for x in preds_ordered[:max_k]]

        row[f'top{max_k}_species'] = ';'.join(topk_species)
        row[f'top{max_k}_genus'] = ';'.join(topk_genus)
        row[f'top{max_k}_family'] = ';'.join(topk_family)

        for k in ks:
            row[f'hit_species@{k}'] = hit_flags[f'hit_species@{k}']
            row[f'hit_genus@{k}'] = hit_flags[f'hit_genus@{k}']
            row[f'hit_family@{k}'] = hit_flags[f'hit_family@{k}']

        detailed_rows.append(row)

        # 新增：对比信息行 - 只显示top1预测用于对比
        if len(preds_ordered) > 0:
            top1_pred = preds_ordered[0]
            comparison_row = {
                'phage_id': phage_id,
                'true_species': true_species,
                'pred_species_top1': top1_pred.get('pred_species'),
                'true_genus': true_genus,
                'pred_genus_top1': top1_pred.get('pred_genus'),
                'true_family': true_family,
                'pred_family_top1': top1_pred.get('pred_family'),
                'hit_species@1': hit_flags.get('hit_species@1', 0),
                'hit_genus@1': hit_flags.get('hit_genus@1', 0),
                'hit_family@1': hit_flags.get('hit_family@1', 0)
            }
            comparison_rows.append(comparison_row)

    detailed_df = pd.DataFrame(detailed_rows)
    comparison_df = pd.DataFrame(comparison_rows)  # 新增对比DataFrame

    denom = n_phages_with_label if n_phages_with_label > 0 else 0

    summary_k_rows = []
    if denom == 0:
        print("Warning: no phage with known true labels found in truth file. Summary will be zeros.")
    for k in ks:
        species_count = summary_counts.get(f'hit_species@{k}', 0)
        genus_count = summary_counts.get(f'hit_genus@{k}', 0)
        family_count = summary_counts.get(f'hit_family@{k}', 0)
        species_rate = species_count / denom if denom > 0 else 0.0
        genus_rate = genus_count / denom if denom > 0 else 0.0
        family_rate = family_count / denom if denom > 0 else 0.0
        summary_k_rows.append({
            'k': k,
            'Hit_species': species_rate,
            'Count_species': species_count,
            'Hit_genus': genus_rate,
            'Count_genus': genus_count,
            'Hit_family': family_rate,
            'Count_family': family_count
        })

    summary_df = pd.DataFrame(summary_k_rows)

    out_prefix = args.out_prefix or 'result'
    out_details = f"{out_prefix}_hit_results.tsv"
    out_summary = f"{out_prefix}_hit_summary.tsv"
    out_comparison = f"{out_prefix}_label_comparison.tsv"  # 新增对比文件

    detailed_df.to_csv(out_details, sep='\t', index=False)
    summary_df.to_csv(out_summary, sep='\t', index=False)
    comparison_df.to_csv(out_comparison, sep='\t', index=False)  # 保存对比文件

    print(f"\nTotal phage predictions processed (unique phage_id): {phage_total}")
    print(f"Phages with known true label used for Hit@k (denominator): {denom}")
    print("Hit@k summary (per-k rows):")
    if not summary_df.empty:
        print(summary_df.to_string(index=False, float_format='{:0.4f}'.format))
    else:
        print("(empty)")

    print(f"\nWrote detailed results to: {out_details}")
    print(f"Wrote summary to: {out_summary}")
    print(f"Wrote label comparison to: {out_comparison}")  # 新增输出信息


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Hit@k (species, genus, family) from prediction + taxonomy + truth files.")
    parser.add_argument('--pred', required=True, help="Prediction TSV file (must contain phage_id; preferably host_id, host_species, rank or score).")
    parser.add_argument('--taxonomy', required=True, help="taxonomy TSV (columns: taxid,parent_taxid,name,rank,alias).")
    parser.add_argument('--truth', required=True, help="Truth TSV (must contain refseq_id, genus, species; optionally family).")
    parser.add_argument('--k', required=True, nargs='+', help="List of k values (e.g. --k 1 3 5)", metavar='K')
    parser.add_argument('--out_prefix', default='result', help="Prefix for output files (default: result).")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


'''
    python acc.py --pred /home/wangjingyuan/wys/inference/extracted_phages_results_species/final_predictions_all.tsv\
          --taxonomy /home/wangjingyuan/wys/wys_shiyan/taxonomy_with_alias.tsv\
            --truth phage_host.tsv\
                  --k 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30\
        --out_prefix /home/wangjingyuan/wys/inference/extracted_phages_results_species/final_predictions_accuracy_result
'''