#!/usr/bin/env python3
"""
edge_to_host_taxid.py

将边表 (src_id dst_id edge_type weight) 转换为:
phage_id  host_gcf  host_species_taxid  label  source

使用 taxonomy_with_alias.tsv 的 alias 列（包含 GCF ids）来查找 host_species_taxid。
支持 dst_id 单元格里有多个 GCF（用 ; 分隔），会展开为多行。

用法:
python edge_to_host_taxid.py --edges edges.tsv --taxonomy taxonomy_with_alias.tsv --out output.tsv

可选:
--gcf-col-name  指定边表里 GCF 列名 (默认 dst_id)
--weight-col    指定权重/label 列名 (默认 weight)
--source-name   覆盖 source 字段，默认使用 edges 文件的 basename
"""
import argparse
from pathlib import Path
import pandas as pd
import re
import csv
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", required=True, help="边文件 (tsv) 包含 src_id dst_id edge_type weight")
    p.add_argument("--taxonomy", required=True, help="taxonomy_with_alias.tsv，包含 taxid,name,rank,alias 等列")
    p.add_argument("--out", required=True, help="输出 tsv 文件")
    p.add_argument("--gcf-col-name", default="dst_id", help="边文件中表示 host GCF 的列名 (默认 dst_id)")
    p.add_argument("--weight-col", default="weight", help="边文件中表示权重/label 的列名 (默认 weight)")
    p.add_argument("--src-col", default="src_id", help="边文件中 src 列名 (默认 src_id)")
    p.add_argument("--source-name", default=None, help="写入 output 的 source 字段，默认为 edges 文件名")
    return p.parse_args()

def build_alias_map(taxonomy_df):
    """
    Build a map: alias_token -> taxid
    alias field might contain multiple tokens separated by ; , | whitespace. We extract tokens that look like GCF_* 
    but also map any token in alias to the taxid as fallback.
    """
    alias_map = {}
    # iterate rows
    for _, row in taxonomy_df.iterrows():
        taxid = str(row.get("taxid") if "taxid" in taxonomy_df.columns else row.iloc[0]).strip()
        alias_field = row.get("alias", "")
        if pd.isna(alias_field) or str(alias_field).strip() == "":
            continue
        alias_str = str(alias_field)
        # split by common separators ; , | whitespace
        parts = re.split(r"[;,|\s]+", alias_str)
        for part in parts:
            token = part.strip()
            if not token:
                continue
            # keep mapping for token -> taxid (prefer exact token)
            alias_map[token] = taxid
            # also if token contains 'GCF_' possibly with version or suffix, map normalized form
            # e.g. GCF_000005845.1  -> map GCF_000005845
            m = re.match(r"^(GCF_[0-9]+)(?:\.\d+)?$", token, flags=re.IGNORECASE)
            if m:
                alias_map[m.group(1)] = taxid
    return alias_map

def normalize_gcf_token(tok):
    """normalize common forms, strip whitespace and possible .1 version suffix"""
    tok = str(tok).strip()
    if not tok:
        return tok
    # remove surrounding quotes
    tok = tok.strip('"').strip("'")
    m = re.match(r"^(GCF_[0-9]+)(?:\.\d+)?$", tok, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return tok

def main():
    args = parse_args()
    edges_path = Path(args.edges)
    tax_path = Path(args.taxonomy)
    out_path = Path(args.out)

    if not edges_path.exists():
        print("edges file not found:", edges_path, file=sys.stderr)
        sys.exit(2)
    if not tax_path.exists():
        print("taxonomy file not found:", tax_path, file=sys.stderr)
        sys.exit(2)

    # read taxonomy table (try tab delim)
    tax_df = pd.read_csv(tax_path, sep="\t", dtype=str, low_memory=False)
    tax_df.columns = [c.strip() for c in tax_df.columns]

    # build alias->taxid map
    alias_map = build_alias_map(tax_df)

    # read edges
    edges_df = pd.read_csv(edges_path, sep="\t", dtype=str, low_memory=False)
    edges_df.columns = [c.strip() for c in edges_df.columns]

    src_col = args.src_col
    gcf_col = args.gcf_col_name
    weight_col = args.weight_col

    if src_col not in edges_df.columns:
        raise KeyError(f"src column '{src_col}' not found in edges file. Columns: {list(edges_df.columns)}")
    if gcf_col not in edges_df.columns:
        raise KeyError(f"gcf column '{gcf_col}' not found in edges file. Columns: {list(edges_df.columns)}")
    if weight_col not in edges_df.columns:
        # if weight absent, create default 1
        edges_df[weight_col] = "1"

    # explode rows if gcf cell contains multiple GCF ids separated by ;
    # allow separators ; , | whitespace
    def split_gcf_cell(cell):
        if pd.isna(cell):
            return []
        parts = re.split(r"[;,|]\s*", str(cell))
        return [p for p in (x.strip() for x in parts) if p]

    expanded_rows = []
    missing_gcf = set()
    for _, row in edges_df.iterrows():
        src = str(row[src_col]).strip()
        weight = str(row[weight_col]).strip()
        gcf_cell = row[gcf_col]
        gcf_list = split_gcf_cell(gcf_cell)
        if not gcf_list:
            # skip if no gcf
            continue
        for gcf in gcf_list:
            gcf_norm = normalize_gcf_token(gcf)
            taxid = alias_map.get(gcf_norm)
            if taxid is None:
                # try with original gcf (case-insensitive)
                taxid = alias_map.get(gcf_norm.upper()) or alias_map.get(gcf_norm.lower())
            if taxid is None:
                missing_gcf.add(gcf_norm)
            expanded_rows.append({
                "phage_id": src,
                "host_gcf": gcf_norm,
                "host_species_taxid": taxid if taxid is not None else "",
                "label": weight,
                "source": args.source_name if args.source_name else edges_path.name
            })

    # write output
    out_df = pd.DataFrame(expanded_rows, columns=["phage_id","host_gcf","host_species_taxid","label","source"])
    out_df.to_csv(out_path, sep="\t", index=False)

    # write missing list
    missing_path = out_path.with_name(out_path.stem + "_missing_gcf_ids.txt")
    with open(missing_path, "w") as mh:
        for m in sorted(missing_gcf):
            mh.write(f"{m}\n")

    print("Wrote output:", out_path)
    print("Missing GCF ids (written to):", missing_path)
    print("Total output rows:", len(out_df))
    print("Total unique missing GCF ids:", len(missing_gcf))

if __name__ == "__main__":
    main()
'''
python generate_train_pair.py --edges VHM_PAIR_TAX_filter_new_GCF_na_edges.tsv --taxonomy taxonomy_with_alias.tsv --out pairs_train_na.tsv
'''