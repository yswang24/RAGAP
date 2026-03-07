#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import traceback

def read_mapping(map_tsv: Path) -> dict:
    """
    读取 GCF→taxid 映射（src_id→dst_id），忽略 edge_type/weight。
    要求列：src_id, dst_id
    """
    df = pd.read_csv(map_tsv, sep="\t", dtype=str)
    # 宽容大小写/空白
    df.columns = [c.strip() for c in df.columns]
    if "src_id" not in df.columns or "dst_id" not in df.columns:
        raise ValueError("映射表必须包含列：src_id, dst_id")
    df["src_id"] = df["src_id"].str.strip()
    df["dst_id"] = df["dst_id"].str.strip()
    # 去重：保留首次出现
    df = df.drop_duplicates(subset=["src_id"], keep="first")
    return dict(zip(df["src_id"], df["dst_id"]))

def load_parquet_safe(pq_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pq_path)
    # 规范列名
    df.columns = [c.strip() for c in df.columns]
    # 确保必需列存在
    need = {"sequence_id", "embedding", "source_file"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{pq_path.name} 缺少列: {missing}")
    return df

def ensure_list(obj):
    """
    将 embedding 统一转为 Python list（避免某些 parquet 存的是 np.ndarray 或字符串）。
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, str):
        s = obj.strip()
        # 尝试解析像 "[1,2,3]" 的字符串
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                return json.loads(s.replace("(", "[").replace(")", "]"))
            except Exception:
                pass
        # 兜底：按逗号分割
        parts = [p for p in s.strip("[]() ").split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except Exception:
            return parts
    # 其他类型直接包一层
    return [obj]

def process_one_file(pq_path: Path, gcf_to_taxid: dict, out_dir: Path) -> dict:
    """
    读取一个源 parquet，生成目标 parquet。
    返回统计信息。
    """
    gcf = pq_path.stem  # e.g., "GCF_000007005"
    # 兼容稀有情况：文件名带双后缀 .fa.parquet 或中文句号
    gcf = gcf.split(".")[0]
    gcf = gcf.replace("。", ".")  # 防中文句号
    # 若文件名像 "GCF_000007005.fa" → 取前段
    if gcf.endswith(".fa"):
        gcf = gcf[:-3]

    df_src = load_parquet_safe(pq_path)

    # 映射 taxid
    taxid = gcf_to_taxid.get(gcf, None)

    # 目标 DataFrame
    tgt = pd.DataFrame({
        "host_gcf": gcf,
        "sequence_id": df_src["sequence_id"].astype(str).str.strip(),
        "host_species_taxid": taxid if taxid is not None else np.nan,
        "host_dna_emb": df_src["embedding"].map(ensure_list),
        "tangent_emb": [[] for _ in range(len(df_src))],  # 空列表，占位
    })

    out_path = out_dir / f"{gcf}.parquet"
    tgt.to_parquet(out_path, index=False)

    return {
        "gcf": gcf,
        "rows": len(tgt),
        "taxid_found": taxid is not None,
        "out": str(out_path)
    }

def main():
    ap = argparse.ArgumentParser(
        description="将 selected_fasta_na_dna 中的 GCF_*.parquet 转为目标列，并按 TSV 映射填充 host_species_taxid。")
    ap.add_argument("--input_dir", required=True, help="输入目录（含多个 GCF_*.parquet）")
    ap.add_argument("--map_tsv", required=True, help="GCF→taxid 映射 TSV（列：src_id, dst_id, ...）")
    ap.add_argument("--out_dir", required=True, help="输出目录（每个 GCF 输出一个目标 parquet）")
    ap.add_argument("--combined_out", default="", help="（可选）合并输出为一个总 parquet 的路径")
    ap.add_argument("--unmatched_report", default="", help="（可选）未匹配到 taxid 的 GCF 报告 TSV 路径")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    map_tsv = Path(args.map_tsv).expanduser().resolve()

    try:
        gcf_to_taxid = read_mapping(map_tsv)
    except Exception as e:
        print(f"[ERROR] 读取映射表失败：{e}", file=sys.stderr)
        sys.exit(1)

    stats = []
    combined_parts = []

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".parquet" and p.name.startswith("GCF_")])
    if not files:
        print(f"[WARN] 输入目录中未发现 GCF_*.parquet：{in_dir}", file=sys.stderr)

    for pq in files:
        try:
            rec = process_one_file(pq, gcf_to_taxid, out_dir)
            stats.append(rec)
            # 用于合并输出
            df_part = pd.read_parquet(rec["out"])
            combined_parts.append(df_part)
            ok = "✓" if rec["taxid_found"] else "✗"
            print(f"[DONE] {pq.name} → {rec['rows']} 行，taxid映射 {ok} → {rec['out']}")
        except Exception as e:
            print(f"[FAIL] {pq.name}: {e}", file=sys.stderr)
            traceback.print_exc()

    # 写未匹配列表（按 GCF）
    if args.unmatched_report:
        unmatched = [s["gcf"] for s in stats if not s["taxid_found"]]
        pd.DataFrame({"host_gcf": unmatched}).drop_duplicates().to_csv(args.unmatched_report, sep="\t", index=False)
        print(f"[REPORT] 未匹配 taxid 的 GCF 写至：{args.unmatched_report}（共 {len(unmatched)} 条）")

    # 合并输出
    if args.combined_out:
        if combined_parts:
            df_all = pd.concat(combined_parts, ignore_index=True)
            Path(args.combined_out).parent.mkdir(parents=True, exist_ok=True)
            df_all.to_parquet(args.combined_out, index=False)
            print(f"[COMBINED] 合并写出：{args.combined_out}（共 {len(df_all)} 行）")
        else:
            print("[COMBINED] 无可合并内容。")

if __name__ == "__main__":
    main()
'''
python build_host_parquets.py \
  --input_dir /home/wangjingyuan/wys/duibi/selected_fasta_na_dna \
  --map_tsv /home/wangjingyuan/wys/duibi/edges_na/host_taxonomy_edges.tsv \
  --out_dir host_parquets0 \
  --combined_out out_parquets0/all_hosts0.parquet \
  --unmatched_report out_parquets/unmatched_gcf.tsv
'''