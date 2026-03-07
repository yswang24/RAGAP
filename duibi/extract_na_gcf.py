#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

# ---------- 工具函数 ----------

def norm_species(s: str) -> str:
    """标准化物种名：去首尾空格、压缩中间空白为单空格、小写"""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def read_table(path: Path) -> pd.DataFrame:
    """优先按制表符读取，失败再按逗号"""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        if df.shape[1] == 1:  # 只有一列，可能是逗号分隔
            df = pd.read_csv(path, sep=",", dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=",", dtype=str)
    return df

_GCF_RE = re.compile(r"^(GCF_\d+)(?:\.(\d+))?$", re.IGNORECASE)

def parse_gcf_parts(acc: str):
    """
    将 GCF accession 拆成 (base, version_int):
    - 'GCF_000006745.2' -> ('GCF_000006745', 2)
    - 'GCF_000006745'   -> ('GCF_000006745', 0)
    非 GCF 返回 (None, None)
    """
    if not acc:
        return None, None
    m = _GCF_RE.match(str(acc).strip())
    if not m:
        return None, None
    base = m.group(1)
    ver = int(m.group(2)) if m.group(2) is not None else 0
    return base, ver

def scan_prokaryote_repo(root: Path):
    """
    递归扫描 prokaryote 仓库中可用的 GCF fasta 文件。
    支持的后缀：.fa, .fna, .fasta, 及其 .gz 压缩。
    返回：
      - available_exact: set[str]  形如 'GCF_000006745' 或 'GCF_000006745.2'
      - best_by_base: dict[str -> str]  映射 'GCF_000006745' -> 'GCF_000006745.2'（版本最高者）
    """
    exts = {".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz"}
    available_exact = set()
    best_by_base_ver = dict()  # base -> (ver, chosen_name)

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        # 允许文件名形如 GCF_000006745.2.fna.gz / GCF_000006745.fa / xxx_GCF_000006745.1.fna 等少见情况
        # 优先从文件名起始提取；否则在整名里找第一个 GCF_... 片段
        m = re.match(r"^(GCF_\d+(?:\.\d+)?)", name, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(GCF_\d+(?:\.\d+)?)", name, flags=re.IGNORECASE)
        if not m:
            continue

        acc_like = m.group(1)
        # 检查后缀是否在允许集合（以便过滤掉非fasta类文件）
        suf = "".join(p.suffixes[-2:]) if "".join(p.suffixes[-2:]) in exts else "".join(p.suffixes[-1:])
        if suf.lower() not in exts:
            continue

        base, ver = parse_gcf_parts(acc_like)
        if base is None:
            continue

        # 记录 exact 形式（带版本则保留版本）
        exact = f"{base}.{ver}" if ver > 0 else base
        available_exact.add(exact)

        # 更新 base 的最佳版本
        prev = best_by_base_ver.get(base)
        if (prev is None) or (ver > prev[0]):
            best_by_base_ver[base] = (ver, exact)

    best_by_base = {base: exact for base, (ver, exact) in best_by_base_ver.items()}
    return available_exact, best_by_base

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="仅当 GCF 在 prokaryote 仓库存在时才回填 NA；否则输出未找到的 species。")
    ap.add_argument("--input_dir", required=True, help="输入目录：多个 TSV（至少含 accession,species,GCF_ids；大小写不敏感）")
    ap.add_argument("--ref_tsv", required=True, help="参考表：至少含 Accession,Species 列")
    ap.add_argument("--prokaryote_root", required=True, help="prokaryote 仓库根目录（递归扫描 GCF_*.fa|fna|fasta[.gz]）")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--input_suffix", default=".tsv", help="输入文件后缀，默认 .tsv")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ref_path = Path(args.ref_tsv)
    prok_root = Path(args.prokaryote_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读参考表并建 species -> 候选 GCF 列表（按字典序稳定）
    ref = read_table(ref_path)
    need = {"Accession", "Species"}
    if not need.issubset(ref.columns):
        raise ValueError(f"参考表缺少列 {need}；实际列：{list(ref.columns)}")

    ref["_species_norm"] = ref["Species"].map(norm_species)

    species_to_accs = defaultdict(list)
    for sp, sub in ref.groupby("_species_norm"):
        if not sp:
            continue
        accs = sorted({str(a).strip() for a in sub["Accession"].dropna() if str(a).strip()})
        # 仅保留 GCF 前缀的 accession
        accs = [a for a in accs if a.upper().startswith("GCF_")]
        if accs:
            species_to_accs[sp] = accs

    # 2) 扫描 prokaryote 仓库
    print(f"[SCAN] 扫描仓库：{prok_root} ...")
    available_exact, best_by_base = scan_prokaryote_repo(prok_root)
    print(f"[SCAN] 可用条目（exact）: {len(available_exact)}；基座可用: {len(best_by_base)}")

    # 3) 遍历输入文件
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix == args.input_suffix])
    if not files:
        print(f"[WARN] 输入目录 {in_dir} 下没有后缀 {args.input_suffix} 的文件。")

    total_na, total_filled, total_unresolved = 0, 0, 0
    summary_rows = []

    for f in files:
        df = read_table(f)
        original_cols = list(df.columns)
        lower_map = {c.lower(): c for c in df.columns}
        for must in ["accession", "species", "gcf_ids"]:
            if must not in lower_map:
                raise ValueError(f"{f.name} 缺少必要列 {must}；实际列：{list(df.columns)}")
        acc_col = lower_map["accession"]
        sp_col  = lower_map["species"]
        gcf_col = lower_map["gcf_ids"]

        # 识别 NA（真缺失 + 伪缺失）
        gcf_series = df[gcf_col]
        is_missing = gcf_series.isna()
        gcf_str = gcf_series.astype(str).str.strip()
        is_pseudo_na = gcf_str.str.upper().isin(["", "NA", "N/A", "NAN", "NONE", "NULL"])
        is_na = is_missing | is_pseudo_na

        na_rows = df[is_na].copy()
        na_count = len(na_rows)
        total_na += na_count

        na_rows["_species_norm"] = na_rows[sp_col].map(norm_species)

        filled_records = []
        na_filled_accs = []
        unresolved_species = set()

        # 针对每个 NA 行，尝试在参考表 -> 仓库命中可用 GCF
        for i, row in na_rows.iterrows():
            sp_norm = row["_species_norm"]
            if not sp_norm:
                unresolved_species.add("(empty_species)")
                continue

            candidates = species_to_accs.get(sp_norm, [])
            if not candidates:
                unresolved_species.add(str(row[sp_col]))
                continue

            # 优先 exact 命中仓库；否则按 base 匹配仓库中最高版本
            chosen = None
            # 先 exact
            for a in sorted(candidates):
                if a in available_exact:
                    chosen = a
                    break
            # 再 base -> best version in repo
            if chosen is None:
                for a in sorted(candidates):
                    base, _ = parse_gcf_parts(a)
                    if base and base in best_by_base:
                        chosen = best_by_base[base]  # 回填仓库中“该 base 的最高版本”名
                        break

            if chosen is None:
                unresolved_species.add(str(row[sp_col]))
                continue

            old_val = df.at[i, gcf_col] if gcf_col in df.columns else ""
            df.at[i, gcf_col] = chosen
            na_filled_accs.append(chosen)
            filled_records.append({
                "row_index": i,
                "accession": str(df.at[i, acc_col]),
                "species": str(row[sp_col]),
                "new_GCF_ids": chosen,
                "old_GCF_ids": str(old_val)
            })

        # 写更新后的表（保持原列顺序）
        stem = f.stem
        updated_path = out_dir / f"{stem}_updated.tsv"
        df[original_cols].to_csv(updated_path, sep="\t", index=False)

        # 只输出“原本为 NA 的行”填回的 GCF（去重）
        if na_filled_accs:
            with (out_dir / f"{stem}_NA_filled_accessions.txt").open("w", encoding="utf-8") as w:
                for a in sorted(set(na_filled_accs)):
                    w.write(f"{a}\n")

        # 未能在仓库找到对应 GCF 的 species 名称
        if unresolved_species:
            with (out_dir / f"{stem}_not_found_in_repo_species.txt").open("w", encoding="utf-8") as w:
                for sp in sorted(unresolved_species):
                    w.write(f"{sp}\n")

        # 记录明细
        if filled_records:
            pd.DataFrame(filled_records).to_csv(out_dir / f"{stem}_filled_rows.tsv", sep="\t", index=False)

        filled_cnt = len(filled_records)
        unresolved_cnt = len(unresolved_species)
        total_filled += filled_cnt
        total_unresolved += unresolved_cnt

        summary_rows.append({
            "file": f.name,
            "na_rows": na_count,
            "filled_rows": filled_cnt,
            "unresolved_species_unique": unresolved_cnt,
            "updated_tsv": str(updated_path),
            "na_filled_txt": str(out_dir / f"{stem}_NA_filled_accessions.txt") if na_filled_accs else "",
            "not_found_species_txt": str(out_dir / f"{stem}_not_found_in_repo_species.txt") if unresolved_species else "",
            "filled_log": str(out_dir / f"{stem}_filled_rows.tsv") if filled_records else ""
        })
        print(f"[{f.name}] NA={na_count}, 填充={filled_cnt}, 未在仓库找到的物种(去重)={unresolved_cnt}")

    # 汇总
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_per_file.csv"
    summary.to_csv(summary_path, index=False)

    print("\n====== 汇总 ======")
    print(f"总 NA 行数: {total_na}")
    print(f"总成功填充: {total_filled}")
    print(f"总未在仓库找到的物种(唯一): {total_unresolved}")
    print(f"按文件统计: {summary_path}")

if __name__ == "__main__":
    main()

'''
python extract_na_gcf.py \
  --input_dir /home/wangjingyuan/wys/duibi/na \
  --ref_tsv /home/wangjingyuan/wys/duibi/CHERRY/dataset/prokaryote.tsv \
  --prokaryote_root /home/wangjingyuan/wys/duibi/CHERRY/prokaryote\
  --out_dir output_na
'''