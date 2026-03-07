#!/usr/bin/env python3
"""
collect_gcf_fastas.py

用途：
 从一个或多个 mapping TSV（含 GCF_ids 列）中收集所有 GCF IDs，
 在指定的 source 目录中查找对应的 GCF fasta 文件，并把找到的文件复制到 out_dir。

用法示例：
 python extract_host.py --maps VHM_PAIR_TAX_filter_new_GCF_nona.tsv TEST_PAIR_TAX_filter_new_GCF_nona.tsv \
     --source-dir /home/wangjingyuan/wys/host_fasta_final --out-dir ./cherry_host

如果 mapping 文件中 GCF 列名不是 "GCF_ids"，可以用 --gcf-col 指定列名或用列索引（0-based）。

参数：
 --maps         : 一个或多个 mapping tsv 文件（必填）
 --source-dir   : 包含大量 GCF fasta 文件的目录（必填）
 --out-dir      : 复制到的目标目录（必填）
 --gcf-col      : mapping 文件中 GCF id 所在列（默认 "GCF_ids"；也支持数字索引）
 --extensions   : 以逗号分隔的文件扩展名列表（默认 fna,fa,fasta,faa，脚本也会匹配这些后缀加 .gz）
 --dry-run      : 仅打印但不复制
"""

import argparse
import csv
import shutil
from pathlib import Path
import re
import sys

COMMON_EXTS = ["fna", "fa", "fasta", "faa"]

def parse_args():
    p = argparse.ArgumentParser(description="Collect GCF fasta files referenced by mapping TSV(s).")
    p.add_argument("--maps", nargs="+", required=True, help="One or more mapping TSV files (with a GCF_ids column).")
    p.add_argument("--source-dir", required=True, help="Directory containing GCF fasta files to search.")
    p.add_argument("--out-dir", required=True, help="Directory to copy matched fasta files into.")
    p.add_argument("--gcf-col", default="GCF_ids", help="Column name or 0-based index for GCF IDs in mapping TSV(s). Default 'GCF_ids'.")
    p.add_argument("--extensions", default="fna,fa,fasta,faa", help="Comma-separated fasta extensions to match (default: fna,fa,fasta,faa).")
    p.add_argument("--dry-run", action="store_true", help="Don't actually copy files, only report what would be done.")
    return p.parse_args()

def read_gcf_ids_from_tsv(path, gcf_col_arg):
    path = Path(path)
    gcf_ids = []
    with path.open("r", newline='') as fh:
        # try to sniff delimiter = tab
        reader = csv.reader(fh, delimiter='\t')
        rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    # detect column index
    try:
        col_idx = int(gcf_col_arg)
    except Exception:
        # try find header name (case-sensitive first, then case-insensitive)
        if gcf_col_arg in header:
            col_idx = header.index(gcf_col_arg)
        else:
            lowered = [h.lower() for h in header]
            if gcf_col_arg.lower() in lowered:
                col_idx = lowered.index(gcf_col_arg.lower())
            else:
                # fallback: assume last column is GCF ids
                col_idx = len(header) - 1
    # collect from all subsequent rows (skip header row possibly duplicated)
    for r in rows[1:]:
        if len(r) <= col_idx:
            continue
        val = r[col_idx].strip()
        if val == "":
            continue
        # if multiple GCF ids in one cell (comma/semicolon/pipe), split
        parts = re.split(r"[;,|]\s*", val)
        for p in parts:
            p = p.strip()
            if p:
                gcf_ids.append(p)
    return gcf_ids

def build_search_patterns(gcf_id, exts):
    """
    Build filename patterns (as regex) to match a given GCF id.
    We'll match files that contain the GCF id as a token, e.g. 'GCF_016728825' or 'GCF_016728825.1'
    followed by allowed extension (optionally .gz).
    Also match if file name equals gcf_id + ext, or contains it with separators.
    """
    escaped = re.escape(gcf_id)
    # token boundary: start or non-word char before, end or non-word after
    # We'll allow optional version suffix like .1 after GCF id before extension
    patterns = []
    for ext in exts:
        # e.g. r".*\bGCF_016728825(\.\d+)?\.fna(\.gz)?$"
        pat = re.compile(rf".*\b{escaped}(?:\.\d+)?\.{re.escape(ext)}(?:\.gz)?$", flags=re.IGNORECASE)
        patterns.append(pat)
    # also add pattern to match directory/file names that contain id anywhere (fallback)
    fallback = re.compile(rf".*{escaped}.*", flags=re.IGNORECASE)
    patterns.append(fallback)
    return patterns

def find_matching_file(source_dir: Path, gcf_id: str, exts):
    """
    Search under source_dir (non-recursive only top-level) for a file matching gcf_id.
    If not found at top-level, perform a recursive search.
    Returns Path or None.
    """
    patterns = build_search_patterns(gcf_id, exts)
    # First try non-recursive top level for speed
    for p in source_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if any(pat.match(name) for pat in patterns[:-1]):  # prefer strict patterns
            return p
    # try recursive with strict patterns
    for pat in patterns[:-1]:
        for p in source_dir.rglob(f"*"):
            if not p.is_file():
                continue
            if pat.match(p.name):
                return p
    # fallback: any file whose name contains gcf id
    for p in source_dir.rglob(f"*"):
        if not p.is_file():
            continue
        if patterns[-1].match(p.name):
            return p
    return None

def main():
    args = parse_args()
    maps = [Path(p) for p in args.maps]
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    exts = [e.strip().lstrip(".") for e in args.extensions.split(",") if e.strip()]
    if not exts:
        exts = COMMON_EXTS

    if not source_dir.exists():
        print(f"ERROR: source dir does not exist: {source_dir}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    # collect GCF ids from all maps
    gcf_set = []
    for m in maps:
        if not m.exists():
            print(f"WARNING: map file not found: {m}", file=sys.stderr)
            continue
        ids = read_gcf_ids_from_tsv(m, args.gcf_col)
        print(f"Read {len(ids)} GCF ids from {m}")
        gcf_set.extend(ids)
    # deduplicate while preserving order
    seen = set()
    gcf_list = []
    for g in gcf_set:
        if g not in seen:
            seen.add(g)
            gcf_list.append(g)

    print(f"Total unique GCF ids to find: {len(gcf_list)}")

    copied = []
    missing = []
    manifest_rows = []

    for gcf in gcf_list:
        found = find_matching_file(source_dir, gcf, exts)
        if found:
            dest = out_dir / found.name
            if args.dry_run:
                action = "DRY-COPY"
            else:
                # if a file with same name exists, append a suffix to avoid overwrite
                if dest.exists():
                    # generate unique name
                    base = dest.stem
                    suffix = 1
                    while True:
                        candidate = out_dir / f"{base}_{suffix}{dest.suffix}"
                        if not candidate.exists():
                            dest = candidate
                            break
                        suffix += 1
                shutil.copy2(found, dest)
            copied.append((gcf, str(found), str(dest)))
            manifest_rows.append((gcf, str(found), str(dest)))
        else:
            missing.append(gcf)
    # write manifest
    manifest_path = out_dir / "copied_manifest.tsv"
    with manifest_path.open("w", newline='') as mf:
        w = csv.writer(mf, delimiter="\t")
        w.writerow(["GCF_id", "source_path", "dest_path"])
        for row in manifest_rows:
            w.writerow(row)

    # summary
    print("=== SUMMARY ===")
    print(f"Requested GCF ids: {len(gcf_list)}")
    print(f"Found & (copied) : {len(copied)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("Missing list (first 50):")
        for x in missing[:50]:
            print(" ", x)
    print(f"Manifest written to: {manifest_path}")
    print("Done.")

if __name__ == '__main__':
    main()
