#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import pandas as pd


def split_concat_id(val: str):
    """
    兼容形如 'GCF_013389765NZ_CP058322.1_1' 的单列拼接写法：
    - host_id 识别为以 GCF_ 开头的编号 (GCF_ + 数字)
    - protein_id 识别为其后的 RefSeq/GenBank 常见前缀（如 NZ_, NC_, CP 等）开头的剩余部分
    """
    if not isinstance(val, str):
        return None, None
    val = val.strip()
    # 常见蛋白/基因组条目前缀：NZ_, NC_, CP, CM, AE, AL 等；示例中为 NZ_CP...
    m = re.match(r'^(GCF_\d+)([A-Z]{2}_.+)$', val)
    if m:
        return m.group(1), m.group(2)
    # 宽松兜底：只要能提取到 GCF_XXXX，其余作为 protein_id
    m2 = re.match(r'^(GCF_\d+)(.+)$', val)
    if m2:
        return m2.group(1), m2.group(2).lstrip('_')
    return None, None


def main():
    ap = argparse.ArgumentParser(description="将(host_id, protein_id)整理为(src_id, dst_id, edge_type, weight)")
    ap.add_argument("--parquet", required=True, help="输入 Parquet 文件路径")
    ap.add_argument("--out", required=True, help="输出 TSV 文件路径")
    ap.add_argument("--host-col", default="host_id", help="host_id 列名（默认 host_id）")
    ap.add_argument("--prot-col", default="protein_id", help="protein_id 列名（默认 protein_id）")
    ap.add_argument("--concat-col", default=None, help="当只有一列拼接时的列名（如 GCF_xxxNZ_xxx），自动拆分")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)

    # 情况 A：标准两列
    if args.host_col in df.columns and args.prot_col in df.columns:
        tmp = df[[args.host_col, args.prot_col]].copy()
        tmp = tmp.dropna()
        tmp = tmp.rename(columns={args.host_col: "src_id", args.prot_col: "dst_id"})
    # 情况 B：只有一列拼接（自动拆）
    elif args.concat_col and args.concat_col in df.columns:
        tmp = df[[args.concat_col]].copy().dropna()
        split_res = tmp[args.concat_col].apply(split_concat_id)
        tmp["src_id"] = split_res.apply(lambda x: x[0])
        tmp["dst_id"] = split_res.apply(lambda x: x[1])
        tmp = tmp.drop(columns=[args.concat_col])
        # 去除无法拆分的行
        before = len(tmp)
        tmp = tmp.dropna(subset=["src_id", "dst_id"])
        removed = before - len(tmp)
        if removed > 0:
            print(f"[提示] 有 {removed} 行无法自动拆分，已丢弃。")
    else:
        raise ValueError(
            f"找不到所需列：两列模式需要 '{args.host_col}' 与 '{args.prot_col}'；"
            f"或提供 --concat-col 并确保该列存在。当前列为：{list(df.columns)}"
        )

    # 生成目标四列
    tmp["edge_type"] = "host-protein"
    tmp["weight"] = 1

    # 排序与去重（可选）
    tmp = tmp[["src_id", "dst_id", "edge_type", "weight"]].drop_duplicates()

    # 输出 TSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.to_csv(out_path, sep="\t", index=False)
    print(f"[完成] 已写出：{out_path}，共 {len(tmp)} 条边。")


if __name__ == "__main__":
    main()




'''
python host_protein.py \
  --parquet /home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/host_protein_edges.parquet \
  --out /home/wangjingyuan/wys/duibi/host_protein_edges.tsv \
  --host-col host_id \
  --prot-col protein_id
  
  python host_protein.py \
  --parquet /home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_protein_edges.parquet \
  --out /home/wangjingyuan/wys/duibi/phage_protein_edges.tsv \
  --host-col phage_id \
  --prot-col protein_id
  
  '''