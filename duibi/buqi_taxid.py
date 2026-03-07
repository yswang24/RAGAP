#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def is_na_like(s: pd.Series) -> pd.Series:
    # 把空串/空白/NA/NaN/None 当成缺失
    return s.isna() | s.astype(str).str.strip().str.upper().isin(["", "NA", "NAN", "NONE"])
print('ssdja')
def norm_gcf(s: pd.Series) -> pd.Series:
    # GCF 统一规范：去空白、转大写、去掉尾部版本号 .1/.2
    return (s.astype(str).str.strip().str.upper()
            .str.replace(r"\.\d+$", "", regex=True))

def main():
    ap = argparse.ArgumentParser(description="仅补齐缺失的 host_species_taxid")
    ap.add_argument("--pairs", required=True, help="成对表 TSV（含列：host_gcf, host_species_taxid）")
    ap.add_argument("--map",   required=True, help="映射表 TSV（含列：GCF, taxid）")
    ap.add_argument("--out",   required=True, help="输出 TSV 路径")
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    df = pd.read_csv(args.pairs, sep=args.sep, dtype=str, encoding=args.encoding)
    mp = pd.read_csv(args.map,   sep=args.sep, dtype=str, encoding=args.encoding)

    # 只保留映射所需列，并做规范化键
    df["_need_fill"]   = is_na_like(df["host_species_taxid"])
    df["_host_gcf_n"]  = norm_gcf(df["host_gcf"])
    mp["_GCF_n"]       = norm_gcf(mp["GCF"])
    mp = mp.drop_duplicates("_GCF_n", keep="first")[["_GCF_n", "taxid"]]

    # 仅对需要补的行做 merge
    to_fill = df.loc[df["_need_fill"], ["_host_gcf_n"]].merge(
        mp, left_on="_host_gcf_n", right_on="_GCF_n", how="left"
    )["taxid"].values

    # 写回（只填缺失）
    df.loc[df["_need_fill"], "host_species_taxid"] = to_fill

    # 清理临时列并保存
    df.drop(columns=["_need_fill", "_host_gcf_n"], errors="ignore") \
      .to_csv(args.out, sep=args.sep, index=False, encoding=args.encoding)

if __name__ == "__main__":
    main()


'''
python buqi_taxid.py \
  --pairs /home/wangjingyuan/wys/duibi/pairs_val_na.tsv \
  --map /home/wangjingyuan/wys/duibi/gcf_check_placeable_but_missing.tsv \
  --out /home/wangjingyuan/wys/duibi/pairs_val_taxid_na.tsv 
'''