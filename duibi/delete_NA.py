import pandas as pd

inp = "/home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF.tsv"          # 原始文件
outp = "/home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF_nona.tsv"      # 过滤后的输出

df = pd.read_csv(inp, sep="\t", dtype=str, keep_default_na=False)
# 仅删除 GCF_ids 恰为 "NA" 的行（去掉首尾空格后判断）
mask = df["GCF_ids"].astype(str).str.strip() != "NA"
df_filtered = df[mask].copy()
df_filtered.to_csv(outp, sep="\t", index=False)
print(f"已输出 {len(df_filtered)} 行到 {outp}")


