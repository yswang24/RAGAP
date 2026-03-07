import os
import pandas as pd
import numexpr as ne
# ===== 读取输入文件 =====
file1 = "/home/wangjingyuan/wys/build_new_phage/newphage_predictions_cherry_2w.tsv"
file2 = "virus_host_with_GCF.tsv"
file3 = "taxid_genus.tsv"


# ===== 读取（强制为字符串） =====
df1 = pd.read_csv(file1, sep="\t", dtype=str)
df2 = pd.read_csv(file2, sep="\t", dtype=str)
df3 = pd.read_csv(file3, sep="\t", dtype=str)

# ===== 拆分 Extracted_GCFs（以分号 ; 分割） =====
df2["Extracted_GCFs"] = df2["Extracted_GCFs"].fillna("")
df2 = df2.assign(Extracted_GCFs=df2["Extracted_GCFs"].str.split(";"))
df2 = df2.explode("Extracted_GCFs")  # 多 GCF 展开为多行
df2["Extracted_GCFs"] = df2["Extracted_GCFs"].str.strip()  # 去除空格

# ===== 去除空行与重复 =====
df2 = df2[df2["Extracted_GCFs"] != ""]
df2 = df2.drop_duplicates(subset=["Extracted_GCFs", "host_taxid"])

# ===== 第一次合并：host_id ↔ Extracted_GCFs =====
merged = pd.merge(df1, df2, left_on="host_id", right_on="Extracted_GCFs", how="left")

# ===== 第二次合并：host_taxid ↔ taxid 找 genus =====
merged = pd.merge(merged, df3, left_on="host_taxid", right_on="taxid", how="left")

# ===== 选取最终列 =====
result = merged[["phage_id", "rank", "host_id", "genus", "score"]].rename(columns={"genus": "host_genus"})

# ===== 输出结果 =====
result.to_csv("/home/wangjingyuan/wys/build_new_phage/newphage_predictions_cherry_2w_genus.tsv", sep="\t", index=False)

print("✅ 输出完成：phage_host_genus.tsv")
print(f"共输出 {len(result)} 条记录。")