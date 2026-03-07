import pandas as pd

# ===== 输入文件 =====
file1 = "/home/wangjingyuan/wys/build_new_phage/newphage_predictions_cherry_2w.tsv"
file3 = "taxid_genus.tsv"
output_path = "/home/wangjingyuan/wys/build_new_phage/newphage_predictions_cherry_2w_genus.tsv"

# ===== 读取 TSV 文件（强制字符串） =====
df1 = pd.read_csv(file1, sep="\t", dtype=str)
df3 = pd.read_csv(file3, sep="\t", dtype=str)

# ===== 合并 taxid → genus =====
merged = pd.merge(df1, df3, left_on="host_species_taxid", right_on="taxid", how="left")

# ===== 生成新列并重命名 =====
result = merged.rename(
    columns={
        "host_species_taxid": "host_genus_taxid",
        "genus": "host_genus_name"
    }
)

# ===== 选取最终输出列（按指定顺序） =====
result = result[["phage_id", "host_genus_taxid", "host_genus_name", "score", "rank"]]

# ===== 输出结果 =====
result.to_csv(output_path, sep="\t", index=False)

print(f"✅ 输出完成：{output_path}")
print(f"共输出 {len(result)} 条记录。")
