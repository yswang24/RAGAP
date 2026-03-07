import pandas as pd

# 读取 taxonomy_with_alias.tsv
tax = pd.read_csv("taxonomy_with_alias.tsv", sep="\t")

# 重命名列
tax.rename(columns={'parent_taxid': 'parent'}, inplace=True)

# 先处理 parent 列：缺失值保留为 NaN，其余转成 int
if "parent" in tax.columns:
    tax["parent"] = tax["parent"].astype("Int64")  # pandas 的 nullable int 类型，可以保留 NaN

# 处理 taxid 列
tax["taxid"] = tax["taxid"].astype("int64")

# 保存为 parquet
tax.to_parquet("taxonomy_with_alias.parquet", index=False)
print("保存完成：taxonomy_with_alias.parquet")
print(tax.dtypes)
