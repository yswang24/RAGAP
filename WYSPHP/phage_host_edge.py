# import pandas as pd

# # === 配置输入输出文件路径 ===
# input_file = "virus_host_with_GCF.tsv"   # 你的原始 TSV
# output_file = "phage_host_edges.tsv"

# # === 读取数据 ===
# df = pd.read_csv(input_file, sep="\t")

# # === 转换格式 ===
# edges_df = pd.DataFrame({
#     "src_id": df["refseq_id"],
#     "dst_id": df["Extracted_GCFs"],
#     "edge_type": "phage-host",
#     "weight": 1
# })

# # === 保存 ===
# edges_df.to_csv(output_file, sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条 phage-host 边到 {output_file}")



# import pandas as pd

# # === 配置输入输出文件路径 ===
# input_file = "/home/wangjingyuan/wys/wys_shiyan/host_protein_edges.parquet"  # 你的 parquet 文件
# output_file = "host_protein_edges.tsv"

# # === 读取 parquet 文件 ===
# df = pd.read_parquet(input_file)

# # === 转换成标准边格式 ===
# edges_df = pd.DataFrame({
#     "src_id": df["host_id"],
#     "dst_id": df["protein_id"],
#     "edge_type": "host-protein",
#     "weight": 1
# })

# # === 保存为 TSV ===
# edges_df.to_csv(output_file, sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条 phage-protein 边到 {output_file}")


# import pandas as pd

# # === 输入输出路径 ===
# input_file = "/home/wangjingyuan/wys/WYSPHP/taxonomy_with_alias.tsv"  # 你的输入文件
# output_file = "taxonomy_taxonomy_edges.tsv"

# # === 用字符串读取，避免把 taxid 当成浮点数 ===
# df = pd.read_csv(input_file, sep="\t", dtype=str)
# # 检查 parent_taxid 为空的行
# empty_parent = df[df["parent_taxid"].isna() | (df["parent_taxid"].str.strip() == "")]
# if len(empty_parent) > 0:
#     print(f"⚠️ 发现 {len(empty_parent)} 行 parent_taxid 为空，打印前 10 行：")
#     print(empty_parent.head(10).to_string(index=False))

# # 转换成边格式，保留空值行（用空字符串表示）
# edges_df = pd.DataFrame({
#     "src_id": df["taxid"].fillna(""),       # 保证没有 NaN
#     "dst_id": df["parent_taxid"].fillna(""),
#     "edge_type": "taxonomy-parent",
#     "weight": "1"
# })

# # 保存为 TSV
# edges_df.to_csv(output_file, sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条 taxonomy-parent 边到 {output_file}")



# import pandas as pd

# # === 输入输出路径 ===
# input_file = "/home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF.tsv"  # 你的输入 TSV
# output_file = "host_taxonomy_edges.tsv"

# # === 读取 TSV 文件 ===
# df = pd.read_csv(input_file, sep="\t", dtype=str)  # dtype=str 避免数字变浮点

# # 转换成标准边格式
# edges_df = pd.DataFrame({
#     "src_id": df["Extracted_GCFs"].fillna(""),
#     "dst_id": df["host_taxid"].fillna(""),
#     "edge_type": "host-taxonomy",
#     "weight": "1"
# })
# edges_df = edges_df.drop_duplicates(subset=["src_id", "dst_id", "edge_type"])

# # 保存为 TSV
# edges_df.to_csv(output_file, sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条 host-taxonomy 边到 {output_file}")


import pandas as pd
from pathlib import Path

# 配置：输入/输出文件
input_path = "/home/wangjingyuan/wys/wys_shiyan/protein_clusters_emb.parquet"   # <- 改成你的输入文件名
output_path = "/home/wangjingyuan/wys/wys_shiyan/protein_clusters_emb_new.parquet"  # <- 改成你想要的输出文件名

# 读取 parquet
df = pd.read_parquet(input_path)

# 检查列是否存在
if "cluster_id" not in df.columns:
    raise KeyError("输入文件中找不到列 'cluster_id'，请确认列名。")

# 1) 将原来的 cluster_id 改名为 cluster_id_source
df = df.rename(columns={"cluster_id": "cluster_id_source"})

# 2) 新增 cluster_id（取第一个 '|' 之前的部分；若没有 '|' 则取整段）
# 先把 NaN 转成空字符串以免出错
df["cluster_id_source"] = df["cluster_id_source"].fillna("").astype(str)
df["cluster_id"] = df["cluster_id_source"].str.split("|", n=1).str[0]

# 3) 把新列放到最前面，后面紧跟 cluster_id_source，其它列保持原有顺序
other_cols = [c for c in df.columns if c not in ("cluster_id", "cluster_id_source")]
new_order = ["cluster_id", "cluster_id_source"] + other_cols
df = df.loc[:, new_order]

# 4) 保存为 parquet
df.to_parquet(output_path, index=False)

# 打印检查信息
print(f"✅ 已保存: {Path(output_path).absolute()}")
print(f"行数: {len(df)}，列顺序：{df.columns.tolist()}")
print("前 5 行预览：")
print(df.head(5).to_string(index=False))
