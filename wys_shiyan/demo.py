# import pandas as pd
# h = pd.read_parquet('data_processed/host_catalog.parquet')
# p = pd.read_parquet('data_processed/phage_catalog.parquet')
# pairs = pd.read_csv('/home/wangjingyuan/wys/wys_shiyan/data_processed/random_data/pairs_all.tsv', sep='\t')

# print("hosts:", len(h))
# print("phage:", len(p))
# print("pairs total:", len(pairs))
# print("pairs unique phage:", pairs['phage_id'].nunique())
# print("pairs unique host_gcf:", pairs['host_gcf'].nunique())

# # check embedding dims
# print("host tangent dim (example):", len(h['tangent_emb'].dropna().iloc[0]))
# print("host dna dim (example):", len(h['host_dna_emb'].dropna().iloc[0]))
# print("phage dna dim (example):", len(p['phage_dna_emb'].dropna().iloc[0]))


import pandas as pd

# 读取原始 tsv 文件
df = pd.read_csv("protein_protein_edges.tsv", sep="\t")

# 处理 src_id 和 dst_id，保留第一个 "|" 之前的内容
df["src_id"] = df["src_id"].str.split("|").str[0]
df["dst_id"] = df["dst_id"].str.split("|").str[0]

# 保存结果
df.to_csv("protein_protein_edges.tsv", sep="\t", index=False)

print(f"✅ 已处理完成，保存到 protein_protein_edges_clean.tsv，行数: {len(df)}")
print(df.head())