import pandas as pd
import pyarrow.parquet as pq

# 文件路径
tsv_file = "sequence_ids_count_phage.tsv"          # 你的 TSV 文件
parquet_file = "/home/wangjingyuan/wys/WYSPHP/RBP_650/protein_phage_catalog_650.parquet"   # 你的 Parquet 文件
output_file = "/home/wangjingyuan/wys/WYSPHP/RBP_650/protein_cluster_phage_catalog_650.parquet"  # 输出匹配行
not_found_file = "not_found_phage_ids.tsv"

# 1️⃣ 读取 TSV
tsv_df = pd.read_csv(tsv_file, sep="\t")
sequence_ids = set(tsv_df['Sequence_ID'])

# 2️⃣ 打开 Parquet 文件
pf = pq.ParquetFile(parquet_file)
matched_dfs = []

for i in range(pf.num_row_groups):
    # 读取每个 row group
    table = pf.read_row_group(i)
    df = table.to_pandas()
    matched_chunk = df[df['protein_id'].isin(sequence_ids)]
    if not matched_chunk.empty:
        matched_dfs.append(matched_chunk)

# 合并所有匹配结果
if matched_dfs:
    matched_df = pd.concat(matched_dfs, ignore_index=True)
else:
    matched_df = pd.DataFrame(columns=['protein_id', 'source_type', 'source_id', 'embedding'])

# 统计匹配情况
found_ids = set(matched_df['protein_id'])
not_found_ids = sequence_ids - found_ids

print(f"✅ 匹配到的序列行数: {len(matched_df)}")
print(f"❌ TSV 中未找到的序列数: {len(not_found_ids)}")

# 保存结果
matched_df.to_parquet(output_file, index=False)
pd.DataFrame({"Sequence_ID": list(not_found_ids)}).to_csv(not_found_file, sep="\t", index=False)

print(f"✅ 匹配行已保存到: {output_file}")
print(f"✅ 未匹配序列已保存到: {not_found_file}")