import pandas as pd

# 读取两个 TSV 文件
file1 = "/home/wangjingyuan/wys/wys_shiyan/data_processed_new/pairs_test.tsv"   # 包含 phage_id
file2 = "/home/wangjingyuan/wys/duibi/VHM_PAIR_TAX_filter.tsv"         # 包含 accession

df1 = pd.read_csv(file1, sep='\t')
df2 = pd.read_csv(file2, sep='\t')

# 取交集
common = set(df1['phage_id']).intersection(set(df2['accession']))

# 统计数量
print(f"共有 {len(common)} 个 phage_id 与 accession 相同。")

# 如果想查看具体相同的 ID
matched = df1[df1['phage_id'].isin(common)]
print(matched.head())



