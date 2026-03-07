import pandas as pd

# ===== 输入输出路径 =====
input_tsv = "/home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF.tsv"          # 原始 TSV 文件路径
output_tsv = "phage_taxonomy_edges.tsv"  # 转换后输出路径

# ===== 读取文件 =====
df = pd.read_csv(input_tsv, sep='\t')

# ===== 构建新的DataFrame =====
edges = pd.DataFrame({
    'src_id': df['refseq_id'],
    'dst_id': df['virus_taxid'],
    'edge_type': ['phage-taxonomy'] * len(df),
    'weight': [1] * len(df)
})

# ===== 输出结果 =====
edges.to_csv(output_tsv, sep='\t', index=False)

print(f"已生成文件: {output_tsv}")
print(edges.head())
