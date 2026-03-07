import pandas as pd

# ===== 输入文件路径 =====
file1 = "/home/wangjingyuan/wys/WYSPHP/RBP_650/protein_cluster_host_catalog_650.parquet"
file2 = "/home/wangjingyuan/wys/WYSPHP/RBP_650/protein_cluster_phage_catalog_650.parquet"
output_file = "/home/wangjingyuan/wys/WYSPHP/RBP_650/cluster_phage_host_catalog_650.parquet"

# ===== 读取两个Parquet文件 =====
df1 = pd.read_parquet(file1)
df2 = pd.read_parquet(file2)

# ===== 合并（按行拼接）=====
merged = pd.concat([df1, df2], ignore_index=True)

# ===== 保存合并后的文件 =====
merged.to_parquet(output_file, index=False)

print(f"合并完成！输出文件: {output_file}")
print(f"合并后行数: {len(merged)}")
