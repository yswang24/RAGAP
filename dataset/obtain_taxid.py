# import pandas as pd

# # 读取TSV文件
# file_path = 'assembly_summary_bacteria.tsv'  # 替换为你的文件路径
# df = pd.read_csv(file_path, sep='\t')

# # 保留第一列、第六列和第十二列
# df_filtered = df.iloc[:, [0, 5, 11]]

# # 筛选第十二列中包含 "Complete Genome" 的数据
# df_filtered = df_filtered[df_filtered.iloc[:, 2] == 'Complete Genome']

# # 将结果保存到新文件
# df_filtered.to_csv('complete_genome_bacteria.tsv', sep='\t', index=False)


import pandas as pd

# 读取TSV文件
file_path = 'complete_genome_bacteria.tsv'  # 替换为你的文件路径
df = pd.read_csv(file_path, sep='\t')

# 保留第二列（列索引为1）
second_column = df.iloc[:, 1]

# 将第二列的数据保存到TXT文件
output_path = 'bacteria_taxid.txt'  # 设置输出文件路径
second_column.to_csv(output_path, index=False, header=False, sep='\n')
