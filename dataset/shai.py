#从tsv最后一列数据中筛选出txt文件对应的每一行数据  tsv文件对应的行信息不变
import pandas as pd

# 读取 TSV 文件
df = pd.read_csv("phage.tsv", sep="\t", header=None)

# 读取 TXT 文件，获取筛选条件（物种名称）
with open("unique_species.txt", "r") as f:
    species_list = set(line.strip() for line in f)

# 筛选符合条件的行
filtered_df = df[df.iloc[:, -1].isin(species_list)]

# 保存筛选后的数据
filtered_df.to_csv("filtered_unique.tsv", sep="\t", index=False, header=False)

print(f"筛选完成，共筛选出 {filtered_df.shape[0]} 行数据，结果已保存到 filtered_output.tsv")
