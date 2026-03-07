# import pandas as pd

# # 读取TSV文件
# # df = pd.read_csv('phagesmiao.tsv', sep='\t', header=None)
# df = pd.read_csv('phagesmiao.csv', sep=',', header=None)
# # 获取最后一列的名称
# last_column = df.iloc[:, -1]

# # 获取不同的名称（种类）
# unique_names = last_column.nunique()

# # 输出不同名称的数量
# print(f"不同的名称（种类）数量: {unique_names}")


import pandas as pd

# 读取TSV文件
df = pd.read_csv("sample200.tsv", sep="\t", header=None)

# 提取最后一列
last_column = df.iloc[:, -1]

# 获取唯一的物种名称
unique_species = last_column.unique()

# 统计不同的物种数量
species_count = len(unique_species)
print(f"不同的物种数量: {species_count}")

# 将不同的物种名称保存到TXT文件
with open("unique_species.txt", "w") as f:
    for species in unique_species:
        f.write(str(species) + "\n")

print("物种名称已保存到 unique_species.txt")
