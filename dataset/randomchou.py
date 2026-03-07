import pandas as pd
import random

# 读取TSV文件
df = pd.read_csv('filtered_unique.tsv', sep='\t', header=None)

# 随机抽取200行
random_rows = df.sample(n=200, random_state=42)

# 将随机抽取的内容保存为新的TSV文件
random_rows.to_csv('sample200.tsv', sep='\t', header=False, index=False)

# 将第一列单独保存为TXT文件
first_column = random_rows.iloc[:, 0]
first_column.to_csv('sample200.txt', index=False, header=False)
