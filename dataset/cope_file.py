

# import csv

# # 读取并去重
# input_file = "host_repeat.tsv"  # 输入文件名
# output_file = "host.tsv"  # 输出文件名

# unique_rows = set()
# with open(input_file, 'r', newline='', encoding='utf-8') as infile:
#     reader = csv.reader(infile, delimiter='\t')
#     header = next(reader)  # 保存头部
#     rows = [row for row in reader if row[0] not in unique_rows and not unique_rows.add(row[0])]

# # 写入去重后的数据
# with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
#     writer = csv.writer(outfile, delimiter='\t')
#     writer.writerow(header)  # 写入头部
#     writer.writerows(rows)  # 写入去重后的内容

# print(f"去重完成，结果保存在 {output_file}")



#提取第一列生成txt文件
# import pandas as pd

# # 输入 TSV 文件路径
# tsv_file = 'phage.tsv'
# # 输出 TXT 文件路径
# txt_file = 'phage_accession.txt'

# # 读取 TSV 文件
# df = pd.read_csv(tsv_file, sep='\t')

# # 提取第一列
# first_column = df.iloc[:, 0]

# # 保存为 TXT 文件，每个值占一行
# first_column.to_csv(txt_file, index=False, header=False, sep='\n')

# print(f"第一列数据提取完成！已保存为: {txt_file}")


# #根据txt 筛选Accession
import csv
import os

# 输入文件路径
tsv_file = 'phage_host_old_all.tsv'
txt_file = 'phage_accession.txt'
output_tsv = 'phage_host1.tsv'

# 检查文件是否存在
if not os.path.exists(tsv_file) or not os.path.exists(txt_file):
    print("输入文件不存在，请检查文件路径！")
    exit()

# 读取TXT文件中的Accession号
with open(txt_file, 'r', encoding='utf-8') as txt:
    accession_set = {line.strip() for line in txt if line.strip()}

# 用于记录已写入的Accession号
written_accessions = set()

# 筛选TSV文件中的匹配行并去重
try:
    with open(tsv_file, 'r', encoding='utf-8') as tsv_in, open(output_tsv, 'w', encoding='utf-8', newline='') as tsv_out:
        reader = csv.reader(tsv_in, delimiter='\t')
        writer = csv.writer(tsv_out, delimiter='\t')
        
        for row in reader:
            if len(row) < 1:  # 检查是否有至少一列
                continue
            accession = row[0].strip()
            if accession in accession_set and accession not in written_accessions:
                writer.writerow(row)
                written_accessions.add(accession)
except Exception as e:
    print(f"处理TSV文件时发生错误：{e}")
else:
    print(f"筛选完成并去重，结果已保存到 {output_tsv}")
