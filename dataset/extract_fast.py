import os
import shutil

# 设置相关路径
txt_file = 'phage_accession.txt'  # 包含Accession号的TXT文件
fasta_folder = 'split_sequences'  # 原始FASTA文件夹
output_folder = 'phage_fasta'  # 新的文件夹，用来存放筛选的FASTA文件

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取TXT文件中的Accession号
accession_ids = set()  # 使用集合来存储Accession号，去重

with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        # 按逗号分割Accession号
        accession_list = line # 使用逗号分割
        accession_ids.update([acc for acc in accession_list])  # 去除可能的空格并加入集合

# 遍历FASTA文件夹中的所有文件
for file_name in os.listdir(fasta_folder):
    if file_name.endswith('.fasta'):  # 只处理.fasta文件
        accession = file_name.split('.')[0]  # 提取文件名中的Accession号（假设是文件名的第一个部分）

        # 如果Accession号在TXT文件中，复制该FASTA文件到新的文件夹
        if accession in accession_ids:
            source_path = os.path.join(fasta_folder, file_name)
            destination_path = os.path.join(output_folder, file_name)
            shutil.copy(source_path, destination_path)
            print(f"已复制: {file_name}")

print("操作完成，相关FASTA文件已复制到新的文件夹。")


# #拆分Accession
# import csv

# # 设置输入输出文件路径
# input_file = 'phage_repeat.tsv'  # 输入的TSV文件
# output_file = 'phage.tsv'  # 输出的TSV文件

# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
#     reader = csv.reader(infile, delimiter='\t')  # 读取TSV文件
#     writer = csv.writer(outfile, delimiter='\t')  # 写入TSV文件

#     for row in reader:
#         # 处理第一列（Accession号），假设第一列是Accession号
#         accession_column = row[0].strip()  # 获取第一列并去除多余空格
#         other_columns = row[1:]  # 获取其余列

#         # 如果Accession号中包含逗号，拆分Accession号
#         accession_list = accession_column.split(',')

#         # 对每个拆分后的Accession号，写入新行，保留其余列的内容
#         for accession in accession_list:
#             new_row = [accession.strip()] + other_columns  # 将Accession号与其它列内容合并
#             writer.writerow(new_row)  # 写入新的行

# print(f"操作完成，输出文件已保存为 {output_file}")



# # 读取TSV文件并提取第一列
# input_file = 'phage.tsv'  # 替换为你的TSV文件路径
# output_file = 'phage_accession.txt'  # 输出的TXT文件路径

# with open(input_file, 'r', encoding='utf-8') as infile:
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             # 按制表符分割每行，提取第一列
#             columns = line.strip().split('\t')
#             # 将第一列写入输出文件
#             outfile.write(columns[0] + '\n')

# print("第一列已成功提取并保存到 output.txt")
