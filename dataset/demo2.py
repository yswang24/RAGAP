# #csv转tsv
# import csv

# # 定义 CSV 和 TSV 文件的路径
# csv_file = 'bacteria_lineage.csv'
# tsv_file = 'bacteria_lineage.tsv'

# # 读取 CSV 文件并写入 TSV 文件
# with open(csv_file, mode='r', newline='', encoding='utf-8') as csv_f:
#     reader = csv.reader(csv_f)
#     # 使用 'tab' 字符作为分隔符，写入 TSV 文件
#     with open(tsv_file, mode='w', newline='', encoding='utf-8') as tsv_f:
#         writer = csv.writer(tsv_f, delimiter='\t')
#         for row in reader:
#             writer.writerow(row)

# print("CSV 文件已成功转换为 TSV 文件。")

#添加

# import csv

# # 定义 TSV 文件的路径
# tsv_file1 = 'phage_old.tsv'
# tsv_file2 = 'virus_lineage_new.tsv'
# output_file = 'phage_merge.tsv'

# # 读取 TSV 文件 1 的第一列
# with open(tsv_file1, mode='r', newline='', encoding='utf-8') as file1:
#     reader1 = csv.reader(file1, delimiter='\t')
#     file1_first_column = [row[0] for row in reader1]  # 读取第一列

# # 读取 TSV 文件 2，并将文件 1 的第一列添加到文件 2 的第一列
# with open(tsv_file2, mode='r', newline='', encoding='utf-8') as file2:
#     reader2 = csv.reader(file2, delimiter='\t')
#     rows2 = [row for row in reader2]

# # 确保文件 1 和 文件 2 的行数相同
# if len(file1_first_column) != len(rows2):
#     raise ValueError("文件 1 和 文件 2 的行数不相同，无法合并")

# # 合并文件 1 的第一列和文件 2 的其他列
# merged_rows = []
# for i in range(len(rows2)):
#     merged_rows.append([file1_first_column[i]] + rows2[i])

# # 将合并后的数据写入新的 TSV 文件
# with open(output_file, mode='w', newline='', encoding='utf-8') as output:
#     writer = csv.writer(output, delimiter='\t')
#     writer.writerows(merged_rows)

# print("文件已成功合并，并保存为", output_file)





#删除taxid列


import csv

# 定义输入和输出文件的路径
input_file = 'phage_new.tsv'
output_file = 'phage_new.tsv'

# 打开并读取 TSV 文件
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile, delimiter='\t')
    
    # 读取表头
    header = next(reader)
    
    # 检查 'col2' 是否在表头中
    if 'col2' in header:
        col2_index = header.index('col2')  # 获取 'col2' 列的索引
        header.remove('col2')  # 删除 'col2' 列的列名
    else:
        print("没有找到 col2 列。")
        col2_index = None

    # 读取剩余数据并删除 'col2' 列
    rows = []
    for row in reader:
        if col2_index is not None:
            del row[col2_index]  # 删除 'col2' 列
        rows.append(row)

# 将修改后的数据写入新的 TSV 文件
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(header)  # 写入表头
    writer.writerows(rows)  # 写入数据行

print(f"文件已成功保存为 {output_file}，'col2' 列已删除。")


