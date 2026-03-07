import csv

# 定义文件路径
input_file = 'phage.tsv'
output_file = 'phage_shouzimu.tsv'

# 读取并排序文件
with open(input_file, 'r', encoding='utf-8') as infile:
    # 读取TSV文件内容
    reader = csv.reader(infile, delimiter='\t')
    # 将内容转换为列表并排序
    rows = list(reader)
    rows.sort(key=lambda x: x[0].lower())  # 按第一列的首字母排序，不区分大小写

# 将排序后的内容写入新文件
with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(rows)

print(f"文件已成功排序并保存为 {output_file}")
