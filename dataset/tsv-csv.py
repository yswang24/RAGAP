import csv

def tsv_to_csv(input_tsv_path, output_csv_path):
    # 打开TSV文件进行读取，同时打开CSV文件进行写入
    with open(input_tsv_path, 'r', encoding='utf-8') as tsv_file, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        
        # 创建一个TSV读取器和CSV写入器
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        csv_writer = csv.writer(csv_file)
        
        # 逐行读取TSV文件并写入CSV文件
        for row in tsv_reader:
            csv_writer.writerow(row)

# 示例调用
input_tsv_path = 'filtered_unique.tsv'  # 替换为你的输入TSV文件路径
output_csv_path = 'filtered_unique.csv'  # 替换为你想要保存的输出CSV文件路径
tsv_to_csv(input_tsv_path, output_csv_path)

print(f"已成功将 {input_tsv_path} 转换为 {output_csv_path}")