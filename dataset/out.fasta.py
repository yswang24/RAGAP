
import os

# 输入文件夹路径和TXT文件路径
folder_path = 'phage_fasta'  # 这里替换为你的文件夹路径
txt_file_path = 'sample200.txt'  # 这里替换为你的TXT文件路径
output_fasta_path = 'phages_sample200.fasta'  # 合成后的输出文件路径

# 从TXT文件读取名称（去掉文件后缀）
with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
    fasta_names = [line.strip() for line in txt_file.readlines()]

# 准备写入合并后的FASTA文件
with open(output_fasta_path, 'w', encoding='utf-8') as output_file:
    for fasta_name in fasta_names:
        fasta_file_path = os.path.join(folder_path, f'{fasta_name}.fasta')

        # 检查该FASTA文件是否存在
        if os.path.exists(fasta_file_path):
            with open(fasta_file_path, 'r', encoding='utf-8') as fasta_file:
                # 将FASTA文件内容写入输出文件
                output_file.write(fasta_file.read() + '\n')
        else:
            print(f"警告: 文件 {fasta_file_path} 未找到，跳过该文件。")

print(f"所有文件已按顺序合并到 {output_fasta_path}")
