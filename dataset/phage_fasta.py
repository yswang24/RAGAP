
import os
import shutil

# 设置相关路径
txt_file = 'phage_accession.txt'  # 包含 Accession 号的 TXT 文件
fasta_folder = 'split_sequences'  # 原始 FASTA 文件夹
output_folder = 'phage_fasta'  # 新的文件夹，用来存放筛选的 FASTA 文件

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取 TXT 文件中的 Accession 号，保持顺序
accession_ids = []  # 用列表来存储 Accession 号，保持顺序
seen = set()  # 用于去重

with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        accession_list = line.strip().split(',')  # 使用逗号分割
        for acc in accession_list:
            acc = acc.strip()
            if acc and acc not in seen:  # 避免重复
                accession_ids.append(acc)
                seen.add(acc)

# 遍历 Accession 号，按顺序复制文件
missing_files = []  # 记录找不到的文件
copied_files = []  # 记录已复制的文件，用于输出

for accession in accession_ids:
    file_name = f"{accession}.fasta"  # 假设文件名格式为 'Accession.fasta'
    source_path = os.path.join(fasta_folder, file_name)
    destination_path = os.path.join(output_folder, file_name)

    if os.path.exists(source_path):  # 只有当文件存在时才复制
        shutil.copy(source_path, destination_path)
        copied_files.append(file_name)
    else:
        missing_files.append(file_name)  # 记录找不到的文件

# 输出已复制的文件（按顺序）
print("\n已复制的 FASTA 文件（按顺序）：")
for copied in copied_files:
    print(copied)

# 输出缺失的文件
if missing_files:
    print("\n以下 Accession 号对应的 FASTA 文件未找到：")
    for missing in missing_files:
        print(missing)

print("\n操作完成，相关 FASTA 文件已复制到新的文件夹。")
