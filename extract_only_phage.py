
import os
import shutil

# 路径参数
folder1 = "/home/wangjingyuan/wys/phage_fasta"   # 替换为你的文件夹1路径
folder2 = "/home/wangjingyuan/wys/phage_fasta_final"   # 替换为你的文件夹2路径
output_folder = "/home/wangjingyuan/wys/phage_fasta_unique"  # 输出目录

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 获取两个文件夹中的文件名（只取 .fasta 文件）
files1 = {f for f in os.listdir(folder1) if f.endswith(".fasta")}
files2 = {f for f in os.listdir(folder2) if f.endswith(".fasta")}

# 找出文件夹1独有的文件
unique_files = files1 - files2

# 复制独有文件到新目录
for f in unique_files:
    src = os.path.join(folder1, f)
    dst = os.path.join(output_folder, f)
    shutil.copy2(src, dst)

print(f"完成！共复制 {len(unique_files)} 个独有的 fasta 文件到 {output_folder}")

