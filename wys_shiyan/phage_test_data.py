import os
import pandas as pd
import shutil

# ======= 用户输入部分 =======
tsv_file = "/home/wangjingyuan/wys/wys_shiyan/data_processed_new/pairs_train.tsv"       # 输入的 TSV 文件路径
phage_dir = "/home/wangjingyuan/wys/phage_fasta_final"            # 原始 fasta 文件夹
output_dir = "/home/wangjingyuan/wys/phage_train_data_613"             # 输出文件夹
# ===========================

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取 TSV 文件
df = pd.read_csv(tsv_file, sep="\t")

# 检查是否存在 phage_id 列
if 'phage_id' not in df.columns:
    raise ValueError("❌ TSV 文件中未找到 'phage_id' 列，请检查输入文件。")

# 提取 phage_id 列（去除空值与重复）
phage_ids = df['phage_id'].dropna().unique()

print(f"共检测到 {len(phage_ids)} 个噬菌体 ID。开始复制对应的 .fasta 文件...\n")

copied = 0
missing = 0

for pid in phage_ids:
    fasta_name = f"{pid}.fasta"
    src_path = os.path.join(phage_dir, fasta_name)
    dst_path = os.path.join(output_dir, fasta_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied += 1
        print(f"✅ 已复制: {fasta_name}")
    else:
        missing += 1
        print(f"⚠️ 未找到: {fasta_name}")

print("\n====== 复制完成 ======")
print(f"成功复制文件: {copied}")
print(f"未找到文件: {missing}")
print(f"输出文件夹: {os.path.abspath(output_dir)}")
