
import os
import shutil
import random

# 路径参数
input_folder = "/home/wangjingyuan/wys/WYSPHP/dnabert4_phage_embeddings_final_unique"   # 替换为你的源文件夹路径
output_folder = "random_phage_emb" # 输出目录
sample_size = 1000                # 抽取数量

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有 .parquet 文件
parquet_files = [f for f in os.listdir(input_folder) if f.endswith(".parquet")]

# 如果文件数量不足，取全部
if len(parquet_files) <= sample_size:
    sampled_files = parquet_files
    print(f"源文件不足 {sample_size} 个，仅复制 {len(sampled_files)} 个文件。")
else:
    sampled_files = random.sample(parquet_files, sample_size)

# 复制抽样文件
for f in sampled_files:
    src = os.path.join(input_folder, f)
    dst = os.path.join(output_folder, f)
    shutil.copy2(src, dst)

print(f"完成！共复制 {len(sampled_files)} 个 parquet 文件到 {output_folder}")
