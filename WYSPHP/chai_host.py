import os
from pathlib import Path
from Bio import SeqIO
import pandas as pd

# 输入和输出路径
input_dir = "/home/wangjingyuan/wys/host_fasta_final"       # 存放原始fasta文件的文件夹
output_dir = "/home/wangjingyuan/wys/host_fasta_final_sequence"      # 输出拆分后的序列
tsv_path = "sequence_source.tsv"  # 记录来源的TSV

os.makedirs(output_dir, exist_ok=True)

records_info = []  # 用于存储 (SeqID, SourceFile)

# 遍历文件夹中的所有fasta文件
for fasta_file in Path(input_dir).glob("*.fasta"):
    source_name = fasta_file.stem  # e.g. GCF_000006925

    # 读取 fasta 文件
    for record in SeqIO.parse(fasta_file, "fasta"):
        # 取 ID （> 后面的第一个字段）
        seq_id = record.id.split()[0]

        # 生成输出文件名
        out_fasta = Path(output_dir) / f"{seq_id}.fasta"

        # 写入单条序列到新文件
        SeqIO.write(record, out_fasta, "fasta")

        # 保存来源关系
        records_info.append([seq_id, source_name])

# 输出到 TSV
df = pd.DataFrame(records_info, columns=["SeqID", "SourceFile"])
df.to_csv(tsv_path, sep="\t", index=False)

print(f"序列已拆分到 {output_dir}")
print(f"来源关系已保存到 {tsv_path}")
