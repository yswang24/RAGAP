# #!/usr/bin/env python3
# import sys

# def extract_representatives(clstr_file, output_tsv):
#     cluster_id = None
#     representatives = []

#     with open(clstr_file, 'r') as infile:
#         for line in infile:
#             line = line.strip()
#             if not line:
#                 continue

#             if line.startswith('>Cluster'):
#                 # 解析簇号
#                 cluster_id = line.split()[1]
#             elif line.endswith('*'):
#                 # 找到代表序列
#                 parts = line.split('>')
#                 if len(parts) > 1:
#                     seq_id = parts[1].split('...')[0]  # 去掉 ... 及后续
#                     seq_id = seq_id.split()[0]         # 去掉空格及后面内容
#                     representatives.append((cluster_id, seq_id))

#     # 输出结果
#     with open(output_tsv, 'w') as out:
#         out.write("Cluster_ID\tRepresentative_ID\n")
#         for cid, rid in representatives:
#             out.write(f"{cid}\t{rid}\n")

#     print(f"✅ 提取完成：共 {len(representatives)} 个簇，结果已保存至 {output_tsv}")


# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("用法: python extract_representatives.py <input.clstr> <output.tsv>")
#         sys.exit(1)

#     clstr_file = sys.argv[1]
#     output_tsv = sys.argv[2]
#     extract_representatives(clstr_file, output_tsv)

'''
python extract_representatives.py all_proteins_phage_nr80.clstr phage_reps.tsv
python extract_representatives.py all_proteins_host_nr80 host_reps.tsv
'''



input_fasta = "all_proteins_phage_nr80"  # 你的FASTA文件
output_tsv = "sequence_ids_count_phage.tsv"

sequence_ids = []

with open(input_fasta, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            # 只保留第一个空格前的ID
            seq_id = line[1:].split()[0]
            sequence_ids.append(seq_id)

# 输出统计信息
total_sequences = len(sequence_ids)
print(f"✅ 总序列数: {total_sequences}")

# 保存ID到TSV文件
with open(output_tsv, "w") as out:
    out.write("Sequence_ID\n")
    for sid in sequence_ids:
        out.write(sid + "\n")

print(f"✅ 所有序列ID已保存到 {output_tsv}")
