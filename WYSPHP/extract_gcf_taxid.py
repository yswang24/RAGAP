#########提取GCF编号


# import os

# # 设置目标文件夹路径
# input_dir = "/home/wangjingyuan/wys/host_fasta"  # <- 替换为你的实际路径
# output_file = "GCF.txt"

# # 存储所有提取到的文件名（不含后缀）
# fasta_ids = []

# # 遍历文件夹（含子目录）
# for root, dirs, files in os.walk(input_dir):
#     for file in files:
#         if file.endswith(".fasta"):
#             # 去除扩展名
#             base_name = os.path.splitext(file)[0]
#             fasta_ids.append(base_name)

# # 写入到 txt 文件中
# with open(output_file, "w") as f:
#     for fasta_id in fasta_ids:
#         f.write(fasta_id + "\n")

# print(f"已提取 {len(fasta_ids)} 个 fasta 文件名，保存至 {output_file}")



# import sys

# # 用户输入文件路径
# GCF_LIST_FILE = "GCF.txt"  # 包含你所有 GCF 编号（不含版本号）
# ASSEMBLY_SUMMARY_FILE = "assembly_summary_refseq.txt"  # NCBI 官方 summary 文件
# OUTPUT_FILE = "GCF_taxid_mapping.tsv"  # 输出文件

# # 加载所有 GCF 编号（不含版本号）
# with open(GCF_LIST_FILE) as f:
#     gcf_set = set(line.strip().split('.')[0] for line in f if line.strip())

# # 创建映射字典
# gcf_to_taxid = {}

# with open(ASSEMBLY_SUMMARY_FILE) as f:
#     for line in f:
#         if line.startswith("#"):
#             continue
#         parts = line.strip().split("\t")
#         if len(parts) < 6:
#             continue
#         full_gcf = parts[0]             # 带版本号
#         taxid = parts[5]
#         gcf_no_version = full_gcf.split('.')[0]
#         if gcf_no_version in gcf_set:
#             # 只记录首次匹配（最新版本）
#             if gcf_no_version not in gcf_to_taxid:
#                 gcf_to_taxid[gcf_no_version] = taxid

# # 输出结果
# with open(OUTPUT_FILE, "w") as out:
#     for gcf in sorted(gcf_set):
#         taxid = gcf_to_taxid.get(gcf, "NA")
#         out.write(f"{gcf}\t{taxid}\n")

# print(f"✅ 提取完成，共匹配到 {len(gcf_to_taxid)} 个 GCF → TaxID 映射，输出到 {OUTPUT_FILE}")



import csv

# 文件路径
ACCESSION_FILE = "/home/wangjingyuan/wys/dataset/phage_accession.txt"          # 每行是一个 accession（如 GCF_000005825 或 FN436268）
VIRUS_HOST_DB_FILE = "/home/wangjingyuan/wys/dataset/virushostdb.daily_all.tsv"    # Virus-Host DB 全数据表
OUTPUT_FILE = "phage_taxid_mapping.tsv"        # 输出 TSV 文件

# 加载 accession 列表
with open(ACCESSION_FILE) as f:
    accessions = [line.strip() for line in f if line.strip()]

# 构建 refseq id → virus tax id 映射
refseq_to_taxid = {}

with open(VIRUS_HOST_DB_FILE, newline='') as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        refseq_id = row["refseq id"].strip()
        virus_taxid = row["virus tax id"].strip()
        if refseq_id and virus_taxid:
            refseq_to_taxid[refseq_id] = virus_taxid

# 写出 accession → taxid 映射
with open(OUTPUT_FILE, "w") as out:
    for acc in accessions:
        taxid = refseq_to_taxid.get(acc, "NA")
        out.write(f"{acc}\t{taxid}\n")

print(f"✅ 映射完成，共写出 {len(accessions)} 条记录，输出文件：{OUTPUT_FILE}")
