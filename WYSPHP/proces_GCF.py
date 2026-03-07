'''
python proces_GCF.py \
  --virus_host /home/wangjingyuan/wys/WYSPHP/virus_host_taxid.tsv \
  --assembly /home/wangjingyuan/wys/WYSPHP/assembly_summary_refseq_reference_genome.tsv \
  --out virus_host_with_GCF.tsv \
  --log extraction_log.tsv
'''
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import csv
# import argparse
# import re

# def extract_gcf_id(assembly_accession):
#     """提取 GCF_xxxxxxxx（去掉版本号）"""
#     m = re.match(r"(GCF_\d+)", assembly_accession)
#     return m.group(1) if m else None

# def main():
#     parser = argparse.ArgumentParser(description="根据 host_taxid 匹配 assembly_info 里的 species_taxid，提取 GCF 编号（忽略原始 GCF_id）")
#     parser.add_argument("--virus_host", required=True, help="virus_host.tsv 输入文件")
#     parser.add_argument("--assembly", required=True, help="assembly_info.tsv 输入文件")
#     parser.add_argument("--out", required=True, help="输出文件 (TSV)")
#     parser.add_argument("--log", required=True, help="日志文件 (TSV)")
#     args = parser.parse_args()

#     # === 读取 assembly_info，建立 species_taxid -> GCF 列表 ===
#     mapping = {}
#     with open(args.assembly, newline="") as f:
#         reader = csv.DictReader(f, delimiter="\t")
#         first_col = reader.fieldnames[0]
#         for row in reader:
#             gcf = extract_gcf_id(row[first_col])
#             species_taxid = row.get("species_taxid", "")
#             if gcf and species_taxid:
#                 mapping.setdefault(species_taxid, set()).add(gcf)

#     # === 处理 virus_host.tsv ===
#     with open(args.virus_host, newline="") as f_in, \
#          open(args.out, "w", newline="") as f_out, \
#          open(args.log, "w") as f_log:

#         reader = csv.DictReader(f_in, delimiter="\t")
#         fieldnames = reader.fieldnames + ["Extracted_GCFs"]
#         writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=fieldnames)
#         writer.writeheader()

#         f_log.write("host_taxid\tstatus\tcount\tgcf_list\n")

#         for row in reader:
#             host_taxid = row.get("host_taxid", "")
#             gcf_list = sorted(mapping.get(host_taxid, []))

#             if gcf_list:
#                 row["Extracted_GCFs"] = ";".join(gcf_list)
#                 f_log.write(f"{host_taxid}\tFound\t{len(gcf_list)}\t{','.join(gcf_list)}\n")
#             else:
#                 row["Extracted_GCFs"] = "NA"
#                 f_log.write(f"{host_taxid}\tNot Found\t0\t\n")

#             writer.writerow(row)

#     print(f"✅ 完成: 结果写入 {args.out}, 日志写入 {args.log}")

# if __name__ == "__main__":
#     main()


'''
python proces_GCF.py \
  --input virus_host_with_GCF.tsv \
  --output all_host_GCFs.txt
  '''




# import csv
# import argparse

# def main():
#     parser = argparse.ArgumentParser(description="从 virus_host_with_GCF.tsv 提取所有 GCF 并输出到 txt")
#     parser.add_argument("--input", required=True, help="输入文件：virus_host_with_GCF.tsv")
#     parser.add_argument("--output", required=True, help="输出文件：all_GCFs.txt")
#     args = parser.parse_args()

#     all_gcfs = set()

#     with open(args.input, newline="") as f:
#         reader = csv.DictReader(f, delimiter="\t")
#         for row in reader:
#             gcf_field = row.get("Extracted_GCFs", "")
#             if gcf_field and gcf_field != "NA":
#                 all_gcfs.update(gcf_field.split(";"))

#     with open(args.output, "w") as f:
#         for gcf in sorted(all_gcfs):
#             f.write(gcf + "\n")

#     print(f"✅ 提取完成，共 {len(all_gcfs)} 个唯一 GCF 写入 {args.output}")

# if __name__ == "__main__":
#     main()

'''
python proces_GCF.py \
  --input virus_host_with_GCF0.tsv \
  --output virus_host_with_GCF_noNA.tsv

'''



# import csv
# import argparse

# def main():
#     parser = argparse.ArgumentParser(description="删除 Extracted_GCFs 列为 NA 的行")
#     parser.add_argument("--input", required=True, help="输入文件：virus_host_with_GCF.tsv")
#     parser.add_argument("--output", required=True, help="输出文件：filtered.tsv")
#     args = parser.parse_args()

#     with open(args.input, newline="") as f_in, open(args.output, "w", newline="") as f_out:
#         reader = csv.DictReader(f_in, delimiter="\t")
#         writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames, delimiter="\t")
#         writer.writeheader()

#         kept = 0
#         dropped = 0
#         for row in reader:
#             if row.get("Extracted_GCFs", "") != "NA":
#                 writer.writerow(row)
#                 kept += 1
#             else:
#                 dropped += 1

#     print(f"✅ 处理完成: 保留 {kept} 行, 删除 {dropped} 行, 输出写入 {args.output}")


# if __name__ == "__main__":
#     main()



# import pandas as pd


# # 读取原始文件
# df = pd.read_csv("/home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF0.tsv", sep="\t", dtype=str)

# # 把 "NA" 处理成缺失值
# df["Extracted_GCFs"] = df["Extracted_GCFs"].replace("NA", pd.NA)  
# df["GCF_id"] = df["GCF_id"].replace("NA", pd.NA)

# # 优先保留 Extracted_GCFs，不存在时用 GCF_id
# df["Extracted_GCFs"] = df["Extracted_GCFs"].fillna(df["GCF_id"])

# # 删除 Extracted_GCFs 仍为空的行
# df = df.dropna(subset=["Extracted_GCFs"])

# # 保存新的 tsv 文件
# df.to_csv("virus_host_GCF.tsv", sep="\t", index=False)

# # 额外保存 Extracted_GCFs 的唯一列表
# df["Extracted_GCFs"].dropna().unique().tofile("host_GCFs.txt", sep="\n", format="%s")

# import re
# import pandas as pd




# # 输入输出文件名
# input_file = "/home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF_noNA.tsv"
# output_file = "/home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF.tsv"
# import csv



# with open(input_file, "r", encoding="utf-8") as infile, \
#      open(output_file, "w", encoding="utf-8", newline="") as outfile:

#     reader = csv.DictReader(infile, delimiter="\t")
#     writer = csv.DictWriter(outfile, fieldnames=["virus_taxid", "refseq_id", "host_taxid","Extracted_GCFs"], delimiter="\t")

#     # 写表头
#     writer.writeheader()

#     # 按列提取并写入
#     for row in reader:
#         writer.writerow({
#             "virus_taxid": row.get("virus_taxid", ""),
#             "refseq_id": row.get("refseq_id", ""),
#             "host_taxid":row.get("host_taxid", ""),
#             "Extracted_GCFs": row.get("Extracted_GCFs", "")
#         })

# print(f"✅ 已保存结果到 {output_file}")


# import os
# import shutil
# import csv

# # === 配置 ===
# tsv_file = "/home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF.tsv"           # 你的 tsv 文件
# refseq_source = "/home/wangjingyuan/wys/phage_fasta"     # refseq parquet 文件所在目录
# gcf_source = "/home/wangjingyuan/wys/host_fasta"           # GCF parquet 文件所在目录
# refseq_outdir = "/home/wangjingyuan/wys/phage_fasta_final"        # refseq parquet 文件输出目录
# gcf_outdir = "/home/wangjingyuan/wys/host_fasta_final"              # GCF parquet 文件输出目录


# # 确保输出目录存在
# os.makedirs(refseq_outdir, exist_ok=True)
# os.makedirs(gcf_outdir, exist_ok=True)

# # 用 set 去重
# refseq_files = set()
# gcf_files = set()
# missing_gcf = set()

# with open(tsv_file, "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f, delimiter="\t")

#     for row in reader:
#         refseq_id = row["refseq_id"].strip()
#         gcf_ids = row["Extracted_GCFs"].strip().split(";")  # 拆分多个 GCF

#         # === 处理 refseq 文件 ===
#         if refseq_id:
#             refseq_file = os.path.join(refseq_source, refseq_id + ".parquet")
#             target_file = os.path.join(refseq_outdir, refseq_id + ".parquet")
#             if not os.path.exists(target_file):  # 目标目录不存在才复制
#                 if os.path.exists(refseq_file):
#                     shutil.copy(refseq_file, target_file)
#                     refseq_files.add(refseq_id)
#                 else:
#                     print(f"⚠️ 找不到 refseq 文件: {refseq_file}")
#             else:
#                 refseq_files.add(refseq_id)  # 已存在也加入集合

#         # === 处理多个 GCF 文件 ===
#         for gcf_id in gcf_ids:
#             gcf_id = gcf_id.strip()
#             if gcf_id:
#                 gcf_file = os.path.join(gcf_source, gcf_id + ".parquet")
#                 target_file = os.path.join(gcf_outdir, gcf_id + ".parquet")
#                 if not os.path.exists(target_file):  # 目标目录不存在才复制
#                     if os.path.exists(gcf_file):
#                         shutil.copy(gcf_file, target_file)
#                         gcf_files.add(gcf_id)
#                     else:
#                         print(f"⚠️ 找不到 GCF 文件: {gcf_file}")
#                         missing_gcf.add(gcf_id)
#                 else:
#                     gcf_files.add(gcf_id)  # 已存在也加入集合

# # === 写入 txt 文件（排序一下，避免乱序） ===
# with open("refseq_files.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(sorted(refseq_files)))

# with open("gcf_files.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(sorted(gcf_files)))

# with open("missing_gcf.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(sorted(missing_gcf)))

# print("✅ 去重 & 拆分完成，已跳过已有 parquet 文件。")
# print(f"- refseq 文件夹: {refseq_outdir}, 列表: refseq_files.txt")
# print(f"- GCF 文件夹: {gcf_outdir}, 列表: gcf_files.txt")
# print(f"- 缺失的 GCF 名称: missing_gcf.txt")




# import os
# import glob
# import shutil

# # 原始 genomes 目录
# genomes_dir = "fasta_only"
# # 输出 parquet 目录
# fasta_dir = "host_fasta_final0"
# os.makedirs(fasta_dir, exist_ok=True)

# # 遍历所有 *_genomic.fna 文件
# for file_path in glob.glob(os.path.join(genomes_dir, "**", "*_genomic.fna"), recursive=True):
#     # 获取文件名
#     filename = os.path.basename(file_path)
#     # 提取 GCF 编号部分（去掉版本号和 _genomic）
#     # 示例: GCF_000154385.1_ASM15438v1_genomic.fna -> GCF_000154385.parquet
#     gcf_id = filename.split("_ASM")[0]  # 先去掉版本和 ASM 信息
#     gcf_id = gcf_id.split(".")[0]      # 去掉 .1 等版本号
#     new_name = gcf_id + ".parquet"

#     target_path = os.path.join(fasta_dir, new_name)

#     # 如果目标文件已存在，跳过
#     if os.path.exists(target_path):
#         print(f"⚠️ {new_name} 已存在，跳过")
#         continue

#     # 复制并重命名
#     shutil.copy(file_path, target_path)
#     print(f"✅ {filename} -> {new_name}")

# print("✅ 所有文件重命名完成，保存在:", fasta_dir)



# import os
# import shutil

# # 配置
# fasta_dir = "/home/wangjingyuan/wys/WYSPHP/esm_embeddings_35_phage_parquet"  # 原 parquet 文件夹
# name_file = "phage_refseq.txt"             # TXT 文件，每行一个名字，不含后缀
# out_dir = "/home/wangjingyuan/wys/WYSPHP/esm_embeddings_35_phage_parquet_final"               # 输出目录
# os.makedirs(out_dir, exist_ok=True)

# # 读取 TXT 文件中的名字
# with open(name_file, "r") as f:
#     names = set(line.strip() for line in f if line.strip())

# # 遍历 fasta_dir，把匹配的文件复制到输出目录
# for name in names:
#     src_path = os.path.join(fasta_dir, name + ".parquet")
#     dst_path = os.path.join(out_dir, name + ".parquet")
#     if os.path.exists(src_path):
#         shutil.copy(src_path, dst_path)
#         print(f"✅ 复制 {name}.parquet")
#     else:
#         print(f"⚠️ 未找到 {name}.parquet")

# print("✅ 提取完成，保存在:", out_dir)



# 提取 host_taxid 并去重
input_file = "virus_host_with_GCF.tsv"
output_file = "host_taxids.txt"

host_ids = set()

with open(input_file, "r") as f:
    header = f.readline()  # 跳过表头
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 3:  # host_taxid 是第 3 列
            host_ids.add(parts[2])

# 写入输出文件
with open(output_file, "w") as f:
    for host_id in sorted(host_ids):
        f.write(host_id + "\n")

print(f"✅ 已提取 {len(host_ids)} 个唯一 host_taxid，结果保存在 {output_file}")
