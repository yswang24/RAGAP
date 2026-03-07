# import pandas as pd

# # 读取 parquet 文件
# df = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/dnabert6_host_embeddings_final/GCF_028596125.parquet")
# df1 = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/dnabert6_phage_embeddings_final/AB626963.parquet")
# df2 = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/taxonomy_tree/taxonomy_with_alias.parquet")
# df3 = pd.read_parquet("/home/wangjingyuan/wys/wys_shiyan/data_processed/phage_catalog.parquet")
# df4 = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/esm_embeddings_35_host_parquet_final/GCF_000005845.parquet")
# df5 = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/mmseq/protein_clusters.parquet")
# df6 = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/protein_clusters_emb.parquet")
# df7 = pd.read_parquet("/home/wangjingyuan/wys/wys_shiyan/host_protein_edges.parquet")



# print(df7.info())       # 列名、类型、非空数量
# print(df7.head(10))       # 前几行
# print(df7.shape)        # 行数、列数

# 看基本信息
# print(df.info())       # 列名、类型、非空数量
# print(df.head())       # 前几行
# print(df.shape)        # 行数、列数

# print(df1.info())       # 列名、类型、非空数量
# print(df1.head())       # 前几行
# print(df1.shape) 
# print(df2.info())       # 列名、类型、非空数量
# print(df2.head())       # 前几行
# print(df2.shape) 

import pandas as pd
import io

# # 假设 df3 已经存在

# with open("df3p_info.txt", "w", encoding="utf-8") as f:
#     # 保存 df3.info()
#     buffer = io.StringIO()
#     df3.info(buf=buffer)
#     info_str = buffer.getvalue()
#     f.write("=== df.info() ===\n")
#     f.write(info_str + "\n\n")

#     # 保存 df3.head(10)
#     f.write("=== df.head(10) ===\n")
#     f.write(df3.head(100).to_string() + "\n\n")

#     # 保存 df3.shape
#     f.write("=== df.shape ===\n")
#     f.write(str(df3.shape) + "\n")


# print(df3.info())       # 列名、类型、非空数量
# print(df3.head(10))       # 前几行
# print(df3.shape)        # 行数、列数


# with open("df2_info.txt", "w", encoding="utf-8") as f:
#     # 保存 df3.info()
#     buffer = io.StringIO()
#     df2.info(buf=buffer)
#     info_str = buffer.getvalue()
#     f.write("=== df.info() ===\n")
#     f.write(info_str + "\n\n")

#     # 保存 df3.head(10)
#     f.write("=== df.head(10) ===\n")
#     f.write(df2.head(100).to_string() + "\n\n")

#     # 保存 df3.shape
#     f.write("=== df.shape ===\n")
#     f.write(str(df2.shape) + "\n")


# print(df4.info())       # 列名、类型、非空数量
# print(df4.head(10))       # 前几行
# print(df4.shape)        # 行

# print(df5.info())       # 列名、类型、非空数量
# print(df5.head(10))       # 前几行
# print(df5.shape)        # 行
# with open("df5_info.txt", "w", encoding="utf-8") as f:
#     # 保存 df3.info()
#     buffer = io.StringIO()
#     df5.info(buf=buffer)
#     info_str = buffer.getvalue()
#     f.write("=== df.info() ===\n")
#     f.write(info_str + "\n\n")

#     # 保存 df3.head(10)
#     f.write("=== df.head(10) ===\n")
#     f.write(df5.head(100).to_string() + "\n\n")

#     # 保存 df3.shape
#     f.write("=== df.shape ===\n")
#     f.write(str(df5.shape) + "\n")



# print(df6.info())       # 列名、类型、非空数量
# print(df6.head(10))       # 前几行
# print(df6.shape)        # 行
# with open("df6_info.txt", "w", encoding="utf-8") as f:
#     # 保存 df3.info()
#     buffer = io.StringIO()
#     df6.info(buf=buffer)
#     info_str = buffer.getvalue()
#     f.write("=== df.info() ===\n")
#     f.write(info_str + "\n\n")

#     # 保存 df3.head(10)
#     f.write("=== df.head(10) ===\n")
#     f.write(df6.head(100).to_string() + "\n\n")

#     # 保存 df3.shape
#     f.write("=== df.shape ===\n")
#     f.write(str(df6.shape) + "\n")

# import os
# import pandas as pd

# # 输入文件夹 (存放 parquet 文件)
# PARQUET_DIR = "/home/wangjingyuan/wys/WYSPHP/dnabert4_host_embeddings_final0"
# # TSV 文件 (含 GCF -> host_taxid 映射)
# TSV_FILE = "/home/wangjingyuan/wys/WYSPHP/virus_host_with_GCF.tsv"
# # 输出文件夹
# OUTPUT_DIR = "/home/wangjingyuan/wys/WYSPHP/dnabert4_host_embeddings_final"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 读取 TSV 文件
# tsv_df = pd.read_csv(TSV_FILE, sep="\t")

# # 构建 GCF -> host_taxid 映射，支持分号分隔的多个 GCF
# gcf_to_taxid = {}
# for _, row in tsv_df.iterrows():
#     host_taxid = row["host_taxid"]
#     gcf_list = str(row["Extracted_GCFs"]).split(";")
#     for gcf in gcf_list:
#         gcf = gcf.strip()
#         if gcf:  # 避免空值
#             gcf_to_taxid[gcf] = host_taxid

# # 遍历 parquet 文件
# for fname in os.listdir(PARQUET_DIR):
#     if fname.endswith(".parquet"):
#         fpath = os.path.join(PARQUET_DIR, fname)

#         # 提取 GCF 编号 (去掉扩展名)
#         host_gcf = os.path.splitext(fname)[0]

#         # 读取 parquet 文件
#         df = pd.read_parquet(fpath)

#         # 添加 host_gcf
#         df["host_gcf"] = host_gcf

#         # 添加 host_species_taxid (可能不存在映射，用 get)
#         df["host_species_taxid"] = gcf_to_taxid.get(host_gcf, "NA")

#         # 输出到新 parquet
#         outpath = os.path.join(OUTPUT_DIR, fname)
#         df.to_parquet(outpath, index=False)

#         print(f"Processed {fname} -> {outpath}")



import os
import pandas as pd

# 输入目录 (存放 ABxxxxxx.parquet 等)
PARQUET_DIR = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_dnaemb"
# TSV 文件 (virus_taxid/refseq_id/host_taxid/Extracted_GCFs)
TSV_FILE = "/home/wangjingyuan/wys/duibi/pairs_all.tsv"
# 输出目录
OUTPUT_DIR = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_dnaemb_cherry"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 TSV 文件
tsv_df = pd.read_csv(TSV_FILE, sep="\t")

# 构建 refseq_id -> virus_taxid 映射
refseq_to_virus = dict(zip(tsv_df["refseq_id"], tsv_df["virus_taxid"]))

# 统计没匹配到的
unmatched = []

# 遍历 parquet 文件
for fname in os.listdir(PARQUET_DIR):
    if fname.endswith(".parquet"):
        fpath = os.path.join(PARQUET_DIR, fname)

        # 提取 ID (去掉扩展名)
        file_id = os.path.splitext(fname)[0]

        # 读取 parquet 文件
        df = pd.read_parquet(fpath)

        # 匹配 virus_taxid
        virus_taxid = refseq_to_virus.get(file_id, None)

        if virus_taxid is None:
            df["virus_taxid"] = "NA"
            unmatched.append(file_id)
            print(f"[WARNING] {file_id} 未匹配到 virus_taxid")
        else:
            df["virus_taxid"] = virus_taxid

        # 输出新 parquet
        outpath = os.path.join(OUTPUT_DIR, fname)
        df.to_parquet(outpath, index=False)

        print(f"Processed {fname} -> {outpath}")

print("\n========== 统计结果 ==========")
print(f"总文件数: {len([f for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet')])}")
print(f"未匹配到的数量: {len(unmatched)}")
if unmatched:
    print("未匹配到的ID列表:")
    for uid in unmatched:
        print("  ", uid)
