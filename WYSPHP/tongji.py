# import pandas as pd

# # ---------- 第一部分：统计 cluster 成员数 ----------
# csv_path = "host_clusters_0.95.csv"  # 替换为你的 cluster csv 文件路径
# df = pd.read_csv(csv_path)

# # 统计每个 cluster 的成员数
# cluster_counts = df['cluster'].value_counts().reset_index()
# cluster_counts.columns = ['cluster_id', 'member_count']

# # ---------- 第二部分：提取唯一 GCF_id ----------
# tsv_path = "virus_host_taxid.tsv"  # 替换为你的 tsv 文件路径
# df_tsv = pd.read_csv(tsv_path, sep='\t')

# # 删除 GCF_id 中的缺失值
# df_tsv = df_tsv.dropna(subset=['GCF_id'])

# # 提取唯一 GCF_id
# unique_gcfs = df_tsv['GCF_id'].unique()
# gcf_count = len(unique_gcfs)

# # ---------- 输出到 txt 文件 ----------
# output_file = "summary_results.txt"

# with open(output_file, 'w') as f:
#     # 写入 cluster 统计结果
#     f.write("==== Cluster 成员数量统计（按数量降序） ====\n")
#     for _, row in cluster_counts.iterrows():
#         f.write(f"Cluster {row['cluster_id']}: {row['member_count']} 个成员\n")

#     # 写入 GCF 统计结果
#     f.write("\n==== virus_host_taxid.tsv 中唯一 GCF_id 统计 ====\n")
#     f.write(f"共有唯一 GCF_id 数量：{gcf_count}\n")
#     f.write("GCF 编号列表如下：\n")
#     for gcf in unique_gcfs:
#         f.write(f"{gcf}\n")

# print(f"结果已保存到 {output_file}")


# import pandas as pd
# import re

# # ---------- 文件路径 ----------
# cluster_csv = "host_clusters_0.95.csv"  # 你说的两个文件是同一个
# tsv_path = "virus_host_taxid.tsv"
# output_file = "summary_results1.txt"

# # ---------- 读取 cluster 文件 ----------
# df_cluster = pd.read_csv(cluster_csv)

# # 提取 GCF_id，例如从路径中提取 GCF_000005825
# def extract_gcf_id(path):
#     match = re.search(r"(GCF_\d+)", path)
#     return match.group(1) if match else None

# df_cluster['GCF_id'] = df_cluster['id'].apply(extract_gcf_id)

# # 构建 GCF → cluster 映射
# gcf_to_cluster = dict(zip(df_cluster['GCF_id'], df_cluster['cluster']))

# # 统计每个 cluster 的成员数
# cluster_counts = df_cluster['cluster'].value_counts().reset_index()
# cluster_counts.columns = ['cluster_id', 'member_count']

# # ---------- 读取 virus_host_taxid.tsv ----------
# df_tsv = pd.read_csv(tsv_path, sep='\t')
# df_tsv = df_tsv.dropna(subset=['GCF_id'])

# # 提取唯一 GCF_id
# unique_gcfs = sorted(df_tsv['GCF_id'].unique())
# gcf_count = len(unique_gcfs)

# # ---------- 写入 summary_results.txt ----------
# with open(output_file, 'w') as f:
#     # 第一部分：cluster 成员数
#     f.write("==== Cluster 成员数量统计（按数量降序） ====\n")
#     for _, row in cluster_counts.iterrows():
#         f.write(f"Cluster {row['cluster_id']}: {row['member_count']} 个成员\n")
    
#     # 第二部分：唯一 GCF_id 统计
#     f.write("\n==== virus_host_taxid.tsv 中唯一 GCF_id 统计 ====\n")
#     f.write(f"共有唯一 GCF_id 数量：{gcf_count}\n")

#     # 第三部分：每个 GCF_id 所属 cluster
#     f.write("\n==== 每个 GCF_id 所属的 cluster（来自 host_clusters_0.95.csv） ====\n")
#     for gcf in unique_gcfs:
#         cluster = gcf_to_cluster.get(gcf, "未找到")
#         f.write(f"{gcf} → Cluster {cluster}\n")

# print(f"✅ 统计完成，结果已保存到：{output_file}")


# import pandas as pd
# import re

# # ---------- 文件路径 ----------
# cluster_csv = "host_clusters_0.95.csv"  # cluster 文件（也是 your_cluster_file.csv）
# tsv_path = "virus_host_taxid.tsv"
# output_file = "summary_results2.txt"

# # ---------- 读取 cluster 文件 ----------
# df_cluster = pd.read_csv(cluster_csv)

# # 提取 GCF_id（如 GCF_000005825）
# def extract_gcf_id(path):
#     match = re.search(r"(GCF_\d+)", path)
#     return match.group(1) if match else None

# df_cluster['GCF_id'] = df_cluster['id'].apply(extract_gcf_id)

# # 构建 GCF_id → cluster 映射
# gcf_to_cluster = dict(zip(df_cluster['GCF_id'], df_cluster['cluster']))

# # 构建 cluster → 成员数 映射
# cluster_member_count = df_cluster['cluster'].value_counts().to_dict()

# # ---------- cluster 总成员统计 ----------
# cluster_counts = df_cluster['cluster'].value_counts().reset_index()
# cluster_counts.columns = ['cluster_id', 'member_count']

# # ---------- 读取 virus_host_taxid.tsv ----------
# df_tsv = pd.read_csv(tsv_path, sep='\t')
# df_tsv = df_tsv.dropna(subset=['GCF_id'])

# # 提取唯一 GCF_id
# unique_gcfs = sorted(df_tsv['GCF_id'].unique())
# gcf_count = len(unique_gcfs)

# # ---------- 查找这些 GCF_id 来自哪些 cluster ----------
# used_clusters = {}
# for gcf in unique_gcfs:
#     cluster = gcf_to_cluster.get(gcf)
#     if cluster is not None:
#         if cluster not in used_clusters:
#             used_clusters[cluster] = cluster_member_count.get(cluster, 0)

# # ---------- 写入 summary_results.txt ----------
# with open(output_file, 'w') as f:
#     # 第一部分：cluster 成员数
#     f.write("==== Cluster 成员数量统计（按数量降序） ====\n")
#     for _, row in cluster_counts.iterrows():
#         f.write(f"Cluster {row['cluster_id']}: {row['member_count']} 个成员\n")
    
#     # 第二部分：唯一 GCF_id 总数
#     f.write("\n==== virus_host_taxid.tsv 中唯一 GCF_id 统计 ====\n")
#     f.write(f"共有唯一 GCF_id 数量：{gcf_count}\n")

#     # 第三部分：每个 GCF_id 所属 cluster
#     f.write("\n==== 每个 GCF_id 所属的 cluster（来自 host_clusters_0.95.csv） ====\n")
#     for gcf in unique_gcfs:
#         cluster = gcf_to_cluster.get(gcf, "未找到")
#         f.write(f"{gcf} → Cluster {cluster}\n")

#     # 第四部分：这些 GCF_id 来源的不同 cluster 数量和成员统计
#     f.write("\n==== virus_host_taxid.tsv 中唯一 GCF_id 来源的 cluster 统计 ====\n")
#     f.write(f"共来自 {len(used_clusters)} 个不同的 cluster，详细如下：\n\n")
#     for cluster_id, count in sorted(used_clusters.items(), key=lambda x: -x[1]):
#         f.write(f"Cluster {cluster_id}: {count} 个成员\n")

# print(f"✅ 统计完成，结果已保存到：{output_file}")

# import pandas as pd
# import re

# # ---------- 文件路径 ----------
# cluster_csv = "/home/wangjingyuan/wys/create_edge/host_clusters.csv"
# tsv_path = "virus_host_taxid.tsv"
# output_summary = "summary_results1.txt"
# output_in_clusters = "in_virus_clusters_gcf1.txt"
# output_not_in_clusters = "not_in_virus_clusters_gcf1.txt"

# # ---------- 读取 cluster 文件 ----------
# df_cluster = pd.read_csv(cluster_csv)

# # 提取 GCF_id
# def extract_gcf_id(path):
#     match = re.search(r"(GCF_\d+)", path)
#     return match.group(1) if match else None

# df_cluster['GCF_id'] = df_cluster['id'].apply(extract_gcf_id)

# # 构建 GCF → cluster 映射
# gcf_to_cluster = dict(zip(df_cluster['GCF_id'], df_cluster['cluster']))

# # cluster → 成员 GCF 列表
# cluster_to_gcf_list = df_cluster.groupby('cluster')['GCF_id'].apply(set).to_dict()

# # ---------- cluster 成员数量统计 ----------
# cluster_counts = df_cluster['cluster'].value_counts().reset_index()
# cluster_counts.columns = ['cluster_id', 'member_count']

# # ---------- 读取病毒来源的 GCF_id ----------
# df_tsv = pd.read_csv(tsv_path, sep='\t')
# df_tsv = df_tsv.dropna(subset=['GCF_id'])
# unique_gcfs = sorted(df_tsv['GCF_id'].unique())
# gcf_count = len(unique_gcfs)

# # ---------- 查找病毒 GCF 所在的 cluster ----------
# used_clusters = {}
# for gcf in unique_gcfs:
#     cluster = gcf_to_cluster.get(gcf)
#     if cluster is not None:
#         if cluster not in used_clusters:
#             used_clusters[cluster] = cluster_counts[cluster_counts['cluster_id'] == cluster]['member_count'].values[0]

# # ---------- 收集 in-cluster 和 not-in-cluster GCF 列表 ----------
# all_gcf_set = set(df_cluster['GCF_id'].dropna())
# in_cluster_gcf_set = set()
# for cluster_id in used_clusters:
#     in_cluster_gcf_set.update(cluster_to_gcf_list.get(cluster_id, set()))

# not_in_cluster_gcf_set = all_gcf_set - in_cluster_gcf_set

# # ---------- 写入 summary_results.txt ----------
# with open(output_summary, 'w') as f:
#     f.write("==== Cluster 成员数量统计（按数量降序） ====\n")
#     for _, row in cluster_counts.iterrows():
#         f.write(f"Cluster {row['cluster_id']}: {row['member_count']} 个成员\n")
    
#     f.write("\n==== virus_host_taxid.tsv 中唯一 GCF_id 统计 ====\n")
#     f.write(f"共有唯一 GCF_id 数量：{gcf_count}\n")

#     f.write("\n==== 每个 GCF_id 所属的 cluster（来自 host_clusters_0.95.csv） ====\n")
#     for gcf in unique_gcfs:
#         cluster = gcf_to_cluster.get(gcf, "未找到")
#         f.write(f"{gcf} → Cluster {cluster}\n")

#     f.write("\n==== virus_host_taxid.tsv 中唯一 GCF_id 来源的 cluster 统计 ====\n")
#     f.write(f"共来自 {len(used_clusters)} 个不同的 cluster，详细如下：\n\n")
#     for cluster_id, count in sorted(used_clusters.items(), key=lambda x: -x[1]):
#         f.write(f"Cluster {cluster_id}: {count} 个成员\n")

# # ---------- 写入两个 GCF 列表 ----------
# with open(output_in_clusters, 'w') as f:
#     for gcf in sorted(in_cluster_gcf_set):
#         f.write(f"{gcf}\n")

# with open(output_not_in_clusters, 'w') as f:
#     for gcf in sorted(not_in_cluster_gcf_set):
#         f.write(f"{gcf}\n")

# print(f"✅ 分析完成！\n结果摘要写入：{output_summary}\n病毒相关簇的 GCF 编号写入：{output_in_clusters}\n其余编号写入：{output_not_in_clusters}")




# import os
# import shutil

# # ==== 配置路径 ====
# pkl_dir = "/home/wangjingyuan/wys/WYSPHP/esm_embeddings_35_phage_host"  # 存放 .pkl 文件的目录
# esm_output_dir = "esm_embeddingsd"  # 输出的 .pkl 目标目录
# host_cluster_dir = "/home/wangjingyuan/wys/WYSPHP/annotation_out/host"  # 存放 .faa 文件的目录
# process_host_dir = "process_host"  # 输出的 .faa 目标目录
# id_txt_file = "/home/wangjingyuan/wys/WYSPHP/in_virus_clusters_gcf1.txt"  # txt 文件，包含目标 GCF 编号

# missing_faa_file = "missing_faa.txt"  # ⬅ 输出没找到 .faa 的编号

# # ==== 创建目标文件夹 ====
# os.makedirs(esm_output_dir, exist_ok=True)
# os.makedirs(process_host_dir, exist_ok=True)

# # ==== 读取目标 GCF 编号 ====
# with open(id_txt_file) as f:
#     target_ids = {line.strip() for line in f if line.strip()}

# # ==== 处理 .pkl 文件 ====
# existing_ids = set()
# for gcf_id in target_ids:
#     pkl_file = os.path.join(pkl_dir, f"{gcf_id}.pkl")
#     if os.path.exists(pkl_file):
#         shutil.move(pkl_file, os.path.join(esm_output_dir, f"{gcf_id}.pkl"))
#         existing_ids.add(gcf_id)

# # ==== 找出没找到的 .pkl 编号 ====
# missing_ids = target_ids - existing_ids

# # ==== 找出 .faa 缺失的编号（最终输出） ====
# missing_faa_ids = []

# # ==== 将 .faa 复制到目标目录，记录缺失项 ====
# for gcf_id in missing_ids:
#     faa_file = os.path.join(host_cluster_dir, f"{gcf_id}.faa")
#     if os.path.exists(faa_file):
#         shutil.copy(faa_file, os.path.join(process_host_dir, f"{gcf_id}.faa"))
#     else:
#         print(f"[警告] 没有找到对应的 .faa 文件: {faa_file}")
#         missing_faa_ids.append(gcf_id)

# # ==== 将没找到 .faa 的编号写入 TXT ====
# with open(missing_faa_file, "w") as f:
#     for gcf_id in missing_faa_ids:
#         f.write(gcf_id + "\n")

# print(f"\n✅ 共 {len(missing_faa_ids)} 个 GCF 没有找到 .faa 文件，已写入 {missing_faa_file}")



# import os
# import shutil
# import pandas as pd

# # ====== 配置部分 ======
# tsv_file = "virus_host_with_GCF.tsv"  # 你的tsv文件路径
# source_folder = "/home/wangjingyuan/wys/WYSPHP/annotation_out/phage"  # 存放原始faa文件的文件夹
# target_folder = "/home/wangjingyuan/wys/WYSPHP/annotation_out/phage_final"  # 目标输出文件夹

# # 创建目标文件夹（如果不存在）
# os.makedirs(target_folder, exist_ok=True)

# # 读取tsv文件
# df = pd.read_csv(tsv_file, sep="\t")

# # 遍历refseq_id列
# for refseq_id in df["refseq_id"].dropna().unique():
#     source_path = os.path.join(source_folder, f"{refseq_id}.faa")
#     target_path = os.path.join(target_folder, f"{refseq_id}.faa")
    
#     if os.path.exists(source_path):
#         shutil.copy2(source_path, target_path)
#         print(f"✅ 已复制: {refseq_id}.faa")
#     else:
#         print(f"⚠️ 未找到: {refseq_id}.faa")



import os
from pathlib import Path

# ====== 配置部分 ======
phage_folder = "/home/wangjingyuan/wys/WYSPHP/annotation_out/phage_final"  # 存放phage .faa文件的文件夹
host_folder = "/home/wangjingyuan/wys/WYSPHP/annotation_out/host_final"    # 存放host .faa文件的文件夹
output_file = "/home/wangjingyuan/wys/WYSPHP/annotation_out/phage&host.faa"  # 合并后的输出文件

# 创建输出文件
with open(output_file, "w") as out_f:
    unique_id = 1  # 全局唯一ID计数

    # 遍历phage文件夹
    for faa_file in Path(phage_folder).glob("*.faa"):
        phage_id = faa_file.stem  # 取文件名作为phage_id
        with open(faa_file) as f:
            for line in f:
                if line.startswith(">"):
                    # 构造新的序列名：>uniqueID_proteinName|source=phage|phage_id=xxx
                    protein_name = line[1:].strip().split()[0]  # 去掉'>'，取第一个token
                    new_header = f">{unique_id}_{protein_name}|source=phage|phage_id={phage_id}"
                    out_f.write(new_header + "\n")
                    unique_id += 1
                else:
                    out_f.write(line)

    # 遍历host文件夹
    for faa_file in Path(host_folder).glob("*.faa"):
        host_id = faa_file.stem  # 取文件名作为host_id
        with open(faa_file) as f:
            for line in f:
                if line.startswith(">"):
                    # 构造新的序列名：>uniqueID_proteinName|source=host|host_id=xxx
                    protein_name = line[1:].strip().split()[0]
                    new_header = f">{unique_id}_{protein_name}|source=host|host_id={host_id}"
                    out_f.write(new_header + "\n")
                    unique_id += 1
                else:
                    out_f.write(line)

print(f"✅ 合并完成，输出文件: {output_file}")
