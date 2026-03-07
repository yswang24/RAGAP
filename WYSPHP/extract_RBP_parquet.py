# import os
# import pandas as pd

# # 输入文件夹路径
# input_folder = "/home/wangjingyuan/wys/WYSPHP/annotation_out/PhageRBPdetect_v4output_host"
# # 输出文件路径
# output_file = "RBP_host.tsv"

# # 用于保存所有结果
# filtered_dfs = []

# # 遍历文件夹中的所有tsv文件
# for filename in os.listdir(input_folder):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(input_folder, filename)
#         df = pd.read_csv(file_path, sep=",")  # 假设是逗号分隔
#         filtered = df[df['preds'] == 1]  # 过滤第二列为1的行
#         if not filtered.empty:
#             filtered['source_file'] = filename  # 可选：保留文件名，方便追踪
#             filtered_dfs.append(filtered)

# # 合并所有结果
# if filtered_dfs:
#     result = pd.concat(filtered_dfs, ignore_index=True)
#     result.to_csv(output_file, sep="\t", index=False)
#     print(f"已保存结果到 {output_file}，共 {len(result)} 行")
# else:
#     print("没有找到preds=1的行")



import pandas as pd

# 输入文件
tsv_file = "RBP_phage.tsv"        # 刚生成的tsv
parquet_file = "/home/wangjingyuan/wys/WYSPHP/RBP_650/protein_phage_catalog_650.parquet"      # 原始parquet
output_parquet = "/home/wangjingyuan/wys/WYSPHP/RBP_650/RBP_phage.parquet"
not_found_file = "/home/wangjingyuan/wys/WYSPHP/RBP_650/not_found.tsv"

# 1. 读取tsv文件
tsv_df = pd.read_csv(tsv_file, sep="\t")
names = set(tsv_df['protein_name'].unique())  # 取出所有名字

# 2. 读取parquet文件
parquet_df = pd.read_parquet(parquet_file)

# 3. 根据protein_id过滤
matched_df = parquet_df[parquet_df['protein_id'].isin(names)]
not_found = names - set(matched_df['protein_id'])

# 4. 保存结果
matched_df.to_parquet(output_parquet, index=False)
pd.DataFrame(sorted(not_found), columns=['not_found_protein_name']).to_csv(not_found_file, sep="\t", index=False)

print(f"匹配成功 {len(matched_df)} 行，已保存到 {output_parquet}")
print(f"未找到 {len(not_found)} 个名字，已保存到 {not_found_file}")
