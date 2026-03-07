# import pandas as pd

# # 读取文件
# #pred_df = pd.read_csv("phage_prediction_results_test_topk.tsv", sep="\t")
# #pred_df = pd.read_csv("/home/wangjingyuan/wys/wys_shiyan/phage_prediction_results_epoch_2000.tsv", sep="\t")
# pred_df = pd.read_csv("phage_prediction_results_val_topk_genus.tsv", sep="\t")


# true_df = pd.read_csv("phage.tsv", sep="\t")


# # 只保留 rank=1 的预测
# top1_df = pred_df[pred_df['rank'] == 1]

# # 合并预测和真实标签
# merged = pd.merge(top1_df, true_df[['phage_id', 'genus']], on='phage_id', how='inner')

# # 判断预测是否正确
# merged['correct'] = merged['host_genus'] == merged['genus']

# # 计算hit@1准确率
# accuracy = merged['correct'].mean()
# print(f"Hit@1 accuracy: {accuracy:.4f} ({merged['correct'].sum()}/{len(merged)})")

# # 打印预测错误的 phage_id 及其预测/真实结果
# wrong = merged[~merged['correct']]
# if not wrong.empty:
#     print("\n预测错误的 phage 列表:")
#     print(wrong[['phage_id', 'host_genus', 'genus']].to_string(index=False))
# else:
#     print("\n所有预测都正确 ✅")



import os
import pandas as pd
import numexpr as ne

# ===== 路径设置 =====
file1 = "phage_prediction_results_epoch_40000.tsv"
file2 = "virus_host_with_GCF.tsv"
file3 = "taxid_genus.tsv"
true_file = "phage.tsv"
output_file = "phage_prediction_results_epoch_40000_genus_topk.tsv"

# ===== 读取输入文件 =====
df1 = pd.read_csv(file1, sep="\t", dtype=str)
df2 = pd.read_csv(file2, sep="\t", dtype=str)
df3 = pd.read_csv(file3, sep="\t", dtype=str)

# ===== 拆分 Extracted_GCFs（以分号 ; 分割） =====
df2["Extracted_GCFs"] = df2["Extracted_GCFs"].fillna("")
df2 = df2.assign(Extracted_GCFs=df2["Extracted_GCFs"].str.split(";"))
df2 = df2.explode("Extracted_GCFs")  # 多 GCF 展开为多行
df2["Extracted_GCFs"] = df2["Extracted_GCFs"].str.strip()  # 去除空格

# ===== 去除空行与重复 =====
df2 = df2[df2["Extracted_GCFs"] != ""]
df2 = df2.drop_duplicates(subset=["Extracted_GCFs", "host_taxid"])

# ===== 第一次合并：host_id ↔ Extracted_GCFs =====
merged = pd.merge(df1, df2, left_on="host_id", right_on="Extracted_GCFs", how="left")

# ===== 第二次合并：host_taxid ↔ taxid 找 genus =====
merged = pd.merge(merged, df3, left_on="host_taxid", right_on="taxid", how="left")

# ===== 选取最终列 =====
result = merged[["phage_id", "rank", "host_id", "genus", "score"]].rename(columns={"genus": "host_genus"})

# ===== 输出中间文件 =====
result.to_csv(output_file, sep="\t", index=False)
print(f"✅ 输出完成：{output_file}")
print(f"共输出 {len(result)} 条记录。")

# ===== 读取真实标签文件 =====
true_df = pd.read_csv(true_file, sep="\t", dtype=str)

# ===== 合并预测和真实属标签 =====
merged_eval = pd.merge(result, true_df[['phage_id', 'genus']], on='phage_id', how='inner')

# 确保 rank 为数值类型
merged_eval['rank'] = merged_eval['rank'].astype(int)

# ===== 定义计算 Hit@k 函数 =====
def calc_hit_at_k(df, k):
    """计算 Hit@k：真实属是否出现在 top-k 预测中"""
    hits = 0
    total = df['phage_id'].nunique()
    for pid, group in df.groupby('phage_id'):
        topk = group.nsmallest(k, 'rank')['host_genus'].dropna().unique()
        true_genus = group['genus'].iloc[0]
        if true_genus in topk:
            hits += 1
    return hits / total, hits, total

# ===== 计算 Hit@1、Hit@5、Hit@10 =====
hit1, h1_n, total = calc_hit_at_k(merged_eval, 1)
hit5, h5_n, _ = calc_hit_at_k(merged_eval, 5)
hit10, h10_n, _ = calc_hit_at_k(merged_eval, 10)

# ===== 打印结果 =====
print("\n🎯 模型预测准确率统计：")
print(f"Hit@1  = {hit1:.4f} ({h1_n}/{total})")
print(f"Hit@5  = {hit5:.4f} ({h5_n}/{total})")
print(f"Hit@10 = {hit10:.4f} ({h10_n}/{total})")

# ===== 打印预测错误的 phage（Hit@1） =====
top1_df = merged_eval[merged_eval['rank'] == 1]
top1_df['correct'] = top1_df['host_genus'] == top1_df['genus']
wrong = top1_df[~top1_df['correct']]

if not wrong.empty:
    print("\n❌ Hit@1 预测错误的 phage 列表:")
    print(wrong[['phage_id', 'host_genus', 'genus']].to_string(index=False))
else:
    print("\n✅ Hit@1 所有预测都正确！")