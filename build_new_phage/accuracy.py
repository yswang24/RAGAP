import pandas as pd

# 读取文件
# pred_df = pd.read_csv("phage_prediction_results_test_topk.tsv", sep="\t")
# pred_df = pd.read_csv("/home/wangjingyuan/wys/wys_shiyan/phage_prediction_results_epoch_2000.tsv", sep="\t")
pred_df = pd.read_csv("/home/wangjingyuan/wys/build_new_phage/newphage_predictions_cherry_pp_out.tsv", sep="\t")
true_df = pd.read_csv("/home/wangjingyuan/wys/build_new_phage/cherry_dataset/virus.csv", sep=",")

# ---- Hit@1 ----
top1_df = pred_df[pred_df['rank'] == 1]
merged = pd.merge(top1_df, true_df[['phage_id', 'species']], on='phage_id', how='inner')
merged['correct'] = merged['host_species_name'] == merged['species']
accuracy1 = merged['correct'].mean()
print(f"Hit@1 accuracy: {accuracy1:.4f} ({merged['correct'].sum()}/{len(merged)})")

# ---- Hit@5 ----
top5_df = pred_df[pred_df['rank'] <= 5]
hit5 = top5_df.groupby('phage_id')['host_species_name'].apply(list).reset_index()
hit5 = pd.merge(hit5, true_df[['phage_id', 'species']], on='phage_id', how='inner')
hit5['correct'] = hit5.apply(lambda row: row['species'] in row['host_species_name'], axis=1)
accuracy5 = hit5['correct'].mean()
print(f"Hit@5 accuracy: {accuracy5:.4f} ({hit5['correct'].sum()}/{len(hit5)})")

# ---- Hit@10 ----
top10_df = pred_df[pred_df['rank'] <= 10]
hit10 = top10_df.groupby('phage_id')['host_species_name'].apply(list).reset_index()
hit10 = pd.merge(hit10, true_df[['phage_id', 'species']], on='phage_id', how='inner')
hit10['correct'] = hit10.apply(lambda row: row['species'] in row['host_species_name'], axis=1)
accuracy10 = hit10['correct'].mean()
print(f"Hit@10 accuracy: {accuracy10:.4f} ({hit10['correct'].sum()}/{len(hit10)})")

# ---- 打印预测错误的 phage_id (只看Top1错误的) ----
wrong = merged[~merged['correct']]
if not wrong.empty:
    print("\n预测错误的 phage 列表:")
    print(wrong[['phage_id', 'host_species_name', 'species']].to_string(index=False))
else:
    print("\n所有预测都正确 ✅")
