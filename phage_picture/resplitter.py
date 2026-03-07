import pandas as pd
import os
import random
from collections import defaultdict

# ================= 配置区域 =================
# 1. CD-HIT 生成的 .clstr 文件路径 (程序跑完后会有这个文件)
CLSTR_FILE = "/home/wangjingyuan/wys/phage_picture/clustering_result/phages_clustered_0.95.clstr"

# 2. 你原始的所有数据 (Train + Val + Test)
# 如果你有单独的三个文件，我们需要先把它们读进来合并
ORIGINAL_FILES = [
    "/home/wangjingyuan/wys/wys_shiyan/data_processed_new/pairs_train.tsv",
    "/home/wangjingyuan/wys/wys_shiyan/data_processed_new/pairs_val.tsv",
    "/home/wangjingyuan/wys/wys_shiyan/data_processed_new/pairs_test.tsv"
]

# 3. 输出目录
OUTPUT_DIR = "final_leakage_free_dataset"
# ===========================================

def parse_clstr(clstr_path):
    """
    解析 CD-HIT .clstr 文件
    返回字典: {cluster_id: [phage_id_1, phage_id_2, ...]}
    """
    print(f"📖 Parsing cluster file: {clstr_path} ...")
    clusters = defaultdict(list)
    current_cluster_id = None
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                # line example: >Cluster 0
                current_cluster_id = int(line.split()[1])
            else:
                # line example: 0	497513nt, >MT127620... *
                # 提取 > 和 ... 之间的内容
                part = line.split('>')[1].split('...')[0]
                phage_id = part.strip()
                clusters[current_cluster_id].append(phage_id)
    
    print(f"✅ Found {len(clusters)} clusters.")
    return clusters

def main():
    # 1. 读取并合并原始数据
    print("📂 Loading original interaction data...")
    df_list = []
    for fpath in ORIGINAL_FILES:
        if os.path.exists(fpath):
            df_list.append(pd.read_csv(fpath, sep='\t'))
    
    full_df = pd.concat(df_list, ignore_index=True)
    # 确保 phage_id 是字符串，去除可能的空格
    full_df['phage_id'] = full_df['phage_id'].astype(str).str.strip()
    
    print(f"📊 Total interactions loaded: {len(full_df)}")

    # 2. 解析聚类结果
    if not os.path.exists(CLSTR_FILE):
        print(f"❌ Error: Cluster file not found: {CLSTR_FILE}")
        print("请等待 CD-HIT 运行完毕后再运行此脚本。")
        return
    
    cluster_dict = parse_clstr(CLSTR_FILE)
    
    # 反向映射：知道每个 phage_id 属于哪个 cluster
    phage_to_cluster = {}
    for cid, phages in cluster_dict.items():
        for pid in phages:
            phage_to_cluster[pid] = cid

    # 3. 检查是否有噬菌体没被聚类 (理论上不应该发生)
    missing_phages = 0
    # 给数据打上 Cluster 标签
    def get_cluster(pid):
        return phage_to_cluster.get(pid, -1)

    full_df['cluster_id'] = full_df['phage_id'].apply(get_cluster)
    
    # 移除没找到聚类信息的行 (通常是因为ID匹配不上)
    missing_df = full_df[full_df['cluster_id'] == -1]
    if len(missing_df) > 0:
        print(f"⚠️ Warning: {len(missing_df)} interactions have phage IDs not found in clustering results.")
        # print(missing_df.head())
        full_df = full_df[full_df['cluster_id'] != -1]

    # 4. 按 Cluster 进行划分 (8:1:1)
    print("🎲 Splitting data by CLUSTERS (not sequences)...")
    
    unique_clusters = list(full_df['cluster_id'].unique())
    random.seed(42) # 保证结果可复现
    random.shuffle(unique_clusters)
    
    n_clusters = len(unique_clusters)
    n_train = int(n_clusters * 0.8)
    n_val = int(n_clusters * 0.1)
    
    train_clusters = set(unique_clusters[:n_train])
    val_clusters = set(unique_clusters[n_train : n_train + n_val])
    test_clusters = set(unique_clusters[n_train + n_val:])
    
    # 5. 根据 Cluster ID 提取数据
    train_df = full_df[full_df['cluster_id'].isin(train_clusters)]
    val_df   = full_df[full_df['cluster_id'].isin(val_clusters)]
    test_df  = full_df[full_df['cluster_id'].isin(test_clusters)]
    
    # 移除临时的 cluster_id 列
    train_df = train_df.drop(columns=['cluster_id'])
    val_df = val_df.drop(columns=['cluster_id'])
    test_df = test_df.drop(columns=['cluster_id'])

    # 6. 保存结果
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "pairs_train_cleaned.tsv"), sep='\t', index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "pairs_val_cleaned.tsv"), sep='\t', index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "pairs_test_cleaned.tsv"), sep='\t', index=False)

    print("\n✅ Done! New datasets saved to:", OUTPUT_DIR)
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples:   {len(val_df)}")
    print(f"   Test samples:  {len(test_df)}")
    print("\n✨ Check Leakage now: 现在测试集里的任何噬菌体，在训练集里都没有相似度 >95% 的亲戚了。")

if __name__ == "__main__":
    main()