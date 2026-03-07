#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_hosts_threshold.py

基于 sourmash compare 输出的 Jaccard 相似度矩阵，
使用层次聚类（Average linkage）按距离阈值切簇，并选出每簇的 Medoid 代表。

依赖：
    pip install numpy pandas scipy

用法示例：
    python3 cluster_hosts_threshold.py \
      --matrix /home/wangjingyuan/wys/create_edge/work-21-PP/compare/compare_matrix.npz \
      --labels /home/wangjingyuan/wys/create_edge/work-21-PP/compare/compare_matrix.npz.labels.txt \
      --threshold 0.05 \
      --out_cluster phage_clusters.csv \
      --out_representatives phage_representatives.txt

     python3 cluster_hosts_threshold.py \
      --matrix /home/wangjingyuan/wys/create_edge/work-21-HH/compare/compare_matrix.npz \
      --labels /home/wangjingyuan/wys/create_edge/work-21-HH/compare/compare_matrix.npz.labels.txt \
      --threshold 0.05 \
      --out_cluster host_clusters.csv \
      --out_representatives host_representatives.txt
"""

import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

def parse_args():
    p = argparse.ArgumentParser(
        description="基于 Jaccard 相似度矩阵按距离阈值做层次聚类，并选 Medoid 代表"
    )
    p.add_argument('--matrix', required=True,
                   help='sourmash compare 输出的 compare_matrix.npz 文件（或 .npy 文件）')
    p.add_argument('--labels', required=True,
                   help='compare_matrix.npz.labels.txt 文件，每行一个样本 ID')
    p.add_argument('--threshold', type=float, default=0.2,
                   help='距离阈值 t，cutoff distance = t，距 ≤ t 的样本归同簇；默认 0.2 (相似度 ≥ 0.8)')
    p.add_argument('--out_cluster', default='host_clusters.csv',
                   help='输出 CSV：每个样本所属簇编号')
    p.add_argument('--out_representatives', default='host_representatives.txt',
                   help='输出 TXT：每簇选出的代表样本 ID')
    return p.parse_args()

def main():
    args = parse_args()

    # 1. 加载相似度矩阵 & 标签
    data = np.load(args.matrix, allow_pickle=True)
    # 如果是 NpzFile，就从 data.files 中取；否则直接当 ndarray 用
    if hasattr(data, 'files'):
        # sourmash compare 默认 key 是 'data'，否则取第一个
        key = 'data' if 'data' in data.files else data.files[0]
        sim_matrix = data[key]
    else:
        sim_matrix = data
    labels = [line.strip() for line in open(args.labels, 'r')]

    n = sim_matrix.shape[0]
    assert n == len(labels), f"样本数不匹配：矩阵 {n} vs 标签 {len(labels)}"

    # 2. 转为距离矩阵（1 - Jaccard）
    dist_matrix = 1.0 - sim_matrix
    dist_vector = squareform(dist_matrix, checks=False)

    # 3. 层次聚类（average linkage）
    linkage_matrix = linkage(dist_vector, method='average')

    # 4. 按距离阈值切簇
    clusters = fcluster(linkage_matrix, t=args.threshold, criterion='distance')

    # 5. 保存聚类结果
    df = pd.DataFrame({'id': labels, 'cluster': clusters})
    df.to_csv(args.out_cluster, index=False)
    print(f"[+] 聚类完成：共 {df['cluster'].nunique()} 个簇，结果保存至 {args.out_cluster}")

    # 6. 每簇选 medoid 代表
    representatives = []
    for cid in sorted(df['cluster'].unique()):
        members = df[df['cluster'] == cid]['id'].tolist()
        idxs = [labels.index(m) for m in members]
        subdist = dist_matrix[np.ix_(idxs, idxs)]
        medoid_idx = np.argmin(subdist.sum(axis=1))
        representatives.append(members[medoid_idx])

    # 7. 保存代表列表
    with open(args.out_representatives, 'w') as f:
        for rep in representatives:
            f.write(rep + "\n")
    print(f"[+] 每簇 Medoid 代表已写入 {args.out_representatives}")

if __name__ == '__main__':
    main()