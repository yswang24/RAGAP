# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import normalize
# import faiss

# # 读取簇嵌入
# df = pd.read_parquet("protein_clusters_emb.parquet")

# # 检查每个 embedding 的维度长度
# lengths = df["cluster_emb"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else None)
# main_length = lengths.mode()[0]  # 主流维度
# bad_rows = df[lengths != main_length]  # 异常行

# if len(bad_rows) > 0:
#     print(f"⚠️ 检测到 {len(bad_rows)} 行 embedding 维度异常，将被忽略 (占比 {len(bad_rows)/len(df):.2%})")
#     df = df[lengths == main_length]

# # 构建 embedding 矩阵
# embeddings = np.vstack(df["cluster_emb"].to_numpy())
# cluster_ids = df["cluster_id"].to_numpy()

# # 归一化向量（余弦相似性 = 内积）
# embeddings = normalize(embeddings, axis=1)

# # 建立 faiss 索引
# d = embeddings.shape[1]
# index = faiss.IndexFlatIP(d)  # 内积 index
# index.add(embeddings)

# # 搜索 KNN
# K = 20
# sims, idxs = index.search(embeddings, K+1)  # K+1 因为第一个是自己

# # 构建边
# edges = []
# for i, cluster in enumerate(cluster_ids):
#     for j, sim in zip(idxs[i][1:], sims[i][1:]):  # 去掉自身
#         if sim > 0.6:  # 阈值筛选
#             edges.append([cluster, cluster_ids[j], "protein-protein", float(sim)])

# # 保存
# edges_df = pd.DataFrame(edges, columns=["src_id", "dst_id", "edge_type", "weight"])
# edges_df.to_csv("protein_protein_edges_0.6.tsv", sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条 protein-protein 边")



# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import normalize
# import faiss

# # 读取簇嵌入
# df = pd.read_parquet("protein_clusters_emb.parquet")

# # 检查每个 embedding 的维度长度
# lengths = df["cluster_emb"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else None)
# main_length = lengths.mode()[0]  # 主流维度
# bad_rows = df[lengths != main_length]  # 异常行

# if len(bad_rows) > 0:
#     print(f"⚠️ 检测到 {len(bad_rows)} 行 embedding 维度异常，将被忽略 (占比 {len(bad_rows)/len(df):.2%})")
#     df = df[lengths == main_length]

# # 构建 embedding 矩阵
# embeddings = np.vstack(df["cluster_emb"].to_numpy())
# cluster_ids = df["cluster_id"].to_numpy()

# # 归一化向量（余弦相似性 = 内积）
# embeddings = normalize(embeddings, axis=1)

# # ---- GPU 加速部分 ----
# d = embeddings.shape[1]
# faiss.omp_set_num_threads(16)  # 多线程加速（CPU 和 GPU 混合都有效）

# cpu_index = faiss.IndexFlatIP(d)  # 精确内积索引
# res = faiss.StandardGpuResources()  # 分配 GPU 资源
# gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 把索引搬到 GPU，0 表示第 0 块 GPU

# # 添加向量
# gpu_index.add(embeddings)

# # 搜索 KNN
# K = 20
# sims, idxs = gpu_index.search(embeddings, K+1)  # K+1 因为第一个是自己

# # 构建边
# edges = []
# for i, cluster in enumerate(cluster_ids):
#     for j, sim in zip(idxs[i][1:], sims[i][1:]):  # 去掉自身
#         if sim > 0.9:  # 阈值筛选
#             edges.append([cluster, cluster_ids[j], "protein-protein", float(sim)])

# # 保存
# edges_df = pd.DataFrame(edges, columns=["src_id", "dst_id", "edge_type", "weight"])
# edges_df.to_csv("protein_protein_edges_0.9.tsv", sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条 protein-protein 边")



# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import normalize
# import faiss

# # 读取簇嵌入
# df = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/RBP_phage.parquet")

# # 检查每个 embedding 的维度长度
# lengths = df["embedding"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else None)
# main_length = lengths.mode()[0]  # 主流维度
# bad_rows = df[lengths != main_length]  # 异常行

# if len(bad_rows) > 0:
#     print(f"⚠️ 检测到 {len(bad_rows)} 行 embedding 维度异常，将被忽略 (占比 {len(bad_rows)/len(df):.2%})")
#     df = df[lengths == main_length]

# # 构建 embedding 矩阵
# embeddings = np.vstack(df["embedding"].to_numpy())
# cluster_ids = df["protein_id"].to_numpy()

# # 归一化向量（余弦相似性 = 内积）
# embeddings = normalize(embeddings, axis=1)

# # ---- GPU 加速部分 ----
# d = embeddings.shape[1]
# faiss.omp_set_num_threads(16)

# cpu_index = faiss.IndexFlatIP(d)  # 精确内积索引
# res = faiss.StandardGpuResources()  # 分配 GPU 资源
# gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# # 添加向量
# gpu_index.add(embeddings)

# # 搜索 KNN
# K = 2000
# sims, idxs = gpu_index.search(embeddings, K+1)  # K+1 因为第一个是自己

# # 构建邻居字典
# neighbors = {cluster_ids[i]: set(cluster_ids[idxs[i][1:]]) for i in range(len(cluster_ids))}

# # 构建互为 Top-K 的边
# edges = []
# for i, cluster in enumerate(cluster_ids):
#     for j_idx, sim in zip(idxs[i][1:], sims[i][1:]):
#         neighbor_id = cluster_ids[j_idx]
#         if sim > 0.9 and cluster in neighbors[neighbor_id]:  # 互为 Top-K
#             # 为了避免重复，只保留 cluster < neighbor_id 的组合
#             if cluster < neighbor_id:
#                 edges.append([cluster, neighbor_id, "protein-protein", float(sim)])




# # 1) sims 基本统计与分布（查看范围）
# import numpy as np
# all_sims = sims[:,1:].ravel()   # 去掉 self-sim (第一列)
# print("sims shape:", sims.shape)
# print("sim min/median/mean/max:", float(all_sims.min()), float(np.median(all_sims)),
#       float(all_sims.mean()), float(all_sims.max()))
# # 查看不同阈值对应的候选数量
# for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
#     cnt = (all_sims > t).sum()
#     print(f"sim > {t}: {cnt} entries ({cnt/len(all_sims):.2%})")

# # 2) 看 Top-K 互为关系受 K 的影响（统计互为候选数）

# # neighbors 已经是 {id: set(topK_ids)}
# mutual_count = 0
# for i, cluster in enumerate(cluster_ids):
#     for j_idx, sim in zip(idxs[i][1:], sims[i][1:]):
#         neighbor_id = cluster_ids[j_idx]
#         if cluster in neighbors.get(neighbor_id, set()):
#             mutual_count += 1
# print("total mutual directed pairs (counting A->B and B->A):", mutual_count)
# # 转为无向边数（大致）
# print("approx mutual undirected edges (upper bound):", mutual_count//2)

# # 3) 检查是否存在大量几乎相同向量（重复）
# # 随机抽 1000 对计算相似度，看是否很多接近 1.0
# import random
# n = min(1000, len(cluster_ids))
# idxs_sample = random.sample(range(len(cluster_ids)), n)
# pairs = []
# for i in idxs_sample:
#     j = idxs[i][1]  # each vector's top neighbor (excluding itself)
#     pairs.append(sims[i][1])
# pairs = np.array(pairs)
# print("sample top-1 sims: min/mean/max:", pairs.min(), pairs.mean(), pairs.max())



# import numpy as np
# import pandas as pd

# # 假设你已有 embeddings, cluster_ids, sims, idxs
# # embeddings: (N, D) numpy array, dtype float32
# # sims: (N, K+1) numpy array from faiss
# # idxs: (N, K+1) numpy array

# # 1) 检查向量范数（是否正确归一化）
# norms = np.linalg.norm(embeddings, axis=1)
# print("norms: min/median/mean/max:", norms.min(), np.median(norms), norms.mean(), norms.max())

# # 2) 查看有多少完全相同的向量（字面相同）
# # 将每行转为 bytes/hash 进行去重检测（对浮点有容忍度）
# rounded = np.round(embeddings, 6)  # 量化到 1e-6，防止浮点微小差异
# as_tuples = [tuple(row) for row in rounded]
# s = pd.Series(as_tuples)
# n_unique = s.nunique()
# N = embeddings.shape[0]
# print("total vectors:", N, "unique after rounding(1e-6):", n_unique,
#       "duplicates:", N - n_unique, f"({(N-n_unique)/N:.2%})")

# # 3) 查看 top1 相似度分布（你已部分看到，但更详细）
# top1 = sims[:,1]   # 第1是 top-1 excluding self
# print("top1 min/mean/median/max:", top1.min(), top1.mean(), np.median(top1), top1.max())

# # 4) 若存在大量近似重复，显示几个重复示例（蛋白 id）
# from collections import defaultdict
# hash_map = defaultdict(list)
# for i, row in enumerate(rounded):
#     hash_map[tuple(row)].append(i)

# # 打印几个重复严重的键（若有）
# dups = [(k, v) for k,v in hash_map.items() if len(v) > 1]
# dups = sorted(dups, key=lambda x: len(x[1]), reverse=True)[:5]
# for k,v in dups:
#     print("one vector duplicated count:", len(v), "indices sample:", v[:10])









# # 保存
# edges_df = pd.DataFrame(edges, columns=["src_id", "dst_id", "edge_type", "weight"])
# edges_df.to_csv("/home/wangjingyuan/wys/WYSPHP/RBP_650/protein_protein_RBP_phage_650_edges_0.9_test.tsv", sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条互为 Top-{K} 的无向 protein-protein 边")







import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import faiss
import random
import matplotlib.pyplot as plt

# 读取簇嵌入
df = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/RBP_650/RBP_phage_650.parquet")

# 1) 检查 embedding 维度一致性（更稳健）
lengths = df["embedding"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else np.nan)
main_length = lengths.value_counts(dropna=True).idxmax()
bad_rows = df[lengths != main_length]
if len(bad_rows) > 0:
    print(f"⚠️ 检测到 {len(bad_rows)} 行 embedding 维度异常，将被忽略 (占比 {len(bad_rows)/len(df):.2%})")
    df = df[lengths == main_length].copy()

# 构建 embedding 矩阵（确保顺序不变）
emb_list = df["embedding"].tolist()
embeddings = np.asarray(emb_list)  # 形状 (n, d)
n, d = embeddings.shape
print("n, d:", n, d)

# 归一化并转换为 float32、C 连续
embeddings = normalize(embeddings, axis=1)
embeddings = np.asarray(embeddings, dtype="float32", order="C")

# cluster ids
cluster_ids = np.asarray(df["protein_id"].to_numpy())  # 确保 numpy array

# ---- FAISS GPU 初始化（稳健处理 K 越界） ----
faiss.omp_set_num_threads(16)  # 影响 CPU 部分
cpu_index = faiss.IndexFlatIP(d)  # 内积索引（归一化后即 cosine）
# 若数据量非常大，IndexFlatIP 内存会很大——注意
res = faiss.StandardGpuResources()
gpu_id = 0
gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)

gpu_index.add(embeddings)  # 添加向量到索引

# K 不能超过 n-1
K = 2000
K = min(K, n-1)
search_K = K + 1  # 包含可能的 self
sims, idxs = gpu_index.search(embeddings, search_K)  # 返回 shape (n, search_K)

print("sims shape:", sims.shape)
all_sims = sims[:, 1:].ravel()  # 去掉第二列开始的 self 假设（但我们下面会更稳健地跳过 self）
print("sim min/median/mean/max:", float(all_sims.min()), float(np.median(all_sims)),
      float(all_sims.mean()), float(all_sims.max()))
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    cnt = (all_sims > t).sum()
    print(f"sim > {t}: {cnt} entries ({cnt/len(all_sims):.2%})")

# 可视化相似度分布（建议）
plt.figure(figsize=(6,3))
plt.hist(all_sims, bins=100)
plt.title("Distribution of pairwise top-K sims (excluding self)")
plt.xlabel("cosine similarity")
plt.ylabel("count")
plt.tight_layout()

# 保存为高分辨率 PNG 文件
plt.savefig("similarity_distribution.png", dpi=300)
plt.close()


# 2) 构建 neighbors：排除 self 并只取前 K 有效 neighbor ids（更稳健）
neighbors = {}
for i in range(n):
    neigh_ids = []
    for j in idxs[i]:
        if j == i:
            continue
        neigh_ids.append(cluster_ids[j])
        if len(neigh_ids) >= K:
            break
    neighbors[cluster_ids[i]] = set(neigh_ids)

# 3) 构建互为 Top-K 的无向边集合（使用 frozenset 去重）
edges = []
edge_set = set()
for i in range(n):
    cid = cluster_ids[i]
    # iterate through returned neighbors and sims — 找到非 self 的 pairs
    for j_pos, j_idx in enumerate(idxs[i]):
        if j_idx == i:
            continue
        if j_pos >= search_K:  # safety, although loop over idxs ensures length
            break
        sim = sims[i, j_pos]
        neighbor_id = cluster_ids[j_idx]
        if sim > 0.9 and cid in neighbors.get(neighbor_id, set()):  # 互为 Top-K 且阈值
            pair = frozenset({cid, neighbor_id})
            if pair not in edge_set:
                edge_set.add(pair)
                # 记录一条边（附上相似度，可以用两者相似度的最大/平均作为权重）
                edges.append([cid, neighbor_id, "protein-protein", float(sim)])

print("num edges (undirected, threshold 0.9):", len(edges))

# 4) 互为候选数统计（更准确地统计无向）
mutual_pairs = set()
for a, neighs in neighbors.items():
    for b in neighs:
        if a in neighbors.get(b, set()):
            mutual_pairs.add(frozenset({a, b}))
print("mutual undirected edges (upper bound):", len(mutual_pairs))

# 5) 检查 top-1 样本（稳健获取第一个非 self 的 neighbor）
n_sample = min(1000, n)
idxs_sample = random.sample(range(n), n_sample)
pairs = []
for i in idxs_sample:
    # 找第一个 j != i
    first_sim = None
    for j_pos, j_idx in enumerate(idxs[i]):
        if j_idx == i:
            continue
        first_sim = sims[i, j_pos]
        break
    if first_sim is not None:
        pairs.append(first_sim)
pairs = np.array(pairs)
print("sample top-1 sims: min/mean/max:", pairs.min(), pairs.mean(), pairs.max())


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

emb = embeddings.copy()  # 你的 (n, d) 矩阵，已 normalize 为 float32

# 1) 检查重复向量（完全相等或几乎相等）
# 完全相等（字符串哈希）
rows_as_tuples = [tuple(row) for row in emb]
unique_count = len(set(rows_as_tuples))
print("n, unique exact rows:", emb.shape[0], unique_count, "duplicates:", emb.shape[0]-unique_count)

# 近重复（使用随机抽样的最近邻距离）
def sample_nearest_stats(mat, sample_n=1000):
    import faiss
    m, d = mat.shape
    sample_idx = np.random.choice(m, min(sample_n, m), replace=False)
    sub = mat[sample_idx].astype('float32')
    idx = faiss.IndexFlatIP(d)
    idx.add(mat.astype('float32'))
    sims, idxs = idx.search(sub, 2)  # self + nearest
    # take nearest non-self
    nn_sims = []
    for i in range(len(sample_idx)):
        if idxs[i,0] == sample_idx[i]:
            nn_sims.append(sims[i,1])
        else:
            nn_sims.append(sims[i,0])
    nn_sims = np.array(nn_sims)
    return nn_sims.min(), nn_sims.mean(), nn_sims.max(), np.quantile(nn_sims, [0.25,0.5,0.75])
print("sample nn sims stats (min/mean/max/quantiles):", sample_nearest_stats(emb, sample_n=2000))

# 2) 每维方差与零/常数列检查
stds = emb.std(axis=0)
print("embedding dim:", emb.shape[1], "std mean/min/max:", stds.mean(), stds.min(), stds.max())
const_dims = (stds == 0).sum()
print("constant dims (std==0):", const_dims)

# 3) 全局均向量（mean vector）与去中心化后相似度差异
mean_vec = emb.mean(axis=0, keepdims=True)
mean_norm = np.linalg.norm(mean_vec)
print("mean vector norm:", mean_norm)
# cosine between each vector and mean
cos_to_mean = (emb @ (mean_vec.T)).ravel()
print("cosine to mean: min/median/max:", cos_to_mean.min(), np.median(cos_to_mean), cos_to_mean.max())

# 4) PCA: 看前几个主成分解释的方差比例（是否被少数主成分统治）
pca = PCA(n_components=min(50, emb.shape[1]))
pca.fit(emb)
explained = pca.explained_variance_ratio_
print("PCA top 10 var ratios:", explained[:10])
print("sum top 1/5/10:", explained[0], explained[:5].sum(), explained[:10].sum())

# 保存
# edges_df = pd.DataFrame(edges, columns=["src_id", "dst_id", "edge_type", "weight"])
# edges_df.to_csv("/home/wangjingyuan/wys/WYSPHP/RBP_650/protein_protein_RBP_phage_650_edges_0.9.tsv", sep="\t", index=False)
# print(f"✅ 已保存 {len(edges_df)} 条互为 Top-{K} 的无向 protein-protein 边")

