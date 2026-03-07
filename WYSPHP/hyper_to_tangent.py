
# import pandas as pd
# df = pd.read_parquet("/home/wangjingyuan/wys/WYSPHP/taxonomy_tree/zuiyou/taxonomy_poincare_dep_neg_120.parquet")
# import pandas as pd
# import numpy as np


# # 展示前 5 行
# for i in range(5):
#     taxid = df.loc[i, "taxid"]
#     hyper = df.loc[i, "hyperbolic_emb"]
#     tangent = df.loc[i, "tangent_emb"]

#     print(f"Taxid: {taxid}")
#     print(f"  Hyperbolic emb (前10维): {hyper[:10]}")
#     print(f"  Tangent emb    (前10维): {tangent[:10]}")
#     print("-" * 60)

# # 如果想展开成矩阵形式
# H = np.stack(df["hyperbolic_emb"].to_numpy())  # (N, dim)
# T = np.stack(df["tangent_emb"].to_numpy())     # (N, dim)

# print("双曲空间向量矩阵 shape:", H.shape)
# print("切空间向量矩阵 shape:", T.shape)


import pandas as pd
df=pd.read_parquet('/home/wangjingyuan/wys/WYSPHP/esm_embeddings_35_phage_parquet_final/AB626963.parquet')


# 查看前几行
print(df.head(10))

# 查看列和数据类型
print(df.info())

# 查看基本统计信息
print(df.describe())

# 查看行数和列数
print(df.shape)


# emb = df["embedding"].iloc[0]

# print(type(emb))       # 看看存的是什么类型（list / numpy array / string）
# print(len(emb))