
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
print(pd.read_parquet('/wys/WYSPHP/dnabert6_phage_embeddings/AB626963.parquet.parquet').columns)
