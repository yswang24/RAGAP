#!/usr/bin/env python3
"""
compute_cluster_embeddings.py

输入:
  - protein_catalog.parquet (columns: protein_id, source_type, source_id, embedding)
  - protein_clusters.parquet (columns: protein_id, cluster_id)

输出:
  - protein_cluster_emb.parquet (columns: cluster_id, cluster_emb, medoid_protein, size, members_sample (optional))
  - optional cluster_cluster_sim.tsv (topk similarities)

用法示例:
python compute_cluster_embeddings.py \
  --protein_catalog protein_catalog.parquet \
  --protein_clusters protein_clusters.parquet \
  --out_cluster_parquet protein_clusters_emb.parquet \
  --method mean_medoid \
  --emb_col embedding \
  --topk_sim 10
"""
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def load_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protein_catalog", required=True)
    p.add_argument("--protein_clusters", required=True)
    p.add_argument("--out_cluster_parquet", required=True)
    p.add_argument("--method", choices=["mean","medoid","mean_medoid","weighted","attention","multi_proto"], default="mean_medoid")
    p.add_argument("--emb_col", default="embedding")  # column name in protein_catalog
    p.add_argument("--weight_col", default=None, help="optional weight column name in protein_catalog for weighted pooling")
    p.add_argument("--proto_k", type=int, default=2, help="for multi_proto: prototypes per cluster")
    p.add_argument("--topk_sim", type=int, default=0, help=">0 to compute cluster-cluster topk similarities")
    p.add_argument("--sim_out", default="cluster_cluster_sim.tsv")
    p.add_argument("--sample_member_n", type=int, default=5, help="save sample members for debugging")
    return p.parse_args()

def safe_stack(ser):
    arrs = []
    for v in ser:
        if v is None:
            continue
        arrs.append(np.array(v, dtype=np.float32))
    if len(arrs)==0:
        return np.zeros((0,))
    return np.vstack(arrs)

def compute_medoid(embs):
    if embs.shape[0]==0:
        return None, -1
    if embs.shape[0]==1:
        return 0, embs[0]
    sim = cosine_similarity(embs)
    scores = sim.sum(axis=1)
    idx = int(np.argmax(scores))
    return idx, embs[idx]

def clean_protein_id(pid: str) -> str:
    """
    处理 cluster 文件里的 protein_id:
    e.g. 'NZ_JQLH01000001.1_1723|source=host|host_id=GCF_000744105'
         -> 'NZ_JQLH01000001.1_1723'
    """
    return pid.split("|")[0]

def main():
    args = load_args()
    df_prot = pd.read_parquet(args.protein_catalog)
    df_map = pd.read_parquet(args.protein_clusters)

    # clean IDs
    df_prot['protein_id'] = df_prot['protein_id'].astype(str)
    df_map['protein_id'] = df_map['protein_id'].astype(str).apply(clean_protein_id)

    if 'cluster_id' not in df_map.columns:
        raise RuntimeError("protein_clusters parquet must contain cluster_id column")

    # merge
    df = df_map.merge(df_prot, on='protein_id', how='left')
    missing = df[args.emb_col].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} proteins missing embeddings after merge; they will be ignored in pooling.")

    # group by cluster
    groups = df.groupby('cluster_id')
    cluster_ids, cluster_embs, medoid_prots, sizes, sample_members = [], [], [], [], []

    for cid, g in tqdm(groups, desc="clusters"):
        embs, prot_ids = [], []
        for _, r in g.iterrows():
            emb = r.get(args.emb_col)
            if emb is None or (isinstance(emb, float) and np.isnan(emb)):
                continue
            embs.append(np.array(emb, dtype=np.float32))
            prot_ids.append(str(r['protein_id']))

        n = len(embs)
        sizes.append(n)
        cluster_ids.append(str(cid))

        if n == 0:
            cluster_embs.append(np.zeros((1,), dtype=np.float32).tolist())
            medoid_prots.append("")
            sample_members.append([])
            continue

        embs_mat = np.vstack(embs)
        mean_emb = embs_mat.mean(axis=0)

        if args.method in ("mean", "mean_medoid"):
            chosen_emb = mean_emb
        elif args.method == "medoid":
            midx, med = compute_medoid(embs_mat)
            chosen_emb = med
            medoid_prots.append(prot_ids[midx])
        elif args.method == "weighted":
            if args.weight_col is None:
                raise RuntimeError("weighted method needs --weight_col")
            weights = []
            for pid in prot_ids:
                w = g.loc[g['protein_id'] == pid, args.weight_col].values
                weights.append(float(w[0]) if len(w) and not np.isnan(w[0]) else 1.0)
            w = np.array(weights, dtype=np.float32)
            w = w / (w.sum() + 1e-12)
            chosen_emb = (embs_mat * w[:, None]).sum(axis=0)
        elif args.method == "attention":
            sims = embs_mat.dot(mean_emb)
            alphas = np.exp(sims - sims.max())
            alphas = alphas / (alphas.sum() + 1e-12)
            chosen_emb = (embs_mat * alphas[:, None]).sum(axis=0)
        elif args.method == "multi_proto":
            from sklearn.cluster import KMeans
            k = min(args.proto_k, n)
            km = KMeans(n_clusters=k, random_state=0, n_init=5).fit(embs_mat)
            centers = km.cluster_centers_
            chosen_emb = centers.mean(axis=0)
        else:
            chosen_emb = mean_emb

        if args.method == "mean_medoid":
            midx, med = compute_medoid(embs_mat)
            medoid_prots.append(prot_ids[midx] if midx >= 0 else "")
        elif args.method != "medoid":
            medoid_prots.append("")

        cluster_embs.append(chosen_emb.astype(np.float32).tolist())
        sample_members.append(prot_ids[:args.sample_member_n])

    out_df = pd.DataFrame({
        "cluster_id": cluster_ids,
        "cluster_emb": cluster_embs,
        "medoid_protein": medoid_prots,
        "size": sizes,
        "members_sample": sample_members
    })
    out_df.to_parquet(args.out_cluster_parquet, index=False)
    print("Wrote cluster embeddings to", args.out_cluster_parquet)

if __name__ == "__main__":
    main()
