#!/usr/bin/env python3
# -*- coding: utf-8
"""
Evaluate Poincare/tangent embeddings for taxonomy.

Requirements:
  pip install pyarrow numpy scipy scikit-learn networkx matplotlib umap-learn

Usage:
  python eval_taxonomy_embeddings.py \
    --parquet taxonomy_poincare_dep.parquet \
    --taxonomy taxonomy_with_alias.tsv \
    --sample_pairs 50000 \
    --knn 10
"""
import argparse
import csv
import math
import random
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import networkx as nx
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
try:
    import umap
except Exception:
    umap = None

# --------------------------
# IO
# --------------------------
def load_parquet_embeddings(parquet_path):
    table = pq.read_table(parquet_path)
    taxids = table.column("taxid").to_pylist()
    hyper = table.column("hyperbolic_emb").to_pylist()  # list of lists
    tangent = table.column("tangent_emb").to_pylist()
    hyper = np.array([np.array(x, dtype=np.float32) for x in hyper])
    tangent = np.array([np.array(x, dtype=np.float32) for x in tangent])
    return taxids, hyper, tangent

def read_taxonomy(path):
    tax = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tid = row.get("taxid","").strip()
            pid = row.get("parent_taxid","").strip()
            rank = row.get("rank","").strip()
            alias = row.get("alias","").strip()
            tax[tid] = {"parent": pid, "rank": rank, "alias": alias}
    return tax

# --------------------------
# Graph distances
# --------------------------
def build_graph(taxonomy):
    G = nx.DiGraph()
    for tid, rec in taxonomy.items():
        G.add_node(tid)
        parent = rec.get("parent")
        if parent and parent in taxonomy and parent != tid:
            G.add_edge(parent, tid)
            G.add_edge(tid, parent)  # undirected shortest path
    return G

def sample_node_pairs(taxids, sample_pairs):
    n = len(taxids)
    pairs = set()
    while len(pairs) < sample_pairs:
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        pairs.add((i,j))
    return list(pairs)

# --------------------------
# Metrics
# --------------------------
def compute_tree_distances(G, taxids, pairs, max_dist=100):
    """Return list of tree distances (int) for pairs; if no path, return max_dist."""
    dist_list = []
    for i,j in pairs:
        a = taxids[i]
        b = taxids[j]
        try:
            d = nx.shortest_path_length(G, source=a, target=b)
        except nx.NetworkXNoPath:
            try:
                d = nx.shortest_path_length(G, source=b, target=a)
            except nx.NetworkXNoPath:
                d = max_dist
        dist_list.append(d)
    return np.array(dist_list, dtype=np.int32)

def poincare_distance_matrix(hyper_a, hyper_b=None):
    """Compute pairwise Poincare distance between rows of hyper_a and hyper_b (or itself)"""
    if hyper_b is None:
        hyper_b = hyper_a
    # vectorized formula
    uu = np.sum(hyper_a**2, axis=1, keepdims=True)  # (n,1)
    vv = np.sum(hyper_b**2, axis=1, keepdims=True)  # (m,1)
    # compute squared diff pairs: using broadcasting
    # careful about memory: do chunking if large
    n = hyper_a.shape[0]
    m = hyper_b.shape[0]
    D = np.zeros((n,m), dtype=np.float32)
    for i in range(n):
        diff = hyper_a[i:i+1,:] - hyper_b  # (m,dim)
        dd = np.sum(diff*diff, axis=1)
        denom = (1.0 - uu[i,0]) * (1.0 - vv[:,0])
        denom = np.clip(denom, 1e-15, None)
        arg = 1.0 + 2.0 * dd / denom
        arg = np.clip(arg, 1.0 + 1e-7, None)
        z = np.sqrt(arg - 1.0) * np.sqrt(arg + 1.0)
        dist = np.log(arg + z)
        D[i,:] = dist
    return D

# --------------------------
# Parent MRR / Recall@k
# --------------------------
def parent_mrr_recall(taxids, hyper, taxonomy, idx_map, topk=[1,5,10]):
    # build nearest neighbors on hyper (tangent could be used instead)
    nbrs = NearestNeighbors(n_neighbors=max(topk)+1, metric='euclidean').fit(hyper)  # +1 because self included
    distances, indices = nbrs.kneighbors(hyper)
    # indices[i,0] == i (self)
    rr_total = 0.0
    hits = {k:0 for k in topk}
    N = len(taxids)
    for i, tid in enumerate(taxids):
        parent = taxonomy[tid]['parent']
        if not parent:
            continue
        # find parent's index if present
        pj = idx_map.get(parent)
        if pj is None:
            continue
        # flatten neighbor list skipping self
        neighs = [int(x) for x in indices[i] if int(x) != i]
        # find rank of parent
        try:
            rankpos = neighs.index(pj) + 1  # 1-based
            rr_total += 1.0 / rankpos
        except ValueError:
            # not found in neighbors within max(topk)
            pass
        for k in topk:
            if pj in neighs[:k]:
                hits[k] += 1
    mrr = rr_total / N
    recall_at_k = {k: hits[k] / N for k in topk}
    return mrr, recall_at_k

# --------------------------
# KNN purity
# --------------------------
def knn_purity(taxids, emb, taxonomy, idx_map, K=10, rank_level='genus'):
    nbrs = NearestNeighbors(n_neighbors=K+1, metric='euclidean').fit(emb)
    dists, inds = nbrs.kneighbors(emb)
    purities = []
    for i, tid in enumerate(taxids):
        label = taxonomy[tid].get('rank','')
        # choose taxonomy label at desired level: here we check exact rank string equal.
        # Better: you may provide external mapping taxid->genus_id to compare.
        # For now use 'rank' field crude check: skip if rank != target
        # We will instead use alias/given labels not rank string; user may adapt.
        # So compute fraction of neighbors that share same 'parent' at species/genus if available:
        same_parent_count = 0
        total = 0
        parent = taxonomy[tid].get('parent')
        for j in inds[i]:
            if j == i:
                continue
            other = taxids[j]
            if taxonomy[other].get('parent') == parent:
                same_parent_count += 1
            total += 1
        if total > 0:
            purities.append(same_parent_count / total)
    if len(purities) == 0:
        return 0.0
    return float(np.mean(purities))

# --------------------------
# Visualizations
# --------------------------
def plot_distance_scatter(tree_dists, hyper_dists, out_png="dist_scatter.png", sample_size=2000):
    # sample for plotting
    n = len(tree_dists)
    idx = np.random.choice(n, min(n, sample_size), replace=False)
    x = tree_dists[idx]
    y = hyper_dists[idx]
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=3, alpha=0.4)
    plt.xlabel("Tree distance (shortest path)")
    plt.ylabel("Hyperbolic distance")
    plt.title("Tree dist vs Hyperbolic dist")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Wrote", out_png)

def plot_umap(tangent_emb, taxids, taxonomy, out_png="umap.png"):
    if umap is None:
        print("umap not installed; skip UMAP plot")
        return
    reducer = umap.UMAP(n_components=2, random_state=0)
    emb2 = reducer.fit_transform(tangent_emb)
    # color by family/parent id coarse
    parents = [taxonomy[t]['parent'] for t in taxids]
    uniq = list(sorted(set(parents)))
    cmap = {u:i for i,u in enumerate(uniq)}
    colors = [cmap[p] for p in parents]
    plt.figure(figsize=(6,6))
    plt.scatter(emb2[:,0], emb2[:,1], c=colors, s=6, alpha=0.8)
    plt.title("UMAP of tangent embeddings (color by parent taxid)")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Wrote", out_png)

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="taxonomy_poincare.parquet")
    parser.add_argument("--taxonomy", required=True, help="taxonomy_with_alias.tsv")
    parser.add_argument("--map", required=False, help="taxid_index_map.tsv (optional but recommended)")
    parser.add_argument("--sample_pairs", type=int, default=20000, help="number of node pairs to sample for dist correlation")
    parser.add_argument("--knn", type=int, default=10)
    args = parser.parse_args()

    print("Loading embeddings...")
    taxids, hyper, tangent = load_parquet_embeddings(args.parquet)
    print("Loaded", len(taxids), "taxids; hyper dim:", hyper.shape[1], "tangent dim:", tangent.shape[1])

    taxonomy = read_taxonomy(args.taxonomy)
    G = build_graph(taxonomy)
    print("Graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())

    # build index map
    idx_map = {taxid:i for i,taxid in enumerate(taxids)}
    if args.map:
        # try to load but optional
        try:
            with open(args.map,'r',encoding='utf-8') as f:
                next(f)
                for line in f:
                    a,b=line.strip().split("\t")
                    idx_map[a]=int(b)
        except Exception:
            pass

    # sample pairs
    pairs = sample_node_pairs(taxids, args.sample_pairs)
    print("Sampled pairs:", len(pairs))

    print("Computing tree distances (sampled)...")
    tree_dists = compute_tree_distances(G, taxids, pairs, max_dist=100)

    print("Computing hyperbolic distances (sampled)...")
    # compute hyperbolic distances only for sampled pairs
    hyper_dists = []
    for i,j in pairs:
        u = hyper[i]
        v = hyper[j]
        # use same formula as train
        uu = np.sum(u*u)
        vv = np.sum(v*v)
        dd = np.sum((u-v)**2)
        denom = (1.0-uu)*(1.0-vv)
        denom = max(1e-15, denom)
        arg = 1.0 + 2.0*dd/denom
        arg = max(1.0+1e-7, arg)
        z = math.sqrt(arg-1.0)*math.sqrt(arg+1.0)
        d = math.log(arg+z)
        hyper_dists.append(d)
    hyper_dists = np.array(hyper_dists, dtype=float)

    # Spearman correlation
    rho, pval = spearmanr(tree_dists, hyper_dists)
    print("Spearman rho (tree_dist vs hyper_dist):", rho, "p=", pval)

    # parent retrieval MRR/Recall
    print("Computing parent MRR / Recall@k...")
    mrr, recall_at_k = parent_mrr_recall(taxids, tangent, taxonomy, idx_map, topk=[1,5,10])
    print("Parent MRR:", mrr)
    print("Recall@k:", recall_at_k)

    # KNN purity (coarse: share same parent)
    purity = knn_purity(taxids, tangent, taxonomy, idx_map, K=args.knn)
    print(f"KNN purity@{args.knn} (share parent):", purity)

    # Silhouette (requires labels) -> we use parent ids as labels where available
    labels = []
    valid_idx = []
    for i,t in enumerate(taxids):
        p = taxonomy[t].get('parent')
        if p:
            labels.append(p)
            valid_idx.append(i)
    if len(set(labels)) > 1:
        sil = silhouette_score(tangent[valid_idx], labels)
        print("Silhouette (by parent):", sil)
    else:
        print("Silhouette: not enough label classes")

    # optional plots
    plot_distance_scatter(tree_dists, hyper_dists, out_png="dist_scatter.png")
    if umap is not None:
        plot_umap(tangent, taxids, taxonomy, out_png="umap_tangent.png")

    print("Done.")

if __name__ == "__main__":
    main()
