#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Poincaré taxonomy trainer using geoopt + optional FAISS hard-neg mining.

Saves:
  - embeddings (hyperbolic coordinates) and tangent embeddings (.pt)
  - also optional parquet with taxid + embeddings for interoperability

Key improvements vs earlier script:
 - use geoopt.manifold methods (dist, logmap0) for numeric stability
 - vectorized batch training with margin loss (pos vs neg)
 - robust hard-neg selection using FAISS (if installed) with ancestor/descendant filtering
 - deterministic seeding, clearer device handling and warnings
 - better I/O: .pt dumps + parquet optional
 - evaluation: parent MRR/recall and spearman sampling (controlled sample size)

Usage example:
 python train_poincare_geoopt.py \
   --in taxonomy_with_alias.tsv \
   --out taxonomy_poincare.pt \
   --parquet_out taxonomy_poincare.parquet \
   --dim 64 --epochs 400 --lr 5e-3 --neg 50 --batch 512 --device cuda \
   --max_anc_depth 1 --anc_alpha 1.0 --hard_every 5 --topk_hard 100 --seed 42
"""
import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import geoopt
from geoopt import ManifoldParameter, PoincareBall
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import networkx as nx

# Optional faiss import
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# --------------------------
# Utilities & numeric helpers
# --------------------------
EPS = 1e-7


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_device(device_str: str) -> torch.device:
    """
    Return torch.device with clearer warnings if requested CUDA but unavailable.
    device_str examples: 'cpu', 'cuda', 'cuda:0'
    """
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
        else:
            return torch.device(device_str)
    return torch.device(device_str)


# --------------------------
# IO: taxonomy reading
# --------------------------
def read_taxonomy_table(path: str) -> Dict[str, Dict[str, str]]:
    """
    Read taxonomy TSV with header including at least taxid, parent_taxid, name, rank, alias (alias optional).
    Returns dict[taxid] -> record dict
    """
    tax = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tid = row.get("taxid", "").strip()
            if not tid:
                continue
            tax[tid] = {
                "parent": row.get("parent_taxid", "").strip(),
                "name": row.get("name", "").strip(),
                "rank": row.get("rank", "").strip(),
                "alias": row.get("alias", "").strip() if "alias" in row else "",
            }
    return tax


# --------------------------
# graph & ancestry computation
# --------------------------
def build_graph_from_taxonomy(tax: Dict[str, Dict[str, str]]) -> nx.DiGraph:
    """
    Build directed graph edges child -> parent for ancestry operations.
    Also add nodes even if isolated.
    """
    G = nx.DiGraph()
    for tid in tax.keys():
        G.add_node(tid)
    for tid, rec in tax.items():
        parent = rec.get("parent", "")
        if parent and parent in tax and parent != tid:
            G.add_edge(tid, parent)
    return G


def compute_ancestors_full(tax: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    """
    For each node, compute full ancestor chain up to root (excluding itself).
    This returns list ordered from parent -> grandparent -> ...
    """
    ancestors = {}
    for tid in tax:
        cur = tid
        anc_list = []
        visited = set()
        while True:
            parent = tax.get(cur, {}).get("parent", "")
            if not parent or parent == cur or parent in visited:
                break
            anc_list.append(parent)
            visited.add(parent)
            cur = parent
        ancestors[tid] = anc_list
    return ancestors


def compute_ancestors_with_depth(tax: Dict[str, Dict[str, str]], max_depth: int = 2) -> Dict[str, List[Tuple[str, int]]]:
    """
    Compute ancestors up to max_depth returning list of (ancestor, distance) with distance 1..max_depth
    """
    result = {}
    for tid in tax:
        cur = tid
        pairs = []
        for d in range(1, max_depth + 1):
            parent = tax.get(cur, {}).get("parent", "")
            if not parent or parent == cur:
                break
            pairs.append((parent, d))
            cur = parent
        result[tid] = pairs
    return result


# --------------------------
# positive pair builder
# --------------------------
def build_pos_pairs(nodes: List[str], idx_of: Dict[str, int], ancestors_with_depth: Dict[str, List[Tuple[str, int]]],
                    max_anc_depth: int = 2, alpha: float = 1.0) -> List[Tuple[int, int, float]]:
    """
    Build (child_idx, ancestor_idx, weight) pairs. weight = exp(-alpha*(d-1))
    """
    pos = []
    for child in nodes:
        child_idx = idx_of[child]
        for anc, d in ancestors_with_depth.get(child, [])[:max_anc_depth]:
            if anc in idx_of:
                anc_idx = idx_of[anc]
                w = math.exp(-alpha * (d - 1.0))
                pos.append((child_idx, anc_idx, w))
    return pos


# --------------------------
# Hard negative helpers
# --------------------------
def is_ancestor(a_idx: int, b_idx: int, nodes: List[str], parent_map: Dict[str, str]) -> bool:
    """
    Check if node at index a_idx is an ancestor of node at index b_idx by walking up from b.
    This is exact but potentially linear in tree depth; used for filtering a few candidates.
    """
    a_tid = nodes[a_idx]
    cur = nodes[b_idx]
    seen = set()
    while True:
        parent = parent_map.get(cur, "")
        if not parent or parent == cur or parent in seen:
            return False
        if parent == a_tid:
            return True
        seen.add(parent)
        cur = parent
    # fallback
    return False


def filter_hard_candidates_for_row(candidate_idxs: List[int], pos_idx: int, u_idx: int,
                                  nodes: List[str], parent_map: Dict[str, str], max_checks: int = 1000) -> List[int]:
    """
    Given candidate indices (from FAISS), filter out:
      - positive (the true ancestor v)
      - ancestors of u or descendants of u (i.e., u is ancestor of candidate)
    Return up to needed number later by caller.
    """
    out = []
    checks = 0
    for cand in candidate_idxs:
        if checks >= max_checks:
            break
        checks += 1
        if cand == pos_idx:
            continue
        # if cand is ancestor of u (bad), skip
        if is_ancestor(cand, u_idx, nodes, parent_map):
            continue
        # if u is ancestor of cand (i.e., candidate is descendant of u), skip
        if is_ancestor(u_idx, cand, nodes, parent_map):
            continue
        out.append(cand)
    return out


# --------------------------
# Evaluation utilities
# --------------------------
def sample_pairs_indices(n_nodes: int, n_pairs: int, seed: int = 0) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < n_pairs:
        i = rng.randrange(n_nodes)
        j = rng.randrange(n_nodes)
        if i == j:
            continue
        pairs.add((i, j))
    return list(pairs)


def compute_spearman_sampled(nodes: List[str], hyper_np: np.ndarray, G: nx.Graph, sample_pairs: int = 20000, seed: int = 0) -> float:
    """
    Sample pairs and compute Spearman between tree distance (shortest path) and hyperbolic distance.
    Note: this is sampled for speed.
    """
    N = len(nodes)
    if N == 0 or hyper_np.shape[0] != N:
        return 0.0
    pairs = sample_pairs_indices(N, min(sample_pairs, N * 10), seed=seed)
    tree_dists = []
    hyper_dists = []
    # pre-convert hyper to torch for distance compute using geoopt? We'll compute euclidean radial or use Poincare distance via geoopt manifold
    manifold = PoincareBall()
    emb = torch.tensor(hyper_np, dtype=torch.float32)
    for i, j in pairs:
        a = nodes[i]
        b = nodes[j]
        try:
            td = nx.shortest_path_length(G, source=a, target=b)
        except nx.NetworkXNoPath:
            td = max(10, G.number_of_nodes() // 10)
        tree_dists.append(td)
        with torch.no_grad():
            d = manifold.dist(emb[i].unsqueeze(0), emb[j].unsqueeze(0)).item()
        hyper_dists.append(d)
    if len(set(tree_dists)) <= 1:
        return 0.0
    rho, _ = spearmanr(tree_dists, hyper_dists)
    return float(rho)


def compute_parent_mrr_recall(nodes: List[str], tangent_np: np.ndarray, taxonomy: Dict[str, Dict[str, str]],
                              idx_map: Dict[str, int], topk_list=(1, 5, 10)) -> Tuple[float, Dict[int, float]]:
    """
    Use Euclidean tangent emb (tangent_np) to compute parent MRR and Recall@k.
    """
    if tangent_np.shape[0] == 0:
        return 0.0, {k: 0.0 for k in topk_list}
    knn = max(topk_list) + 1
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean', n_jobs=1).fit(tangent_np)
    distances, indices = nbrs.kneighbors(tangent_np)
    rr_total = 0.0
    hits = {k: 0 for k in topk_list}
    valid_count = 0
    for i, tid in enumerate(nodes):
        parent = taxonomy.get(tid, {}).get('parent', '')
        if not parent:
            continue
        pj = idx_map.get(parent)
        if pj is None:
            continue
        valid_count += 1
        neighs = [int(x) for x in indices[i] if int(x) != i]
        if pj in neighs:
            rankpos = neighs.index(pj) + 1
            rr_total += 1.0 / rankpos
            for k in topk_list:
                if pj in neighs[:k]:
                    hits[k] += 1
    if valid_count == 0:
        return 0.0, {k: 0.0 for k in topk_list}
    mrr = rr_total / valid_count
    recall_at_k = {k: hits[k] / valid_count for k in topk_list}
    return float(mrr), recall_at_k


# --------------------------
# Save functions
# --------------------------
def save_embeddings_pt(out_path: str, taxids: List[str], hyper_np: np.ndarray, tangent_np: np.ndarray, meta: Dict = None):
    """
    Save as torch .pt dictionary for fast reload.
    """
    data = {
        "taxids": taxids,
        "hyper_np": hyper_np.astype(np.float32),
        "tangent_np": tangent_np.astype(np.float32),
        "meta": meta or {}
    }
    torch.save(data, out_path)
    print(f"WROTE PT {out_path} rows={len(taxids)}")


def save_embeddings_parquet(outfile: str, taxids: List[str], hyper_np: np.ndarray, tangent_np: np.ndarray):
    """
    Save taxid (string), hyperbolic_emb (list<float32}), tangent_emb (list<float32}) to parquet.
    """
    taxid_arr = pa.array(taxids, type=pa.string())
    hyper_list = [row.astype(float).tolist() for row in hyper_np]
    tang_list = [row.astype(float).tolist() for row in tangent_np]
    hyper_arr = pa.array(hyper_list, type=pa.list_(pa.float32()))
    tang_arr = pa.array(tang_list, type=pa.list_(pa.float32()))
    table = pa.Table.from_arrays([taxid_arr, hyper_arr, tang_arr], names=["taxid", "hyperbolic_emb", "tangent_emb"])
    pq.write_table(table, outfile, compression="snappy")
    print(f"WROTE PARQUET {outfile} rows={len(taxids)}")


# --------------------------
# Training core
# --------------------------
def train(
    taxonomy_file: str,
    out_pt: str,
    parquet_out: str = None,
    dim: int = 64,
    epochs: int = 300,
    lr: float = 1e-2,
    neg: int = 60,
    batch_pos: int = 512,
    device: str = "cpu",
    max_anc_depth: int = 2,
    anc_alpha: float = 1.0,
    hard_every: int = 5,
    topk_hard: int = 100,
    use_faiss: bool = HAS_FAISS,
    sample_eval_pairs: int = 20000,
    seed: int = 42,
    save_every: int = 50,
    margin: float = 1.0
):
    set_seed(seed)
    dev = safe_device(device)

    # load taxonomy
    tax = read_taxonomy_table(taxonomy_file)
    if len(tax) == 0:
        raise RuntimeError("Empty taxonomy input.")
    # nodes: keep insertion order for reproducibility (python3.7+ preserves dict order)
    nodes = list(tax.keys())
    N = len(nodes)
    idx_of = {tid: i for i, tid in enumerate(nodes)}
    parent_map = {tid: rec.get("parent", "") for tid, rec in tax.items()}

    print(f"Loaded taxonomy nodes: {N}")

    # Build graph & ancestors
    G = build_graph_from_taxonomy(tax)
    ancestors_full = compute_ancestors_full(tax)
    ancestors_with_depth = compute_ancestors_with_depth(tax, max_depth=max_anc_depth)
    print(f"Built graph (directed). edges: {G.number_of_edges()} ancestors_full_sample={next(iter(ancestors_full.items())) if ancestors_full else None}")

    # Build positive pairs
    pos_pairs = build_pos_pairs(nodes, idx_of, ancestors_with_depth, max_anc_depth, anc_alpha)
    print(f"Number of positive pairs (including ancestors up to depth {max_anc_depth}): {len(pos_pairs)}")
    if len(pos_pairs) == 0:
        raise RuntimeError("No positive pairs generated; check taxonomy / parent links.")

    # Setup manifold & embeddings
    manifold = PoincareBall(c=1.0)
    # initialize small random tangent vectors then expmap0
    init_tangent = (torch.randn(N, dim, device=dev) * 1e-3)
    init_on_manifold = manifold.expmap0(init_tangent)
    emb = ManifoldParameter(init_on_manifold, manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([emb], lr=lr)

    # neg split: hard fraction if FAISS enabled
    neg_per_pos = max(1, int(neg))
    nhard_frac = 0.6 if use_faiss else 0.0
    nhard = int(neg_per_pos * nhard_frac)
    nrand = neg_per_pos - nhard
    print(f"neg_per_pos={neg_per_pos} => hard={nhard}, random={nrand} (faiss_enabled={use_faiss})")

    faiss_index = None
    tangent_cpu = None  # cached numpy for FAISS

    best = {"epoch": -1, "spearman": -1.0}
    best_hyper = None
    best_tangent = None

    # Training loop
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        # Build FAISS index on tangent space periodically
        if use_faiss and ((epoch % hard_every) == 1):
            with torch.no_grad():
                tangent_current = manifold.logmap0(emb.detach()).cpu().numpy().astype('float32')
            tangent_cpu = tangent_current
            d = tangent_cpu.shape[1]
            faiss_index = faiss.IndexFlatL2(d)
            faiss_index.add(tangent_cpu)
            print(f"[epoch {epoch}] Built FAISS index dim={d} n={tangent_cpu.shape[0]}")

        # shuffle positives for epoch
        random.shuffle(pos_pairs)
        total_loss_epoch = 0.0
        n_batches = 0

        for start in range(0, len(pos_pairs), batch_pos):
            batch = pos_pairs[start:start + batch_pos]
            B = len(batch)
            if B == 0:
                continue
            child_idx = torch.tensor([p[0] for p in batch], dtype=torch.long, device=dev)
            anc_idx = torch.tensor([p[1] for p in batch], dtype=torch.long, device=dev)
            weights = torch.tensor([p[2] for p in batch], dtype=torch.float32, device=dev)

            u = emb[child_idx]   # shape (B, dim)
            v = emb[anc_idx]     # shape (B, dim)

            # positive distance (smaller is better)
            d_pos = manifold.dist(u, v)  # (B,)

            # prepare random negatives indices (B, nrand)
            if nrand > 0:
                rand_idx = torch.randint(0, N, (B, nrand), device=dev)
            else:
                rand_idx = torch.empty((B, 0), dtype=torch.long, device=dev)

            # prepare hard negatives indices (B, nhard) by FAISS search + filtering
            hard_idx = torch.empty((B, 0), dtype=torch.long, device=dev)
            if nhard > 0 and faiss_index is not None and tangent_cpu is not None:
                # compute u tangent (cpu)
                with torch.no_grad():
                    u_tangent_cpu = manifold.logmap0(u.detach()).cpu().numpy().astype('float32')  # (B, dim)
                D, I = faiss_index.search(u_tangent_cpu, topk_hard)  # I shape (B, topk_hard)
                hard_idxs_list = []
                for i_row in range(B):
                    candidates = [int(x) for x in I[i_row]]
                    # filter out positives/ancestors/descendants
                    filtered = filter_hard_candidates_for_row(candidates, anc_idx[i_row].item(), child_idx[i_row].item(),
                                                              nodes, parent_map)
                    # take up to nhard
                    chosen = filtered[:nhard]
                    # if not enough, pad with random indices
                    if len(chosen) < nhard:
                        # sample random indices not equal to positive (and not ancestors)
                        extra = []
                        tries = 0
                        while len(extra) < (nhard - len(chosen)) and tries < nhard * 10:
                            tries += 1
                            r = random.randrange(N)
                            if r == anc_idx[i_row].item():
                                continue
                            if r in chosen:
                                continue
                            if is_ancestor(r, child_idx[i_row].item(), nodes, parent_map) or is_ancestor(child_idx[i_row].item(), r, nodes, parent_map):
                                continue
                            extra.append(r)
                        chosen += extra
                    # if still short, pad with -1 (we'll mask later)
                    while len(chosen) < nhard:
                        chosen.append(-1)
                    hard_idxs_list.append(chosen)
                hard_idx = torch.tensor(hard_idxs_list, dtype=torch.long, device=dev)  # (B, nhard)
            else:
                # no hard negs
                hard_idx = torch.empty((B, 0), dtype=torch.long, device=dev)

            # combine negatives: shape (B, nneg_total)
            if hard_idx.numel() == 0 and rand_idx.numel() == 0:
                # fallback: sample one random negative per pos to avoid degenerate case
                rand_idx = torch.randint(0, N, (B, 1), device=dev)
            neg_idx = rand_idx if hard_idx.numel() == 0 else torch.cat([rand_idx, hard_idx], dim=1) if rand_idx.numel() > 0 else hard_idx

            # build negative embeddings
            # neg_idx may contain -1 placeholders; mask them out when computing loss
            if neg_idx.numel() == 0:
                # safety, shouldn't happen
                continue
            neg_idx_clamped = neg_idx.clamp(min=0)
            neg_emb = emb[neg_idx_clamped]  # shape (B, nneg, dim)
            # compute distances u vs neg_emb -> (B, nneg)
            u_exp = u.unsqueeze(1).expand_as(neg_emb)
            d_neg = manifold.dist(u_exp.reshape(-1, dim), neg_emb.reshape(-1, dim)).reshape(u_exp.shape[0], u_exp.shape[1])

            # mask invalid positions (where neg_idx == -1) by setting large distance so loss is small
            if (neg_idx == -1).any():
                mask = (neg_idx == -1).to(device=dev)
                # set corresponding d_neg entries to a large positive value so margin loss won't push them
                d_neg = d_neg.masked_fill(mask, 1e6)

            # margin ranking style: want d_pos + margin < d_neg -> minimize relu(d_pos + margin - d_neg)
            # compute pairwise per negative and average
            d_pos_expand = d_pos.unsqueeze(1).expand_as(d_neg)  # (B, nneg)
            per_pair_loss = F.relu(d_pos_expand + margin - d_neg)  # (B, nneg)
            # weight per positive pair
            per_pair_loss = per_pair_loss.mean(dim=1)  # (B,)
            weighted_loss = (per_pair_loss * weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss_epoch += float(weighted_loss.item())
            n_batches += 1

        avg_loss = total_loss_epoch / max(1, n_batches)
        epoch_time = time.time() - t_epoch_start

        # periodic evaluation
        # if epoch % save_every == 0 or epoch == 1 or epoch == epochs:
        #     with torch.no_grad():
        #         hyper_cpu = emb.detach().cpu().numpy()
        #         tangent_cpu_eval = manifold.logmap0(emb.detach()).cpu().numpy()
        #     spearman = compute_spearman_sampled(nodes, hyper_cpu, G.to_undirected(), sample_pairs=sample_eval_pairs, seed=seed + epoch)
        #     mrr, recall_at_k = compute_parent_mrr_recall(nodes, tangent_cpu_eval, tax, idx_of, topk_list=(1, 5, 10))
        #     print(f"[Epoch {epoch}] loss={avg_loss:.6g} time={epoch_time:.1f}s spearman={spearman:.4f} MRR={mrr:.4f} R@1/5/10={[recall_at_k[k] for k in (1,5,10)]}")
        #     # update best
        #     if spearman > best["spearman"]:
        #         best = {"epoch": epoch, "spearman": spearman}
        #         # Save best snapshot
        #         with torch.no_grad():
        #             hyper_np_best = emb.detach().cpu().numpy()
        #             tangent_np_best = manifold.logmap0(emb.detach()).cpu().numpy()
        #         save_embeddings_pt(out_pt, nodes, hyper_np_best, tangent_np_best, meta={"epoch": epoch, "spearman": spearman})
        #         if parquet_out:
        #             save_embeddings_parquet(parquet_out, nodes, hyper_np_best, tangent_np_best)
        # else:
        #     print(f"[Epoch {epoch}] loss={avg_loss:.6g} time={epoch_time:.1f}s")
        import shutil


        if epoch % save_every == 0 or epoch == 1 or epoch == epochs:
            with torch.no_grad():
                hyper_cpu = emb.detach().cpu().numpy()
                tangent_cpu_eval = manifold.logmap0(emb.detach()).cpu().numpy()
            spearman = compute_spearman_sampled(nodes, hyper_cpu, G.to_undirected(),
                                                sample_pairs=sample_eval_pairs, seed=seed + epoch)
            mrr, recall_at_k = compute_parent_mrr_recall(nodes, tangent_cpu_eval, tax, idx_of, topk_list=(1, 5, 10))
            print(f"[Epoch {epoch}] loss={avg_loss:.6g} time={epoch_time:.1f}s "
                f"spearman={spearman:.4f} MRR={mrr:.4f} R@1/5/10={[recall_at_k[k] for k in (1,5,10)]}")

            if spearman > best["spearman"]:
                best = {"epoch": epoch, "spearman": spearman}
                best_hyper = hyper_cpu.copy()
                best_tangent = tangent_cpu_eval.copy()



    # after training loop
    if best_hyper is not None and best_tangent is not None:
        save_embeddings_pt(out_pt, nodes, best_hyper, best_tangent, meta=best)
        if parquet_out:
            save_embeddings_parquet(parquet_out, nodes, best_hyper, best_tangent)
    print("Training finished. Best:", best)


    print("Training finished. Best:", best)
    return best


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Poincare embeddings for taxonomy (improved).")
    p.add_argument("--in", dest="infile", required=True, help="input taxonomy tsv")
    p.add_argument("--out", dest="outpt", required=True, help="output .pt file to save embeddings")
    p.add_argument("--parquet_out", dest="parquet_out", default=None, help="optional parquet output path")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--neg", type=int, default=60)
    p.add_argument("--batch", dest="batch_pos", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_anc_depth", type=int, default=2)
    p.add_argument("--anc_alpha", type=float, default=1.0)
    p.add_argument("--hard_every", type=int, default=5)
    p.add_argument("--topk_hard", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_eval_pairs", type=int, default=20000)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--margin", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    best = train(
        taxonomy_file=args.infile,
        out_pt=args.outpt,
        parquet_out=args.parquet_out,
        dim=args.dim,
        epochs=args.epochs,
        lr=args.lr,
        neg=args.neg,
        batch_pos=args.batch_pos,
        device=args.device,
        max_anc_depth=args.max_anc_depth,
        anc_alpha=args.anc_alpha,
        hard_every=args.hard_every,
        topk_hard=args.topk_hard,
        use_faiss=HAS_FAISS,
        sample_eval_pairs=args.sample_eval_pairs,
        seed=args.seed,
        save_every=args.save_every,
        margin=args.margin
    )



