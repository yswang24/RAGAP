#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-batch HGT link-pred training using LinkNeighborLoader (PyG).

Features:
 - Robust torch.load (weights_only=False)
 - Uses LinkNeighborLoader for link-pred mini-batch training
 - Full-graph eval moved to eval_device (default cpu) to avoid GPU OOM
 - Negative sampling per batch (uniform)
 - Save best model by val AUC, log metrics (AUC/MRR/Hits@1/5/10)
 - Supports data saved as (HeteroData, split_edge) or HeteroData with splits inside.

Usage example:
python train_hgt_phage_host_full.py \
  --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt \
  --device cuda \
  --eval_device cuda \
  --epochs 1 \
  --hidden_dim 128 \
  --out_dim 128 \
  --n_layers 2 \
  --n_heads 1 \
  --num_neighbors 15 10 \
  --batch_size 256 \
  --neg_ratio 2 \
  --eval_neg_ratio 1 \
  --save_path best_hgt_nb.pt\
  --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
  --dropout 0.2\
  --log_every 5








  python train_hgt_phage_host_full.py \
  --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt \
  --device cuda \
  --eval_device cuda \
  --epochs 2\
  --hidden_dim 256 \
  --out_dim 256 \
  --n_layers 2 \
  --n_heads 4 \
  --num_neighbors 15 10 \
  --batch_size 512 \
  --neg_ratio 20 \
  --save_path best_hgt_nb2.pt
"""

import argparse
import time
import math
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.loader import LinkNeighborLoader

# Try to allowlist BaseStorage for torch.load safety (PyTorch >=2.6)
try:
    from torch_geometric.data.storage import BaseStorage
    import torch.serialization as _tser
    _tser.add_safe_globals([BaseStorage])
except Exception:
    pass

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_torch_load(path: str):
    """
    Loads a .pt file. Accepts either:
      - torch.save((data, split_edge), path)
      - torch.save(data, path) where data is HeteroData
      - torch.save({'data':data, 'split_edge': split_edge}, path)
    Returns (data, split_edge_or_None)
    """
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], HeteroData):
        return obj[0], obj[1]
    if isinstance(obj, HeteroData):
        return obj, None
    if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], HeteroData):
        return obj['data'], obj.get('split_edge', None)
    raise RuntimeError("Unsupported .pt content. Please save torch.save((data, split_edge), path) or torch.save(data, path).")


def find_phage_host_splits(data: HeteroData, ext_splits):
    """
    Return three pairs: (train_src, train_dst), (val_src, val_dst), (test_src, test_dst)
    as 1D cpu LongTensors.
    """
    # if provided externally (from saved tuple)
    if ext_splits is not None:
        def as_pair(x):
            if isinstance(x, torch.Tensor) and x.dim() == 2 and x.size(0) == 2:
                return x[0].cpu(), x[1].cpu()
            raise RuntimeError("External split format invalid")
        try:
            return as_pair(ext_splits['train']['edge']), as_pair(ext_splits['val']['edge']), as_pair(ext_splits['test']['edge'])
        except Exception:
            pass

    # find phage->host relation name
    rel = None
    for r in data.edge_types:
        if r[0] == 'phage' and r[2] == 'host':
            rel = r
            break
    if rel is None:
        raise RuntimeError("No ('phage',*, 'host') relation in data.edge_types")

    rec = data[rel]
    # common attribute patterns
    patterns = [
        ('edge_index_train', 'edge_index_val', 'edge_index_test'),
        ('train_pos_edge_index', 'val_pos_edge_index', 'test_pos_edge_index'),
        ('train_edge_index', 'val_edge_index', 'test_edge_index'),
        ('train', 'val', 'test'),
    ]
    for a,b,c in patterns:
        if hasattr(rec, a) and hasattr(rec, b) and hasattr(rec, c):
            A = getattr(rec, a); B = getattr(rec, b); C = getattr(rec, c)
            if isinstance(A, torch.Tensor) and A.dim()==2 and A.size(0)==2:
                return (A[0].cpu(), A[1].cpu()), (B[0].cpu(), B[1].cpu()), (C[0].cpu(), C[1].cpu())

    # fallback: maybe top-level attribute data.split_edge
    if hasattr(data, 'split_edge'):
        se = getattr(data, 'split_edge')
        if isinstance(se, dict) and 'train' in se and 'val' in se and 'test' in se:
            def pair(e):
                if isinstance(e, torch.Tensor) and e.dim()==2 and e.size(0)==2:
                    return e[0].cpu(), e[1].cpu()
                raise RuntimeError("split_edge format invalid")
            return pair(se['train']['edge']), pair(se['val']['edge']), pair(se['test']['edge'])

    raise RuntimeError("Cannot find phage-host splits inside data; please save splits or provide as ext_splits.")


# -------------------------
# Model
# -------------------------
class HGTMiniModel(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=128, out_dim=128, n_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        # per-node input projector
        self.input_proj = nn.ModuleDict()
        for n in self.node_types:
            d = in_dims.get(n, None)
            if d is None:
                raise RuntimeError(f"Missing in_dim for node type {n}")
            self.input_proj[n] = nn.Linear(d, hidden_dim)

        # HGT layers
        self.hgt_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hgt_layers.append(HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=n_heads))

        self.dropout = nn.Dropout(dropout)
        # final node projection
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
        # edge decoder
        self.edge_mlp = nn.Sequential(nn.Linear(2*out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))

    def forward(self, x_dict, edge_index_dict):
        # x_dict: node_type -> [N_ntype, in_dim] (as in the sampled mini-batch)
        h = {}
        for n, x in x_dict.items():
            h[n] = F.relu(self.input_proj[n](x))

        for conv in self.hgt_layers:
            h = conv(h, edge_index_dict)
            for k in list(h.keys()):
                h[k] = F.relu(self.dropout(h[k]))

        #out = {k: self.final_proj[k](v) for k, v in h.items()}
        out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}#归一化

        return out

    def decode(self, z_dict, edge_label_index, etype=('phage','infects','host')):
        # edge_label_index: tensor (2, E) or tuple(src,dst)
        if isinstance(edge_label_index, torch.Tensor) and edge_label_index.dim()==2 and edge_label_index.size(0)==2:
            src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        elif isinstance(edge_label_index, (tuple, list)) and len(edge_label_index)==2:
            src_idx, dst_idx = edge_label_index
        else:
            raise RuntimeError("edge_label_index must be (2,E) or tuple(src,dst)")
        src_type, _, dst_type = etype
        src_z = z_dict[src_type][src_idx]
        dst_z = z_dict[dst_type][dst_idx]
        e = torch.cat([src_z, dst_z], dim=-1)
        return self.edge_mlp(e).view(-1)


# -------------------------
# Metrics (full-graph eval)
# -------------------------
@torch.no_grad()
def compute_metrics_fullgraph(model, data, train_pairs, val_pairs, test_pairs,
                              eval_device='cpu', eval_neg_ratio=1, k_list=(1,5,10),
                              host_id2taxid=None, taxid2species=None):
    """
    Compute full-graph metrics on eval_device.
    Returns (train_metrics, val_metrics, test_metrics) where each is (auc, mrr, hits_at_dict).
    - AUC: computed using `eval_neg_ratio` negatives per positive (random negs, chunked decode).
    - MRR & Hits@k: species-level ranking computed by ranking against all hosts (full candidate set).
    """
    import numpy as np
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    # determine phage->host relation key in data
    relation = None
    for et in data.edge_types:
        if et[0] == 'phage' and et[2] == 'host':
            relation = et
            break
    if relation is None:
        raise RuntimeError("phage->host relation not found in data.edge_types")

    # save original device and move model + data features to eval_device
    orig_device = next(model.parameters()).device
    moved_model = False
    try:
        if orig_device != torch.device(eval_device):
            model.to(eval_device)
            moved_model = True

        # move feature tensors and edge_index_dict (non-destructive: data.to may modify in-place;
        # to be safe, we access tensors and move to eval_device for the forward)
        data_eval_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
        edge_index_dict_eval = {}
        for et in data.edge_types:
            if hasattr(data[et], 'edge_index') and data[et].edge_index is not None:
                edge_index_dict_eval[et] = data[et].edge_index.to(eval_device)

        model.eval()
        # compute full graph embeddings once on eval_device
        out = model(data_eval_x, edge_index_dict_eval)
        ph_emb = out['phage']    # shape (n_phage, D)
        host_emb = out['host']   # shape (n_host, D)

        n_hosts = host_emb.size(0)

        def hostid2species(hid: int) -> str:
            if host_id2taxid is None or taxid2species is None:
                print('没有成功找到species名称')
                return str(hid)
            taxid = int(host_id2taxid[hid])
            return taxid2species.get(taxid, f"unk_{taxid}")

        # helper: compute AUC with random negatives (chunked decode to avoid OOM)
        # ===== build all_pos_map (filtered negatives) =====
        from collections import defaultdict
        all_pos_map = defaultdict(set)
        # aggregate known positives from train/val/test (each is tuple (src_cpu, dst_cpu))
        for pair in (train_pairs, val_pairs, test_pairs):
            src_cpu, dst_cpu = pair
            # ensure CPU lists
            if src_cpu.numel() == 0:
                continue
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                all_pos_map[int(s)].add(int(d))
        # ==================================================

        def compute_auc_for_pairs(src_tensor_cpu, dst_tensor_cpu):
            """
            Full-host, filtered AUC:
            - For each positive pair (s,d) we compute scores vs ALL hosts:
                scores_all = sigmoid( ph_emb[s] · host_emb[:] )
            - We treat pos_score = scores_all[d], and negatives = scores_all[all_hosts \ other_known_positives]
            - We collect all pos_scores and neg_scores across all positives and compute roc_auc_score.
            """
            # inputs are CPU tensors of indices
            src_list = src_tensor_cpu.tolist() if src_tensor_cpu.numel() > 0 else []
            dst_list = dst_tensor_cpu.tolist() if dst_tensor_cpu.numel() > 0 else []
            npos = len(src_list)
            if npos == 0:
                return float('nan')

            pos_scores_list = []
            neg_scores_list = []

            # process each positive example one-by-one (keeps memory bounded)
            for s_cpu, d_cpu in zip(src_list, dst_list):
                # make sure indices are ints
                s_idx = int(s_cpu); d_idx = int(d_cpu)
                # compute scores vs all hosts on eval_device
                # ph_emb[s_idx] shape (D,) -> unsqueeze to (1,D) to broadcast
                q = ph_emb[s_idx:s_idx+1]            # (1, D) on eval_device
                scores_all = (q * host_emb).sum(dim=-1).squeeze(0)  # (n_hosts,) on eval_device
                scores_all = torch.sigmoid(scores_all)  # probabilities

                # positive score
                if not (0 <= d_idx < n_hosts):
                    # out-of-range true host (shouldn't happen after earlier validation) -> skip
                    continue
                pos_score = float(scores_all[d_idx].cpu().item())
                pos_scores_list.append(pos_score)

                # build mask for negatives: exclude other known positives of this phage (filtered)
                known = all_pos_map.get(s_idx, set())
                if len(known) == 0:
                    # all other hosts are negatives
                    neg_scores = scores_all.cpu().numpy()
                    # remove the current true from negatives (we keep it only as positive)
                    neg_scores = np.delete(neg_scores, d_idx)
                else:
                    # create boolean mask on device
                    mask = torch.ones(n_hosts, dtype=torch.bool, device=eval_device)
                    # exclude all other known positives except the current true d_idx
                    for oth in known:
                        if oth == d_idx:
                            continue
                        if 0 <= oth < n_hosts:
                            mask[oth] = False
                    # also exclude the current true (so neg_scores only contain negatives)
                    mask[d_idx] = False
                    # if mask has no True entries, skip negatives for this sample
                    if mask.any():
                        neg_scores = scores_all[mask].cpu().numpy()
                    else:
                        neg_scores = np.array([], dtype=float)

                # store negatives (could be empty for pathological cases)
                if neg_scores.size > 0:
                    neg_scores_list.append(neg_scores)
                # else: leave neg list empty for this pos (it will not contribute neg samples)

            # if no positives found (edge case)
            if len(pos_scores_list) == 0:
                return float('nan')

            # flatten negatives into one array
            if len(neg_scores_list) == 0:
                # no negatives available after filtering -- cannot compute AUC reliably
                return float('nan')

            all_pos_arr = np.asarray(pos_scores_list, dtype=float)
            all_neg_arr = np.concatenate(neg_scores_list, axis=0).astype(float)

            # Build labels and scores for roc_auc_score
            y_true = np.concatenate([np.ones_like(all_pos_arr), np.zeros_like(all_neg_arr)])
            y_score = np.concatenate([all_pos_arr, all_neg_arr])

            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = float('nan')
            return auc


        # helper: compute species-level MRR and Hits@k by ranking against ALL hosts
        def compute_rank_metrics(pairs_cpu):
            src_cpu, dst_cpu = pairs_cpu
            if src_cpu.numel() == 0:
                return float('nan'), 0.0, {k: 0.0 for k in k_list} #应该返回两个

            # group true destinations per phage (using CPU indices)
            ph2hosts = defaultdict(list)
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                ph2hosts[s].append(d)

            hits = {k: 0 for k in k_list}
            rr_sum = 0.0
            total_q = 0

            # compute scores for all hosts once per phage (vectorized inside loop)
            for ph_idx_cpu, true_ds in ph2hosts.items():
                total_q += 1
                q_idx = torch.tensor([ph_idx_cpu], dtype=torch.long, device=eval_device)
                # compute scores against all hosts: shape (n_host,)
                # do dot product batch-wise if memory tight
                # here we compute (ph_emb[q] * host_emb).sum(dim=-1)
                q_emb = ph_emb[q_idx]  # (1, D)
                scores = (q_emb * host_emb).sum(dim=-1)  # (n_host,)
                # get top-k hosts (k = max requested)
                K = max(k_list)
                if K >= n_hosts:
                    topk_idx = torch.argsort(scores, descending=True).cpu().numpy().tolist()
                else:
                    topk_idx = torch.topk(scores, K).indices.cpu().numpy().tolist()

                # species sets
                true_species = {hostid2species(h) for h in true_ds}
                topk_species = [hostid2species(h) for h in topk_idx]

                # MRR: find first position where species match
                rank = None
                for pos, sp in enumerate(topk_species, start=1):
                    if sp in true_species:
                        rank = pos
                        break
                if rank is None:
                    rank = K + 1  # not found within top-K window -> treat as rank > K
                rr_sum += 1.0 / rank

                # Hits@k
                for k in k_list:
                    if any(sp in true_species for sp in topk_species[:k]):
                        hits[k] += 1

            mrr = rr_sum / total_q if total_q > 0 else 0.0
            hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}
            return mrr, hits_at

        # compute metrics for each split
        # train AUC & species-metrics
        train_auc = compute_auc_for_pairs(train_pairs[0], train_pairs[1]) if train_pairs[0].numel() > 0 else float('nan')
        train_mrr, train_hits = compute_rank_metrics(train_pairs)

        val_auc = compute_auc_for_pairs(val_pairs[0], val_pairs[1]) if val_pairs[0].numel() > 0 else float('nan')
        val_mrr, val_hits = compute_rank_metrics(val_pairs)

        test_auc = compute_auc_for_pairs(test_pairs[0], test_pairs[1]) if test_pairs[0].numel() > 0 else float('nan')
        test_mrr, test_hits = compute_rank_metrics(test_pairs)

        # Return tuples matching previous format: (auc, mrr, hits_dict)
        train_metrics = (train_auc, train_mrr, train_hits)
        val_metrics = (val_auc, val_mrr, val_hits)
        test_metrics = (test_auc, test_mrr, test_hits)
        return train_metrics, val_metrics, test_metrics

    finally:
        # restore model device if moved
        if moved_model:
            model.to(orig_device)


# -------------------------
# Training entry
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_pt", required=True, help="Path to .pt file containing HeteroData or (data, split_edge)")
    p.add_argument("--taxid2species_tsv", default=None, help="Optional TSV mapping taxid -> species")
    p.add_argument("--device", default="cuda", help="training device")
    p.add_argument("--eval_device", default="cpu", help="eval device (use cpu to save GPU mem)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_neighbors", nargs='+', type=int, default=[15,10], help="neighbors per hop, e.g. --num_neighbors 15 10")
    p.add_argument("--batch_size", type=int, default=2048, help="positive edges per batch")
    p.add_argument("--neg_ratio", type=int, default=1, help="negatives per positive in training batch")
    p.add_argument("--eval_neg_ratio", type=int, default=1)
    p.add_argument("--save_path", default="best_hgt_nb.pt")
    p.add_argument("--log_every", type=int, default=1)
 
    return p.parse_args()
def main():
    import pandas as pd  # 新增导入（你原来也有）
    import random  # 新增导入，CPU 随机采样，减少 GPU 分配
    import os, csv, numpy as np, torch  # 新增导入：用于检查与保存
    import time
    import torch.nn as nn
    from torch_geometric.loader import LinkNeighborLoader

    # 其余已有导入请保持不变（例如 argparse, HGTMiniModel, compute_metrics_fullgraph 等）

    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    eval_device = torch.device(args.eval_device)

    print("Loading data:", args.data_pt)
    data, split_edge = safe_torch_load(args.data_pt)
    print("Data metadata:", data.metadata())

    # --------- 新增函数：检查并（可选）修复 split 越界 / 缺失 id 字段 ----------
    def inspect_and_fix_data(data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu, fix_enable=True, out_dir="debug_out"):
        import os, csv, numpy as np, torch
        os.makedirs(out_dir, exist_ok=True)
        node_counts = {}

        # 1) node counts
        print("== node counts ==")
        for ntype in data.node_types:
            if hasattr(data[ntype], "num_nodes"):
                try:
                    n_nodes = int(data[ntype].num_nodes)
                except Exception:
                    n_nodes = None
            elif 'x' in data[ntype]:
                try:
                    n_nodes = int(data[ntype].x.size(0))
                except Exception:
                    n_nodes = None
            else:
                n_nodes = None
            node_counts[ntype] = n_nodes
            print(f"{ntype}: num_nodes = {n_nodes}")

        # 2) edge_index_dict bounds
        print("\n== edge_index_dict bounds ==")
        bad_items = []
        for etype, eidx in data.edge_index_dict.items():
            if eidx is None:
                print(f"{etype}: edge_index is None")
                continue
            e_cpu = eidx.cpu()
            if e_cpu.numel() == 0:
                print(f"{etype}: empty")
                continue
            if e_cpu.dim() != 2 or e_cpu.size(0) != 2:
                print(f"WARNING {etype}: unexpected shape {tuple(e_cpu.shape)}")
            try:
                src_max = int(e_cpu[0].max()); src_min = int(e_cpu[0].min())
                dst_max = int(e_cpu[1].max()); dst_min = int(e_cpu[1].min())
            except Exception as e:
                print(f"Could not compute min/max for {etype}: {e}")
                continue
            src_type, _, dst_type = etype
            src_n = node_counts.get(src_type)
            dst_n = node_counts.get(dst_type)
            print(f"{etype}: src_min={src_min}, src_max={src_max}, src_n={src_n}; dst_min={dst_min}, dst_max={dst_max}, dst_n={dst_n}")
            if src_n is not None and (src_min < -src_n or src_max >= src_n):
                bad_items.append(("edge_index", etype, "src", src_min, src_max, src_n))
            if dst_n is not None and (dst_min < -dst_n or dst_max >= dst_n):
                bad_items.append(("edge_index", etype, "dst", dst_min, dst_max, dst_n))

        # 3) splits check helper
        def check_split(name, s_cpu, d_cpu, src_type='phage', dst_type='host'):
            s_arr = s_cpu.cpu().numpy()
            d_arr = d_cpu.cpu().numpy()
            smin = int(s_arr.min()); smax = int(s_arr.max())
            dmin = int(d_arr.min()); dmax = int(d_arr.max())
            src_n = node_counts.get(src_type)
            dst_n = node_counts.get(dst_type)
            print(f"{name}: {src_type} min/max = {smin}/{smax} (n={src_n}); {dst_type} min/max = {dmin}/{dmax} (n={dst_n})")
            bad = []
            if src_n is not None and (smin < -src_n or smax >= src_n):
                bad.append((name, src_type, smin, smax, src_n))
            if dst_n is not None and (dmin < -dst_n or dmax >= dst_n):
                bad.append((name, dst_type, dmin, dmax, dst_n))
            return bad, s_arr, d_arr

        print("\n== checking train/val/test splits ==")
        bad, train_s_arr, train_d_arr = check_split("train", train_src_cpu, train_dst_cpu)
        bad += check_split("val", val_src_cpu, val_dst_cpu)[0]
        bad += check_split("test", test_src_cpu, test_dst_cpu)[0]

        if bad:
            print("\n!!! Found out-of-bounds items:")
            for b in bad:
                print(b)
            with open(os.path.join(out_dir, "bad_items.txt"), "w", encoding="utf-8") as fo:
                for b in bad:
                    fo.write(str(b) + "\n")

        # 保存 invalid examples（若存在越界）
        def save_invalid_examples(name, s_arr, d_arr, src_n, dst_n, limit=200):
            invalid = []
            for i, (si, di) in enumerate(zip(s_arr.tolist(), d_arr.tolist())):
                if not ((-src_n <= si < src_n) and (-dst_n <= di < dst_n)):
                    invalid.append((i, int(si), int(di)))
                    if len(invalid) >= limit:
                        break
            if invalid:
                fn = os.path.join(out_dir, f"invalid_{name}_examples.tsv")
                with open(fn, "w", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(["idx","src_idx","dst_idx"])
                    writer.writerows(invalid)
                print(f"Saved {len(invalid)} invalid {name} examples to {fn}")

        if bad:
            phage_n = node_counts.get('phage')
            host_n = node_counts.get('host')
            if phage_n and host_n:
                save_invalid_examples("train", train_s_arr, train_d_arr, phage_n, host_n)
                val_s_arr = val_src_cpu.cpu().numpy(); val_d_arr = val_dst_cpu.cpu().numpy()
                test_s_arr = test_src_cpu.cpu().numpy(); test_d_arr = test_dst_cpu.cpu().numpy()
                save_invalid_examples("val", val_s_arr, val_d_arr, phage_n, host_n)
                save_invalid_examples("test", test_s_arr, test_d_arr, phage_n, host_n)

        # 自动清理越界 split（谨慎：会先备份）
        if fix_enable and bad:
            print("Auto-cleaning split edges (backing up originals to debug_out/)...")
            torch.save((train_src_cpu.clone(), train_dst_cpu.clone()), os.path.join(out_dir, "train_split_backup.pt"))
            torch.save((val_src_cpu.clone(), val_dst_cpu.clone()), os.path.join(out_dir, "val_split_backup.pt"))
            torch.save((test_src_cpu.clone(), test_dst_cpu.clone()), os.path.join(out_dir, "test_split_backup.pt"))

            def filter_pairs(s_cpu, d_cpu, src_n, dst_n):
                s_list=[]; d_list=[]
                for si, di in zip(s_cpu.cpu().tolist(), d_cpu.cpu().tolist()):
                    if (-src_n <= si < src_n) and (-dst_n <= di < dst_n):
                        s_list.append(int(si)); d_list.append(int(di))
                return torch.tensor(s_list, dtype=torch.long), torch.tensor(d_list, dtype=torch.long)
            phage_n = node_counts.get('phage'); host_n = node_counts.get('host')
            if phage_n is None or host_n is None:
                print("Cannot auto-fix because phage/host node counts unknown.")
            else:
                train_src_cpu, train_dst_cpu = filter_pairs(train_src_cpu, train_dst_cpu, phage_n, host_n)
                val_src_cpu, val_dst_cpu = filter_pairs(val_src_cpu, val_dst_cpu, phage_n, host_n)
                test_src_cpu, test_dst_cpu = filter_pairs(test_src_cpu, test_dst_cpu, phage_n, host_n)
                print("After filter train/val/test sizes:", train_src_cpu.size(0), val_src_cpu.size(0), test_src_cpu.size(0))
                torch.save((train_src_cpu, train_dst_cpu), os.path.join(out_dir, "train_split_fixed.pt"))
                torch.save((val_src_cpu, val_dst_cpu), os.path.join(out_dir, "val_split_fixed.pt"))
                torch.save((test_src_cpu, test_dst_cpu), os.path.join(out_dir, "test_split_fixed.pt"))

        # ====== 关键修改：为缺失的 id 添加 **数值型** 占位（避免字符串导致的 np.take -> torch.from_numpy 错误） ======
        for ntype in ['phage', 'host']:
            if not hasattr(data[ntype], 'id'):
                n = node_counts.get(ntype)
                if n is not None:
                    print(f"Adding numeric placeholder {ntype}.id = arange({n}) (dtype=int64)")
                    data[ntype].id = torch.arange(n, dtype=torch.long)
                else:
                    print(f"Cannot add {ntype}.id because node count unknown")

        return data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu

    # ----------------- end of inspect_and_fix_data -----------------

    # get splits
    train_pair, val_pair, test_pair = find_phage_host_splits(data, split_edge)
    train_src_cpu, train_dst_cpu = train_pair
    val_src_cpu, val_dst_cpu = val_pair
    test_src_cpu, test_dst_cpu = test_pair
    print("Train/Val/Test counts:", train_src_cpu.size(0), val_src_cpu.size(0), test_src_cpu.size(0))

    # 在创建 train_loader 之前进行检查与可选自动修复（严格保持后续逻辑不变）
    data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu = inspect_and_fix_data(
        data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu, fix_enable=True, out_dir="debug_out"
    )

    # ensure node features present
    in_dims = {}
    for n in data.node_types:
        if 'x' not in data[n]:
            raise RuntimeError(f"Node {n} missing .x features")
        in_dims[n] = data[n].x.size(1)
        print(f"node {n} in_dim = {in_dims[n]}")

    # ---------- DIAGNOSTIC & SAFE MODEL INIT ----------
    print("=== Diagnostic before model instantiation ===")
    for n in data.node_types:
        x = data[n].x
        try:
            num_nodes = x.size(0)
            dim = x.size(1)
            nz = (x != 0).sum().item() if x.numel() > 0 else 0
            nz_ratio = nz / float(x.numel()) if x.numel() > 0 else 0.0
            print(f"node={n:10s}  num_nodes={num_nodes:6d}  in_dim={dim:6d}  nonzero_ratio={nz_ratio:.6f}  dtype={x.dtype}")
        except Exception as e:
            print(f"  Failed to inspect data[{n}].x: {e}")

    total_first_layer_params = sum(in_dims[n] * args.hidden_dim for n in in_dims)
    bytes_est = total_first_layer_params * 4  # float32 -> 4 bytes
    print(f"Rough params for first mapping (sum in_dim * hidden_dim) = {total_first_layer_params:,} floats ≈ {bytes_est/1024**2:.1f} MB")

    print("Instantiating model ON CPU for diagnostics (won't .to(device) yet)...")
    model = HGTMiniModel(metadata=data.metadata(), in_dims=in_dims,
                        hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total params: {total_params:,}  floats  (~{total_params*4/1024**2:.1f} MB for float32)")
    print("Large parameter shapes (showing >1e6 elements):")
    for name, p in model.named_parameters():
        if p.numel() > 1_000_000:
            print(f"  {name:60s} shape={tuple(p.shape)} numel={p.numel():,}")

    try:
        model = model.to(device)
    except RuntimeError as e:
        print("ERROR: moving model to device failed with:", e)
        print("Suggestion: reduce args.hidden_dim / args.n_heads / args.n_layers, or reduce feature dims.")
        raise
    # ---------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    # Build LinkNeighborLoader for training
    train_edge_index = torch.stack([train_src_cpu, train_dst_cpu], dim=0)
    relation = None
    for r in data.edge_types:
        if r[0]=='phage' and r[2]=='host':
            relation = r
            break
    if relation is None:
        raise RuntimeError("phage->host relation not found")

    train_loader = LinkNeighborLoader(
        data,
        num_neighbors={etype: list(map(int, args.num_neighbors)) for etype in data.edge_types},
        edge_label_index=(relation, train_edge_index),
        edge_label=torch.ones(train_edge_index.size(1), dtype=torch.float),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0
    )
    print("Train loader created. batches:", len(train_loader))

    # ------------------------------- 后续训练、评估逻辑保持不变 -------------------------------
    taxid2species = None
    host_id2taxid = None
    if args.taxid2species_tsv is not None:
        taxmap = pd.read_csv(args.taxid2species_tsv, sep="\t")
        taxid2species = dict(zip(taxmap["taxid"], taxmap["species"]))
        if not hasattr(data['host'], 'taxid'):
            raise RuntimeError("data['host'] must have .taxid attribute for species-level eval")
        host_id2taxid = data['host'].taxid.cpu().numpy()  # host idx -> taxid
    # -------------------------------

    # 全局正边集合（保留 val/test 正边，按你的原实现）
    all_pos_edges = set(zip(train_src_cpu.tolist(), train_dst_cpu.tolist()))
    all_pos_edges |= set(zip(val_src_cpu.tolist(), val_dst_cpu.tolist()))
    all_pos_edges |= set(zip(test_src_cpu.tolist(), test_dst_cpu.tolist()))

    best_val_auc = -1.0
    best_ckpt = None

    # training loop (mini-batch)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model({nt: batch[nt].x for nt in batch.node_types},
                        batch.edge_index_dict)

            edge_label_index = batch[relation].edge_label_index
            if edge_label_index is None or edge_label_index.numel() == 0:
                continue

            pos_scores = model.decode(out, edge_label_index, etype=relation)
            labels_pos = torch.ones_like(pos_scores, device=device)

            # ----------- 负采样（保持你原逻辑） -----------
            # 映射 batch 内局部索引 -> 全局 id
            if not hasattr(batch['phage'], 'n_id') or not hasattr(batch['host'], 'n_id'):
                # 这种情况不常见，但做保护性判断
                raise RuntimeError("Expected batch['phage'].n_id and batch['host'].n_id present in batch")

            phage_nid = batch['phage'].n_id.cpu().tolist()
            host_nid = batch['host'].n_id.cpu().tolist()
            host_n = len(host_nid)

            neg_src_list, neg_dst_list = [], []
            for src_local in edge_label_index[0].cpu().tolist():
                src_global = phage_nid[src_local]
                count = 0
                # 防止死循环：限制 attempts
                attempts = 0
                while count < args.neg_ratio and attempts < args.neg_ratio * 10:
                    dst_local = random.randrange(host_n)
                    dst_global = host_nid[dst_local]
                    if (src_global, dst_global) not in all_pos_edges:
                        neg_src_list.append(src_local)
                        neg_dst_list.append(dst_local)
                        count += 1
                    attempts += 1
                # 如果在限定 attempts 内找不到足够负样本，则允许少于 neg_ratio 的负样本继续
            if len(neg_src_list) == 0:
                continue

            # 分块 decode 避免显存暴涨
            chunk_size = 2048
            neg_loss_total = 0.0
            neg_src_tensor = torch.tensor(neg_src_list, dtype=torch.long, device=device)
            neg_dst_tensor = torch.tensor(neg_dst_list, dtype=torch.long, device=device)
            n_chunks = (len(neg_src_tensor) + chunk_size - 1) // chunk_size

            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(neg_src_tensor))
                neg_chunk = (neg_src_tensor[start:end], neg_dst_tensor[start:end])
                neg_scores = model.decode(out, neg_chunk, etype=relation)
                labels_neg = torch.zeros_like(neg_scores, device=device)
                neg_loss_total += loss_fn(neg_scores, labels_neg)

            # 总 loss
            loss = loss_fn(pos_scores, labels_pos) + neg_loss_total
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        t1 = time.time()
        avg_loss = epoch_loss / max(1, n_batches)

        # 评估（保持你原有的 compute_metrics_fullgraph 调用）
        if epoch % args.log_every == 0 or epoch == args.epochs:
            try:
                train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
                    model,
                    data,
                    (train_src_cpu, train_dst_cpu),
                    (val_src_cpu, val_dst_cpu),
                    (test_src_cpu, test_dst_cpu),
                    eval_device=args.eval_device,
                    eval_neg_ratio=args.eval_neg_ratio,
                    host_id2taxid=host_id2taxid,
                    taxid2species=taxid2species
                )

                # ------------------- 修改点：使用 full-graph embedding 在 eval_device(GPU) 上计算 test edge 分数 -------------------
                # 说明：
                # - 我们把模型短暂移动到 eval_device（如果需要），在全图（data）上计算所有节点的表示，
                #   然后在 eval_device 上对 test edges 做 decode（候选 host 全体在 GPU）
                # - 之后再把模型移回原训练 device，继续训练（保证训练流程不变）
                model.eval()
                with torch.no_grad():
                    # 保存原设备以便恢复
                    orig_dev = next(model.parameters()).device
                    moved = False
                    try:
                        if orig_dev != eval_device:
                            model.to(eval_device)
                            moved = True

                        # prepare full features on eval_device
                        full_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
                        # prepare full edge_index_dict on eval_device (use existing relation keys)
                        full_edge_index_dict = {}
                        for et in data.edge_types:
                            if hasattr(data[et], 'edge_index'):
                                ei = data[et].edge_index
                                if ei is None:
                                    continue
                                full_edge_index_dict[et] = ei.to(eval_device)

                        # compute full-graph embeddings
                        out_full = model(full_x, full_edge_index_dict)

                        # build test edge_index (global indices) on eval_device
                        edge_index_full = torch.stack([test_src_cpu.to(eval_device), test_dst_cpu.to(eval_device)], dim=0)
                        ###原来
                        # decode using full-graph embeddings (on eval_device) -> logits -> sigmoid to get probabilities
                        scores_tensor = torch.sigmoid(model.decode(out_full, edge_index_full, etype=relation))
                        scores = scores_tensor.cpu().numpy()

                        # # 假设 out_full 是 full-graph 计算出来的节点嵌入
                        # # 获取 phage 和 host 的 embedding
                        # phage_emb = out_full['phage']  # [num_phage, hidden_dim]
                        # host_emb  = out_full['host']   # [num_host, hidden_dim]

                        # # 获取所有 phage 和 host 节点索引
                        # num_phage = phage_emb.shape[0]
                        # num_host  = host_emb.shape[0]
                        # phage_idx = torch.arange(num_phage, device=eval_device)
                        # host_idx  = torch.arange(num_host, device=eval_device)

                        # # 构造所有 phage × host 对
                        # phage_expand = phage_idx.unsqueeze(1).repeat(1, num_host).reshape(-1)
                        # host_expand  = host_idx.repeat(num_phage)

                        # # 用模型 decode 计算得分（注意这里不再传 edge_index_full，而是自己构造）
                        # scores_tensor = torch.sigmoid(model.decode(
                        #     (phage_emb[phage_expand], host_emb[host_expand]),
                        #     etype=('phage', 'interacts', 'host')
                        # ))

                        # # 转成矩阵 [num_phage, num_host]
                        # scores_matrix = scores_tensor.view(num_phage, num_host)


                    finally:
                        # move model back to original training device if we moved it
                        if moved:
                            model.to(orig_dev)

                    #now write predictions using global ids -> readable ids (data[<ntype>].id assumed present)
                    # phage_ids = []
                    # host_ids = []
                    # print(data['phage'].keys())
                    # print(data['host'].keys())

                    # for nid in test_src_cpu.tolist():
                    #     # data['phage'].id may be tensor of strings or longs; ensure convert to python str/int
                    #     pid = data['phage'].id[nid]
                    #     if isinstance(pid, torch.Tensor):
                    #         pid = pid.item()
                    #     phage_ids.append(str(pid))
                    
                    # for nid in test_dst_cpu.tolist():
                    #     hid = data['host'].id[nid]
                    #     if isinstance(hid, torch.Tensor):
                    #         hid = hid.item()
                    #     host_ids.append(str(hid))

                    # if host_id2taxid is not None and taxid2species is not None:
                    #     host_species = [taxid2species[host_id2taxid[hid]] for hid in test_dst_cpu.tolist()]
                    # else:
                    #     host_species = ["NA"] * len(host_ids)

                    # # save TSV
                    # with open("phage_prediction_results.tsv", "w", newline="", encoding="utf-8") as f:
                    #     writer = csv.writer(f, delimiter="\t")
                    #     writer.writerow(["phage_id", "host_id", "host_species", "score"])
                    #     for pid, hid, hs, s in zip(phage_ids, host_ids, host_species, scores):
                    #         writer.writerow([pid, hid, hs, float(s)])
                    # print("Phage-host prediction results saved to phage_prediction_results.tsv")
                    
                    #-----------------------------------------------------------------------------------------------------------------
                    
                    import json
                    import csv
                    import torch

                    # ---- 读取 node_maps.json 并反转索引映射 ----
                    with open("node_maps.json", "r", encoding="utf-8") as f:
                        node_maps = json.load(f)

                    phage_map = node_maps.get("phage_map", {})
                    host_map  = node_maps.get("host_map", {})

                    # 反转：index(int) -> id(str)
                    phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
                    host_idx2id  = {int(v): str(k) for k, v in host_map.items()}

                    # ---- 根据索引取回 ID ----
                    phage_ids = [phage_idx2id.get(int(nid), str(nid)) for nid in test_src_cpu.tolist()]
                    host_ids  = [host_idx2id.get(int(nid), str(nid)) for nid in test_dst_cpu.tolist()]

                    # ---- 保留你原来的物种逻辑 ----
                    if host_id2taxid is not None and taxid2species is not None:
                        host_species = [taxid2species[host_id2taxid[hid]] for hid in test_dst_cpu.tolist()]
                    else:
                        host_species = ["NA"] * len(host_ids)

                    # ---- 保存结果 ----
                    with open("phage_prediction_results.tsv", "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, delimiter="\t")
                        writer.writerow(["phage_id", "host_id", "host_species", "score"])
                        for pid, hid, hs, s in zip(phage_ids, host_ids, host_species, scores):
                            writer.writerow([pid, hid, hs, float(s)])

                    print("Phage-host prediction results saved to phage_prediction_results.tsv")
                    # import json
                    # import csv
                    # import torch
                    # import numpy as np

                    # # --------------------- 读取 node_maps.json 并反转索引映射 ---------------------
                    # with open("node_maps.json", "r", encoding="utf-8") as f:
                    #     node_maps = json.load(f)

                    # phage_map = node_maps.get("phage_map", {})
                    # host_map  = node_maps.get("host_map", {})

                    # # index(int) -> original id(str)
                    # phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
                    # host_idx2id  = {int(v): str(k) for k, v in host_map.items()}

                    # # --------------------- 如果需要预测全矩阵，先构造 phage × host 全部得分 ---------------------
                    # if scores_tensor.ndim == 1:
                    #     # 当前是单边预测模式，只输出真实边
                    #     phage_ids = [phage_idx2id.get(int(nid), str(nid)) for nid in test_src_cpu.tolist()]
                    #     host_ids  = [host_idx2id.get(int(nid), str(nid)) for nid in test_dst_cpu.tolist()]

                    #     if host_id2taxid is not None and taxid2species is not None:
                    #         host_species = [taxid2species.get(host_id2taxid.get(hid, -1), "NA") for hid in test_dst_cpu.tolist()]
                    #     else:
                    #         host_species = ["NA"] * len(host_ids)

                    #     scores = scores_tensor.cpu().numpy()

                    #     output_file = "phage_prediction_1results.tsv"
                    #     with open(output_file, "w", newline="", encoding="utf-8") as f:
                    #         writer = csv.writer(f, delimiter="\t")
                    #         writer.writerow(["phage_id", "host_id", "host_species", "score"])
                    #         for pid, hid, hs, s in zip(phage_ids, host_ids, host_species, scores):
                    #             writer.writerow([pid, hid, hs, float(s)])

                    #     print(f"[INFO] 单边预测模式，结果已保存到 {output_file}")

                    # else:
                    #     # 当前是全矩阵预测模式，可以输出 Top-K
                    #     scores = scores_tensor.cpu().numpy()  # [num_phage, num_host]
                    #     num_phage, num_host = scores.shape

                    #     topk = 5  # 每个 phage 的 Top-K 宿主
                    #     topk_indices = np.argsort(scores, axis=1)[:, ::-1][:, :topk]
                    #     topk_scores  = np.take_along_axis(scores, topk_indices, axis=1)

                    #     output_file = "phage_prediction_TOPK_results.tsv"
                    #     with open(output_file, "w", newline="", encoding="utf-8") as f:
                    #         writer = csv.writer(f, delimiter="\t")
                    #         writer.writerow(["phage_id", "topk_host_ids", "topk_species", "topk_scores"])

                    #         for pid_idx, indices, scs in zip(range(num_phage), topk_indices, topk_scores):
                    #             pid = phage_idx2id.get(pid_idx, str(pid_idx))
                    #             top_hosts   = [host_idx2id.get(int(i), str(i)) for i in indices]
                    #             top_species = [taxid2species.get(host_id2taxid.get(host_idx2id.get(int(i), ""), -1), "NA") 
                    #                         if host_id2taxid and taxid2species else "NA" for i in indices]

                    #             writer.writerow([
                    #                 pid,
                    #                 ",".join(top_hosts),
                    #                 ",".join(top_species),
                    #                 ",".join([f"{s:.4f}" for s in scs])
                    #             ])

                    #     print(f"[INFO] Top-{topk} 预测结果已保存到 {output_file}")



                    

#-----------------------------------------------------------------------------------------------------------------


                train_auc, train_mrr, train_hits = train_metrics
                val_auc, val_mrr, val_hits = val_metrics
                test_auc, test_mrr, test_hits = test_metrics
            except Exception as e:
                print("Warning: full-graph eval failed:", e)
                train_auc = val_auc = test_auc = float('nan')
                train_mrr = val_mrr = test_mrr = 0.0
                train_hits = val_hits = test_hits = {}

            print(f"[Epoch {epoch:03d}] loss={avg_loss:.6f} time={t1-t0:.1f}s "
                  f"val_auc={val_auc:.4f} val_mrr={val_mrr:.4f} hits@1/5/10={val_hits}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_ckpt = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc
                }
                torch.save(best_ckpt, args.save_path)
                print("Saved best model ->", args.save_path)

    # 使用最佳 checkpoint 进行最终测试
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt['model_state'])
    print("Evaluating final test (full-graph eval on eval_device)...")
    train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
        model,
        data,
        (train_src_cpu, train_dst_cpu),
        (val_src_cpu, val_dst_cpu),
        (test_src_cpu, test_dst_cpu),
        eval_device=args.eval_device,
        eval_neg_ratio=args.eval_neg_ratio,
        host_id2taxid=host_id2taxid,
        taxid2species=taxid2species
    )
    print("FINAL TEST metrics (AUC, MRR, Hits@1/5/10):", test_metrics)


if __name__ == "__main__":
    main()
