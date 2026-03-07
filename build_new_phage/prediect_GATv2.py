#!/usr/bin/env python3
# predict_new_phages_gatv2.py
# 完整预测脚本（GATv2 + edge_attr 支持）
# 保存后运行前请调整路径和参数

# ---------- 必要 imports（把这些放在文件头部，若已有重复可忽略已有导入） ----------
import typing
import torch
import json
import pandas as pd
from torch_geometric.data import HeteroData
import argparse
import time
import math
from collections import defaultdict
import logging
import os
import csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import LinkNeighborLoader

# NEW imports for GATv2 implementation
from torch_geometric.nn import GATv2Conv, HeteroConv

# ---------- GATv2MiniModel (替换掉原 HGTMiniModel) ----------
class GATv2MiniModel(nn.Module):
    def __init__(
        self,
        metadata: tuple,                 # (node_types, edge_types)
        in_dims: dict,                   # {ntype: in_dim}
        hidden_dim: int = 256,
        out_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        decoder: str = "mlp",
        use_edge_attr: bool = True,      # 我们要用带权重的模型，默认启用
        edge_attr_dim: int = 1,          # 权重为标量 -> dim = 1
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.decoder = decoder
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout_p = dropout
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim

        # input projection
        self.input_proj = nn.ModuleDict()
        for n in self.node_types:
            d = in_dims.get(n)
            if d is None:
                raise RuntimeError(f"Missing in_dim for node type {n}")
            self.input_proj[n] = nn.Linear(d, hidden_dim)

        # Use concat=False to avoid needing hidden_dim % n_heads == 0
        concat_flag = False
        out_channels = hidden_dim

        # per-layer ModuleDict -> HeteroConv
        self.edge_conv_md_list = nn.ModuleList()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            convs_md = nn.ModuleDict()
            for (src, rel, dst) in self.edge_types:
                str_key = f"{src}__{rel}__{dst}"
                add_self_loops_flag = (src == dst)  # only allow self-loops for homogeneous relations
                if self.use_edge_attr:
                    conv = GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=out_channels,
                        heads=n_heads,
                        concat=concat_flag,
                        dropout=dropout,
                        edge_dim=self.edge_attr_dim,
                        add_self_loops=add_self_loops_flag
                    )
                else:
                    conv = GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=out_channels,
                        heads=n_heads,
                        concat=concat_flag,
                        dropout=dropout,
                        add_self_loops=add_self_loops_flag
                    )
                convs_md[str_key] = conv

            self.edge_conv_md_list.append(convs_md)
            conv_map = {etype: convs_md[f"{etype[0]}__{etype[1]}__{etype[2]}"] for etype in self.edge_types}
            self.layers.append(HeteroConv(conv_map, aggr='sum'))

        self.dropout = nn.Dropout(self.dropout_p)

        # final proj and decoder (keep same behavior as HGTMiniModel)
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))
        if decoder == "mlp":
            self.decoder_mlp = self.edge_mlp

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: typing.Optional[dict] = None,   # etype -> scalar or tensor
    ) -> dict[str, torch.Tensor]:
        # 1) input projection
        h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

        # 2) per-layer propagation
        for layer in self.layers:
            if self.use_edge_attr and edge_attr_dict is not None:
                processed = {}
                for etype, edge_index in edge_index_dict.items():
                    E = edge_index.size(1)
                    if etype in edge_attr_dict:
                        val = edge_attr_dict[etype]
                        if isinstance(val, (float, int)):
                            # scalar -> expand
                            if self.edge_attr_dim == 1:
                                t = torch.full((E,), float(val), dtype=torch.float, device=edge_index.device)
                            else:
                                t = torch.full((E, self.edge_attr_dim), float(val), dtype=torch.float, device=edge_index.device)
                        elif isinstance(val, torch.Tensor):
                            t = val.to(edge_index.device)
                            if t.dim() == 1:
                                if self.edge_attr_dim == 1:
                                    if t.size(0) != E:
                                        raise RuntimeError(f"edge_attr for {etype} len {t.size(0)} != expected {E}")
                                else:
                                    if t.size(0) != E:
                                        raise RuntimeError(f"edge_attr for {etype} len {t.size(0)} != expected {E}")
                                    t = t.view(-1, 1).repeat(1, self.edge_attr_dim)
                            elif t.dim() == 2:
                                if t.size(0) != E or t.size(1) != self.edge_attr_dim:
                                    raise RuntimeError(f"edge_attr for {etype} shape {tuple(t.size())} != expected (E,{self.edge_attr_dim})")
                            else:
                                raise RuntimeError(f"edge_attr for {etype} must be 1D or 2D tensor")
                        else:
                            raise RuntimeError(f"Unsupported edge_attr type for {etype}: {type(val)}")
                        processed[etype] = t
                # pass processed as *_dict (HeteroConv expects *_dict)
                h = layer(h, edge_index_dict, edge_attr_dict=processed)
            else:
                h = layer(h, edge_index_dict)

            for k in list(h.keys()):
                h[k] = F.relu(self.dropout(h[k]))

        out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
        #out = {k: self.final_proj[k](v) for k, v in h.items()}
        return out

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        etype: tuple[str, str, str]
    ) -> torch.Tensor:
        if isinstance(edge_label_index, torch.Tensor) and edge_label_index.dim() == 2 and edge_label_index.size(0) == 2:
            src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        elif isinstance(edge_label_index, (tuple, list)) and len(edge_label_index) == 2:
            src_idx, dst_idx = edge_label_index
        else:
            raise RuntimeError("edge_label_index must be (2,E) or tuple(src,dst)")
        src_type, _, dst_type = etype
        src_z = z_dict[src_type][src_idx]
        dst_z = z_dict[dst_type][dst_idx]
        if self.decoder == "cosine":
            return F.cosine_similarity(src_z, dst_z)
        elif self.decoder == "mlp":
            e = torch.cat([src_z, dst_z], dim=-1)
            return self.edge_mlp(e).view(-1)
        else:
            raise ValueError(f"Unknown decoder {self.decoder}")

# ---------- load_model: instantiate GATv2MiniModel and load ckpt ----------
def load_model(ckpt_path, data, args, device, use_edge_attr=True, edge_attr_dim=1):
    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_dims = {n: data[n].x.size(1) for n in data.node_types}

    model = GATv2MiniModel(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        decoder="mlp",
        use_edge_attr=use_edge_attr,
        edge_attr_dim=edge_attr_dim
    ).to(device)

    # checkpoint expected to contain 'model_state' (as in your training script)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        # fallback: assume ckpt is state_dict directly
        model.load_state_dict(ckpt)
    model.eval()
    print(f"✅ loaded checkpoint from {ckpt_path} (keys: {list(ckpt.keys())[:5]})")
    return model

# ---------- predict_new_phages: use model with edge weights ----------
#def predict_new_phages(graph_path, map_path, ckpt_path, taxid2species_tsv, args, out_tsv, device="cuda", edge_type_weight_map: typing.Optional[dict]=None):
    """
    edge_type_weight_map: optional dict mapping etype tuple -> scalar weight, e.g. {('phage','infects','host'): 0.5}
    Priority for edge attributes:
      1) data[etype].edge_weight (if present)
      2) edge_type_weight_map scalar (if provided)
    If neither provided for an etype, that etype will have no edge_attr (model will run without it).
    """
    # 1. load graph + mapping
    data: HeteroData = torch.load(graph_path, map_location="cpu", weights_only=False)
    with open(map_path) as f:
        phage_map = json.load(f)

    # 2. load model (use edge_attr support)
    device_t = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    model = load_model(ckpt_path, data, args, device_t, use_edge_attr=True, edge_attr_dim=1)

    # 3. build full_x and full_edge_index_dict (on cpu then move to device)
    full_x = {nt: data[nt].x.to(device_t) for nt in data.node_types}
    full_edge_index_dict = {
        et: data[et].edge_index.to(device_t)
        for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
    }

    # 4. Build global_edge_attr dict:
    # default edge_type_weight_map if not provided (adjust as you trained)
    if edge_type_weight_map is None:
        edge_type_weight_map = {
            ('phage','infects','host'): 2.0,
            ('protein','similar','protein'): 0.5,
            ('host','has_sequence','host_sequence'): 1.0,
            ('phage','interacts','phage'): 1.0,
            ('host','interacts','host'): 1.0,
            ('phage','encodes','protein'): 1.0,
            ('host','encodes','protein'): 1.0,
            ('host','belongs_to','taxonomy'): 1.0,
            ('taxonomy','related','taxonomy'): 1.0,
        }

    global_edge_attr = {}
    for et, edge_index in full_edge_index_dict.items():
        E = edge_index.size(1)
        # priority 1: data[et].edge_weight
        if hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
            val = data[et].edge_weight.to(device_t)
            if val.dim() == 0:
                val = val.expand(E)
            global_edge_attr[et] = val
        # priority 2: edge_type_weight_map scalar
        elif et in edge_type_weight_map:
            w = float(edge_type_weight_map[et])
            global_edge_attr[et] = torch.full((E,), w, device=device_t, dtype=torch.float)
        else:
            # no edge_attr provided for this etype
            pass

    # 5. forward to get embeddings (with edge_attr_dict passed)
    model.eval()
    with torch.no_grad():
        h_dict = model(full_x, full_edge_index_dict, edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None)

    # 6. host taxid -> species mapping
    host_taxids = data["host"].taxid.cpu().numpy().tolist() if hasattr(data["host"], "taxid") else []
    taxid2species = {}
    if taxid2species_tsv is not None:
        df_tax = pd.read_csv(taxid2species_tsv, sep="\t")
        # try to detect column names
        if "taxid" in df_tax.columns and "species" in df_tax.columns:
            taxid_col, species_col = "taxid", "species"
        elif "taxid" in df_tax.columns and "species_name" in df_tax.columns:
            taxid_col, species_col = "taxid", "species_name"
        else:
            # fallback to first two columns
            taxid_col, species_col = df_tax.columns[0], df_tax.columns[1]
        taxid2species = dict(zip(df_tax[taxid_col].astype(int), df_tax[species_col].astype(str)))

    host_id2species = {}
    for idx, taxid in enumerate(host_taxids):
        try:
            t_int = int(taxid)
        except Exception:
            t_int = taxid
        host_id2species[idx] = (t_int, taxid2species.get(t_int, "Unknown"))

    # 7. find phage->host relation type
    relation = None
    for r in data.edge_types:
        if r[0] == "phage" and r[2] == "host":
            relation = r
            break
    if relation is None:
        raise RuntimeError("phage->host relation not found in graph")

    # 8. predict for each new phage
    host_num = int(data["host"].num_nodes)
    host_ids = list(range(host_num))
    results = []
    k = 10  # top-k
    for phage_id, phage_idx in phage_map.items():
        phage_tensor = torch.tensor([int(phage_idx)] * host_num, device=device_t, dtype=torch.long)
        host_tensor = torch.tensor(host_ids, device=device_t, dtype=torch.long)

        with torch.no_grad():
            scores = model.decode(h_dict, (phage_tensor, host_tensor), etype=relation)
            scores = torch.sigmoid(scores)
            scores = scores.detach().cpu().numpy()

        # sort & take top-k
        topk_idx = scores.argsort()[::-1][:k]
        for rank, hi in enumerate(topk_idx, 1):
            taxid, species = host_id2species[int(hi)]
            results.append({
                "phage_id": phage_id,
                "phage_node_idx": int(phage_idx),
                "host_node_idx": int(hi),
                "host_taxid": int(taxid) if isinstance(taxid, (int, np.integer)) else taxid,
                "host_species_name": species,
                "score": float(scores[int(hi)]),
                "rank": rank
            })
    # 假设 full_edge_index_dict, global_edge_attr 如脚本中生成
    print("edge_index types:", list(full_edge_index_dict.keys()))
    print("edge_attr types:", list(global_edge_attr.keys()))
    for et in full_edge_index_dict.keys():
        print(et, "E =", full_edge_index_dict[et].size(1),
            "has_attr?", et in global_edge_attr,
            "attr_shape:", None if et not in global_edge_attr else tuple(global_edge_attr[et].shape))

    # 9. save results
    df = pd.DataFrame(results)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"✅ saved predictions to {out_tsv}")
# ---------- predict_new_phages: use model with edge weights ----------
# ---------- predict_new_phages: use model with edge weights ----------
def predict_new_phages(graph_path, map_path, ckpt_path, taxid2species_tsv, args, out_tsv, device="cuda", edge_type_weight_map: typing.Optional[dict]=None):
    """
    edge_type_weight_map: optional dict mapping etype tuple -> scalar weight, e.g. {('phage','infects','host'): 0.5}
    Priority for edge attributes:
      1) data[etype].edge_weight (if present)
      2) edge_type_weight_map scalar (if provided)
    If neither provided for an etype, that etype will have no edge_attr (model will run without it).
    """
    # 1. load graph + mapping
    data: HeteroData = torch.load(graph_path, map_location="cpu", weights_only=False)
    with open(map_path) as f:
        phage_map = json.load(f)

    # 2. load model (use edge_attr support)
    device_t = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    model = load_model(ckpt_path, data, args, device_t, use_edge_attr=True, edge_attr_dim=1)

    # 3. build full_x and full_edge_index_dict
    full_x = {nt: data[nt].x.to(device_t) for nt in data.node_types}
    full_edge_index_dict = {
        et: data[et].edge_index.to(device_t)
        for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
    }

    # 4. Build global_edge_attr dict
    if edge_type_weight_map is None:
        edge_type_weight_map = {
            ('phage','infects','host'): 2.0,
            ('protein','similar','protein'): 0.5,
            ('host','has_sequence','host_sequence'): 1.0,
            ('phage','interacts','phage'): 1.0,
            ('host','interacts','host'): 1.0,
            ('phage','encodes','protein'): 1.0,
            ('host','encodes','protein'): 1.0,
            ('host','belongs_to','taxonomy'): 1.0,
            ('taxonomy','related','taxonomy'): 1.0,
        }

    global_edge_attr = {}
    for et, edge_index in full_edge_index_dict.items():
        E = edge_index.size(1)
        # priority 1: data[et].edge_weight
        if hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
            val = data[et].edge_weight.to(device_t)
            if val.dim() == 0:
                val = val.expand(E)
            global_edge_attr[et] = val
        # priority 2: edge_type_weight_map scalar
        elif et in edge_type_weight_map:
            w = float(edge_type_weight_map[et])
            global_edge_attr[et] = torch.full((E,), w, device=device_t, dtype=torch.float)
        else:
            # no edge_attr provided for this etype
            pass

    # 5. forward to get embeddings (with edge_attr_dict passed)
    model.eval()
    with torch.no_grad():
        # ⚠️ 不再手动调用 final_proj，保持训练输出一致
        h_dict = model(full_x, full_edge_index_dict, edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None)

    # 6. host taxid -> species mapping
    host_taxids = data["host"].taxid.cpu().numpy().tolist() if hasattr(data["host"], "taxid") else []
    taxid2species = {}
    if taxid2species_tsv is not None:
        df_tax = pd.read_csv(taxid2species_tsv, sep="\t")
        if "taxid" in df_tax.columns and "species" in df_tax.columns:
            taxid_col, species_col = "taxid", "species"
        elif "taxid" in df_tax.columns and "species_name" in df_tax.columns:
            taxid_col, species_col = "taxid", "species_name"
        else:
            taxid_col, species_col = df_tax.columns[0], df_tax.columns[1]
        taxid2species = dict(zip(df_tax[taxid_col].astype(int), df_tax[species_col].astype(str)))

    host_id2species = {}
    for idx, taxid in enumerate(host_taxids):
        try:
            t_int = int(taxid)
        except Exception:
            t_int = taxid
        host_id2species[idx] = (t_int, taxid2species.get(t_int, "Unknown"))

    # 7. find phage->host relation type
    relation = None
    for r in data.edge_types:
        if r[0] == "phage" and r[2] == "host":
            relation = r
            break
    if relation is None:
        raise RuntimeError("phage->host relation not found in graph")

    # 8. predict for each new phage
    host_num = int(data["host"].num_nodes)
    host_ids = list(range(host_num))
    results = []
    k = 10  # top-k
    for phage_id, phage_idx in phage_map.items():
        phage_tensor = torch.tensor([int(phage_idx)] * host_num, device=device_t, dtype=torch.long)
        host_tensor = torch.tensor(host_ids, device=device_t, dtype=torch.long)

        with torch.no_grad():
            scores = model.decode(h_dict, (phage_tensor, host_tensor), etype=relation)
            scores = torch.sigmoid(scores)
            scores = scores.detach().cpu().numpy()

        topk_idx = scores.argsort()[::-1][:k]
        for rank, hi in enumerate(topk_idx, 1):
            taxid, species = host_id2species[int(hi)]
            results.append({
                "phage_id": phage_id,
                "phage_node_idx": int(phage_idx),
                "host_node_idx": int(hi),
                "host_taxid": int(taxid) if isinstance(taxid, (int, np.integer)) else taxid,
                "host_species_name": species,
                "score": float(scores[int(hi)]),
                "rank": rank
            })

    # 9. 打印 edge 信息（debug）
    print("edge_index types:", list(full_edge_index_dict.keys()))
    print("edge_attr types:", list(global_edge_attr.keys()))
    for et in full_edge_index_dict.keys():
        print(et, "E =", full_edge_index_dict[et].size(1),
              "has_attr?", et in global_edge_attr,
              "attr_shape:", None if et not in global_edge_attr else tuple(global_edge_attr[et].shape))

    # 10. save results
    df = pd.DataFrame(results)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"✅ saved predictions to {out_tsv}")

# ==== Example usage ====
if __name__ == "__main__":
    class Args:
        hidden_dim = 512
        out_dim = 256
        n_layers = 2
        n_heads = 4
        dropout = 0.2
    args = Args()


    predict_new_phages(
        graph_path="graph_with_newphages_ppedges.pt",
        map_path="merged_phage_mapping.json",
        ckpt_path="best_hgt_nb_RBP_GAT_4heads_weight_20.10_hid512_1024_pp2_out.pt",
        taxid2species_tsv="taxid_species.tsv",   # 需要一张映射表
        args=args,
        out_tsv="newphage_predictions_cherry_pp_out.tsv",
        device="cuda"
    )
