import torch
import json
import pandas as pd
from torch_geometric.data import HeteroData
import argparse
import time
import math
import json
from collections import defaultdict
import logging
import os
import csv
import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.loader import LinkNeighborLoader
class HGTMiniModel(nn.Module):
    def __init__(self, metadata: tuple, in_dims: dict[str, int], hidden_dim: int = 256, out_dim: int = 256, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.2, decoder="mlp"):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.input_proj = nn.ModuleDict()
        self.decoder = decoder
        for n in self.node_types:
            d = in_dims.get(n)
            if d is None:
                raise RuntimeError(f"Missing in_dim for node type {n}")
            self.input_proj[n] = nn.Linear(d, hidden_dim)

        self.hgt_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hgt_layers.append(HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=n_heads))

        self.dropout = nn.Dropout(dropout)
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
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

    # def forward(
    #     self,
    #     x_dict: dict[str, torch.Tensor],
    #     edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    #     edge_type_weights: Optional[dict[tuple[str, str, str], float]] = None
    # ) -> dict[str, torch.Tensor]:
    #     """
    #     Forward pass with adjustable edge-type weights.

    #     Args:
    #         x_dict: 节点特征 {ntype: tensor}
    #         edge_index_dict: 边索引 {etype: tensor([2, E])}
    #         edge_type_weights: 每种边类型的缩放系数字典
    #                         例如 {('phage','infects','host'): 0.5, ('protein','similar','protein'): 2.0}
    #                         如果不给，则所有边权重=1.0
    #                         ['phage', 'host', 'protein', 'taxonomy'], [('phage', 'interacts', 'phage'), ('host', 'interacts', 'host'), ('phage', 'encodes', 'protein'), ('host', 'encodes', 'protein'), ('protein', 'similar', 'protein'), ('host', 'belongs_to', 'taxonomy'), ('taxonomy', 'related', 'taxonomy'), ('phage', 'infects', 'host')]
    #     """
    #     h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

    #     # 构造 edge_weight_dict
    #     edge_weight_dict = {
    #         ('protein', 'similar', 'protein'): 0.5
    #     }
    #     for etype, edge_index in edge_index_dict.items():
    #         w = 1.0
    #         if edge_type_weights is not None and etype in edge_type_weights:
    #             w = edge_type_weights[etype]
    #         edge_weight_dict[etype] = torch.full(
    #             (edge_index.size(1),),
    #             float(w),
    #             dtype=torch.float,
    #             device=edge_index.device
    #         )

    #     # 逐层卷积
    #     for conv in self.hgt_layers:
    #         h = conv(h, edge_index_dict, edge_weight_dict=edge_weight_dict)
    #         for k in list(h.keys()):
    #             h[k] = F.relu(self.dropout(h[k]))

    #     out = {k: self.final_proj[k](v) for k, v in h.items()}
    #     return out

    def decode(self, z_dict: dict[str, torch.Tensor], edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], etype: tuple[str, str, str]) -> torch.Tensor:
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

def load_model(ckpt_path, data, args, device):
    ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)
    in_dims = {n: data[n].x.size(1) for n in data.node_types}

    model = HGTMiniModel(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        decoder="mlp"
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ loaded checkpoint from {ckpt_path} (epoch {ckpt['epoch']})")
    return model


def predict_new_phages(graph_path, map_path, ckpt_path, taxid2species_tsv, args, out_tsv, device="cuda"):
    # 1. load graph + mapping
    data: HeteroData = torch.load(graph_path, map_location=device,weights_only=False)
    with open(map_path) as f:
        phage_map = json.load(f)

    # 2. load model
    model = load_model(ckpt_path, data, args, device)

    # 3. forward: get embeddings for all nodes
    with torch.no_grad():
        h_dict = model({nt: data[nt].x.to(device) for nt in data.node_types},
                       {k: v.to(device) for k, v in data.edge_index_dict.items()})

    # 4. host taxid → species name mapping
    host_taxids = data["host"].taxid.cpu().numpy().tolist()
    taxid2species = {}
    if taxid2species_tsv is not None:
        df_tax = pd.read_csv(taxid2species_tsv, sep="\t")
        taxid2species = dict(zip(df_tax["taxid"].astype(int), df_tax["species"]))

    host_id2species = {}
    for idx, taxid in enumerate(host_taxids):
        if taxid in taxid2species:
            host_id2species[idx] = (taxid, taxid2species[taxid])
        else:
            host_id2species[idx] = (taxid, "Unknown")

    # 5. find phage->host relation type
    relation = None
    for r in data.edge_types:
        if r[0] == "phage" and r[2] == "host":
            relation = r
            break
    if relation is None:
        raise RuntimeError("phage->host relation not found in graph")

    host_ids = list(range(data["host"].num_nodes))

    # 6. predict for each new phage
    results = []
    for phage_id, phage_idx in phage_map.items():
        phage_tensor = torch.tensor([phage_idx] * len(host_ids), device=device)
        host_tensor = torch.tensor(host_ids, device=device)

        scores = model.decode(h_dict, (phage_tensor, host_tensor), etype=relation)
        scores = torch.sigmoid(scores)
        scores = scores.detach().cpu().numpy()
        

        # sort & take top-k
        k = 10
        topk_idx = scores.argsort()[::-1][:k]
        for rank, hi in enumerate(topk_idx, 1):
            taxid, species = host_id2species[hi]
            results.append({
                "phage_id": phage_id,
                "host_species_taxid": taxid,
                "host_species_name": species,
                "score": float(scores[hi]),
                "rank": rank
            })

    # 7. save results
    df = pd.DataFrame(results)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"✅ saved predictions to {out_tsv}")


# ==== Example usage ====
if __name__ == "__main__":
    class Args:
        hidden_dim = 256
        out_dim = 256
        n_layers = 2
        n_heads = 8
        dropout = 0.2
    args = Args()

    predict_new_phages(
        graph_path="graph_with_newphages_cherry.pt",
        map_path="newphage_mapping_cherry.json",
        ckpt_path="best_hgt_nb_RBP_5000-512_15_5e_ph.pt",
        taxid2species_tsv="taxid_species.tsv",   # 需要一张映射表
        args=args,
        out_tsv="newphage_predictions_cherry_old.tsv",
        device="cuda"
    )
