# #!/usr/bin/env python3
# # predict_new_phages_gatv2_modified.py
# # 与训练模型保持一致的预测脚本（包含可学习 logit_scale，支持 cosine/mlp decoder，保持 forward 归一化）
# # 保存后运行前请调整路径和参数

# # ---------- 必要 imports（把这些放在文件头部，若已有重复可忽略已有导入） ----------
# import typing
# import torch
# import json
# import pandas as pd
# from torch_geometric.data import HeteroData
# import argparse
# import time
# import math
# from collections import defaultdict
# import logging
# import os
# import csv
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional
# from sklearn.metrics import roc_auc_score
# from torch_geometric.loader import LinkNeighborLoader

# # NEW imports for GATv2 implementation
# from torch_geometric.nn import GATv2Conv, HeteroConv

# # ---------- GATv2MiniModel (与训练版本保持一致) ----------
# class GATv2MiniModel(nn.Module):
#     def __init__(
#         self,
#         metadata: tuple,                 # (node_types, edge_types)
#         in_dims: dict,                   # {ntype: in_dim}
#         hidden_dim: int = 256,
#         out_dim: int = 256,
#         n_layers: int = 2,
#         n_heads: int = 4,
#         dropout: float = 0.2,
#         decoder: str = "mlp",            # 支持 "mlp" 或 "cosine"
#         use_edge_attr: bool = True,      # 我们要用带权重的模型，默认启用
#         edge_attr_dim: int = 1,          # 权重为标量 -> dim = 1
#     ):
#         super().__init__()
#         self.metadata = metadata
#         self.node_types, self.edge_types = metadata
#         # <<< MOD: use attribute name decoder_type to match training model
#         self.decoder_type = decoder
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.n_heads = n_heads
#         self.dropout_p = dropout
#         self.use_edge_attr = use_edge_attr
#         self.edge_attr_dim = edge_attr_dim

#         # input projection
#         self.input_proj = nn.ModuleDict()
#         for n in self.node_types:
#             d = in_dims.get(n)
#             if d is None:
#                 raise RuntimeError(f"Missing in_dim for node type {n}")
#             self.input_proj[n] = nn.Linear(d, hidden_dim)

#         # Use concat=False to avoid needing hidden_dim % n_heads == 0
#         concat_flag = False
#         out_channels = hidden_dim

#         # per-layer ModuleDict -> HeteroConv
#         self.edge_conv_md_list = nn.ModuleList()
#         self.layers = nn.ModuleList()
#         for _ in range(n_layers):
#             convs_md = nn.ModuleDict()
#             for (src, rel, dst) in self.edge_types:
#                 str_key = f"{src}__{rel}__{dst}"
#                 add_self_loops_flag = (src == dst)  # only allow self-loops for homogeneous relations
#                 if self.use_edge_attr:
#                     conv = GATv2Conv(
#                         in_channels=hidden_dim,
#                         out_channels=out_channels,
#                         heads=n_heads,
#                         concat=concat_flag,
#                         dropout=dropout,
#                         edge_dim=self.edge_attr_dim,
#                         add_self_loops=add_self_loops_flag
#                     )
#                 else:
#                     conv = GATv2Conv(
#                         in_channels=hidden_dim,
#                         out_channels=out_channels,
#                         heads=n_heads,
#                         concat=concat_flag,
#                         dropout=dropout,
#                         add_self_loops=add_self_loops_flag
#                     )
#                 convs_md[str_key] = conv

#             self.edge_conv_md_list.append(convs_md)
#             conv_map = {etype: convs_md[f"{etype[0]}__{etype[1]}__{etype[2]}"] for etype in self.edge_types}
#             self.layers.append(HeteroConv(conv_map, aggr='sum'))

#         self.dropout = nn.Dropout(self.dropout_p)

#         # final proj and decoder (keep same behavior as HGTMiniModel)
#         self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
#         self.edge_mlp = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))
#         if self.decoder_type == "mlp":
#             # <<< MOD: keep name decoder_mlp like training model
#             self.decoder_mlp = self.edge_mlp

#         # <<< MOD: add learnable logit_scale parameter (stored in log-space)
#         # initial logit_scale = 0.0 -> scale = 1.0
#         self.logit_scale = nn.Parameter(torch.tensor(0.0))

#     def forward(
#         self,
#         x_dict: dict[str, torch.Tensor],
#         edge_index_dict: dict[tuple, torch.Tensor],
#         edge_attr_dict: typing.Optional[dict] = None,   # etype -> scalar or tensor
#     ) -> dict[str, torch.Tensor]:
#         # 1) input projection
#         h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

#         # 2) per-layer propagation
#         for layer in self.layers:
#             if self.use_edge_attr and edge_attr_dict is not None:
#                 processed = {}
#                 for etype, edge_index in edge_index_dict.items():
#                     E = edge_index.size(1)
#                     if etype in edge_attr_dict:
#                         val = edge_attr_dict[etype]
#                         if isinstance(val, (float, int)):
#                             # scalar -> expand
#                             if self.edge_attr_dim == 1:
#                                 t = torch.full((E,), float(val), dtype=torch.float, device=edge_index.device)
#                             else:
#                                 t = torch.full((E, self.edge_attr_dim), float(val), dtype=torch.float, device=edge_index.device)
#                         elif isinstance(val, torch.Tensor):
#                             t = val.to(edge_index.device)
#                             if t.dim() == 1:
#                                 if self.edge_attr_dim == 1:
#                                     if t.size(0) != E:
#                                         raise RuntimeError(f"edge_attr for {etype} len {t.size(0)} != expected {E}")
#                                 else:
#                                     if t.size(0) != E:
#                                         raise RuntimeError(f"edge_attr for {etype} len {t.size(0)} != expected {E}")
#                                     t = t.view(-1, 1).repeat(1, self.edge_attr_dim)
#                             elif t.dim() == 2:
#                                 if t.size(0) != E or t.size(1) != self.edge_attr_dim:
#                                     raise RuntimeError(f"edge_attr for {etype} shape {tuple(t.size())} != expected (E,{self.edge_attr_dim})")
#                             else:
#                                 raise RuntimeError(f"edge_attr for {etype} must be 1D or 2D tensor")
#                         else:
#                             raise RuntimeError(f"Unsupported edge_attr type for {etype}: {type(val)}")
#                         processed[etype] = t
#                 # pass processed as *_dict (HeteroConv expects *_dict)
#                 # <<< MOD: HeteroConv in this code will accept edge_attr_dict keyword as processed
#                 h = layer(h, edge_index_dict, edge_attr_dict=processed)
#             else:
#                 h = layer(h, edge_index_dict)

#             for k in list(h.keys()):
#                 h[k] = F.relu(self.dropout(h[k]))

#         # <<< MOD: return normalized embeddings (L2=1) to match training model behavior
#         out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
#         return out

#     def decode(
#         self,
#         z_dict: dict[str, torch.Tensor],
#         edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
#         etype: tuple[str, str, str]
#     ) -> torch.Tensor:
#         if isinstance(edge_label_index, torch.Tensor) and edge_label_index.dim() == 2 and edge_label_index.size(0) == 2:
#             src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
#         elif isinstance(edge_label_index, (tuple, list)) and len(edge_label_index) == 2:
#             src_idx, dst_idx = edge_label_index
#         else:
#             raise RuntimeError("edge_label_index must be (2,E) or tuple(src,dst)")
#         src_type, _, dst_type = etype
#         src_z = z_dict[src_type][src_idx]
#         dst_z = z_dict[dst_type][dst_idx]
#         if self.decoder_type == "cosine":
#             # defensive normalization (idempotent if forward already normalized)
#             src_n = F.normalize(src_z, p=2, dim=-1)
#             dst_n = F.normalize(dst_z, p=2, dim=-1)
#             sim = F.cosine_similarity(src_n, dst_n)   # in [-1,1]
#             # <<< MOD: apply learnable scale (positive) using exp(logit_scale)
#             return sim * torch.exp(self.logit_scale)
#         elif self.decoder_type == "mlp":
#             e = torch.cat([src_z, dst_z], dim=-1)
#             return self.decoder_mlp(e).view(-1)
#         else:
#             raise ValueError(f"Unknown decoder {self.decoder_type}")

# # ---------- load_model: instantiate GATv2MiniModel and load ckpt ----------
# def load_model(ckpt_path, data, args, device, use_edge_attr=True, edge_attr_dim=1):
#     # load checkpoint
#     ckpt = torch.load(ckpt_path, map_location=device)
#     # build in_dims from data
#     in_dims = {n: data[n].x.size(1) for n in data.node_types}

#     model = GATv2MiniModel(
#         metadata=data.metadata(),
#         in_dims=in_dims,
#         hidden_dim=args.hidden_dim,
#         out_dim=args.out_dim,
#         n_layers=args.n_layers,
#         n_heads=args.n_heads,
#         dropout=args.dropout,
#         decoder=args.decoder if hasattr(args, "decoder") else "mlp",  # <<< MOD: allow choosing decoder via args
#         use_edge_attr=use_edge_attr,
#         edge_attr_dim=edge_attr_dim
#     ).to(device)

#     # checkpoint may contain different key naming; attempt common keys
#     if isinstance(ckpt, dict):
#         # common naming: 'model_state' or 'state_dict' or direct state dict
#         if "model_state" in ckpt:
#             state = ckpt["model_state"]
#         elif "state_dict" in ckpt:
#             state = ckpt["state_dict"]
#         else:
#             # assume ckpt is state dict mapping param->tensor (or contains them at top-level)
#             state = ckpt
#     else:
#         state = ckpt

#     # strip module. prefix if present
#     new_state = {}
#     for k, v in state.items():
#         new_k = k.replace("module.", "") if k.startswith("module.") else k
#         new_state[new_k] = v

#     # load (allow missing / unexpected keys but warn)
#     missing, unexpected = model.load_state_dict(new_state, strict=False)
#     if missing:
#         print(f"[warn] Missing keys when loading model: {missing}")
#     if unexpected:
#         print(f"[warn] Unexpected keys when loading model: {unexpected}")

#     model.eval()
#     print(f"✅ loaded checkpoint from {ckpt_path}")
#     return model

# # ---------- predict_new_phages: use model with edge weights ----------
# def predict_new_phages(graph_path, map_path, ckpt_path, taxid2species_tsv, args, out_tsv, device="cuda", edge_type_weight_map: typing.Optional[dict]=None):
#     """
#     edge_type_weight_map: optional dict mapping etype tuple -> scalar weight, e.g. {('phage','infects','host'): 0.5}
#     Priority for edge attributes:
#       1) data[etype].edge_weight (if present)
#       2) edge_type_weight_map scalar (if provided)
#     If neither provided for an etype, that etype will have no edge_attr (model will run without it).
#     """
#     # 1. load graph + mapping
#     data: HeteroData = torch.load(graph_path, map_location="cpu",weights_only=False)
#     with open(map_path) as f:
#         phage_map = json.load(f)

#     # 2. load model (use edge_attr support)
#     device_t = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
#     model = load_model(ckpt_path, data, args, device_t, use_edge_attr=True, edge_attr_dim=1)

#     # 3. build full_x and full_edge_index_dict
#     full_x = {nt: data[nt].x.to(device_t) for nt in data.node_types}
#     full_edge_index_dict = {
#         et: data[et].edge_index.to(device_t)
#         for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
#     }

#     # 4. Build global_edge_attr dict
#     if edge_type_weight_map is None:
#         edge_type_weight_map = {
#             ('phage','infects','host'): 1.0,
#             ('protein','similar','protein'): 1.0,
#             ('host','has_sequence','host_sequence'): 1.0,
#             ('phage','interacts','phage'): 1.0,
#             ('host','interacts','host'): 1.0,
#             ('phage','encodes','protein'): 1.0,
#             ('host','encodes','protein'): 1.0,
#             ('host','belongs_to','taxonomy'): 1.0,
#             ('taxonomy','related','taxonomy'): 1.0,
#         }

#     global_edge_attr = {}
#     for et, edge_index in full_edge_index_dict.items():
#         E = edge_index.size(1)
#         # priority 1: data[et].edge_weight
#         if hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
#             val = data[et].edge_weight.to(device_t)
#             if val.dim() == 0:
#                 val = val.expand(E)
#             global_edge_attr[et] = val
#         # priority 2: edge_type_weight_map scalar
#         elif et in edge_type_weight_map:
#             w = float(edge_type_weight_map[et])
#             global_edge_attr[et] = torch.full((E,), w, device=device_t, dtype=torch.float)
#         else:
#             # no edge_attr provided for this etype
#             pass

#     # 5. forward to get embeddings (with edge_attr_dict passed)
#     model.eval()
#     with torch.no_grad():
#         h_dict = model(full_x, full_edge_index_dict, edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None)

#     # 6. host taxid -> species mapping
#     host_taxids = data["host"].taxid.cpu().numpy().tolist() if hasattr(data["host"], "taxid") else []
#     taxid2species = {}
#     if taxid2species_tsv is not None:
#         df_tax = pd.read_csv(taxid2species_tsv, sep="\t")
#         if "taxid" in df_tax.columns and "species" in df_tax.columns:
#             taxid_col, species_col = "taxid", "species"
#         elif "taxid" in df_tax.columns and "species_name" in df_tax.columns:
#             taxid_col, species_col = "taxid", "species_name"
#         else:
#             taxid_col, species_col = df_tax.columns[0], df_tax.columns[1]
#         taxid2species = dict(zip(df_tax[taxid_col].astype(int), df_tax[species_col].astype(str)))

#     host_id2species = {}
#     for idx, taxid in enumerate(host_taxids):
#         try:
#             t_int = int(taxid)
#         except Exception:
#             t_int = taxid
#         host_id2species[idx] = (t_int, taxid2species.get(t_int, "Unknown"))

#     # 7. find phage->host relation type
#     relation = None
#     for r in data.edge_types:
#         if r[0] == "phage" and r[2] == "host":
#             relation = r
#             break
#     if relation is None:
#         raise RuntimeError("phage->host relation not found in graph")

#     # 8. predict for each new phage
#     host_num = int(data["host"].num_nodes)
#     host_ids = list(range(host_num))
#     results = []
#     k = 10  # top-k
#     for phage_id, phage_idx in phage_map.items():
#         phage_tensor = torch.tensor([int(phage_idx)] * host_num, device=device_t, dtype=torch.long)
#         host_tensor = torch.tensor(host_ids, device=device_t, dtype=torch.long)

#         with torch.no_grad():
#             scores = model.decode(h_dict, (phage_tensor, host_tensor), etype=relation)
#             # NOTE: decode returns logits scaled by logit_scale if cosine; if you want probabilities you can sigmoid.
#             # Keep original behavior: convert to probabilities with sigmoid (optional)
#             probs = torch.sigmoid(scores).detach().cpu().numpy()

#         topk_idx = probs.argsort()[::-1][:k]
#         for rank, hi in enumerate(topk_idx, 1):
#             taxid, species = host_id2species[int(hi)]
#             results.append({
#                 "phage_id": phage_id,
#                 "phage_node_idx": int(phage_idx),
#                 "host_node_idx": int(hi),
#                 "host_taxid": int(taxid) if isinstance(taxid, (int, np.integer)) else taxid,
#                 "host_species_name": species,
#                 "score": float(probs[int(hi)]),   # stored as probability after sigmoid
#                 "rank": rank
#             })

#     # 9. 打印 edge 信息（debug）
#     print("edge_index types:", list(full_edge_index_dict.keys()))
#     print("edge_attr types:", list(global_edge_attr.keys()))
#     for et in full_edge_index_dict.keys():
#         print(et, "E =", full_edge_index_dict[et].size(1),
#               "has_attr?", et in global_edge_attr,
#               "attr_shape:", None if et not in global_edge_attr else tuple(global_edge_attr[et].shape))

#     # 10. save results
#     df = pd.DataFrame(results)
#     df.to_csv(out_tsv, sep="\t", index=False)
#     print(f"✅ saved predictions to {out_tsv}")


# # ==== Example usage ====
# if __name__ == "__main__":
#     class Args:
#         hidden_dim = 512
#         out_dim = 256
#         n_layers = 2
#         n_heads = 4
#         dropout = 0.2
#         decoder = "cosine"  # <<< MOD: allow "cosine" if you trained with cosine
#     args = Args()


#     predict_new_phages(
#         graph_path="graph_with_newphages_ppedges.pt",
#         map_path="merged_phage_mapping.json",
#         ckpt_path="/home/wangjingyuan/wys/build_new_phage/best_GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg30_evl20_ph2_p0.5_1e-5_cos.pt",
#         taxid2species_tsv="taxid_species.tsv",   # 需要一张映射表
#         args=args,
#         out_tsv="newphage_predictions_cherry_pp_out.tsv",
#         device="cuda"
#     )












#!/usr/bin/env python3
# predict_new_phages_gatv2_modified.py
# 与训练模型保持一致的预测脚本（包含可学习 logit_scale，支持 cosine/mlp decoder，保持 forward 归一化）
# 保存后运行前请调整路径和参数

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
from torch_geometric.nn import GATv2Conv, HeteroConv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Model (与训练版本保持一致) ----------
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
        decoder: str = "mlp",            # 支持 "mlp" 或 "cosine"
        use_edge_attr: bool = True,
        edge_attr_dim: int = 1,
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.decoder_type = decoder
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

        concat_flag = False
        out_channels = hidden_dim

        self.edge_conv_md_list = nn.ModuleList()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            convs_md = nn.ModuleDict()
            for (src, rel, dst) in self.edge_types:
                str_key = f"{src}__{rel}__{dst}"
                add_self_loops_flag = (src == dst)
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

        # final proj + decoder
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))
        if self.decoder_type == "mlp":
            self.decoder_mlp = self.edge_mlp

        # learnable logit_scale in log-space (init 0 -> scale=1)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: typing.Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

        for layer in self.layers:
            if self.use_edge_attr and edge_attr_dict is not None:
                processed = {}
                for etype, edge_index in edge_index_dict.items():
                    E = edge_index.size(1)
                    if etype in edge_attr_dict:
                        val = edge_attr_dict[etype]
                        if isinstance(val, (float, int)):
                            if self.edge_attr_dim == 1:
                                t = torch.full((E,), float(val), dtype=torch.float, device=edge_index.device)
                            else:
                                t = torch.full((E, self.edge_attr_dim), float(val), dtype=torch.float, device=edge_index.device)
                        elif isinstance(val, torch.Tensor):
                            t = val.to(edge_index.device)
                            if t.dim() == 0:
                                t = t.expand(E)
                            if t.dim() == 1:
                                if self.edge_attr_dim == 1 and t.size(0) != E:
                                    raise RuntimeError(f"edge_attr for {etype} len {t.size(0)} != expected {E}")
                                elif self.edge_attr_dim != 1:
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
                h = layer(h, edge_index_dict, edge_attr_dict=processed)
            else:
                h = layer(h, edge_index_dict)

            for k in list(h.keys()):
                h[k] = F.relu(self.dropout(h[k]))

        out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
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
        if self.decoder_type == "cosine":
            src_n = F.normalize(src_z, p=2, dim=-1)
            dst_n = F.normalize(dst_z, p=2, dim=-1)
            sim = F.cosine_similarity(src_n, dst_n)
            return sim * torch.exp(self.logit_scale)
        elif self.decoder_type == "mlp":
            e = torch.cat([src_z, dst_z], dim=-1)
            return self.decoder_mlp(e).view(-1)
        else:
            raise ValueError(f"Unknown decoder {self.decoder_type}")


# ---------- load_model: instantiate GATv2MiniModel and load ckpt ----------
def load_model(ckpt_path: str, data: HeteroData, args, device: torch.device, use_edge_attr=True, edge_attr_dim=1):
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dims = {n: data[n].x.size(1) for n in data.node_types}

    model = GATv2MiniModel(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        decoder=args.decoder if hasattr(args, "decoder") else "mlp",
        use_edge_attr=use_edge_attr,
        edge_attr_dim=edge_attr_dim
    ).to(device)

    # Attempt to extract state dict from various ckpt formats
    state = None
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # likely a state_dict already
            state = ckpt
        else:
            # fallback: maybe ckpt contains nested states
            # try to find something that looks like a state dict
            for k, v in ckpt.items():
                if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    state = v
                    break
    else:
        # ckpt might directly be state_dict
        state = ckpt

    if state is None:
        logger.warning("Couldn't interpret checkpoint structure; attempting to load as state_dict anyway")
        state = ckpt

    # strip module. prefix if present
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logger.warning("Missing keys when loading model: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading model: %s", unexpected)

    model.eval()
    logger.info("Loaded checkpoint from %s", ckpt_path)
    return model

def predict_new_phages(graph_path: str,
                       map_path: str,
                       ckpt_path: str,
                       taxid2species_tsv: Optional[str],
                       args,
                       out_tsv: str,
                       device: str = "cuda",
                       edge_type_weight_map: typing.Optional[dict] = None,
                       node_maps_path: str = "node_maps.json",
                       top_k: int = 10):
    """
    edge_type_weight_map: optional dict mapping etype tuple -> scalar weight, e.g. {('phage','infects','host'): 0.5}
    Priority for edge attributes:
      1) data[etype].edge_weight (if present)
      2) edge_type_weight_map scalar (if provided)
    If neither provided for an etype, that etype will have no edge_attr (model will run without it).
    """
    import os
    # Fix thread/numexpr warnings (putting here is safe; ideally set at program entry)
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    # reduce fragmentation risk
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:128")

    device_t = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    # 1. load graph + mapping
    data: HeteroData = torch.load(graph_path, map_location="cpu", weights_only=False)

    # map_path expected to be JSON mapping phage_id -> node_index (same as in training pipeline)
    with open(map_path, "r", encoding="utf-8") as f:
        phage_map = json.load(f)  # e.g. { "phageA": 123, ... }

    # 2. load model
    model = load_model(ckpt_path, data, args, device_t, use_edge_attr=True, edge_attr_dim=1)

    # 3. build full_x & full_edge_index_dict (on device_t for normal flow; may be moved to CPU later)
    full_x = {nt: data[nt].x.to(device_t) for nt in data.node_types}
    full_edge_index_dict = {
        et: data[et].edge_index.to(device_t)
        for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
    }

    # 4. Build global_edge_attr dict (on device_t)
    if edge_type_weight_map is None:
        edge_type_weight_map = {
            ('phage','infects','host'): 1.0,
            ('protein','similar','protein'): 1.0,
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
        if hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
            val = data[et].edge_weight.to(device_t)
            if val.dim() == 0:
                val = val.expand(E)
            global_edge_attr[et] = val
        elif et in edge_type_weight_map:
            w = float(edge_type_weight_map[et])
            global_edge_attr[et] = torch.full((E,), w, device=device_t, dtype=torch.float)
        else:
            # no edge_attr for this etype
            pass

    # 5. forward to get embeddings (with edge_attr_dict passed)
    # Use CPU-forward to avoid GATv2Conv creating huge temporary tensors on GPU and causing OOM.
    model_device_orig = next(model.parameters()).device
    forward_on_cpu = True  # compute whole-graph embeddings on CPU to avoid GPU OOM

    if forward_on_cpu:
        logger.info("Performing full-graph forward on CPU to avoid GPU OOM (may be slower).")
        # move model to cpu
        model_cpu = model.to("cpu")

        # Build CPU versions of inputs (use data to avoid moving already-on-gpu tensors back and forth)
        full_x_cpu = {nt: data[nt].x.to("cpu") for nt in data.node_types}
        full_edge_index_dict_cpu = {
            et: data[et].edge_index.to("cpu")
            for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
        }
        global_edge_attr_cpu = {et: t.to("cpu") for et, t in global_edge_attr.items()} if len(global_edge_attr) > 0 else None

        with torch.no_grad():
            h_dict_cpu = model_cpu(full_x_cpu, full_edge_index_dict_cpu, edge_attr_dict=global_edge_attr_cpu)

        # move model back to its original device
        model.to(model_device_orig)

        # keep embeddings on CPU (to avoid GPU peak memory blowup)
        phage_emb_all_cpu = h_dict_cpu['phage'].cpu()
        host_emb_all_cpu  = h_dict_cpu['host'].cpu()

        # provide a small dict for compatibility if other code expects h_dict
        h_dict = {'phage': phage_emb_all_cpu, 'host': host_emb_all_cpu}

        # cleanup large temporaries
        del h_dict_cpu, full_x_cpu, full_edge_index_dict_cpu, global_edge_attr_cpu
        torch.cuda.empty_cache()
        logger.info("CPU forward done: phage_emb shape %s, host_emb shape %s",
                    tuple(phage_emb_all_cpu.size()), tuple(host_emb_all_cpu.size()))
    else:
        logger.info("Performing forward on device (may OOM).")
        with torch.no_grad():
            h_dict = model(full_x, full_edge_index_dict, edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None)
        phage_emb_all_cpu = h_dict['phage'].cpu()
        host_emb_all_cpu  = h_dict['host'].cpu()
        del h_dict
        torch.cuda.empty_cache()

    # 6. host taxid -> species mapping
    host_taxids = data["host"].taxid.cpu().numpy().tolist() if hasattr(data["host"], "taxid") else []
    taxid2species = {}
    if taxid2species_tsv is not None:
        df_tax = pd.read_csv(taxid2species_tsv, sep="\t")
        # auto-detect columns
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

    # 8. predict for each new phage: score against all hosts and take top_k
    host_num = int(data["host"].num_nodes)
    host_ids = list(range(host_num))

    results = []
    k = int(top_k)

    # Use the CPU-resident embeddings (these were assigned above)
    phage_emb_all = phage_emb_all_cpu  # (N_phage, D) on CPU
    host_emb_all  = host_emb_all_cpu   # (N_host, D) on CPU

    # precompute scale (if cosine decoder). Move scale to device when used.
    scale_val = float(torch.exp(model.logit_scale).item()) if hasattr(model, "logit_scale") else 1.0

    # chunking parameters (tune if you have more/less GPU memory)
    # default host_chunk_size is conservative; increase if you have free GPU memory
    host_chunk_size = 65536

    # iterate phages from mapping
    for phage_id, phage_idx in phage_map.items():
        try:
            pid_int = int(phage_idx)
        except Exception:
            logger.warning("Invalid phage_idx %s for phage_id %s; skipping", phage_idx, phage_id)
            continue

        if pid_int < 0 or pid_int >= phage_emb_all.size(0):
            logger.warning("phage index %s for id %s out of bounds (0..%d). Skipping.",
                           phage_idx, phage_id, phage_emb_all.size(0)-1)
            continue

        # compute scores against all hosts in a memory-aware way
        if model.decoder_type == "cosine":
            # take phage vector (CPU) -> move to device for matmul
            ph_vec = phage_emb_all[pid_int].to(device_t).unsqueeze(0)  # (1, D) on device
            scores_chunks = []
            # iterate host chunks (move each chunk to device, compute, move back)
            for s in range(0, host_num, host_chunk_size):
                e = min(host_num, s + host_chunk_size)
                host_chunk = host_emb_all[s:e].to(device_t)  # (chunk, D)
                with torch.no_grad():
                    scores_chunk = (ph_vec @ host_chunk.t()).squeeze(0)  # (chunk,)
                    if scale_val != 1.0:
                        scores_chunk = scores_chunk * scale_val
                scores_chunks.append(scores_chunk.cpu())  # keep on CPU list
                # free GPU temp
                del host_chunk, scores_chunk
                torch.cuda.empty_cache()
            # concat CPU tensors -> apply sigmoid
            if len(scores_chunks) == 0:
                continue
            scores_tensor = torch.cat(scores_chunks, dim=0)  # CPU tensor of length host_num
            probs = torch.sigmoid(scores_tensor).numpy()     # numpy array
        else:
            # For non-cosine (mlp) decoder: we must call model.decode in chunks.
            # We'll create tensors of phage indices and host index chunks and call model.decode on device.
            # phage index repeated for each host chunk.
            phage_chunk_scores = []
            for s in range(0, host_num, host_chunk_size):
                e = min(host_num, s + host_chunk_size)
                host_idx_chunk = torch.arange(s, e, dtype=torch.long, device=device_t)
                ph_idx_repeat = torch.full((e - s,), pid_int, dtype=torch.long, device=device_t)
                with torch.no_grad():
                    # We need embeddings on device for decode: move corresponding CPU embeddings to device
                    # Build z_dict-like small dict with slices moved to device
                    z_dict = {
                        'phage': phage_emb_all[pid_int:pid_int+1].to(device_t),  # (1, D)
                        'host' : host_emb_all[s:e].to(device_t)                 # (chunk, D)
                    }
                    # decode expects (src_idx, dst_idx) local to these tensors; for (1,chunk) src_idx zeros
                    # but our decode implementation expects global indices into z_dict; easier: call with tensors
                    # matching our model.decode interface: src indices [0..] for the local phage slice
                    # We'll call model.decode with src_idx tensor of zeros and dst_idx 0..chunk-1
                    src_local = torch.zeros((e - s,), dtype=torch.long, device=device_t)
                    dst_local = torch.arange(0, e - s, dtype=torch.long, device=device_t)
                    scores_chunk = model.decode(z_dict, (src_local, dst_local), etype=relation)
                    # scores_chunk on device: move to cpu
                    phage_chunk_scores.append(scores_chunk.cpu())
                    del z_dict, src_local, dst_local, scores_chunk
                    torch.cuda.empty_cache()
            scores_tensor = torch.cat(phage_chunk_scores, dim=0)  # CPU tensor
            probs = torch.sigmoid(scores_tensor).numpy()

        # take top-k host indices by probability
        topk_idx = probs.argsort()[::-1][:k]
        for rank, hi in enumerate(topk_idx, 1):
            taxid, species = host_id2species.get(int(hi), (None, "NA"))
            results.append({
                "phage_id": phage_id,
                "phage_node_idx": int(pid_int),
                "host_node_idx": int(hi),
                "host_taxid": int(taxid) if isinstance(taxid, (int, np.integer)) else taxid,
                "host_species_name": species,
                "score": float(probs[int(hi)]),
                "rank": rank
            })

    # 9. debug printing of edge info
    logger.info("edge_index types: %s", list(full_edge_index_dict.keys()))
    logger.info("edge_attr types: %s", list(global_edge_attr.keys()))
    for et in full_edge_index_dict.keys():
        E = full_edge_index_dict[et].size(1)
        has_attr = et in global_edge_attr
        attr_shape = None if not has_attr else tuple(global_edge_attr[et].shape)
        logger.info("%s E=%d has_attr=%s attr_shape=%s", et, E, has_attr, attr_shape)

    # 10. save results to TSV
    df = pd.DataFrame(results)
    df.to_csv(out_tsv, sep="\t", index=False)
    logger.info("✅ saved predictions to %s (rows=%d)", out_tsv, len(df))

# ==== Example usage as script ====
if __name__ == "__main__":
    # Example args: adjust as needed
    class Args:
        hidden_dim = 512
        out_dim = 256
        n_layers = 2
        n_heads = 4
        dropout = 0.2
        decoder = "cosine"  # or "mlp" to match training
    args = Args()

    # Example call: change paths to your files
    predict_new_phages(
        graph_path="graph_with_newphages_ppedges.pt",
        map_path="merged_phage_mapping.json",
        ckpt_path="/home/wangjingyuan/wys/build_new_phage/best_GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg30_evl20_ph2_p0.5_1e-5_cos.pt",
        taxid2species_tsv="taxid_species.tsv",
        args=args,
        out_tsv="newphage_predictions_cherry_pp_out.tsv",
        device="cuda",
        edge_type_weight_map = {
                # 你指定的特殊权重：
                ('phage', 'infects', 'host'): 2.0,
                ('protein', 'similar', 'protein'): 1.0,

                # 其他边类型默认设为 1.0（中性影响）
                ('host', 'has_sequence', 'host_sequence'): 1.0,
                ('phage', 'interacts', 'phage'): 1.0,
                ('host', 'interacts', 'host'): 1.0,
                ('phage', 'encodes', 'protein'): 0.5,
                ('host', 'encodes', 'protein'): 0.5,
                ('host', 'belongs_to', 'taxonomy'): 1.0,
                ('taxonomy', 'related', 'taxonomy'): 1.0,
                ('phage', 'belongs_to', 'taxonomy'): 1.0,
            },   # or pass the same mapping used in training
        node_maps_path="node_maps_cluster_650.json",
        top_k=10
    )
