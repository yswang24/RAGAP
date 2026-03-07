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
from torch_geometric.nn import HGTConv,GATv2Conv,HeteroConv
from torch_geometric.loader import LinkNeighborLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def safe_torch_load(path: str) -> tuple[HeteroData, typing.Union[dict, None]]:
    """
    Loads a .pt file. Accepts either:
      - torch.save((data, split_edge), path)
      - torch.save(data, path) where data is HeteroData
      - torch.save({'data':data, 'split_edge': split_edge}, path)
    Returns (data, split_edge_or_None)
    """
    obj = torch.load(path, weights_only=False, map_location='cpu')
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], HeteroData):
        return obj[0], obj[1]
    if isinstance(obj, HeteroData):
        return obj, None
    if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], HeteroData):
        return obj['data'], obj.get('split_edge', None)
    raise RuntimeError("Unsupported .pt content. Please save torch.save((data, split_edge), path) or torch.save(data, path).")

def find_phage_host_splits(data: HeteroData, ext_splits: typing.Union[dict, None]) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
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
    for a, b, c in patterns:
        if hasattr(rec, a) and hasattr(rec, b) and hasattr(rec, c):
            A = getattr(rec, a); B = getattr(rec, b); C = getattr(rec, c)
            if isinstance(A, torch.Tensor) and A.dim() == 2 and A.size(0) == 2:
                return (A[0].cpu(), A[1].cpu()), (B[0].cpu(), B[1].cpu()), (C[0].cpu(), C[1].cpu())

    # fallback: maybe top-level attribute data.split_edge
    if hasattr(data, 'split_edge'):
        se = getattr(data, 'split_edge')
        if isinstance(se, dict) and 'train' in se and 'val' in se and 'test' in se:
            def pair(e):
                if isinstance(e, torch.Tensor) and e.dim() == 2 and e.size(0) == 2:
                    return e[0].cpu(), e[1].cpu()
                raise RuntimeError("split_edge format invalid")
            return pair(se['train']['edge']), pair(se['val']['edge']), pair(se['test']['edge'])

    raise RuntimeError("Cannot find phage-host splits inside data; please save splits or provide as ext_splits.")

# -------------------------
# Data Inspection and Fixing
# -------------------------
def check_node_counts(data: HeteroData) -> dict[str, typing.Union[int, None]]:
    node_counts = {}
    for ntype in data.node_types:
        if hasattr(data[ntype], "num_nodes"):
            n_nodes = int(data[ntype].num_nodes)
        elif 'x' in data[ntype]:
            n_nodes = int(data[ntype].x.size(0))
        else:
            n_nodes = None
        node_counts[ntype] = n_nodes
    return node_counts

def check_edge_bounds(data: HeteroData, node_counts: dict[str, typing.Union[int, None]]) -> list[tuple]:
    bad_items = []
    for etype, eidx in data.edge_index_dict.items():
        if eidx is None or eidx.numel() == 0:
            continue
        e_cpu = eidx.cpu()
        if e_cpu.dim() != 2 or e_cpu.size(0) != 2:
            continue
        src_max = int(e_cpu[0].max()); src_min = int(e_cpu[0].min())
        dst_max = int(e_cpu[1].max()); dst_min = int(e_cpu[1].min())
        src_type, _, dst_type = etype
        src_n = node_counts.get(src_type)
        dst_n = node_counts.get(dst_type)
        if src_n is not None and (src_min < 0 or src_max >= src_n):  # Assume non-negative indices
            bad_items.append(("edge_index", etype, "src", src_min, src_max, src_n))
        if dst_n is not None and (dst_min < 0 or dst_max >= dst_n):
            bad_items.append(("edge_index", etype, "dst", dst_min, dst_max, dst_n))
    return bad_items

def check_split_bounds(name: str, s_cpu: torch.Tensor, d_cpu: torch.Tensor, src_n: typing.Union[int, None], dst_n: typing.Union[int, None]) -> list[tuple]:
    if s_cpu.numel() == 0:
        return []
    s_arr = s_cpu.numpy()
    d_arr = d_cpu.numpy()
    smin, smax = int(s_arr.min()), int(s_arr.max())
    dmin, dmax = int(d_arr.min()), int(d_arr.max())
    bad = []
    if src_n is not None and (smin < 0 or smax >= src_n):
        bad.append((name, 'phage', smin, smax, src_n))
    if dst_n is not None and (dmin < 0 or dmax >= dst_n):
        bad.append((name, 'host', dmin, dmax, dst_n))
    return bad

def save_invalid_examples(out_dir: str, name: str, s_arr: np.ndarray, d_arr: np.ndarray, src_n: int, dst_n: int, limit: int = 200):
    invalid = []
    for i, (si, di) in enumerate(zip(s_arr.tolist(), d_arr.tolist())):
        if not (0 <= si < src_n and 0 <= di < dst_n):
            invalid.append((i, int(si), int(di)))
        if len(invalid) >= limit:
            break
    if invalid:
        fn = os.path.join(out_dir, f"invalid_{name}_examples.tsv")
        with open(fn, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["idx", "src_idx", "dst_idx"])
            writer.writerows(invalid)
        logger.info(f"Saved {len(invalid)} invalid {name} examples to {fn}")

def filter_pairs(s_cpu: torch.Tensor, d_cpu: torch.Tensor, src_n: int, dst_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    s_list, d_list = [], []
    for si, di in zip(s_cpu.tolist(), d_cpu.tolist()):
        if 0 <= si < src_n and 0 <= di < dst_n:
            s_list.append(int(si))
            d_list.append(int(di))
    return torch.tensor(s_list, dtype=torch.long), torch.tensor(d_list, dtype=torch.long)

def add_placeholder_ids(data: HeteroData, node_counts: dict[str, typing.Union[int, None]]):
    for ntype in ['phage', 'host']:
        if not hasattr(data[ntype], 'id') and node_counts.get(ntype) is not None:
            n = node_counts[ntype]
            logger.info(f"Adding numeric placeholder {ntype}.id = arange({n}) (dtype=int64)")
            data[ntype].id = torch.arange(n, dtype=torch.long)

def inspect_and_fix_data(
    data: HeteroData,
    train_src_cpu: torch.Tensor,
    train_dst_cpu: torch.Tensor,
    val_src_cpu: torch.Tensor,
    val_dst_cpu: torch.Tensor,
    test_src_cpu: torch.Tensor,
    test_dst_cpu: torch.Tensor,
    fix_enable: bool = True,
    out_dir: str = "debug_out"
) -> tuple[HeteroData, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    os.makedirs(out_dir, exist_ok=True)
    node_counts = check_node_counts(data)
    bad_items = check_edge_bounds(data, node_counts)

    phage_n = node_counts.get('phage')
    host_n = node_counts.get('host')

    bad_splits = []
    bad_splits += check_split_bounds("train", train_src_cpu, train_dst_cpu, phage_n, host_n)
    bad_splits += check_split_bounds("val", val_src_cpu, val_dst_cpu, phage_n, host_n)
    bad_splits += check_split_bounds("test", test_src_cpu, test_dst_cpu, phage_n, host_n)

    bad_items += bad_splits

    if bad_items:
        logger.warning("Found out-of-bounds items:")
        for b in bad_items:
            logger.warning(str(b))
        with open(os.path.join(out_dir, "bad_items.txt"), "w", encoding="utf-8") as fo:
            for b in bad_items:
                fo.write(str(b) + "\n")

    if bad_splits and phage_n is not None and host_n is not None:
        save_invalid_examples(out_dir, "train", train_src_cpu.numpy(), train_dst_cpu.numpy(), phage_n, host_n)
        save_invalid_examples(out_dir, "val", val_src_cpu.numpy(), val_dst_cpu.numpy(), phage_n, host_n)
        save_invalid_examples(out_dir, "test", test_src_cpu.numpy(), test_dst_cpu.numpy(), phage_n, host_n)

    if fix_enable and bad_splits and phage_n is not None and host_n is not None:
        logger.info("Auto-cleaning split edges (backing up originals to debug_out/)...")
        torch.save((train_src_cpu.clone(), train_dst_cpu.clone()), os.path.join(out_dir, "train_split_backup.pt"))
        torch.save((val_src_cpu.clone(), val_dst_cpu.clone()), os.path.join(out_dir, "val_split_backup.pt"))
        torch.save((test_src_cpu.clone(), test_dst_cpu.clone()), os.path.join(out_dir, "test_split_backup.pt"))

        train_src_cpu, train_dst_cpu = filter_pairs(train_src_cpu, train_dst_cpu, phage_n, host_n)
        val_src_cpu, val_dst_cpu = filter_pairs(val_src_cpu, val_dst_cpu, phage_n, host_n)
        test_src_cpu, test_dst_cpu = filter_pairs(test_src_cpu, test_dst_cpu, phage_n, host_n)

        logger.info("After filter train/val/test sizes: %d / %d / %d", train_src_cpu.size(0), val_src_cpu.size(0), test_src_cpu.size(0))
        torch.save((train_src_cpu, train_dst_cpu), os.path.join(out_dir, "train_split_fixed.pt"))
        torch.save((val_src_cpu, val_dst_cpu), os.path.join(out_dir, "val_split_fixed.pt"))
        torch.save((test_src_cpu, test_dst_cpu), os.path.join(out_dir, "test_split_fixed.pt"))

    add_placeholder_ids(data, node_counts)

    return data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu

# -------------------------
# Model
# -------------------------


# class HGTConvWithEdgeWeight(HGTConv):
#     def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
#         # 构造 edge_attr_dict，用于传给父类
#         edge_attr_dict = None
#         if edge_weight_dict is not None:
#             edge_attr_dict = {}
#             for etype, weight in edge_weight_dict.items():
#                 edge_index = edge_index_dict[etype]
#                 edge_attr_dict[etype] = weight.to(edge_index.device)
#         return super().forward(x_dict, edge_index_dict, edge_attr=edge_attr_dict)
#class GATv2MiniModel(nn.Module):
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
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.decoder_type = decoder
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout_p = dropout

        # --- 输入投影（把每种节点类型投到统一 hidden_dim） ---
        self.input_proj = nn.ModuleDict()
        for n in self.node_types:
            d = in_dims.get(n)
            if d is None:
                raise RuntimeError(f"Missing in_dim for node type {n}")
            self.input_proj[n] = nn.Linear(d, hidden_dim)

        # 我这里使用 concat=False（输出维度由 out_channels 决定），这样不要求 hidden_dim 能被 n_heads 整除。
        # 如果你想用 concat=True（输出 = out_per_head * heads），请把 concat_flag=True 并确保 hidden_dim % n_heads == 0。
        concat_flag = False
        out_channels = hidden_dim  # 当 concat=False 时，输出维度为 out_channels

        # --- 为每一层创建 HeteroConv（内部每个 edge-type 用 GATv2Conv） ---
        # 为保证子模块被正确注册，我们把每层的 GATv2Conv 放在一个 ModuleDict 中（键为字符串）
        self.edge_conv_md_list = nn.ModuleList()  # 每层的 ModuleDict（string keys）
        self.layers = nn.ModuleList()             # 每层对应的 HeteroConv

        for _ in range(n_layers):
            convs_md = nn.ModuleDict()
            # ModuleDict 的键必须是字符串，所以将 etype tuple 转为字符串 key
            for (src, rel, dst) in self.edge_types:
                key = f"{src}__{rel}__{dst}"
                convs_md[key] = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    heads=n_heads,
                    concat=concat_flag,
                    dropout=dropout
                )
            self.edge_conv_md_list.append(convs_md)

            # 构造 HeteroConv 的 mapping： (src,rel,dst) -> conv_module
            conv_map = {etype: convs_md[f"{etype[0]}__{etype[1]}__{etype[2]}"] for etype in self.edge_types}
            self.layers.append(HeteroConv(conv_map, aggr='sum'))

        self.dropout = nn.Dropout(dropout)

        # --- 输出投影 与 原 HGTMiniModel 保持一致 ---
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})

        # 与原 HGTMiniModel 保持一致的 edge MLP decoder（2*out_dim -> out_dim -> 1）
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

        if decoder == "mlp":
            self.decoder_mlp = self.edge_mlp

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        x_dict: {ntype: tensor[num_nodes_ntype, in_dim]}
        edge_index_dict: {(src, rel, dst): tensor(2, E)}
        returns: out_dict {ntype: tensor[num_nodes_ntype, out_dim_normalized]}
        """
        # 1) 输入线性投影
        h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

        # 2) 层叠 HeteroConv (每层内部每个 edge-type 用对应的 GATv2Conv)
        for layer in self.layers:
            h = layer(h, edge_index_dict)  # HeteroConv 会把每个 etype 的 conv 应用
            for k in list(h.keys()):
                h[k] = F.relu(self.dropout(h[k]))

        # 3) final projection + L2 归一化（保持原行为）
        out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
        return out

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        etype: tuple[str, str, str]
    ) -> torch.Tensor:
        # 保持 decode 接口与原来一致
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
            return F.cosine_similarity(src_z, dst_z)
        elif self.decoder_type == "mlp":
            e = torch.cat([src_z, dst_z], dim=-1)
            return self.decoder_mlp(e).view(-1)
        else:
            raise ValueError(f"Unknown decoder {self.decoder_type}")
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
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.decoder_type = decoder
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout_p = dropout

        # --- 输入投影（把每种节点类型投到统一 hidden_dim） ---
        self.input_proj = nn.ModuleDict()
        for n in self.node_types:
            d = in_dims.get(n)
            if d is None:
                raise RuntimeError(f"Missing in_dim for node type {n}")
            self.input_proj[n] = nn.Linear(d, hidden_dim)

        # 为保持输出维度为 hidden_dim，当 concat=True 时需要 hidden_dim % n_heads == 0
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads}) when concat=True. "
                             "Either change hidden_dim or n_heads, or set concat=False in GATv2Conv.")

        out_per_head = hidden_dim // n_heads  # 每个 head 的 out_channels

        # --- 为每一层创建 HeteroConv（内部每个 edge-type 用 GATv2Conv） ---
        # 为了正确注册模块，我们把每层的 GATv2Conv 集合放到一个 ModuleDict 中并注册到 self.edge_conv_md_list（ModuleList）
        self.edge_conv_md_list = nn.ModuleList()  # list of ModuleDicts (one per layer)
        self.layers = nn.ModuleList()             # list of HeteroConv (one per layer)

        for _ in range(n_layers):
            convs_md = nn.ModuleDict()
            # ModuleDict 的键必须是字符串，所以将 etype tuple 转为字符串 key
            for (src, rel, dst) in self.edge_types:
                key = f"{src}__{rel}__{dst}"
                convs_md[key] = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=out_per_head,   # concat=True -> 输出维度 = out_per_head * heads = hidden_dim
                    heads=n_heads,
                    dropout=dropout,
                    concat=True
                )
            # 将 convs_md 注册到模块列表中（以保证参数被注册）
            self.edge_conv_md_list.append(convs_md)

            # 构造 HeteroConv 需要一个 mapping: etype_tuple -> module
            conv_map = {etype: convs_md[f"{etype[0]}__{etype[1]}__{etype[2]}"] for etype in self.edge_types}
            self.layers.append(HeteroConv(conv_map, aggr='sum'))

        self.dropout = nn.Dropout(dropout)

        # --- 输出投影 与 原 HGTMiniModel 保持一致 ---
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})

        # 与原 HGTMiniModel 保持一致的 edge MLP decoder（2*out_dim -> out_dim -> 1）
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

        # decoder mlp alias（如果用户选 mlp）
        if decoder == "mlp":
            self.decoder_mlp = self.edge_mlp

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        x_dict: {ntype: tensor[num_nodes_ntype, in_dim]}
        edge_index_dict: {(src, rel, dst): tensor(2, E)}
        returns: out_dict {ntype: tensor[num_nodes_ntype, out_dim_normalized]}
        """
        # 1) 输入线性投影
        h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

        # 2) 层叠 HeteroConv (每层内部每个 edge-type 用对应的 GATv2Conv)
        for layer in self.layers:
            # HeteroConv 会把每个 etype 的 conv 应用到对应的 (src, dst, edge_index)
            h = layer(h, edge_index_dict)
            # 激活 + dropout
            for k in list(h.keys()):
                h[k] = F.relu(self.dropout(h[k]))

        # 3) final projection + L2 归一化（保持原行为）
        out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
        return out

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        etype: tuple[str, str, str]
    ) -> torch.Tensor:
        # 保持 decode 接口与原来一致
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
            # 余弦相似度
            return F.cosine_similarity(src_z, dst_z)
        elif self.decoder_type == "mlp":
            e = torch.cat([src_z, dst_z], dim=-1)
            return self.decoder_mlp(e).view(-1)
        else:
            raise ValueError(f"Unknown decoder {self.decoder_type}")
    def __init__(self, metadata, in_dims, hidden_dim, out_dim, n_layers=2, n_heads=4, dropout=0.2, decoder="mlp"):
        super().__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.decoder_type = decoder
        self.hidden_dim = hidden_dim

        # 每种节点类型独立的输入投影
        self.input_proj = nn.ModuleDict({
            n: nn.Linear(in_dims[n], hidden_dim) for n in self.node_types
        })

        # 多层 HeteroConv，每层内部是 GATv2Conv
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            convs = nn.ModuleDict()
            for (src, rel, dst) in self.edge_types:
                convs[(src, rel, dst)] = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    concat=True
                )
            self.layers.append(HeteroConv(convs, aggr='sum'))

        # 输出投影层
        self.final_proj = nn.ModuleDict({
            n: nn.Linear(hidden_dim, out_dim) for n in self.node_types
        })

        # 解码器
        if decoder == "mlp":
            self.decoder_mlp = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, 1)
            )

    def forward(self, x_dict, edge_index_dict):
        # 1️⃣ 输入投影
        x_dict = {k: F.relu(self.input_proj[k](x)) for k, x in x_dict.items()}

        # 2️⃣ GATv2 层堆叠
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # 3️⃣ 输出层
        out_dict = {}
        for n, x in x_dict.items():
            out_dict[n] = self.final_proj[n](x)
            out_dict[n] = F.normalize(out_dict[n], p=2, dim=-1)

        return out_dict

    def decode(self, h_dict, edge_index, etype):
        src, dst = edge_index
        src_type, _, dst_type = etype
        h_src = h_dict[src_type][src]
        h_dst = h_dict[dst_type][dst]

        if self.decoder_type == "cosine":
            return (h_src * h_dst).sum(dim=-1)
        else:
            h_cat = torch.cat([h_src, h_dst], dim=-1)
            return self.decoder_mlp(h_cat).view(-1)

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

        out = {k: self.final_proj[k](v) for k, v in h.items()}
        #out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}#归一化
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
#         decoder: str = "mlp",
#         use_edge_attr: bool = False,     # <- 新增：是否启用 edge_attr/edge_weight 支持
#         edge_attr_dim: int = 1,          # <- 新增：edge_attr 的维度（1 表示标量权重）
#     ):
#         super().__init__()
#         self.metadata = metadata
#         self.node_types, self.edge_types = metadata
#         self.decoder_type = decoder
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.n_heads = n_heads
#         self.dropout_p = dropout
#         self.use_edge_attr = use_edge_attr
#         self.edge_attr_dim = edge_attr_dim

#         # 输入投影
#         self.input_proj = nn.ModuleDict()
#         for n in self.node_types:
#             d = in_dims.get(n)
#             if d is None:
#                 raise RuntimeError(f"Missing in_dim for node type {n}")
#             self.input_proj[n] = nn.Linear(d, hidden_dim)

#         # 这里使用 concat=False（避免 hidden_dim 必须被 n_heads 整除）。
#         concat_flag = False
#         out_channels = hidden_dim

#         # 每层的 GATv2Conv 集合和对应的 HeteroConv
#         self.edge_conv_md_list = nn.ModuleList()
#         self.layers = nn.ModuleList()
#         for _ in range(n_layers):
#             convs_md = nn.ModuleDict()
#             for (src, rel, dst) in self.edge_types:
#                 key = f"{src}__{rel}__{dst}"
#                 # 如果启用了 edge_attr，设置 edge_dim=edge_attr_dim
#                 if self.use_edge_attr:
#                     convs_md[key] = GATv2Conv(
#                         in_channels=hidden_dim,
#                         out_channels=out_channels,
#                         heads=n_heads,
#                         concat=concat_flag,
#                         dropout=dropout,
#                         edge_dim=self.edge_attr_dim
#                     )
#                 else:
#                     convs_md[key] = GATv2Conv(
#                         in_channels=hidden_dim,
#                         out_channels=out_channels,
#                         heads=n_heads,
#                         concat=concat_flag,
#                         dropout=dropout
#                     )
#             self.edge_conv_md_list.append(convs_md)
#             conv_map = {etype: convs_md[f"{etype[0]}__{etype[1]}__{etype[2]}"] for etype in self.edge_types}
#             self.layers.append(HeteroConv(conv_map, aggr='sum'))

#         self.dropout = nn.Dropout(dropout)

#         # final proj + decoder (保持原来接口)
#         self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
#         self.edge_mlp = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))
#         if decoder == "mlp":
#             self.decoder_mlp = self.edge_mlp

#     def forward(
#         self,
#         x_dict: dict[str, torch.Tensor],
#         edge_index_dict: dict[tuple, torch.Tensor],
#         edge_attr_dict: typing.Optional[dict] = None,   # <- 新增：etype -> tensor/float 的映射
#     ) -> dict[str, torch.Tensor]:
#         """
#         edge_attr_dict 可以是：
#           - { (src,rel,dst): Tensor(num_edges, ) }  或 Tensor(num_edges, edge_attr_dim)
#           - { (src,rel,dst): float } （会自动扩展为长度为 num_edges 的 Tensor）
#           - 可以只为部分 etype 提供值
#         """
#         # 1) input proj
#         h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

#         # 2) for each layer, prepare per-edge-type edge_attr mapping (if provided)
#         for layer in self.layers:
#             if self.use_edge_attr and edge_attr_dict is not None:
#                 # build processed_edge_attr: mapping from etype tuple -> tensor or None
#                 processed = {}
#                 for etype, edge_index in edge_index_dict.items():
#                     E = edge_index.size(1)
#                     if etype in edge_attr_dict:
#                         val = edge_attr_dict[etype]
#                         # scalar -> expand
#                         if isinstance(val, (float, int)):
#                             t = torch.full((E,), float(val), dtype=torch.float, device=edge_index.device)
#                         elif isinstance(val, torch.Tensor):
#                             t = val.to(edge_index.device)
#                             # allow (E,) or (E, edge_attr_dim)
#                             if t.dim() == 1:
#                                 # shape (E,) -> ok if edge_attr_dim==1
#                                 if self.edge_attr_dim != 1:
#                                     # expand to (E, edge_attr_dim) by repeating (not ideal but fallback)
#                                     t = t.view(-1, 1).repeat(1, self.edge_attr_dim)
#                             elif t.dim() == 2:
#                                 # ensure second dim matches edge_attr_dim
#                                 if t.size(1) != self.edge_attr_dim:
#                                     raise RuntimeError(f"edge_attr for {etype} has dim {t.size(1)} but model expects {self.edge_attr_dim}")
#                             else:
#                                 raise RuntimeError(f"edge_attr for {etype} must be 1D or 2D tensor")
#                         else:
#                             raise RuntimeError(f"Unsupported edge_attr type for {etype}: {type(val)}")
#                         processed[etype] = t
#                     else:
#                         # not provided -> do not include key (HeteroConv will call conv without that extra arg)
#                         pass

#                 # HeteroConv accepts additional keyword args where values are dicts keyed by etype
#                 # The key name must match the forward argument name of the child convs, which for GATv2Conv is `edge_attr`.
#                 h = layer(h, edge_index_dict, edge_attr=processed)
#             else:
#                 # default behavior (no edge attributes)
#                 h = layer(h, edge_index_dict)

#             # activation + dropout
#             for k in list(h.keys()):
#                 h[k] = F.relu(self.dropout(h[k]))

#         out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
#         return out

#     def decode(
#         self,
#         z_dict: dict[str, torch.Tensor],
#         edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
#         etype: tuple[str, str, str]
#     ) -> torch.Tensor:
#         # 保持原来 decode 接口
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
#             return F.cosine_similarity(src_z, dst_z)
#         elif self.decoder_type == "mlp":
#             e = torch.cat([src_z, dst_z], dim=-1)
#             return self.decoder_mlp(e).view(-1)
#         else:
#             raise ValueError(f"Unknown decoder {self.decoder_type}")
# -------------------------
# Metrics (full-graph eval)
# -------------------------
@torch.no_grad()
# def compute_metrics_fullgraph(
#     model: HGTMiniModel,
#     data: HeteroData,
#     train_pairs: tuple[torch.Tensor, torch.Tensor],
#     val_pairs: tuple[torch.Tensor, torch.Tensor],
#     test_pairs: tuple[torch.Tensor, torch.Tensor],
#     relation: tuple[str, str, str],
#     eval_device: str = 'cpu',
#     eval_neg_ratio: int = 1,
#     k_list: tuple[int, ...] = (1, 5, 10),
#     host_id2taxid: typing.Union[np.ndarray, None] = None,
#     taxid2species: typing.Union[dict[int, str], None] = None,
#     save_path: Optional[str] = None,   # ⭐ 新增：保存 top-k 文件
#    # ⭐ 新增：保存 top-k 文件
#     top_k: int = 5                # ⭐ 新增：top-k host
# ) -> tuple[tuple[float, float, dict[int, float]],
#            tuple[float, float, dict[int, float]],
#            tuple[float, float, dict[int, float]]]:
#     """
#     Compute full-graph metrics (AUC, MRR, Hits@K) and optionally save top-k predictions.
#     """

#     from collections import defaultdict
#     import numpy as np
#     import torch
#     from sklearn.metrics import roc_auc_score
#     import pandas as pd
#     import json
#     import logging

#     logger = logging.getLogger(__name__)

#     # ======================
#     # Load phage/host idx -> real id
#     # ======================
#     try:
#         with open("node_maps_RBP.json", "r", encoding="utf-8") as f:
#             node_maps = json.load(f)
#         phage_map = node_maps.get("phage_map", {})
#         host_map = node_maps.get("host_map", {})
#         phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
#         host_idx2id = {int(v): str(k) for k, v in host_map.items()}
#     except FileNotFoundError:
#         logger.warning("node_maps.json not found, using index as ID")
#         phage_idx2id = {}
#         host_idx2id = {}

#     # ======================
#     # 设备准备
#     # ======================
#     orig_device = next(model.parameters()).device
#     moved_model = False
#     if orig_device != torch.device(eval_device):
#         model.to(eval_device)
#         moved_model = True

#     try:
#         data_eval_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
#         edge_index_dict_eval = {
#             et: data[et].edge_index.to(eval_device)
#             for et in data.edge_types
#             if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
#         }

#         model.eval()
#         out = model(data_eval_x, edge_index_dict_eval)
#         n_hosts = out['host'].size(0)

#         # ======================
#         # 辅助函数
#         # ======================
#         def hostid2species(hid: int) -> str:
#             if host_id2taxid is None or taxid2species is None:
#                 return f"unknown_{hid}"
#             taxid = int(host_id2taxid[hid])
#             return taxid2species.get(taxid, f"unknown_{taxid}")

#         def build_pos_map(pairs: tuple[torch.Tensor, torch.Tensor]) -> defaultdict[int, set[int]]:
#             pos_map = defaultdict(set)
#             src_cpu, dst_cpu = pairs
#             for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
#                 pos_map[int(s)].add(int(d))
#             return pos_map

#         train_pos_map = build_pos_map(train_pairs)
#         val_pos_map = build_pos_map(val_pairs)
#         test_pos_map = build_pos_map(test_pairs)

#         # ======================
#         # AUC 计算
#         # ======================
#         def compute_auc_for_pairs(pairs: tuple[torch.Tensor, torch.Tensor], pos_map: defaultdict[int, set[int]]) -> float:
#             src_cpu, dst_cpu = pairs
#             if src_cpu.numel() == 0:
#                 return float('nan')

#             pos_scores, neg_scores_list = [], []
#             for s_idx, d_idx in zip(src_cpu.tolist(), dst_cpu.tolist()):
#                 if not (0 <= d_idx < n_hosts):
#                     continue

#                 pos_score = torch.sigmoid(
#                     model.decode(out, (torch.tensor([s_idx], device=eval_device),
#                                        torch.tensor([d_idx], device=eval_device)),
#                                  etype=relation)
#                 ).item()
#                 pos_scores.append(pos_score)

#                 # 负样本
#                 known = pos_map.get(s_idx, set())
#                 mask = torch.ones(n_hosts, dtype=torch.bool, device=eval_device)
#                 for oth in known:
#                     if 0 <= oth < n_hosts:
#                         mask[oth] = False
#                 mask[d_idx] = False
#                 cand_neg_idx = mask.nonzero(as_tuple=True)[0]
#                 if cand_neg_idx.numel() > 0:
#                     sample_size = min(eval_neg_ratio * len(known), cand_neg_idx.numel())
#                     neg_idx = cand_neg_idx[torch.randperm(cand_neg_idx.numel(), device=eval_device)[:sample_size]]
#                     neg_scores = torch.sigmoid(
#                         model.decode(out, (torch.full((len(neg_idx),), s_idx, device=eval_device),
#                                            neg_idx),
#                                      etype=relation)
#                     ).cpu().numpy()
#                     neg_scores_list.append(neg_scores)

#             if len(pos_scores) == 0 or len(neg_scores_list) == 0:
#                 return float('nan')

#             y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(sum(len(n) for n in neg_scores_list))])
#             y_score = np.concatenate([np.array(pos_scores), np.concatenate(neg_scores_list)])
#             return roc_auc_score(y_true, y_score)

#         # ======================
#         # MRR & Hits@K
#         # ======================
#         def compute_rank_metrics(pairs: tuple[torch.Tensor, torch.Tensor], save_path: Optional[str] = None):
#             src_cpu, dst_cpu = pairs
#             if src_cpu.numel() == 0:
#                 return 0.0, {k: 0.0 for k in k_list}

#             ph2hosts = defaultdict(list)
#             for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
#                 ph2hosts[s].append(d)

#             hits = {k: 0 for k in k_list}
#             rr_sum, total_q = 0.0, len(ph2hosts)

#             prediction_rows = []

#             for ph_idx, true_ds in ph2hosts.items():
#                 cand_hosts = torch.arange(n_hosts, device=eval_device)
#                 scores_tensor = torch.sigmoid(
#                     model.decode(out, (torch.full((n_hosts,), ph_idx, device=eval_device),
#                                        cand_hosts),
#                                  etype=relation)
#                 )

#                 scores_np = scores_tensor.cpu().numpy()
#                 topk_idx = scores_np.argsort()[::-1][:top_k]

#                 phage_real_id = phage_idx2id.get(ph_idx, str(ph_idx))
#                 true_species = {hostid2species(h) for h in true_ds}

#                 # 保存 top-k host
#                 for rank, h in enumerate(topk_idx, 1):
#                     host_real_id = host_idx2id.get(h, str(h))
#                     host_species_name = hostid2species(h)
#                     score = scores_np[h]
#                     prediction_rows.append({
#                         "phage_id": phage_real_id,
#                         "rank": rank,
#                         "host_id": host_real_id,
#                         "host_species": host_species_name,
#                         "score": score
#                     })

#                 # MRR / Hits@K
#                 rank_val = None
#                 for pos, h in enumerate(topk_idx, 1):
#                     if hostid2species(h) in true_species:
#                         rank_val = pos
#                         break
#                 if rank_val is None:
#                     rank_val = top_k + 1
#                 rr_sum += 1.0 / rank_val

#                 for k in k_list:
#                     if any(hostid2species(h) in true_species for h in topk_idx[:k]):
#                         hits[k] += 1

#             if save_path is not None and len(prediction_rows) > 0:
#                 pd.DataFrame(prediction_rows).to_csv(save_path, sep="\t", index=False)

#             mrr = rr_sum / total_q if total_q > 0 else 0.0
#             hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}
#             return mrr, hits_at

#         # ======================
#         # 计算指标
#         # ======================
#         train_auc = compute_auc_for_pairs(train_pairs, train_pos_map)
#         train_mrr, train_hits = compute_rank_metrics(train_pairs, save_path=f"{save_path}_train_topk.tsv" if save_path else None)

#         val_auc = compute_auc_for_pairs(val_pairs, val_pos_map)
#         val_mrr, val_hits = compute_rank_metrics(val_pairs, save_path=f"{save_path}_val_topk.tsv" if save_path else None)

#         test_auc = compute_auc_for_pairs(test_pairs, test_pos_map)
#         test_mrr, test_hits = compute_rank_metrics(test_pairs, save_path=f"{save_path}_test_topk.tsv" if save_path else None)

#         return (train_auc, train_mrr, train_hits), (val_auc, val_mrr, val_hits), (test_auc, test_mrr, test_hits)

#     finally:
#         if moved_model:
#             model.to(orig_device)
#         torch.cuda.empty_cache() if torch.cuda.is_available() else None
def compute_metrics_fullgraph(
    model: HGTMiniModel,
    #model: GATv2MiniModel,
    data: HeteroData,
    train_pairs: tuple[torch.Tensor, torch.Tensor],
    val_pairs: tuple[torch.Tensor, torch.Tensor],
    test_pairs: tuple[torch.Tensor, torch.Tensor],
    relation: tuple[str, str, str],
    eval_device: str = 'cpu',
    eval_neg_ratio: int = 10,
    k_list: tuple[int, ...] = (1, 5, 10),
    host_id2taxid: typing.Union[np.ndarray, None] = None,
    taxid2species: typing.Union[dict[int, str], None] = None,
    save_path: Optional[str] = None,
    top_k: int = 10
) -> tuple[tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]]]:
    """
    Compute full-graph metrics (AUC, MRR, Hits@K) and optionally save top-k predictions.
    """

    from collections import defaultdict
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    import json
    import logging

    logger = logging.getLogger(__name__)

    # ======================
    # Load phage/host idx -> real id
    # ======================
    try:
        with open("node_maps_RBP.json", "r", encoding="utf-8") as f:
            node_maps = json.load(f)
        phage_map = node_maps.get("phage_map", {})
        host_map = node_maps.get("host_map", {})
        phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
        host_idx2id = {int(v): str(k) for k, v in host_map.items()}
    except FileNotFoundError:
        logger.warning("node_maps.json not found, using index as ID")
        phage_idx2id = {}
        host_idx2id = {}

    # ======================
    # 设备准备
    # ======================
    orig_device = next(model.parameters()).device
    moved_model = False
    if orig_device != torch.device(eval_device):
        model.to(eval_device)
        moved_model = True

    try:
        data_eval_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
        edge_index_dict_eval = {
            et: data[et].edge_index.to(eval_device)
            for et in data.edge_types
            if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
        }

        model.eval()
        out = model(data_eval_x, edge_index_dict_eval)
        n_hosts = out['host'].size(0)

        # ======================
        # 辅助函数
        # ======================
        def hostid2species(hid: int) -> str:
            if host_id2taxid is None or taxid2species is None:
                return f"unknown_{hid}"
            taxid = int(host_id2taxid[hid])
            return taxid2species.get(taxid, f"unknown_{taxid}")

        def build_pos_map(pairs: tuple[torch.Tensor, torch.Tensor]) -> defaultdict[int, set[int]]:
            pos_map = defaultdict(set)
            src_cpu, dst_cpu = pairs
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                pos_map[int(s)].add(int(d))
            return pos_map

        train_pos_map = build_pos_map(train_pairs)
        val_pos_map = build_pos_map(val_pairs)
        test_pos_map = build_pos_map(test_pairs)

        # ======================
        # ensure top_k covers requested k_list
        # ======================
        effective_top_k = max(top_k, max(k_list))

        # ======================
        # AUC 计算 (修正后的负采样)
        # ======================
        def compute_auc_for_pairs(pairs: tuple[torch.Tensor, torch.Tensor], pos_map: defaultdict[int, set[int]]) -> float:
            src_cpu, dst_cpu = pairs
            if src_cpu.numel() == 0:
                return float('nan')

            pos_scores = []
            neg_scores_list = []
            all_hosts_idx = torch.arange(n_hosts, device=eval_device)

            for s_idx, d_idx in zip(src_cpu.tolist(), dst_cpu.tolist()):
                if not (0 <= d_idx < n_hosts):
                    continue

                # positive score
                pos_score = torch.sigmoid(
                    model.decode(out, (torch.tensor([s_idx], device=eval_device),
                                       torch.tensor([d_idx], device=eval_device)),
                                 etype=relation)
                ).item()
                pos_scores.append(pos_score)

                # negative candidates
                known = pos_map.get(s_idx, set())
                mask = torch.ones(n_hosts, dtype=torch.bool, device=eval_device)
                for oth in known:
                    if 0 <= oth < n_hosts:
                        mask[oth] = False
                if 0 <= d_idx < n_hosts:
                    mask[d_idx] = False

                cand_neg_idx = mask.nonzero(as_tuple=True)[0]
                if cand_neg_idx.numel() == 0:
                    continue

                # sample up to eval_neg_ratio negatives
                sample_size = min(int(eval_neg_ratio), int(cand_neg_idx.numel()))
                if sample_size <= 0:
                    continue
                perm = torch.randperm(cand_neg_idx.numel(), device=eval_device)[:sample_size]
                neg_idx = cand_neg_idx[perm]

                neg_scores = torch.sigmoid(
                    model.decode(out, (torch.full((neg_idx.numel(),), s_idx, device=eval_device),
                                       neg_idx),
                                 etype=relation)
                ).cpu().numpy()
                neg_scores_list.append(neg_scores)

            if len(pos_scores) == 0 or len(neg_scores_list) == 0:
                return float('nan')

            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(sum(len(n) for n in neg_scores_list))])
            y_score = np.concatenate([np.array(pos_scores), np.concatenate(neg_scores_list)])
            try:
                return float(roc_auc_score(y_true, y_score))
            except Exception:
                return float('nan')

        # ======================
        # MRR & Hits@K
        # ======================
        def compute_rank_metrics(pairs: tuple[torch.Tensor, torch.Tensor], save_path: Optional[str] = None):
            src_cpu, dst_cpu = pairs
            if src_cpu.numel() == 0:
                return 0.0, {k: 0.0 for k in k_list}

            ph2hosts = defaultdict(list)
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                ph2hosts[int(s)].append(int(d))

            hits = {k: 0 for k in k_list}
            rr_sum, total_q = 0.0, len(ph2hosts)

            prediction_rows = []
            cand_hosts = torch.arange(n_hosts, device=eval_device)

            for ph_idx, true_ds in ph2hosts.items():
                scores_tensor = torch.sigmoid(
                    model.decode(out, (torch.full((n_hosts,), ph_idx, device=eval_device),
                                       cand_hosts),
                                 etype=relation)
                )

                scores_np = scores_tensor.cpu().numpy()
                topk_idx = scores_np.argsort()[::-1][:effective_top_k]

                phage_real_id = phage_idx2id.get(ph_idx, str(ph_idx))
                true_species = {hostid2species(h) for h in true_ds}

                for rank, h in enumerate(topk_idx, 1):
                    host_real_id = host_idx2id.get(int(h), str(int(h)))
                    host_species_name = hostid2species(int(h))
                    score = float(scores_np[int(h)])
                    prediction_rows.append({
                        "phage_id": phage_real_id,
                        "rank": rank,
                        "host_id": host_real_id,
                        "host_species": host_species_name,
                        "score": score
                    })

                # reciprocal rank
                rank_val = None
                for pos, h in enumerate(topk_idx, 1):
                    if hostid2species(int(h)) in true_species:
                        rank_val = pos
                        break
                if rank_val is None:
                    rank_val = effective_top_k + 1
                rr_sum += 1.0 / rank_val

                for k in k_list:
                    if k <= effective_top_k:
                        if any(hostid2species(int(h)) in true_species for h in topk_idx[:k]):
                            hits[k] += 1

            if save_path is not None and len(prediction_rows) > 0:
                pd.DataFrame(prediction_rows).to_csv(save_path, sep="\t", index=False)

            mrr = rr_sum / total_q if total_q > 0 else 0.0
            hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}
            return mrr, hits_at

        # ======================
        # 计算指标
        # ======================
        train_auc = compute_auc_for_pairs(train_pairs, train_pos_map)
        train_mrr, train_hits = compute_rank_metrics(train_pairs, save_path=f"{save_path}_train_topk.tsv" if save_path else None)

        val_auc = compute_auc_for_pairs(val_pairs, val_pos_map)
        val_mrr, val_hits = compute_rank_metrics(val_pairs, save_path=f"{save_path}_val_topk.tsv" if save_path else None)

        test_auc = compute_auc_for_pairs(test_pairs, test_pos_map)
        test_mrr, test_hits = compute_rank_metrics(test_pairs, save_path=f"{save_path}_test_topk.tsv" if save_path else None)

        return (train_auc, train_mrr, train_hits), (val_auc, val_mrr, val_hits), (test_auc, test_mrr, test_hits)

    finally:
        if moved_model:
            model.to(orig_device)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# -------------------------
# Save Predictions
# -------------------------
def save_predictions(
    model: HGTMiniModel,
    #model: GATv2MiniModel,
    data: HeteroData,
    test_src_cpu: torch.Tensor,
    test_dst_cpu: torch.Tensor,
    relation: tuple[str, str, str],
    eval_device: torch.device,
    host_id2taxid: typing.Union[np.ndarray, None] = None,
    taxid2species: typing.Union[dict[int, str], None] = None,
    output_file: str = "phage_prediction_results.tsv",
    top_k: int = 5 ,# ⭐ 新增：控制输出 top-k
    k_list=(1, 5, 10)
):
    orig_dev = next(model.parameters()).device
    moved = False
    if orig_dev != eval_device:
        model.to(eval_device)
        moved = True

    try:
        full_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
        full_edge_index_dict = {
            et: data[et].edge_index.to(eval_device)
            for et in data.edge_types if hasattr(data[et], 'edge_index')
        }

        model.eval()
        out_full = model(full_x, full_edge_index_dict)

        edge_index_full = torch.stack(
            [test_src_cpu.to(eval_device), test_dst_cpu.to(eval_device)], dim=0
        )
        scores_tensor = torch.sigmoid(model.decode(out_full, edge_index_full, etype=relation))
        scores = scores_tensor.cpu().detach().numpy()

        # Load node maps for ID reversal
        try:
            with open("node_maps_RBP.json", "r", encoding="utf-8") as f:
                node_maps = json.load(f)
            phage_map = node_maps.get("phage_map", {})
            host_map = node_maps.get("host_map", {})
            phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
            host_idx2id = {int(v): str(k) for k, v in host_map.items()}
        except FileNotFoundError:
            logger.warning("node_maps.json not found, using index as ID")
            phage_idx2id = {}
            host_idx2id = {}

        phage_ids = [phage_idx2id.get(int(nid), str(nid)) for nid in test_src_cpu.tolist()]
        host_ids = [host_idx2id.get(int(nid), str(nid)) for nid in test_dst_cpu.tolist()]

        host_species = ["NA"] * len(host_ids)
        if host_id2taxid is not None and taxid2species is not None:
            host_species = [
                taxid2species.get(int(host_id2taxid[int(nid)]), "NA")
                for nid in test_dst_cpu.tolist()
            ]

        # ⭐ 核心：按 phage 分组，取 top-k host
        phage2preds = {}
        for pid, hid, hs, s in zip(phage_ids, host_ids, host_species, scores):
            if pid not in phage2preds:
                phage2preds[pid] = []
            phage2preds[pid].append((hid, hs, s))

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["phage_id", "rank", "host_id", "host_species", "score"])

            for pid, preds in phage2preds.items():
                preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]
                for rank, (hid, hs, s) in enumerate(preds_sorted, 1):
                    writer.writerow([pid, rank, hid, hs, float(s)])

        logger.info(f"Top-{top_k} Phage-host prediction results saved to {output_file}")

    finally:
        if moved:
            model.to(orig_dev)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# -------------------------
# Training entry
# -------------------------
def parse_args() -> argparse.Namespace:
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
def bpr_loss(pos_scores, neg_scores):

    """
    BPR loss:
    pos_scores: [num_pos]
    neg_scores: [num_pos * neg_ratio] (flattened)
    """
    # 如果是一正多负 -> reshape
    num_pos = pos_scores.size(0)
    neg_ratio = neg_scores.size(0) // num_pos
    neg_scores = neg_scores.view(num_pos, neg_ratio)  # [num_pos, neg_ratio]

    # 广播比较正负
    diff = pos_scores.unsqueeze(1) - neg_scores  # [num_pos, neg_ratio]
    return -torch.mean(F.logsigmoid(diff))
def softmax_ce_loss(phage_emb_batch, host_emb_batch, pos_phage_local_idx, pos_host_local_idx, tau=1.0):
    """
    Compute softmax cross-entropy loss for ranking:
      - phage_emb_batch: (num_phage_in_batch, D)
      - host_emb_batch:  (num_host_in_batch,  D)
      - pos_phage_local_idx: LongTensor shape (num_pos,) -- local indices into phage_emb_batch
      - pos_host_local_idx:  LongTensor shape (num_pos,) -- local indices into host_emb_batch (labels)
    Returns scalar loss (on same device).
    Loss = CrossEntropy( logits = (phage_vec @ host_emb_batch.T) / tau, labels = pos_host_local_idx )
    Note: returns mean over num_pos
    """
    device = phage_emb_batch.device
    # select phage vectors corresponding to positive pairs
    phage_vecs = phage_emb_batch[pos_phage_local_idx]  # (num_pos, D)
    # logits: (num_pos, num_host_in_batch)
    logits = torch.matmul(phage_vecs, host_emb_batch.t())  # (num_pos, host_n)
    if tau != 1.0:
        logits = logits / float(tau)
    # labels are pos_host_local_idx (already in range [0, host_n-1])
    labels = pos_host_local_idx.to(device).long()
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss
def main():
    import pandas as pd
    import random

    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    eval_device = torch.device(args.eval_device)

    logger.info("Loading data: %s", args.data_pt)
    data, split_edge = safe_torch_load(args.data_pt)
    logger.info("Data metadata: %s", data.metadata())

    train_pair, val_pair, test_pair = find_phage_host_splits(data, split_edge)
    train_src_cpu, train_dst_cpu = train_pair
    val_src_cpu, val_dst_cpu = val_pair
    test_src_cpu, test_dst_cpu = test_pair
    logger.info("Train/Val/Test counts: %d / %d / %d", train_src_cpu.size(0), val_src_cpu.size(0), test_src_cpu.size(0))

    data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu = inspect_and_fix_data(
        data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu, fix_enable=True
    )

    in_dims = {}
    for n in data.node_types:
        if 'x' not in data[n]:
            raise RuntimeError(f"Node {n} missing .x features")
        in_dims[n] = data[n].x.size(1)
        logger.info("node %s in_dim = %d", n, in_dims[n])

    logger.info("Instantiating model...")
    model = HGTMiniModel(metadata=data.metadata(), in_dims=in_dims,
                         hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                         n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, decoder="mlp").to(device)


    # 在 main() 开头或合适位置定义你想要的按-edge-type 标量权重映射
    # edge_type_weight_map = {
    #     ('phage','infects','host'): 2.0,
    #     ('protein','similar','protein'): 0.5,
    #     # 添加你要修改权重的其他 edge types
    # }
    # model = GATv2MiniModel(metadata=data.metadata(), in_dims=in_dims,
    #                      hidden_dim=args.hidden_dim, out_dim=args.out_dim,
    #                      n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, decoder="mlp",use_edge_attr=True,edge_attr_dim=1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    relation = None
    for r in data.edge_types:
        if r[0] == 'phage' and r[2] == 'host':
            relation = r
            break
    if relation is None:
        raise RuntimeError("phage->host relation not found")

    train_edge_index = torch.stack([train_src_cpu, train_dst_cpu], dim=0)
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors={etype: args.num_neighbors for etype in data.edge_types},
        edge_label_index=(relation, train_edge_index),
        edge_label=torch.ones(train_edge_index.size(1), dtype=torch.float),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    logger.info("Train loader created. batches: %d", len(train_loader))

    taxid2species = None
    host_id2taxid = None
    if args.taxid2species_tsv:
        taxmap = pd.read_csv(args.taxid2species_tsv, sep="\t")
        taxid2species = dict(zip(taxmap["taxid"], taxmap["species"]))
        if hasattr(data['host'], 'taxid'):
            host_id2taxid = data['host'].taxid.cpu().numpy()
        else:
            logger.warning("data['host'] missing .taxid, species-level eval disabled")

    # Only train positives for filtering to avoid leakage
    train_pos_edges = set(zip(train_src_cpu.tolist(), train_dst_cpu.tolist()))

    best_val_auc = -1.0
    best_val_mrr = -1.0
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model({nt: batch[nt].x for nt in batch.node_types}, batch.edge_index_dict)

            edge_label_index = batch[relation].edge_label_index
            if edge_label_index is None or edge_label_index.numel() == 0:
                continue

            pos_scores = model.decode(out, edge_label_index, etype=relation)
            labels_pos = torch.ones_like(pos_scores)

            # Negative sampling (filtered by train_pos_edges only)
            phage_nid = batch['phage'].n_id.cpu().tolist()
            host_nid = batch['host'].n_id.cpu().tolist()
            host_n = len(host_nid)

            neg_src_list, neg_dst_list = [], []
            for src_local in edge_label_index[0].cpu().tolist():
                src_global = phage_nid[src_local]
                count = 0
                attempts = 0
                max_attempts = args.neg_ratio * 20  # Increased to avoid skips
                while count < args.neg_ratio and attempts < max_attempts:
                    dst_local = random.randrange(host_n)
                    dst_global = host_nid[dst_local]
                    if (src_global, dst_global) not in train_pos_edges:
                        neg_src_list.append(src_local)
                        neg_dst_list.append(dst_local)
                        count += 1
                    attempts += 1
                if count < args.neg_ratio:
                    logger.debug("Could not find enough negatives for phage %d after %d attempts", src_global, max_attempts)

            if len(neg_src_list) == 0:
                continue

            chunk_size = 2048
            neg_loss_total = 0.0
            neg_src_tensor = torch.tensor(neg_src_list, device=device)
            neg_dst_tensor = torch.tensor(neg_dst_list, device=device)
            n_chunks = math.ceil(len(neg_src_tensor) / chunk_size)

            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(neg_src_tensor))
                neg_chunk = (neg_src_tensor[start:end], neg_dst_tensor[start:end])
                neg_scores = model.decode(out, neg_chunk, etype=relation)
                labels_neg = torch.zeros_like(neg_scores)
                neg_loss_total += loss_fn(neg_scores, labels_neg) * (end - start) / len(neg_src_tensor)  # Weighted average

            loss = loss_fn(pos_scores, labels_pos) + neg_loss_total
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        t1 = time.time()
        avg_loss = epoch_loss / max(1, n_batches)

        if epoch % args.log_every == 0 or epoch == args.epochs:
            try:
                train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
                    model, data, train_pair, val_pair, test_pair, relation=relation,
                    eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
                    host_id2taxid=host_id2taxid, taxid2species=taxid2species,k_list=(1, 5, 10), # <- 这里加上
                    save_path="phage_prediction_results"
                )
                train_auc, train_mrr, train_hits = train_metrics
                val_auc, val_mrr, val_hits = val_metrics
                test_auc, test_mrr, test_hits = test_metrics

                # Save predictions with epoch suffix
                pred_file = f"phage_prediction_results_epoch_{epoch}.tsv"
                save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file=pred_file,k_list=(10,) )

            except Exception as e:
                logger.warning("Full-graph eval failed: %s", e)
                train_auc = val_auc = test_auc = float('nan')
                train_mrr = val_mrr = test_mrr = 0.0
                train_hits = val_hits = test_hits = {k: 0.0 for k in (1,5,10)}

            logger.info("[Epoch %03d] loss=%.6f time=%.1fs val_auc=%.4f val_mrr=%.4f hits@1/5/10=%s",
                        epoch, avg_loss, t1 - t0, val_auc, val_mrr, val_hits)

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_ckpt = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'val_mrr': val_mrr
                }
                torch.save(best_ckpt, args.save_path)
                logger.info("Saved best model -> %s", args.save_path)

    if best_ckpt is not None:
        model.load_state_dict(best_ckpt['model_state'])
    logger.info("Evaluating final test...")
    train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
        model, data, train_pair, val_pair, test_pair, relation=relation,
        eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
        host_id2taxid=host_id2taxid, taxid2species=taxid2species,k_list=(1, 5, 10),   # <- 这里加上
        save_path="phage_prediction_results"
    )
    logger.info("FINAL TEST metrics (AUC, MRR, Hits@1/5/10): %s", test_metrics)

    # Final predictions
    save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file="phage_prediction_results_final.tsv",k_list=(10,))

# def main():
#     import pandas as pd
#     import random

#     args = parse_args()
#     set_seed(args.seed)
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     eval_device = torch.device(args.eval_device)

#     logger.info("Loading data: %s", args.data_pt)
#     data, split_edge = safe_torch_load(args.data_pt)
#     logger.info("Data metadata: %s", data.metadata())

#     train_pair, val_pair, test_pair = find_phage_host_splits(data, split_edge)
#     train_src_cpu, train_dst_cpu = train_pair
#     val_src_cpu, val_dst_cpu = val_pair
#     test_src_cpu, test_dst_cpu = test_pair
#     logger.info("Train/Val/Test counts: %d / %d / %d", train_src_cpu.size(0), val_src_cpu.size(0), test_src_cpu.size(0))

#     data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu = inspect_and_fix_data(
#         data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu, fix_enable=True
#     )

#     in_dims = {}
#     for n in data.node_types:
#         if 'x' not in data[n]:
#             raise RuntimeError(f"Node {n} missing .x features")
#         in_dims[n] = data[n].x.size(1)
#         logger.info("node %s in_dim = %d", n, in_dims[n])

#     logger.info("Instantiating model...")
#     model = HGTMiniModel(metadata=data.metadata(), in_dims=in_dims,
#                          hidden_dim=args.hidden_dim, out_dim=args.out_dim,
#                          n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

#     '''lr=1e-3, weight_decay=1e-4

#         lr=5e-4, weight_decay=5e-4

#         lr=2e-3, weight_decay=1e-5
#     '''

#     relation = None
#     for r in data.edge_types:
#         if r[0] == 'phage' and r[2] == 'host':
#             relation = r
#             break
#     if relation is None:
#         raise RuntimeError("phage->host relation not found")

#     train_edge_index = torch.stack([train_src_cpu, train_dst_cpu], dim=0)
#     train_loader = LinkNeighborLoader(
#         data,
#         num_neighbors={etype: args.num_neighbors for etype in data.edge_types},
#         edge_label_index=(relation, train_edge_index),
#         edge_label=torch.ones(train_edge_index.size(1), dtype=torch.float),
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=0
#     )
#     logger.info("Train loader created. batches: %d", len(train_loader))

#     taxid2species = None
#     host_id2taxid = None
#     if args.taxid2species_tsv:
#         taxmap = pd.read_csv(args.taxid2species_tsv, sep="\t")
#         taxid2species = dict(zip(taxmap["taxid"], taxmap["species"]))
#         if hasattr(data['host'], 'taxid'):
#             host_id2taxid = data['host'].taxid.cpu().numpy()
#         else:
#             logger.warning("data['host'] missing .taxid, species-level eval disabled")

#     # Only train positives for filtering to avoid leakage
#     train_pos_edges = set(zip(train_src_cpu.tolist(), train_dst_cpu.tolist()))

#     best_val_mrr = -1.0
#     best_ckpt = None
#     ###
#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         model.train()
#         epoch_loss = 0.0
#         n_batches = 0

#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()

#             out = model({nt: batch[nt].x for nt in batch.node_types}, batch.edge_index_dict)

#             edge_label_index = batch[relation].edge_label_index
#             if edge_label_index is None or edge_label_index.numel() == 0:
#                 continue

#             # 正样本打分
#             pos_scores = model.decode(out, edge_label_index, etype=relation)

#             # # -------- Negative sampling --------
#             # phage_nid = batch['phage'].n_id.cpu().tolist()
#             # host_nid = batch['host'].n_id.cpu().tolist()
#             # host_n = len(host_nid)

#             # neg_src_list, neg_dst_list = [], []
#             # for src_local in edge_label_index[0].cpu().tolist():
#             #     src_global = phage_nid[src_local]
#             #     count, attempts = 0, 0
#             #     max_attempts = args.neg_ratio * 20
#             #     while count < args.neg_ratio and attempts < max_attempts:
#             #         dst_local = random.randrange(host_n)
#             #         dst_global = host_nid[dst_local]
#             #         if (src_global, dst_global) not in train_pos_edges:
#             #             neg_src_list.append(src_local)
#             #             neg_dst_list.append(dst_local)
#             #             count += 1
#             #         attempts += 1

#             # if len(neg_src_list) == 0:
#             #     continue

#             # # -------- Negative decoding in chunks --------
#             # chunk_size = 2048
#             # neg_scores_all = []
#             # neg_src_tensor = torch.tensor(neg_src_list, device=device)
#             # neg_dst_tensor = torch.tensor(neg_dst_list, device=device)
#             # n_chunks = math.ceil(len(neg_src_tensor) / chunk_size)

#             # for i in range(n_chunks):
#             #     start = i * chunk_size
#             #     end = min((i + 1) * chunk_size, len(neg_src_tensor))
#             #     neg_chunk = (neg_src_tensor[start:end], neg_dst_tensor[start:end])
#             #     neg_scores_chunk = model.decode(out, neg_chunk, etype=relation)
#             #     neg_scores_all.append(neg_scores_chunk)

#             # neg_scores = torch.cat(neg_scores_all, dim=0)

#             # # -------- BPR Loss --------
#             # loss = bpr_loss(pos_scores, neg_scores)


#             # ------------------ SoftmaxCE loss computation ------------------
#             # We will compute logits between each positive phage and ALL host nodes in the batch,
#             # and apply cross-entropy with the true host local index as label.
#             #
#             # edge_label_index contains local indices: [phage_local_idx, host_local_idx]
#             pos_phage_local = edge_label_index[0].to(device).long()  # indices into out['phage']
#             pos_host_local = edge_label_index[1].to(device).long()   # indices into out['host']

#             # Ensure out contains 'phage' and 'host' keys
#             if 'phage' not in out or 'host' not in out:
#                 raise RuntimeError("Model output missing 'phage' or 'host' embeddings needed for softmaxCE.")

#             phage_emb_batch = out['phage']  # (num_phage_in_batch, D)
#             host_emb_batch = out['host']    # (num_host_in_batch, D)

#             # temperature tau: you can expose this via args (default 1.0)
#             tau = getattr(args, "softmax_tau", 0.5)

#             # compute softmax cross-entropy loss
#             loss = softmax_ce_loss(phage_emb_batch, host_emb_batch, pos_phage_local, pos_host_local, tau=tau)

#             # -------- Backward --------
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#             n_batches += 1

#         t1 = time.time()
#         avg_loss = epoch_loss / max(1, n_batches)

#         if epoch % args.log_every == 0 or epoch == args.epochs:
#             try:
#                 train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
#                     model, data, train_pair, val_pair, test_pair, relation=relation,
#                     eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
#                     host_id2taxid=host_id2taxid, taxid2species=taxid2species,k_list=(1, 5, 10), # <- 这里加上
#                     save_path="phage_prediction_results"
#                 )
#                 train_auc, train_mrr, train_hits = train_metrics
#                 val_auc, val_mrr, val_hits = val_metrics
#                 test_auc, test_mrr, test_hits = test_metrics

#                 # Save predictions with epoch suffix
#                 pred_file = f"phage_prediction_results_epoch_{epoch}.tsv"
#                 save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file=pred_file,k_list=(10,) )

#             except Exception as e:
#                 logger.warning("Full-graph eval failed: %s", e)
#                 train_auc = val_auc = test_auc = float('nan')
#                 train_mrr = val_mrr = test_mrr = 0.0
#                 train_hits = val_hits = test_hits = {k: 0.0 for k in (1,5,10)}

#             logger.info("[Epoch %03d] loss=%.6f time=%.1fs val_auc=%.4f val_mrr=%.4f hits@1/5/10=%s",
#                         epoch, avg_loss, t1 - t0, val_auc, val_mrr, val_hits)

#             if val_mrr > best_val_mrr:
#                 best_val_mrr = val_mrr
#                 best_ckpt = {
#                     'model_state': model.state_dict(),
#                     'optimizer_state': optimizer.state_dict(),
#                     'epoch': epoch,
#                     'val_mrr': val_mrr
#                 }
#                 torch.save(best_ckpt, args.save_path)
#                 logger.info("Saved best model -> %s", args.save_path)

#     if best_ckpt is not None:
#         model.load_state_dict(best_ckpt['model_state'])
#     logger.info("Evaluating final test...")
#     train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
#         model, data, train_pair, val_pair, test_pair, relation=relation,
#         eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
#         host_id2taxid=host_id2taxid, taxid2species=taxid2species,k_list=(1, 5, 10),   # <- 这里加上
#         save_path="phage_prediction_results"
#     )
#     logger.info("FINAL TEST metrics (AUC, MRR, Hits@1/5/10): %s", test_metrics)

#     # Final predictions
#     save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file="phage_prediction_results_final.tsv",k_list=(10,))

if __name__ == "__main__":
    main()



'''
python train_hgt_phage_host.py \
  --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP.pt \
  --device cuda \
  --eval_device cuda \
  --epochs 5 \
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
'''





edge_type_weight_map = {
        # 你指定的特殊权重：
        ('phage', 'infects', 'host')
        ('protein', 'similar', 'protein')
        ('phage', 'interacts', 'phage')
        ('host', 'interacts', 'host')
        ('phage', 'encodes', 'protein')
        ('host', 'encodes', 'protein')
        ('host', 'belongs_to', 'taxonomy')
        ('taxonomy', 'related', 'taxonomy')
        ('phage', 'belongs_to', 'taxonomy')
    }