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

import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

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
        use_edge_attr: bool = True,      # 是否启用 edge_attr 支持
        edge_attr_dim: int = 1,          # edge_attr 的维度（1 表示标量权重）
    ):
        """
        GATv2MiniModel for heterogeneous graphs using HeteroConv(GATv2Conv).
        - metadata: (node_types_list, edge_types_list), where edge_types are tuples (src, rel, dst)
        - in_dims: dict mapping node type -> input feature dim
        - use_edge_attr: whether to pass edge_attr to GATv2Conv (requires GATv2Conv to support edge_dim)
        - edge_attr_dim: number of channels for edge_attr (1 for scalar weight)
        """
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

        # 输入投影（每种节点类型投到 hidden_dim）
        self.input_proj = nn.ModuleDict()
        for n in self.node_types:
            d = in_dims.get(n)
            if d is None:
                raise RuntimeError(f"Missing in_dim for node type {n}")
            self.input_proj[n] = nn.Linear(d, hidden_dim)

        # 使用 concat=False（输出维度由 out_channels 决定），避免 hidden_dim 必须可被 n_heads 整除
        concat_flag = False
        out_channels = hidden_dim

        # 为每一层创建 ModuleDict（字符串键）来注册 GATv2Conv 子模块，并构造对应的 HeteroConv
        self.edge_conv_md_list = nn.ModuleList()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            convs_md = nn.ModuleDict()
            for (src, rel, dst) in self.edge_types:
                str_key = f"{src}__{rel}__{dst}"
                # 对于异构边 (src != dst) 强制禁用 self-loops（add_self_loops=False）
                add_self_loops_flag = (src == dst)

                if self.use_edge_attr:
                    # 注意：GATv2Conv 必须支持 edge_dim 参数（某些旧版本 PyG 不支持）
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

            # 保存 ModuleDict（参数会被正确注册）
            self.edge_conv_md_list.append(convs_md)

            # 构造 HeteroConv 需要的映射： (src,rel,dst) -> module
            conv_map = {
                et: convs_md[f"{et[0]}__{et[1]}__{et[2]}"] for et in self.edge_types
            }
            self.layers.append(HeteroConv(conv_map, aggr='sum'))

        self.dropout = nn.Dropout(self.dropout_p)

        # 输出投影 + decoder（与原 HGTMiniModel 保持行为一致）
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))
        if decoder == "mlp":
            self.decoder_mlp = self.edge_mlp

        # ========== 新增: 可学习的 logit scale（以 log-space 存放） ==========
        # 用 exp(self.logit_scale) 作为实际放缩系数 (保证正)
        # 初始为 0.0 -> scale = 1.0；如需更大初始 scale 可改为 torch.log(torch.tensor(5.0))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: typing.Optional[dict] = None,   # etype -> tensor/float 的映射
    ) -> dict[str, torch.Tensor]:
        """
        edge_attr_dict keys must be etype tuples like (src,rel,dst) to match edge_index_dict keys.
        Values can be:
        - float/int scalar (will be expanded to shape (E,) or (E, edge_attr_dim))
        - torch.Tensor of shape (E,) or (E, edge_attr_dim)
        If edge_attr_dict is None or an etype not present, that etype is forwarded without edge_attr.
        """
        # 1) input proj
        h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

        # 2) for each layer, prepare per-edge-type edge_attr mapping (if provided)
        for layer in self.layers:
            if self.use_edge_attr and edge_attr_dict is not None:
                # build processed_edge_attr: mapping from etype tuple -> tensor
                processed = {}
                for etype, edge_index in edge_index_dict.items():
                    E = edge_index.size(1)
                    if etype in edge_attr_dict:
                        val = edge_attr_dict[etype]
                        # scalar -> expand
                        if isinstance(val, (float, int)):
                            if self.edge_attr_dim == 1:
                                t = torch.full((E,), float(val), dtype=torch.float, device=edge_index.device)
                            else:
                                t = torch.full((E, self.edge_attr_dim), float(val),
                                            dtype=torch.float, device=edge_index.device)
                        elif isinstance(val, torch.Tensor):
                            t = val.to(edge_index.device)
                            # allow (E,) or (E, edge_attr_dim)
                            if t.dim() == 1:
                                if self.edge_attr_dim == 1:
                                    if t.size(0) != E:
                                        raise RuntimeError(f"edge_attr for {etype} has length {t.size(0)} but expected {E}")
                                else:
                                    if t.size(0) != E:
                                        raise RuntimeError(f"edge_attr for {etype} has length {t.size(0)} but expected {E}")
                                    t = t.view(-1, 1).repeat(1, self.edge_attr_dim)
                            elif t.dim() == 2:
                                if t.size(0) != E:
                                    raise RuntimeError(f"edge_attr for {etype} has first dim {t.size(0)} but expected {E}")
                                if t.size(1) != self.edge_attr_dim:
                                    raise RuntimeError(f"edge_attr for {etype} has dim {t.size(1)} but model expects {self.edge_attr_dim}")
                            else:
                                raise RuntimeError(f"edge_attr for {etype} must be 1D or 2D tensor")
                        else:
                            raise RuntimeError(f"Unsupported edge_attr type for {etype}: {type(val)}")
                        processed[etype] = t
                    else:
                        # not provided -> do not include key (HeteroConv will call conv without that extra arg)
                        pass

                # IMPORTANT: HeteroConv 要求 kwargs 以 `_dict` 结尾，该 dict 将会被展开并以 `edge_attr=...` 的形式传给子 conv
                h = layer(h, edge_index_dict, edge_attr_dict=processed)
                
            else:
                # default behavior (no edge attributes)
                h = layer(h, edge_index_dict)

            # activation + dropout
            for k in list(h.keys()):
                h[k] = F.relu(self.dropout(h[k]))

        # 默认返回归一化的 embedding（L2=1）
        out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
        
        return out

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        etype: tuple[str, str, str]
    ) -> torch.Tensor:
        # same interface as your original HGTMiniModel.decode
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
            # 防御性归一化（如果 forward 已归一化这一步是幂等的）
            src_n = F.normalize(src_z, p=2, dim=-1)
            dst_n = F.normalize(dst_z, p=2, dim=-1)
            sim = F.cosine_similarity(src_n, dst_n)   # in [-1,1]
            # 用可学习的 scale 放大 logits（保证为正）
            return sim * torch.exp(self.logit_scale)
        elif self.decoder_type == "mlp":
            e = torch.cat([src_z, dst_z], dim=-1)
            return self.decoder_mlp(e).view(-1)
        else:
            raise ValueError(f"Unknown decoder {self.decoder_type}")



# -------------------------
# Metrics (full-graph eval)
# -------------------------
@torch.no_grad()

# def compute_metrics_fullgraph(
#     #model: HGTMiniModel,
#     model: GATv2MiniModel,
#     data: HeteroData,
#     train_pairs: tuple[torch.Tensor, torch.Tensor],
#     val_pairs: tuple[torch.Tensor, torch.Tensor],
#     test_pairs: tuple[torch.Tensor, torch.Tensor],
#     relation: tuple[str, str, str],
#     eval_device: str = 'cpu',
#     eval_neg_ratio: int = 10,
#     k_list: tuple[int, ...] = (1, 5, 10),
#     host_id2taxid: typing.Union[np.ndarray, None] = None,
#     taxid2species: typing.Union[dict[int, str], None] = None,
#     save_path: Optional[str] = None,
#     top_k: int = 10)-> tuple[tuple[float, float, dict[int, float]],
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
#         #out = model(data_eval_x, edge_index_dict_eval)
#         # 构建全图级别的 edge_attr 字典（优先使用 data[etype].edge_weight，否则使用 edge_type_weight_map）
#         edge_type_weight_map = {
#         # 你指定的特殊权重：
#         ('phage', 'infects', 'host'): 1.0,
#         ('protein', 'similar', 'protein'): 1.0,

#         # 其他边类型默认设为 1.0（中性影响）
#         ('host', 'has_sequence', 'host_sequence'): 1.0,
#         ('phage', 'interacts', 'phage'): 1.0,
#         ('host', 'interacts', 'host'): 1.0,
#         ('phage', 'encodes', 'protein'): 1.0,
#         ('host', 'encodes', 'protein'): 1.0,
#         ('host', 'belongs_to', 'taxonomy'): 1.0,
#         ('taxonomy', 'related', 'taxonomy'): 1.0,
#     }
#         global_edge_attr = {}
#         for et in data.edge_types:
#             # only include if edge_index exists in edge_index_dict_eval
#             if et not in edge_index_dict_eval:
#                 continue
#             if hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
#                 global_edge_attr[et] = data[et].edge_weight.to(eval_device)
#             elif et in edge_type_weight_map:
#                 E = data[et].edge_index.size(1)
#                 global_edge_attr[et] = torch.full((E,), float(edge_type_weight_map[et]), device=eval_device)
#             else:
#                 pass

#         model.eval()
#         out = model(data_eval_x, edge_index_dict_eval, edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None)

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
#         # ensure top_k covers requested k_list
#         # ======================
#         effective_top_k = max(top_k, max(k_list))

#         # ======================
#         # AUC 计算 (修正后的负采样)
#         # ======================
#         def compute_auc_for_pairs(pairs: tuple[torch.Tensor, torch.Tensor], pos_map: defaultdict[int, set[int]]) -> float:
#             src_cpu, dst_cpu = pairs
#             if src_cpu.numel() == 0:
#                 return float('nan')

#             pos_scores = []
#             neg_scores_list = []
#             all_hosts_idx = torch.arange(n_hosts, device=eval_device)

#             for s_idx, d_idx in zip(src_cpu.tolist(), dst_cpu.tolist()):
#                 if not (0 <= d_idx < n_hosts):
#                     continue

#                 # positive score
#                 pos_score = torch.sigmoid(
#                     model.decode(out, (torch.tensor([s_idx], device=eval_device),
#                                        torch.tensor([d_idx], device=eval_device)),
#                                  etype=relation)
#                 ).item()
#                 pos_scores.append(pos_score)

#                 # negative candidates
#                 known = pos_map.get(s_idx, set())
#                 mask = torch.ones(n_hosts, dtype=torch.bool, device=eval_device)
#                 for oth in known:
#                     if 0 <= oth < n_hosts:
#                         mask[oth] = False
#                 if 0 <= d_idx < n_hosts:
#                     mask[d_idx] = False

#                 cand_neg_idx = mask.nonzero(as_tuple=True)[0]
#                 if cand_neg_idx.numel() == 0:
#                     continue

#                 # sample up to eval_neg_ratio negatives
#                 sample_size = min(int(eval_neg_ratio), int(cand_neg_idx.numel()))
#                 if sample_size <= 0:
#                     continue
#                 perm = torch.randperm(cand_neg_idx.numel(), device=eval_device)[:sample_size]
#                 neg_idx = cand_neg_idx[perm]

#                 neg_scores = torch.sigmoid(
#                     model.decode(out, (torch.full((neg_idx.numel(),), s_idx, device=eval_device),
#                                        neg_idx),
#                                  etype=relation)
#                 ).cpu().numpy()
#                 neg_scores_list.append(neg_scores)

#             if len(pos_scores) == 0 or len(neg_scores_list) == 0:
#                 return float('nan')

#             y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(sum(len(n) for n in neg_scores_list))])
#             y_score = np.concatenate([np.array(pos_scores), np.concatenate(neg_scores_list)])
#             try:
#                 return float(roc_auc_score(y_true, y_score))
#             except Exception:
#                 return float('nan')

#         # ======================
#         # MRR & Hits@K
#         # ======================
#         def compute_rank_metrics(pairs: tuple[torch.Tensor, torch.Tensor], save_path: Optional[str] = None):
#             src_cpu, dst_cpu = pairs
#             if src_cpu.numel() == 0:
#                 return 0.0, {k: 0.0 for k in k_list}

#             ph2hosts = defaultdict(list)
#             for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
#                 ph2hosts[int(s)].append(int(d))

#             hits = {k: 0 for k in k_list}
#             rr_sum, total_q = 0.0, len(ph2hosts)

#             prediction_rows = []
#             cand_hosts = torch.arange(n_hosts, device=eval_device)

#             for ph_idx, true_ds in ph2hosts.items():
#                 scores_tensor = torch.sigmoid(
#                     model.decode(out, (torch.full((n_hosts,), ph_idx, device=eval_device),
#                                        cand_hosts),
#                                  etype=relation)
#                 )

#                 scores_np = scores_tensor.cpu().numpy()
#                 topk_idx = scores_np.argsort()[::-1][:effective_top_k]

#                 phage_real_id = phage_idx2id.get(ph_idx, str(ph_idx))
#                 true_species = {hostid2species(h) for h in true_ds}

#                 for rank, h in enumerate(topk_idx, 1):
#                     host_real_id = host_idx2id.get(int(h), str(int(h)))
#                     host_species_name = hostid2species(int(h))
#                     score = float(scores_np[int(h)])
#                     prediction_rows.append({
#                         "phage_id": phage_real_id,
#                         "rank": rank,
#                         "host_id": host_real_id,
#                         "host_species": host_species_name,
#                         "score": score
#                     })

#                 # reciprocal rank
#                 rank_val = None
#                 for pos, h in enumerate(topk_idx, 1):
#                     if hostid2species(int(h)) in true_species:
#                         rank_val = pos
#                         break
#                 if rank_val is None:
#                     rank_val = effective_top_k + 1
#                 rr_sum += 1.0 / rank_val

#                 for k in k_list:
#                     if k <= effective_top_k:
#                         if any(hostid2species(int(h)) in true_species for h in topk_idx[:k]):
#                             hits[k] += 1

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
# -------------------------
# Compute & Save (modified to evaluate by host GCF_id and keep species in outputs)
# -------------------------


def compute_metrics_fullgraph(
    model: GATv2MiniModel,
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
    # -------------- new optional mapping: host index -> GCF id (string) --------------
    host_idx2gcf: typing.Union[dict, list, np.ndarray, None] = None,
    save_path: typing.Optional[str] = None,
    top_k: int = 10
) -> tuple[tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]]]:
    """
    Compute full-graph metrics (AUC, MRR, Hits@K) and optionally save top-k predictions.
    Evaluation correctness is checked by **GCF_id** (host-level GCF).
    Output files include both host_gcf and host_species (if available).
    """

    # ======================
    # Load phage/host idx -> real id (existing logic)
    # ======================
    try:
        with open("node_maps_RBP.json", "r", encoding="utf-8") as f:
            node_maps = json.load(f)
        phage_map = node_maps.get("phage_map", {})
        host_map = node_maps.get("host_map", {})
        # phage_idx -> phage id string
        phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
        # host_idx -> host id string (likely GCF string if node maps were created from host_gcf)
        host_idx2id = {int(v): str(k) for k, v in host_map.items()}
    except FileNotFoundError:
        logger.warning("node_maps_RBP.json not found, attempting to fallback to data attributes")
        phage_idx2id = {}
        host_idx2id = {}

    # if explicit host_idx2gcf provided, prefer it
    if host_idx2gcf is None:
        # try to use host_idx2id (from node_maps) as GCF if possible
        host_idx2gcf = host_idx2id if host_idx2id else None

    # ======================
    # Device & forward
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

        #build global edge_attr same as before (kept)
        edge_type_weight_map = {
        # 你指定的特殊权重：
        ('phage', 'infects', 'host'): 1.0,
        ('protein', 'similar', 'protein'): 1.0,

        # 其他边类型默认设为 1.0（中性影响）
        ('host', 'has_sequence', 'host_sequence'): 1.0,
        ('phage', 'interacts', 'phage'): 1.0,
        ('host', 'interacts', 'host'): 1.0,
        ('phage', 'encodes', 'protein'): 1.0,
        ('host', 'encodes', 'protein'): 1.0,
        ('host', 'belongs_to', 'taxonomy'): 1.0,
        ('taxonomy', 'related', 'taxonomy'): 1.0,
    }
        global_edge_attr = {}
        for et in data.edge_types:
            if et not in edge_index_dict_eval:
                continue
            if hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
                global_edge_attr[et] = data[et].edge_weight.to(eval_device)
            elif et in edge_type_weight_map:
                E = data[et].edge_index.size(1)
                global_edge_attr[et] = torch.full((E,), float(edge_type_weight_map[et]), device=eval_device)
            else:
                pass

        model.eval()
        out = model(data_eval_x, edge_index_dict_eval, edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None)

        n_hosts = out['host'].size(0)

        # ======================
        # helper functions for mapping
        # ======================
        def hostidx2gcf(hidx: int) -> str:
            # prefer explicit mapping, then try node_maps host_idx2id
            try:
                if host_idx2gcf is None:
                    return f"unknown_{hidx}"
                if isinstance(host_idx2gcf, dict):
                    return host_idx2gcf.get(int(hidx), f"unknown_{hidx}")
                elif isinstance(host_idx2gcf, (list, np.ndarray)):
                    return str(host_idx2gcf[int(hidx)])
                else:
                    return str(host_idx2gcf.get(int(hidx), f"unknown_{hidx}"))
            except Exception:
                return f"unknown_{hidx}"

        def hostidx2species(hidx: int) -> str:
            if host_id2taxid is None or taxid2species is None:
                return f"unknown_{hidx}"
            try:
                taxid = int(host_id2taxid[int(hidx)])
                return taxid2species.get(taxid, f"unknown_{taxid}")
            except Exception:
                return f"unknown_{hidx}"

        def build_pos_map(pairs: tuple[torch.Tensor, torch.Tensor]) -> defaultdict[int, set[int]]:
            pos_map = defaultdict(set)
            src_cpu, dst_cpu = pairs
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                pos_map[int(s)].add(int(d))
            return pos_map

        train_pos_map = build_pos_map(train_pairs)
        val_pos_map = build_pos_map(val_pairs)
        test_pos_map = build_pos_map(test_pairs)

        effective_top_k = max(top_k, max(k_list))

        # ======================
        # AUC (unchanged, keep previous negative sampling for AUC)
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

                pos_score = torch.sigmoid(
                    model.decode(out, (torch.tensor([s_idx], device=eval_device),
                                       torch.tensor([d_idx], device=eval_device)),
                                 etype=relation)
                ).item()
                pos_scores.append(pos_score)

                # negative candidates (all except known positives and the positive one)
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
        # MRR & Hits@K -- evaluate by **GCF** equality
        # ======================
        def compute_rank_metrics(pairs: tuple[torch.Tensor, torch.Tensor], save_path: typing.Optional[str] = None):
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
                # scores for all hosts for this phage
                scores_tensor = torch.sigmoid(
                    model.decode(out, (torch.full((n_hosts,), ph_idx, device=eval_device),
                                       cand_hosts),
                                 etype=relation)
                )

                scores_np = scores_tensor.cpu().numpy()
                topk_idx = scores_np.argsort()[::-1][:effective_top_k]

                phage_real_id = phage_idx2id.get(ph_idx, str(ph_idx))
                # true set in GCF-space
                true_gcf_set = {hostidx2gcf(d) for d in true_ds}

                for rank, h in enumerate(topk_idx, 1):
                    host_real_id = host_idx2id.get(int(h), str(int(h))) if 'host_idx2id' in locals() else hostidx2gcf(int(h))
                    host_gcf_str = hostidx2gcf(int(h))
                    host_species_name = hostidx2species(int(h))
                    score = float(scores_np[int(h)])
                    prediction_rows.append({
                        "phage_id": phage_real_id,
                        "rank": rank,
                        "host_idx": int(h),
                        "host_gcf": host_gcf_str,
                        "host_species": host_species_name,
                        "score": score
                    })

                # reciprocal rank: check if predicted host_gcf in true_gcf_set
                rank_val = None
                for pos, h in enumerate(topk_idx, 1):
                    if hostidx2gcf(int(h)) in true_gcf_set:
                        rank_val = pos
                        break
                if rank_val is None:
                    rank_val = effective_top_k + 1
                rr_sum += 1.0 / rank_val

                for k in k_list:
                    if k <= effective_top_k:
                        if any(hostidx2gcf(int(h)) in true_gcf_set for h in topk_idx[:k]):
                            hits[k] += 1

            if save_path is not None and len(prediction_rows) > 0:
                pd.DataFrame(prediction_rows).to_csv(save_path, sep="\t", index=False)

            mrr = rr_sum / total_q if total_q > 0 else 0.0
            hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}
            return mrr, hits_at

        # ======================
        # compute metrics (train/val/test)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -------------------------
# Save Predictions
# -------------------------



def save_predictions(
    # model: HGTMiniModel,
    model: GATv2MiniModel,
    data: HeteroData,
    test_src_cpu: torch.Tensor,
    test_dst_cpu: torch.Tensor,
    relation: tuple[str, str, str],
    eval_device: torch.device,
    host_id2taxid: typing.Union[np.ndarray, None] = None,
    taxid2species: typing.Union[dict[int, str], None] = None,
    output_file: str = "phage_prediction_results.tsv",
    top_k: int = 10,  # ⭐ 新增：控制输出 top-k
    k_list=(1, 5, 10),
    edge_type_weight_map: typing.Optional[dict] = None,   # <- 新增：标量权重映射，key=(src,rel,dst)
    edge_attr_dict: typing.Optional[dict] = None         # <- 新增：显式传入的 edge_attr（优先）
):
    """
    保存 top-k 的 phage->host 预测。
    edge_attr_dict (优先) 可以是 {(src,rel,dst): tensor_or_scalar, ...}
    edge_type_weight_map 为 {(src,rel,dst): scalar, ...}（可选）
    """
    orig_dev = next(model.parameters()).device
    moved = False
    if orig_dev != eval_device:
        model.to(eval_device)
        moved = True

    try:
        # 全图节点特征与边索引
        full_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
        full_edge_index_dict = {
            et: data[et].edge_index.to(eval_device)
            for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
        }

        # --- 构建 global_edge_attr ---
        # 优先使用显式传入的 edge_attr_dict；
        # 否则使用 data[etype].edge_weight（如果存在）；
        # 否则使用 edge_type_weight_map 标量（如果提供）。
        global_edge_attr = {}
        # Normalize inputs
        edge_type_weight_map = edge_type_weight_map or {}

        for et, edge_index in full_edge_index_dict.items():
            E = edge_index.size(1)
            # 1) 显式传入优先
            if edge_attr_dict is not None and et in edge_attr_dict:
                val = edge_attr_dict[et]
                if isinstance(val, (float, int)):
                    global_edge_attr[et] = torch.full((E,), float(val), device=eval_device)
                elif isinstance(val, torch.Tensor):
                    t = val.to(eval_device)
                    if t.dim() == 1 and t.size(0) == E:
                        global_edge_attr[et] = t
                    elif t.dim() == 2 and t.size(0) == E:
                        global_edge_attr[et] = t
                    else:
                        raise RuntimeError(f"edge_attr for {et} has invalid shape {tuple(t.size())}, expected (E,) or (E, dim).")
                else:
                    raise RuntimeError(f"Unsupported edge_attr type for {et}: {type(val)}")
            # 2) else try data[et].edge_weight
            elif hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
                global_edge_attr[et] = data[et].edge_weight.to(eval_device)
                # If it's scalar stored somehow, ensure shape (E,)
                if global_edge_attr[et].dim() == 0:
                    global_edge_attr[et] = global_edge_attr[et].expand(E)
            # 3) else try edge_type_weight_map scalar
            elif et in edge_type_weight_map:
                w = float(edge_type_weight_map[et])
                global_edge_attr[et] = torch.full((E,), w, device=eval_device)
            else:
                # not provided for this etype -> do not include key
                pass

        # 若没有任何 edge_attr, 传 None，使 model 在 forward 分支走无 edge_attr 的路径
        edge_attr_arg = global_edge_attr if len(global_edge_attr) > 0 else None

        # --- Forward (eval) ---
        model.eval()
        # 注意 model 的 forward 已支持 edge_attr_dict 参数
        out_full = model(full_x, full_edge_index_dict, edge_attr_dict=edge_attr_arg)

        # 计算 test 对应的分数（与你原先保持一致）
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
        # fix typo and call correctly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# -------------------------
# Training entry
# -------------------------

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
#     # model = HGTMiniModel(metadata=data.metadata(), in_dims=in_dims,
#     #                      hidden_dim=args.hidden_dim, out_dim=args.out_dim,
#     #                      n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, decoder="mlp").to(device)


#     # 在 main() 开头或合适位置定义你想要的按-edge-type 标量权重映射
#     edge_type_weight_map = {
#         # 你指定的特殊权重：
#         ('phage', 'infects', 'host'): 1.0,
#         ('protein', 'similar', 'protein'): 1.0,

#         # 其他边类型默认设为 1.0（中性影响）
#         ('host', 'has_sequence', 'host_sequence'): 1.0,
#         ('phage', 'interacts', 'phage'): 1.0,
#         ('host', 'interacts', 'host'): 1.0,
#         ('phage', 'encodes', 'protein'): 1.0,
#         ('host', 'encodes', 'protein'): 1.0,
#         ('host', 'belongs_to', 'taxonomy'): 1.0,
#         ('taxonomy', 'related', 'taxonomy'): 1.0,
#     }
#     model = GATv2MiniModel(metadata=data.metadata(), in_dims=in_dims,
#                          hidden_dim=args.hidden_dim, out_dim=args.out_dim,
#                          n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, decoder="cosine",use_edge_attr=True,edge_attr_dim=1).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
#     loss_fn = nn.BCEWithLogitsLoss()

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

#     best_val_auc = -1.0
#     best_val_mrr = -1.0
#     best_ckpt = None

#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         model.train()
#         epoch_loss = 0.0
#         n_batches = 0

#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             #out = model({nt: batch[nt].x for nt in batch.node_types}, batch.edge_index_dict)

#             # 准备 x_dict 与 edge_index_dict（你之前已有）
#             x_dict = {nt: batch[nt].x for nt in batch.node_types}
#             edge_index_dict = batch.edge_index_dict  # keys are tuples (src,rel,dst)

#             # 构造 batch_edge_attr: 优先使用 batch 中已有的 edge_weight，否则使用你指定的 scalar map
#             batch_edge_attr = {}
#             for et in batch.edge_types:
#                 # et is tuple like ('phage','infects','host')
#                 if hasattr(batch[et], 'edge_weight') and batch[et].edge_weight is not None:
#                     # 若 batch 自带 per-edge weights，直接用它
#                     batch_edge_attr[et] = batch[et].edge_weight.to(batch[et].edge_index.device)
#                 elif et in edge_type_weight_map:
#                     # 否则，如果 edge_type_weight_map 给了一个标量权重，就扩展成长度为 E 的张量
#                     E = batch[et].edge_index.size(1)
#                     batch_edge_attr[et] = torch.full((E,), float(edge_type_weight_map[et]), device=batch[et].edge_index.device)
#                 else:
#                     # 没有提供 -> 不在字典中（模型会按无 edge_attr 分支走）
#                     pass

#             # 调用模型，传入 edge_attr_dict（如果为空可以传 None）
#             edge_attr_arg = batch_edge_attr if len(batch_edge_attr) > 0 else None
#             out = model(x_dict, edge_index_dict, edge_attr_dict=edge_attr_arg)

#             edge_label_index = batch[relation].edge_label_index
#             if edge_label_index is None or edge_label_index.numel() == 0:
#                 continue

#             pos_scores = model.decode(out, edge_label_index, etype=relation)
#             labels_pos = torch.ones_like(pos_scores)

#             # Negative sampling (filtered by train_pos_edges only)
#             phage_nid = batch['phage'].n_id.cpu().tolist()
#             host_nid = batch['host'].n_id.cpu().tolist()
#             host_n = len(host_nid)

#             neg_src_list, neg_dst_list = [], []
#             for src_local in edge_label_index[0].cpu().tolist():
#                 src_global = phage_nid[src_local]
#                 count = 0
#                 attempts = 0
#                 max_attempts = args.neg_ratio * 20  # Increased to avoid skips
#                 while count < args.neg_ratio and attempts < max_attempts:
#                     dst_local = random.randrange(host_n)
#                     dst_global = host_nid[dst_local]
#                     if (src_global, dst_global) not in train_pos_edges:
#                         neg_src_list.append(src_local)
#                         neg_dst_list.append(dst_local)
#                         count += 1
#                     attempts += 1
#                 if count < args.neg_ratio:
#                     logger.debug("Could not find enough negatives for phage %d after %d attempts", src_global, max_attempts)

#             if len(neg_src_list) == 0:
#                 continue

#             chunk_size = 2048
#             neg_loss_total = 0.0
#             neg_src_tensor = torch.tensor(neg_src_list, device=device)
#             neg_dst_tensor = torch.tensor(neg_dst_list, device=device)
#             n_chunks = math.ceil(len(neg_src_tensor) / chunk_size)

#             for i in range(n_chunks):
#                 start = i * chunk_size
#                 end = min((i + 1) * chunk_size, len(neg_src_tensor))
#                 neg_chunk = (neg_src_tensor[start:end], neg_dst_tensor[start:end])
#                 neg_scores = model.decode(out, neg_chunk, etype=relation)
#                 labels_neg = torch.zeros_like(neg_scores)
#                 neg_loss_total += loss_fn(neg_scores, labels_neg) * (end - start) / len(neg_src_tensor)  # Weighted average

#             loss = loss_fn(pos_scores, labels_pos) + neg_loss_total
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
#                 edge_type_weight_map = {
#                 # 你指定的特殊权重：
#                 ('phage', 'infects', 'host'): 1.0,
#                 ('protein', 'similar', 'protein'): 1.0,

#                 # 其他边类型默认设为 1.0（中性影响）
#                 ('host', 'has_sequence', 'host_sequence'): 1.0,
#                 ('phage', 'interacts', 'phage'): 1.0,
#                 ('host', 'interacts', 'host'): 1.0,
#                 ('phage', 'encodes', 'protein'): 1.0,
#                 ('host', 'encodes', 'protein'): 1.0,
#                 ('host', 'belongs_to', 'taxonomy'): 1.0,
#                 ('taxonomy', 'related', 'taxonomy'): 1.0,
#             }
#                 save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file=pred_file,k_list=(10,),edge_type_weight_map=edge_type_weight_map )

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
#                     'val_auc': val_auc,
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
#     edge_type_weight_map = {
#         # 你指定的特殊权重：
#         ('phage', 'infects', 'host'): 1.0,
#         ('protein', 'similar', 'protein'): 1.0,

#         # 其他边类型默认设为 1.0（中性影响）
#         ('host', 'has_sequence', 'host_sequence'): 1.0,
#         ('phage', 'interacts', 'phage'): 1.0,
#         ('host', 'interacts', 'host'): 1.0,
#         ('phage', 'encodes', 'protein'): 1.0,
#         ('host', 'encodes', 'protein'): 1.0,
#         ('host', 'belongs_to', 'taxonomy'): 1.0,
#         ('taxonomy', 'related', 'taxonomy'): 1.0,
#     }
#     save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file="phage_prediction_results_final.tsv",k_list=(10,),edge_type_weight_map=edge_type_weight_map)
# -------------------------
# Training entry (modified: add taxonomy & hard negative sampling)
# -------------------------
import argparse
import os
import time
import math
import random
import json
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pt", required=True, help="Path to .pt file containing HeteroData or (data, split_edge)")
    p.add_argument("--taxid2species_tsv", default=None, help="Optional TSV mapping taxid -> species")
    p.add_argument("--taxonomy_tsv", default=None, help="taxonomy_with_alias TSV (taxid,parent_taxid,name,rank,alias) for climbing to genus")
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
    p.add_argument("--hard_neg_ratio", type=int, default=None, help="if set, #hard negatives per positive (falls back to neg_ratio)")
    p.add_argument("--use_hard_neg", action='store_true', help="enable same-genus hard negative sampling in training")
    p.add_argument("--save_path", default="best_hgt_nb.pt")
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()

def load_taxonomy_parents(taxonomy_tsv: str):
    """
    Read taxonomy_with_alias TSV and return maps:
      parent_of[taxid] = parent_taxid (int)
      rank_of[taxid] = rank (str)
    """
    parent_of = {}
    rank_of = {}
    if taxonomy_tsv is None:
        return parent_of, rank_of

    df_tax = pd.read_csv(taxonomy_tsv, sep="\t", dtype={"taxid":str, "parent_taxid":str, "rank":str}, keep_default_na=False)
    for _, row in df_tax.iterrows():
        try:
            tid = int(row["taxid"])
            parent = int(row["parent_taxid"]) if row["parent_taxid"] != "" else tid
        except Exception:
            continue
        parent_of[tid] = parent
        rank_of[tid] = str(row.get("rank", "")).lower()
    return parent_of, rank_of

def find_ancestor_of_rank(taxid: int, parent_of: dict, rank_of: dict, target_rank: str = "genus", max_hops: int = 50):
    """
    Walk up parent_of until rank_of[taxid] == target_rank, or stop.
    Return found taxid (int) or -1 if not found.
    """
    if taxid is None:
        return -1
    cur = int(taxid)
    hops = 0
    visited = set()
    while hops < max_hops:
        r = rank_of.get(cur, "")
        if r == target_rank:
            return cur
        parent = parent_of.get(cur, cur)
        if parent == cur or parent in visited:
            break
        visited.add(cur)
        cur = parent
        hops += 1
    return -1

def main():
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

    # Load taxid->species mapping if provided (unchanged)
    taxid2species = None
    host_id2taxid = None
    if args.taxid2species_tsv:
        taxmap = pd.read_csv(args.taxid2species_tsv, sep="\t")
        taxid2species = dict(zip(taxmap["taxid"], taxmap["species"]))
        if hasattr(data['host'], 'taxid'):
            host_id2taxid = data['host'].taxid.cpu().numpy()
        else:
            logger.warning("data['host'] missing .taxid, species-level eval disabled")

    # Build host_idx2gcf (try to obtain from node_maps_RBP.json; fallback to data['host'].id or to indices)
    host_idx2gcf = None
    try:
        with open("node_maps_RBP.json", "r", encoding="utf-8") as f:
            node_maps = json.load(f)
        host_map = node_maps.get("host_map", {})
        # host_map: original_id_string -> idx
        host_idx2gcf = {int(v): str(k) for k, v in host_map.items()}
        logger.info("Loaded host_idx2gcf from node_maps_RBP.json")
    except Exception:
        # fallback: if data['host'] has an attribute containing gcf strings, use it (rare)
        if hasattr(data['host'], 'gcf'):
            try:
                host_idx2gcf = {i: str(x) for i, x in enumerate(data['host'].gcf)}
            except Exception:
                host_idx2gcf = None
        else:
            host_idx2gcf = None
            logger.info("No node_maps_RBP.json and no data['host'].gcf. Outputs will show index if gcf missing.")

    # ---------------------
    # Taxonomy: compute host -> genus mapping (global-level)
    # ---------------------
    parent_of, rank_of = load_taxonomy_parents(args.taxonomy_tsv)
    host_genus_map = {}  # global_host_idx -> genus_taxid (int) or -1
    if args.use_hard_neg:
        if not hasattr(data['host'], 'taxid'):
            logger.warning("use_hard_neg requested but data['host'] lacks .taxid. Falling back to random negatives.")
            args.use_hard_neg = False
        elif args.taxonomy_tsv is None:
            logger.warning("use_hard_neg requested but no taxonomy_tsv provided. Falling back to random negatives.")
            args.use_hard_neg = False
        else:
            # compute genus for each global host id
            host_taxid_arr = data['host'].taxid.cpu().numpy().tolist()
            for gid, spec_taxid in enumerate(host_taxid_arr):
                try:
                    spec_taxid = int(spec_taxid)
                except Exception:
                    host_genus_map[gid] = -1
                    continue
                genus_tax = find_ancestor_of_rank(spec_taxid, parent_of, rank_of, target_rank="genus")
                host_genus_map[gid] = int(genus_tax) if genus_tax is not None else -1
            logger.info("Built host_genus_map (size=%d) for hard negative sampling", len(host_genus_map))

    # ---------------------
    # Build train_pos_edges global set and per-phage mapping for quick filtering
    # ---------------------
    train_pos_edges = set(zip(train_src_cpu.tolist(), train_dst_cpu.tolist()))
    train_pos_map_global = defaultdict(set)
    for s, d in train_pos_edges:
        train_pos_map_global[int(s)].add(int(d))

    # ---------------------
    # instantiate model (unchanged)
    # ---------------------
    edge_type_weight_map = {
        ('phage', 'infects', 'host'): 1.0,
        ('protein', 'similar', 'protein'): 1.0,
        ('host', 'has_sequence', 'host_sequence'): 1.0,
        ('phage', 'interacts', 'phage'): 1.0,
        ('host', 'interacts', 'host'): 1.0,
        ('phage', 'encodes', 'protein'): 1.0,
        ('host', 'encodes', 'protein'): 1.0,
        ('host', 'belongs_to', 'taxonomy'): 1.0,
        ('taxonomy', 'related', 'taxonomy'): 1.0,
    }
    model = GATv2MiniModel(metadata=data.metadata(), in_dims=in_dims,
                         hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                         n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, decoder="cosine",use_edge_attr=True,edge_attr_dim=1).to(device)

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

    # ---------------------
    # Training loop: modified negative sampling to incorporate same-genus hard negatives
    # ---------------------
    best_val_auc = -1.0
    best_val_mrr = -1.0
    best_ckpt = None

    # default hard_neg_ratio
    if args.hard_neg_ratio is None:
        hard_neg_ratio = args.neg_ratio
    else:
        hard_neg_ratio = args.hard_neg_ratio

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            edge_index_dict = batch.edge_index_dict

            # batch-level edge_attr (unchanged)
            batch_edge_attr = {}
            for et in batch.edge_types:
                if hasattr(batch[et], 'edge_weight') and batch[et].edge_weight is not None:
                    batch_edge_attr[et] = batch[et].edge_weight.to(batch[et].edge_index.device)
                elif et in edge_type_weight_map:
                    E = batch[et].edge_index.size(1)
                    batch_edge_attr[et] = torch.full((E,), float(edge_type_weight_map[et]), device=batch[et].edge_index.device)
                else:
                    pass
            edge_attr_arg = batch_edge_attr if len(batch_edge_attr) > 0 else None

            out = model(x_dict, edge_index_dict, edge_attr_dict=edge_attr_arg)

            edge_label_index = batch[relation].edge_label_index
            if edge_label_index is None or edge_label_index.numel() == 0:
                continue

            pos_scores = model.decode(out, edge_label_index, etype=relation)
            labels_pos = torch.ones_like(pos_scores)

            # Negative sampling (filtered by train_pos_edges only) --- MODIFIED for hard negatives
            phage_nid = batch['phage'].n_id.cpu().tolist()   # local->global mapping list
            host_nid = batch['host'].n_id.cpu().tolist()     # local->global mapping list
            host_n = len(host_nid)

            neg_src_list, neg_dst_list = [], []

            # Precompute mapping local->global for convenience
            local_to_global_host = {local: g for local, g in enumerate(host_nid)}

            # For each positive local pair in this batch, sample negatives
            for src_local in edge_label_index[0].cpu().tolist():
                src_local = int(src_local)
                src_global = phage_nid[src_local]
                count = 0
                attempts = 0
                max_attempts = max(args.neg_ratio, hard_neg_ratio) * 50

                # determine true global hosts for this phage from training positives
                true_globals = train_pos_map_global.get(src_global, set())

                # collect candidate local indices in batch that satisfy same-genus and are not positives
                candidates_same_genus = []
                if args.use_hard_neg and len(true_globals) > 0:
                    # union of genus taxids for this phage's true hosts
                    true_genera = set()
                    for tg in true_globals:
                        g_tax = host_genus_map.get(tg, -1)
                        if g_tax != -1:
                            true_genera.add(g_tax)
                    if len(true_genera) > 0:
                        for local_idx, g_global in local_to_global_host.items():
                            if g_global in true_globals:
                                continue
                            g_genus = host_genus_map.get(g_global, -1)
                            if g_genus in true_genera:
                                # candidate hard negative (local index)
                                candidates_same_genus.append(local_idx)

                # try to sample hard negatives first up to hard_neg_ratio
                needed = args.neg_ratio
                # sample from candidates_same_genus without replacement
                if args.use_hard_neg and len(candidates_same_genus) > 0:
                    random.shuffle(candidates_same_genus)
                    use_cnt = min(len(candidates_same_genus), hard_neg_ratio)
                    sampled = candidates_same_genus[:use_cnt]
                    for local_dst in sampled:
                        # ensure (src_global, dst_global) not a training positive
                        dst_global = local_to_global_host[local_dst]
                        if (src_global, dst_global) in train_pos_edges:
                            continue
                        neg_src_list.append(src_local)
                        neg_dst_list.append(local_dst)
                        count += 1
                        if count >= needed:
                            break

                # if not enough negatives yet, fallback to random sampling among batch hosts (as original)
                while count < needed and attempts < max_attempts:
                    dst_local = random.randrange(host_n)
                    dst_global = host_nid[dst_local]
                    if (src_global, dst_global) not in train_pos_edges:
                        neg_src_list.append(src_local)
                        neg_dst_list.append(dst_local)
                        count += 1
                    attempts += 1
                if count < needed:
                    logger.debug("Could not find enough negatives for phage %d after %d attempts (got %d)", src_global, max_attempts, count)

            if len(neg_src_list) == 0:
                continue

            # chunked negative scoring as before
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
                neg_loss_total += loss_fn(neg_scores, labels_neg) * (end - start) / len(neg_src_tensor)

            loss = loss_fn(pos_scores, labels_pos) + neg_loss_total
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        t1 = time.time()
        avg_loss = epoch_loss / max(1, n_batches)

        if epoch % args.log_every == 0 or epoch == args.epochs:
            try:
                # pass host_idx2gcf so evaluation uses GCF matching
                train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
                    model, data, train_pair, val_pair, test_pair, relation=relation,
                    eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
                    host_id2taxid=host_id2taxid, taxid2species=taxid2species,
                    host_idx2gcf=host_idx2gcf,
                    k_list=(1, 5, 10), save_path="phage_prediction_results"
                )
                train_auc, train_mrr, train_hits = train_metrics
                val_auc, val_mrr, val_hits = val_metrics
                test_auc, test_mrr, test_hits = test_metrics

                pred_file = f"phage_prediction_results_epoch_{epoch}.tsv"
                save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file=pred_file, top_k=10, k_list=(10,), edge_type_weight_map=edge_type_weight_map, edge_attr_dict=None)
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

    # end epochs

    if best_ckpt is not None:
        model.load_state_dict(best_ckpt['model_state'])
    logger.info("Evaluating final test...")

    # final eval with host_idx2gcf
    train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
        model, data, train_pair, val_pair, test_pair, relation=relation,
        eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
        host_id2taxid=host_id2taxid, taxid2species=taxid2species,
        host_idx2gcf=host_idx2gcf,
        k_list=(1, 5, 10), save_path="phage_prediction_results"
    )
    logger.info("FINAL TEST metrics (AUC, MRR, Hits@1/5/10): %s", test_metrics)

    # Final predictions (include host_gcf via node_maps if available)
    save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, output_file="phage_prediction_results_final.tsv", top_k=10, k_list=(10,), edge_type_weight_map=edge_type_weight_map, edge_attr_dict=None)

if __name__ == "__main__":
    main()




'''
python train_GAT_phage_host.py \
  --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP.pt \
  --taxonomy_tsv /home/wangjingyuan/wys/wys_shiyan/taxonomy_with_alias.tsv \
  --device cuda \
  --eval_device cpu \
  --epochs 1 \
  --hidden_dim 128 \
  --out_dim 128 \
  --n_layers 2 \
  --n_heads 1 \
  --num_neighbors 15 10 \
  --batch_size 256 \
  --neg_ratio 2 \
  --hard_neg_ratio 2 \
  --use_hard_neg \
  --eval_neg_ratio 1 \
  --save_path best_hgt_nb_testpt \
  --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv \
  --dropout 0.2 \
  --log_every 1

'''

