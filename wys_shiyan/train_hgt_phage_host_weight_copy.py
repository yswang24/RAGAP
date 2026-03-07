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
        # __init__ 末尾处（logit_scale 后）新增
        self.rel_logw = nn.ParameterDict()
        # 可选：传入一个初始化map，没传则默认1.0
        rel_init_map = getattr(self, "rel_init_map", None)

        for (src, rel, dst) in self.edge_types:
            if rel_init_map is not None and (src, rel, dst) in rel_init_map:
                init_w = float(rel_init_map[(src, rel, dst)])
            else:
                # 没给就用 1.0；你也可以在这里写死一些常用先验
                init_w = 1.0
            p = nn.Parameter(torch.log(torch.tensor(init_w, dtype=torch.float)))
            self.rel_logw[f"{src}__{rel}__{dst}"] = p
    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: typing.Optional[dict] = None,   # etype -> tensor/float
    ) -> dict[str, torch.Tensor]:

        # 1) 节点特征输入投影
        h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}

        # 2) 逐层消息传递
        for layer in self.layers:
            if self.use_edge_attr:
                processed = {}
                for etype, edge_index in edge_index_dict.items():
                    E = edge_index.size(1)
                    key = f"{etype[0]}__{etype[1]}__{etype[2]}"
                    # 可学习的“关系门控”，标量 > 0
                    alpha = torch.exp(self.rel_logw[key])  # shape: (), 标量

                    # 逐边数值 v_e：若传了 edge_attr_dict[etype] 就用它，否则用 1
                    if edge_attr_dict is not None and etype in edge_attr_dict:
                        v = edge_attr_dict[etype]
                        if not torch.is_tensor(v):
                            # 标量 -> (E, edge_attr_dim)
                            v = torch.full(
                                (E, self.edge_attr_dim),
                                float(v),
                                dtype=torch.float,
                                device=edge_index.device
                            )
                        else:
                            v = v.to(edge_index.device)
                            # 统一形状为 (E, edge_attr_dim)
                            if v.dim() == 1:
                                v = v.view(-1, 1)  # -> (E,1)
                            elif v.dim() == 2:
                                if v.size(1) != self.edge_attr_dim:
                                    # 若维度不对，截断或pad；这里选择截断到需要的维度
                                    if v.size(1) > self.edge_attr_dim:
                                        v = v[:, :self.edge_attr_dim]
                                    else:
                                        pad = self.edge_attr_dim - v.size(1)
                                        v = F.pad(v, (0, pad), value=1.0)  # 用1补齐
                            else:
                                raise RuntimeError(f"edge_attr for {etype} must be 1D or 2D tensor")
                            if v.size(0) != E:
                                raise RuntimeError(f"edge_attr for {etype} length {v.size(0)} != edge count {E}")
                    else:
                        # 没传逐边数值 -> 用全1
                        v = torch.ones((E, self.edge_attr_dim), device=edge_index.device)

                    # 关系门控乘逐边数值 -> 最终喂给 GATv2
                    # alpha 标量 -> 扩展成 (E, edge_attr_dim) 参与广播
                    processed[etype] = v * alpha

                # 以 *_dict 格式传入，HeteroConv 会把它展开为 edge_attr=...
                h = layer(h, edge_index_dict, edge_attr_dict=processed)
            else:
                # 不使用边特征
                h = layer(h, edge_index_dict)

            # 激活 + dropout
            for k in list(h.keys()):
                h[k] = F.relu(self.dropout(h[k]))

        # 3) 输出投影 + L2 归一化（配合 cosine 解码）
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
    save_path: Optional[str] = None,
    top_k: int = 10,
    node_maps_path: str = "node_maps.json",                       # <- NEW
    edge_type_weight_map: typing.Optional[dict] = None            # <- NEW
 ) -> tuple[tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]]]:
    """
    Compute full-graph metrics (AUC, MRR, Hits@K) and optionally save top-k predictions.
    node_maps_path: path to JSON with phage_map / host_map
    edge_type_weight_map: optional mapping {(src,rel,dst): scalar_weight, ...}
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
    # Load phage/host idx -> real id (use provided node_maps_path)
    # ======================
    try:
        with open(node_maps_path, "r", encoding="utf-8") as f:
            node_maps = json.load(f)
        phage_map = node_maps.get("phage_map", {})
        host_map = node_maps.get("host_map", {})
        phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
        host_idx2id = {int(v): str(k) for k, v in host_map.items()}
    except FileNotFoundError:
        logger.warning("%s not found, using index as ID", node_maps_path)
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

        # ======================
        # Build global edge_attr using provided edge_type_weight_map if present
        # ======================
        global_edge_attr = {}
        edge_type_weight_map = edge_type_weight_map or {}
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

        out = model(data_eval_x, edge_index_dict_eval, edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None)

        n_hosts = out['host'].size(0)

        # ======================
        # helpers (unchanged)
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

        effective_top_k = max(top_k, max(k_list))

        # ======================
        # AUC 计算（保留原逻辑；可后续向量化）
        # ======================
        def compute_auc_for_pairs(pairs: tuple[torch.Tensor, torch.Tensor], pos_map: defaultdict[int, set[int]]) -> float:
            src_cpu, dst_cpu = pairs
            if src_cpu.numel() == 0:
                return float('nan')

            pos_scores = []
            neg_scores_list = []
            for s_idx, d_idx in zip(src_cpu.tolist(), dst_cpu.tolist()):
                if not (0 <= d_idx < n_hosts):
                    continue

                pos_score = torch.sigmoid(
                    model.decode(out, (torch.tensor([s_idx], device=eval_device),
                                       torch.tensor([d_idx], device=eval_device)),
                                 etype=relation)
                ).item()
                pos_scores.append(pos_score)

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
        # Vectorized path when decoder == "cosine"
        # ======================
        def compute_rank_metrics(pairs: tuple[torch.Tensor, torch.Tensor], save_path: Optional[str] = None):
            src_cpu, dst_cpu = pairs
            if src_cpu.numel() == 0:
                return 0.0, {k: 0.0 for k in k_list}

            ph2hosts = defaultdict(list)
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                ph2hosts[int(s)].append(int(d))

            hits = {k: 0 for k in k_list}
            total_q = len(ph2hosts)
            prediction_rows = []

            # Fast path for cosine decoder (out embeddings already L2-normalized in model.forward)
            if getattr(model, "decoder_type", None) == "cosine":
                # out['phage'] and out['host'] are normalized; use matrix multiplication
                ph_indices = list(ph2hosts.keys())
                if len(ph_indices) == 0:
                    return 0.0, {k: 0.0 for k in k_list}
                ph_emb = out['phage'][ph_indices]              # (P, D)
                host_emb = out['host']                         # (n_hosts, D)
                # scores: (P, n_hosts)
                scores = torch.matmul(ph_emb, host_emb.t()) * torch.exp(model.logit_scale)
                scores_np = scores.cpu().numpy()
                # topk
                topk_idx = np.argpartition(-scores_np, range(min(effective_top_k, scores_np.shape[1])), axis=1)[:, :effective_top_k]
                # sort topk rows for ranks
                row_indices = np.arange(scores_np.shape[0])[:, None]
                topk_scores = scores_np[row_indices, topk_idx]
                order = np.argsort(-topk_scores, axis=1)
                topk_sorted_idx = topk_idx[row_indices, order].reshape(scores_np.shape[0], effective_top_k)

                # build predictions + compute MRR/Hits
                for i_row, ph_global in enumerate(ph_indices):
                    topk_for_ph = topk_sorted_idx[i_row]
                    phage_real_id = phage_idx2id.get(ph_global, str(ph_global))
                    true_species = {hostid2species(h) for h in ph2hosts[ph_global]}
                    # record top-k rows
                    for rank_pos, h_idx in enumerate(topk_for_ph, 1):
                        host_real_id = host_idx2id.get(int(h_idx), str(int(h_idx)))
                        host_species_name = hostid2species(int(h_idx))
                        score = float(scores_np[i_row, int(h_idx)])
                        prediction_rows.append({
                            "phage_id": phage_real_id,
                            "rank": rank_pos,
                            "host_id": host_real_id,
                            "host_species": host_species_name,
                            "score": score
                        })
                    # reciprocal rank
                    rank_val = None
                    for pos, h in enumerate(topk_for_ph, 1):
                        if hostid2species(int(h)) in true_species:
                            rank_val = pos
                            break
                    if rank_val is None:
                        rank_val = effective_top_k + 1
                    # update rr and hits
                    # we'll compute rr_sum separately
                    rr = 1.0 / rank_val
                    # accumulate hits
                    for k in k_list:
                        if k <= effective_top_k:
                            if any(hostid2species(int(h)) in true_species for h in topk_for_ph[:k]):
                                hits[k] += 1
                    # store rr by adding to a running list
                    # (we'll average below)
                    # to avoid extra storage, we accumulate rr_sum here:
                    if 'rr_sum_acc' not in locals():
                        rr_sum_acc = rr
                    else:
                        rr_sum_acc += rr

                mrr = rr_sum_acc / total_q if total_q > 0 else 0.0
                hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}

            else:
                # fallback: original per-phage loop (keeps previous behavior for mlp decoder)
                rr_sum = 0.0
                for ph_idx, true_ds in ph2hosts.items():
                    scores_tensor = torch.sigmoid(
                        model.decode(out, (torch.full((n_hosts,), ph_idx, device=eval_device),
                                           torch.arange(n_hosts, device=eval_device)),
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

                mrr = rr_sum / total_q if total_q > 0 else 0.0
                hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}

            # save predictions if requested
            if save_path is not None and len(prediction_rows) > 0:
                pd.DataFrame(prediction_rows).to_csv(save_path, sep="\t", index=False)

            return mrr, hits_at

        # ======================
        # run metrics exactly as before (but now accept node_maps_path & edge_type_weight_map)
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

# def save_predictions(
#     # model: HGTMiniModel,
#     model: GATv2MiniModel,
#     data: HeteroData,
#     test_src_cpu: torch.Tensor,
#     test_dst_cpu: torch.Tensor,
#     relation: tuple[str, str, str],
#     eval_device: torch.device,
#     host_id2taxid: typing.Union[np.ndarray, None] = None,
#     taxid2species: typing.Union[dict[int, str], None] = None,
#     output_file: str = "phage_prediction_results.tsv",
#     top_k: int = 10,  # control top-k in output
#     k_list=(1, 5, 10),
#     edge_type_weight_map: typing.Optional[dict] = None,   # scalar weight map {(src,rel,dst): scalar}
#     edge_attr_dict: typing.Optional[dict] = None,        # explicit per-edge edge_attr (priority)
#     node_maps_path: str = "node_maps.json"               # <- NEW: unified node maps path
#  ):
#     """
#     保存 top-k 的 phage->host 预测。
#     参数说明（新增/重要）:
#       - edge_attr_dict: 显式传入的 edge_attr（优先）；可以是 {(src,rel,dst): scalar_or_tensor, ...}
#       - edge_type_weight_map: 标量映射 {(src,rel,dst): scalar, ...}（用于填充未提供的 edge_attr）
#       - node_maps_path: JSON 文件路径，包含 "phage_map" 和 "host_map"（用于 id 反查）
#     其余行为与原函数保持一致（按 phage 分组，输出 top_k）。
#     """
#     orig_dev = next(model.parameters()).device
#     moved = False
#     if orig_dev != eval_device:
#         model.to(eval_device)
#         moved = True

#     try:
#         # 全图节点特征与边索引（移动到 eval_device）
#         full_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
#         full_edge_index_dict = {
#             et: data[et].edge_index.to(eval_device)
#             for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
#         }

#         # --- 构建 global_edge_attr ---
#         global_edge_attr = {}
#         edge_type_weight_map = edge_type_weight_map or {}

#         for et, edge_index in full_edge_index_dict.items():
#             E = edge_index.size(1)
#             # 1) 显式传入优先
#             if edge_attr_dict is not None and et in edge_attr_dict:
#                 val = edge_attr_dict[et]
#                 if isinstance(val, (float, int)):
#                     global_edge_attr[et] = torch.full((E,), float(val), device=eval_device)
#                 elif isinstance(val, torch.Tensor):
#                     t = val.to(eval_device)
#                     if t.dim() == 1 and t.size(0) == E:
#                         global_edge_attr[et] = t
#                     elif t.dim() == 2 and t.size(0) == E:
#                         global_edge_attr[et] = t
#                     else:
#                         raise RuntimeError(f"edge_attr for {et} has invalid shape {tuple(t.size())}, expected (E,) or (E, dim).")
#                 else:
#                     raise RuntimeError(f"Unsupported edge_attr type for {et}: {type(val)}")
#             # 2) 数据里已有 per-edge weight
#             elif hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
#                 w = data[et].edge_weight.to(eval_device)
#                 if w.dim() == 0:
#                     w = w.expand(E)
#                 elif w.dim() == 1 and w.size(0) != E:
#                     raise RuntimeError(f"data[{et}].edge_weight length {w.size(0)} != E({E})")
#                 global_edge_attr[et] = w
#             # 3) edge_type_weight_map 标量填充
#             elif et in edge_type_weight_map:
#                 w = float(edge_type_weight_map[et])
#                 global_edge_attr[et] = torch.full((E,), w, device=eval_device)
#             else:
#                 # 未提供 edge_attr -> 不包含该 etype（模型会走无 edge_attr 的分支）
#                 pass

#         edge_attr_arg = global_edge_attr if len(global_edge_attr) > 0 else None

#         # --- Forward (eval) ---
#         model.eval()
#         out_full = model(full_x, full_edge_index_dict, edge_attr_dict=edge_attr_arg)

#         # 计算 test 对应的分数（保持原先行为）
#         edge_index_full = torch.stack(
#             [test_src_cpu.to(eval_device), test_dst_cpu.to(eval_device)], dim=0
#         )
#         scores_tensor = torch.sigmoid(model.decode(out_full, edge_index_full, etype=relation))
#         scores = scores_tensor.cpu().detach().numpy()

#         # Load node maps using provided node_maps_path (统一)
#         try:
#             with open(node_maps_path, "r", encoding="utf-8") as f:
#                 node_maps = json.load(f)
#             phage_map = node_maps.get("phage_map", {})
#             host_map = node_maps.get("host_map", {})
#             phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
#             host_idx2id = {int(v): str(k) for k, v in host_map.items()}
#         except Exception:
#             logger = logging.getLogger(__name__)
#             logger.warning("%s not found or invalid JSON, using index as ID", node_maps_path)
#             phage_idx2id = {}
#             host_idx2id = {}

#         phage_ids = [phage_idx2id.get(int(nid), str(nid)) for nid in test_src_cpu.tolist()]
#         host_ids = [host_idx2id.get(int(nid), str(nid)) for nid in test_dst_cpu.tolist()]

#         host_species = ["NA"] * len(host_ids)
#         if host_id2taxid is not None and taxid2species is not None:
#             host_species = [
#                 taxid2species.get(int(host_id2taxid[int(nid)]), "NA")
#                 for nid in test_dst_cpu.tolist()
#             ]

#         # 按 phage 分组并取 top-k
#         phage2preds = {}
#         for pid, hid, hs, s in zip(phage_ids, host_ids, host_species, scores):
#             if pid not in phage2preds:
#                 phage2preds[pid] = []
#             phage2preds[pid].append((hid, hs, s))

#         with open(output_file, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f, delimiter="\t")
#             writer.writerow(["phage_id", "rank", "host_id", "host_species", "score"])

#             for pid, preds in phage2preds.items():
#                 preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)[:top_k]
#                 for rank, (hid, hs, s) in enumerate(preds_sorted, 1):
#                     writer.writerow([pid, rank, hid, hs, float(s)])

#         logger = logging.getLogger(__name__)
#         logger.info(f"Top-{top_k} Phage-host prediction results saved to {output_file}")

#     finally:
#         if moved:
#             model.to(orig_dev)
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
def save_predictions(
    model: GATv2MiniModel,
    data: HeteroData,
    test_src_cpu: torch.Tensor,
    test_dst_cpu: torch.Tensor,
    relation: tuple[str, str, str],
    eval_device: torch.device,
    host_id2taxid: typing.Union[np.ndarray, None] = None,
    taxid2species: typing.Union[dict[int, str], None] = None,
    output_file: str = "phage_prediction_results.tsv",
    top_k: int = 10,  # control top-k in output
    k_list=(1, 5, 10),
    edge_type_weight_map: typing.Optional[dict] = None,
    edge_attr_dict: typing.Optional[dict] = None,
    node_maps_path: str = "node_maps.json"
 ):
    """
    保存 top-k 的 phage->host 预测（向量化 / 分批两种策略）。
    - 如果 model.decoder_type == "cosine": 使用一次性矩阵乘法计算 (num_phage, num_host) scores。
    - 否则：对每个 phage 分批调用 model.decode（避免 OOM）。
    """
    orig_dev = next(model.parameters()).device
    moved = False
    if orig_dev != eval_device:
        model.to(eval_device)
        moved = True

    try:
        # 全图节点特征与边索引（移动到 eval_device）
        full_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
        full_edge_index_dict = {
            et: data[et].edge_index.to(eval_device)
            for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
        }

        # --- 构建 global_edge_attr (same logic as before) ---
        global_edge_attr = {}
        edge_type_weight_map = edge_type_weight_map or {}
        for et, edge_index in full_edge_index_dict.items():
            E = edge_index.size(1)
            if edge_attr_dict is not None and et in edge_attr_dict:
                val = edge_attr_dict[et]
                if isinstance(val, (float, int)):
                    global_edge_attr[et] = torch.full((E,), float(val), device=eval_device)
                elif isinstance(val, torch.Tensor):
                    t = val.to(eval_device)
                    if t.dim() == 0:
                        t = t.expand(E)
                    if t.dim() == 1 and t.size(0) != E:
                        raise RuntimeError(f"edge_attr for {et} has length {t.size(0)} != E({E})")
                    global_edge_attr[et] = t
                else:
                    raise RuntimeError(f"Unsupported edge_attr type for {et}: {type(val)}")
            elif hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
                w = data[et].edge_weight.to(eval_device)
                if w.dim() == 0:
                    w = w.expand(E)
                elif w.dim() == 1 and w.size(0) != E:
                    raise RuntimeError(f"data[{et}].edge_weight length {w.size(0)} != E({E})")
                global_edge_attr[et] = w
            elif et in edge_type_weight_map:
                w = float(edge_type_weight_map[et])
                global_edge_attr[et] = torch.full((E,), w, device=eval_device)
            else:
                pass

        edge_attr_arg = global_edge_attr if len(global_edge_attr) > 0 else None

        # --- Forward (eval) to get embeddings for all nodes ---
        model.eval()
        with torch.no_grad():
            out_full = model(full_x, full_edge_index_dict, edge_attr_dict=edge_attr_arg)

            # all host embeddings and phage embeddings (full graph)
            phage_emb_all = out_full['phage']  # (N_phage, D)
            host_emb_all = out_full['host']    # (N_host, D)

            # Unique phage ids we need to output predictions for:
            # We will output per unique phage present in test_src_cpu
            uniq_phage_idx = torch.unique(test_src_cpu).to(eval_device)  # global phage indices (CPU -> move)
            # Map to int list for iteration / output ordering
            uniq_phage_list = uniq_phage_idx.cpu().tolist()

            # Load node maps
            try:
                with open(node_maps_path, "r", encoding="utf-8") as f:
                    node_maps = json.load(f)
                phage_map = node_maps.get("phage_map", {})
                host_map = node_maps.get("host_map", {})
                phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
                host_idx2id = {int(v): str(k) for k, v in host_map.items()}
            except Exception:
                logger = logging.getLogger(__name__)
                logger.warning("%s not found or invalid JSON, using index as ID", node_maps_path)
                phage_idx2id = {}
                host_idx2id = {}

            # host species mapping (if available)
            host_species_lookup = None
            if host_id2taxid is not None and taxid2species is not None:
                host_species_lookup = (host_id2taxid, taxid2species)

            phage2preds = {}

            if model.decoder_type == "cosine":
                # Vectorized scoring: (P_subset, D) @ (N_host, D).T -> (P_subset, N_host)
                # Use float32 on device; multiply by exp(logit_scale) to match decode behavior.
                scale = float(torch.exp(model.logit_scale).to(eval_device)) if hasattr(model, "logit_scale") else 1.0

                # To avoid very large memory, we can chunk phages (and/or hosts) if needed.
                # We'll iterate over phage chunks.
                ph_chunk = 256  # adjust if OOM; increase if memory allows
                N_host = host_emb_all.size(0)
                for i in range(0, len(uniq_phage_list), ph_chunk):
                    ph_idxs = uniq_phage_list[i:i+ph_chunk]
                    ph_tensor = phage_emb_all[torch.tensor(ph_idxs, device=eval_device)]  # (P_chunk, D)
                    # scores matrix:
                    scores_mat = torch.matmul(ph_tensor, host_emb_all.t()) * scale  # (P_chunk, N_host)
                    # convert to probabilities consistent with other code (you often used sigmoid)
                    probs_mat = torch.sigmoid(scores_mat)  # (P_chunk, N_host)

                    # get top-k per phage row
                    k = min(top_k, N_host)
                    top_vals, top_idxs = torch.topk(probs_mat, k=k, dim=1)
                    top_vals = top_vals.cpu().numpy()
                    top_idxs = top_idxs.cpu().numpy()

                    for r, pid in enumerate(ph_idxs):
                        ph_real = phage_idx2id.get(int(pid), str(int(pid)))
                        row_vals = top_vals[r]
                        row_idxs = top_idxs[r]
                        phage2preds[ph_real] = []
                        for rank_i, (hid_idx, score_val) in enumerate(zip(row_idxs.tolist(), row_vals.tolist()), start=1):
                            host_real = host_idx2id.get(int(hid_idx), str(int(hid_idx)))
                            host_sp = "NA"
                            if host_species_lookup is not None:
                                hid_global = int(hid_idx)
                                taxid_arr, taxmap = host_species_lookup
                                host_sp = taxmap.get(int(taxid_arr[hid_global]), "NA") if hid_global < len(taxid_arr) else "NA"
                            phage2preds[ph_real].append((host_real, host_sp, float(score_val)))

            else:
                # Non-cosine decoder: score each phage against all hosts but do it in batches
                # to avoid OOM; call model.decode on chunks of host indices.
                N_host = host_emb_all.size(0)
                host_idx_tensor_all = torch.arange(N_host, device=eval_device)
                host_chunk = 1024  # adjust
                for pid in uniq_phage_list:
                    ph_real = phage_idx2id.get(int(pid), str(int(pid)))
                    phage2preds[ph_real] = []
                    # compute scores against all hosts in chunks
                    for start in range(0, N_host, host_chunk):
                        end = min(N_host, start + host_chunk)
                        host_chunk_idx = host_idx_tensor_all[start:end]
                        # build edge_label_index for decode: (2, H_chunk) with phage repeated
                        ph_idx_tensor = torch.full((host_chunk_idx.numel(),), int(pid), dtype=torch.long, device=eval_device)
                        edge_label_index = (ph_idx_tensor, host_chunk_idx)
                        # model.decode expects z_dict (out_full) and edge_label_index
                        logits = model.decode(out_full, edge_label_index, etype=relation)  # (H_chunk,)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        for hid_local, score_val in zip(host_chunk_idx.cpu().tolist(), probs.tolist()):
                            host_real = host_idx2id.get(int(hid_local), str(int(hid_local)))
                            host_sp = "NA"
                            if host_species_lookup is not None:
                                taxid_arr, taxmap = host_species_lookup
                                host_sp = taxmap.get(int(taxid_arr[int(hid_local)]), "NA") if int(hid_local) < len(taxid_arr) else "NA"
                            phage2preds[ph_real].append((host_real, host_sp, float(score_val)))
                    # after collecting all hosts, keep only top_k to save memory/disk
                    phage2preds[ph_real].sort(key=lambda x: x[2], reverse=True)
                    phage2preds[ph_real] = phage2preds[ph_real][:top_k]

        # --- write out top_k per phage ---
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["phage_id", "rank", "host_id", "host_species", "score"])
            for pid, preds in phage2preds.items():
                for rank, (hid, hs, s) in enumerate(preds, start=1):
                    writer.writerow([pid, rank, hid, hs, float(s)])

        logger = logging.getLogger(__name__)
        logger.info(f"Top-{top_k} Phage-host prediction results saved to {output_file}")

    finally:
        if moved:
            model.to(orig_dev)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    # NEW: node maps path (统一默认 node_maps.json)
    p.add_argument("--node_maps", default="node_maps_cluster_650.json", help="JSON file mapping node ids (phage_map, host_map). Default: node_maps.json")
    p.add_argument("--out_dir", default="outputs", help="directory to place all outputs (checkpoints, predictions, debug files)")
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
def softmax_ce_loss(phage_emb_batch: torch.Tensor,
                     host_emb_batch: torch.Tensor,
                     pos_phage_local_idx: torch.LongTensor,
                     pos_host_local_idx: torch.LongTensor,
                     tau: float = 0.1) -> torch.Tensor:
    """
    In-batch sampled softmax cross-entropy loss for ranking.
    - phage_emb_batch: tensor (P_batch, D)
    - host_emb_batch:  tensor (H_batch, D)
    - pos_phage_local_idx: LongTensor (num_pos,) indices into phage_emb_batch
    - pos_host_local_idx:  LongTensor (num_pos,) indices into host_emb_batch (labels)
    - tau: temperature (smaller -> sharper distribution). Default 0.1 recommended.
    Returns scalar loss (on same device).
    Behavior: for each positive pair i, compute logits = phage_vec_i @ host_emb_batch.T / tau,
              and treat pos_host_local_idx[i] as the correct class (CrossEntropy).
    """
    device = phage_emb_batch.device
    # ensure indices are on same device
    pos_phage_local_idx = pos_phage_local_idx.to(device)
    pos_host_local_idx = pos_host_local_idx.to(device)

    # Select phage vectors corresponding to positive pairs (num_pos, D)
    phage_vecs = phage_emb_batch[pos_phage_local_idx]  # (N_pos, D)

    # logits: (N_pos, H_batch)
    logits = torch.matmul(phage_vecs, host_emb_batch.t())  # (N_pos, H)
    if tau != 1.0 and tau > 0.0:
        logits = logits / float(tau)

    labels = pos_host_local_idx.long()  # (N_pos,)
    # CrossEntropy expects logits (N, C) and labels in [0,C-1]
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss

def main():

    import pandas as pd
    import random
        # ---------- 在 main() 里唯一定义 edge_type_weight_map ----------
    edge_type_weight_map = {
        # ('phage', 'infects', 'host'): 2.0,
        # ('protein', 'similar', 'protein'): 1.0,
        # ('host', 'has_sequence', 'host_sequence'): 1.0,
        # ('phage', 'interacts', 'phage'): 1.0,
        # ('host', 'interacts', 'host'): 1.0,
        # ('phage', 'encodes', 'protein'): 0.5,
        # ('host', 'encodes', 'protein'): 0.5,
        # ('host', 'belongs_to', 'taxonomy'): 1.0,
        # ('taxonomy', 'related', 'taxonomy'): 1.0,
        # ('phage', 'belongs_to', 'taxonomy'): 1.0,
    }

    args = parse_args()
    # ---- create output dir and subdirs ----
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # subfolders
    debug_out_dir = os.path.join(out_dir, "debug_out")
    preds_dir = os.path.join(out_dir, "predictions")
    os.makedirs(debug_out_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    # adjust save_path to be inside out_dir (preserve basename)
    args.save_path = os.path.join(out_dir, os.path.basename(args.save_path))

    # ---- add file logger into logging (besides console) ----
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

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
    data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu,
    fix_enable=True, out_dir=debug_out_dir
    )


    in_dims = {}
    for n in data.node_types:
        if 'x' not in data[n]:
            raise RuntimeError(f"Node {n} missing .x features")
        in_dims[n] = data[n].x.size(1)
        logger.info("node %s in_dim = %d", n, in_dims[n])

    logger.info("Instantiating model...")
    # model = HGTMiniModel(metadata=data.metadata(), in_dims=in_dims,
    #                      hidden_dim=args.hidden_dim, out_dim=args.out_dim,
    #                      n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, decoder="mlp").to(device)

    model = GATv2MiniModel(metadata=data.metadata(), in_dims=in_dims,
                         hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                         n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout, decoder="cosine",use_edge_attr=True,edge_attr_dim=1).to(device)
    model.rel_init_map = edge_type_weight_map
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # optimizer = torch.optim.AdamW([
    # {"params": [p for n, p in model.named_parameters() if "logit_scale" not in n]},
    # {"params": [model.logit_scale], "lr": args.lr * 0.1}  # logit_scale 的学习率更小
    # ], lr=args.lr, weight_decay=1e-5)
    optimizer = torch.optim.AdamW([
    {"params": [p for n,p in model.named_parameters() if ("logit_scale" not in n and "rel_logw" not in n)], "lr": args.lr},
    {"params": [model.logit_scale], "lr": args.lr * 0.1},
    {"params": list(model.rel_logw.parameters()), "lr": args.lr * 0.1}
    ], weight_decay=1e-5)


    loss_fn = softmax_ce_loss  # 你自定义的对比损失


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
            #out = model({nt: batch[nt].x for nt in batch.node_types}, batch.edge_index_dict)

            # 准备 x_dict 与 edge_index_dict（你之前已有）
            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            edge_index_dict = batch.edge_index_dict  # keys are tuples (src,rel,dst)

            # 构造 batch_edge_attr: 优先使用 batch 中已有的 edge_weight，否则使用你指定的 scalar map
            batch_edge_attr = {}
            for et in batch.edge_types:
                if hasattr(batch[et], 'edge_weight') and batch[et].edge_weight is not None:
                    v = batch[et].edge_weight.to(batch[et].edge_index.device)
                    batch_edge_attr[et] = v.view(-1, 1)  # 统一二维
                elif et in edge_type_weight_map:
                    # 有“类型先验”也不用在这里生成（由模型里的 rel_logw 学），
                    # 真要乘就把它当常数 per-edge 数值：
                    E = batch[et].edge_index.size(1)
                    batch_edge_attr[et] = torch.full((E,1), float(edge_type_weight_map[et]),
                                                    device=batch[et].edge_index.device)
            # 如果你希望“类型先验”只作为初始化，不想重复乘，就干脆不填 batch_edge_attr，让模型内部乘 alpha * 1。
            edge_attr_arg = batch_edge_attr if len(batch_edge_attr)>0 else None
            out = model(x_dict, edge_index_dict, edge_attr_dict=edge_attr_arg)

            # edge_label_index = batch[relation].edge_label_index
            # if edge_label_index is None or edge_label_index.numel() == 0:
            #     continue

            # pos_scores = model.decode(out, edge_label_index, etype=relation)
            # labels_pos = torch.ones_like(pos_scores)

            # # Negative sampling (filtered by train_pos_edges only)
            # phage_nid = batch['phage'].n_id.cpu().tolist()
            # host_nid = batch['host'].n_id.cpu().tolist()
            # host_n = len(host_nid)

            # neg_src_list, neg_dst_list = [], []
            # for src_local in edge_label_index[0].cpu().tolist():
            #     src_global = phage_nid[src_local]
            #     count = 0
            #     attempts = 0
            #     max_attempts = args.neg_ratio * 20  # Increased to avoid skips
            #     while count < args.neg_ratio and attempts < max_attempts:
            #         dst_local = random.randrange(host_n)
            #         dst_global = host_nid[dst_local]
            #         if (src_global, dst_global) not in train_pos_edges:
            #             neg_src_list.append(src_local)
            #             neg_dst_list.append(dst_local)
            #             count += 1
            #         attempts += 1
            #     if count < args.neg_ratio:
            #         logger.debug("Could not find enough negatives for phage %d after %d attempts", src_global, max_attempts)

            # if len(neg_src_list) == 0:
            #     continue

            # chunk_size = 2048
            # neg_loss_total = 0.0
            # neg_src_tensor = torch.tensor(neg_src_list, device=device)
            # neg_dst_tensor = torch.tensor(neg_dst_list, device=device)
            # n_chunks = math.ceil(len(neg_src_tensor) / chunk_size)

            # for i in range(n_chunks):
            #     start = i * chunk_size
            #     end = min((i + 1) * chunk_size, len(neg_src_tensor))
            #     neg_chunk = (neg_src_tensor[start:end], neg_dst_tensor[start:end])
            #     neg_scores = model.decode(out, neg_chunk, etype=relation)
            #     labels_neg = torch.zeros_like(neg_scores)
            #     neg_loss_total += loss_fn(neg_scores, labels_neg) * (end - start) / len(neg_src_tensor)  # Weighted average

            # loss = loss_fn(pos_scores, labels_pos) + neg_loss_total
            # loss.backward()
            # ========== 替换开始：用 in-batch softmax CE 代替原先的 BCE pos/neg ==========
            edge_label_index = batch[relation].edge_label_index
            if edge_label_index is None or edge_label_index.numel() == 0:
                continue

            # out 已由 model(...) 返回，为字典：out['phage'], out['host'], ...
            # 获取本 batch 的 phage / host embedding 矩阵（局部索引空间）
            phage_emb_batch = out['phage']   # shape (P_batch, D)
            host_emb_batch = out['host']     # shape (H_batch, D)

            # positive pairs (local indices in this batch's phage/host sets)
            pos_src_local = edge_label_index[0].to(device)  # (num_pos,)
            pos_dst_local = edge_label_index[1].to(device)  # (num_pos,)

            # Compute in-batch softmax CE loss.
            # Treat host_emb_batch as the classification candidates for each phage vector.
            # temperature tau can be tuned; 0.1 is a good default to encourage sharper ranks.
            tau = 0.1
            loss = softmax_ce_loss(phage_emb_batch, host_emb_batch, pos_src_local, pos_dst_local, tau=tau)
            
            # 反向传播
            loss.backward()
            # 可选：梯度裁剪，防止梯度爆炸（数值更稳）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 更新参数
            optimizer.step()

            # 在更新后对 logit_scale 做 clamping，防止其变得过大/过小导致不稳定
            with torch.no_grad():
                model.logit_scale.clamp_(-10.0, 10.0)

            # 清除梯度（你在循环开始也做了 zero_grad，这里在 step 后再清一次是安全的）
            optimizer.zero_grad()

            epoch_loss += float(loss.item())
            n_batches += 1

        t1 = time.time()
        avg_loss = epoch_loss / max(1, n_batches)

        if epoch % args.log_every == 0 or epoch == args.epochs:
            try:
                train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
                    model, data, train_pair, val_pair, test_pair, relation=relation,
                    eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
                    host_id2taxid=host_id2taxid, taxid2species=taxid2species,k_list=(1, 5, 10), # <- 这里加上
                    save_path=os.path.join(preds_dir, "phage_prediction_results"),node_maps_path=args.node_maps,
                    edge_type_weight_map=edge_type_weight_map
                )
                train_auc, train_mrr, train_hits = train_metrics
                val_auc, val_mrr, val_hits = val_metrics
                test_auc, test_mrr, test_hits = test_metrics

                # Save predictions with epoch suffix
                pred_file = os.path.join(preds_dir, f"phage_prediction_results_epoch_{epoch}.tsv")
                save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device,
                                host_id2taxid, taxid2species,
                                output_file=pred_file, k_list=(10,), edge_type_weight_map=edge_type_weight_map,node_maps_path=args.node_maps)

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
        save_path=os.path.join(preds_dir, "phage_prediction_results"),node_maps_path=args.node_maps,
        edge_type_weight_map=edge_type_weight_map
    )
    logger.info("FINAL TEST metrics (AUC, MRR, Hits@1/5/10): %s", test_metrics)

    # Final predictions
    
    final_pred_file = os.path.join(preds_dir, "phage_prediction_results_final.tsv")
    save_predictions(model, data, test_src_cpu, test_dst_cpu, relation, eval_device, host_id2taxid, taxid2species, k_list=(10,),edge_type_weight_map=edge_type_weight_map,node_maps_path=args.node_maps,output_file=final_pred_file)

if __name__ == "__main__":
    main()



'''
python train_hgt_phage_host_weight_copy.py \
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
  --log_every 5\
  --out_dir teat
'''

