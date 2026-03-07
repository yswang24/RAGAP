# import torch
# import pandas as pd
# import json
# from torch_geometric.data import HeteroData


# def safe_load_parquet_embeddings(path, id_col, emb_col):
#     df = pd.read_parquet(path)
#     ids = df[id_col].astype(str).tolist()
#     mat = df[emb_col].apply(lambda x: torch.tensor(x, dtype=torch.float32)).tolist()
#     mat = torch.stack(mat, dim=0)
#     return ids, mat

# def add_new_phages(old_graph_path, new_phage_parquet, out_graph_path, out_map_path):
#     # 1. load old graph
#     data: HeteroData = torch.load(old_graph_path,weights_only=False)
#     old_num = data["phage"].x.size(0)

#     # 2. load new phage embeddings
#     phage_ids, phage_x_new = safe_load_parquet_embeddings(
#         new_phage_parquet, id_col="phage_id", emb_col="phage_dna_emb"
#     )

#     # 3. append new phage nodes
#     data["phage"].x = torch.cat([data["phage"].x, phage_x_new], dim=0)

#     # 4. build mapping: 原始 phage_id -> 新的节点索引
#     id_mapping = {}
#     for i, pid in enumerate(phage_ids):
#         id_mapping[pid] = old_num + i

#     # 5. save new graph + mapping
#     torch.save(data, out_graph_path)
#     with open(out_map_path, "w") as f:
#         json.dump(id_mapping, f, indent=2)

#     print(f"✅ old phage count: {old_num}")
#     print(f"✅ new phage count: {len(phage_ids)}")
#     print(f"✅ total phage count: {data['phage'].x.size(0)}")
#     print(f"✅ mapping saved to {out_map_path}")
#     return data, id_mapping

# if __name__ == "__main__":
#     old_graph_path = "/home/wangjingyuan/wys/build_new_phage/hetero_graph_with_features_splits4_RBP.pt"
#     new_phage_parquet = "/home/wangjingyuan/wys/build_new_phage/newphage_cherry_catalog"
#     out_graph_path = "graph_with_newphages_cherry.pt"
#     out_map_path = "newphage_mapping_cherry.json"

#     add_new_phages(old_graph_path, new_phage_parquet, out_graph_path, out_map_path)







###加phage-phage边
#!/usr/bin/env python3
"""
add_new_phages_with_ppedges.py

功能：
 - 在已有 HeteroData 图上追加新的 phage 节点（从 parquet 读取 embedding）
 - 可选：从一个 tsv 添加 phage-phage 边（两列：src_id \t dst_id）
 - 支持传入旧的 node_map JSON（phage_id -> idx），以解析 TSV 中的旧 phage id
 - 输出新图和合并后的 mapping JSON

Usage example:
python build_new_heteroData.py \
  --old-graph hetero_graph_with_features_splits4_cluster_650.pt \
  --new-parquet /home/wangjingyuan/wys/build_new_phage/newphage_cherry_catalog/new_phage.parquet \
  --pp-edge-tsv /home/wangjingyuan/wys/build_new_phage/newphage-compare/new_phage_phage_edges.tsv \
  --old-map node_maps_cluster_650.json \
  --out-graph graph_with_newphages_ppedges.pt \
  --out-map merged_phage_mapping.json
"""

import os
import json
import argparse
import torch
import pandas as pd
from torch_geometric.data import HeteroData


def safe_load_parquet_embeddings(path, id_col, emb_col):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if id_col not in df.columns or emb_col not in df.columns:
        raise RuntimeError(f"{path} must contain columns '{id_col}' and '{emb_col}'")
    ids = df[id_col].astype(str).tolist()
    emb_list = df[emb_col].tolist()
    # convert each emb to torch tensor robustly
    rows = []
    for i, v in enumerate(emb_list):
        arr = torch.tensor(v, dtype=torch.float32)
        if arr.ndim != 1:
            raise RuntimeError(f"Embedding at row {i} is not 1-D")
        rows.append(arr)
    mat = torch.stack(rows, dim=0)
    return ids, mat, df


import os, json

def load_json_map(path):
    """
    更稳健地加载 node map JSON，返回 {str(id): int(idx)}。
    支持文件结构：
      - {"MT123": 42, ...}
      - {"phage_map": {"MT123": 42, ...}, "host_map": {...}}
      - {"MT123": {"idx": 42}, ...}
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path) as fh:
        d = json.load(fh)

    # 如果顶层就是一个包含子 maps 的 dict（如 {'phage_map': {...}, 'host_map': {...}}）
    # 那就尝试提取最合适的子 map（优先查找包含 'phage' 的键）
    if isinstance(d, dict):
        # quick detect: if values are dicts and some keys look like submaps
        # pick candidate submaps
        candidate = None
        # 1) if there is a key that contains 'phage' prefer that
        for k in d.keys():
            if 'phage' in k.lower() and isinstance(d[k], dict):
                candidate = d[k]
                break
        # 2) else find first key that endswith '_map' or value is a dict
        if candidate is None:
            for k, v in d.items():
                if k.lower().endswith('_map') and isinstance(v, dict):
                    candidate = v
                    break
            if candidate is None:
                # 3) if all top-level values are ints/str -> treat d itself as mapping
                simple_values = all(not isinstance(v, dict) for v in d.values())
                if not simple_values:
                    # try to find any dict-valued entry and merge them
                    merged = {}
                    for k, v in d.items():
                        if isinstance(v, dict):
                            # if v looks like {id: idx} merge
                            keys_are_ids = all(isinstance(x, str) for x in v.keys())
                            if keys_are_ids:
                                merged.update(v)
                    if merged:
                        candidate = merged
        if candidate is not None:
            d = candidate

    # Now d should be either a mapping id->idx OR id-> {inner dict}
    mapping = {}
    if not isinstance(d, dict):
        raise RuntimeError(f"Unexpected json map format at {path}: top-level not a dict")

    for k, v in d.items():
        ks = str(k)
        # if value is an int-like
        if isinstance(v, (int, float, str)) and not isinstance(v, dict):
            # try to cast to int
            try:
                mapping[ks] = int(v)
                continue
            except Exception:
                # fallback: skip or continue to parsing below
                pass

        # if value is a dict, try to find inner integer fields
        if isinstance(v, dict):
            found = False
            for inner_key in ('idx', 'index', 'node_idx', 'node_index', 'id', 'index0'):
                if inner_key in v:
                    try:
                        mapping[ks] = int(v[inner_key])
                        found = True
                        break
                    except Exception:
                        pass
            # some maps store {'value': 42} or similar
            if not found:
                # try to find any int-like in the dict values
                for inner_v in v.values():
                    if isinstance(inner_v, (int, float, str)):
                        try:
                            mapping[ks] = int(inner_v)
                            found = True
                            break
                        except Exception:
                            continue
            if found:
                continue
            else:
                # Could not interpret inner dict -> skip with warning
                # (we don't raise to be tolerant)
                # print(f"Warning: could not parse inner mapping for {ks} -> {v}")
                continue

        # if value is a list, try to take first element as idx
        if isinstance(v, (list, tuple)):
            if len(v) > 0:
                try:
                    mapping[ks] = int(v[0])
                    continue
                except Exception:
                    pass

        # otherwise skip entry (couldn't parse)
        # print(f"Skipping mapping for key {ks}: unsupported value type {type(v)}")
        continue

    return mapping



def read_pp_edge_tsv(path):
    """Read a two-column TSV (src_id, dst_id). Returns list of (src, dst) as strings."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 2:
        raise RuntimeError(f"{path} must have at least 2 columns (src_id, dst_id)")
    srcs = df.iloc[:, 0].astype(str).tolist()
    dsts = df.iloc[:, 1].astype(str).tolist()
    return list(zip(srcs, dsts))


def add_new_phages(old_graph_path, new_phage_parquet, out_graph_path, out_map_path,
                   pp_edge_tsv=None, old_map_path=None):
    # 1) load old graph
    if not os.path.exists(old_graph_path):
        raise FileNotFoundError(old_graph_path)
    data: HeteroData = torch.load(old_graph_path, map_location="cpu",weights_only=False)
    if "phage" not in data.node_types:
        raise RuntimeError("old graph has no 'phage' node type")
    old_phage_count = int(data["phage"].x.size(0))

    # 2) load new phage embeddings (parquet)
    new_ids, new_x, _ = safe_load_parquet_embeddings(new_phage_parquet, id_col="phage_id", emb_col="phage_dna_emb")
    new_count = int(new_x.size(0))

    # 3) check dims
    phage_dim = int(data["phage"].x.size(1))
    if int(new_x.size(1)) != phage_dim:
        raise RuntimeError(f"Embedding dim mismatch: graph phage dim={phage_dim}, new emb dim={int(new_x.size(1))}")

    # 4) append new nodes
    data = data.clone()  # don't modify original in-place unexpectedly
    data["phage"].x = torch.cat([data["phage"].x, new_x], dim=0)

    # 5) build new mapping for appended phages
    new_map = {str(pid): int(old_phage_count + i) for i, pid in enumerate(new_ids)}

    # 6) try to load old_map if provided
    old_map = {}
    if old_map_path:
        old_map = load_json_map(old_map_path)
        print(f"Loaded old_map with {len(old_map)} entries from {old_map_path}")
    else:
        print("No old_map provided (will only resolve ids present in new_map)")

    # 7) merge maps (old_map keys take precedence if conflict)
    merged_map = {}
    merged_map.update(new_map)
    merged_map.update(old_map)  # old_map may override new_map if same keys (unlikely)
    # but better to ensure old_map entries are integers and correct

    # 8) if pp_edge_tsv provided, read and convert to indices
    pp_added = 0
    pp_dropped = 0
    if pp_edge_tsv:
        raw_edges = read_pp_edge_tsv(pp_edge_tsv)
        src_idx = []
        dst_idx = []
        missing_ids = set()
        for s_id, d_id in raw_edges:
            # try to find s and d in merged_map
            s_idx = merged_map.get(str(s_id))
            d_idx = merged_map.get(str(d_id))
            if s_idx is None or d_idx is None:
                # try swapped: sometimes TSV uses only old ids (in old_map)
                missing_ids.add((s_id, d_id))
                pp_dropped += 1
                continue
            src_idx.append(int(s_idx))
            dst_idx.append(int(d_idx))
            pp_added += 1

        if len(src_idx) > 0:
            new_edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            # if graph already has phage-phage relation, merge (concatenate)
            rel = ("phage", "interacts", "phage")
            if rel in data.edge_types and hasattr(data[rel], "edge_index") and data[rel].edge_index is not None:
                old_ei = data[rel].edge_index
                # concatenate and remove duplicates
                combined = torch.cat([old_ei, new_edge_index], dim=1)
                # unique columns removal (workaround using set on tuples)
                cols = set()
                keep_src = []
                keep_dst = []
                for a, b in combined.t().tolist():
                    if (a, b) not in cols:
                        cols.add((a, b))
                        keep_src.append(a)
                        keep_dst.append(b)
                data[rel].edge_index = torch.tensor([keep_src, keep_dst], dtype=torch.long)
                print(f"Merged {pp_added} new phage-phage edges with existing edges; total now {data[rel].edge_index.size(1)}")
            else:
                # just set new relation
                data[("phage", "interacts", "phage")].edge_index = new_edge_index
                print(f"Added phage-phage relation with {new_edge_index.size(1)} edges")
        else:
            print("No valid phage-phage edges found (all dropped).")
        if pp_dropped > 0:
            print(f"Note: {pp_dropped} edges dropped due to missing phage ids in mapping (examples shown below).")
            # print up to 10 missing examples
            for a, b in list(missing_ids)[:10]:
                print("  missing:", a, b)

    # 9) save outputs
    torch.save(data, out_graph_path)
    # prefer to save merged_map: included old_map + new_map
    with open(out_map_path, "w") as fh:
        json.dump({k: int(v) for k, v in merged_map.items()}, fh, indent=2)

    print(f"Saved new graph -> {out_graph_path}")
    print(f"Saved merged mapping -> {out_map_path}")
    print(f"Old phage count: {old_phage_count}, New phage added: {new_count}, Total phage now: {int(data['phage'].x.size(0))}")
    if pp_edge_tsv:
        print(f"Phage-phage edges added: {pp_added}, dropped: {pp_dropped}")

    return data, merged_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--old-graph", required=True, help="原始 HeteroData .pt 文件")
    p.add_argument("--new-parquet", required=True, help="新 phage parquet（必须包含 phage_id 和 phage_dna_emb）")
    p.add_argument("--pp-edge-tsv", default=None, help="可选：phage-phage 边的 tsv，2 列 (src_id, dst_id)")
    p.add_argument("--old-map", default=None, help="可选：旧的 phage_id->idx 映射 JSON（训练时保存的 map_out）")
    p.add_argument("--out-graph", default="graph_with_newphages.pt", help="输出新图路径")
    p.add_argument("--out-map", default="merged_phage_mapping.json", help="输出合并映射 json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_new_phages(
        old_graph_path=args.old_graph,
        new_phage_parquet=args.new_parquet,
        out_graph_path=args.out_graph,
        out_map_path=args.out_map,
        pp_edge_tsv=args.pp_edge_tsv,
        old_map_path=args.old_map
    )
