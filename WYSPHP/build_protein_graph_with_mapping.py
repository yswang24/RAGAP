# build_protein_graph_with_mapping.py

import os
import json
import numpy as np
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
import torch

# PDB 输入目录（AlphaFold 输出或 PDB 文件）
PDB_DIR = "af2_out"
# mapping 表（蛋白ID → 来源 Phage 或 Host ID）
# 格式示例：{"protein1": {"source_type": "phage", "source_id": "phage_XYZ"}, ...}
SOURCE_MAP = "protein_to_source.json"

OUT_DIR = "GCN_data"
os.makedirs(OUT_DIR, exist_ok=True)

# 加载来源映射
with open(SOURCE_MAP, "r") as f:
    protein_source_map = json.load(f)

# Config：residue-level, Cα contact threshold 默认
config = ProteinGraphConfig(granularity="CA",
    edge_construction_functions=[add_distance_threshold])

all_edge_indices = {}
all_feature_dict = {}
mapping_list = []

for pdb_file in os.listdir(PDB_DIR):
    if not pdb_file.endswith(".pdb"):
        continue
    protein_id = os.path.splitext(pdb_file)[0]
    pdb_path = os.path.join(PDB_DIR, pdb_file)
    G = construct_graph(config=config, pdb_path=pdb_path)

    # 提取 edge_index
    edges = list(G.edges())
    edge_index = np.array(edges).T  # shape [2, E]
    all_edge_indices[protein_id] = edge_index

    # 提取节点特征（作为示例：Cα coords）
    coords = np.array([data["coords"] for _, data in G.nodes(data=True)], dtype=np.float32)
    all_feature_dict[protein_id] = coords

    # 记录 mapping：每个 node 的 global index 和来源
    source_info = protein_source_map.get(protein_id, {})
    source_type = source_info.get("source_type")
    source_id = source_info.get("source_id")
    num_nodes = coords.shape[0]
    for local_idx in range(num_nodes):
        mapping_list.append({
            "protein_id": protein_id,
            "local_idx": local_idx,
            "source_type": source_type,
            "source_id": source_id
        })

    print(f"{protein_id}: {edge_index.shape[1]} edges, {coords.shape[0]} nodes")

# 保存结构
np.savez(os.path.join(OUT_DIR, "protein_edge_index.npz"), **all_edge_indices)
np.savez(os.path.join(OUT_DIR, "protein_node_coords.npz"), **all_feature_dict)
with open(os.path.join(OUT_DIR, "protein_mapping.json"), "w") as f:
    json.dump(mapping_list, f, indent=2)

print("Saved edges, features and mapping for all proteins.")
