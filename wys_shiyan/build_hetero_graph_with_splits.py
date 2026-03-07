#!/usr/bin/env python3
"""
build_hetero_graph_with_splits.py

Build a PyG HeteroData containing node features and edges, and include
phage-host train/val/test split edge_index attributes.

Usage example:
python build_hetero_graph_with_splits.py \
  --phage_catalog phage_catalog4.parquet \
  --host_catalog host_catalog4.parquet \
  --protein_clusters protein_clusters_emb.parquet \
  --taxonomy taxonomy_poincare_tangent.parquet \
  --edge_dir edges/ \
  --pairs_train pairs_train.tsv \
  --pairs_val pairs_val.tsv \
  --pairs_test pairs_test.tsv \
  --out hetero_graph_with_features_splits4.pt \
  --map_out node_maps.json

  

python build_hetero_graph_with_splits.py \
  --phage_catalog phage_catalog4.parquet \
  --host_catalog host_catalog4.parquet \
  --protein_clusters RBP_phage_host.parquet \
  --taxonomy taxonomy_poincare_tangent.parquet \
  --edge_dir edges_RBP/ \
  --pairs_train pairs_train_taxa.tsv \
  --pairs_val pairs_val_taxa.tsv \
  --pairs_test pairs_test_taxa.tsv \
  --out hetero_graph_with_features_splits4_RBP_taxa.pt \
  --map_out node_maps_RBP_taxa.json
Notes:
- Edge files are expected as TSV with src_id in col0, dst_id in col1 (any extra columns ignored).
- This script is robust to some missing embeddings: those nodes are dropped and edges involving them are skipped (logged).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phage_catalog", required=True)
    p.add_argument("--host_catalog", required=True)
    p.add_argument("--protein_clusters", required=True)
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--edge_dir", required=True, help="folder containing edge TSVs (names as in defaults) or pass full paths per arg")
    p.add_argument("--pairs_train", required=True)
    p.add_argument("--pairs_val", required=True)
    p.add_argument("--pairs_test", required=True)
    p.add_argument("--out", default="hetero_graph_with_features_splits.pt")
    p.add_argument("--map_out", default="node_maps.json")
    return p.parse_args()


# ---------------- utilities for loading embeddings ----------------
# def df_to_emb_matrix(df, id_col, emb_col="embedding", prefix_dim="dim_"):
#     """
#     Convert df with embeddings into (id_list, emb_matrix numpy float32).
#     Handles:
#       - single column emb_col where each row is list/ndarray
#       - or many columns named prefix_dim + i (dim_1,...)
#     Returns:
#       ids_kept (list of str), emb_mat (np.ndarray shape (N, D)), dropped_ids(list)
#     """
#     # if embedding stored as multi columns like dim_1..dim_480
#     dim_cols = [c for c in df.columns if c.startswith(prefix_dim)]
#     if emb_col in df.columns:
#         ids = df[id_col].astype(str).tolist()
#         raw = df[emb_col].tolist()
#         kept_ids = []
#         rows = []
#         dropped = []
#         # detect first non-missing to get dimension
#         for i, v in enumerate(raw):
#             if v is None:
#                 dropped.append(ids[i]); continue
#             # often v is list or numpy array
#             try:
#                 arr = np.asarray(v, dtype=np.float32)
#             except Exception:
#                 dropped.append(ids[i]); continue
#             if arr.ndim == 0:
#                 # scalar - treat as length-1 vector
#                 arr = arr.reshape(1)
#             rows.append(arr)
#             kept_ids.append(ids[i])
#         if len(rows) == 0:
#             raise RuntimeError(f"No valid embeddings found in column {emb_col}")
#         # ensure consistent dimensionality: pad or trim to max
#         lengths = [r.shape[0] for r in rows]
#         D = max(lengths)
#         if not all(l == D for l in lengths):
#             # pad shorter with zeros
#             rows_p = [np.pad(r, (0, D - r.shape[0]), mode='constant') if r.shape[0] < D else (r[:D] if r.shape[0] > D else r) for r in rows]
#             mat = np.vstack(rows_p).astype(np.float32)
#         else:
#             mat = np.vstack(rows).astype(np.float32)
#         return kept_ids, mat, dropped
#     elif len(dim_cols) > 0:
#         ids = df[id_col].astype(str).tolist()
#         mat = df[dim_cols].to_numpy(dtype=np.float32)
#         return ids, mat, []
#     else:
#         raise RuntimeError(f"Embedding column '{emb_col}' not found and no '{prefix_dim}*' columns in dataframe.")
def df_to_emb_matrix(df, id_col, emb_col="embedding", prefix_dim="dim_"):
    """
    Convert df with embeddings into:
      ids_kept (list of str),
      emb_mat (np.ndarray shape (N, D)),
      dropped_ids (list of str),
      df_filtered (pd.DataFrame only rows kept).
    Handles:
      - single column emb_col where each row is list/ndarray
      - or many columns named prefix_dim + i (dim_1,...)
    """
    dim_cols = [c for c in df.columns if c.startswith(prefix_dim)]

    if emb_col in df.columns:
        ids = df[id_col].astype(str).tolist()
        raw = df[emb_col].tolist()
        kept_ids, rows, dropped = [], [], []

        # detect first non-missing to get dimension
        for i, v in enumerate(raw):
            if v is None:
                dropped.append(ids[i])
                continue
            try:
                arr = np.asarray(v, dtype=np.float32)
            except Exception:
                dropped.append(ids[i])
                continue

            if arr.ndim == 0:  # scalar - treat as length-1 vector
                arr = arr.reshape(1)

            rows.append(arr)
            kept_ids.append(ids[i])

        if len(rows) == 0:
            raise RuntimeError(f"No valid embeddings found in column {emb_col}")

        # ensure consistent dimensionality: pad or trim
        lengths = [r.shape[0] for r in rows]
        D = max(lengths)
        if not all(l == D for l in lengths):
            rows_p = [
                np.pad(r, (0, D - r.shape[0]), mode="constant")
                if r.shape[0] < D
                else (r[:D] if r.shape[0] > D else r)
                for r in rows
            ]
            mat = np.vstack(rows_p).astype(np.float32)
        else:
            mat = np.vstack(rows).astype(np.float32)

        df_filtered = df[df[id_col].astype(str).isin(kept_ids)].copy()
        return kept_ids, mat, dropped, df_filtered

    elif len(dim_cols) > 0:
        ids = df[id_col].astype(str).tolist()
        mat = df[dim_cols].to_numpy(dtype=np.float32)
        df_filtered = df.copy()
        return ids, mat, [], df_filtered

    else:
        raise RuntimeError(
            f"Embedding column '{emb_col}' not found and no '{prefix_dim}*' columns in dataframe."
        )


# def safe_load_parquet_embeddings(path, id_col, emb_col="embedding", prefix_dim="dim_"):
#     df = pd.read_parquet(path)
#     if id_col not in df.columns:
#         raise RuntimeError(f"{path} missing id column '{id_col}'")
#     ids, mat, dropped = df_to_emb_matrix(df, id_col=id_col, emb_col=emb_col, prefix_dim=prefix_dim)
#     return ids, mat, dropped, df  # also return original df for optional extra columns
def safe_load_parquet_embeddings(path, id_col, emb_col="embedding", prefix_dim="dim_"):
    """
    Load parquet file and extract embeddings with safe handling.
    Returns:
      ids (list of str), mat (np.ndarray float32), dropped (list of str), df (original DataFrame)
    """
    df = pd.read_parquet(path)
    if id_col not in df.columns:
        raise RuntimeError(f"{path} missing id column '{id_col}'")
    ids, mat, dropped, df = df_to_emb_matrix(
        df, id_col=id_col, emb_col=emb_col, prefix_dim=prefix_dim
    )
    return ids, mat, dropped, df


# ---------------- utilities for reading edges ----------------
def read_edge_tsv(path):
    # read first two columns as strings
    df = pd.read_csv(path, sep="\t", header=0, dtype=str)
    if df.shape[1] < 2:
        raise RuntimeError(f"Edge file {path} should have at least 2 columns (src,dst)")
    srcs = df.iloc[:, 0].astype(str).tolist()
    dsts = df.iloc[:, 1].astype(str).tolist()
    return srcs, dsts, df

# def check_heterodata_ids(data, phage_ids, host_ids, prot_ids, tax_ids, phage_map, host_map, protein_map, tax_map, out_file="check_ids.txt"):
#     lines = []

#     # ---- 1. 节点 keys ----
#     lines.append("=== Node keys ===")
#     for node_type in ["phage", "host", "protein", "taxonomy"]:
#         lines.append(f"{node_type} keys: {list(data[node_type].keys())}")

#     # ---- 2. 长度匹配 ----
#     lines.append("\n=== Length check ===")
#     lines.append(f"Phage: {len(data['phage'].id)} vs {data['phage'].x.shape[0]}")
#     lines.append(f"Host: {len(data['host'].id)} vs {data['host'].x.shape[0]}")
#     lines.append(f"Protein: {len(data['protein'].id)} vs {data['protein'].x.shape[0]}")
#     lines.append(f"Taxonomy: {len(data['taxonomy'].id)} vs {data['taxonomy'].x.shape[0]}")

#     # ---- 3. 抽样前5个 ----
#     lines.append("\n=== Sample check ===")
#     lines.append(f"Phage IDs: {phage_ids[:5]} -> {data['phage'].id[:5].tolist()}")
#     lines.append(f"Host IDs: {host_ids[:5]} -> {data['host'].id[:5]}")
#     lines.append(f"Protein IDs: {prot_ids[:5]} -> {data['protein'].id[:5]}")
#     lines.append(f"Taxonomy IDs: {tax_ids[:5]} -> {data['taxonomy'].id[:5]}")

#     # ---- 4. map 对齐检查 ----
#     lines.append("\n=== Map check ===")
#     if len(phage_ids) > 0:
#         lines.append(f"phage_map[{phage_ids[0]}] = {phage_map[phage_ids[0]]}, "
#                      f"data['phage'].id[0] = {data['phage'].id[0].item()}")
#     if len(host_ids) > 0:
#         lines.append(f"host_map[{host_ids[0]}] = {host_map[host_ids[0]]}, "
#                      f"data['host'].id[0] = {data['host'].id[0].item()}")
#     if len(prot_ids) > 0:
#         lines.append(f"protein_map[{prot_ids[0]}] = {protein_map[prot_ids[0]]}, "
#                      f"data['protein'].id[0] = {data['protein'].id[0].item()}")
#     if len(tax_ids) > 0:
#         lines.append(f"tax_map[{tax_ids[0]}] = {tax_map[str(tax_ids[0])]}, "
#                      f"data['taxonomy'].id[0] = {data['taxonomy'].id[0].item()}")

#     # ---- 写入文件 ----
#     with open(out_file, "w", encoding="utf-8") as f:
#         f.write("\n".join(map(str, lines)))

#     print(f"✅ 检查结果已写入 {out_file}")



# ---------------- main builder ----------------
# def build_hetero(args):
#     # 1) load node embeddings (efficiently -> numpy -> torch)
#     print("Loading phage embeddings from", args.phage_catalog)
#     phage_ids, phage_mat, phage_dropped, phage_df = safe_load_parquet_embeddings(
#         args.phage_catalog, id_col="phage_id", emb_col="phage_dna_emb"
#     )
#     print(f"  phage: kept {len(phage_ids)}, dropped {len(phage_dropped)}")
#     #####原来
#     print("Loading host embeddings from", args.host_catalog)
#     # host: may have host_dna_emb and tangent_emb to concatenate
#     host_ids, host_mat, host_dropped, host_df = safe_load_parquet_embeddings(
#         args.host_catalog, id_col="host_gcf", emb_col="host_dna_emb"
#     )
#     print(f"  host: kept {len(host_ids)}, dropped {len(host_dropped)}")
#     print(f"  host: using dna_emb only, dim={host_mat.shape[1]}")




#     # if "tangent_emb" in host_df.columns:
#     #     # 保证顺序和 host_ids 对齐
#     #     tang_raw = host_df.set_index("host_gcf").loc[host_ids]["tangent_emb"].to_list()
#     #     host_tang_mat = np.stack([np.asarray(v, dtype=np.float32) for v in tang_raw])
#     #     host_mat = np.concatenate([host_mat_dna, host_tang_mat], axis=1)
#     #     print(
#     #         f"  host: dna_dim={host_mat_dna.shape[1]}, tangent_dim={host_tang_mat.shape[1]} "
#     #         f"-> concatenated_dim={host_mat.shape[1]}"
#     #     )
#     # else:
#     #     host_mat = host_mat_dna
#     #     print(f"  host: tangent_emb not present, using dna_emb dim={host_mat.shape[1]}")
#     ####cluster protein
#     # print("Loading protein cluster embeddings from", args.protein_clusters)
#     # prot_ids, prot_mat, prot_dropped, prot_df = safe_load_parquet_embeddings(
#     #     args.protein_clusters, id_col="cluster_id", emb_col="cluster_emb"
#     # )
#     # print(f"  protein clusters: kept {len(prot_ids)}, dropped {len(prot_dropped)}")
#     #####RBP
#     print("Loading protein cluster embeddings from", args.protein_clusters)
#     prot_ids, prot_mat, prot_dropped, prot_df = safe_load_parquet_embeddings(
#         args.protein_clusters, id_col="protein_id", emb_col="embedding"
#     )
#     print(f"  protein clusters: kept {len(prot_ids)}, dropped {len(prot_dropped)}")

#     print("Loading taxonomy embeddings from", args.taxonomy)
#     tax_ids, tax_mat, tax_dropped, tax_df = safe_load_parquet_embeddings(
#         args.taxonomy, id_col="taxid", emb_col="tangent_emb"
#     )
#     print(f"  taxonomy: kept {len(tax_ids)}, dropped {len(tax_dropped)}")


#     # convert to torch tensors
#     phage_x = torch.from_numpy(phage_mat)
#     host_x = torch.from_numpy(host_mat)
#     protein_x = torch.from_numpy(prot_mat)
#     taxonomy_x = torch.from_numpy(tax_mat)


#     # 保存 host 的 taxid 属性
#     # 注意 host_df 可能包含未被保留的节点，先对 host_ids 过滤
#     # 先过滤 host_df，确保只保留被选中的 host_ids
#     host_df_filtered = host_df[host_df["host_gcf"].isin(host_ids)].copy()

#     # 按照 host_ids 的顺序对齐 taxid
#     host_df_filtered = host_df_filtered.set_index("host_gcf").reindex(host_ids)

#     host_taxid_raw = host_df_filtered["host_species_taxid"].tolist()
#     # host_taxid_raw = host_df.set_index("host_gcf").loc[host_ids]["host_species_taxid"].tolist()
#     # 将字符串 taxid 转为整数，如果有缺失可以填 -1 或 0
#     host_taxid_int = []
#     for t in host_taxid_raw:
#         try:
#             host_taxid_int.append(int(t))
#         except Exception:
#             host_taxid_int.append(-1)  # 或者 np.nan / 0，根据你的需求
#     # 转成 long tensor
#     host_taxid = torch.tensor(host_taxid_int, dtype=torch.long)



#     # create maps id -> idx
#     phage_map = {pid: i for i, pid in enumerate(phage_ids)}
#     host_map = {hid: i for i, hid in enumerate(host_ids)}
#     protein_map = {cid: i for i, cid in enumerate(prot_ids)}
#     tax_map = {str(tid): i for i, tid in enumerate(tax_ids)}  # taxids may be ints/strings: map to str keys

#     print("Node counts: phage", phage_x.shape[0], "host", host_x.shape[0], "protein", protein_x.shape[0], "taxonomy", taxonomy_x.shape[0])

#     # prepare HeteroData and attach features
#     data = HeteroData()
#     data["phage"].x = phage_x
#     data["host"].x = host_x
#     data["host"].taxid = host_taxid
#     data["protein"].x = protein_x
#     data["taxonomy"].x = taxonomy_x
#     #data["phage"].refid = phage_ids


#     # check_heterodata_ids(
#     #     data,
#     #     phage_ids, host_ids, prot_ids, tax_ids,
#     #     phage_map, host_map, protein_map, tax_map,
#     #     out_file="heterodata_check.txt"
#     # )


#     # 随机抽几个 host 对应的 taxid
#     import random
#     for i in random.sample(range(len(host_ids)), 5):
#         print(f"GCF: {host_ids[i]}  ->  taxid: {data['host'].taxid[i].item()}")

#     print("host_df columns:", host_df.columns)
#     print("host_df sample:")
#     print(host_df.head())

#     print("host_ids 前10:", host_ids[:10])

#     # 确认 taxid 提取
#     try:
#         host_taxid_raw = host_df.set_index("host_gcf").loc[host_ids]["host_species_taxid"].tolist()
#         print("提取到 taxid 数量:", len(host_taxid_raw))
#         print("前10个 taxid:", host_taxid_raw[:10])
#         print("唯一 taxid 数量:", len(set(host_taxid_raw)))
#     except Exception as e:
#         print("ERROR extracting host_species_taxid:", e)

#     # define standard edge files names (in edge_dir)
#     edges_spec = {
#         ("phage", "interacts", "phage"): os.path.join(args.edge_dir, "phage_phage_edges.tsv"),
#         ("host", "interacts", "host"): os.path.join(args.edge_dir, "host_host_edges.tsv"),
#         ("phage", "encodes", "protein"): os.path.join(args.edge_dir, "phage_protein_edges.tsv"),
#         ("host", "encodes", "protein"): os.path.join(args.edge_dir, "host_protein_edges.tsv"),
#         ("protein", "similar", "protein"): os.path.join(args.edge_dir, "protein_protein_edges.tsv"),
#         ("host", "belongs_to", "taxonomy"): os.path.join(args.edge_dir, "host_taxonomy_edges.tsv"),
#         ("taxonomy", "related", "taxonomy"): os.path.join(args.edge_dir, "taxonomy_taxonomy_edges.tsv"),
#     }

#     # helper to add a relation edge_index (map ids -> ints; drop missing)
#     def add_relation(src_type, rel, dst_type, path, src_map, dst_map):
#         if not os.path.exists(path):
#             print(f"  WARNING: edge file {path} not found, skipping relation {(src_type, rel, dst_type)}")
#             return 0
#         srcs, dsts, df_raw = read_edge_tsv(path)
#         # map
#         src_idx = []
#         dst_idx = []
#         dropped = 0
#         for s, d in zip(srcs, dsts):
#             key_s = s
#             key_d = d
#             # taxonomy keys might be ints in string form; normalize to str
#             if src_type == "taxonomy":
#                 key_s = str(s)
#             if dst_type == "taxonomy":
#                 key_d = str(d)
#             i = src_map.get(key_s)
#             j = dst_map.get(key_d)
#             if i is None or j is None:
#                 dropped += 1
#                 continue
#             src_idx.append(i)
#             dst_idx.append(j)
#         if len(src_idx) == 0:
#             print(f"  Relation {(src_type, rel, dst_type)} produced 0 edges (all dropped).")
#             return 0
#         edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
#         data[(src_type, rel, dst_type)].edge_index = edge_index
#         print(f"  Relation {(src_type, rel, dst_type)} added: {edge_index.shape[1]} edges (dropped {dropped})")
#         return edge_index.shape[1]

#     # add relations
#     for (src_type, rel, dst_type), path in edges_spec.items():
#         src_map = {"phage": phage_map, "host": host_map, "protein": protein_map, "taxonomy": tax_map}[src_type]
#         dst_map = {"phage": phage_map, "host": host_map, "protein": protein_map, "taxonomy": tax_map}[dst_type]
#         add_relation(src_type, rel, dst_type, path, src_map, dst_map)

#     # phage-host: read train/val/test splits and attach union + per-split indices
#     def read_pairs_to_idx(path, src_map, dst_map):
#         if not os.path.exists(path):
#             return [], []
#         df = pd.read_csv(path, sep="\t", header=0, dtype=str)
#         if df.shape[1] < 2:
#             raise RuntimeError(f"{path} must have at least 2 columns: phage_id, host_gcf")
#         s = df.iloc[:, 0].astype(str).tolist()
#         d = df.iloc[:, 1].astype(str).tolist()
#         s_idx = []
#         d_idx = []
#         dropped = 0
#         for si, di in zip(s, d):
#             i = src_map.get(si)
#             j = dst_map.get(di)
#             if i is None or j is None:
#                 dropped += 1
#                 continue
#             s_idx.append(i); d_idx.append(j)
#         if dropped > 0:
#             print(f"  NOTE: {dropped} pairs in {path} dropped because phage/host missing from node tables")
#         return s_idx, d_idx

#     print("Processing phage-host train/val/test splits...")
#     train_s, train_d = read_pairs_to_idx(args.pairs_train, phage_map, host_map)
#     val_s, val_d = read_pairs_to_idx(args.pairs_val, phage_map, host_map)
#     test_s, test_d = read_pairs_to_idx(args.pairs_test, phage_map, host_map)

#     # union all edges into full edge_index (deduplicate)
#     all_pairs = set()
#     for a, b in zip(train_s, train_d): all_pairs.add((a, b))
#     for a, b in zip(val_s, val_d): all_pairs.add((a, b))
#     for a, b in zip(test_s, test_d): all_pairs.add((a, b))
#     if len(all_pairs) == 0:
#         print("Warning: no phage-host pairs found in the splits!")
#     all_src = [p[0] for p in sorted(all_pairs)]
#     all_dst = [p[1] for p in sorted(all_pairs)]
#     if len(all_src) > 0:
#         data[("phage", "infects", "host")].edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
#     else:
#         print("No phage-host edges to add.")

#     # store per-split edge_index attributes (train/val/test) on the same relation
#     def set_split(name, s_list, d_list):
#         if len(s_list) == 0:
#             setattr(data[("phage", "infects", "host")], f"edge_index_{name}", torch.empty((2,0), dtype=torch.long))
#         else:
#             setattr(data[("phage", "infects", "host")], f"edge_index_{name}", torch.tensor([s_list, d_list], dtype=torch.long))
#         print(f"  Stored phage-host split '{name}': {len(s_list)} edges")

#     set_split("train", train_s, train_d)
#     set_split("val", val_s, val_d)
#     set_split("test", test_s, test_d)

#     # optionally save nodes maps for traceability
#     node_maps = {
#         "phage_map": phage_map,
#         "host_map": host_map,
#         "protein_map": protein_map,
#         "tax_map": tax_map
#     }
#     with open(args.map_out, "w") as fh:
#         json.dump(node_maps, fh)
#     print("Saved node maps to", args.map_out)

#     # save hetero data
#     torch.save(data, args.out)
#     print("Saved HeteroData to", args.out)
#     return data

def build_hetero(args):

    # 1) load node embeddings (efficiently -> numpy -> torch)
    print("Loading phage embeddings from", args.phage_catalog)
    phage_ids, phage_mat, phage_dropped, phage_df = safe_load_parquet_embeddings(
        args.phage_catalog, id_col="phage_id", emb_col="phage_dna_emb"
    )
    print(f"  phage: kept {len(phage_ids)}, dropped {len(phage_dropped)}")


    #### protein ####
    print("Loading protein cluster embeddings from", args.protein_clusters)
    prot_ids, prot_mat, prot_dropped, prot_df = safe_load_parquet_embeddings(
        args.protein_clusters, id_col="protein_id", emb_col="embedding"
    )
    print(f"  protein clusters: kept {len(prot_ids)}, dropped {len(prot_dropped)}")

    #### taxonomy ####
    print("Loading taxonomy embeddings from", args.taxonomy)
    tax_ids, tax_mat, tax_dropped, tax_df = safe_load_parquet_embeddings(
        args.taxonomy, id_col="taxid", emb_col="tangent_emb"
    )
    print(f"  taxonomy: kept {len(tax_ids)}, dropped {len(tax_dropped)}")

    # convert to torch tensors
    phage_x = torch.from_numpy(phage_mat)
    protein_x = torch.from_numpy(prot_mat)
    taxonomy_x = torch.from_numpy(tax_mat)


        ##### 修改 host 部分 #####
    print("Loading host embeddings from", args.host_catalog)
    host_df = pd.read_parquet(args.host_catalog)

    # host_gcf 节点列表（去重）
    host_ids = sorted(host_df["host_gcf"].astype(str).unique().tolist())

    # host_sequence 节点列表 + embedding
    seq_ids_kept, seq_mat, seq_dropped, _ = df_to_emb_matrix(
        host_df, id_col="sequence_id", emb_col="host_dna_emb"
    )
    sequence_x = torch.from_numpy(seq_mat)
        # create maps id -> idx
    phage_map = {pid: i for i, pid in enumerate(phage_ids)}
    host_map = {hid: i for i, hid in enumerate(host_ids)}
    sequence_map = {sid: i for i, sid in enumerate(seq_ids_kept)}
    protein_map = {cid: i for i, cid in enumerate(prot_ids)}
    tax_map = {str(tid): i for i, tid in enumerate(tax_ids)}
    # host_gcf 用零向量，维度取自 sequence embedding
    if sequence_x.shape[0] > 0:
        dim = sequence_x.shape[1]
    else:
        dim = 128  # fallback
    # host_gcf 用 sequence 节点的 mean pooling 作为初始特征
    host_x = torch.zeros((len(host_ids), dim), dtype=torch.float32)

    # 建立 host_gcf -> sequence 映射
    host_to_seqs = {hid: [] for hid in host_ids}
    for _, row in host_df.iterrows():
        g = str(row["host_gcf"])
        s = str(row["sequence_id"])
        if g in host_map and s in sequence_map:
            host_to_seqs[g].append(sequence_map[s])

    # 计算 mean pooling
    for g, seq_indices in host_to_seqs.items():
        if len(seq_indices) > 0:
            emb = sequence_x[seq_indices].mean(dim=0)
            host_x[host_map[g]] = emb


    print(f"  host: host_gcf count={len(host_ids)}, dim={dim} (zero init)")
    print(f"  host_sequence: kept {len(seq_ids_kept)}, dropped {len(seq_dropped)}")

    # 保存 host 的 taxid 属性（挂在 host_gcf 上）
    host_df_filtered = host_df.drop_duplicates("host_gcf").set_index("host_gcf").reindex(host_ids)
    host_taxid_raw = host_df_filtered["host_species_taxid"].tolist()
    host_taxid_int = []
    for t in host_taxid_raw:
        try:
            host_taxid_int.append(int(t))
        except Exception:
            host_taxid_int.append(-1)
    host_taxid = torch.tensor(host_taxid_int, dtype=torch.long)

    print("Node counts: phage", phage_x.shape[0],
          "host", host_x.shape[0],
          "host_sequence", sequence_x.shape[0],
          "protein", protein_x.shape[0],
          "taxonomy", taxonomy_x.shape[0])

    # prepare HeteroData and attach features
    data = HeteroData()
    data["phage"].x = phage_x
    data["host"].x = host_x
    data["host"].taxid = host_taxid
    data["host_sequence"].x = sequence_x
    data["protein"].x = protein_x
    data["taxonomy"].x = taxonomy_x

    # host_gcf – sequence 边
    src_idx = []
    dst_idx = []
    for _, row in host_df.iterrows():
        g = str(row["host_gcf"])
        s = str(row["sequence_id"])
        if g in host_map and s in sequence_map:
            src_idx.append(host_map[g])
            dst_idx.append(sequence_map[s])
    data[("host", "has_sequence", "host_sequence")].edge_index = torch.tensor(
        [src_idx, dst_idx], dtype=torch.long
    )

    print(f"  host–sequence edges: {len(src_idx)}")

    # 随机抽几个 host 对应的 taxid
    import random
    for i in random.sample(range(len(host_ids)), min(5, len(host_ids))):
        print(f"GCF: {host_ids[i]}  ->  taxid: {data['host'].taxid[i].item()}")
    # 检查 host 节点特征是否全是 0
    num_zero_hosts = (host_x.abs().sum(dim=1) == 0).sum().item()
    print(f"Host 节点特征为零的数量: {num_zero_hosts} / {host_x.shape[0]}")

    # 随机打印几个 host embedding 看看
    import random
    for i in random.sample(range(len(host_ids)), min(5, len(host_ids))):
        print(f"Host {host_ids[i]} embedding 前5维: {host_x[i][:5].tolist()}")

    #### edges from tsv ####
    edges_spec = {
        ("phage", "interacts", "phage"): os.path.join(args.edge_dir, "phage_phage_edges.tsv"),
        ("host", "interacts", "host"): os.path.join(args.edge_dir, "host_host_edges.tsv"),
        ("phage", "encodes", "protein"): os.path.join(args.edge_dir, "phage_protein_edges.tsv"),
        ("host", "encodes", "protein"): os.path.join(args.edge_dir, "host_protein_edges.tsv"),
        ("protein", "similar", "protein"): os.path.join(args.edge_dir, "protein_protein_edges.tsv"),
        ("host", "belongs_to", "taxonomy"): os.path.join(args.edge_dir, "host_taxonomy_edges.tsv"),
        ("taxonomy", "related", "taxonomy"): os.path.join(args.edge_dir, "taxonomy_taxonomy_edges.tsv"),
    }

    def add_relation(src_type, rel, dst_type, path, src_map, dst_map):
        if not os.path.exists(path):
            print(f"  WARNING: edge file {path} not found, skipping relation {(src_type, rel, dst_type)}")
            return 0
        srcs, dsts, df_raw = read_edge_tsv(path)
        src_idx, dst_idx, dropped = [], [], 0
        for s, d in zip(srcs, dsts):
            key_s, key_d = s, d
            if src_type == "taxonomy":
                key_s = str(s)
            if dst_type == "taxonomy":
                key_d = str(d)
            i = src_map.get(key_s)
            j = dst_map.get(key_d)
            if i is None or j is None:
                dropped += 1
                continue
            src_idx.append(i); dst_idx.append(j)
        if len(src_idx) == 0:
            print(f"  Relation {(src_type, rel, dst_type)} produced 0 edges (all dropped).")
            return 0
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_index = edge_index
        print(f"  Relation {(src_type, rel, dst_type)} added: {edge_index.shape[1]} edges (dropped {dropped})")
        return edge_index.shape[1]

    for (src_type, rel, dst_type), path in edges_spec.items():
        src_map = {"phage": phage_map, "host": host_map, "host_sequence": sequence_map,
                   "protein": protein_map, "taxonomy": tax_map}[src_type]
        dst_map = {"phage": phage_map, "host": host_map, "host_sequence": sequence_map,
                   "protein": protein_map, "taxonomy": tax_map}[dst_type]
        add_relation(src_type, rel, dst_type, path, src_map, dst_map)

    #### phage-host splits ####
    def read_pairs_to_idx(path, src_map, dst_map):
        if not os.path.exists(path):
            return [], []
        df = pd.read_csv(path, sep="\t", header=0, dtype=str)
        if df.shape[1] < 2:
            raise RuntimeError(f"{path} must have at least 2 columns: phage_id, host_gcf")
        s = df.iloc[:, 0].astype(str).tolist()
        d = df.iloc[:, 1].astype(str).tolist()
        s_idx, d_idx, dropped = [], [], 0
        for si, di in zip(s, d):
            i = src_map.get(si)
            j = dst_map.get(di)
            if i is None or j is None:
                dropped += 1
                continue
            s_idx.append(i); d_idx.append(j)
        if dropped > 0:
            print(f"  NOTE: {dropped} pairs in {path} dropped because phage/host missing from node tables")
        return s_idx, d_idx

    print("Processing phage-host train/val/test splits...")
    train_s, train_d = read_pairs_to_idx(args.pairs_train, phage_map, host_map)
    val_s, val_d = read_pairs_to_idx(args.pairs_val, phage_map, host_map)
    test_s, test_d = read_pairs_to_idx(args.pairs_test, phage_map, host_map)

    all_pairs = set()
    for a, b in zip(train_s, train_d): all_pairs.add((a, b))
    for a, b in zip(val_s, val_d): all_pairs.add((a, b))
    for a, b in zip(test_s, test_d): all_pairs.add((a, b))
    all_src = [p[0] for p in sorted(all_pairs)]
    all_dst = [p[1] for p in sorted(all_pairs)]
    if len(all_src) > 0:
        data[("phage", "infects", "host")].edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    else:
        print("No phage-host edges to add.")

    def set_split(name, s_list, d_list):
        if len(s_list) == 0:
            setattr(data[("phage", "infects", "host")], f"edge_index_{name}", torch.empty((2,0), dtype=torch.long))
        else:
            setattr(data[("phage", "infects", "host")], f"edge_index_{name}", torch.tensor([s_list, d_list], dtype=torch.long))
        print(f"  Stored phage-host split '{name}': {len(s_list)} edges")

    set_split("train", train_s, train_d)
    set_split("val", val_s, val_d)
    set_split("test", test_s, test_d)

    node_maps = {
        "phage_map": phage_map,
        "host_map": host_map,
        "host_sequence_map": sequence_map,
        "protein_map": protein_map,
        "tax_map": tax_map
    }
    with open(args.map_out, "w") as fh:
        json.dump(node_maps, fh)
    print("Saved node maps to", args.map_out)

    torch.save(data, args.out)
    print("Saved HeteroData to", args.out)
    return data

if __name__ == "__main__":
    args = parse_args()
    build_hetero(args)
    

