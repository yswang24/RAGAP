# import torch
# import sys
# from torch_geometric.data import HeteroData
# from torch_geometric.data.storage import BaseStorage

# def inspect_hetero_graph(path):
#     print(f"正在加载图文件: {path}")

#     # 显式允许 BaseStorage，避免 PyTorch 2.6 安全限制报错
#     torch.serialization.add_safe_globals([BaseStorage])

#     data = torch.load(path, weights_only=False)
#     print(f"加载完成，类型: {type(data)}\n")

#     if isinstance(data, HeteroData):
#         print("=== 基本信息 ===")
#         print(data)

#         print("\n=== 节点类型和特征 ===")
#         for node_type in data.node_types:
#             node_data = data[node_type]
#             num_nodes = node_data.num_nodes if hasattr(node_data, "num_nodes") else "未知"
#             x_shape = node_data.x.shape if "x" in node_data else "无特征"
#             print(f"节点类型: {node_type:<15} 节点数: {num_nodes:<10} 特征: {x_shape}")

#         print("\n=== 边类型和数量 ===")
#         for edge_type in data.edge_types:
#             edge_data = data[edge_type]
#             edge_count = edge_data.edge_index.shape[1] if "edge_index" in edge_data else 0
#             print(f"边类型: {edge_type} 边数量: {edge_count}")
#             if edge_count > 0:
#                 print("前5条边索引:\n", edge_data.edge_index[:, :min(5, edge_count)].tolist())

#     elif isinstance(data, dict):
#         print("不是 HeteroData，而是字典，包含键:")
#         for k in data.keys():
#             print(f" - {k}: {type(data[k])}")
#     else:
#         print("未知类型，请手动检查数据结构")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("用法: python inspect_yigou.py /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt")
#     else:
#         inspect_hetero_graph(sys.argv[1])



# import torch
# from torch_geometric.data import HeteroData

# # 读取 .pt 文件
# data = torch.load("/home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt",weights_only=False)

# # 查看整个数据对象（会输出节点和边的基本信息）
# print(data)

# # 查看有哪些节点类型
# print("Node types:", data.node_types)

# # 查看有哪些边类型
# print("Edge types:", data.edge_types)

# print("Phage keys:", data["phage"].keys())
# print("Host keys:", data["host"].keys())
# print("Protein keys:", data["protein"].keys())
# print("Taxonomy keys:", data["taxonomy"].keys())


# # 如果存在 id 字段，就打印前几个
# if 'id' in data['phage']:
#     print("phage id:", data['phage'].id[:10])
# if 'id' in data['host']:
#     print("host id:", data['host'].id[:10])
# if 'id' in data['protein']:
#     print("protein id:", data['protein'].id[:10])
# if 'id' in data['taxonomy']:
#     print("taxonomy id:", data['taxonomy'].id[:10])




# # check_heterograph_ids.py

# import torch
# from torch_geometric.data import HeteroData

# # --------- 配置 ---------
# hetero_graph_path = "/home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt"  # HeteroData 保存文件
# output_prefix = "inspect_vectors_"  # 输出文件前缀
# num_preview = 5  # 打印前几条查看

# # --------- 1. 读取异构图 ---------
# data = torch.load(hetero_graph_path,weights_only=False)
# print("Loaded HeteroData from", hetero_graph_path)
# print("Node types:", data.node_types)

# # --------- 2. 遍历每种节点类型 ---------
# for ntype in data.node_types:
#     print(f"\n--- Checking {ntype} nodes ---")
    
#     # 确认 id 与向量长度一致
#     if not hasattr(data[ntype], "id"):
#         print(f"Warning: {ntype} nodes do not have 'id' attribute!")
#         continue

#     assert data[ntype].x.shape[0] == len(data[ntype].id), f"Mismatch in {ntype} nodes!"

#     # --------- 3. 打印前几条 ---------
#     print(f"{ntype} first {num_preview} nodes (id, first 5 vector dims):")
#     for i in range(min(num_preview, data[ntype].num_nodes)):
#         node_id = data[ntype].id[i].item() if torch.is_tensor(data[ntype].id[i]) else data[ntype].id[i]
#         vector = data[ntype].x[i].tolist()
#         print(f"  {node_id} -> {vector[:5]} ...")

#     # --------- 4. 保存到文件 ---------
#     output_file = f"{output_prefix}{ntype}.txt"
#     with open(output_file, "w") as f:
#         f.write("id\tvector\n")
#         for i in range(data[ntype].num_nodes):
#             node_id = data[ntype].id[i].item() if torch.is_tensor(data[ntype].id[i]) else data[ntype].id[i]
#             vector = data[ntype].x[i].tolist()
#             vector_str = ",".join([f"{v:.6f}" for v in vector])
#             f.write(f"{node_id}\t{vector_str}\n")
#     print(f"Saved all {ntype} nodes to {output_file}")





# import torch

# from torch_geometric.data import HeteroData

# # 读取 .pt 文件
# data = torch.load("/home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt",weights_only=False)

# def check_node_alignment(data, ntype="phage", n=5):
#     """
#     随机抽取 n 个节点，检查 id / refseq_id / 向量 是否对应
#     """
#     print(f"\n=== Checking {ntype} nodes ===")

#     refid = data[ntype].refid
#     x = data[ntype].x

#     # 打印基本信息
#     print(f"Feature dim: {x.shape[1] if x.ndim > 1 else 1}")

#     # 随机抽几个
#     idxs = torch.randperm(len(id))[:n].tolist()
#     for i in idxs:
#         print(f"\nIndex {i}:")
#         print(f"  refid    -> {ref_ids[i]}")
#         print(f"  feature vec  -> {x[i].tolist()[:10]} ...")  # 只打印前10个维度
# if __name__ == "__main__":
#     check_node_alignment(data, "phage", n=5)





# import torch

# # 你的模型文件路径
# model_path = "/home/wangjingyuan/wys/wys_shiyan/best_hgt_nb_30_20.pt"
# output_path = "model_info.txt"

# # 加载 checkpoint
# checkpoint = torch.load(model_path, map_location="cpu",weights_only=False)

# with open(output_path, "w", encoding="utf-8") as f:
#     f.write("===== Keys in checkpoint =====\n")
#     f.write(str(list(checkpoint.keys())) + "\n\n")

#     f.write("===== Model state_dict keys =====\n")
#     if "model_state_dict" in checkpoint:
#         for k, v in checkpoint["model_state_dict"].items():
#             if isinstance(v, torch.Tensor):
#                 f.write(f"{k}: {tuple(v.shape)}\n")
#             else:
#                 f.write(f"{k}: {type(v)}\n")
#     elif isinstance(checkpoint, dict):
#         for k, v in checkpoint.items():
#             if isinstance(v, torch.Tensor):
#                 f.write(f"{k}: {tuple(v.shape)}\n")
#             else:
#                 f.write(f"{k}: {type(v)}\n")
#     else:
#         f.write("模型文件不是典型的 state_dict 格式\n")

#     f.write("\n===== Optimizer info =====\n")
#     if "optimizer_state_dict" in checkpoint:
#         opt_state = checkpoint["optimizer_state_dict"]
#         f.write(f"Optimizer state has {len(opt_state['state'])} parameter groups\n")
#         f.write(f"Optimizer param_groups: {opt_state['param_groups']}\n")
#     else:
#         f.write("No optimizer_state_dict found\n")

#     f.write("\n===== Extra info =====\n")
#     for k in checkpoint.keys():
#         if k not in ["model_state_dict", "optimizer_state_dict"]:
#             f.write(f"{k}: {checkpoint[k]}\n")

# print(f"模型信息已保存到 {output_path}")



# import torch

# data = torch.load("/home/wangjingyuan/wys/wys_shiyan/best_model/best_hgt_nb_RBP_5000-512_15_5e_ph.pt", map_location="cpu",weights_only=False)
# print(type(data))   # 一般是 torch_geometric.data.HeteroData
# if isinstance(data, dict):
#     print(data.keys())
# #print(data)         # 概览信息




import torch
from collections import defaultdict
from torch_geometric.data import HeteroData
data = torch.load("/home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_cluster_650.pt", map_location="cpu",weights_only=False)

# =====================================================
# 假设你已经有一个 HeteroData 对象叫 data
# 例如:
# data = ...  # load_your_hetero_graph()
# =====================================================

print("=" * 60)
print("📊 Graph Inspection Report")
print("=" * 60)

# 节点信息
print("\n[节点类型]")
for ntype in data.node_types:
    n_nodes = data[ntype].num_nodes
    print(f"  - {ntype}: {n_nodes} nodes")

# 边信息
print("\n[边类型]")
for etype in data.edge_types:
    edge_index = data[etype].edge_index
    n_edges = edge_index.size(1)
    print(f"  - {etype}: {n_edges} edges")

# 平均度数（每个节点平均有多少邻居）
print("\n[平均度数 (per node)]")
degree_info = {}
for etype in data.edge_types:
    src, rel, dst = etype
    edge_index = data[etype].edge_index

    src_counts = torch.bincount(edge_index[0], minlength=data[src].num_nodes)
    dst_counts = torch.bincount(edge_index[1], minlength=data[dst].num_nodes)

    avg_src_degree = src_counts.float().mean().item()
    avg_dst_degree = dst_counts.float().mean().item()

    print(f"  - {src} -> {rel} -> {dst}")
    print(f"      {src}: avg degree {avg_src_degree:.2f}")
    print(f"      {dst}: avg degree {avg_dst_degree:.2f}")

    degree_info[(src, rel, dst)] = (avg_src_degree, avg_dst_degree)

# 推荐 num_neighbors
print("\n[推荐 num_neighbors 参数]")
# 取所有边的平均度数的均值作为参考
all_avg_degrees = [deg for vals in degree_info.values() for deg in vals if deg > 0]
if len(all_avg_degrees) > 0:
    mean_degree = sum(all_avg_degrees) / len(all_avg_degrees)
    first_layer = max(5, int(mean_degree * 0.7))   # 第一层取 70% 平均度数
    second_layer = max(3, int(first_layer * 0.5))  # 第二层减半
    print(f"  建议: --num_neighbors {first_layer} {second_layer}")
else:
    print("  无法推荐 num_neighbors (图中没有边)")

print("=" * 60)
print("✅ Done")
