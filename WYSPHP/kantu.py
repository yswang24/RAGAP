#!/usr/bin/env python3
"""
draw_hetero_svg.py

Usage examples:
# 1) 从 PyG HeteroData (.pt) 文件画图
python draw_hetero_svg.py --hetero_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt --out graph.svg --max_nodes 500

# 2) 从边表 (tsv) 画图（需要至少 src_id,dst_id 列）
python draw_hetero_svg.py --edges edges.tsv --out graph.svg --max_nodes 500

# 3) 从边表 + 节点表（包含 node_id,type）画图
python draw_hetero_svg.py --edges edges.tsv --nodes nodes.tsv --node_id_col id --node_type_col type --out graph.svg
"""
import argparse
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import sys
hetero_data = torch.load("/home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt", weights_only=False)
def build_graph_from_edges(edges_path, src_col="src_id", dst_col="dst_id", type_col="edge_type"):
    df = pd.read_csv(edges_path, sep=None, engine="python")  # 支持 tsv 或 csv
    # 尧：如果文件列不是这些名字，可通过参数设置
    if src_col not in df.columns or dst_col not in df.columns:
        raise ValueError(f"输入边表缺少列: {src_col} 或 {dst_col}. 实际列: {list(df.columns)}")
    G = nx.Graph()
    # 添加边并标注 edge_type
    for _, r in df.iterrows():
        u = str(r[src_col])
        v = str(r[dst_col])
        et = str(r[type_col]) if type_col in df.columns else "edge"
        # 把节点 type 临时标为空，之后可用 nodes 表补齐
        if not G.has_node(u):
            G.add_node(u, ntype=None)
        if not G.has_node(v):
            G.add_node(v, ntype=None)
        # 如果多条边存在，保留最大权重（若有 weight 列）
        w = None
        if "weight" in df.columns:
            try:
                w = float(r["weight"])
            except Exception:
                w = None
        # 使用 edge_type 在属性里保存（如果多条边，合并为列表）
        if G.has_edge(u, v):
            # 添加/更新属性
            G[u][v].setdefault("edge_types", set()).add(et)
            if w is not None:
                # 存最大权重
                prev = G[u][v].get("weight", -1.0)
                if w > prev: G[u][v]["weight"] = w
        else:
            G.add_edge(u, v, edge_types=set([et]), weight=w if w is not None else 1.0)
    return G

def build_graph_from_pyg(hetero_pt_path):
    try:
        import torch
        from torch_geometric.data import HeteroData
        from torch_geometric.utils import to_networkx
    except Exception as e:
        raise RuntimeError("加载 torch_geometric 失败，请先安装 pytorch_geometric。错误: " + str(e))
    data = torch.load(hetero_pt_path)
    # to_networkx 会把 hetero 转成单一 Graph；保留节点 type 作为属性
    G = to_networkx(data, to_undirected=True)
    # 如果 to_networkx 没带 type，需要从 data 手动添加：
    # nodes: data.node_types; their indices mapped by data[node_type].node_id 或直接编号
    return G

def sample_graph(G, max_nodes):
    if max_nodes is None or len(G) <= max_nodes:
        return G
    # 保留高 degree 的一部分 + 随机补足，避免只取度0节点
    deg = dict(G.degree())
    sorted_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)
    topk = int(max_nodes * 0.6)
    top_nodes = [n for n,_ in sorted_nodes[:topk]]
    remaining = list(set(G.nodes()) - set(top_nodes))
    need = max_nodes - len(top_nodes)
    if need > 0:
        samp = random.sample(remaining, min(need, len(remaining)))
    else:
        samp = []
    nodes_keep = set(top_nodes + samp)
    return G.subgraph(nodes_keep).copy()

def auto_assign_node_types(G, nodes_df=None, node_id_col="id", node_type_col="type"):
    # 如果提供 nodes_df，用它补全节点类型
    if nodes_df is not None:
        for _, r in nodes_df.iterrows():
            nid = str(r[node_id_col])
            if nid in G:
                G.nodes[nid]["ntype"] = str(r[node_type_col])
    # 若仍然没有类型，尝试依据 node name 中的关键词判断
    for n in G.nodes():
        if G.nodes[n].get("ntype") is None:
            name = str(n)
            if "|source=phage" in name or "phage_id" in name:
                G.nodes[n]["ntype"] = "phage"
            elif "|source=host" in name or name.startswith("GCF_") or name.isdigit():
                G.nodes[n]["ntype"] = "host"
            elif "CDS" in name or "protein" in name or name.startswith("NC_"):
                G.nodes[n]["ntype"] = "protein"
            else:
                G.nodes[n]["ntype"] = "other"

def draw_and_save_svg(G, out_svg, title=None, figsize=(12,10), seed=42):
    # 计算布局。对于中小图用 spring_layout；大图用 spectral or kamada_kawai
    n = len(G)
    if n == 0:
        raise ValueError("图为空")
    random.seed(seed)
    if n <= 300:
        pos = nx.spring_layout(G, seed=seed, k=None, iterations=200)
    elif n <= 2000:
        pos = nx.spring_layout(G, seed=seed, k=0.1, iterations=150)
    else:
        pos = nx.spectral_layout(G)
    # 颜色映射
    types = [G.nodes[n].get("ntype","other") for n in G.nodes()]
    uniq_types = sorted(list(set(types)))
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]
    color_map = {t: palette[i % len(palette)] for i,t in enumerate(uniq_types)}
    node_colors = [color_map[t] for t in types]
    # node sizes by degree
    degs = dict(G.degree())
    sizes = [max(20, min(800, 20 + degs[n]*10)) for n in G.nodes()]

    plt.figure(figsize=figsize)
    ax = plt.gca()
    # draw edges lightly
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.7)
    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, linewidths=0.3, edgecolors='k', alpha=0.9)
    # optionally draw labels for small graphs
    if n <= 200:
        nx.draw_networkx_labels(G, pos, font_size=8)
    # legend
    for t in uniq_types:
        ax.scatter([], [], c=color_map[t], label=t)
    ax.legend(scatterpoints=1, fontsize=8)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()
    print(f"✅ 已保存 SVG 到 {out_svg}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hetero_pt", help="PyG HeteroData .pt 文件路径（可选）")
    parser.add_argument("--edges", help="边表文件路径 (tsv/csv)，需要 src_id,dst_id 列")
    parser.add_argument("--nodes", help="可选节点表 (tsv/csv)，带 node id 和 type 列")
    parser.add_argument("--src_col", default="src_id")
    parser.add_argument("--dst_col", default="dst_id")
    parser.add_argument("--edge_type_col", default="edge_type")
    parser.add_argument("--node_id_col", default="id")
    parser.add_argument("--node_type_col", default="type")
    parser.add_argument("--out", required=True, help="输出 svg 文件路径")
    parser.add_argument("--max_nodes", type=int, default=800, help="若图太大，采样保留节点数")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    
    if not args.hetero_pt and not args.edges:
        parser.error("必须指定 --hetero_pt 或 --edges 其中之一")

    # 构建 NetworkX 图
    if args.hetero_pt:
        print("从 PyG HeteroData 加载并转换为 NetworkX ...")
        G = build_graph_from_pyg(args.hetero_pt)
    else:
        print("从边表加载图 ...")
        G = build_graph_from_edges(args.edges, src_col=args.src_col, dst_col=args.dst_col, type_col=args.edge_type_col)

    # 可选加载节点类型文件用于着色
    nodes_df = None
    if args.nodes:
        nodes_df = pd.read_csv(args.nodes, sep=None, engine="python", dtype=str)

    # 若节点太多，采样子图（保留高 degree 节点优先）
    if args.max_nodes and len(G) > args.max_nodes:
        print(f"图节点数 {len(G)} 超过 max_nodes={args.max_nodes}，将采样子图用于可视化 ...")
        G = sample_graph(G, args.max_nodes)

    # 自动赋予节点类型（若没有）
    auto_assign_node_types(G, nodes_df, node_id_col=args.node_id_col, node_type_col=args.node_type_col)

    # 最后绘制并保存 svg
    draw_and_save_svg(G, args.out, title=args.title, figsize=(12,10))

if __name__ == "__main__":
    main()
