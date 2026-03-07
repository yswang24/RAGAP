import pandas as pd

# 输入文件
edges_file = "/home/wangjingyuan/wys/WYSPHP/sequence_sequence_edges.tsv"
map_file = "/home/wangjingyuan/wys/WYSPHP/sequence_source.tsv"
output_file = "/home/wangjingyuan/wys/WYSPHP/sequence_sequence_edges_.tsv"

# 读取数据
edges = pd.read_csv(edges_file, sep="\t")
mapping = pd.read_csv(map_file, sep="\t")

# 建立字典：SeqID -> SourceFile
map_dict = dict(zip(mapping["SeqID"], mapping["SourceFile"]))

# 替换 src_id 和 dst_id
edges["src_id"] = edges["src_id"].map(map_dict).fillna(edges["src_id"])
edges["dst_id"] = edges["dst_id"].map(map_dict).fillna(edges["dst_id"])

# 去掉自己和自己的情况
edges = edges[edges["src_id"] != edges["dst_id"]]

# 保存结果
edges.to_csv(output_file, sep="\t", index=False)

print(f"处理完成，结果保存到 {output_file}")
