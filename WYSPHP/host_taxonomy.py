import pandas as pd

# 读入两个文件
edges = pd.read_csv("/home/wangjingyuan/wys/WYSPHP/host_taxonomy_edges.tsv", sep="\t")
tax = pd.read_csv("/home/wangjingyuan/wys/WYSPHP/taxonomy_with_alias.tsv", sep="\t")

# 构建 taxid -> (parent_taxid, name, rank) 的映射
tax_dict = tax.set_index("taxid").to_dict(orient="index")

def get_lineage(taxid):
    lineage = {"domain": None, "phylum": None, "class": None, "order": None,
               "family": None, "genus": None, "species": None}
    visited = set()
    while taxid in tax_dict and taxid not in visited:
        visited.add(taxid)
        info = tax_dict[taxid]
        rank = info["rank"]
        if rank in lineage:
            lineage[rank] = info["name"]
        if info["parent_taxid"] == taxid:  # root
            break
        taxid = info["parent_taxid"]
    return lineage

# 对每个 dst_id 计算分类信息
records = []
for _, row in edges.iterrows():
    src = row["src_id"]
    taxid = row["dst_id"]
    lineage = get_lineage(taxid)
    lineage["src_id"] = src
    lineage["taxid"] = taxid
    records.append(lineage)

result = pd.DataFrame(records)
result.to_csv("host_taxonomy_lineage.tsv", sep="\t", index=False)
print(result)
