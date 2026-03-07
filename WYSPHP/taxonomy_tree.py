'''
读取 nodes.dmp 和 names.dmp

读取目标 taxid 列表（phage 和 host）

构建一个只包含相关 taxid 及其祖先的子树

输出一个 TSV，包含：taxid, parent_taxid, name, rank
'''

# import os

# def load_nodes(nodes_file):
#     taxid2parent = {}
#     taxid2rank = {}
#     with open(nodes_file, "r") as f:
#         for line in f:
#             parts = [x.strip() for x in line.split("|")]
#             taxid = parts[0]
#             parent = parts[1]
#             rank = parts[2]
#             taxid2parent[taxid] = parent
#             taxid2rank[taxid] = rank
#     return taxid2parent, taxid2rank

# def load_names(names_file):
#     taxid2name = {}
#     with open(names_file, "r") as f:
#         for line in f:
#             parts = [x.strip() for x in line.split("|")]
#             taxid = parts[0]
#             name = parts[1]
#             name_class = parts[3]
#             if name_class == "scientific name":
#                 taxid2name[taxid] = name
#     return taxid2name

# def load_taxids(file):
#     return set(line.strip() for line in open(file))

# def get_lineage(taxid, taxid2parent):
#     lineage = []
#     while taxid in taxid2parent and taxid != "1":  # "1" = root
#         lineage.append(taxid)
#         taxid = taxid2parent[taxid]
#     lineage.append("1")
#     return lineage

# def build_subtree(phage_file, host_file, nodes_file, names_file, output_file):
#     taxid2parent, taxid2rank = load_nodes(nodes_file)
#     taxid2name = load_names(names_file)
#     phage_taxids = load_taxids(phage_file)
#     host_taxids = load_taxids(host_file)

#     all_taxids = set()
#     for t in phage_taxids.union(host_taxids):
#         all_taxids.update(get_lineage(t, taxid2parent))

#     with open(output_file, "w") as out:
#         out.write("taxid\tparent_taxid\tname\trank\n")
#         for t in all_taxids:
#             parent = taxid2parent.get(t, "")
#             name = taxid2name.get(t, "NA")
#             rank = taxid2rank.get(t, "NA")
#             out.write(f"{t}\t{parent}\t{name}\t{rank}\n")

# if __name__ == "__main__":
#     build_subtree(
#         phage_file="phage_taxids.txt",
#         host_file="host_taxids.txt",
#         nodes_file="nodes.dmp",
#         names_file="names.dmp",
#         output_file="taxonomy_subtree.tsv"
#     )

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import csv

# def load_names(names_file):
#     """读取 names.dmp，返回 {taxid: name}"""
#     taxid2name = {}
#     with open(names_file, "r") as f:
#         for line in f:
#             parts = line.strip().split("\t|\t")
#             if len(parts) >= 2:
#                 taxid = parts[0].strip()
#                 name_txt = parts[1].strip()
#                 if "scientific name" in line:  # 只取学名
#                     taxid2name[taxid] = name_txt
#     return taxid2name


# def load_nodes(nodes_file):
#     """读取 nodes.dmp，返回 {taxid: (parent, rank)}"""
#     taxid2parent = {}
#     taxid2rank = {}
#     with open(nodes_file, "r") as f:
#         for line in f:
#             parts = line.strip().split("\t|\t")
#             if len(parts) >= 3:
#                 taxid = parts[0].strip()
#                 parent = parts[1].strip()
#                 rank = parts[2].strip()
#                 taxid2parent[taxid] = parent
#                 taxid2rank[taxid] = rank
#     return taxid2parent, taxid2rank


# def process_mapping(input_file, names_file, nodes_file, output_file):
#     # 加载 taxonomy 信息
#     taxid2name = load_names(names_file)
#     taxid2parent, taxid2rank = load_nodes(nodes_file)

#     with open(input_file, "r") as fin, open(output_file, "w", newline="") as fout:
#         reader = csv.DictReader(fin, delimiter="\t")
#         writer = csv.writer(fout, delimiter="\t")
#         writer.writerow(["taxid", "parent_taxid", "name", "rank"])

#         for row in reader:
#             virus_taxid = row["virus_taxid"]
#             host_taxid = row["host_taxid"]
#             refseq_id = row["refseq_id"]
#             gcf = row["Extracted_GCFs"]

#             # 病毒节点
#             if virus_taxid in taxid2parent:
#                 name = taxid2name.get(virus_taxid, "NA") + f" ({refseq_id})"
#                 writer.writerow([
#                     virus_taxid,
#                     taxid2parent[virus_taxid],
#                     name,
#                     taxid2rank[virus_taxid]
#                 ])

#             # 宿主节点
#             if host_taxid in taxid2parent:
#                 name = taxid2name.get(host_taxid, "NA") + f" ({gcf})"
#                 writer.writerow([
#                     host_taxid,
#                     taxid2parent[host_taxid],
#                     name,
#                     taxid2rank[host_taxid]
#                 ])


# if __name__ == "__main__":
#     process_mapping(
#         input_file="virus_host_with_GCF.tsv",  # 输入：你的病毒-宿主文件
#         names_file="names.dmp",              # NCBI taxonomy 名称文件
#         nodes_file="nodes.dmp",              # NCBI taxonomy 结构文件
#         output_file="taxonomy_subtree_accession.tsv"   # 输出：taxid parent name rank
#     )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入:
  - nodes.dmp
  - names.dmp
  - virus_host.tsv (包含列: virus_taxid, refseq_id, host_taxid, Extracted_GCFs)
输出:
  - taxonomy_with_alias.tsv (只包含 virus_taxid/host_taxid 及其祖先节点)
字段:
  taxid\tparent_taxid\tname\trank\talias
说明:
  alias: 如果该 taxid 是 virus_taxid 则为 refseq_id；否则如果是 host_taxid 则为 Extracted_GCFs（原文中提供的字段）
"""

import sys
import csv
import argparse

def load_nodes(nodes_file):
    taxid2parent = {}
    taxid2rank = {}
    with open(nodes_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            taxid = parts[0]
            parent = parts[1]
            rank = parts[2]
            taxid2parent[taxid] = parent
            taxid2rank[taxid] = rank
    return taxid2parent, taxid2rank

def load_names(names_file):
    taxid2name = {}
    with open(names_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            taxid = parts[0]
            name_txt = parts[1]
            name_class = parts[3]
            if name_class == "scientific name":
                taxid2name[taxid] = name_txt
    return taxid2name

def load_virus_host_map(vh_file):
    """
    读取 virus_host.tsv，返回三个字典:
      virus_taxids: set of virus taxids (strings)
      virus2refseq: {virus_taxid: refseq_id}
      host2gcf: {host_taxid: extracted_gcf_string}
    """
    virus_taxids = set()
    virus2refseq = {}
    host_taxids = set()
    host2gcf = {}

    with open(vh_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Expect columns: virus_taxid, refseq_id, host_taxid, Extracted_GCFs (case-sensitive)
        # tolerate missing columns gracefully
        for row in reader:
            vt = row.get("virus_taxid", "").strip()
            rt = row.get("refseq_id", "").strip()
            ht = row.get("host_taxid", "").strip()
            gcf = row.get("Extracted_GCFs", "").strip()

            if vt:
                virus_taxids.add(vt)
                if rt:
                    virus2refseq[vt] = rt
            if ht:
                host_taxids.add(ht)
                if gcf:
                    host2gcf[ht] = gcf

    return virus_taxids, virus2refseq, host_taxids, host2gcf

def collect_ancestors(seed_taxids, taxid2parent):
    """
    对每个 seed taxid 向上追溯到 root (taxid == parent 或 父不存在)；
    返回包含 seed 与所有祖先的集合 (strings)
    """
    collected = set()
    for t in seed_taxids:
        cur = t
        # guard against empty strings
        if not cur:
            continue
        while True:
            if cur in collected:
                break
            collected.add(cur)
            parent = taxid2parent.get(cur)
            if parent is None:
                # could not find parent in nodes.dmp -> stop
                break
            if parent == cur:
                # root/self-loop
                break
            cur = parent
    return collected

def build_output(out_file, taxids_set, taxid2parent, taxid2rank, taxid2name, virus2refseq, host2gcf):
    # sort for deterministic output: numeric by int where possible
    def _key(x):
        try:
            return int(x)
        except:
            return x
    taxid_list = sorted(taxids_set, key=_key)
    with open(out_file, "w", encoding="utf-8") as out:
        out.write("taxid\tparent_taxid\tname\trank\talias\n")
        for taxid in taxid_list:
            parent = taxid2parent.get(taxid, "")
            rank = taxid2rank.get(taxid, "")
            name = taxid2name.get(taxid, "")
            alias = ""
            # Prefer virus alias if exists, else host alias
            if taxid in virus2refseq:
                alias = virus2refseq[taxid]
            elif taxid in host2gcf:
                alias = host2gcf[taxid]
            out.write(f"{taxid}\t{parent}\t{name}\t{rank}\t{alias}\n")
    print(f"wrote {out_file} ({len(taxid_list)} rows)")

def main(args):
    taxid2parent, taxid2rank = load_nodes(args.nodes)
    taxid2name = load_names(args.names)
    virus_taxids, virus2refseq, host_taxids, host2gcf = load_virus_host_map(args.vh)

    # seeds = union of virus and host taxids
    seeds = set()
    seeds.update(virus_taxids)
    seeds.update(host_taxids)
    if not seeds:
        print("No virus/host taxids found in VH file. Exiting.")
        return

    # collect ancestors
    all_taxids = collect_ancestors(seeds, taxid2parent)

    # ensure seeds present even if parent missing
    all_taxids.update(seeds)

    build_output(args.out, all_taxids, taxid2parent, taxid2rank, taxid2name, virus2refseq, host2gcf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build taxonomy subtree (taxid,parent,name,rank,alias) for given virus/host taxids")
    parser.add_argument("--nodes", required=True, help="path to nodes.dmp")
    parser.add_argument("--names", required=True, help="path to names.dmp")
    parser.add_argument("--vh", required=True, help="virus-host tsv with columns: virus_taxid, refseq_id, host_taxid, Extracted_GCFs")
    parser.add_argument("--out", default="taxonomy_with_alias.tsv", help="output file")
    args = parser.parse_args()
    main(args)

'''
python taxonomy_tree.py \
  --nodes nodes.dmp \
  --names names.dmp \
  --vh virus_host_with_GCF.tsv \
  --out taxonomy_with_alias.tsv
'''
