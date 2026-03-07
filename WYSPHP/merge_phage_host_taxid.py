#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def load_phage_set(path):
    """读取 phage.tsv，返回所有 accession_number 的集合。"""
    s = set()
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0]:
                s.add(parts[0])
    return s

def load_gcf_map(path):
    """读取 GCF_taxid_mapping.tsv，返回 taxid->GCF 字典。"""
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            gcf, taxid = line.strip().split('\t')
            d[taxid.split('.')[0]] = gcf
    return d

def main():
    # —— 文件路径（根据需要修改） ——
    virus_host_file = "/home/wangjingyuan/wys/dataset/virushostdb.daily_all.tsv"
    phage_file = "/home/wangjingyuan/wys/dataset/phage.tsv"
    gcf_map_file = "GCF_taxid_mapping.tsv"
    output_file = "virus_host_taxid.tsv"

    phage_set = load_phage_set(phage_file)
    gcf_map   = load_gcf_map(gcf_map_file)

    kept = 0
    with open(output_file, 'w', encoding='utf-8') as fout, \
         open(virus_host_file, 'r', encoding='utf-8') as fin:

        # 写入表头
        fout.write("virus_taxid\trefseq_id\thost_taxid\tGCF_id\n")

        header = fin.readline().rstrip('\n').split('\t')
        idx_v = header.index("virus tax id")
        idx_r = header.index("refseq id")
        idx_h = header.index("host tax id")

        for line in fin:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(idx_v, idx_r, idx_h):
                continue

            refseq = parts[idx_r]
            if refseq not in phage_set:
                continue

            v_taxid = parts[idx_v].split('.')[0]
            h_taxid = parts[idx_h].split('.')[0]
            gcf     = gcf_map.get(h_taxid)

            # 只有 gcf 不为空才写入
            if gcf:
                fout.write(f"{v_taxid}\t{refseq}\t{h_taxid}\t{gcf}\n")
                kept += 1

    print(f"✅ 完成：共写入 {kept} 条有 GCF 的记录到 {output_file}")

if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# def load_phage_set(path):
#     s = set()
#     with open(path, 'r', encoding='utf-8') as f:
#         header = f.readline()
#         for line in f:
#             acc = line.strip().split('\t')[0]
#             if acc: s.add(acc)
#     return s

# def load_gcf_map(path):
#     d = {}
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             gcf, taxid = line.strip().split('\t')
#             d[taxid.split('.')[0]] = gcf
#     return d

# def main():
#     # —— 文件路径（根据需要修改） ——
#     virus_host_file = "/home/wangjingyuan/wys/dataset/virushostdb.daily_all.tsv"
#     phage_file = "/home/wangjingyuan/wys/dataset/phage.tsv"
#     gcf_map_file = "GCF_taxid_mapping.tsv"
#     output_file = "virus_host_taxid.tsv"

#     phage_set = load_phage_set(phage_file)
#     print(f"1) 总共 phage accession_count = {len(phage_set)}")

#     gcf_map = load_gcf_map(gcf_map_file)
#     print(f"2) GCF mappings loaded = {len(gcf_map)}")

#     matched_in_virus = 0
#     found_gcf = 0

#     with open(output_file, 'w', encoding='utf-8') as fout, \
#          open(virus_host_file, 'r', encoding='utf-8') as fin:

#         fout.write("virus_taxid\trefseq_id\thost_taxid\tGCF_id\n")
#         header = fin.readline().rstrip('\n').split('\t')
#         idx_v = header.index("virus tax id")
#         idx_r = header.index("refseq id")
#         idx_h = header.index("host tax id")

#         for line in fin:
#             parts = line.rstrip('\n').split('\t')
#             if len(parts) <= max(idx_v, idx_r, idx_h):
#                 continue

#             refseq = parts[idx_r]
#             if refseq not in phage_set:
#                 continue

#             matched_in_virus += 1
#             v_taxid = parts[idx_v].split('.')[0]
#             h_taxid = parts[idx_h].split('.')[0]
#             gcf     = gcf_map.get(h_taxid, "")

#             if gcf:
#                 found_gcf += 1

#             fout.write(f"{v_taxid}\t{refseq}\t{h_taxid}\t{gcf}\n")

#     print(f"3) 匹配到 virus_host.tsv 的 phage = {matched_in_virus}")
#     print(f"4) 在这些匹配中成功找到 GCF = {found_gcf}")
#     print(f"5) 输出写入 {output_file}（包括 GCF_id 为空的行）")

# if __name__ == "__main__":
#     main()
