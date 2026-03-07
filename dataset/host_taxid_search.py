# def build_accession_taxid_map(file_path):
#     """
#     从 nucl_gb.accession2taxid 文件构建 Accession 到 TaxID 的映射。
#     """
#     accession_to_taxid = {}
#     with open(file_path, "r") as file:
#         next(file)  # 跳过标题行
#         for line in file:
#             fields = line.strip().split("\t")
#             accession = fields[0]  # accession.version
#             taxid = fields[2]      # taxid
#             accession_to_taxid[accession] = taxid
#     return accession_to_taxid

# # 加载 Accession 到 TaxID 的映射
# accession_taxid_map = build_accession_taxid_map("nucl_gb.accession2taxid")
# print('构建字典完成')

# # 读取 Accession 号列表并查找 TaxID，生成 TSV 文件
# with open("host_version_accesssion.txt", "r") as infile, open("accession_to_taxid.tsv", "w") as outfile:
#     # 写入 TSV 文件的标题行
#     outfile.write("Accession\tTaxID\n")
    
#     for line in infile:
#         accession = line.strip()
#         taxid = accession_taxid_map.get(accession, "Not Found")  # 查找 TaxID
#         outfile.write(f"{accession}\t{taxid}\n")


import pandas as pd

def get_taxid_for_accessions(accession_file, accession_mapping_file, output_file):
    """
    根据 Accession 文件，生成 Accession 对应的 TaxID 文件（保留未匹配到的 Accession）。
    :param accession_file: 包含 Accession 列表的 .txt 文件路径
    :param accession_mapping_file: accession2taxid 映射文件路径
    :param output_file: 输出的 Accession 与 TaxID 对应的文件路径
    """
    # 读取 Accession 文件，每行一个 Accession
    accession_data = pd.read_csv(accession_file, header=None, names=["accession"], dtype=str)
    
    # 读取 accession2taxid 数据（修改列映射）
    accession_mapping_data = pd.read_csv(
        accession_mapping_file,
        sep='\t',
        header=None,
        names=["accession", "accession.version", "taxid", "gi"],  # 对应的列名
        usecols=["accession", "taxid"],  # 仅保留需要的列
        dtype=str,
        low_memory=False
    )
    
    # 按 Accession 进行合并
    merged_data = pd.merge(accession_data, accession_mapping_data, on="accession", how="left")
    
    # 保存结果到文件（包括未匹配到的 Accession）
    merged_data[["accession", "taxid"]].to_csv(output_file, sep='\t', index=False, header=True)

# 文件路径
accession_input = "host_accession.txt"  # 输入 Accession 文件（每行一个 Accession）
accession_mapping_file = "nucl_gb.accession2taxid"  # accession2taxid 文件路径
output_txt = "accession_to_taxid_with_unmatched.txt"  # 输出文件路径

# 调用函数生成结果
get_taxid_for_accessions(accession_input, accession_mapping_file, output_txt)
