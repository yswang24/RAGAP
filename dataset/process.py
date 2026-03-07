
import pandas as pd

def get_accession_for_taxids(taxid_file, accession_file, output_file):
    """
    根据 TaxID 文件，生成 TaxID 对应的 Accession 号文件（保留未匹配到的 TaxID）。
    :param taxid_file: 包含 TaxID 列表的 .txt 文件路径
    :param accession_file: accession2taxid 映射文件路径
    :param output_file: 输出的 TaxID 与 Accession 对应的文件路径
    """
    # 读取 TaxID 文件
    taxid_data = pd.read_csv(taxid_file, sep='\t', header=None, names=["taxid"], dtype=str)
    
    # 读取 accession2taxid 数据（修改列映射）
    accession_data = pd.read_csv(
        accession_file,
        sep='\t',
        header=None,
        names=["accession", "accession.version", "taxid", "gi"],  # 对应的列名
        usecols=["accession", "taxid"],  # 仅保留需要的列
        dtype=str,
        low_memory=False
    )
    
    # 按 TaxID 进行合并
    merged_data = pd.merge(taxid_data, accession_data, on="taxid", how="left")
    
    # 保存结果到文件（包括未匹配到的 TaxID）
    merged_data[["taxid", "accession"]].to_csv(output_file, sep='\t', index=False, header=True)

# 文件路径
taxid_input = "virus_taxid.txt"  # 输入 TaxID 文件（每行一个 TaxID）
accession_mapping_file = "nucl_gb.accession2taxid"  # accession2taxid 文件路径
output_txt = "taxid_to_accession_with_unmatched.txt"  # 输出文件路径

# 调用函数生成结果
get_accession_for_taxids(taxid_input, accession_mapping_file, output_txt)
