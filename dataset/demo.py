from Bio import SeqIO
import os
import pandas as pd
import subprocess
#拆分文件
'''
def split_dna_file(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for record in SeqIO.parse(input_file, "fasta"):
        output_path = os.path.join(output_dir, f"{record.id}.fasta")
        SeqIO.write(record, output_path, "fasta")

split_dna_file("virushostdb.formatted.genomic.fna", "split_sequences/")
'''
#筛选phage
'''
def filter_phage(input_tsv, output_tsv):
    data = pd.read_csv(input_tsv, sep="\t", header=None)
    phage_data = data[data[1].str.contains("phage", case=False, na=False)]
    phage_data.to_csv(output_tsv, sep="\t", index=False, header=False)

filter_phage("virushostdb.daily.tsv", "filtered_phage.tsv")
'''
#删除真核生物记录
'''
def remove_eukaryotic_records(input_tsv, output_tsv):
    data = pd.read_csv(input_tsv, sep="\t", header=None)  
    # 删除包含 "Eukaryota" 的记录
    prokaryotic_data = data[~data[9].str.contains("Eukaryota", case=False, na=False)]   
    # 保存结果
    prokaryotic_data.to_csv(output_tsv, sep="\t", index=False, header=False)
remove_eukaryotic_records("filtered_phage.tsv", "prokaryotic_hosts.tsv")
'''
#筛选ref
'''
import pandas as pd

# 读取TSV文件
file_path = 'prokaryotic_hosts.tsv'  # 请替换为你的文件路径
df = pd.read_csv(file_path, sep='\t')  # 使用sep='\t'指定制表符分隔

# 筛选第12列中包含"RefSeq"的行，并确保保留所有列信息
filtered_df = df[df.iloc[:, 11].str.contains('RefSeq', na=False)]  # iloc[:, 11]表示第12列

# 显示筛选后的结果，确保包含所有原始列
print(filtered_df)

# 如果你想将筛选后的数据保存为新的TSV文件，保留所有列
filtered_df.to_csv('prokaryotic_hosts_ref.tsv', sep='\t', index=False)
'''
# #获取宿主分类标签

# def get_host_lineages(taxid_file, output_file):
#     #command = f"taxonkit lineage {taxid_file} | taxonkit reformat -F -P"
#     command = f"taxonkit lineage --show-rank {taxid_file} | sed 's/; t__[^;]*//g' > {output_file}"
#     #command = f"taxonkit lineage {taxid_file} > {output_file}"

#     subprocess.run(command, shell=True)

# #提取宿主的 taxid
# def extract_taxid(input_tsv, taxid_file):
#     data = pd.read_csv(input_tsv, sep="\t", header=None)
#     data[7] = data[7].fillna(0)#NAN填充0
#     #data = data.dropna(subset=[7]) #删除NAN
#     data[7]=data[7].astype(int)
#     data[7].to_csv(taxid_file, index=False, header=False)
  
# extract_taxid("prokaryotic_hosts_ref.tsv", "host_taxid.txt")
# get_host_lineages("host_taxid.txt", "host_lineages.txt")

# #提取virus的 taxid
# def virus_extract_taxid(input_tsv, taxid_file):
#     data = pd.read_csv(input_tsv, sep="\t", header=None)
#     data[0] = data[0].fillna(0)#NAN填充0
#     #data = data.dropna(subset=[7]) #删除NAN
#     data[0]=data[0].astype(int)
#     data[0].to_csv(taxid_file, index=False, header=False)

# virus_extract_taxid("prokaryotic_hosts_ref.tsv", "virus_taxid.txt")

# #获取virus分类学标签
# def get_virus_lineages(taxid_file, output_file):
#     #command = f"taxonkit lineage {taxid_file} | taxonkit reformat -F -P"
#     command = f"taxonkit lineage --show-rank {taxid_file} | sed 's/; t__[^;]*//g' > {output_file}"
#     #command = f"taxonkit lineage {taxid_file} > {output_file}"

#     subprocess.run(command, shell=True)

# get_virus_lineages("virus_taxid.txt", "virus_lineages.txt")


#将两文件合并
'''
# 定义文件路径
file1_path = "virus_lineages.txt"
file2_path = "host_lineages.txt"
output_file_path = "merged_output.csv"

# 读取文件1，只取第1列和第2列（索引从0开始）
file1_data = pd.read_csv(file1_path, sep='\t', header=None, usecols=[0, 1], names=['Col1_File1', 'Col2_File1'])

# 读取文件2，只取第2列（索引从0开始）
file2_data = pd.read_csv(file2_path, sep='\t', header=None, usecols=[1], names=['Col2_File2'])

# 合并两组数据
merged_data = pd.concat([file1_data, file2_data], axis=1)

# 将结果保存为一个CSV文件
merged_data.to_csv(output_file_path, index=False)

print(f"文件合并完成，结果保存到 {output_file_path}")
'''

#拓展Accession

import pandas as pd

# 输入 TSV 文件路径
tsv_file = 'phage_host_old.tsv'
# 输出结果 TSV 文件路径
output_file = 'phage_host_old_all.tsv'

# 读取 TSV 文件
df = pd.read_csv(tsv_file, sep='\t')

# 存储结果的列表
expanded_rows = []

# 遍历每一行
for index, row in df.iterrows():
    # 获取第四列的 Accession 数据（可能是多个逗号分隔的Accession号）
    accession_column = row.iloc[0]
    
    # 如果第四列是逗号分隔的多个 Accession 号
    accession_list = accession_column.split(',')
    
    # 对于每个 Accession 号，生成新的行，复制其余列的数据
    for accession in accession_list:
        new_row = row.copy()  # 复制当前行数据
        new_row.iloc[0] = accession  # 将第四列替换为单个 Accession 号
        expanded_rows.append(new_row)

# 将扩展后的数据转换为 DataFrame
expanded_df = pd.DataFrame(expanded_rows)

# 保存为新的 TSV 文件
expanded_df.to_csv(output_file, sep='\t', index=False)

print(f"处理完成！已保存为: {output_file}")


# import pandas as pd

# # 输入 TSV 文件路径
# tsv_file = 'prokaryotic_hosts.tsv'
# # 输入 TXT 文件路径
# txt_file = 'phage_accession.txt'
# # 输出结果 TSV 文件路径
# output_file = 'prokaryotic_host1.tsv'

# # 读取 TSV 文件
# df = pd.read_csv(tsv_file, sep='\t')

# # 读取 TXT 文件中的 Accession 号
# with open(txt_file, 'r') as file:
#     accession_list = [line.strip() for line in file.readlines()]

# # 筛选 TSV 文件中第四列为 Accession 号的行
# filtered_df = df[df.iloc[:, 3].isin(accession_list)]

# # 保存筛选后的结果为新的 TSV 文件
# filtered_df.to_csv(output_file, sep='\t', index=False)

# print(f"筛选完成！已保存为: {output_file}")




