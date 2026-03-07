'''cut -f2 /home/wangjingyuan/wys/wys_shiyan/edges/host_taxonomy_edges.tsv | sort -u | \taxonkit lineage | \taxonkit reformat -f "{g}" > taxid_genus.tsv'''
import pandas as pd

def clean_taxid_species(input_tsv, output_tsv):
    # 读取文件
    df = pd.read_csv(input_tsv, sep='\t', header=None)

    # 保留第1列和第3列
    df = df[[0, 2]]
    df.columns = ['taxid', 'species']

    # 输出结果
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"✅ 整理完成，结果已保存到 {output_tsv}")

# 使用示例
clean_taxid_species("/home/wangjingyuan/wys/wys_shiyan/taxid_genus.tsv", "/home/wangjingyuan/wys/wys_shiyan/taxid_genus.tsv")
