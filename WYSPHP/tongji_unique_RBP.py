import pandas as pd
import re
# ===== 输入输出路径 =====
input_tsv = "RBP_phage.tsv"        # 你的tsv文件路径
output_txt = "unique_phage_RBP.txt"  # 输出txt文件路径

# ===== 读取TSV =====
df = pd.read_csv(input_tsv, sep='\t')

prefixes = set()

for name in df['protein_name']:
    if name.startswith(('NC_', 'NZ_')):
        # 保留前两个下划线前的部分，例如 NC_004167_CDS_... -> NC_004167
        match = re.match(r'^(NC|NZ)_[^_]+', name)
        if match:
            prefixes.add(match.group(0))
    else:
        # 否则保留第一个下划线前的部分
        prefix = name.split('_')[0]
        prefixes.add(prefix)

# ===== 写入TXT =====
with open(output_txt, 'w') as f:
    for p in sorted(prefixes):
        f.write(p + '\n')

print(f"唯一前缀数量: {len(prefixes)}")
print(f"结果已保存至: {output_txt}")