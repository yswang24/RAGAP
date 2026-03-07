import pandas as pd

# === 输入输出文件路径 ===
input_file = "/home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF.tsv"      # 原始文件，例如：KF787094.tsv
output_file = "TEST_PAIR_TAX_filter_new_GCF_na_edges.tsv"  # 输出文件路径

# === 读取 TSV 文件 ===
df = pd.read_csv(input_file, sep="\t")

# === 检查必要列是否存在 ===
required_cols = ["accession", "GCF_ids"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ 缺少必要列: {missing}")

# === 拆分 GCF_ids 列（以分号为分隔符） ===
df["GCF_ids"] = df["GCF_ids"].astype(str)  # 确保是字符串
df = df.assign(GCF_ids=df["GCF_ids"].str.split(";")).explode("GCF_ids")

# === 去除空值和多余空白 ===
df["GCF_ids"] = df["GCF_ids"].str.strip()
df = df[df["GCF_ids"] != ""]

# === 构建新的 DataFrame ===
edges = pd.DataFrame({
    "src_id": df["accession"],
    "dst_id": df["GCF_ids"],
    "edge_type": "phage-host",
    "weight": 1
})

# === 输出为 TSV ===
edges.to_csv(output_file, sep="\t", index=False)

print(f"✅ 已生成关系文件: {output_file}")
print(f"共 {len(edges)} 条边。")