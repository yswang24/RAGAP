import pandas as pd

# 输入与输出文件路径
input_file = "/home/wangjingyuan/wys/duibi/phage_catalog0.parquet"
output_file = "/home/wangjingyuan/wys/duibi/phage_catalog.parquet"

# 读取 parquet 文件
df = pd.read_parquet(input_file)

# 确保存在 phage_id 列
if "phage_id" not in df.columns:
    raise ValueError("❌ parquet 文件中没有 'phage_id' 列")

# 去掉 phage_id 结尾的 .数字，例如 .1 或 .2
df["phage_id"] = df["phage_id"].str.replace(r"\.\d+$", "", regex=True)

# 保存为新的 parquet 文件
df.to_parquet(output_file, index=False)

print("✅ 已去掉 phage_id 的 .数字 后缀，结果保存到：", output_file)
