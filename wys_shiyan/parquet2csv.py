# import pandas as pd

# # 读取 parquet 文件
# df = pd.read_parquet("/home/wangjingyuan/lyf/GraphMetaEvo/data_spilt/PRJCA007414/error_correction/round1_dnabertemb/contig_1.parquet")

# # 保存为 csv 文件
# df.to_csv("/home/wangjingyuan/lyf/GraphMetaEvo/data_spilt/PRJCA007414/error_correction/round1_dnabertemb_csv/contig_1.csv", index=False)  # index=False 表示不保存行索引




import os
import pandas as pd
from pathlib import Path

# 输入与输出目录
input_dir = Path("/home/wangjingyuan/lyf/GraphMetaEvo/data_spilt/PRJCA007414/error_correction/round1_dnabertemb")
output_dir = Path("/home/wangjingyuan/lyf/GraphMetaEvo/data_spilt/PRJCA007414/error_correction/round1_dnabertemb_csv")

# 若输出目录不存在则创建
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历所有 .parquet 文件
for parquet_file in input_dir.glob("*.parquet"):
    try:
        # 读取 parquet 文件
        df = pd.read_parquet(parquet_file)

        # 生成对应的 csv 文件路径
        csv_file = output_dir / f"{parquet_file.stem}.csv"

        # 保存为 csv
        df.to_csv(csv_file, index=False)

        print(f"✅ 已转换: {parquet_file.name} -> {csv_file.name}")

    except Exception as e:
        print(f"❌ 转换失败: {parquet_file.name}, 错误原因: {e}")
