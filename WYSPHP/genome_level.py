
'''✅ L2 归一化 → genome-level 平均聚合'''

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os, random, json
from tqdm import tqdm

INPUT = "/home/wangjingyuan/wys/WYSPHP/protein_catalog.parquet"
OUTPUT_DIR = "genome_level_protein_parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def aggregate_embeddings(df, source_type, output_path):
    print(f"[+] 聚合 {source_type} ({len(df)} 蛋白质条目)...")
    # 展开 embedding 列
    embeddings = np.vstack(df['embedding'].apply(np.array).values)
    embeddings = l2_normalize(embeddings)
    df = df.copy()
    df['embedding'] = list(embeddings)

    grouped = df.groupby("source_id")['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    result_df = pd.DataFrame({
        "source_id": grouped.index,
        "embedding": grouped.values
    })
    result_df.to_parquet(output_path, index=False)
    print(f"[+] 已保存: {output_path} ({len(result_df)} genomes)")
    return result_df

def verify_aggregation(df, aggregated_df, n=3):
    print("[+] 开始随机抽样验证...")
    sample_genomes = random.sample(list(aggregated_df['source_id']), min(n, len(aggregated_df)))
    for genome in sample_genomes:
        raw = df[df['source_id'] == genome]
        raw_emb = np.vstack(raw['embedding'].apply(np.array).values)
        raw_emb = l2_normalize(raw_emb)
        manual_mean = raw_emb.mean(axis=0)
        diff = np.linalg.norm(manual_mean - aggregated_df.loc[aggregated_df['source_id']==genome, 'embedding'].values[0])
        print(f"    - {genome}: 差异范数 = {diff:.6f} (应接近 0)")

def main():
    print(f"[+] 读取 parquet 文件: {INPUT}")
    df = pd.read_parquet(INPUT)
    print(f"[+] 数据规模: {len(df)} 蛋白质, {df['source_id'].nunique()} 个 genomes")
    
    # 聚合 phage
    phage_df = df[df['source_type'] == 'phage']
    phage_out = os.path.join(OUTPUT_DIR, "genome_phage_genome_embeddings.parquet")
    phage_agg = aggregate_embeddings(phage_df, "phage", phage_out)

    # 聚合 host
    host_df = df[df['source_type'] == 'host']
    host_out = os.path.join(OUTPUT_DIR, "genome_host_genome_embeddings.parquet")
    host_agg = aggregate_embeddings(host_df, "host", host_out)

    # 验证环节
    verify_aggregation(phage_df, phage_agg, n=3)
    verify_aggregation(host_df, host_agg, n=3)
    print("[+] 验证完成！")

if __name__ == "__main__":
    main()
