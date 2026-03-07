#!/usr/bin/env python3
"""
build_catalogs.py
合并 embeddings -> 生成 host_catalog.parquet 和 phage_catalog.parquet
用法:
  python build_catalogs.py \
    --phage_dir /home/wangjingyuan/wys/WYSPHP/dnabert4_phage_embeddings_final \
    --host_dir /home/wangjingyuan/wys/WYSPHP/dnabert4_host_embeddings_final \
    --taxonomy_parquet /home/wangjingyuan/wys/WYSPHP/taxonomy_tree/zuiyou/taxonomy_poincare_dep_neg_120_4_.parquet \
    --out_dir data_processed4
"""

import argparse, os, json, glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def ensure_list(vec):
    if isinstance(vec, list):
        return vec
    if isinstance(vec, (np.ndarray,)):
        return vec.tolist()
    try:
        return json.loads(vec) if isinstance(vec, str) else [float(x) for x in vec]
    except Exception:
        return []


def load_all_parquets(folder):
    """读取文件夹下所有 parquet 并合并"""
    dfs = []
    for fn in glob.glob(os.path.join(folder, "*.parquet")):
        try:
            df = pd.read_parquet(fn)
            dfs.append(df)
        except Exception as e:
            print("⚠️ 跳过文件", fn, "错误:", e)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phage_dir", required=True, help="包含 phage parquet 文件的文件夹")
    p.add_argument("--host_dir", required=True, help="包含 host parquet 文件的文件夹")
    p.add_argument("--taxonomy_parquet", required=True)
    p.add_argument("--out_dir", default="data/processed")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading taxonomy parquet ...")
    tax_df = pd.read_parquet(args.taxonomy_parquet)
    tax_df['taxid'] = tax_df['taxid'].astype(str)
    tax_map = {}
    if 'tangent_emb' in tax_df.columns:
        for _, r in tax_df.iterrows():
            tax_map[str(r['taxid'])] = ensure_list(r['tangent_emb'])
    else:
        raise RuntimeError("taxonomy parquet must have 'tangent_emb' column")

    # ==== HOST ====
    print("Loading host parquet files ...")
    host_df = load_all_parquets(args.host_dir)
    if host_df.empty:
        raise RuntimeError("host_dir 中没有有效的 parquet 文件")

    # 确认列名（sequence_id, embedding, source_file, host_gcf, host_species_taxid）
    if 'host_gcf' not in host_df.columns:
        raise RuntimeError("host parquet 缺少 host_gcf 列")
    if 'host_species_taxid' not in host_df.columns:
        raise RuntimeError("host parquet 缺少 host_species_taxid 列")

    host_df['host_species_taxid'] = host_df['host_species_taxid'].astype(str)

    tangent_list, missing_taxids = [], set()
    for _, r in host_df.iterrows():
        st = str(r['host_species_taxid'])
        if st in tax_map:
            tangent_list.append(tax_map[st])
        else:
            tangent_list.append(None)
            missing_taxids.add(st)
    host_df['tangent_emb'] = tangent_list

    print(f"Host rows: {len(host_df)}; missing taxonomy for {len(missing_taxids)} distinct species taxids")
    if missing_taxids:
        with open(os.path.join(args.out_dir, "missing_host_taxids.txt"), "w") as f:
            for t in sorted(missing_taxids):
                f.write(t + "\n")
        print("Wrote missing_host_taxids.txt")

    # 标准化并保存
    host_records = []
    for _, r in host_df.iterrows():
        host_records.append({
            'host_gcf': str(r['host_gcf']),
            'sequence_id': str(r['sequence_id']),
            'host_species_taxid': str(r['host_species_taxid']),
            'host_dna_emb': ensure_list(r.get('embedding', r.get('host_dna_emb', []))),
            'tangent_emb': ensure_list(r['tangent_emb']) if r['tangent_emb'] is not None else None,
        })

    host_out = os.path.join(args.out_dir, "host_catalog.parquet")
    host_table = pa.Table.from_pydict({
        'host_gcf': [r['host_gcf'] for r in host_records],
        'sequence_id': [r['sequence_id'] for r in host_records],
        'host_species_taxid': [r['host_species_taxid'] for r in host_records],
        'host_dna_emb': [r['host_dna_emb'] for r in host_records],
        'tangent_emb': [r['tangent_emb'] for r in host_records],
    })
    pq.write_table(host_table, host_out, compression="snappy")
    print("WROTE", host_out, "rows=", len(host_records))

    # # ==== PHAGE ====
    # print("Loading phage parquet files ...")
    # phage_df = load_all_parquets(args.phage_dir)
    # if phage_df.empty:
    #     raise RuntimeError("phage_dir 中没有有效的 parquet 文件")

    # # 确认列名（sequence_id, embedding, source_file, virus_taxid）
    # if 'sequence_id' not in phage_df.columns or 'embedding' not in phage_df.columns:
    #     raise RuntimeError("phage parquet 缺少 sequence_id 或 embedding 列")

    # phage_records = []
    # for _, r in phage_df.iterrows():
    #     pid = str(r['sequence_id'])
    #     phage_records.append({
    #         'phage_id': pid,
    #         'phage_dna_emb': ensure_list(r.get('embedding', r.get('phage_dna_emb', []))),
    #     })

    # phage_out = os.path.join(args.out_dir, "phage_catalog.parquet")
    # ph_table = pa.Table.from_pydict({
    #     'phage_id': [r['phage_id'] for r in phage_records],
    #     'phage_dna_emb': [r['phage_dna_emb'] for r in phage_records],
    # })
    # pq.write_table(ph_table, phage_out, compression="snappy")
    # print("WROTE", phage_out, "rows=", len(phage_records))

    # print("Done.")


if __name__ == '__main__':
    main()



# #!/usr/bin/env python3
# """
# build_catalogs.py
# 合并 embeddings -> 生成 host_catalog.parquet 和 phage_catalog.parquet
# 用法:
#   python build_catalogs.py \
#     --phage_parquet /home/wangjingyuan/wys/WYSPHP/dnabert4_phage_embeddings_final \
#     --host_parquet /home/wangjingyuan/wys/WYSPHP/dnabert4_host_embeddings_final \
#     --taxonomy_parquet /home/wangjingyuan/wys/WYSPHP/taxonomy_tree/zuiyou/taxonomy_poincare_dep_neg_120_4_.parquet \
#     --out_dir data_processed4
# """

# import argparse, os, json
# import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq


# def ensure_list(vec):
#     """确保向量是 python list 格式"""
#     if isinstance(vec, list):
#         return vec
#     if isinstance(vec, (np.ndarray,)):
#         return vec.tolist()
#     try:
#         return json.loads(vec) if isinstance(vec, str) else [float(x) for x in vec]
#     except Exception:
#         return []


# def load_parquets(path):
#     """加载一个 parquet 文件或整个目录"""
#     if os.path.isdir(path):
#         files = [os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(".parquet")]
#     else:
#         files = [path]

#     dfs = []
#     for f in files:
#         df = pd.read_parquet(f)
#         dfs.append(df)
#     if not dfs:
#         return pd.DataFrame()
#     return pd.concat(dfs, ignore_index=True)

# def deduplicate_df(df, name="df"):
#     # 如果有 sequence_id 和 source_file，就基于它们去重
#     if {"sequence_id", "source_file"} <= set(df.columns):
#         before = len(df)
#         df = df.drop_duplicates(subset=["sequence_id", "source_file"]).reset_index(drop=True)
#         after = len(df)
#         print(f"{name}: dropped {before - after} duplicate rows (based on sequence_id+source_file)")
#     else:
#         # 如果没有，就只用所有非嵌入列
#         non_array_cols = [c for c in df.columns if c != "embedding"]
#         before = len(df)
#         df = df.drop_duplicates(subset=non_array_cols).reset_index(drop=True)
#         after = len(df)
#         print(f"{name}: dropped {before - after} duplicate rows (based on non-array cols)")
#     return df

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--phage_parquet", required=True, help="phage parquet 文件或目录")
#     p.add_argument("--host_parquet", required=True, help="host parquet 文件或目录")
#     p.add_argument("--taxonomy_parquet", required=True, help="taxonomy parquet 文件")
#     p.add_argument("--out_dir", default="data/processed")
#     args = p.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

    

#     # === Taxonomy ===
#     print("Loading taxonomy parquet ...")
#     tax_df = pd.read_parquet(args.taxonomy_parquet)
#     tax_df["taxid"] = tax_df["taxid"].astype(str)
#     tax_map = {}
#     if "tangent_emb" in tax_df.columns:
#         for _, r in tax_df.iterrows():
#             tax_map[str(r["taxid"])] = ensure_list(r["tangent_emb"])
#     else:
#         raise RuntimeError("taxonomy parquet must have 'tangent_emb' column")

#     # === Host ===
#     print("Loading host parquet files ...")
#     host_df = load_parquets(args.host_parquet)

#     # 预期列: host_gcf, host_dna_emb, host_species_taxid
#     if "host_gcf" not in host_df.columns:
#         if "gcf" in host_df.columns:
#             host_df = host_df.rename(columns={"gcf": "host_gcf"})
#         else:
#             raise RuntimeError("host parquet missing 'host_gcf' column")

#     host_df["host_species_taxid"] = host_df["host_species_taxid"].astype(str)

#     # taxonomy 映射
#     tangent_list = []
#     missing_taxids = set()
#     for _, r in host_df.iterrows():
#         st = str(r["host_species_taxid"])
#         if st in tax_map:
#             tangent_list.append(tax_map[st])
#         else:
#             tangent_list.append(None)
#             missing_taxids.add(st)
#     host_df["tangent_emb"] = tangent_list

#     print(f"Host rows (before dedup): {len(host_df)}")
#     if missing_taxids:
#         with open(os.path.join(args.out_dir, "missing_host_taxids.txt"), "w") as f:
#             for t in sorted(missing_taxids):
#                 f.write(t + "\n")
#         print(f"Wrote missing_host_taxids.txt ({len(missing_taxids)} missing)")

#     # 去重
#     host_df = deduplicate_df(host_df, "host_df")
#     print(f"Host rows (after dedup): {len(host_df)}")

#     # 保存
#     host_out = os.path.join(args.out_dir, "host_catalog4.parquet")
#     host_table = pa.Table.from_pandas(host_df)
#     pq.write_table(host_table, host_out, compression="snappy")
#     print("WROTE", host_out, "rows=", len(host_df))

#     # === Phage ===
#     print("Loading phage parquet files ...")
#     phage_df = load_parquets(args.phage_parquet)

#     if "phage_id" not in phage_df.columns and "sequence_id" in phage_df.columns:
#         phage_df = phage_df.rename(columns={"sequence_id": "phage_id"})
#     if "phage_id" not in phage_df.columns:
#         raise RuntimeError("phage parquet missing 'phage_id' or 'refseq'")

#     print(f"Phage rows (before dedup): {len(phage_df)}")
#     phage_df = deduplicate_df(phage_df, "phage_df")
#     print(f"Phage rows (after dedup): {len(phage_df)}")

#     phage_out = os.path.join(args.out_dir, "phage_catalog4.parquet")
#     ph_table = pa.Table.from_pandas(phage_df)
#     pq.write_table(ph_table, phage_out, compression="snappy")
#     print("WROTE", phage_out, "rows=", len(phage_df))

#     print("Done.")


# if __name__ == "__main__":
#     main()
