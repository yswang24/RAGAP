# ####好用 生成总的
# import pandas as pd
# import glob
# import os

# # 输入目录
# phage_dir = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_proteinemb_parquet"
# host_dir = "/home/wangjingyuan/wys/WYSPHP/esm_embeddings_35_host_parquet_final"

# protein_catalog = []
# phage_edges = []
# host_edges = []

# # ===== 处理 phage 蛋白 =====
# for file in glob.glob(os.path.join(phage_dir, "*.parquet")):
#     phage_id = os.path.splitext(os.path.basename(file))[0]  # 比如 AB626963
#     df = pd.read_parquet(file)

#     for _, row in df.iterrows():
#         protein_id = row["seq_id"]
#         emb = row.drop("seq_id").tolist()

#         protein_catalog.append({
#             "protein_id": protein_id,
#             "source_type": "phage",
#             "source_id": phage_id,
#             "embedding": emb
#         })
#         phage_edges.append((phage_id, protein_id))

# # ===== 处理 host 蛋白 =====
# for file in glob.glob(os.path.join(host_dir, "*.parquet")):
#     host_id = os.path.splitext(os.path.basename(file))[0]  # 比如 GCF_000005845
#     df = pd.read_parquet(file)

#     for _, row in df.iterrows():
#         protein_id = row["seq_id"]
#         emb = row.drop("seq_id").tolist()

#         protein_catalog.append({
#             "protein_id": protein_id,
#             "source_type": "host",
#             "source_id": host_id,
#             "embedding": emb
#         })
#         host_edges.append((host_id, protein_id))

# # ===== 保存结果 =====
# protein_catalog = pd.DataFrame(protein_catalog)
# protein_catalog.to_parquet("/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/protein_catalog.parquet", index=False)

# pd.DataFrame(phage_edges, columns=["phage_id", "protein_id"]).to_parquet("/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_protein_edges.parquet", index=False)
# pd.DataFrame(host_edges, columns=["host_id", "protein_id"]).to_parquet("/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/host_protein_edges.parquet", index=False)

# print("✅ 整理完成：protein_catalog.parquet, phage_protein_edges.parquet, host_protein_edges.parquet")

####生成host_RBP_parquet文件

# # stream_write_protein_catalog.py
# import glob, os
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# phage_dir = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_proteinemb_parquet"
# out_protein_parquet = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/protein_phage_catalog.parquet"
# out_phage_edges = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_protein_edges.parquet"

# # # 如果已经存在输出文件，先删除（谨慎）
# # if os.path.exists(out_protein_parquet):
# #     os.remove(out_protein_parquet)
# # if os.path.exists(out_host_edges):
# #     os.remove(out_host_edges)

# # 用于 edges 的临时缓冲（写成 parquet 时也用流式）
# edges_schema = pa.schema([('phage_id', pa.string()), ('protein_id', pa.string())])

# # 准备 ParquetWriter（我们将写入一个列：protein_id, source_type, source_id, embedding(为 list<float>)）
# # 但要提前确定 embedding 长度：下面会在第一次文件里获取 embedding 长度并构建 list 类型
# first = True
# writer = None
# edges_writer = None

# file_list = glob.glob(os.path.join(phage_dir, "*.parquet"))
# file_list.sort()

# for file in file_list:
#     phage_id = os.path.splitext(os.path.basename(file))[0]
#     # 只读必要列：假设原文件有 "seq_id" 和若干 embedding 列
#     # 如果你的文件是列名为 seq_id 与 embedding 列名不同，请调整 columns 参数
#     df = pd.read_parquet(file)  # 这里通常单个文件不大，直接读入内存是可以的

#     # 确保 seq_id 存在
#     if "seq_id" not in df.columns:
#         raise RuntimeError(f"file {file} missing seq_id")

#     # 构造 embedding 列：把除 seq_id 以外的列合并为 list
#     emb_cols = [c for c in df.columns if c != "seq_id"]
#     # 下面生成 list-of-lists
#     embeddings = df[emb_cols].values.tolist()

#     # 构造 pyarrow Table
#     prot_ids = df["seq_id"].astype(str).tolist()
#     src_types = ["phage"] * len(prot_ids)
#     src_ids = [phage_id] * len(prot_ids)

#     # 如果是第一次，建立 writer，指定 embedding 类型为 list<float32>
#     if first:
#         emb_len = len(embeddings[0]) if embeddings else 0
#         # 使用 float32（节省空间），如果你的 embedding 原本是 float64，可改为 float64
#         pa_emb_type = pa.list_(pa.float32())
#         schema = pa.schema([
#             ('protein_id', pa.string()),
#             ('source_type', pa.string()),
#             ('source_id', pa.string()),
#             ('embedding', pa_emb_type)
#         ])
#         writer = pq.ParquetWriter(out_protein_parquet, schema, compression='snappy')
#         edges_writer = pq.ParquetWriter(out_phage_edges, edges_schema, compression='snappy')
#         first = False

#     # 构造 arrow arrays
#     protein_arr = pa.array(prot_ids, type=pa.string())
#     source_type_arr = pa.array(src_types, type=pa.string())
#     source_id_arr = pa.array(src_ids, type=pa.string())
#     # 注意：把嵌套 list 转为 float32
#     # cast to float32
#     emb_lists = [[float(x) for x in row] for row in embeddings]
#     embedding_arr = pa.array(emb_lists, type=pa.list_(pa.float32()))

#     table = pa.Table.from_arrays([protein_arr, source_type_arr, source_id_arr, embedding_arr],
#                                  names=['protein_id','source_type','source_id','embedding'])
#     writer.write_table(table)

#     # 写 edges
#     phage_col = pa.array([phage_id]*len(prot_ids), type=pa.string())
#     prot_col = pa.array(prot_ids, type=pa.string())
#     edges_table = pa.Table.from_arrays([phage_col, prot_col], names=['phage_id','protein_id'])
#     edges_writer.write_table(edges_table)

#     # 小进度提示
#     print(f"wrote {len(prot_ids)} proteins from {phage_id}")

# # 关闭 writer
# if writer:
#     writer.close()
# if edges_writer:
#     edges_writer.close()

# print("✅ 完成：流式写入 protein_catalog 与 host_edges")




# stream_write_protein_catalog_all.py
import glob
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ======== 路径配置 ========
phage_dir = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_proteinemb_parquet"
host_dir  = "/home/wangjingyuan/wys/duibi/esm_embeddings_650_host_parquet_final"

out_protein_parquet = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/protein_catalog.parquet"
out_phage_edges     = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_protein_edges.parquet"
out_host_edges      = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/host_protein_edges.parquet"

# ======== 可选：如需覆盖输出，先删除旧文件（谨慎） ========
# for p in (out_protein_parquet, out_phage_edges, out_host_edges):
#     if os.path.exists(p):
#         os.remove(p)

# ======== 统一 schema（catalog 为合并） ========
# catalog 列：protein_id, source_type, source_id, embedding(list<float32>)
catalog_schema = pa.schema([
    ('protein_id',  pa.string()),
    ('source_type', pa.string()),     # "phage" 或 "host"
    ('source_id',   pa.string()),     # phage_id 或 host_id
    ('embedding',   pa.list_(pa.float32())),
])

# edges 的 schema
phage_edges_schema = pa.schema([('phage_id', pa.string()), ('protein_id', pa.string())])
host_edges_schema  = pa.schema([('host_id',  pa.string()), ('protein_id', pa.string())])

# ======== Writer 准备 ========
catalog_writer = pq.ParquetWriter(out_protein_parquet, schema=catalog_schema, compression="snappy")
phage_edges_writer = pq.ParquetWriter(out_phage_edges, schema=phage_edges_schema, compression="snappy")
host_edges_writer  = pq.ParquetWriter(out_host_edges,  schema=host_edges_schema,  compression="snappy")

def process_dir(dir_path, source_type, edges_writer, id_col_name):
    """
    dir_path: 目录
    source_type: "phage" 或 "host"
    edges_writer: 对应的 edges writer
    id_col_name: edges 中的 id 列名：phage 用 'phage_id'，host 用 'host_id'
    """
    file_list = glob.glob(os.path.join(dir_path, "*.parquet"))
    file_list.sort()

    total_prot = 0
    for file in file_list:
        source_id = os.path.splitext(os.path.basename(file))[0]  # phage_id 或 host_id
        try:
            # 读 parquet（通常单文件不大，直接读内存；若特别大可考虑 pyarrow.dataset 分批读列）
            df = pd.read_parquet(file)

            if "seq_id" not in df.columns:
                print(f"⚠️  跳过（缺少 seq_id 列）: {file}")
                continue

            # 构造 embedding：把除 seq_id 之外的列合成 list
            emb_cols = [c for c in df.columns if c != "seq_id"]
            if len(emb_cols) == 0:
                print(f"⚠️  跳过（没有 embedding 列）: {file}")
                continue

            # 过滤空行
            if df.shape[0] == 0:
                print(f"⚠️  跳过（空文件）: {file}")
                continue

            # 生成 list-of-lists 并转 float32
            embeds = df[emb_cols].values.tolist()
            embeds = [[float(x) for x in row] for row in embeds]

            prot_ids = df["seq_id"].astype(str).tolist()
            src_types = [source_type] * len(prot_ids)
            src_ids   = [source_id]  * len(prot_ids)

            # ---- 写 catalog ----
            table = pa.Table.from_arrays(
                [
                    pa.array(prot_ids, type=pa.string()),
                    pa.array(src_types, type=pa.string()),
                    pa.array(src_ids,   type=pa.string()),
                    pa.array(embeds,    type=pa.list_(pa.float32())),
                ],
                names=['protein_id', 'source_type', 'source_id', 'embedding']
            )
            catalog_writer.write_table(table)

            # ---- 写 edges ----
            edges_table = pa.Table.from_arrays(
                [
                    pa.array([source_id] * len(prot_ids), type=pa.string()),
                    pa.array(prot_ids, type=pa.string())
                ],
                names=[id_col_name, 'protein_id']
            )
            edges_writer.write_table(edges_table)

            total_prot += len(prot_ids)
            print(f"✅ wrote {len(prot_ids):6d} proteins from {source_type}:{source_id}")
        except Exception as e:
            print(f"❌ 处理失败: {file}\n   {e}")

    return total_prot

# ======== 执行 ========
n_phage = process_dir(phage_dir, "phage", phage_edges_writer, "phage_id")
n_host  = process_dir(host_dir,  "host",  host_edges_writer,  "host_id")

# 关闭 writers
catalog_writer.close()
phage_edges_writer.close()
host_edges_writer.close()

print(f"🎉 完成：catalog={n_phage+n_host} 个蛋白；phage_edges≈{n_phage}；host_edges≈{n_host}")
print("输出：")
print(" -", out_protein_parquet)
print(" -", out_phage_edges)
print(" -", out_host_edges)
