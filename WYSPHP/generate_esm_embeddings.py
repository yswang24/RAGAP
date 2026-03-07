# import os
# import torch
# import esm
# from tqdm import tqdm
# import argparse
# import pickle
# from multiprocessing import Pool, cpu_count

# def parse_faa(faa_path):
#     """解析 .faa 文件，返回 (protein_id, seq) 列表。"""
#     seqs = []
#     with open(faa_path) as f:
#         header, seq = None, []
#         for line in f:
#             line = line.strip()
#             if line.startswith(">"):
#                 if header and seq:
#                     seqs.append((header, "".join(seq)))
#                 header = line[1:].split()[0]
#                 seq = []
#             else:
#                 seq.append(line.replace("*", ""))  # 移除尾部 '*'
#         if header and seq:
#             seqs.append((header, "".join(seq)))
#     return seqs

# def process_single_faa(args):
#     faa_file, faa_dir, output_dir, model_name, batch_size, repr_layer, device = args
#     if not faa_file.endswith(".faa"):
#         return

#     base = os.path.splitext(faa_file)[0]
#     out_path = os.path.join(output_dir, base + ".pkl")
#     if os.path.exists(out_path):
#         return f"[跳过] {faa_file}"

#     faa_path = os.path.join(faa_dir, faa_file)
#     seqs = parse_faa(faa_path)
#     if not seqs:
#         return f"[空文件] {faa_file}"

#     # 每个进程加载模型（会占内存/显存）
#     model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
#     model = model.eval().to(device)
#     batch_converter = alphabet.get_batch_converter()

#     seq_embeddings = {}
#     for i in range(0, len(seqs), batch_size):
#         batch = seqs[i : i + batch_size]
#         labels, sequences = zip(*batch)
#         tokens = batch_converter(batch)[2].to(device)
#         with torch.no_grad():
#             out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
#         reps = out["representations"][repr_layer]
#         for j, (prot_id, seq) in enumerate(batch):
#             emb = reps[j, 1:len(seq)+1].mean(0)
#             seq_embeddings[prot_id] = emb.cpu()

#     with open(out_path, "wb") as f:
#         pickle.dump(seq_embeddings, f)
#     return f"[完成] {faa_file} → {out_path}"

# def main(faa_dir, output_dir, model_name, batch_size=4, repr_layer=33, device="cuda", num_workers=4):
#     os.makedirs(output_dir, exist_ok=True)
#     faa_files = [f for f in os.listdir(faa_dir) if f.endswith(".faa")]

#     pool_args = [
#         (faa_file, faa_dir, output_dir, model_name, batch_size, repr_layer, device)
#         for faa_file in faa_files
#     ]

#     print(f">> 开始并行处理 {len(pool_args)} 个 .faa 文件，使用 {num_workers} 进程")
#     with Pool(processes=num_workers) as pool:
#         for result in tqdm(pool.imap_unordered(process_single_faa, pool_args), total=len(pool_args)):
#             if result:
#                 print(result)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="多进程生成 ESM 蛋白嵌入")
#     parser.add_argument("--faa-dir", required=True)
#     parser.add_argument("--out", required=True)
#     parser.add_argument("--model-name", default="esm2_t33_150M_UR50D")
#     parser.add_argument("--batch-size", type=int, default=4)
#     parser.add_argument("--repr-l", type=int, default=33)
#     parser.add_argument("--device", default="cuda")
#     parser.add_argument("--workers", type=int, default=4, help="并行进程数")
#     args = parser.parse_args()

#     main(
#         args.faa_dir,
#         args.out,
#         args.model_name,
#         args.batch_size,
#         args.repr_l,
#         args.device,
#         args.workers,
#     )



import os
import torch
import esm
from tqdm import tqdm
import argparse
import pickle
from multiprocessing import Pool
import gc

def parse_faa(faa_path):
    seqs = []
    with open(faa_path) as f:
        header, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and seq:
                    seqs.append((header, "".join(seq)))
                header = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.replace("*", "").replace("+", "").replace("#", ""))
        if header and seq:
            seqs.append((header, "".join(seq)))
    return seqs

def process_single_faa(args):
    faa_file, faa_dir, output_dir, model_name, batch_size, repr_layer, device = args
    if not faa_file.endswith(".faa"):
        return

    base = os.path.splitext(faa_file)[0]
    out_path = os.path.join(output_dir, base + ".pkl")
    if os.path.exists(out_path):
        return f"[跳过] {faa_file}"

    faa_path = os.path.join(faa_dir, faa_file)
    seqs = parse_faa(faa_path)
    if not seqs:
        return f"[空文件] {faa_file}"

    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()

        seq_embeddings = {}
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i: i + batch_size]
            labels, sequences = zip(*batch)

            try:
                tokens = batch_converter(batch)[2].to(device)

                with torch.no_grad():
                    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)

                reps = out["representations"][repr_layer]

                for j, (prot_id, seq) in enumerate(batch):
                    print(f"[{faa_file}] 正在处理蛋白质序列: {prot_id} (长度: {len(seq)})")
                    emb = reps[j, 1:len(seq)+1].mean(0)
                    seq_embeddings[prot_id] = emb.cpu()

                del tokens, out, reps
                torch.cuda.empty_cache()
                gc.collect()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[警告] 跳过因显存不足的批次 in {faa_file}: {str(e)[:100]}...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue  # 跳过当前 batch
                else:
                    raise e  # 其他错误照常抛出

        with open(out_path, "wb") as f:
            pickle.dump(seq_embeddings, f)

        return f"[完成] {faa_file} → {out_path}"

    except Exception as e:
        return f"[错误] {faa_file} 处理失败: {e}"

def main(faa_dir, output_dir, model_name, batch_size=4, repr_layer=33, device="cuda", num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    faa_files = [f for f in os.listdir(faa_dir) if f.endswith(".faa")]

    pool_args = [
        (faa_file, faa_dir, output_dir, model_name, batch_size, repr_layer, device)
        for faa_file in faa_files
    ]

    print(f">> 开始并行处理 {len(pool_args)} 个 .faa 文件，使用 {num_workers} 进程")
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_faa, pool_args), total=len(pool_args)):
            if result:
                print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多进程生成 ESM 蛋白嵌入")
    parser.add_argument("--faa-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="esm2_t33_150M_UR50D")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repr-l", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    main(
        args.faa_dir,
        args.out,
        args.model_name,
        args.batch_size,
        args.repr_l,
        args.device,
        args.workers,
    )
