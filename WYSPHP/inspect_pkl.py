
import os
import pickle
import torch
import pandas as pd
from datetime import datetime

def process_pkl_file(pkl_path, out_dir, log_file):
    """读取一个pkl文件并保存为parquet"""
    try:
        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}.parquet")

        # 如果已经存在就跳过
        if os.path.exists(out_path):
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now()}] 跳过: {pkl_path} (已存在)\n")
            return

        # 加载 pkl
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 确保是 dict
        if not isinstance(data, dict):
            raise ValueError(f"{pkl_path} 格式不是 dict")

        # 转换成 DataFrame
        records = []
        for seq_id, tensor in data.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.numpy()
            records.append([seq_id] + tensor.tolist())

        df = pd.DataFrame(records)
        df.columns = ["seq_id"] + [f"dim_{i}" for i in range(1, df.shape[1])]

        # 保存 parquet
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(out_path, index=False)

        with open(log_file, "a") as f:
            f.write(f"[{datetime.now()}] 成功: {pkl_path} -> {out_path}\n")

    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now()}] 失败: {pkl_path}, 错误: {str(e)}\n")


def batch_process_pkl(input_dir, out_dir, log_file="esm2_embedding.log"):
    """批量处理文件夹下的所有pkl文件"""
    os.makedirs(out_dir, exist_ok=True)
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]

    if not pkl_files:
        print("⚠️ 没有找到 .pkl 文件")
        return

    for pkl_file in pkl_files:
        pkl_path = os.path.join(input_dir, pkl_file)
        process_pkl_file(pkl_path, out_dir, log_file)


if __name__ == "__main__":
    # 输入pkl文件夹
    input_dir = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_proteinemb"
    # 输出parquet文件夹
    out_dir = "/home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_proteinemb_parquet"
    # 日志文件
    log_file = "esm2_embedding.log"

    batch_process_pkl(input_dir, out_dir, log_file)
    print("✅ 全部完成，详情见日志文件")
