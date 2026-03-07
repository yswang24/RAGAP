#!/usr/bin/env python3
import argparse
import numpy as np
import csv
import os


def load_matrix(path):
    """
    加载 compare 矩阵：
    - 如果是 NpzFile（numpy.savez 生成），读取第一个数组。
    - 如果直接是 ndarray（.npy 或者 np.load 直接返回 ndarray），直接返回。
    """
    data = np.load(path)
    if hasattr(data, 'files'):  # NpzFile：多个数组
        key = data.files[0]
        return data[key]
    else:  # ndarray
        return data


def strip_name(path):
    """从完整路径提取文件名并去掉扩展名"""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def main():
    parser = argparse.ArgumentParser(
        description="从 Sourmash 相似性矩阵提取高相似性序列对，并生成 CSV 和 NTW 文件"
    )
    parser.add_argument('--matrix',    required=True,
                        help="compare 矩阵文件路径（建议带 .npz）")
    parser.add_argument('--labels',    required=True,
                        help="标签文件路径（compare_matrix.npz.labels.txt）")
    parser.add_argument('--threshold', required=True, type=float,
                        help="保留相似度 ≥ 阈值的序列对")
    parser.add_argument('--output_csv',required=True,
                        help="输出 CSV 边列表路径")
    parser.add_argument('--output_ntw',required=True,
                        help="输出 NTW 文件路径（无后缀名信息，仅名字）")
    args = parser.parse_args()

    # 文件检查
    if not os.path.isfile(args.matrix):
        raise FileNotFoundError(f"找不到相似度矩阵文件：{args.matrix}")
    if not os.path.isfile(args.labels):
        raise FileNotFoundError(f"找不到标签文件：{args.labels}")

    # 读取标签
    with open(args.labels, 'r') as fh:
        labels = [line.strip() for line in fh]

    # 读取相似度矩阵
    sim_mat = load_matrix(args.matrix)
    n = len(labels)
    if sim_mat.shape != (n, n):
        raise ValueError(f"矩阵维度 {sim_mat.shape} 与标签数 {n} 不一致")

    # 打开输出文件
    with open(args.output_csv, 'w', newline='') as csvfile, \
         open(args.output_ntw, 'w', newline='') as ntwfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['source', 'target', 'weight'])

        # NTW 文件通常不需要 header
        # 遍历上三角矩阵
        for i in range(n):
            for j in range(i+1, n):
                sim = sim_mat[i, j]
                if sim >= args.threshold:
                    # 写 CSV
                    csv_writer.writerow([labels[i], labels[j], f"{sim:.6f}"])
                    # 写 NTW：只保留名字
                    name_i = strip_name(labels[i])
                    name_j = strip_name(labels[j])
                    ntwfile.write(f"{name_i},{name_j},{sim:.6f}\n")

    print(f"完成：CSV 输出 -> {args.output_csv}" )
    print(f"完成：NTW 输出 -> {args.output_ntw}")


if __name__ == '__main__':
    main()