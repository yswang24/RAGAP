import os

def merge_fasta_files(input_dir, output_file):
    """
    将目录中的所有 .fasta 或 .fa 文件合并为一个文件。
    
    参数:
        input_dir (str): 包含 fasta 文件的文件夹路径
        output_file (str): 合并后输出的文件名
    """
    # 支持的扩展名
    extensions = ('.fasta', '.fa')
    
    with open(output_file, 'w') as outfile:
        for filename in sorted(os.listdir(input_dir)):
            filepath = os.path.join(input_dir, filename)
            # 跳过子目录，只处理文件
            if os.path.isfile(filepath) and filename.lower().endswith(extensions):
                print(f"正在处理: {filename}")
                with open(filepath, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                # 可选：在每个文件末尾加一个换行，避免粘连
                outfile.write("\n")
    print(f"✅ 所有文件已合并到: {output_file}")

# =========================
# === 使用示例 ===
# =========================

if __name__ == "__main__":
    input_directory = "/home/wangjingyuan/wys/phage_test_data_613"   # 修改为你的fasta文件夹路径
    output_filename = "all_test.fasta"    # 输出文件名

    merge_fasta_files(input_directory, output_filename)