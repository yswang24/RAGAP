import os
import shutil

def copy_fasta_from_list(txt_file, source_dir, target_dir):
    # 读取txt文件中的GCF编号
    with open(txt_file, "r") as f:
        gcf_ids = [line.strip() for line in f if line.strip()]
    
    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)
    
    not_found = []  # 用于记录没找到的编号
    
    for gcf in gcf_ids:
        fasta_name = f"{gcf}.fasta"
        source_path = os.path.join(source_dir, fasta_name)
        target_path = os.path.join(target_dir, fasta_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)  # 保留文件属性
            print(f"复制成功: {fasta_name}")
        else:
            not_found.append(gcf)
            print(f"未找到: {fasta_name}")
    
    # 如果有未找到的文件，输出到一个txt
    if not_found:
        not_found_file = os.path.join(target_dir, "not_found.txt")
        with open(not_found_file, "w") as f:
            f.write("\n".join(not_found))
        print(f"有 {len(not_found)} 个文件未找到，已保存到 {not_found_file}")


if __name__ == "__main__":
    txt_file = "/home/wangjingyuan/wys/WYSPHP/in_virus_clusters_gcf1.txt"              # 你的txt文件路径
    source_dir = "/home/wangjingyuan/wys/host_fasta"           # 存放fasta的文件夹
    target_dir = "/home/wangjingyuan/wys/host_fasta_new"      # 输出文件夹
    
    copy_fasta_from_list(txt_file, source_dir, target_dir)
