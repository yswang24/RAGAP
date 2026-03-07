##提取聚类之后的val fasta
import pandas as pd
import os
import shutil

def extract_phage_fastas(tsv_path, source_dir, output_dir, extensions=['.fasta', '.fna', '.fa']):
    """
    根据TSV文件中的phage_id，从源目录提取对应的fasta文件到输出目录。
    
    Args:
        tsv_path: TSV文件的路径
        source_dir: 存放原始fasta文件的文件夹路径
        output_dir: 提取出的文件存放路径
        extensions: 可能的文件后缀名列表 (如果你的文件没有后缀，可设为 [''])
    """
    
    # 1. 创建输出目录 (如果不存在)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 2. 读取 TSV 文件
    try:
        # sep='\t' 指定制表符分隔
        df = pd.read_csv(tsv_path, sep='\t')
        
        # 确保去除列名和ID可能存在的空白字符
        df.columns = df.columns.str.strip()
        phage_ids = df['phage_id'].astype(str).str.strip().unique() # 使用unique去重
        
        print(f"共找到 {len(phage_ids)} 个唯一的 Phage ID。开始提取...")
        
    except Exception as e:
        print(f"读取TSV文件失败: {e}")
        return

    # 3. 遍历并复制文件
    success_count = 0
    missing_list = []

    for pid in phage_ids:
        found = False
        
        # 尝试匹配不同的后缀名
        for ext in extensions:
            # 假设文件名就是 ID + 后缀 (例如 MH791407.fasta)
            filename = f"{pid}{ext}"
            src_file = os.path.join(source_dir, filename)
            
            if os.path.exists(src_file):
                dst_file = os.path.join(output_dir, filename)
                
                # --- 核心操作：复制文件 ---
                shutil.copy2(src_file, dst_file)
                # -----------------------
                
                success_count += 1
                found = True
                break # 找到一种后缀后就停止尝试其他后缀
        
        if not found:
            missing_list.append(pid)

    # 4. 输出结果统计
    print("-" * 30)
    print(f"处理完成。")
    print(f"成功提取: {success_count} 个")
    print(f"未找到文件: {len(missing_list)} 个")
    
    if missing_list:
        # 将未找到的ID保存到日志，方便检查
        log_path = os.path.join(output_dir, "missing_ids.log")
        with open(log_path, 'w') as f:
            for mid in missing_list:
                f.write(f"{mid}\n")
        print(f"未找到的ID列表已保存至: {log_path}")

# ================= 配置区域 =================

# 你的 TSV 文件路径 (请修改这里)
my_tsv_path = '/home/wangjingyuan/wys/phage_picture/final_leakage_free_dataset/pairs_val_cleaned.tsv' 

# 你的源 FASTA 文件夹路径
my_source_dir = '/home/wangjingyuan/wys/phage_fasta_final'

# 你希望文件提取到哪里 (脚本会自动创建这个文件夹)
my_output_dir = './extracted_phages'

# 你的fasta文件后缀是什么？(按优先级排列)
# 如果文件名里没有后缀，就写 my_extensions = ['']
my_extensions = ['.fasta', '.fna', '.fa']

# ================= 运行 =================
if __name__ == '__main__':
    extract_phage_fastas(my_tsv_path, my_source_dir, my_output_dir, my_extensions)