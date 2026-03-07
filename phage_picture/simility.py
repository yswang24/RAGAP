import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time

# ================= 配置区域 =================
# 噬菌体 FASTA 文件夹路径
FASTA_DIR = "/home/wangjingyuan/wys/phage_fasta_final"

# 数据集划分文件路径
TRAIN_PATH = "/home/wangjingyuan/wys/phage_picture/final_leakage_free_dataset/pairs_train_cleaned.tsv"
VAL_PATH   = "/home/wangjingyuan/wys/phage_picture/final_leakage_free_dataset/pairs_val_cleaned.tsv"
TEST_PATH  = "/home/wangjingyuan/wys/phage_picture/final_leakage_free_dataset/pairs_test_cleaned.tsv"

# 参数设置
K_MER_SIZE = 21       # K-mer 长度，21是细菌/噬菌体比较的标准
SKETCH_SIZE = 1000    # 签名大小，越大越准，但越慢。1000对于一般分析足够
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2) # 并行核心数

# ===========================================

def read_fasta_and_sketch(args):
    """
    读取 FASTA 文件并计算 MinHash 签名
    """
    phage_id, filepath, k, s_size = args
    
    if not os.path.exists(filepath):
        return phage_id, None
    
    try:
        seq = ""
        with open(filepath, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    seq += line.strip().upper()
        
        if len(seq) < k:
            return phage_id, None

        # 生成 K-mers 的哈希值
        hashes = [hash(seq[i:i+k]) for i in range(len(seq) - k + 1)]
        
        # MinHash: 取最小的 s_size 个哈希值作为签名
        hashes = np.array(hashes, dtype=np.int64)
        np.random.seed(42) # 简单的置换模拟，这里直接排序取最小即可模拟MinHash逻辑
        hashes.sort()
        sketch = np.unique(hashes)[:s_size]
        
        return phage_id, sketch
    except Exception as e:
        # print(f"Error processing {phage_id}: {e}") # 生产环境避免过多打印
        return phage_id, None

def compute_jaccard_similarity(sketch1, sketch2):
    """
    计算两个签名的 Jaccard 相似度
    J(A,B) = |A n B| / |A u B|
    """
    if sketch1 is None or sketch2 is None:
        return 0.0
    
    intersection = np.intersect1d(sketch1, sketch2, assume_unique=True).size
    union = len(sketch1) + len(sketch2) - intersection
    
    if union == 0:
        return 0.0
    return intersection / union

def load_ids(path):
    df = pd.read_csv(path, sep='\t')
    # 假设第一列是 phage_id，或者列名为 'phage_id'
    if 'phage_id' in df.columns:
        # 使用 unique() 确保每个噬菌体只统计一次，即使它有多个宿主
        return df['phage_id'].unique().tolist()
    else:
        return df.iloc[:, 0].unique().tolist()

def main():
    print(f"🚀 Starting Sequence Leakage Check...")
    print(f"📂 Loading dataset IDs...")
    
    # 确保 load_ids 只获取唯一的噬菌体ID
    train_ids = load_ids(TRAIN_PATH)
    val_ids = load_ids(VAL_PATH)
    test_ids = load_ids(TEST_PATH)
    
    # 因为是基于簇划分的结果，这里的数量是相互作用对的数量，但 MinHash 是基于唯一的噬菌体序列
    # 进一步精简到唯一的噬菌体 ID
    unique_train_phages = list(set(train_ids))
    unique_val_phages = list(set(val_ids))
    unique_test_phages = list(set(test_ids))

    print(f"   - Train phages (unique): {len(unique_train_phages)}")
    print(f"   - Val phages (unique):   {len(unique_val_phages)}")
    print(f"   - Test phages (unique):  {len(unique_test_phages)}")

    # 准备文件路径任务
    all_unique_ids = set(unique_train_phages + unique_val_phages + unique_test_phages)
    tasks = []
    for pid in all_unique_ids:
        # 尝试几种可能的扩展名，或者直接用 .fasta
        fpath = os.path.join(FASTA_DIR, f"{pid}.fasta")
        tasks.append((pid, fpath, K_MER_SIZE, SKETCH_SIZE))

    print(f"🧬 Sketching {len(tasks)} sequences using {NUM_WORKERS} CPU cores (MinHash k={K_MER_SIZE})...")
    
    # 并行计算签名
    sketches = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(read_fasta_and_sketch, tasks), total=len(tasks), unit="seq"))
    
    for pid, sketch in results:
        if sketch is not None:
            sketches[pid] = sketch
            
    print("✅ Sketching complete.")

    # 准备计算矩阵
    # 将 Train 的 sketch 提取出来做成列表，方便遍历
    train_sketches = [sketches[pid] for pid in unique_train_phages if pid in sketches]
    
    if not train_sketches:
        print("❌ Error: No valid training sequences found.")
        return

    # 定义比较函数
    def get_max_similarity_to_train(query_ids, set_name):
        max_sims = []
        
        print(f"🔍 Comparing {set_name} against Training set...")
        
        # 确保只比较唯一的噬菌体ID
        unique_query_ids = list(set(query_ids)) 

        for q_id in tqdm(unique_query_ids, desc=f"Checking {set_name}"):
            if q_id not in sketches:
                continue
            
            q_sketch = sketches[q_id]
            current_max = 0.0
            
            # 迭代训练集的签名进行比对
            for t_sketch in train_sketches:
                sim = compute_jaccard_similarity(q_sketch, t_sketch)
                if sim > current_max:
                    current_max = sim
                if current_max >= 1.0: # 找到完全一样的，可以提前停止
                    break
            
            max_sims.append(current_max)
        return max_sims

    # 计算 Test vs Train 和 Val vs Train
    val_max_sims = get_max_similarity_to_train(val_ids, "Validation Set")
    test_max_sims = get_max_similarity_to_train(test_ids, "Test Set")
# ... (之前的代码保持不变) ...

    # ================= 绘图 (Publication Style - MODIFIED) =================
    print("🎨 Generating Plot (Modified Style)...")
    
    # 设置风格
    sns.set_theme(style="ticks", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    
    # 绘制 KDE 图 (核密度估计)
    sns.kdeplot(val_max_sims, fill=True, label='Validation Set', color='#1f77b4', alpha=0.5, linewidth=2)
    sns.kdeplot(test_max_sims, fill=True, label='Test Set', color='#ff7f0e', alpha=0.5, linewidth=2)
    
    # 添加 95% 阈值线 (通常视为同种/同源的界限)
    plt.axvline(x=0.95, color='black', linestyle='--', alpha=0.7)
    
    # 调整文本位置
    max_density = plt.ylim()[1]
    # 保持 95% 阈值文本，用于标记
    plt.text(0.96, max_density * 0.75, '95% Similarity Threshold', color='black', fontsize=1)

    # **MODIFICATION: 将图例放置在 95% 阈值线的正上方**
    plt.legend(
        title=None,
        # 使用 'lower center' 或 'center' 让图例的中心对齐到锚点
        loc='lower center',      
        # 将锚点设置在 x=0.95，y=1.05 (位于图表上方)
        bbox_to_anchor=(0.95, 1.03), 
        ncol=1,                 # 两列横向排列
        frameon=False,           # 保持图例边框
        fontsize=12,
        title_fontsize=12
    )
    
    # 美化轴标签
    plt.xlabel('Max MinHash Jaccard Similarity with Training Samples', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xlim(0, 1.0)
    
    # 避免 x 轴刻度线太多
    plt.xticks(np.arange(0, 1.1, 0.1))
    
    sns.despine()

    # 保存图片
    output_file = "dataset_leakage_check_clustered_final_aligned.svg"
    plt.savefig(output_file, format='svg', bbox_inches='tight', dpi=600)
    print(f"💾 Plot saved to: {os.path.abspath(output_file)}")
    
    # ... (后续的统计数据输出保持不变) ...    
    # 输出统计数据
    print("\n📊 --- Statistical Summary ---")
    for name, data in [("Validation", val_max_sims), ("Test", test_max_sims)]:
        data = np.array(data)
        leakage_count = np.sum(data > 0.95)
        print(f"dataset: {name} ({len(data)} unique phages)")
        print(f"  - Mean Max Similarity: {np.mean(data):.4f}")
        print(f"  - Median Max Similarity: {np.median(data):.4f}")
        print(f"  - Samples > 95% Sim (High Risk): {leakage_count} ({leakage_count/len(data)*100:.2f}%)")

if __name__ == "__main__":
    main()