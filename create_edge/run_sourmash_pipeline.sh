#!/usr/bin/env bash
set -euo pipefail

# ======== 激活 conda 环境 ========
source /home/wangjingyuan/anaconda3/etc/profile.d/conda.sh
conda activate sourmash_env
echo "✅ 当前 Conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"

#### 用户可调整的参数 ####
INPUT_DIR="/home/wangjingyuan/wys/host_fasta_final_sequence"         # 输入：存放所有 .fasta 文件的文件夹
WORK_DIR="work-21-HH-sequence"                # 中间结果输出目录
KMER_SIZE=31                  # k-mer 大小
SCALED=1000                    # scaled 参数（压缩比）
THRESHOLD=0.8                  # 提取边时的相似性阈值
##############################

# 1. 创建工作目录
mkdir -p "${WORK_DIR}/signatures"
mkdir -p "${WORK_DIR}/compare"

# # 2. 批量生成 .sig 文件
# echo "[1/3] Sketching all FASTA files into Sourmash signatures..."
# find "${INPUT_DIR}" -name "*.fasta" | \
#   parallel --bar \
#     "sourmash sketch dna -p k=${KMER_SIZE},scaled=${SCALED},abund \
#       -o ${WORK_DIR}/signatures/{/.}.sig {}"

# 3. 两两比较，输出矩阵和标签
echo "[2/3] Comparing all signatures (this may take a while)..."
sourmash compare \
  -k ${KMER_SIZE} \
  ${WORK_DIR}/signatures/*.sig \
  -o ${WORK_DIR}/compare/compare_matrix.npz


# # 4. 可选：生成热图（PNG）
# echo "[3/3] Plotting similarity heatmap..."
# sourmash plot \
#   ${WORK_DIR}/compare/compare_matrix.npz \


# 5. 提取高相似性对，调用 Python 脚本
echo "[4/4] Extracting edges with similarity ≥ ${THRESHOLD}..."
python3 extract_edges.py \
  --matrix "${WORK_DIR}/compare/compare_matrix.npz" \
  --labels "${WORK_DIR}/compare/compare_matrix.npz.labels.txt" \
  --threshold ${THRESHOLD} \
  --output_csv "${WORK_DIR}/graph_edges.csv" \
  --output_ntw "${WORK_DIR}/HH.ntw"

  

echo "Pipeline complete! 边列表保存在 ${WORK_DIR}/graph_edges.csv"
