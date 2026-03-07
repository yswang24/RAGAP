#!/usr/bin/env bash
set -euo pipefail

# ======== 激活 conda 环境 ========
source /home/wangjingyuan/anaconda3/etc/profile.d/conda.sh
conda activate sourmash_env
echo "✅ 当前 Conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"

#### 用户可调整的参数 ####
INPUT_DIR="/home/wangjingyuan/wys/phage_fasta_final"   # 参考序列目录
QUERY_DIR="/home/wangjingyuan/wys/build_new_phage/cherry_fasta"         # 新输入的FASTA文件夹
WORK_DIR="newphage-compare"                                  # 工作目录
KMER_SIZE=21
SCALED=1000
THRESHOLD=0.8
##############################

mkdir -p "${WORK_DIR}/signatures/input"
mkdir -p "${WORK_DIR}/signatures/query"
mkdir -p "${WORK_DIR}/compare"
mkdir -p "${WORK_DIR}/query_split"

echo "[1/4] 拆分 QUERY_DIR 中的多序列 fasta..."
find "${QUERY_DIR}" -name "*.fasta" | while read f; do
  base=$(basename "$f" .fasta)
  awk -v outdir="${WORK_DIR}/query_split" -v prefix="${base}" '
    /^>/ {
      # 如果已有序列，则写出上一条
      if (seq != "") {
        print header > (outdir "/" prefix "_" name ".fasta");
        print seq    >> (outdir "/" prefix "_" name ".fasta");
        close(outdir "/" prefix "_" name ".fasta");
      }
      # 提取新的序列名（去掉“>”）
      name = substr($1, 2);
      header = $0;
      seq = "";
      next;
    }
    {
      seq = seq $0;
    }
    END {
      # 写出最后一条序列
      if (seq != "") {
        print header > (outdir "/" prefix "_" name ".fasta");
        print seq    >> (outdir "/" prefix "_" name ".fasta");
        close(outdir "/" prefix "_" name ".fasta");
      }
    }
  ' "$f"
done

echo "[2/4] 生成 signatures..."
# INPUT_DIR 序列
find "${INPUT_DIR}" -name "*.fasta" | parallel --bar \
  "sourmash sketch dna -p k=${KMER_SIZE},scaled=${SCALED},abund \
   -o ${WORK_DIR}/signatures/input/{/.}.sig {}"

# QUERY_DIR 拆分后的序列
find "${WORK_DIR}/query_split" -name "*.fasta" | parallel --bar \
  "sourmash sketch dna -p k=${KMER_SIZE},scaled=${SCALED},abund \
   -o ${WORK_DIR}/signatures/query/{/.}.sig {}"

echo "[3/4] 仅比较 QUERY vs INPUT + QUERY vs QUERY ..."
# sourmash compare 需要指定一组 sig
# 我们使用 sourmash search 循环实现 pairwise 比较
python3 extract_cross_edges.py \
  --input_sigs "${WORK_DIR}/signatures/input" \
  --query_sigs "${WORK_DIR}/signatures/query" \
  --threshold ${THRESHOLD} \
  --output_tsv "${WORK_DIR}/new_phage_phage_edges.tsv"

echo "✅ 完成！结果输出到 ${WORK_DIR}/cross_similarity.tsv"
