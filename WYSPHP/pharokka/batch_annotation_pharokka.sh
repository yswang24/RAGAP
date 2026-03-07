# #!/bin/bash

# #########pharokka_env环境下
# set -euo pipefail

# THREADS=64
# PHAGE_FASTA_DIR="/home/wangjingyuan/wys/WYSPHP/test_phage_fasta/"
# HOST_FASTA_DIR="/home/wangjingyuan/wys/WYSPHP/test_host_fasta/"
# OUT_DIR="annotation_out"
# PHAGE_OUT="$OUT_DIR/phage/"
# HOST_OUT="$OUT_DIR/host/"
# MAPPING_JSON="$OUT_DIR/protein_to_source.json"

# mkdir -p "$PHAGE_OUT" "$HOST_OUT"

# echo "## 注释噬菌体基因组（Pharokka）"
# for f in "$PHAGE_FASTA_DIR"/*.fasta; do
#   base=$(basename "$f" .fasta)
#   outdir="$PHAGE_OUT/${base}"
#   if [ -d "$outdir" ]; then
#     echo "跳过已有输出：$outdir"
#   else
#     pharokka.py -i "$f" \
#       -o "$outdir" \
#       -d /home/wangjingyuan/wys/pharokka_db/pharokka_v1.4.0_databases \
#       -t "$THREADS" \
#       -g phanotate \
     
      
#   fi
# done

# echo "## 注释宿主基因组（Prodigal 翻译）"
# for f in "$HOST_FASTA_DIR"/*.fasta; do
#   base=$(basename "$f" .fasta)
#   faa="$HOST_OUT/${base}.faa"
#   if [ -f "$faa" ]; then
#     echo "跳过已有蛋白 FASTA：$faa"
#   else
#     prodigal -i "$f" \
#       -a "$faa" \
#       -d /dev/null \
#       -o /dev/null
#   fi
# done

# echo "## 调用 Python 脚本生成 protein_to_source.json"
# python3 generate_mapping.py \
#   --phage-dir "$PHAGE_OUT" \
#   --host-dir "$HOST_OUT" \
#   --out-json "$MAPPING_JSON"

# echo "完成：蛋白 FASTA 和来源映射已生成 → 接下来可运行结构图构建脚本"




#!/bin/bash

######### pharokka_env环境下
set -euo pipefail

THREADS=64
PHAGE_FASTA_DIR="/home/wangjingyuan/wys/phage_fasta"
HOST_FASTA_DIR="/home/wangjingyuan/wys/host_fasta_final"
OUT_DIR="annotation_out"
PHAGE_OUT="$OUT_DIR/phage/"
HOST_OUT="$OUT_DIR/host_final/"
MAPPING_JSON="$OUT_DIR/protein_to_source.json"
LOGFILE="annotation.log"
LOGFILE0="annotation0.log"
LOGFILE1="annotation1.log"


mkdir -p "$PHAGE_OUT" "$HOST_OUT"


echo "## 使用 Phanotate 批量生成噬菌体 .faa 文件"
for f in "$PHAGE_FASTA_DIR"/*.fasta; do
  base=$(basename "$f" .fasta)
  faa="$PHAGE_OUT/${base}.faa"
  if [ -f "$faa" ]; then
    echo "已存在 .faa，跳过：$faa" >> "$LOGFILE" 2>&1
  else
    echo "预测蛋白：$base" >> "$LOGFILE" 2>&1
    phanotate.py "$f" -f faa > "$faa"
    if [ $? -ne 0 ] || [ ! -s "$faa" ]; then
      echo "Error：Phanotate 失败或输出为空 —— $base" >> "$LOGFILE" 2>&1
    fi
  fi
done


echo "## 使用 Prodigal 批量生成宿主 .faa 文件" >> "$LOGFILE1" 2>&1
for f in "$HOST_FASTA_DIR"/*.fasta; do
  base=$(basename "$f" .fasta)
  faa="$HOST_OUT/${base}.faa"

  if [ -f "$faa" ]; then
    echo "已存在 .faa，跳过：$faa" >> "$LOGFILE1" 2>&1
  else
    echo "预测蛋白：$base" >> "$LOGFILE1" 2>&1
    prodigal -i "$f" -a "$faa" -d /dev/null -o /dev/null > /dev/null 2>&1

    if [ $? -ne 0 ] || [ ! -s "$faa" ]; then
      echo "Error：Prodigal 失败或输出为空 —— $base" >> "$LOGFILE1" 2>&1
      rm -f "$faa"
    fi
  fi
done


# echo "## 调用 Python 脚本生成 protein_to_source.json"
# python3 generate_mapping.py \
#   --phage-dir "$PHAGE_OUT" \
#   --host-dir "$HOST_OUT" \
#   --out-json "$MAPPING_JSON"

# echo "完成：蛋白 FASTA 和来源映射已生成 → 接下来可运行结构图构建脚本"
