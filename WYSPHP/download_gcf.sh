#!/bin/bash
# 批量下载 missing_gcf.txt 里的 GCF 数据

# 输入文件
GCF_LIST="missing_gcf.txt"

# 输出目录
OUTDIR="genomes"
FASTA_DIR="fasta_only"

mkdir -p "$OUTDIR" "$FASTA_DIR"

# 循环读取 missing_gcf.txt
while read gcf; do
  # 跳过空行
  if [[ -z "$gcf" ]]; then
    continue
  fi

  echo "==== 正在下载 $gcf ===="

  # 下载并保存为 zip
  datasets download genome accession "$gcf" --include genome --filename "${gcf}.zip"

  # 检查是否下载成功
  if [[ -s "${gcf}.zip" ]]; then
    # 解压到 genomes 目录
    unzip -o "${gcf}.zip" -d "$OUTDIR/" >/dev/null
    rm "${gcf}.zip"

    # 提取 fasta 文件到 fasta_only/
    find "$OUTDIR" -name "${gcf}*genomic.fna" -exec cp {} "$FASTA_DIR/" \;
  else
    echo "⚠️  $gcf 下载失败" >> failed_gcf.txt
  fi

done < "$GCF_LIST"

# 生成下载成功清单
ls "$FASTA_DIR" | sed 's/\.fna$//' > downloaded_gcf.txt

echo "✅ 批量下载完成"
echo "成功列表在 downloaded_gcf.txt"
echo "失败列表在 failed_gcf.txt（如有）"
