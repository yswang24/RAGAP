#!/bin/bash

# 输入文件
input_file="complete_genome_archaea.txt"

# 输出文件
output_file="download_links_archaea.txt"

# 遍历每一行并生成对应的下载链接
while read -r line; do
    # 提取最后一部分文件夹名称
    folder_name=$(basename "$line")
    # 拼接成完整的下载链接
    echo "${line}/${folder_name}_genomic.fna.gz" >> "$output_file"
done < "$input_file"

echo "下载链接已生成，保存在 ${output_file} 中。"
