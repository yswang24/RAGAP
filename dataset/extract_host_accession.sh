#!/bin/bash

# 定义两个文件夹路径
folder1="archaea_complete_genomes"
folder2="bacteria_complete_genomes"

# 输出文件
output_file="host_version_accesssion.txt"

# 清空输出文件（如果已存在）
> "$output_file"

# 遍历文件夹 1 和文件夹 2，提取文件名的第二个下划线前的部分
for folder in "$folder1" "$folder2"; do
    for file in "$folder"/*; do
        # 检查文件是否存在
        [ -f "$file" ] || continue
        # 提取文件名
        filename=$(basename "$file")
        # 提取第二个下划线之前的部分
        extracted_name=$(echo "$filename" | awk -F'_' '{print $1"_"$2}')
        # 写入到输出文件
        echo "$extracted_name" >> "$output_file"
    done
done

# 去重排序（可选）
sort -u "$output_file" -o "$output_file"

echo "提取完成，结果保存在 $output_file 中。"


