#输出lineage最好用
'''
import subprocess

# 输入和输出文件
input_file = "archaea_taxid.txt"  # 包含 taxid 的输入文件
output_file = "archaea_lineage.csv"  # 输出的 CSV 文件

# 直接调用 taxonkit 和 csvtk
def process_taxonkit(input_file, output_file):
    try:
        # 构造流水线命令
        command = (
            f"cat {input_file} | "
            "taxonkit lineage | "
            "taxonkit reformat -f \"{k}\\t{p}\\t{c}\\t{o}\\t{f}\\t{g}\\t{s}\" -F -P | "
            "csvtk cut -t -f -2 | "
            f"csvtk add-header -t -n taxid,kindom,phylum,class,order,family,genus,species > {output_file}"
        )
        # 运行命令
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode().strip()}")

        print(f"Processing complete. Output saved to {output_file}")

    except Exception as e:
        print(f"Error processing taxonkit: {e}")

# 主函数
if __name__ == "__main__":
    process_taxonkit(input_file, output_file)
'''
#改进  不带前缀
'''
import subprocess

# 输入和输出文件
input_file = "bacteria_taxid.txt"  # 包含 taxid 的输入文件
output_file = "bacteria_lineage.csv"  # 输出的 CSV 文件

# 调用 taxonkit 并移除前缀
def process_taxonkit(input_file, output_file):
    try:
        # 构造流水线命令
        command = (
            f"cat {input_file} | "
            "taxonkit lineage | "
            "taxonkit reformat -f \"{k}\\t{p}\\t{c}\\t{o}\\t{f}\\t{g}\\t{s}\" -F -P | "
            "sed 's/[a-z]__//g' | "  # 使用 sed 删除前缀
            "csvtk cut -t -f -2 | "
            f"csvtk add-header -t -n taxid,kindom,phylum,class,order,family,genus,species > {output_file}"
        )
        # 运行命令
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode().strip()}")

        print(f"Processing complete. Output saved to {output_file}")

    except Exception as e:
        print(f"Error processing taxonkit: {e}")

# 主函数
if __name__ == "__main__":
    process_taxonkit(input_file, output_file)
'''
# #保持输出行数一致
# import subprocess

# # 输入和输出文件
# input_file = "bacteria_taxid.txt"  # 包含 taxid 的输入文件
# output_file = "bacteria_lineage.csv"  # 输出的 CSV 文件

# # 调用 taxonkit 并移除前缀
# def process_taxonkit(input_file, output_file):
#     try:
#         # 构造流水线命令
#         command = (
#             f"cat {input_file} | "
#             "taxonkit lineage | "
#             "taxonkit reformat -f \"{k}\\t{p}\\t{c}\\t{o}\\t{f}\\t{g}\\t{s}\" -F -P | "
#             "sed 's/[a-z]__//g' | "  # 使用 sed 删除前缀
#             "csvtk cut -t -f -2 | "
#             f"csvtk add-header -t -n taxid,kindom,phylum,class,order,family,genus,species | "
#             "awk 'BEGIN {OFS=\",\"} {if (NF < 8) {print $0, \"missing\"} else {print $0}}' > '{output_file}'"
#         )
#         # 运行命令
#         process = subprocess.Popen(
#             ["bash", "-c", command],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#         stdout, stderr = process.communicate()

#         if process.returncode != 0:
#             raise RuntimeError(f"Command failed: {stderr.decode().strip()}")

#         print(f"Processing complete. Output saved to {output_file}")

#     except Exception as e:
#         print(f"Error processing taxonkit: {e}")

# # 主函数
# if __name__ == "__main__":
#     process_taxonkit(input_file, output_file)


import subprocess

# 输入和输出文件
input_file = "host_taxid.txt"  # 包含 taxid 的输入文件
output_file = "host_lineage_new.tsv"  # 输出的 TSV 文件

# 调用 taxonkit 并移除前缀
def process_taxonkit(input_file, output_file):
    try:
        # 构造流水线命令
        command = (
            f"cat {input_file} | "
            "taxonkit lineage | "
            "taxonkit reformat -f \"{k}\\t{p}\\t{c}\\t{o}\\t{f}\\t{g}\\t{s}\" -F -P | "
            "sed 's/[a-z]__//g' | "  # 使用 sed 删除前缀
            "csvtk cut -t -f -2 | "
            f"csvtk add-header -t -n taxid,kindom,phylum,class,order,family,genus,species | "
            "awk 'BEGIN {OFS=\"\\t\"} {if ($1 == \"\") {print \"\"} else {print $0}}' > {output_file}"
        )
        # 运行命令
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode().strip()}")

        print(f"Processing complete. Output saved to {output_file}")

    except Exception as e:
        print(f"Error processing taxonkit: {e}")

# 主函数
if __name__ == "__main__":
    process_taxonkit(input_file, output_file)

# import subprocess

# # 输入和输出文件
# input_file = "bacteria_taxid.txt"  # 包含 taxid 的输入文件
# output_file = "bacteria_lineage_new.txt"  # 输出的 TXT 文件

# # 调用 taxonkit 并移除前缀
# def process_taxonkit(input_file, output_file):
#     try:
#         # 构造流水线命令
#         command = (
#             f"cat {input_file} | "
#             "taxonkit lineage | "
#             "taxonkit reformat -f \"{k}\\t{p}\\t{c}\\t{o}\\t{f}\\t{g}\\t{s}\" -F -P | "
#             "sed 's/[a-z]__//g' | "  # 使用 sed 删除前缀
#             "csvtk cut -t -f -2 | "
#             f"csvtk add-header -t -n taxid,kindom,phylum,class,order,family,genus,species | "
#             "awk 'BEGIN {OFS=\"\\t\"} {if ($1 == \"\") {print \"\"} else {print $0}}' > '{output_file}'"
#         )
#         # 运行命令
#         process = subprocess.Popen(
#             ["bash", "-c", command],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#         stdout, stderr = process.communicate()

#         if process.returncode != 0:
#             raise RuntimeError(f"Command failed: {stderr.decode().strip()}")

#         print(f"Processing complete. Output saved to {output_file}")

#     except Exception as e:
#         print(f"Error processing taxonkit: {e}")

# # 主函数
# if __name__ == "__main__":
#     process_taxonkit(input_file, output_file)
