import os
from concurrent.futures import ThreadPoolExecutor

def read_file(file_path):
    """读取单个文件的内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            return infile.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def process_folder(folder_path):
    """处理一个文件夹中的所有文件"""
    content_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)  # 使用 file_name 而不是 file_path
        if os.path.isfile(file_path):
            content = read_file(file_path)
            content_list.append((file_name, content))
    return content_list

def merge_folders_content(folder1, folder2, output_file):
    # 使用线程池并发读取两个文件夹中的文件
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(process_folder, folder1)
        future2 = executor.submit(process_folder, folder2)

        results1 = future1.result()
        results2 = future2.result()

    # 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name, content in results1 + results2:
            outfile.write(f"\n-- {file_name} --\n")
            outfile.write(content)




# 使用函数，传入你的文件夹路径和输出文件路径
folder1 = "archaea_complete_genomes"
folder2 = "bacteria_complete_genomes"
output_file = "prokaryote"

merge_folders_content(folder1, folder2, output_file)