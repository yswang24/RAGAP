import os
import shutil
import re

def process_folders(main_folder, output_folder="virus"):
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 统计处理结果
    processed_count = 0
    skipped_folders = []
    
    # 遍历主文件夹中的所有子文件夹
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)
        
        # 只处理目录且符合命名模式 (07_checkv_XXX)
        if not os.path.isdir(folder_path):
            continue
            
        # 检查文件夹名称格式
        match = re.match(r"07_checkv_(\w+)$", folder_name)
        if not match:
            skipped_folders.append(folder_name)
            continue
            
        # 提取标识符 (如 A1C, AU2F)
        identifier = match.group(1)
        
        # 原始文件路径
        src_file = os.path.join(folder_path, "combined.fna")
        
        # 检查文件是否存在
        if not os.path.exists(src_file):
            skipped_folders.append(folder_name)
            continue
            
        # 新文件名 (如 A1C.fna)
        new_file_name = f"{identifier}.fna"
        
        # 重命名操作
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(src_file, new_file_path)
        
        # 复制到汇总文件夹
        dest_file = os.path.join(output_folder, new_file_name)
        shutil.copy2(new_file_path, dest_file)
        
        processed_count += 1
        print(f"处理成功: {folder_name} -> {new_file_name}")
    
    # 输出处理报告
    print("\n===== 处理完成 =====")
    print(f"成功处理文件夹: {processed_count}个")
    print(f"跳过文件夹: {len(skipped_folders)}个")
    
    if skipped_folders:
        print("\n跳过的文件夹列表:")
        for folder in skipped_folders:
            print(f" - {folder}")

if __name__ == "__main__":
    # 配置参数
    MAIN_FOLDER = "/home/wangjingyuan/wys/soil/virus"       # 包含所有子文件夹的主文件夹
    OUTPUT_FOLDER = "soilvirus"    # 输出汇总文件夹
    
    # 执行处理
    process_folders(MAIN_FOLDER, OUTPUT_FOLDER)