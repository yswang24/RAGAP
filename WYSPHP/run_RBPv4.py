# #!/usr/bin/env python3
# import os
# import subprocess
# from multiprocessing import Pool, cpu_count, Manager

# # === 配置区 ===
# INPUT_DIR = "/home/wangjingyuan/wys/WYSPHP/annotation_out/host_final"
# OUTPUT_DIR = "/home/wangjingyuan/wys/WYSPHP/annotation_out/PhageRBPdetect_v4output_host"
# SCRIPT = "/home/wangjingyuan/wys/WYSPHP/PhageRBPdetect_v4_inference.py"
# NUM_PROCESSES = max(1, cpu_count() - 1)  # 使用CPU核心数-1，保留一个核心防止卡死

# # === 准备输出文件夹 ===
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # === 收集输入文件 ===
# faa_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".faa")]
# faa_files.sort()
# total = len(faa_files)
# if total == 0:
#     print(f"❌ 没有找到任何 .faa 文件，请检查路径: {INPUT_DIR}")
#     exit(1)

# print(f"📂 共检测到 {total} 个faa文件，使用 {NUM_PROCESSES} 个进程并行预测...\n")

# # === 单样本处理函数 ===
# def process_sample(fname):
#     sample_name = os.path.splitext(fname)[0]
#     input_path = os.path.join(INPUT_DIR, fname)
#     output_path = os.path.join(OUTPUT_DIR, f"{sample_name}_predictions.csv")

#     if os.path.exists(output_path):
#         return f"✅ 跳过 {sample_name} (已有结果)"

#     cmd = [
#         "python", SCRIPT,
#         "--input", input_path,
#         "--output", output_path
#     ]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         return f"✅ 完成 {sample_name}"
#     except subprocess.CalledProcessError as e:
#         return f"❌ 失败 {sample_name}: {e}"

# # === 多进程执行 ===
# if __name__ == "__main__":
#     with Pool(processes=NUM_PROCESSES) as pool:
#         for idx, result in enumerate(pool.imap_unordered(process_sample, faa_files), 1):
#             print(f"[{idx}/{total}] {result}")
#     print("\n🎯 所有样本处理完成！")

####多卡gpu

#!/usr/bin/env python3
import os
import subprocess
from multiprocessing import Pool, cpu_count
import torch  # 用于检测GPU数量

# === 配置区 ===
INPUT_DIR = "/home/wangjingyuan/wys/WYSPHP/annotation_out/host_final"
OUTPUT_DIR = "/home/wangjingyuan/wys/WYSPHP/annotation_out/PhageRBPdetect_v4output_host"
SCRIPT = "/home/wangjingyuan/wys/WYSPHP/PhageRBPdetect_v4_inference.py"
NUM_PROCESSES = max(1, cpu_count() - 1)  # 使用CPU核心数-1，保留一个核心防止卡死

# === 准备输出文件夹 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 收集输入文件 ===
faa_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".faa")]
faa_files.sort()
total = len(faa_files)
if total == 0:
    print(f"❌ 没有找到任何 .faa 文件，请检查路径: {INPUT_DIR}")
    exit(1)

# === 检测GPU数量 ===
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    print(f"🖥️ 检测到 {GPU_COUNT} 块可用GPU，将自动分配任务")
else:
    GPU_COUNT = 0
    print("⚠️ 没有检测到可用GPU，将使用CPU运行，速度可能会较慢")

print(f"📂 共检测到 {total} 个faa文件，使用 {NUM_PROCESSES} 个进程并行预测...\n")

# === 单样本处理函数 ===
def process_sample(params):
    fname, gpu_id = params
    sample_name = os.path.splitext(fname)[0]
    input_path = os.path.join(INPUT_DIR, fname)
    output_path = os.path.join(OUTPUT_DIR, f"{sample_name}_predictions.csv")

    if os.path.exists(output_path):
        return f"✅ 跳过 {sample_name} (已有结果)"

    cmd = [
        "python", SCRIPT,
        "--input", input_path,
        "--output", output_path
    ]
    if GPU_COUNT > 0:
        cmd += ["--gpu", str(gpu_id)]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # 检查stdout里是否有cuda提示
        used_device = "cuda" if "using cuda" in result.stdout else "cpu"
        return f"✅ 完成 {sample_name} (使用 {used_device}{gpu_id if used_device=='cuda' else ''})"
    except subprocess.CalledProcessError as e:
        return f"""
❌ 失败 {sample_name}
命令: {' '.join(cmd)}
退出码: {e.returncode}
stdout:
{e.stdout.strip()}

stderr:
{e.stderr.strip()}
"""

# === 多进程执行 ===
if __name__ == "__main__":
    # 分配任务，循环分配GPU
    task_list = [(fname, (i % GPU_COUNT) if GPU_COUNT > 0 else None) for i, fname in enumerate(faa_files)]

    with Pool(processes=NUM_PROCESSES) as pool:
        for idx, result in enumerate(pool.imap_unordered(process_sample, task_list), 1):
            print(f"[{idx}/{total}] {result}")

    print("\n🎯 所有样本处理完成！")
