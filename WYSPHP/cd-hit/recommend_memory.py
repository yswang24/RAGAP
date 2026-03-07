import psutil

def recommend_cd_hit_memory(usage_ratio=0.8):
    """
    自动检测系统总内存，并计算 cd-hit 的 -M 参数（单位 MB）
    usage_ratio: 使用比例，默认 0.8 表示使用 80% 的总内存
    """
    total_mem = psutil.virtual_memory().total / (1024 ** 2)  # 转成 MB
    recommended = int(total_mem * usage_ratio)
    print(f"系统总内存: {total_mem:.0f} MB")
    print(f"建议 cd-hit -M 参数（{usage_ratio*100:.0f}% 使用率）: {recommended} MB")
    return recommended

if __name__ == "__main__":
    recommend_cd_hit_memory()
