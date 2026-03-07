import pandas as pd
import os
import sys

# ================= 配置区域 =================
# 您的 TSV 文件路径
DATA_PATH = "phage_host.tsv" 
# 定义需要统计的分类学级别 (从高到低)
TAX_LEVELS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

# ================= 核心统计函数 =================
def analyze_taxonomy_statistics():
    """
    加载数据，计算每个分类级别的唯一计数和丰度百分比。
    """
    print("🚀 正在加载和分析数据集...")
    
    try:
        # 1. 加载数据。TSV 文件使用制表符 '\t' 作为分隔符。
        # 注意: 假设数据中没有多余的空白行或格式错误。
        df_full = pd.read_csv(DATA_PATH, sep='\t')
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {DATA_PATH}。请检查文件路径和名称是否正确。")
        return
    except pd.errors.ParserError:
        print("❌ 错误: 文件解析失败。请确认分隔符是否为制表符（\\t）。")
        return

    # 检查必需的分类学列是否都存在
    missing_cols = [col for col in TAX_LEVELS if col not in df_full.columns]
    if missing_cols:
        print(f"❌ 致命错误: 数据集中缺少必需的分类学列：{missing_cols}。请检查文件头。")
        return

    # 2. 数据清洗和初始化
    # 填充缺失值 (如果存在)，并确保它们不会被计为独特的分类单元
    df_full[TAX_LEVELS] = df_full[TAX_LEVELS].fillna('Unclassified/NA')
    
    # 总相互作用对数 (即总行数)
    total_interactions = len(df_full)
    print(f"\n================ 数据集概览 ================")
    print(f"总相互作用对数: {total_interactions:,}")
    print(f"唯一噬菌体总数: {df_full['virus_taxid'].nunique():,}")
    print(f"唯一宿主物种总数: {df_full['host_taxid'].nunique():,}")
    print("==========================================\n")

    
    # 3. 逐级统计
    results = {}
    
    for level in TAX_LEVELS:
        # 统计每个分类单元的出现次数 (即相互作用对的数量)
        counts = df_full[level].value_counts()
        
        # 计算百分比
        percentages = (counts / total_interactions) * 100
        
        # 结果合并成一个 DataFrame
        df_stats = pd.DataFrame({
            'Interaction_Count': counts,
            'Percentage': percentages.round(2)
        })
        
        # 排序并存储结果
        df_stats = df_stats.sort_values(by='Interaction_Count', ascending=False)
        results[level] = df_stats
        
        # 4. 打印结果 (为论文描述准备)
        unique_units = df_stats.shape[0]
        
        print(f"--- 分类级别: {level.upper()} ---")
        print(f"唯一分类单元总数: {unique_units:,}")
        
        # 仅显示 Top 5 (或 Top 3) 以便在论文中总结
        top_n = min(5, unique_units) 
        
        print(f"Top {top_n} 丰度最高的分类单元:")
        print(df_stats.head(top_n).to_string())
        
        # 统计 Top N 覆盖率
        top_coverage = df_stats['Interaction_Count'].head(top_n).sum()
        top_coverage_perc = df_stats['Percentage'].head(top_n).sum()
        
        print(f"Top {top_n} 分类单元覆盖了总相互作用的 {top_coverage_perc:.2f}% ({top_coverage:,} 对)")
        print("-" * 50 + "\n")

    print("✅ 统计分析完成。请参考以上输出结果编写论文描述。")
    return results

if __name__ == "__main__":
    analyze_taxonomy_statistics()