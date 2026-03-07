import pandas as pd

def filter_failed_species_predictions(input_file, output_file=None):
    """
    从label_comparison.tsv中筛选出物种预测失败的行
    
    参数:
        input_file: 输入的label_comparison.tsv文件路径
        output_file: 输出的文件路径，如果为None则使用默认名称
    """
    # 读取文件
    df = pd.read_csv(input_file, sep='\t')
    
    # 筛选条件：hit_species@1为0，表示物种预测失败
    # 同时确保真实物种不为空（否则无法判断预测是否正确）
    failed_species = df[
        (df['hit_species@1'] == 0) & 
        (df['true_species'].notna()) & 
        (df['true_species'] != '')
    ]
    
    # 选择需要的列，便于分析
    columns_to_keep = [
        'phage_id', 
        'true_species', 
        'pred_species_top1',
        'true_genus',
        'pred_genus_top1', 
        'true_family',
        'pred_family_top1',
        'hit_species@1',
        'hit_genus@1',
        'hit_family@1'
    ]
    
    # 只保留存在的列
    available_columns = [col for col in columns_to_keep if col in df.columns]
    failed_species = failed_species[available_columns]
    
    # 生成输出文件名
    if output_file is None:
        output_file = input_file.replace('.tsv', '_failed_species.tsv')
    
    # 保存结果
    failed_species.to_csv(output_file, sep='\t', index=False)
    
    print(f"总共处理了 {len(df)} 行数据")
    print(f"找到 {len(failed_species)} 行物种预测失败的数据")
    print(f"结果已保存到: {output_file}")
    
    # 显示一些统计信息
    if len(failed_species) > 0:
        print("\n前10个预测失败的例子:")
        print(failed_species.head(10).to_string(index=False))
        
        # 统计属级和科级的正确率
        genus_correct = failed_species['hit_genus@1'].sum()
        family_correct = failed_species['hit_family@1'].sum()
        
        print(f"\n在这些物种预测失败的情况下:")
        print(f"- 属级预测正确的数量: {genus_correct}/{len(failed_species)} ({genus_correct/len(failed_species)*100:.2f}%)")
        print(f"- 科级预测正确的数量: {family_correct}/{len(failed_species)} ({family_correct/len(failed_species)*100:.2f}%)")
    
    return failed_species

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_file = "accuracy_result_data2_label_comparison.tsv"
    
    # 运行筛选
    failed_df = filter_failed_species_predictions(input_file)
    
    # 如果你想要自定义输出文件名，可以这样使用:
    # 
    failed_df = filter_failed_species_predictions(input_file, "acc_failed_predictions.tsv")