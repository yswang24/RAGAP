import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ================== 1. 用户可调参数 ==================

# 选择层级: "species" 或 "genus"
LEVEL = "species"  # 可切换为 "genus"

# 最大的 k（1~25），例如 20 就画 Hit@1~20
K_MAX = 20

# 背景虚线网格颜色 & 透明度（颜色深浅）
GRID_COLOR = "#AAAAAA"  
GRID_ALPHA = 0.7       
GRID_LINEWIDTH = 0.5    

# 输出文件名（自动带上 level）
OUT_FIG = f"hit_at_k_curves_{LEVEL}_selected_retrain.svg"


# ================== 2. 实验数据整理（仅包含 Ours, CHERRY, DeepHost） ==================

# --- Ours 数据 (保持不变) ---
species_ours = [ 0.8120135721017907, 0.8586239396795476, 0.883129123468426, 0.8944392082940622, 0.8991517436380773, 0.9029217719132894, 0.9057492931196984, 0.9095193213949104, 0.9114043355325165, 0.9132893496701225, 0.9151743638077285, 0.9170593779453345, 0.9198868991517436, 0.9208294062205467, 0.9227144203581527, 0.9245994344957588, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, ]
genus_ours = [ 0.9010392082940622, 0.9198868991517436, 0.9368520263901979, 0.939679547596607, 0.9472196041470311, 0.9509896324222432, 0.9547596606974552, 0.9557021677662583, 0.9585296889726673, 0.9613572101790764, 0.9632422243166824, 0.9632422243166824, 0.9641847313854854, 0.9641847313854854, 0.9660697455230914, 0.9670122525918945, 0.9679547596606974, 0.9679547596606974, 0.9679547596606974, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, ]

# --- 重新训练后的 CHERRY 和 DeepHost 数据 ---
species_cherry = [ 0.5301, 0.6335, 0.6947, 0.7340, 0.7636, 0.7847, 0.8057, 0.8153, 0.8258, 0.8344, 0.8459, 0.8517, 0.8593, 0.8651, 0.8794, 0.8823, 0.8852, 0.8861, 0.8871, 0.8919, 0.8938, 0.8957, 0.8986, 0.9014, 0.9043 ]
genus_cherry = [ 0.6459, 0.7646, 0.8211, 0.8431, 0.8660, 0.8766, 0.8909, 0.8947, 0.9033, 0.9100, 0.9158, 0.9215, 0.9273, 0.9292, 0.9330, 0.9340, 0.9349, 0.9359, 0.9388, 0.9416, 0.9435, 0.9464, 0.9474, 0.9493, 0.9522 ]
species_deephost = [ 0.72667295, 0.78793591, 0.81055608, 0.83034873, 0.83977380, 0.84542884, 0.84825636, 0.85202639, 0.85296890, 0.85485391, 0.85768143, 0.86239397, 0.86522149, 0.86710650, 0.86804901, 0.86899152, 0.86993402, 0.87370405, 0.87558907, 0.87653157, 0.87747408, 0.87747408, 0.87841659, 0.87841659, 0.87935910 ]
genus_deephost = [ 0.84071631, 0.86710650, 0.87558907, 0.87841659, 0.88312912, 0.88689915, 0.88972667, 0.88972667, 0.89349670, 0.89632422, 0.89726673, 0.89820924, 0.90009425, 0.90197926, 0.90197926, 0.90386428, 0.90386428, 0.90574929, 0.90857681, 0.90857681, 0.91046183, 0.91234684, 0.91328935, 0.91328935, 0.91328935 ]

# 只包含 Ours, CHERRY, DeepHost
HIT_DATA = {
    "species": { "CHERRY": species_cherry, "DeepHost": species_deephost, "Ours": species_ours, },
    "genus": { "CHERRY": genus_cherry, "DeepHost": genus_deephost, "Ours": genus_ours, },
}

# 颜色和标记 (只为选定的方法)
color_map = {
    "CHERRY": "#ff7f0e",  # 橙
    "DeepHost": "#2ca02c", # 绿
    "Ours": "#d62728",   # 红
}
markers = {
    "CHERRY": "s", "DeepHost": "D", "Ours": "^",
}


# ================== 3. Matplotlib 样式（统一风格） ==================

def setup_matplotlib_style():
    plt.rcParams.update({
        "figure.figsize": (4.5, 3.5), # 统一尺寸
        "font.family": "sans-serif",
        "font.size": 9,       # 基础字体大小
        "axes.labelsize": 10,    # X/Y 轴标签
        "axes.titlesize": 10,    # 标题/子图标题
        "xtick.labelsize": 8,    # X 轴刻度标签
        "ytick.labelsize": 8,    # Y 轴刻度标签
        "legend.fontsize": 8,    # 图例字体大小
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,   # 统一的线宽
        "lines.markersize": 5,    # 统一的标记大小
    })


# ================== 4. 绘图主函数 ==================

def plot_hit_at_k(level: str, k_max: int, out_file: str):
    assert level in HIT_DATA, f"LEVEL 必须是 'species' 或 'genus'，当前: {level}"
    assert 1 <= k_max <= 25, f"K_MAX 必须在 1~25 之间，当前: {k_max}"

    setup_matplotlib_style() # 应用统一的样式

    curves = HIT_DATA[level]
    k_values = np.arange(1, k_max + 1, dtype=float)

    fig, ax = plt.subplots()

    # 画多条曲线
    for name in ["CHERRY", "DeepHost", "Ours"]: # 确保绘图顺序一致
        if name in curves: # 检查方法是否存在
            full_values = curves[name]
            y = np.array(full_values[:k_max], dtype=float)
            ax.plot(
                k_values,
                y,
                label=name,
                color=color_map[name],
                linestyle="-",
                marker=markers.get(name, "o"),
                markersize=4.5, 
                linewidth=1.5, 
                markerfacecolor="white",
                markeredgewidth=0.8,
            )

    # X 轴设置
    ax.set_xlabel("k", labelpad=5) 
    ax.set_xticks(k_values)
    ax.set_xticklabels(
        [f"{int(k)}" for k in k_values],
        rotation=45,
        ha="right",
        fontsize=8, 
    )
    
    # Y 轴设置
    ax.set_ylabel("Hit@k")
    ax.set_title(f"Hit@k Curve ({level.capitalize()} Level)", loc='center', fontsize=10)
    
    # 根据级别调整 Y 轴范围和刻度
    if level == "species":
        ax.set_ylim(0.5, 1.0) # 调整 Y 轴下限以更好地展示差异
        ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else: # genus
        ax.set_ylim(0.5, 1.0) # 调整 Y 轴下限
        ax.yaxis.set_major_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 

    # 边框和网格
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        alpha=GRID_ALPHA,
        linestyle="--",
        linewidth=GRID_LINEWIDTH,
        color=GRID_COLOR,
    )

    # 图例
    ax.legend(
        loc="lower right", 
        frameon=False,
        handlelength=2.5,
        handletextpad=0.6,
        borderaxespad=0.4,
        markerscale=1.0, 
    )

    fig.tight_layout()
    fig.savefig(out_file)
    print(f"Saved figure to: {out_file}")


# ================== 5. 脚本入口 ==================

if __name__ == "__main__":
    plot_hit_at_k("species", K_MAX, f"hit_at_k_curves_species_selected_retrain.svg")
    plot_hit_at_k("genus", K_MAX, f"hit_at_k_curves_genus_selected_retrain.svg")








# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# # ================== 1. 原始数据 (仅包含 Ours, CHERRY, DeepHost) ==================

# # 仅保留 Ours, CHERRY, DeepHost 的 Hit@1 值
# methods = ["CHERRY", "DeepHost", "Ours"]

# species_vals = [
#     0.5301,  # CHERRY (Retrain Hit@1)
#     0.7267,  # DeepHost (Retrain Hit@1, 四舍五入)
#     0.8124   # Ours (原值)
# ] 

# genus_vals = [
#     0.6459,  # CHERRY (Retrain Hit@1)
#     0.8407,  # DeepHost (Retrain Hit@1, 四舍五入)
#     0.9010   # Ours (原值)
# ] 

# # 颜色映射 (仅为选定的方法)
# color_map = {
#     "CHERRY": "#ff7f0e", # 橙
#     "DeepHost": "#2ca02c", # 绿
#     "Ours": "#d62728", # 红
# }
# colors = [color_map[m] for m in methods]

# OUT_FIG = "hit_at1_bar_comparison_selected_retrain.svg"


# # ================== 2. 样式设置（统一风格） ==================

# def setup_matplotlib_style():
#     plt.rcParams.update({
#         "figure.figsize": (4.5, 3.5), # 调整为适合3个柱子的尺寸，并保持与折线图整体视觉一致
#         "font.family": "sans-serif",
#         "font.size": 9, # 基础字体大小
#         "axes.labelsize": 10, # X/Y 轴标签
#         "axes.titlesize": 10, # 标题/子图标题
#         "xtick.labelsize": 8, # X 轴刻度标签
#         "ytick.labelsize": 8, # Y 轴刻度标签
#         "legend.fontsize": 8, # 图例字体大小
#         "axes.linewidth": 0.8,
#         "xtick.major.width": 0.8,
#         "ytick.major.width": 0.8,
#         "savefig.dpi": 600,
#         "savefig.bbox": "tight",
#         "lines.linewidth": 1.5,
#         "lines.markersize": 5,
#         "hatch.linewidth": 0.5, # 柱状图的统一设置
#     })


# # ================== 3. 双子图绘制函数 ==================

# def plot_hit_at1_comparison(species_vals, genus_vals, methods, colors, out_file):
#     setup_matplotlib_style() # 应用统一的样式

#     x = np.arange(len(methods))
    
#     # 将两个柱状图放在一个图窗的两个子图内
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(7.0, 3.5)) # 重新设置 figsize 以适应并排子图

#     # --- Subplot 1: Species Level ---
#     ax1.bar(
#         x, species_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.5
#     )
    
#     # 保持子图标题和字体一致
#     ax1.set_title("Species Level", loc='center', fontsize=10,)
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=8) 
#     ax1.set_ylabel("Hit@1") 
    
#     # Y 轴设置
#     ax1.set_ylim(0.0, 1.0)
#     ax1.yaxis.set_major_locator(MultipleLocator(0.1))
#     ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#     # 数据标签
#     for i, v in enumerate(species_vals):
#         ax1.text(
#             x[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7, 
#         )

#     ax1.spines["top"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color="#AAAAAA")


#     # --- Subplot 2: Genus Level ---
#     ax2.bar(
#         x, genus_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.5
#     )

#     # 保持子图标题和字体一致
#     ax2.set_title("Genus Level", loc='center', fontsize=10, )
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=8) 
#     ax2.set_ylabel("Hit@1")
    
#     # Y 轴设置 (Genus 级别)
#     y_min_genus = 0.6 # 根据新的数据调整下限
#     ax2.set_ylim(y_min_genus, 1.0)
#     ax2.yaxis.set_major_locator(MultipleLocator(0.1)) # 调整为0.1间距
#     ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 

#     # 数据标签
#     for i, v in enumerate(genus_vals):
#         ax2.text(
#             x[i], v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7, 
#         )
    
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color="#AAAAAA")

#     plt.subplots_adjust(wspace=0.3) # 调整子图间距
    
#     fig.tight_layout()
#     fig.savefig(out_file)
#     print(f"Saved figure to: {out_file}")


# # ================== 4. 主入口 ==================

# if __name__ == "__main__":
#     plot_hit_at1_comparison(species_vals, genus_vals, methods, colors, OUT_FIG)