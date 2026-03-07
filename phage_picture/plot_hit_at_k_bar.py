# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Bar plot of Hit@1 (accuracy) for four methods:
# - iPHoP (blue)
# - CHERRY (orange)
# - DeepHost (green)
# - Ours (red)

# Two separate SVG files:
# - hit_at1_bar_species.svg  (species-level Hit@1)
# - hit_at1_bar_genus.svg    (genus-level Hit@1)

# Y-axis: 0.0 ~ 1.0 (小数)
# """

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# # ================== 1. 原始数据 ==================

# # iPHoP
# species_iphop_hit1 = 0.1768
# genus_iphop_hit1   = 0.7453

# # CHERRY
# species_cherry_hit1 = 0.3318
# genus_cherry_hit1   = 0.3754

# # DeepHost
# species_deephost_hit1 = 0.6182846371347785
# genus_deephost_hit1   = 0.7181903864278982

# # Ours
# species_ours_hit1 = 0.8166135721017907
# genus_ours_hit1   = 0.8944392082940622

# methods = ["iPHoP", "CHERRY", "DeepHost", "Ours"]

# species_vals = [
#     species_iphop_hit1,
#     species_cherry_hit1,
#     species_deephost_hit1,
#     species_ours_hit1,
# ]

# genus_vals = [
#     genus_iphop_hit1,
#     genus_cherry_hit1,
#     genus_deephost_hit1,
#     genus_ours_hit1,
# ]

# # 颜色保持和折线图一致
# color_map = {
#     "iPHoP": "#1f77b4",    # 蓝
#     "CHERRY": "#ff7f0e",   # 橙
#     "DeepHost": "#2ca02c", # 绿
#     "Ours": "#d62728",     # 红
# }

# OUT_FIG_SPECIES = "hit_at1_bar_species.svg"
# OUT_FIG_GENUS   = "hit_at1_bar_genus.svg"


# # ================== 2. 样式设置 ==================

# def setup_matplotlib_style():
#     plt.rcParams.update({
#         "figure.figsize": (3.0, 3.5),
#         "font.family": "sans-serif",
#         "font.size": 9,
#         "axes.labelsize": 10,
#         "axes.titlesize": 10,
#         "xtick.labelsize": 8,
#         "ytick.labelsize": 8,
#         "legend.fontsize": 8,
#         "axes.linewidth": 0.8,
#         "xtick.major.width": 0.8,
#         "ytick.major.width": 0.8,
#         "savefig.dpi": 300,
#         "savefig.bbox": "tight",
#     })


# # ================== 3. 单图画图函数 ==================

# def plot_single_level(title: str, vals, out_file: str, ylabel: str = "Accuracy"):
#     setup_matplotlib_style()

#     x = np.arange(len(methods))
#     fig, ax = plt.subplots()

#     for i, m in enumerate(methods):
#         ax.bar(
#             x[i],
#             vals[i],
#             color=color_map[m],
#             width=0.6,
#         )

#     ax.set_title(title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(methods, rotation=30, ha="right")
#     ax.set_ylabel(ylabel)
#     ax.set_ylim(0.0, 1.0)
#     ax.yaxis.set_major_locator(MultipleLocator(0.1))
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid(axis="y", alpha=0.5, linestyle="--", linewidth=0.6, color="#888888")

#     # 在柱子上标数值
#     for i, v in enumerate(vals):
#         ax.text(
#             x[i],
#             v + 0.02,
#             f"{v:.2f}",
#             ha="center",
#             va="bottom",
#             fontsize=7,
#         )

#     fig.tight_layout()
#     fig.savefig(out_file)
#     print(f"Saved figure to: {out_file}")


# # ================== 4. 主入口 ==================

# if __name__ == "__main__":
#     # species 级别
#     plot_single_level("Species", species_vals, OUT_FIG_SPECIES, ylabel="Accuracy")

#     # genus 级别
#     plot_single_level("Genus", genus_vals, OUT_FIG_GENUS, ylabel="Accuracy")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bar plot of Hit@1 (accuracy) for five methods:
- iPHoP (blue)
- CHERRY (orange)
- DeepHost (green)
- PHIST (purple)
- Ours (red)

Two separate SVG files:
- hit_at1_bar_species.svg  (species-level Hit@1)
- hit_at1_bar_genus.svg    (genus-level Hit@1)

Y-axis: 0.0 ~ 1.0 (小数)
"""

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# # ================== 1. 原始数据 ==================

# # iPHoP
# species_iphop_hit1 = 0.1768
# genus_iphop_hit1   = 0.7453

# # CHERRY
# species_cherry_hit1 = 0.3318
# genus_cherry_hit1   = 0.3754

# # DeepHost
# species_deephost_hit1 = 0.6182846371347785
# genus_deephost_hit1   = 0.7181903864278982

# # PHIST
# species_phist_hit1 = 0.3876
# genus_phist_hit1   = 0.6704

# # Ours
# species_ours_hit1 = 0.8166135721017907
# genus_ours_hit1   = 0.8944392082940622

# methods = ["iPHoP", "CHERRY", "DeepHost", "PHIST", "Ours"]

# species_vals = [
#     species_iphop_hit1,
#     species_cherry_hit1,
#     species_deephost_hit1,
#     species_phist_hit1,
#     species_ours_hit1,
# ]

# genus_vals = [
#     genus_iphop_hit1,
#     genus_cherry_hit1,
#     genus_deephost_hit1,
#     genus_phist_hit1,
#     genus_ours_hit1,
# ]

# # 颜色保持和折线图一致
# color_map = {
#     "iPHoP": "#1f77b4",    # 蓝
#     "CHERRY": "#ff7f0e",   # 橙
#     "DeepHost": "#2ca02c", # 绿
#     "PHIST": "#9467bd",    # 紫
#     "Ours": "#d62728",     # 红
# }

# OUT_FIG_SPECIES = "hit_at1_bar_species.svg"
# OUT_FIG_GENUS   = "hit_at1_bar_genus.svg"


# # ================== 2. 样式设置 ==================

# def setup_matplotlib_style():
#     plt.rcParams.update({
#         "figure.figsize": (3.2, 3.5),
#         "font.family": "sans-serif",
#         "font.size": 9,
#         "axes.labelsize": 10,
#         "axes.titlesize": 10,
#         "xtick.labelsize": 8,
#         "ytick.labelsize": 8,
#         "legend.fontsize": 8,
#         "axes.linewidth": 0.8,
#         "xtick.major.width": 0.8,
#         "ytick.major.width": 0.8,
#         "savefig.dpi": 600,
#         "savefig.bbox": "tight",
#     })


# # ================== 3. 单图画图函数 ==================

# def plot_single_level(title: str, vals, out_file: str, ylabel: str = "Accuracy"):
#     setup_matplotlib_style()

#     x = np.arange(len(methods))
#     fig, ax = plt.subplots()

#     for i, m in enumerate(methods):
#         ax.bar(
#             x[i],
#             vals[i],
#             color=color_map[m],
#             width=0.6,
#         )

#     ax.set_title(title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(methods, rotation=30, ha="right")
#     ax.set_ylabel(ylabel)
#     ax.set_ylim(0.0, 1.0)
#     ax.yaxis.set_major_locator(MultipleLocator(0.1))
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid(axis="y", alpha=0.5, linestyle="--", linewidth=0.6, color="#888888")

#     # # 在柱子上标数值
#     # for i, v in enumerate(vals):
#     #     ax.text(
#     #         x[i],
#     #         v + 0.02,
#     #         f"{v:.2f}",
#     #         ha="center",
#     #         va="bottom",
#     #         fontsize=7,
#     #     )

#     fig.tight_layout()
#     fig.savefig(out_file)
#     print(f"Saved figure to: {out_file}")


# # ================== 4. 主入口 ==================

# if __name__ == "__main__":
#     # species 级别
#     plot_single_level("Species", species_vals, OUT_FIG_SPECIES, ylabel="Accuracy")

#     # genus 级别
#     plot_single_level("Genus", genus_vals, OUT_FIG_GENUS, ylabel="Accuracy")




import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ================== 1. 原始数据 (Original Data) ==================

# 各方法在 Species/Genus 级别的 Hit@1 值
species_vals = [0.1768, 0.3318, 0.6183, 0.3876, 0.8124] # 简化小数位数
genus_vals = [0.7453, 0.3754, 0.7182, 0.6704, 0.8944] # 简化小数位数

methods = ["iPHoP", "CHERRY", "DeepHost", "PHIST", "Ours"]

# 颜色保持一致
color_map = {
    "iPHoP": "#1f77b4",  # 蓝
    "CHERRY": "#ff7f0e",  # 橙
    "DeepHost": "#2ca02c", # 绿
    "PHIST": "#9467bd", # 紫
    "Ours": "#d62728",  # 红
}
colors = [color_map[m] for m in methods]

OUT_FIG = "hit_at1_bar_comparison.svg"


# ================== 2. 样式设置 (Top-Tier Style Setup) ==================

def setup_matplotlib_style():
    plt.rcParams.update({
        "figure.figsize": (7.0, 3.5), # 调整为并排的宽图
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9, # 增大 X 轴标签
        "ytick.labelsize": 9, # 增大 Y 轴标签
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

# ================== 3. 双子图绘制函数 (Main Plotting Function) ==================

def plot_hit_at1_comparison(species_vals, genus_vals, methods, colors, out_file):
    setup_matplotlib_style()

    x = np.arange(len(methods))
    
    # 核心改进：创建包含两个子图的图窗
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False) # 不共享 Y 轴，以便分别优化

    # --- Subplot 1: Species Level ---
    
    # 绘制柱状图
    ax1.bar(
        x,
        species_vals,
        color=colors,
        width=0.6,
        edgecolor='black', # 添加黑色边框，增强视觉分离度
        linewidth=0.5
    )

    # 设置子图标签和标题
    ax1.set_title("(a) Species Level", loc='left', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylabel("Hit@1") # 专业化 Y 轴标签
    
    # 优化 Y 轴范围和刻度
    ax1.set_ylim(0.0, 1.0)
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    #数据标签（在柱子顶部标数值）
    for i, v in enumerate(species_vals):
        ax1.text(
            x[i],
            v + 0.02,
            f"{v:.3f}", # 保留三位小数，符合学术规范
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # 边框和网格线
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color="#AAAAAA")


    # --- Subplot 2: Genus Level ---

    # 绘制柱状图
    ax2.bar(
        x,
        genus_vals,
        color=colors,
        width=0.6,
        edgecolor='black',
        linewidth=0.5
    )

    # 设置子图标签和标题
    ax2.set_title("(b) Genus Level", loc='left', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylabel("Hit@1")
    
    # 优化 Y 轴范围和刻度 (Genus 普遍较高，可以提高下限以突出差异)
    # 最小值约为 0.3754
    y_min_genus = 0 # 设置下限，放大差异
    ax2.set_ylim(y_min_genus, 1.0)
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # 两位小数

    # 数据标签（在柱子顶部标数值）
    for i, v in enumerate(genus_vals):
        ax2.text(
            x[i],
            v + 0.015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    
    # 边框和网格线
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color="#AAAAAA")

    # 调整子图间的距离
    plt.subplots_adjust(wspace=0.3)
    
    fig.tight_layout()
    fig.savefig(out_file)
    print(f"Saved figure to: {out_file}")


# ================== 4. 主入口 (Script Entry) ==================

if __name__ == "__main__":
    plot_hit_at1_comparison(species_vals, genus_vals, methods, colors, OUT_FIG)