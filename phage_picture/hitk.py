import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ================== 1. 用户可调参数 ==================

# 选择层级: "species" 或 "genus"
LEVEL = "species"     # 改成 "species" 就画物种水平的曲线

# 最大的 k（1~25），例如 20 就画 Hit@1~20
K_MAX = 20

# 背景虚线网格颜色 & 透明度（颜色深浅）
GRID_COLOR = "#AAAAAA"  
GRID_ALPHA = 0.7       
GRID_LINEWIDTH = 0.5    

# 输出文件名（自动带上 level）
OUT_FIG = f"hit_at_k_curves_{LEVEL}.svg"


# ================== 2. 实验数据整理（与原数据保持一致） ==================

species_iphop = [ 0.1768, 0.2547, 0.2989, 0.3200, 0.3305, 0.3537, 0.3621, 0.3895, 0.4042, 0.4189, 0.4316, 0.4379, 0.4442, 0.4484, 0.4505, 0.4568, 0.4589, 0.4589, 0.4611, 0.4632, 0.4653, 0.4653, 0.4653, 0.4674, 0.4674, ]
genus_iphop = [ 0.7453, 0.8021, 0.8042, 0.8042, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, 0.8063, ]
species_cherry = [ 0.3318, 0.3640, 0.3697, 0.3810, 0.3848, 0.3896, 0.3915, 0.3943, 0.3962, 0.4076, 0.4209, 0.4351, 0.4502, 0.4550, 0.4550, 0.4664, 0.4682, 0.4720, 0.4720, 0.4739, 0.4787, 0.4815, 0.4825, 0.4853, 0.4910, ]
genus_cherry = [ 0.3754, 0.4085, 0.4142, 0.4284, 0.4322, 0.4370, 0.4445, 0.4493, 0.4540, 0.4692, 0.4844, 0.5062, 0.5213, 0.5299, 0.5318, 0.5479, 0.5498, 0.5536, 0.5545, 0.5573, 0.5630, 0.5659, 0.5678, 0.5697, 0.5744, ]
species_deephost = [ 0.6182846371347785, 0.6786050895381716, 0.6936852026390198, 0.7078228086710651, 0.7115928369462771, 0.7181903864278982, 0.7219604147031102, 0.7257304429783223, 0.7285579641847314, 0.7323279924599434, 0.7360980207351555, 0.7370405278039586, 0.7389255419415646, 0.7389255419415646, 0.7436380772855796, 0.7455230914231856, 0.7511781338360037, 0.7530631479736098, 0.7568331762488218, 0.7596606974552309, 0.7596606974552309, 0.76248821866164, 0.765315739868049, 0.7662582469368521, 0.767200754005655, ]
genus_deephost = [ 0.7181903864278982, 0.7832233741753063, 0.8039585296889726, 0.8143261074458058, 0.82186616399623, 0.8294062205466541, 0.8341187558906692, 0.8378887841658812, 0.8407163053722903, 0.8444863336475024, 0.8482563619227145, 0.8529688972667295, 0.8576814326107446, 0.8595664467483506, 0.8633364750235627, 0.8671065032987747, 0.8689915174363808, 0.8718190386427899, 0.8727615457115928, 0.8737040527803959, 0.8746465598491989, 0.8755890669180019, 0.8784165881244109, 0.879359095193214, 0.88124410933082, ]
species_ours = [ 0.8210135721017907, 0.8586239396795476, 0.883129123468426, 0.8944392082940622, 0.8991517436380773, 0.9029217719132894, 0.9057492931196984, 0.9095193213949104, 0.9114043355325165, 0.9132893496701225, 0.9151743638077285, 0.9170593779453345, 0.9198868991517436, 0.9208294062205467, 0.9227144203581527, 0.9245994344957588, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, 0.9255419415645617, ]
genus_ours = [ 0.90104392082940622, 0.9198868991517436, 0.9368520263901979, 0.939679547596607, 0.9472196041470311, 0.9509896324222432, 0.9547596606974552, 0.9557021677662583, 0.9585296889726673, 0.9613572101790764, 0.9632422243166824, 0.9632422243166824, 0.9641847313854854, 0.9641847313854854, 0.9660697455230914, 0.9670122525918945, 0.9679547596606974, 0.9679547596606974, 0.9679547596606974, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, 0.9688972667295005, ]
species_phist = [ 0.3876, 0.3951, 0.3970, 0.4007, 0.4082, 0.4082, 0.4101, 0.4101, 0.4101, 0.4101, 0.4101, 0.4101, 0.4101, 0.4101, 0.4101, 0.4120, 0.4120, 0.4120, 0.4120, 0.4120, 0.4120, 0.4120, 0.4120, 0.4120, 0.4120, ]
genus_phist = [ 0.6704, 0.6835, 0.6854, 0.6873, 0.6873, 0.6873, 0.6873, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, 0.6891, ]

HIT_DATA = {
    "species": { "iPHoP": species_iphop, "CHERRY": species_cherry, "DeepHost": species_deephost, "PHIST": species_phist, "Ours": species_ours, },
    "genus": { "iPHoP": genus_iphop, "CHERRY": genus_cherry, "DeepHost": genus_deephost, "PHIST": genus_phist, "Ours": genus_ours, },
}

# 颜色和标记
color_map = {
    "iPHoP": "#1f77b4",# 蓝
    "CHERRY": "#ff7f0e",# 橙
    "DeepHost": "#2ca02c", # 绿
    "PHIST": "#9467bd",# 紫
    "Ours": "#d62728",# 红
}
markers = {
    "iPHoP": "o", "CHERRY": "s", "DeepHost": "D", "PHIST": "v", "Ours": "^",
}


# ================== 3. Matplotlib 样式（统一风格） ==================

def setup_matplotlib_style():
    plt.rcParams.update({
        "figure.figsize": (4.5, 3.5), # 统一尺寸
        "font.family": "sans-serif",
        "font.size": 9,# 基础字体大小
        "axes.labelsize": 10, # X/Y 轴标签
        "axes.titlesize": 10,# 标题/子图标题
        "xtick.labelsize": 8,# X 轴刻度标签
        "ytick.labelsize": 8,# Y 轴刻度标签
        "legend.fontsize": 8,# 图例字体大小
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,# 统一的线宽
        "lines.markersize": 5, # 统一的标记大小
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
    for name, full_values in curves.items():
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
        fontsize=8, # 强制使用 8pt
    )
    
    # Y 轴设置
    ax.set_ylabel("Hit@k")
    ax.set_title(f"Hit@k Curve ({level.capitalize()} Level)", loc='center', fontsize=10) # 居中标题
    
    if level == "species":
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else: # genus
        # min_y = min([min(v[:k_max]) for v in curves.values()])
        # y_min = max(0.0, np.floor(min_y * 10) / 10 - 0.1) 
        # ax.set_ylim(y_min, 1.0)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.05)) 
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

plot_hit_at_k(LEVEL, K_MAX, OUT_FIG)



# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# # ================== 1. 原始数据 ==================

# # 简化小数位数，保证数据整洁
# species_vals = [0.1768, 0.3318, 0.6183, 0.3876, 0.8210] 
# genus_vals = [0.7453, 0.3754, 0.7182, 0.6704, 0.9010] 

# methods = ["iPHoP", "CHERRY", "DeepHost", "PHIST", "Ours"]

# # 颜色保持一致
# color_map = {
#     "iPHoP": "#1f77b4", # 蓝
#     "CHERRY": "#ff7f0e",  # 橙
#     "DeepHost": "#2ca02c", # 绿
#     "PHIST": "#9467bd", # 紫
#     "Ours": "#d62728",  # 红
# }
# colors = [color_map[m] for m in methods]

# OUT_FIG = "hit_at1_bar_comparison.svg"


# # ================== 2. 样式设置（统一风格） ==================

# def setup_matplotlib_style():
#     plt.rcParams.update({
#         "figure.figsize": (7.0, 3.5), # 为双子图调整尺寸
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
    
#     # 核心改进：创建包含两个子图的图窗
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False) 

#     # --- Subplot 1: Species Level ---
#     ax1.bar(
#         x, species_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.5
#     )
    
#     # 保持子图标题和字体一致
#     ax1.set_title("Species Level", loc='center', fontsize=10,)
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=8) # 强制使用 8pt
#     ax1.set_ylabel("Hit@1") 
    
#     # Y 轴设置
#     ax1.set_ylim(0.0, 1.0)
#     ax1.yaxis.set_major_locator(MultipleLocator(0.1))
#     ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#     # 数据标签
#     for i, v in enumerate(species_vals):
#         ax1.text(
#             x[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7, # 标签字体 7pt
#         )

#     ax1.spines["top"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color="#AAAAAA")


#     # --- Subplot 2: Genus Level ---
#     ax2.bar(
#         x, genus_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.5
#     )

#     # 保持子图标题和字体一致
#     ax2.set_title("Genus Level", loc='center', fontsize=10,)
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=8) # 强制使用 8pt
#     ax2.set_ylabel("Hit@1")
    
#     # Y 轴设置
#     y_min_genus = 0.35 
#     ax2.set_ylim(y_min_genus, 1.0)
#     ax2.yaxis.set_major_locator(MultipleLocator(0.1))
#     ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 

#     # 数据标签
#     for i, v in enumerate(genus_vals):
#         ax2.text(
#             x[i], v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7, # 标签字体 7pt
#         )
    
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.grid(axis="y", alpha=0.6, linestyle="--", linewidth=0.5, color="#AAAAAA")

#     plt.subplots_adjust(wspace=0.3)
    
#     fig.tight_layout()
#     fig.savefig(out_file)
#     print(f"Saved figure to: {out_file}")


# # ================== 4. 主入口 ==================

# if __name__ == "__main__":
#     plot_hit_at1_comparison(species_vals, genus_vals, methods, colors, OUT_FIG)