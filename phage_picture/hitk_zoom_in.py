import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ================== 1. 实验数据 (保持不变) ==================

# --- Ours 数据 ---
species_ours = [0.8120, 0.8586, 0.8831, 0.8944, 0.8992, 0.9029, 0.9057, 0.9095, 0.9114, 0.9133, 
                0.9152, 0.9171, 0.9199, 0.9208, 0.9227, 0.9246, 0.9255, 0.9255, 0.9255, 0.9255]
genus_ours = [0.9010, 0.9199, 0.9369, 0.9397, 0.9472, 0.9510, 0.9548, 0.9557, 0.9585, 0.9614, 
              0.9632, 0.9632, 0.9642, 0.9642, 0.9661, 0.9670, 0.9680, 0.9680, 0.9680, 0.9689]

# --- 重新训练后的 CHERRY 和 DeepHost 数据 ---
species_cherry = [0.5301, 0.6335, 0.6947, 0.7340, 0.7636, 0.7847, 0.8057, 0.8153, 0.8258, 0.8344, 
                  0.8459, 0.8517, 0.8593, 0.8651, 0.8794, 0.8823, 0.8852, 0.8861, 0.8871, 0.8919]
genus_cherry = [0.6459, 0.7646, 0.8211, 0.8431, 0.8660, 0.8766, 0.8909, 0.8947, 0.9033, 0.9100, 
                0.9158, 0.9215, 0.9273, 0.9292, 0.9330, 0.9340, 0.9349, 0.9359, 0.9388, 0.9416]

species_deephost = [0.7267, 0.7879, 0.8106, 0.8303, 0.8398, 0.8454, 0.8483, 0.8520, 0.8530, 0.8549, 
                    0.8577, 0.8624, 0.8652, 0.8671, 0.8680, 0.8690, 0.8699, 0.8737, 0.8756, 0.8765]
genus_deephost = [0.8407, 0.8671, 0.8756, 0.8784, 0.8831, 0.8869, 0.8897, 0.8897, 0.8935, 0.8963, 
                  0.8973, 0.8982, 0.9001, 0.9020, 0.9020, 0.9039, 0.9039, 0.9057, 0.9086, 0.9086]

# 数据字典
HIT_DATA = {
    "species": { "CHERRY": species_cherry, "DeepHost": species_deephost, "Ours": species_ours },
    "genus": { "CHERRY": genus_cherry, "DeepHost": genus_deephost, "Ours": genus_ours },
}

# 颜色和标记配置
color_map = {
    "CHERRY": "#ff7f0e",   # 橙色
    "DeepHost": "#2ca02c", # 绿色
    "Ours": "#d62728",     # 红色
}
markers = {
    "CHERRY": "s",    # 方块
    "DeepHost": "D",  # 菱形
    "Ours": "^",      # 三角
}

# ================== 2. 绘图样式设置 ==================

def setup_matplotlib_style():
    plt.rcParams.update({
        "figure.figsize": (6, 4.8),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
        "font.weight": "normal",      # <--- 强制全局字体为正常粗细 (非粗体)
        "axes.labelweight": "normal", # <--- 坐标轴标签不加粗
        "axes.titleweight": "normal", # <--- 标题不加粗
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

# ================== 3. 绘图核心函数 ==================

def plot_hit_k_with_zoom(level, k_max=20, out_file=None):
    setup_matplotlib_style()
    
    # 获取数据
    data_dict = HIT_DATA[level]
    k_values = np.arange(1, k_max + 1)
    
    # 创建画布
    fig, ax = plt.subplots()
    
    # ---------------- 主图绘制 ----------------
    for name in ["Ours", "DeepHost", "CHERRY"]:
        if name not in data_dict: continue
        
        y_values = data_dict[name][:k_max]
        
        ax.plot(k_values, y_values,
                label=name,
                color=color_map[name],
                marker=markers[name],
                linestyle='-',
                markeredgecolor='white',
                markeredgewidth=0.8,
                alpha=0.9)

    # 主图坐标轴设置 (移除 fontweight='bold')
    ax.set_xlabel("k")
    ax.set_ylabel("Hit@k Accuracy")
    ax.set_title(f"Hit@k Curve ({level.capitalize()} Level)", pad=15)
    
    ax.set_xticks(np.arange(0, k_max + 1, 2))
    ax.set_xlim(0.5, k_max + 0.5)
    
    if level == "species":
        ax.set_ylim(0.4, 1.02)
    else:
        ax.set_ylim(0.55, 1.02)
        
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例
    ax.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.0, 0.05))

    # ---------------- 局部放大图 (Inset Zoom) ----------------
    # 【平移指南】修改 bbox_to_anchor 中的第 2 个参数
    # (x, y, width, height) -> y 越小，图越靠下
    # 原来是 0.1，现在改为 -0.15 试试 (根据需要调整)
    VERTICAL_SHIFT = 0

    
    axins = inset_axes(ax, width="40%", height="35%", 
                       loc='center right', 
                       borderpad=1.5,
                       # ↓↓↓↓↓↓↓↓↓ 修改这里控制位置 ↓↓↓↓↓↓↓↓↓
                       bbox_to_anchor=(0.05, VERTICAL_SHIFT, 1, 1), 
                       # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                       bbox_transform=ax.transAxes)
    
    zoom_k = 5
    zoom_k_vals = k_values[:zoom_k]
    
    for name in ["Ours", "DeepHost", "CHERRY"]:
        y_vals = data_dict[name][:zoom_k]
        
        axins.plot(zoom_k_vals, y_vals,
                   color=color_map[name],
                   marker=markers[name],
                   linestyle='-',
                   linewidth=1.5,
                   markersize=5,
                   markeredgecolor='white',
                   markeredgewidth=0.5)
                   
    # 放大图设置 (移除 bold)
    axins.set_title(f"Top-{zoom_k}", fontsize=9)
    axins.set_xlim(0.8, zoom_k + 0.2)
    axins.set_xticks(np.arange(1, zoom_k + 1))
    axins.tick_params(labelsize=8)
    
    # 动态调整放大图 Y 轴范围
    all_zoom_y = []
    for n in ["Ours", "DeepHost", "CHERRY"]:
        all_zoom_y.extend(data_dict[n][:zoom_k])
    y_min, y_max = min(all_zoom_y), max(all_zoom_y)
    margin = (y_max - y_min) * 0.1
    axins.set_ylim(y_min - margin, y_max + margin)
    
    axins.grid(True, linestyle=':', alpha=0.3)
    
    # 添加连接线
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.5, linestyle='--')

    # ---------------- 保存 ----------------
    if out_file:
        plt.tight_layout()
        plt.savefig(out_file, format='svg')
        print(f"Saved: {out_file}")
    plt.show()

# ================== 4. 运行 ==================
if __name__ == "__main__":
    plot_hit_k_with_zoom("species", k_max=20, out_file="hit_at_k_zoom_species.svg")
    plot_hit_k_with_zoom("genus", k_max=20, out_file="hit_at_k_zoom_genus.svg")