import matplotlib.pyplot as plt
import numpy as np

# ================== 1. 数据准备 ==================
data = [
    ('Full Model', 0.812, 0.0),
    ('A', 0.768, -0.044),
    ('B', 0.776, -0.036),
    ('C', 0.724, -0.088),
    ('D', 0.773, -0.039),
    ('E', 0.746, -0.066),
    ('F', 0.785, -0.027)
]

labels = [x[0] for x in data]
values = [x[1] for x in data]
deltas = [x[2] for x in data]

# 将原图例信息直接整合进 x 轴刻度标签，减少图内文字干扰。
tick_labels = [
    'Full Model',
    'A\nw/o Protein nodes\nand related edges',
    'B\nw/o Taxonomy nodes\nand related edges',
    'C\nw/o Protein and\nTaxonomy',
    'D\nFixed relation\nweights',
    'E\nBPR loss',
    'F\nMLP decoder',
]

# ================== 2. 风格设置 ==================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 1.0,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(11, 6.8))

# ================== 3. 顶刊红色配色方案 ==================
COLOR_BASE = '#b22222'     # 深红（Baseline）
COLOR_DROP = '#e07b7b'     # 去饱和浅红（Ablation）
COLOR_GRID = '#e5e5e5'

BAR_WIDTH = 0.6
BASELINE_VAL = values[0]

# ================== 4. 绘制瀑布图 ==================
x = np.arange(len(labels))

# Baseline
ax.bar(
    x[0], BASELINE_VAL,
    color=COLOR_BASE,
    width=BAR_WIDTH,
    edgecolor='black',
    linewidth=0.8,
    zorder=3
)
ax.text(
    x[0], BASELINE_VAL + 0.005,
    f"{BASELINE_VAL:.3f}",
    ha='center', va='bottom',
    fontsize=10.5, color=COLOR_BASE
)

# Baseline reference line
ax.axhline(
    y=BASELINE_VAL,
    color=COLOR_BASE,
    linestyle='--',
    linewidth=1,
    alpha=0.5,
    zorder=1
)

# Ablation bars
for i in range(1, len(data)):
    drop = abs(deltas[i])
    ax.bar(
        x[i], drop,
        bottom=BASELINE_VAL - drop,
        color=COLOR_DROP,
        width=BAR_WIDTH,
        edgecolor='black',
        linewidth=0.6,
        zorder=3
    )
    ax.text(
        x[i],
        BASELINE_VAL - drop - 0.015,
        f"{deltas[i]:.3f}",
        ha='center', va='top',
        fontsize=9, color='#333333'
    )

# ================== 5. 坐标轴与网格 ==================
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=9.5, linespacing=1.15)
ax.tick_params(axis='x', length=0, pad=10)

ax.set_ylabel('Hit@1 Accuracy', fontsize=11)
ax.set_xlabel('Model Variant', fontsize=11, labelpad=12)
ax.set_ylim(min(values) - 0.05, BASELINE_VAL + 0.06)

# ✅ 显示 Y 轴（期刊推荐）
ax.spines['left'].set_visible(True)
ax.spines['left'].set_linewidth(0.8)
ax.spines['left'].set_color('black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.8)

ax.grid(
    axis='y',
    linestyle='--',
    linewidth=0.8,
    color=COLOR_GRID,
    zorder=0
)

# ================== 7. 输出 ==================
plt.tight_layout()
plt.savefig('waterfull_ablation_waterfall_red_journal_change.svg', format='svg')
plt.show()
