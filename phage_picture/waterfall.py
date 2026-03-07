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

# A–F 对应实验说明（只保留这一列）
legend_rows = [
    ['A', 'Remove Protein nodes and related edges'],
    ['B', 'Remove Taxonomy nodes and related edges'],
    ['C', 'Remove Protein and Taxonomy'],
    ['D', 'Fix relation weighting parameters to a constant'],
    ['E', 'Replace with traditional BPR Pairwise Loss'],
    ['F', 'Use MLP instead of Cosine Decoder'],
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

fig, ax = plt.subplots(figsize=(10, 6))

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
ax.set_xticklabels(labels, fontsize=10)

ax.set_ylabel('Hit@1 Accuracy', fontsize=11)
ax.set_ylim(min(values) - 0.05, BASELINE_VAL + 0.06)

ax.set_title(
    'Ablation Study',
    fontsize=12, loc='left', pad=16
)

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

# ================== 6. 右上角 A–F 实验说明表（精简版图例） ==================
# ================== 6. 右上角图例（3 × 2，文本型 legend） ==================

legend_items = [
    ('A', 'Remove Protein nodes and related edges'),
    ('B', 'Remove Taxonomy nodes and related edges'),
    ('C', 'Remove Protein and Taxonomy'),
    ('D', 'Fix relation weighting parameters to a constant'),
    ('E', 'Replace with traditional BPR Pairwise Loss'),
    ('F', 'Use MLP instead of Cosine Decoder'),
]

# 图例起始位置（axes 坐标系：0~1）
x_left  = 0.20
x_right = 0.60
y_start = 0.98
y_step  = 0.065

# 左列：A B C
for i, (key, text) in enumerate(legend_items[:3]):
    ax.text(
        x_left,
        y_start - i * y_step,
        f"{key}: {text}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        color='black'
    )

# 右列：D E F
for i, (key, text) in enumerate(legend_items[3:]):
    ax.text(
        x_right,
        y_start - i * y_step,
        f"{key}: {text}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        color='black'
    )


# ================== 7. 输出 ==================
plt.tight_layout()
plt.savefig('waterfull_ablation_waterfall_red_journal.svg', format='svg')
plt.show()
