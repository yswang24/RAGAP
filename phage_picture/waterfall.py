import matplotlib.pyplot as plt
import numpy as np
import textwrap

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

short_tick_labels = ['Full\nModel', 'A', 'B', 'C', 'D', 'E', 'F']
annotation_titles = [
    'Reference',
    'Ablation A',
    'Ablation B',
    'Ablation C',
    'Ablation D',
    'Ablation E',
    'Ablation F',
]
annotation_texts = [
    'All modules retained',
    'Remove Protein nodes and related edges',
    'Remove Taxonomy nodes and related edges',
    'Remove Protein and Taxonomy',
    'Fix relation weighting parameters to a constant',
    'Replace with traditional BPR Pairwise Loss',
    'Use MLP instead of Cosine Decoder',
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

fig, (ax, ax_note) = plt.subplots(
    2,
    1,
    figsize=(12.5, 7.4),
    sharex=True,
    gridspec_kw={"height_ratios": [4.8, 1.9], "hspace": 0.02}
)

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
    x[0], BASELINE_VAL + 0.0035,
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
        BASELINE_VAL - drop - 0.010,
        f"{deltas[i]:.3f}",
        ha='center', va='top',
        fontsize=9, color='#333333'
    )

# ================== 5. 坐标轴与网格 ==================
ax.set_xticks(x)
ax.set_xticklabels(short_tick_labels, fontsize=10, linespacing=1.1, fontweight='bold')
ax.tick_params(axis='x', length=0, pad=7, labelbottom=True)

ax.set_ylabel('Hit@1 Accuracy', fontsize=12)
ax.set_ylim(min(values) - 0.025, 0.825)
ax.set_yticks([0.70, 0.75, 0.80, 0.825])

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

# ================== 6. 与 x 轴对齐的注释带 ==================
ax_note.set_ylim(0, 1)
ax_note.set_xlim(-0.5, len(labels) - 0.5)
ax_note.set_yticks([])
ax_note.tick_params(axis='x', length=0, labelbottom=False)

for side in ['left', 'right', 'bottom']:
    ax_note.spines[side].set_visible(False)
ax_note.spines['top'].set_visible(True)
ax_note.spines['top'].set_linewidth(0.8)
ax_note.spines['top'].set_color('black')

for boundary in np.arange(-0.5, len(labels), 1.0):
    ax_note.axvline(
        x=boundary,
        ymin=0.30,
        ymax=0.90,
        color='#d9d9d9',
        linewidth=0.8,
        zorder=0
    )

for i, (title, text) in enumerate(zip(annotation_titles, annotation_texts)):
    wrapped_text = textwrap.fill(text, width=18)
    ax_note.text(
        x[i],
        0.8,
        title,
        ha='center',
        va='center',
        fontsize=12 ,
        fontweight='bold',
        color=COLOR_BASE if i == 0 else '#222222'
    )
    ax_note.text(
        x[i],
        0.5,
        wrapped_text,
        ha='center',
        va='center',
        fontsize=12 ,
        color='#333333',
        linespacing=1.15
    )

# ================== 7. 输出 ==================
fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.10)
plt.savefig('waterfull_ablation_waterfall_red_journal.svg', format='svg')
plt.show()
