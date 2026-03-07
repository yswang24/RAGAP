import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import patches

# === Data ===
data = {
    'Method': ['iPHoP', 'CHERRY', 'PHIST', 'DeepHost', 'Ours'],
    'Hit_1': [0.177, 0.332, 0.388, 0.618, 0.812],
    'Type': ['Baseline', 'Baseline', 'Baseline', 'SOTA', 'Proposed']
}

# Color mapping requested by user
color_map = {
    "iPHoP": "#1f77b4",  # 蓝
    "CHERRY": "#ff7f0e", # 橙
    "DeepHost": "#2ca02c",# 绿
    "PHIST": "#9467bd",  # 紫
    "Ours": "#d62728"    # 红
}

df = pd.DataFrame(data).sort_values('Hit_1', ascending=True).reset_index(drop=True)

# === Publication-quality style ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

fig, ax = plt.subplots(figsize=(6, 4.8), dpi=600)

# Positions
y_pos = np.arange(len(df))

# Subtle horizontal guide lines
ax.hlines(y=y_pos, xmin=0, xmax=df['Hit_1'], color='#e6e6e6', linewidth=2, zorder=1)
ax.set_axisbelow(True)
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.6)

# Points, white edge for crisp look
for idx, row in df.iterrows():
    ax.scatter(row['Hit_1'], y_pos[idx],
               s=150 if row['Method']=='Ours' else 150,
               color=color_map[row['Method']],
               edgecolor='white', linewidth=1.6, zorder=5)
    ax.plot([0, row['Hit_1']], [y_pos[idx], y_pos[idx]], color='#dcdcdc', linewidth=0.6, zorder=2)

    # Value label
    ax.text(row['Hit_1'] + 0.02, y_pos[idx], f"{row['Hit_1']:.3f}",
            va='center', fontsize=10, fontweight='bold', color=color_map[row['Method']])

# Y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(df['Method'])
ax.invert_yaxis()  # highest value on top

# Labels and title
ax.set_xlabel('Hit@1 Accuracy', labelpad=8)
ax.set_xlim(0, 1.0)
ax.set_title('Top-1 Accuracy Comparison (Species level)', pad=12, fontweight='bold')

# Clean spines
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_linewidth(0.8)

# Legend (ordered by the DataFrame order)
legend_patches = [patches.Patch(color=color_map[m], label=m) for m in df['Method']]
ax.legend(handles=legend_patches, frameon=False, ncol=5, bbox_to_anchor=(0.5, -0.18), loc='upper center')

# Annotate improvement between DeepHost and Ours
deephost_val = df.loc[df['Method']=='DeepHost', 'Hit_1'].values[0]
ours_val = df.loc[df['Method']=='Ours', 'Hit_1'].values[0]
y_deep = int(df.index[df['Method']=='DeepHost'][0])
y_ours = int(df.index[df['Method']=='Ours'][0])

mid_x = (deephost_val + ours_val) / 2
annot_y = min(y_deep, y_ours) - 0.6  # place annotation slightly above (axis inverted)

# bracket-style connector
ax.plot([deephost_val, deephost_val, ours_val, ours_val],
        [annot_y+0.15, annot_y, annot_y, annot_y+0.15],
        color=color_map['Ours'], linewidth=1.6, solid_capstyle='butt', zorder=4)

impr_pct = (ours_val - deephost_val) / deephost_val * 100
ax.text(mid_x, annot_y - 0.06, f'+{19.4}% improvement', ha='center',
        fontsize=10, fontweight='bold', color=color_map['Ours'])

plt.tight_layout()

# Save as SVG
outpath = 'dullbell_top1_comparison.svg'
fig.savefig(outpath, format='svg', bbox_inches='tight')
print(f'Saved SVG to: {outpath}')
