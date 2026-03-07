# ================== 独立图例生成器：修复 Newick Error ==================
import pandas as pd
from ete3 import Tree, TreeStyle, RectFace, TextFace
import os
import colorsys
from PIL import Image 

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- 🔧 核心参数调节区 (与主图保持一致) 🔧 ---
INPUT_META = "phage_host.tsv"
TOP_N = 10          # 显示的分类数量 (沿用主图的 TOP_N=30)
# -----------------------------------------------------------

print("1. 读取注释并计算配色方案...")
df = pd.read_csv(INPUT_META, sep='\t')
df['host_taxid'] = df['host_taxid'].astype(str)

# 统计 Top N
counts = df['order'].value_counts()
top_families = counts.head(TOP_N).index.tolist()
other_count = len(df) - counts.head(TOP_N).sum()

# 自动生成配色方案
def get_n_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        hex_col = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_col)
    return colors

auto_colors = get_n_colors(len(top_families))
color_map = {fam: auto_colors[i] for i, fam in enumerate(top_families)}
if other_count > 0:
    color_map["Other"] = "#FBFF00"

# 2. 创建一个虚拟树和图例专用样式
print("2. 正在创建图例画布 (已修复 Newick Error)...")
# FIX: 使用最稳定的空树构造函数，绕过 Newick 字符串解析错误
t_legend = Tree() 
root = t_legend.get_tree_root()

# 图例专用的 TreeStyle (设置为矩形模式，边距归零)
ts_legend = TreeStyle()
ts_legend.mode = "r" # 矩形模式
ts_legend.show_leaf_name = False
ts_legend.show_branch_length = False
ts_legend.show_scale = False
ts_legend.min_leaf_separation = 0 

# 强制将所有边距设为小值，让图片大小紧贴内容
ts_legend.margin_top = ts_legend.margin_bottom = 10
ts_legend.margin_left = ts_legend.margin_right = 10


# 3. 绘制图例内容
# 图例标题
root.add_face(TextFace(f"Phage Host Order (Top {TOP_N})", fsize=30, bold=True), 
              column=0, position="aligned")
root.add_face(TextFace(" ", fsize=15), column=0, position="aligned") # 空行

# 绘制 Top N 分类
for fam in top_families:
    col = color_map[fam]
    cnt = counts[fam]
    pct = cnt / len(df) * 100
    
    # 图例图标（第 0 列）
    icon = RectFace(width=40, height=40, fgcolor=col, bgcolor=col)
    root.add_face(icon, column=0, position="aligned") 
    
    # 图例文字（第 1 列）
    text = TextFace(f"  {fam} ({cnt}, {pct:.1f}%)", fsize=20)
    root.add_face(text, column=1, position="aligned")

# Other 分类
if other_count > 0:
    col = color_map["Other"]
    pct = other_count / len(df) * 100
    icon = RectFace(width=40, height=40, fgcolor=col, bgcolor=col)
    root.add_face(icon, column=0, position="aligned")
    text = TextFace(f"  Other ({other_count}, {pct:.1f}%)", fsize=20)
    root.add_face(text, column=1, position="aligned")

# 4. 渲染和白底处理
raw_png = f"Standalone_Legend_Raw.png"
final_png = f"Standalone_Legend_Fixed.png"
final_svg = f"Standalone_Legend_Fixed.svg"

print("3. 正在渲染图例...")
# 渲染（w=1000px 保证清晰度）
t_legend.render(final_svg, w=1000, tree_style=ts_legend)
t_legend.render(raw_png, w=1000, tree_style=ts_legend)

print("4. 处理白底...")
try:
    img = Image.open(raw_png)
    bg = Image.new("RGB", img.size, (255, 255, 255))
    if img.mode == 'RGBA':
        bg.paste(img, mask=img.split()[3])
    else:
        bg.paste(img)
    bg.save(final_png)
    print(f"✅ 成功！独立图例已生成: {os.path.abspath(final_png)}")
except Exception as e:
    print(f"❌ 背景处理失败: {e}")