# import pandas as pd

# # 1. 改成你的真实文件名！！！
# filename = "phage_host.tsv"   # 例如：host_info.tsv  或  phage_host_table.tsv 等

# # 2. 先尝试有表头读取（99% 的情况都是这样）
# try:
#     df = pd.read_csv(filename, sep='\t')                     # 有表头
#     print("检测到文件自带表头，读取成功")
# except:
#     df = pd.read_csv(filename, sep='\t', header=None)         # 没有表头兜底

# # 3. 统一处理列名（去除空格、BOM、换行符等）
# df.columns = df.columns.astype(str).str.strip().str.replace('﻿', '')

# # 4. 如果检测到第一行是表头（包含 'host_taxid' 字符串），就删掉它
# if 'host_taxid' in df.iloc[0].values:
#     print("发现重复的表头行，已自动删除")
#     df = df[df['host_taxid'] != 'host_taxid']   # 删除那一行

# # 5. 确保 host_taxid 是字符串类型，并去掉可能的空值
# df['host_taxid'] = df['host_taxid'].astype(str).str.strip()
# df = df[df['host_taxid'].str.isdigit()]   # 只保留纯数字的 taxid

# # 6. 提取唯一宿主
# unique_hosts = df[['host_taxid', 'domain', 'phylum', 'class', 'order',
#                    'order', 'genus', 'species']].drop_duplicates().reset_index(drop=True)

# # 7. 转为整数列表，用于 NCBI 查询
# taxids = unique_hosts['host_taxid'].astype(int).tolist()

# print(f"最终得到 {len(taxids)} 个有效宿主 taxid：")
# print(taxids[:10])   # 只显示前10个
# from Bio import Entrez
# from ete3 import NCBITaxa  # ETE3 内置 NCBI 分类数据库

# Entrez.email = "wys1215070346@gmail"  # NCBI 访问所需

# # 初始化 NCBI 分类（如果需要，下载数据库）
# ncbi = NCBITaxa()

# # 获取唯一的 taxid
# taxids = unique_hosts['host_taxid'].tolist()  # [104099, 438]

# # 获取谱系（从根到叶的祖先 taxid 列表）
# lineages = {taxid: ncbi.get_lineage(taxid) for taxid in taxids}

# # 将 taxid 翻译为名称用于标签
# names = {taxid: ncbi.get_taxid_translator([taxid])[taxid] for taxid in taxids}

# # 构建树：找到最低共同祖先 (LCA) 并构建 Newick
# lca = ncbi.get_topology(taxids).get_common_ancestor(taxids)  # ETE3 拓扑
# tree = ncbi.get_topology(taxids)  # 从根的完整子树

# # 保存为 Newick 用于可视化
# tree.write(format=1, outfile="host_tree.nwk")  # Newick 格式
# print(tree)  # 查看树结构


# fixed_colors = [
    # "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
    # "#ffd92f", "#e5c494", "#b3b3b3", "#1f78b4", "#33a02c",
    # "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#17becf"
# ]










# ================== 终极平衡版：适度重叠 + 完美色块 + 无延长 ==================
import pandas as pd
from ete3 import Tree, TreeStyle, RectFace, TextFace, NodeStyle
import os
import colorsys
from PIL import Image 

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- 🔧 核心参数调节区 🔧 ---
INPUT_TREE = "host_phylogenetic_tree.nwk"
INPUT_META = "phage_host.tsv"
TOP_N = 10          
TREE_SCALE = 3000   # 树的大小

# 【1. 解决“像一条线”的问题】
# 加大色环的径向宽度 (Width)，让它看起来更厚实
RING_WIDTH = 2000    

# 【2. 解决“不是色块” & “延长”的问题】
# 不要用自动计算，也不要用20倍重叠。
# 这里直接给一个适中的固定值：50像素。
# 50px 足够让相邻色块融合在一起形成“色块”，又不会像 200px 那样突出圆外。
MANUAL_ARC_HEIGHT = 320

# 线条粗细
BRANCH_THICKNESS = 50 
# -----------------------------------------------------------
print("1. 加载树并整理形状...")
t = Tree(INPUT_TREE)

# 强制所有分支长度为 1 (形成标准圆)
for node in t.traverse():
    node.dist = 1
t.convert_to_ultrametric()

# 设置树枝粗细
nstyle = NodeStyle()
nstyle["hz_line_width"] = BRANCH_THICKNESS
nstyle["vt_line_width"] = BRANCH_THICKNESS
nstyle["size"] = 0 
for node in t.traverse():
    node.set_style(nstyle)

# 2. 读取数据
print("2. 读取注释...")
df = pd.read_csv(INPUT_META, sep='\t')
df['host_taxid'] = df['host_taxid'].astype(str)
unique_hosts = df[['host_taxid', 'order']].drop_duplicates()
taxid_to_order = dict(zip(unique_hosts['host_taxid'], unique_hosts['order']))

# 3. 统计 Top N
print(f"3. 统计 Top {TOP_N}...")
counts = df['order'].value_counts()
top_families = counts.head(TOP_N).index.tolist()
other_count = len(df) - counts.head(TOP_N).sum()

# 4. 配色
print("4. 生成配色...")
def get_n_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
        hex_col = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_col)
    return colors

auto_colors = get_n_colors(len(top_families))
color_map = {fam: auto_colors[i] for i, fam in enumerate(top_families)}
if other_count > 0:
    color_map["Other"] = "#FBFF00"

# 5. 绘制色环 
t.ladderize()
for leaf in t.iter_leaves():
    taxid = leaf.name.strip()
    fam = taxid_to_order.get(taxid, "Other")
    if fam not in top_families: fam = "Other"
    col = color_map.get(fam, "#E0E0E0")
    
    # 宽度和高度确保是厚实的色块，且能重叠融合
    ring = RectFace(width=RING_WIDTH, height=MANUAL_ARC_HEIGHT, fgcolor=col, bgcolor=col)
    leaf.add_face(ring, column=0, position="branch-right")

# 6. 布局设置
ts = TreeStyle()
ts.mode = "c"
ts.arc_span = 360
ts.force_topology = False 
ts.show_leaf_name = False
ts.show_branch_length = False
ts.show_branch_support = False

# 【关键修复】增加边距，确保图例可见
ts.margin_top = 100 
ts.margin_bottom = 10
ts.margin_left = 100
ts.margin_right = 10

ts.scale = TREE_SCALE 
ts.rotation = 90
ts.draw_guiding_lines = False 

# 7. 图例 (使用 ts.legend 独立层)
print("6. 生成图例...")
ts.legend_position = 1 # 放置在左上角
ts.legend.add_face(TextFace(f"Legend (Top {TOP_N})", fsize=24, bold=True), column=0)
ts.legend.add_face(TextFace(" ", fsize=10), column=0)

for fam in top_families:
    col = color_map[fam]
    cnt = counts[fam]
    pct = cnt / len(df) * 100
    # 图标和文字
    icon = RectFace(width=40, height=40, fgcolor=col, bgcolor=col)
    text = TextFace(f"  {fam} ({cnt}, {pct:.1f}%)", fsize=16)
    ts.legend.add_face(icon, column=0)
    ts.legend.add_face(text, column=1) # 放在下一列

if other_count > 0:
    col = color_map["Other"]
    pct = other_count / len(df) * 100
    icon = RectFace(width=40, height=40, fgcolor=col, bgcolor=col)
    text = TextFace(f"  Other ({other_count}, {pct:.1f}%)", fsize=16)
    ts.legend.add_face(icon, column=0)
    ts.legend.add_face(text, column=1)

# 8. 渲染与背景修复
raw_png = f"Tree_Top{TOP_N}_Legend_Raw.png"
final_png = f"Tree_Top{TOP_N}_Legend_Fix.png"
final_svg = f"Tree_Top{TOP_N}_Legend_Fix.svg"

print("7. 正在渲染...")
# 渲染更大的画布，防止裁剪
t.render(final_svg, tree_style=ts)
t.render(raw_png, w=5000, units="px", tree_style=ts) # 宽度增加到 5000px

print("8. 处理白底...")
try:
    img = Image.open(raw_png)
    bg = Image.new("RGB", img.size, (255, 255, 255))
    if img.mode == 'RGBA':
        bg.paste(img, mask=img.split()[3])
    else:
        bg.paste(img)
    bg.save(final_png)
    print(f"✅ 成功！最终图已生成 (白底、带图例): {os.path.abspath(final_png)}")
except Exception as e:
    print(f"❌ 背景处理失败: {e}")


