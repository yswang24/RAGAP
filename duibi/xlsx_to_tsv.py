#!/usr/bin/env python3
"""
xlsx_to_tsv.py

将 Excel 文件 (.xlsx/.xls) 转换为 TSV 文件。
支持：
 - 指定 sheet 名称或索引导出单个 sheet
 - --all 导出所有 sheet（每个 sheet 输出一个独立的 tsv）
 - 可选是否包含行索引、NA 表示、输出编码

用法示例：
# 导出默认第一个 sheet
python xlsx_to_tsv.py input.xlsx output.tsv

# 指定 sheet 名称
python xlsx_to_tsv.py input.xlsx output.tsv --sheet "Sheet1"

# 指定 sheet 索引（0-based）
python xlsx_to_tsv.py input.xlsx output.tsv --sheet-index 2

# 导出所有 sheet，输出文件会变为 input_SheetName.tsv
python xlsx_to_tsv.py input.xlsx --all

# 不导出行索引，指定 NA 表示为空字符串
python xlsx_to_tsv.py input.xlsx output.tsv --no-index --na-rep ""
"""
import argparse
import os
import sys
import pandas as pd

def convert_single_sheet(input_path: str,
                         output_path: str,
                         sheet_name_or_index,
                         include_index: bool = True,
                         na_rep: str = '',
                         encoding: str = 'utf-8'):
    # pandas 自动根据类型接受 sheet 名称（str）或索引（int）
    df = pd.read_excel(input_path, sheet_name=sheet_name_or_index, engine='xlrd')

    df.to_csv(output_path, sep='\t', index=include_index, na_rep=na_rep, encoding=encoding)
    print(f'Wrote: {output_path}')

def convert_all_sheets(input_path: str,
                       out_dir: str = None,
                       include_index: bool = True,
                       na_rep: str = '',
                       encoding: str = 'utf-8'):
    # 读取所有 sheets 返回 dict of DataFrames
    sheets = pd.read_excel(input_path, sheet_name=None, engine='openpyxl')
    base = os.path.splitext(os.path.basename(input_path))[0]
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    for sheet_name, df in sheets.items():
        # 生成安全的文件名（移除或替换特殊字符）
        safe_sheet = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in sheet_name).strip()
        filename = f"{base}_{safe_sheet}.tsv"
        out_path = os.path.join(out_dir, filename)
        df.to_csv(out_path, sep='\t', index=include_index, na_rep=na_rep, encoding=encoding)
        print(f'Wrote: {out_path}')
    print(f"Exported {len(sheets)} sheets.")

def parse_args():
    p = argparse.ArgumentParser(description="Convert Excel (.xlsx/.xls) to TSV.")
    p.add_argument("input", help="Input Excel file (.xlsx or .xls)")
    p.add_argument("output", nargs='?', help="Output TSV file. If --all is used, this can be omitted (output folder used).")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--sheet", help="Sheet name to export (string).")
    group.add_argument("--sheet-index", type=int, help="Sheet index to export (0-based).")
    p.add_argument("--all", action="store_true", help="Export all sheets to separate TSV files. If used, output (if provided) is treated as output directory.")
    p.add_argument("--no-index", dest="include_index", action="store_false", help="Do not write row index to TSV.")
    p.add_argument("--na-rep", default='', help="String representation for missing values (default: empty).")
    p.add_argument("--encoding", default='utf-8', help="Output file encoding (default: utf-8).")
    return p.parse_args()

def main():
    args = parse_args()
    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    if args.all:
        # output may be provided as directory or omitted
        out_dir = args.output if args.output else None
        convert_all_sheets(input_path, out_dir=out_dir, include_index=args.include_index, na_rep=args.na_rep, encoding=args.encoding)
        return

    # single sheet mode
    if not args.output:
        print("Error: output path required for single-sheet conversion (unless --all).", file=sys.stderr)
        sys.exit(2)
    output_path = args.output

    # determine sheet selection
    sheet_sel = None
    if args.sheet is not None:
        sheet_sel = args.sheet
    elif args.sheet_index is not None:
        sheet_sel = args.sheet_index
    else:
        # default: first sheet -> pandas read_excel uses 0 by default; but to be explicit:
        sheet_sel = 0

    convert_single_sheet(input_path, output_path, sheet_sel, include_index=args.include_index, na_rep=args.na_rep, encoding=args.encoding)

if __name__ == "__main__":
    main()
