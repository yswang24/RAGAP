#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-accession FASTA from a big FASTA by matching accession.version in headers.
Example header: >KF787094.1 |Achromobacter phage JWDelta, complete genome
Output file name: KF787094.fasta
"""

import os
import re
import argparse
import gzip
import csv

HEADER_ACC_RE = re.compile(r'^>(\S+)')  # the first non-space token after '>' (e.g., KF787094.1)

def open_maybe_gzip(path, mode='rt'):
    return gzip.open(path, mode) if path.endswith(('.gz', '.bgz', '.bgzip')) else open(path, mode)

def read_accessions_from_tsv(tsv_path):
    accs = []
    with open(tsv_path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        # 必须包含 "accession" 列
        if 'accession' not in reader.fieldnames:
            raise ValueError(f"TSV 缺少 'accession' 列，实际列为: {reader.fieldnames}")
        for row in reader:
            acc = row['accession'].strip()
            if acc:
                accs.append(acc)
    return accs

def iter_fasta_records(fasta_path):
    """
    Stream through fasta (or fasta.gz), yielding (header_line, seq_string_without_newlines).
    header_line includes the leading '>'.
    """
    with open_maybe_gzip(fasta_path, 'rt') as fh:
        header = None
        seq_chunks = []
        for line in fh:
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header.rstrip('\n'), ''.join(seq_chunks)
                header = line.rstrip('\n')
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        # last record
        if header is not None:
            yield header.rstrip('\n'), ''.join(seq_chunks)

def main():
    ap = argparse.ArgumentParser(description="Split big FASTA into small FASTAs by accessions from TSV.")
    ap.add_argument('--tsv', required=True, help='包含 accession 列的 TSV 文件')
    ap.add_argument('--fasta', required=True, help='大型 FASTA（可为 .gz）')
    ap.add_argument('--outdir', required=True, help='输出文件夹（若不存在将创建）')
    ap.add_argument('--strict_version', action='store_true',
                    help='严格要求版本号必须为 .1（默认允许 .2/.3 等，只要前缀是 accession.）')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    targets = read_accessions_from_tsv(args.tsv)
    target_set = set(targets)
    found = {acc: False for acc in target_set}

    # 为每个 accession 预构建匹配函数：header 中的第一个 token 必须是 "accession.<digits>"
    def match_header(header):
        m = HEADER_ACC_RE.match(header)
        if not m:
            return None, None  # no token
        token = m.group(1)  # e.g., KF787094.1
        # 拆成 accession 和 version
        if '.' in token:
            acc_prefix, ver = token.split('.', 1)
        else:
            acc_prefix, ver = token, None

        if acc_prefix not in target_set:
            return None, None

        # 版本号校验
        if args.strict_version:
            # 仅接受 .1
            if ver != '1':
                return None, None
        else:
            # 接受任意数字版本（.1、.2、...）
            if ver is None or not ver.isdigit():
                return None, None

        return acc_prefix, token  # return accession (no version) and full token

    # 遍历 FASTA，命中就写出
    write_counts = 0
    for header, seq in iter_fasta_records(args.fasta):
        acc_nover, full_token = match_header(header)
        if acc_nover is None:
            continue
        out_path = os.path.join(args.outdir, f"{acc_nover}.fasta")
        with open(out_path, 'w') as out:
            # 输出用不带版本的 accession 命名，但 header 保留原 token（含版本）
            out.write(f">{full_token}\n")
            # 每行 60 列可读性更好
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + "\n")
        if not found[acc_nover]:
            write_counts += 1
        found[acc_nover] = True

    # 报告
    missing = [acc for acc, ok in found.items() if not ok]
    print(f"[完成] 已输出 {write_counts} 个小 FASTA 到: {args.outdir}")
    if missing:
        print(f"[警告] 下列 accession 未在 FASTA 中找到匹配（按 {'严格' if args.strict_version else '宽松'} 版本规则）:")
        for acc in missing:
            print("  -", acc)

if __name__ == '__main__':
    main()
'''
python extract_train_data.py \
  --tsv /home/wangjingyuan/wys/duibi/VHM_PAIR_TAX_filter_new_GCF.tsv \
  --fasta /home/wangjingyuan/wys/duibi/CHERRY/dataset/nucl.fasta \
  --outdir cherry_train_fasta_withna


  python extract_train_data.py \
  --tsv /home/wangjingyuan/wys/duibi/TEST_PAIR_TAX_filter_new_GCF.tsv \
  --fasta /home/wangjingyuan/wys/duibi/test_contigs.fa \
  --outdir cherry_test_fasta_withna
'''