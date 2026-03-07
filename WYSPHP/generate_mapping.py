#!/usr/bin/env python3
import os, json, glob
import argparse

def build_mapping(phage_dir, host_dir):
    mapping = {}
    # 噬菌体
    for sample in os.listdir(phage_dir):
        sample_dir = os.path.join(phage_dir, sample)
        faa = os.path.join(sample_dir, f"{sample}.faa")
        if not os.path.isfile(faa): continue
        with open(faa) as f:
            for line in f:
                if line.startswith(">"):
                    pid = line[1:].split()[0]
                    mapping[pid] = {"source_type": "phage", "source_id": sample}
    # 宿主
    for faa in glob.glob(os.path.join(host_dir, "*.faa")):
        host_id = os.path.splitext(os.path.basename(faa))[0]
        with open(faa) as f:
            for line in f:
                if line.startswith(">"):
                    pid = line[1:].split()[0]
                    mapping[pid] = {"source_type": "host", "source_id": host_id}
    return mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phage-dir", required=True)
    parser.add_argument("--host-dir", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    mapping = build_mapping(args.phage_dir, args.host_dir)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Mapping table generated: {len(mapping)} proteins → {args.out_json}")

if __name__ == "__main__":
    main()
