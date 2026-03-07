"""
PhageRBPdetect (ESM2-fine) - inference
@author: dimiboeckaerts
@date: 2023-12-21

FIRST, DOWNLOAD THE MODEL FILE FROM ZENODO: https://zenodo.org/records/10515367 
THEN, SET THE PATHS & FILES BELOW & RUN THE SCRIPT

INPUTS: a FASTA file with proteins you want to make predictions for, and a fine-tuned ESM-2 model
OUTPUTS: predictions and scores for each protein in the FASTA file

Notes: 
You will probably want to run this script on a GPU-enabled machine (e.g. Google Colab or Kaggle).
The ESM-2 T12 model can run on a single GPU with 16GB of memory.
You can also directly access our Google Colab notebook here: https://colab.research.google.com/drive/1b0DSqMmnEgoXoWW53VxKpT-N8moPU2DA?usp=sharing.
The results will be saved in a `predictions.csv` file in the path. The file will contain 3 columns: the protein names, 
the binary prediction (0: predicted not an RBP, 1: predicted an RBP) and the associated score 
(between 0 and 1, the higher, the more confident the model is in it being an RBP).

Any feedback or questions? Feel free to send me an email: dimi.boeckaerts@gmail.com.
"""

# # 0 - SET THE PATHS
# # ------------------------------------------
# path = '/home/wangjingyuan/wys/WYSPHP/PhageRBPdetection/data'
# fasta_name = 'sequences.fasta'
# model_name = 'RBPdetect_v4_ESMfine' # should be a folder in the path!

# # 1 - TRAINING THE MODEL
# # ------------------------------------------
# # load libraries
# import os
# import torch
# import pandas as pd
# import numpy as np
# from Bio import SeqIO
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import os

# GPU = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
# device = torch.device("cuda")

# # initiation the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained(path+'/'+model_name)
# model = AutoModelForSequenceClassification.from_pretrained(path+'/'+model_name)
# if torch.cuda.is_available():
#     model.eval().cuda()
#     print("using cuda")
# else:
#     model.eval()
#     print("Using CPU")

# # make predictions
# sequences = [str(record.seq) for record in SeqIO.parse(path+'/'+fasta_name, 'fasta')]
# names = [record.id for record in SeqIO.parse(path+'/'+fasta_name, 'fasta')]

# predictions = []
# scores = []
# for sequence in tqdm(sequences):
#     encoding = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)
#     with torch.no_grad():
#         output = model(**encoding)
#         predictions.append(int(output.logits.argmax(-1)))
#         scores.append(float(output.logits.softmax(-1)[:, 1]))

# # save the results
# results = pd.concat([pd.DataFrame(names, columns=['protein_name']),
#                          pd.DataFrame(predictions, columns=['preds']),
#                         pd.DataFrame(scores, columns=['score'])], axis=1)
# results.to_csv(path+'/predictions.csv', index=False)

"""### Step 3: save predictions!

The results will be saved in a `predictions.csv` file in the content folder on the left, where you uploaded the FASTA file as well. From there, you can right click the .csv file and download the predictions!

The file will contain 3 columns: the protein names, the binary prediction (0: predicted not an RBP, 1: predicted an RBP) and the associated score (between 0 and 1, the higher, the more confident the model is in it being an RBP).

Any feedback or questions? Feel free to send me an email: dimi.boeckaerts@gmail.com.
"""

# === 修改 PhageRBPdetect_v4_inference.py 开头部分 ===

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhageRBPdetect v4 inference
批量预测单个FASTA/FAA文件中的蛋白序列是否为RBP。
输出一个CSV文件，包含：
protein_name, preds (0/1), score (置信度)
"""

import argparse
import os
import torch
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------- 解析参数 ----------
parser = argparse.ArgumentParser(description="PhageRBPdetect v4 inference")
parser.add_argument("--input", required=True, help="输入FASTA/FAA文件路径")
parser.add_argument("--output", required=True, help="输出CSV文件路径")
parser.add_argument("--model_dir", default="/home/wangjingyuan/wys/WYSPHP/PhageRBPdetection/data/RBPdetect_v4_ESMfine",
                    help="模型所在目录（默认：RBPdetect_v4_ESMfine）")
parser.add_argument("--gpu", type=int, default=0, help="使用的GPU编号（默认：0）")
args = parser.parse_args()

# --------- 路径检查 ----------
fasta_path = args.input
if not os.path.isfile(fasta_path):
    raise FileNotFoundError(f"❌ 输入文件不存在: {fasta_path}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)

# --------- 设置设备 ----------
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- 加载模型 ----------
print(f"🔧 加载模型: {args.model_dir}")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.eval().to(device)

print("✅ using cuda" if torch.cuda.is_available() else "⚠️ Using CPU")

# --------- 读取序列 ----------
print(f"📂 读取序列文件: {fasta_path}")
records = list(SeqIO.parse(fasta_path, 'fasta'))
if len(records) == 0:
    raise ValueError(f"❌ 输入文件 {fasta_path} 中没有找到任何FASTA序列！")

sequences = [str(record.seq) for record in records]
names = [record.id for record in records]

# --------- 推理 ----------
predictions = []
scores = []
print(f"🚀 开始预测，共 {len(sequences)} 条序列")
for sequence in tqdm(sequences):
    encoding = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output = model(**encoding)
        predictions.append(int(output.logits.argmax(-1)))
        scores.append(float(output.logits.softmax(-1)[:, 1]))

# --------- 保存结果 ----------
results = pd.DataFrame({
    "protein_name": names,
    "preds": predictions,
    "score": scores
})
results.to_csv(args.output, index=False)
print(f"✅ 结果已保存到: {args.output}")




'''
python PhageRBPdetect_v4_inference.py  --input /home/wangjingyuan/wys/WYSPHP/annotation_out/host_final  --output /home/wangjingyuan/wys/WYSPHP/annotation_out/PhageRBPdetect_v4output_host
'''