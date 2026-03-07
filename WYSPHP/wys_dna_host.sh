#!/bin/bash
#SBATCH --job-name=dnabert_host       # 作业名称
#SBATCH --output=dnabert_host.txt        # 输出日志的文件名
#SBATCH --time=2400:00:00            # 执行时间限制为1小时
#SBATCH --ntasks=1              # 任务数为1
#SBATCH --cpus-per-task=64        # 每个任务使用2个 CPU 核心
#SBATCH --mem=400G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl   # 队列名称为test-hpc-1
#SBATCH --gres=gpu:1              # 如果需要，使用1个GPU.

# 确保优先加载conda里的 libstdc++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# python dna_bert_embed.py \
#   --fasta /home/wangjingyuan/wys/host_fasta_final \
#   --model /home/wangjingyuan/wys/WYSPHP/DNA_bert_6 \
#   --out_dir dnabert4_host_embeddings_final/ \
#   --k 4 \
#   --window_tokens 510 \
#   --stride_tokens 510 \
#   --batch_size 32 \
#   --device cuda \
#   --precision fp16 \
#   --max_windows 800 \
#   --log dna_bert_batch.log


python dna_bert_embed.py \
  --fasta /home/wangjingyuan/wys/duibi/selected_fasta_na \
  --model /home/wangjingyuan/wys/WYSPHP/DNA_bert_4 \
  --out_dir /home/wangjingyuan/wys/duibi/selected_fasta_na_dna \
  --k 4 \
  --window_tokens 510 \
  --stride_tokens 510 \
  --batch_size 32 \
  --device cuda \
  --precision fp16 \
  --max_windows 800 \
  --log dna_bert_batch.log