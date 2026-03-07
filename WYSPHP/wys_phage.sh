#!/bin/bash
#SBATCH --job-name=esm_phage       # 作业名称
#SBATCH --output=output-esm-phage-150.txt        # 输出日志的文件名
#SBATCH --time=2400:00:00            # 执行时间限制为1小时
#SBATCH --ntasks=1              # 任务数为1
#SBATCH --cpus-per-task=64        # 每个任务使用2个 CPU 核心
#SBATCH --mem=300G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl   # 队列名称为test-hpc-1
#SBATCH --gres=gpu:1              # 如果需要，使用1个GPU.

echo "开始时间：$(date)"
echo "运行节点：$(hostname)"
echo "使用 GPU：$CUDA_VISIBLE_DEVICES"
echo "使用 CPU 核数：$SLURM_CPUS_PER_TASK"
echo "-----------------------------------------------------"


# python generate_esm_embeddings.py \
#   --faa-dir /home/wangjingyuan/wys/WYSPHP/annotation_out/host0\
#   --out /home/wangjingyuan/wys/WYSPHP/esm_embeddings_35 \
#   --model-name esm2_t12_35M_UR50D \
#   --batch-size 2 \
#   --repr-l 12 \
#   --device cuda \
#   --workers 2

python generate_esm_embeddings_phage.py \
  --faa-dir /home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage\
  --out /home/wangjingyuan/wys/duibi/cherry_all_fasta_faa/phage_proteinemb\
  --model-name esm2_t33_650M_UR50D \
  --batch-size 2 \
  --repr-l 12 \
  --device cuda \
  --workers 2