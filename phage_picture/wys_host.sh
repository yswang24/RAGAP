#!/bin/bash
#SBATCH --job-name=cdhit-host       # 作业名称
#SBATCH --output=cdhit-host.txt        # 输出日志的文件名
#SBATCH --time=2400:00:00            # 执行时间限制为1小时
#SBATCH --ntasks=1              # 任务数为1
#SBATCH --cpus-per-task=64        # 每个任务使用2个 CPU 核心
#SBATCH --mem=150G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl   # 队列名称为test-hpc-1
#SBATCH --gres=gpu:0              # 如果需要，使用1个GPU.




python cd-hit.py