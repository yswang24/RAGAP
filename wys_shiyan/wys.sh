#!/bin/bash
#SBATCH --job-name=train        # 作业名称
#SBATCH --output=output-train-GAT-_4heads_weight_10.20_hid256_512_neg15_evl10_ph2_2layer_1e-3_cos.pt.txt        # 输出日志的文件名
#SBATCH --time=2400:00:00            # 执行时间限制为1小时
#SBATCH --ntasks=1              # 任务数为1
#SBATCH --cpus-per-task=64        # 每个任务使用2个 CPU 核心
#SBATCH --mem=300G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl   # 队列名称为test-hpc-1
#SBATCH --gres=gpu:1               # 如果需要，使用1个GPU.

  # python train_hgt_phage_host_full.py \
  # --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits.pt \
  # --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv \
  # --device cuda \
  # --eval_device cuda \
  # --epochs 1000 \
  # --hidden_dim 256 \
  # --out_dim 256 \
  # --n_layers 2 \
  # --n_heads 4 \
  # --num_neighbors 15 10 \
  # --batch_size 512 \
  # --neg_ratio 30 \
  # --save_path best_hgt_nb2.pt


# python train_hgt_phage_host.py \
#   --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 400 \
#   --hidden_dim 256 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 15 10 \
#   --batch_size 512 \
#   --neg_ratio 30 \
#   --eval_neg_ratio 1 \
#   --save_path best_hgt_nb3.pt\
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --log_every 40


# python train_hgt_phage_host.py \
#   --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 8000 \
#   --hidden_dim 512 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 20 10 \
#   --batch_size 512 \
#   --neg_ratio 20 \
#   --eval_neg_ratio 10 \
#   --save_path best_hgt_nb_HGT_4heads.pt\
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --lr 5e-4\
#   --log_every 1000

# python train_hgt_phage_host_weight.py \
#   --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 8000 \
#   --hidden_dim 256 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 30 15 \
#   --batch_size 1024 \
#   --neg_ratio 20 \
#   --eval_neg_ratio 10 \
#   --save_path best_hgt_nb_RBP_GAT_4heads_weight.pt\
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --lr 5e-4\
#   --log_every 800





python train_hgt_phage_host_weight.py \
  --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP_650.pt \
  --device cuda \
  --eval_device cpu \
  --epochs 8000 \
  --hidden_dim 256 \
  --out_dim 256 \
  --n_layers 2 \
  --n_heads 4 \
  --num_neighbors 10 20 \
  --batch_size 512 \
  --neg_ratio 15 \
  --eval_neg_ratio 10 \
  --save_path best_hgt_nb_RBP_GAT_4heads_weight_10.20_hid256_512_neg15_evl10_ph2_2layer_1e-3_cos.pt\
  --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
  --dropout 0.2\
  --lr 1e-3\
  --log_every 800



