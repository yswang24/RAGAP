#!/bin/bash
#SBATCH --job-name=train        # 作业名称
#SBATCH --output=output-train-GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg70_evl20_ph2_p0.5_1e-5_cos_new_40000_True.txt        # 输出日志的文件名
#SBATCH --time=2400:00:00            # 执行时间限制为1小时
#SBATCH --ntasks=1              # 任务数为1
#SBATCH --cpus-per-task=64        # 每个任务使用2个 CPU 核心
#SBATCH --mem=300G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl   # 队列名称为test-hpc-1
#SBATCH --gres=gpu:1               # 如果需要，使用1个GPU.





python train_hgt_phage_host_weight_copy.py \
  --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_cluster_650_613.pt \
  --device cuda \
  --eval_device cpu \
  --epochs 40000 \
  --hidden_dim 512 \
  --out_dim 256 \
  --n_layers 2 \
  --n_heads 4 \
  --num_neighbors 20 10 \
  --batch_size 1024 \
  --neg_ratio 70 \
  --eval_neg_ratio 20 \
  --save_path best_GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg70_evl20_ph2_p0.5_1e-5_cos_new_40000_True.pt\
  --node_maps /home/wangjingyuan/wys/wys_shiyan/node_maps_cluster_650_613.json \
  --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
  --dropout 0.2\
  --lr 1e-5\
  --log_every 4000\
  --out_dir  output-train-GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg70_evl20_ph2_p0.5_1e-5_cos_new_40000_True

# python train_hgt_phage_host_weight_copy.py \
#   --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_cluster_650.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 40000 \
#   --hidden_dim 512 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 20 10 \
#   --batch_size 1024 \
#   --neg_ratio 30 \
#   --eval_neg_ratio 20 \
#   --save_path best_GAT_cluster_softce_hid512_2layer_4heads_20.10_1024_neg30_evl20_ph2_p0.5_1e-5_tau0.05cos.pt\
#   --node_maps /home/wangjingyuan/wys/wys_shiyan/node_maps_cluster_650.json \
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --lr 1e-5\
#   --log_every 4000