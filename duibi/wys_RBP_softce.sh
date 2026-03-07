#!/bin/bash
#SBATCH --job-name=train        # 作业名称
#SBATCH --output=output-train-GAT_softce_cluster_hid512_2layer_4heads_20.10_2048_neg50_evl-1_drop0.2_1e-5_cos_new_40000_True_noleak_30.txt        # 输出日志的文件名
#SBATCH --time=2400:00:00            # 执行时间限制为1小时
#SBATCH --ntasks=1              # 任务数为1
#SBATCH --cpus-per-task=64        # 每个任务使用2个 CPU 核心
#SBATCH --mem=200G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl   # 队列名称为test-hpc-1
#SBATCH --gres=gpu:1               # 如果需要，使用1个GPU.

# python train_hgt_phage_host_weight_RBP.py \
#   --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP_650_613.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 40000 \
#   --hidden_dim 512 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 20 10 \
#   --batch_size 2048 \
#   --neg_ratio 60 \
#   --eval_neg_ratio 20 \
#   --save_path best_GAT_softce_RBP_hid512_2layer_4heads_20.10_2048_neg60_evl20_drop0.10_5e-4_cos_new_40000_False.pt\
#   --node_maps /home/wangjingyuan/wys/wys_shiyan/node_maps_RBP_650_613.json \
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.10\
#   --lr 5e-4\
#   --log_every 4000\
#   --out_dir  output-train-GAT_softce_RBP_hid512_2layer_4heads_20.10_2048_neg60_evl20_drop0.10_5e-4_cos_new_40000_False




# python train_hgt_phage_host_weight_RBP_noleak.py \
#   --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP_650_613.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 40000 \
#   --hidden_dim 512 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 20 10 \
#   --batch_size 2048 \
#   --neg_ratio 60 \
#   --eval_neg_ratio 50 \
#   --save_path best_GAT_softce_RBP_hid512_2layer_4heads_20.10_2048_neg60_evl50_drop0.2_5e-4_cos_new_40000_True_noleak_study2.pt\
#   --node_maps /home/wangjingyuan/wys/wys_shiyan/node_maps_RBP_650_613.json \
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --lr 5e-4\
#   --log_every 4000\
#   --out_dir  output-train-GAT_softce_RBP_hid512_2layer_4heads_20.10_2048_neg60_evl50_drop0.2_5e-4_cos_new_40000_True_noleak_study2

# python train_hgt_phage_host_weight_copy.py \
#   --data_pt /home/wangjingyuan/wys/wys_shiyan/hetero_graph_with_features_splits4_RBP_650.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 40000 \
#   --hidden_dim 512 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 20 10 \
#   --batch_size 2048 \
#   --neg_ratio 30 \
#   --eval_neg_ratio 20 \
#   --save_path best_GAT_RBP_softce_hid512_2layer_4heads_20.10_2048_neg30_evl20_ph2_p0.5_5e-4_tau0.05cos.pt\
#   --node_maps /home/wangjingyuan/wys/wys_shiyan/node_maps_RBP_650.json \
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --lr 5e-4\
#   --log_every 4000


# python train_hgt_phage_host_weight_RBP_noleak.py \
#   --data_pt /home/wangjingyuan/wys/duibi/hetero_graph_with_features_splits_cherry_retrain.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 30000 \
#   --hidden_dim 256 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 20 10 \
#   --batch_size 512 \
#   --neg_ratio 20 \
#   --eval_neg_ratio 10 \
#   --save_path cherry_hid256_2layer_4heads_20.10_512_neg20_evl10_drop0.2_1e-5_cos_new_30000.pt\
#   --node_maps /home/wangjingyuan/wys/duibi/node_maps_cherry_retrain.json \
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --lr 1e-5\
#   --log_every 3000\
#   --out_dir  output-train-GAT_softce_cluster_hid256_2layer_4heads_20.10_512_neg20_evl10_drop0.2_1e-5_cos_new_30000关键







python train_hgt_phage_host_weight_RBP_noleak.py \
  --data_pt /home/wangjingyuan/wys/duibi/hetero_graph_with_features_splits_cherry_retrain_common_na.pt \
  --device cuda \
  --eval_device cpu \
  --epochs 30000 \
  --hidden_dim 256 \
  --out_dim 256 \
  --n_layers 2 \
  --n_heads 4 \
  --num_neighbors 20 10 \
  --batch_size 1024 \
  --neg_ratio 70 \
  --eval_neg_ratio 20 \
  --save_path new_hid512_2layer_4heads_20.10_1024_neg70_evl5_drop0.2_1e-5_cos_new_30000_common_new1.pt\
  --node_maps /home/wangjingyuan/wys/duibi/node_maps_cherry_retrain_common_na.json \
  --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
  --dropout 0.2\
  --lr 1e-5\
  --log_every 3000\
  --out_dir  new40-train-GAT_softce_cluster_hid512_2layer_4heads_20.10_1024_neg70_evl5_drop0.2_1e-5_cos_new_30000_common_sum_0.05_new.






# python train_hgt_phage_host_weight_RBP_noleak.py \
#   --data_pt /home/wangjingyuan/wys/duibi/hetero_graph_with_features_splits_cherry_retrain_common_na.pt \
#   --device cuda \
#   --eval_device cpu \
#   --epochs 1 \
#   --hidden_dim 256 \
#   --out_dim 256 \
#   --n_layers 2 \
#   --n_heads 4 \
#   --num_neighbors 20 10 \
#   --batch_size 256 \
#   --neg_ratio 70 \
#   --eval_neg_ratio 20 \
#   --save_path new_hid512_2layer_4heads_20.10_1024_neg70_evl5_drop0.2_1e-5_cos_new_30000_common_new1.pt\
#   --node_maps /home/wangjingyuan/wys/duibi/node_maps_cherry_retrain_common_na.json \
#   --taxid2species_tsv /home/wangjingyuan/wys/wys_shiyan/taxid_species.tsv\
#   --dropout 0.2\
#   --lr 1e-4\
#   --log_every 1\
#   --out_dir  new40-train-GAT_softce_cl
