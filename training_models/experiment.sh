#!/bin/bash

#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:4
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

python ./main.py \
  --model efficientnet_b0 \
  --mode train_with_configurable \
  --epoch 200 \
  --save_path cifar_results/efficientnet_b0 \
  --dataset cifar \
  --batch_size 32 \
  --start_revision 0 \
  --task classification \
  --threshold 0.3 \
  --seed 123