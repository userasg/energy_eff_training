#!/bin/bash

#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:4
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

python ./main.py \
  --model mobilenet_v2 \
  --mode train_with_switching \
  --epoch 50 \
  --save_path cifar10_results/mobilenet_v2 \
  --dataset cifar10 \
  --batch_size 32 \
  --start_revision 0 \
  --task classification \
  --threshold 0.3 \
  --seed 42