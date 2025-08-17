#!/bin/bash

#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:4
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

python optuna_tune_configdropout.py \
  --study configdrop-mbv2-c10 \
  --storage sqlite:///optuna_studies/configdrop.db \
  --n-trials 20 \
  --target-acc 0.90