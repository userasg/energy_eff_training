#!/bin/bash

#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:4
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

CD_EARLY_STOP_ACC=0.90
python optuna_tune_configdropout.py --n-trials 10 --epochs 30 --target-acc 0.90
