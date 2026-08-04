#!/bin/zsh
set -e

/opt/anaconda3/bin/python train.py \
  --config configs/mean_teacher_supcon.yaml \
  --method baseline

/opt/anaconda3/bin/python train.py \
  --config configs/mean_teacher_supcon.yaml \
  --method mean_teacher
