#!/bin/bash

export PYTHONPATH=./

log_dir='BECFNet_A5_surface+sym+encCBFF+decCBEFF'

CUDA_VISIBLE_DEVICES=0 python3 tool/train_BECFNet.py --log_dir ${log_dir} --dataset S3DIS \
          --batch_size 4 \
          --batch_size_val 4 \
          --workers 8 \
          --gpus 0 \
          --model S_PointNet++.CBFLNet \
          --optimizer AdamW \
          --min_val 60 \
          --epoch 100 \
          --lr_decay_epochs 60 80 \
          --test_area 5 \
          --learning_rate 0.006 \
          --lr_decay 0.1 \
          --weight_decay 1e-2 \
          --aug_scale \
          --color_contrast \
          --color_shift \
          --color_jitter \
          --hs_shift \
          --nsample 8 16 16 16 16 \
          --fea_dim 6 \
          --voxel_size 0.02