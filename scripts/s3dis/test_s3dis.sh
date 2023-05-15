#!/bin/bash

export PYTHONPATH=./

log_dir='S3DIS_A5_k=16'

CUDA_VISIBLE_DEVICES=0 python3 tool/test_s3dis.py --log_dir ${log_dir} \
          --batch_size_test 12 \
          --gpu_id 0 \
          --model CBFLNet.CBFLNet \
          --test_area 5 \
          --filter