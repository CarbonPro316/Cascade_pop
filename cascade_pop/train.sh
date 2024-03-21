#!/bin/bash
nohup python -u src/pre_training.py --name=dblp --input=./datasets/aps/ --projection_head=4-1  --cuda_device='2' --PE=True --s_weight=dblp > log/dblp/pretrain.log 2>&1 &
nohup python -u src/pre_training.py --name=weibo --input=./datasets/weibo/ --projection_head=4-1  --cuda_device='2' --PE=True --s_weight=weibo > log/weibo/pretrain.log 2>&1 &


