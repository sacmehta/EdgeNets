#!/usr/bin/env bash

train='vision_datasets/pascal_voc/VOCdevkit'
bsz=24
dst='pascal'

CUDA_VISIBLE_DEVICES=4 python train_detection.py --data-path $train --batch-size $bsz --dataset $dst
