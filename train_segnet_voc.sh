#!/usr/bin/env bash

data_path='vision_datasets/pascal_voc/VOCdevkit'
coco_path='vision_datasets/coco_preprocess'
imSz=256
bsz=40
arch='espnet'
s=2.0
dset='pascal'

CUDA_VISIBLE_DEVICES=7 python train_segmentation.py --data-path $data_path --crop-size $imSz $imSz --batch-size $bsz --coco-path $coco_path --s $s --model $arch --dataset $dset
