#!/usr/bin/env bash

data_path='vision_datasets/cityscapes'
imSz1=512
imSz2=256
bsz=32
arch='espnet'
s=2.0
dset='city'

CUDA_VISIBLE_DEVICES=6 python train_segmentation.py --data-path $data_path --crop-size $imSz1 $imSz2 --batch-size $bsz --s $s --model $arch --dataset $dset
