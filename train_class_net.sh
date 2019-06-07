#!/usr/bin/env bash

data_path='/mnt/disk1/datasets/imagenet/' # location of the dataset
#data_path='/home/sachinm/imagenet/' # for VS11
#data_path='/home/sacmehta/Desktop/ILSVRC2015/Data/CLS-LOC/'

model='shuffle_vw' # which model (basic, esp, shuffle, res)
scheduler='clr' # LR scheduler (cyclic or fixed)
workers=32

s=1.0 # factor by which output channels should be scaled
exp='main'
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_classification.py --s $s --model $model --scheduler $scheduler --data $data_path --exp-type $exp --scale 0.2 1.0 --workers $workers