# Object detection

## Dataset

You can download the PASCAL VOC and the MS-COCO data by following below commands:
```
cd  ./data_loader/detection/scripts
bash download_voc.sh
bash download_coco.sh
```
 
## Training and Testing

### Training

We train our detection network using below commands:
```
train='vision_datasets/pascal_voc/VOCdevkit'
bsz=24
dst='pascal'

CUDA_VISIBLE_DEVICES=4 python train_detection.py --data-path ./vision_datasets/pascal_voc/VOCdevkit --batch-size 24 --dataset pascal --im-size 512
```

```
For COCO
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ../vision_datasets/coco --lr-type hybrid --lr 0.01 --clr-max 61 --batch-size 160 --epochs 100 --im-size 300
For PASCAL
CUDA_VISIBLE_DEVICES=0 python train_detection.py --model espnetv2 --s 2.0 --dataset pascal --data-path ../vision_datasets/pascal/VOCdevkit --lr-type hybrid --lr 0.01 --clr-max 61 --batch-size 40 --epochs 100 --im-size 300 
```

### Testing

We test our detection network using below command:
```
CUDA_VISIBLE_DEVICES='' python test_detection.py --data-path ./vision_datasets/pascal_voc/VOCdevkit/ --im-size 512 
```

See command line arguments for more details.