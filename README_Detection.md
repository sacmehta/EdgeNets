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

### Testing

We test our detection network using below command:
```
CUDA_VISIBLE_DEVICES='' python test_detection.py --data-path ./vision_datasets/pascal_voc/VOCdevkit/ --im-size 512 
```

See command line arguments for more details.