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
****** Image-size: 300x300 ***********
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_detection.py --model espnetv2 --s 2.0 --dataset pascal --data-path ./vision_datasets/pascal_voc/VOCdevkit/ --lr-type hybrid --lr 0.01 --clr-max 61 --batch-size 192 --epochs 100 --im-size 300
****** Image-size: 512x512 ***********
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_detection.py --model espnetv2 --s 2.0 --dataset pascal --data-path ./vision_datasets/pascal_voc/VOCdevkit/ --lr-type hybrid --lr 0.005 --clr-max 61 --batch-size 60 --epochs 100 --im-size 512 --freeze-bn --finetune <weights_file_lcoation>
```

```
###### MS-COCO ###########
****** Image-size: 300x300 ***********
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ./vision_datasets/coco --lr-type hybrid --lr 0.01 --clr-max 61 --batch-size 192 --epochs 100 --im-size 300
****** Image-size: 512x512 ***********
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ./vision_datasets/coco --lr-type hybrid --lr 0.005 --clr-max 61 --batch-size 60 --epochs 100 --im-size 512 --freeze-bn --finetune <weights_file_lcoation>
```

where `<weights_file_lcoation>` is the location of the weights that you obtained from training a model at image resolution of `300x300`.

### Testing

We test our detection network on the PASCAL VOC 2007 or MSCOCO dataset, please use below command:
```
###### PASCAL VOC ###########
CUDA_VISIBLE_DEVICES=0 python test_detection.py --model espnetv2 --s 2.0 --dataset pascal --data-path ./vision_datasets/pascal_voc/VOCdevkit/ --im-size 300
CUDA_VISIBLE_DEVICES=0 python test_detection.py --model espnetv2 --s 2.0 --dataset pascal --data-path ./vision_datasets/pascal_voc/VOCdevkit/ --im-size 512

###### MS-COCO ###########
CUDA_VISIBLE_DEVICES=0 python test_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ./vision_datasets/coco --im-size 300
CUDA_VISIBLE_DEVICES=0 python test_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ./vision_datasets/coco --im-size 512
```

To test on your network, please use below commands:
``` 
CUDA_VISIBLE_DEVICES=0 python test_detection.py --model <model_name> --s <model_network_width> --dataset <dataset_name> --data-path <data_location> --im-size <image_size> --weights-test <weights_location>
```


For more details about the supported arguments, please see the `test_detection.py` file.