# Image Classification

## Training and Testing on the ImageNet

This repository supports training and testing of following models:
 * [DiCENet]()
 * [ESPNetv2 (CVPR'19)](https://arxiv.org/abs/1811.11431)
 * [ShuffleNetv2 (ECCV'18)](https://arxiv.org/abs/1807.11164)
 
 
## Dataset preparation
For dataset preparation, please follow the instuctions given [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

## Training
For smaller models (< 80 MFLOPs), we used a scale augmentation of `(0.2, 1.0)` while for other models, we used a scale augmentation of `(0.08, 1.0)`

For training, you can use below command
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_classification.py --s $s --model <model> --scheduler <scheduler> --data <data-path> --dataset imagenet --scale <scale>
```
Example for DiCENet with a network width scaling factor of 0.2 model is shown below.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_classification.py --s 0.2 --model dicenet --scheduler hybrid --epochs 120 --clr-max 61 --data ./imagenet --scale 0.2 1.0 
``` 

### Testing
Testing can be done in two ways:
 * The first option loads the weights automatically from weight dictionary defined in `/model/weight_locations` and you can use below command to test the models

```
CUDA_VISIBLE_DEVICES=0 python test_classification.py --model dicenet --s 0.2 --dataset imagenet --data <imagenet-loc>
```
 * The second option allows you to specify the location of a pretrained `weights` file, as shown below
```
CUDA_VISIBLE_DEVICES=0 python test_classification.py --model dicenet --s 0.2 --dataset imagenet --data <imagenet-loc> --weights <weights-loc
```

## Training and Testing on the MS-COCO (multi-label classification)

If you want to use the MS-COCO dataset, change the `--dataset imagenet` to `--dataset coco` in above commands. Obviously, you need to provide the dataset location too using the `--data` flag.
