# DiCENet: Depth-wise Convolutions for Efficient Networks

This repo contains source code of our work on designing efficient networks for different computer vision tasks: 
 * Image classification
    * Single-label classification on the ImageNet dataset
    * Multi-label classification on the MS-COCO dataset
 * Semantic segmentation
    * PASCAL VOC
    * Cityscapes
 * Object detection
    * PASCAL VOC
    * MS-COCO 

Some of our relevant papers are:
 * [ESPNet (ECCV'18)](https://arxiv.org/abs/1803.06815)
 * [ESPNetv2 (CVPR'19)](https://arxiv.org/abs/1811.11431)
 * [DiCENet (arxiv)]()
 
If you find this repository helpful, please feel free to cite our work:
```
@inproceedings{mehta2018espnet,
  title={Espnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation},
  author={Mehta, Sachin and Rastegari, Mohammad and Caspi, Anat and Shapiro, Linda and Hajishirzi, Hannaneh},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={552--568},
  year={2018}
}

@inproceedings{mehta2018espnetv2,
  title={ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network},
  author={Mehta, Sachin and Rastegari, Mohammad and Shapiro, Linda and Hajishirzi, Hannaneh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2019}
}

```

**Key highlights**
 * Object classification on the ImageNet and MS-COCO (multi-label)
 * Semantic Segmentation on the PASCAL VOC and the CityScapes
 * Object Detection on the PASCAL VOC and the MS-COCO
 * Supports PyTorch 0.4 and 1.0
 * Integrated with Tensorboard for easy visualization of training logs. 
 * Scripts for downloading different datasets.
 
 
## Object Classification

### Training and Testing on the ImageNet

### Training
For smaller models (< 80 MFLOPs), we used a scale augmentation of `(0.2, 1.0)` while for other models, we used a scale augmentation of `(0.08, 1.0)`

For training, you can use below command
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_classification.py --s $s --model <model> --scheduler <scheduler> --data <data-path> --dataset imagenet --scale <scale>
```
Example for DiCENet with a network width scaling factor of 0.2 model is shown below.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_classification.py --s 0.2 --model dicenet --scheduler hybrid --epochs 120 --clr-max 61 --data ./imagenet --scale 0.2 1.0 
``` 

#### Testing
Testing can be done in two ways:
 * The first option loads the weights automatically from weight dictionary defined in `/model/weight_locations` and you can use below command to test the models

```
CUDA_VISIBLE_DEVICES=0 python test_classification.py --model dicenet --s 0.2 --dataset imagenet --data <imagenet-loc>
```
 * The second option allows you to specify the location of a pretrained `weights` file, as shown below
```
CUDA_VISIBLE_DEVICES=0 python test_classification.py --model dicenet --s 0.2 --dataset imagenet --data <imagenet-loc> --weights <weights-loc
```

### Training and Testing on the MS-COCO


## Semantic Segmentation

### Training and Testing on the PASCAL VOC

**Dataset:** A standard practice to train segmentation model on the PASCAL VOC is with additional images from the MS-COCO and .
You can download the dataset from


### Training and Testing on the CityScapes


## Object Detection

### Training and Testing on the PASCAL VOC

### Training and Testing on the MS-COCO
    

# EffCNN
# EdgeNets
