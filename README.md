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
 
 
## Image Classification

### Training and Testing on the ImageNet

This repository supports training and testing of following models:
 * [DiCENet]()
 * [ESPNetv2 (CVPR'19)](https://arxiv.org/abs/1811.11431)
 * [ShuffleNetv2 (ECCV'18)](https://arxiv.org/abs/1807.11164)
 
 
### Dataset preparation
For dataset preparation, please follow the instuctions given [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

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

### Training and Testing on the MS-COCO (multi-label classification)

If you want to use the MS-COCO dataset, change the `--dataset imagenet` to `--dataset coco` in above commands. Obviously, you need to provide the dataset location too using the `--data` flag.


## Semantic Segmentation

### Dataset

#### PASCAL VOC 2012
A standard practice to train segmentation model on the PASCAL VOC is with additional images from the MS-COCO. We also follow the standard procedure.

Follow below steps to download data, including directory set-up:
 * First download the `COCO` and `VOC` data. You can do this by executing following commands (I am assuming that you are at the root directory i.e. inside `EdgeNets`):
 ```
 cd ./data_loader/segmentation/scripts 
 bash download_voc.sh 
 bash download_coco.sh
 ```
 * Above commands will download the PASCAL VOC and the COCO datasets and place it in `EdgeNets/vision_datasets` directory.
 * Next, you need to prepare the COCO dataset in the PASCAL VOC format because you have 80 classes in COCO while PASCAL VOC has only 21 classes including background.
 * After you have successfully downloaded the COCO dataset, execute following commands to prepare COCO dataset in the PASCAL VOC format:
 ```
 cd ./data_loader/segmentation
 python3 coco.py 
 ```
 * This processing will take few hours. Be patient.
 * That's all. You are set for training on the PASCAL VOC dataset now.
 
#### Cityscapes dataset
For your convenience, we provide bash scripts that allows you to download the dataset without using web browser. Follow below steps for downloading and setting-up the Cityscapes dataset.

* Go to `scripts` directory
```
cd  ./data_loader/segmentation/scripts 
``` 

 * Using any text editor, modify the `uname` and `pass` variables inside the `download_cityscapes.sh` file. These variables correspond to your user name and password for the Cityscapes dataset.
 ```
 # enter user details
uname='' #
pass='' 
 ```
 * After you have entered your credential, execute the `download_cityscapes.sh` bash script to download the data.
 * Next, you need to process Cityscapes segmentation masks for training. To do so, follow below commands:
 ```
 cd ./data_loader/cityscape_scripts 
 python3 process_cityscapes.py
 python3 generate_mappings.py
 ```
 * Now, you are set for training on the Cityscapes dataset.
 
Once you complete all above steps, directory structure will look like this:
```
.
EdgeNets
+-- commons
+-- data_loader
+-- loss_fns
+-- model
+-- nn_layers
+-- transforms
+-- utilities
+-- vision-datasets
|   +-- pascal_voc
|   +-- coco
|   +-- cityscapes
|   +-- coco_preprocess
```


### Training and Testing

#### Training


#### Testing

You can test the 
```
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset city --data-path ../vision_datasets/cityscapes/ --split val --im-size 1024 512
```

## Object Detection

### Training and Testing on the PASCAL VOC

### Training and Testing on the MS-COCO
    

# EffCNN
# EdgeNets
