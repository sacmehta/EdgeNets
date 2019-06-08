# Efficient networks for Computer Vision

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
 * [DiCENet (arxiv)](Coming soon)
 
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

## Key highlights
 * Object classification on the ImageNet and MS-COCO (multi-label)
 * Semantic Segmentation on the PASCAL VOC and the CityScapes
 * Object Detection on the PASCAL VOC and the MS-COCO
 * Supports PyTorch 1.0
 * Integrated with Tensorboard for easy visualization of training logs. 
 * Scripts for downloading different datasets.
 
## Blogs

 * [Faster Training for Efficient Networks](https://medium.com/p/faster-training-of-efficient-cnns-657953aa080?source=email-dc17ff22fa63--writer.postDistributed&sk=f60110289b6157de4c9e0c00c77f51e9)
 * [Semantic segmentation using ESPNetv2](https://medium.com/@sachinmehta.ngb/espnetv2-for-semantic-segmentation-9e80f155d522?source=friends_link&sk=91bca9326b088a972c170d1f7f5063e8)
 
 
## Image Classification
Details about training and testing are provided [here](README_Classification.md).

Details about performance of different models are provided [here](model/classification/model_zoo/README.md).

## Semantic segmentation
Details about training and testing are provided [here](README_Segmentation.md).

Details about performance of different models are provided [here](model/segmentation/model_zoo/README.md).


## Object Detection

Details about training and testing are provided [here](README_Detection.md).

## License
By downloading this software, you acknowledge that you agree to the terms and conditions given [here](License).


## Acknowledgements
Most of our object detection code is adapted from [SSD in pytorch](https://github.com/amdegroot/ssd.pytorch). We thank authors for such an amazing work.

