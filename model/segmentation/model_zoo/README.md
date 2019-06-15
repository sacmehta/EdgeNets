# Results on semantic segmentation datasets


## DiCENet models

### PASCAL VOC

| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.2 | 256x256 | 60.98 | 0.05 | 33.77 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.2_pascal_256x256.pth) |
| 0.5 | 256x256 | 82.33 | 0.08 | 42.48 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.5_pascal_256x256.pth) |
| 0.75 | 256x256 | 127.22 | 0.19 | 49.52 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.75_pascal_256x256.pth) |
| 1.0 | 256x256 | 141.54 | 0.28 | 54.31 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.0_pascal_256x256.pth) |
| 1.25 | 256x256 | 155.39 | 0.39 | 57.59 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.25_pascal_256x256.pth) |
| 1.75 | 256x256 | 289.96 | 0.81 | 58.39 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.75_pascal_256x256.pth) |


| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.2 | 384x384 | 135.64 | 0.05 | 34.52 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.2_pascal_384x384.pth) |
| 0.5 | 384x384 | 182.91 | 0.08 | 47.74 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.5_pascal_384x384.pth) |
| 0.75 | 384x384 | 282.05 | 0.19 | 55.67 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.75_pascal_384x384.pth) |
| 1.0 | 384x384 | 312.76 | 0.28 | 60.23 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.0_pascal_384x384.pth) |
| 1.25 | 384x384 | 342.52 | 0.39 | 62.55 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.25_pascal_384x384.pth) |
| 1.75 | 384x384 | 641.98 | 0.81 | 64.76 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.75_pascal_384x384.pth) |
| -- | 384x384 | 680 | -- | 66.50 | [Leaderboard](http://host.robots.ox.ac.uk:8080/anonymous/Q8DARH.html) |
| -- | 384x384 | 680 | -- | 67.31 (test) | [Leaderboard](http://host.robots.ox.ac.uk:8080/anonymous/T44DHQ.html) |


### Cityscapes

| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.5 | 512x256 | 162.77 | 0.08 | 47.3 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.5_city_512x256.pth) |
| 0.75 | 512x256 | 251.29 | 0.19 | 50.0 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.75_city_512x256.pth) |
| 1.0 | 512x256 | 278.92 | 0.28 | 51.2 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.0_city_512x256.pth) |
| 1.25 | 512x256 | 305.68 | 0.39 | 51.6 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.25_city_512x256.pth) |
| 1.5 | 512x256 | 376.41 | 0.55 | 52.1 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.5_city_512x256.pth) |
| 1.75 | 512x256 | 572.59 | 0.81 | 54.2 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.75_city_512x256.pth) |

| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.5 | 1024x512 | 641.68 | 0.08 | 53.4 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.5_city_1024x512.pth) |
| 0.75 | 1024x512 | 988.39 | 0.19 | 58.7 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_0.75_city_1024x512.pth) |
| 1.0 | 1024x512 | 1093.01 | 0.28 | 60.9 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.0_city_1024x512.pth) |
| 1.25 | 1024x512 | 1194.53 | 0.39 | 61.1 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.25_city_1024x512.pth) |
| 1.5 | 1024x512 | 1471.11 | 0.55 | 62.2 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.5_city_1024x512.pth) |
| 1.75 | 1024x512 | 2249.06 | 0.81 | 63.4 | [here](/model/segmentation/model_zoo/dicenet/dicenet_s_1.75_city_1024x512.pth) |


## ESPNetv2 models


### PASCAL VOC

| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.5 | 256x256 |  74.88 | 0.08 |  39.82 | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_0.5_pascal_256x256.pth) |
| 1.0 | 256x256 | 136.0 | 0.23 |  54.14  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.0_pascal_256x256.pth) |
| 1.5 | 256x256 | 214.51 | 0.47 |  59.39  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.5_pascal_256x256.pth) |
| 2. 0 | 256x256 | 337.58 | 0.79 |  63.36  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_2.0_pascal_256x256.pth) |

| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.5 | 384x384 |  168.46 | 0.08 | 47.18 | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_0.5_pascal_384x384.pth) |
| 1.0 | 384x384 | 306.00 | 0.23 | 60.22  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.0_pascal_384x384.pth) |
| 1.5 | 384x384 | 482.63 | 0.47 | 63.67 | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.5_pascal_384x384.pth) |
| 2. 0 | 384x384 | 759.52 | 0.79 | 67.01  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_2.0_pascal_384x384.pth)|
| 2. 0 | 384x384 | 759.52 | 0.79 | 67.95 (test) | [Leaderboard](http://host.robots.ox.ac.uk:8080/anonymous/DAMVRR.html) |

### Cityscapes

| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.5 | 512x256 | 149.40 | 0.08 | 47.3 | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_0.5_city_512x256.pth) |
| 1.0 | 512x256 | 271.66 | 0.23 | 52.2  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.0_city_512x256.pth) |
| 1.5 | 512x256 | 428.66 | 0.47 |  54.6  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.5_city_512x256.pth) |
<!--
| 2. 0 | 512x256 | 674.78 | 0.79 | 50.4 (val) | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_512x256.pth) |
-->
| s | Image Size | FLOPs (in million) | Params (in million) | mIOU (class-wise) | Link |
|---|---|---|---|---|---|
| 0.5 | 1024x512 |  597.56 | 0.08 |  53.9  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_0.5_city_1024x512.pth)|
| 1.0 | 1024x512 | 1086.59 | 0.23 |  60.1 | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.0_city_1024x512.pth)|
| 1.5 | 1024x512 | 1714.57 | 0.47 | 63.8  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_1.5_city_1024x512.pth)|
| 2.0 | 1024x512 | 2699.05 | 0.79 | 66.4  | [here](/model/segmentation/model_zoo/espnetv2/espnetv2_s_2.0_city_1024x512.pth)|
| 2.0 | 1024x512 | 2699.05 | 0.79 | 66.2 (test) | [Leaderboard](https://www.cityscapes-dataset.com/benchmarks/) |

