# Semantic Segmentation

## Dataset

### PASCAL VOC 2012
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
 
### Cityscapes dataset
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


## Training and Testing

### Training


### Testing

You can test the 
```
CUDA_VISIBLE_DEVICES=0 python test_segmentation.py --model espnetv2 --s 2.0 --dataset city --data-path ../vision_datasets/cityscapes/ --split val --im-size 1024 512
```