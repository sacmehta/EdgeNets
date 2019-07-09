# Results on object detection datasets


## ESPNetv2 models

### MS-COCO

| s | Data | Image Size | FLOPs (in billion) | Params (in million) | mAP @ 0.50:0.95 | map @ 0.50 | mAP @ 0.75| Link |
|---|---|---|---|---|---| --- | --- | --- |
| 2.0 | COCO | 300x300 | 1.1 | 5.81 | 20.80 | 37.05 | 20.36 | [here](/model/detection/model_zoo/espnetv2/espnetv2_s_2.0_coco_300x300.pth) |
| 2.0 | COCO | 512x512 | 3.2 | 5.81 | 24.54 | 42.04 | 25.21 | [here](/model/detection/model_zoo/espnetv2/espnetv2_s_2.0_coco_512x512.pth) |


### PASCAL VOC 2007
Below table compares the results on the PASCAL VOC 2007 dataset. It is important to NOTE that we do not use COCO data for below models.

| s | Data | Image Size | FLOPs (in billion) | Params (in million) | mAP | Link |
|---|---|---|---|---|---| --- |
| 2.0 | 07+12 | 300x300 | 0.9 | 4.75 | 70.27 | [here](/model/detection/model_zoo/espnetv2/espnetv2_s_2.0_pascal_300x300.pth) |
| 2.0 | 07+12 | 512x512 | 2.5 | 4.75 | 75.01 | [here](/model/detection/model_zoo/espnetv2/espnetv2_s_2.0_pascal_512x512.pth) |


See below for class-wise mAP

|             | 512x512 | 300x300 | 
|-------------|---------|---------|
| aeroplane   | 0.7793  | 0.7465  | 
| bicycle     | 0.8199  | 0.8113  | 
| bird        | 0.7269  | 0.6517  | 
| boat        | 0.638   | 0.5972  | 
| bottle      | 0.4414  | 0.3506  | 
| bus         | 0.8376  | 0.78    | 
| car         | 0.8416  | 0.7865  | 
| cat         | 0.8533  | 0.829   | 
| chair       | 0.5915  | 0.5151  | 
| cow         | 0.802   | 0.7194  | 
| diningtable | 0.7317  | 0.722   | 
| dog         | 0.819   | 0.7653  | 
| horse       | 0.8575  | 0.8389  | 
| motorbike   | 0.8179  | 0.7986  | 
| person      | 0.7803  | 0.7214  | 
| pottedplant | 0.5353  | 0.4697  | 
| sheep       | 0.7553  | 0.6836  | 
| sofa        | 0.7884  | 0.7414  | 
| train       | 0.8449  | 0.8073  | 
| tvmonitor   | 0.7407  | 0.7166  | 
| mAP         | 0.7501  | 0.7026  | 




