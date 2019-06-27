This folder contains ONNX compatible models for ESPNetv2.

The file name follows this format:
```
<model>_s_<network-width-multiplier>_imsize_<input-image-resolution>_<dataset>.onnx
```

For example, `espnetv2_s_2.0_imsize_224x224_imagenet.onnx` indicates that 
 * model is `espnetv2`
 * network width multiplies is `2.0`
 * image size is `224x224`
 * Dataset is `ImageNet`
