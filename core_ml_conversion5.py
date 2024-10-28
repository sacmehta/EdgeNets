import torch
import torchvision
from onnx_coreml import convert
import torch

mlmodel = convert(model='./onnx_models/classification/espnetv2/espnetv2_s_1.0_imsize_224x224_imagenet.onnx', 
                  minimum_ios_deployment_target='13')

# Traceback (most recent call last):
#   File "/home/ubuntu/ML/EdgeNets/core_ml_conversion5.py", line 3, in <module>
#     from onnx_coreml import convert
#   File "/home/ubuntu/anaconda3/envs/edgenets3/lib/python3.9/site-packages/onnx_coreml/__init__.py", line 6, in <module>
#     from .converter import convert
#   File "/home/ubuntu/anaconda3/envs/edgenets3/lib/python3.9/site-packages/onnx_coreml/converter.py", line 35, in <module>
#     from coremltools.converters.nnssa.coreml.graph_pass.mlmodel_passes import remove_disconnected_layers, transform_conv_crop
# ModuleNotFoundError: No module named 'coremltools.converters.nnssa'