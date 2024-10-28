# Paste Python code snippet here, complete with any required import statements.
import coremltools as ct
import torch
import torchvision

# Load PyTorch model (and perform tracing)
torch_model = torchvision.models.mobilenet_v2()
torch_model.eval() 

example_input = torch.rand(1, 3, 256, 256)
# traced_model = torch.jit.trace(torch_model, example_input)
torch.onnx.export(torch_model, example_input, "core_ml_exp2/simple_model.onnx")

# Convert the ONNX model to CoreML
onnx_model_path = "core_ml_exp2/simple_model.onnx"
coreml_model = ct.converters.onnx.convert(model=onnx_model_path)
coreml_model.save("core_ml_exp2/simple_model.mlmodel")