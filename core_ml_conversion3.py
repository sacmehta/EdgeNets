# Paste Python code snippet here, complete with any required import statements.
import coremltools as ct
import torch
import torchvision

# Load PyTorch model (and perform tracing)
torch_model = torchvision.models.mobilenet_v2()
torch_model.eval() 

example_input = torch.rand(1, 3, 256, 256)
traced_model = torch.jit.trace(torch_model, example_input)

# Convert using the same API. Note that we need to provide "inputs" for pytorch conversion.
model_from_torch = ct.convert(traced_model,
                              inputs=[ct.TensorType(name="input", 
                                                    shape=example_input.shape)],
                              debug=True)
model_from_torch.save('core_ml_exp/model.mlpackage')