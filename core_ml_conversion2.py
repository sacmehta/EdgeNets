import torch
import torch.onnx
import coremltools as ct

# Other guide: https://github.com/vincentfpgarcia/from-pytorch-to-coreml

# Define a simple PyTorch model for demonstration
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(16*112*112, 10)  # Assuming input size is (3, 224, 224)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# Instantiate and export the model
pytorch_model = SimpleModel()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(pytorch_model, dummy_input, "simple_model.onnx")

# Convert the ONNX model to CoreML
onnx_model_path = "simple_model.onnx"
coreml_model = ct.converters.onnx.convert(model=onnx_model_path)
coreml_model.save("simple_model.mlmodel")
