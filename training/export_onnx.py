import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10
DEVICE = "cpu"   # export on CPU

# Load model architecture
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Load trained weights
model.load_state_dict(
    torch.load("../models/mobilenetv2_cifar10.pth", map_location=DEVICE)
)

model.eval()

# Dummy input (batch size 1)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "../models/mobilenetv2_cifar10.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("ONNX export complete.")
