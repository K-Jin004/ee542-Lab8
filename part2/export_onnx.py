import torch
import torchvision

# Load pretrained ResNet-18
model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()

# -------------------------
# FP32 export
# -------------------------
x_fp32 = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)

torch.onnx.export(
    model,
    x_fp32,
    "resnet18_fp32.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=18,
)

print("Exported resnet18_fp32.onnx")

# -------------------------
# FP16 export
# -------------------------
model_fp16 = model.half()

x_fp16 = torch.randn(
    1, 3, 224, 224,
    device="cuda",
    dtype=torch.float16
)

torch.onnx.export(
    model_fp16,
    x_fp16,
    "resnet18_fp16.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=18,
)

print("Exported resnet18_fp16.onnx")
