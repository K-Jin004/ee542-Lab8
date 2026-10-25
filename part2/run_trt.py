import torch
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)

def run_engine(engine_path, batch, dtype):
    # 1. Load TensorRT engine
    runtime = trt.Runtime(LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # 2. Set dynamic input shape
    context.set_input_shape("input", (batch, 3, 224, 224))

    # 3. Allocate input/output directly on GPU
    inp = torch.randn(
        batch, 3, 224, 224,
        device="cuda",
        dtype=dtype
    )

    out = torch.empty(
        batch, 1000,
        device="cuda",
        dtype=dtype
    )

    # 4. Give TensorRT the GPU memory addresses
    context.set_tensor_address("input", inp.data_ptr())
    context.set_tensor_address("logits", out.data_ptr())

    # 5. Use current PyTorch CUDA stream
    stream = torch.cuda.current_stream()

    # 6. Run inference
    ok = context.execute_async_v3(stream.cuda_stream)

    if not ok:
        raise RuntimeError("TensorRT inference failed")

    # Wait for GPU work to finish before reading result
    stream.synchronize()

    return inp, out


# FP32 test
inp32, out32 = run_engine(
    "resnet18_fp32.engine",
    batch=1,
    dtype=torch.float32
)

print("FP32:")
print("Input shape:", inp32.shape)
print("Output shape:", out32.shape)
print("Top-1 class:", out32.argmax(dim=1).item())


# FP16 test
inp16, out16 = run_engine(
    "resnet18_fp16.engine",
    batch=1,
    dtype=torch.float16
)

print("\nFP16:")
print("Input shape:", inp16.shape)
print("Output shape:", out16.shape)
print("Top-1 class:", out16.argmax(dim=1).item())
