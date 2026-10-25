import time
import torch
import torchvision
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)

BATCH_SIZES = [1, 8, 32]
WARMUP = 20
RUNS = 100


def load_engine(path):
    runtime = trt.Runtime(LOGGER)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return runtime, engine


def run_trt(engine, inp, dtype):
    batch = inp.shape[0]

    context = engine.create_execution_context()
    context.set_input_shape("input", (batch, 3, 224, 224))

    out = torch.empty(
        batch,
        1000,
        device="cuda",
        dtype=dtype
    )

    context.set_tensor_address("input", inp.data_ptr())
    context.set_tensor_address("logits", out.data_ptr())

    # Non-default CUDA stream
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        context.execute_async_v3(stream.cuda_stream)

    stream.synchronize()

    return out


def benchmark_pytorch(model, inp):
    # Warm-up
    with torch.no_grad():
        for _ in range(WARMUP):
            model(inp)

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(RUNS):
            model(inp)

    torch.cuda.synchronize()

    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / RUNS
    return avg_ms


def benchmark_trt(engine, inp, dtype):
    batch = inp.shape[0]

    context = engine.create_execution_context()
    context.set_input_shape("input", (batch, 3, 224, 224))

    out = torch.empty(
        batch,
        1000,
        device="cuda",
        dtype=dtype
    )

    context.set_tensor_address("input", inp.data_ptr())
    context.set_tensor_address("logits", out.data_ptr())

    stream = torch.cuda.Stream()

    # Warm-up
    for _ in range(WARMUP):
        context.execute_async_v3(stream.cuda_stream)

    stream.synchronize()

    start = time.perf_counter()

    for _ in range(RUNS):
        context.execute_async_v3(stream.cuda_stream)

    stream.synchronize()

    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / RUNS
    return avg_ms, out


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

print("Loading PyTorch ResNet-18...")

model_fp32 = torchvision.models.resnet18(
    weights="DEFAULT"
).eval().cuda()

runtime32, engine32 = load_engine("resnet18_fp32.engine")
runtime16, engine16 = load_engine("resnet18_fp16.engine")


# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

print()
print("=" * 100)
print(
    f"{'Batch':<8}"
    f"{'PyTorch FP32(ms)':<20}"
    f"{'TRT FP32(ms)':<18}"
    f"{'TRT FP16(ms)':<18}"
    f"{'FP32 Speedup':<16}"
    f"{'FP16 Speedup':<16}"
)
print("=" * 100)

for batch in BATCH_SIZES:

    # Same FP32 random input used for PyTorch and TensorRT FP32
    inp_fp32 = torch.randn(
        batch,
        3,
        224,
        224,
        device="cuda",
        dtype=torch.float32
    )

    # FP16 version of the same data
    inp_fp16 = inp_fp32.half()

    # ----------------------
    # PyTorch reference
    # ----------------------
    with torch.no_grad():
        pytorch_out = model_fp32(inp_fp32)

    # ----------------------
    # TensorRT correctness
    # ----------------------
    trt32_out = run_trt(
        engine32,
        inp_fp32,
        torch.float32
    )

    trt16_out = run_trt(
        engine16,
        inp_fp16,
        torch.float16
    )

    # ----------------------
    # Correctness metrics
    # ----------------------
    diff32 = (
        pytorch_out - trt32_out
    ).abs().max().item()

    diff16 = (
        pytorch_out - trt16_out.float()
    ).abs().max().item()

    pytorch_top1 = pytorch_out.argmax(dim=1)

    trt32_top1 = trt32_out.argmax(dim=1)
    trt16_top1 = trt16_out.argmax(dim=1)

    agreement32 = (
        pytorch_top1 == trt32_top1
    ).float().mean().item() * 100

    agreement16 = (
        pytorch_top1 == trt16_top1
    ).float().mean().item() * 100

    # ----------------------
    # Timing
    # ----------------------
    pytorch_ms = benchmark_pytorch(
        model_fp32,
        inp_fp32
    )

    trt32_ms, _ = benchmark_trt(
        engine32,
        inp_fp32,
        torch.float32
    )

    trt16_ms, _ = benchmark_trt(
        engine16,
        inp_fp16,
        torch.float16
    )

    # ----------------------
    # Speedup / throughput
    # ----------------------
    speedup32 = pytorch_ms / trt32_ms
    speedup16 = pytorch_ms / trt16_ms

    throughput_pt = batch / (pytorch_ms / 1000)
    throughput_32 = batch / (trt32_ms / 1000)
    throughput_16 = batch / (trt16_ms / 1000)

    print(
        f"{batch:<8}"
        f"{pytorch_ms:<20.4f}"
        f"{trt32_ms:<18.4f}"
        f"{trt16_ms:<18.4f}"
        f"{speedup32:<16.2f}"
        f"{speedup16:<16.2f}"
    )

    print(
        f"  PyTorch throughput : {throughput_pt:.2f} images/s"
    )

    print(
        f"  TRT FP32 throughput: {throughput_32:.2f} images/s"
    )

    print(
        f"  TRT FP16 throughput: {throughput_16:.2f} images/s"
    )

    print(
        f"  FP32 max abs diff  : {diff32:.6f}"
    )

    print(
        f"  FP16 max abs diff  : {diff16:.6f}"
    )

    print(
        f"  FP32 top-1 agreement: {agreement32:.2f}%"
    )

    print(
        f"  FP16 top-1 agreement: {agreement16:.2f}%"
    )

    print("-" * 100)
