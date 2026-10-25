import os
import time

import torch
import tensorrt as trt


# ============================================================
# Configuration
# ============================================================

NUM_IMAGES = 16
HEIGHT = 224
WIDTH = 224
CHANNELS = 3

ENGINE_PATH = "resnet18_fp16.engine"

ORIGINAL_DIR = "original_images"
RECONSTRUCTED_DIR = "reconstructed"

LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# Load TensorRT Engine
# ============================================================

def load_engine(path):
    runtime = trt.Runtime(LOGGER)

    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("Failed to load TensorRT engine")

    return runtime, engine


# ============================================================
# Load one raw RGB image into PINNED CPU memory
# ============================================================

def load_raw_image_pinned(path):
    """
    Read the raw 224x224x3 uint8 image from disk.

    The returned tensor is allocated in pinned CPU memory.
    """

    expected_bytes = HEIGHT * WIDTH * CHANNELS

    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes, "
            f"got {len(raw)}"
        )

    # Allocate pinned CPU tensor
    cpu_tensor = torch.empty(
        expected_bytes,
        dtype=torch.uint8,
        pin_memory=True
    )

    # Convert file bytes into a temporary CPU tensor
    src = torch.frombuffer(
        bytearray(raw),
        dtype=torch.uint8
    )

    cpu_tensor.copy_(src)

    # HWC layout
    cpu_tensor = cpu_tensor.view(
        HEIGHT,
        WIDTH,
        CHANNELS
    )

    return cpu_tensor


# ============================================================
# CPU -> GPU copy timing
# ============================================================

def copy_to_gpu(cpu_tensor):
    """
    Copy pinned CPU memory to GPU asynchronously
    and measure the transfer time with CUDA events.
    """

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):

        start.record(stream)

        gpu_uint8 = cpu_tensor.to(
            device="cuda",
            non_blocking=True
        )

        end.record(stream)

    end.synchronize()

    elapsed_ms = start.elapsed_time(end)

    return gpu_uint8, elapsed_ms


# ============================================================
# GPU preprocessing
# ============================================================

def preprocess_on_gpu(gpu_uint8):
    """
    Input:
        H x W x C uint8

    Output:
        1 x C x H x W FP16

    Preprocessing follows ImageNet ResNet normalization.
    """

    # HWC -> CHW
    x = gpu_uint8.permute(2, 0, 1)

    # uint8 [0,255] -> float [0,1]
    x = x.to(torch.float32) / 255.0

    # ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device="cuda",
        dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device="cuda",
        dtype=torch.float32
    ).view(3, 1, 1)

    x = (x - mean) / std

    # Add batch dimension
    x = x.unsqueeze(0)

    # TensorRT engine is FP16
    x = x.half()

    # Ensure contiguous memory layout
    x = x.contiguous()

    return x


# ============================================================
# TensorRT inference
# ============================================================

def infer(engine, inp):
    context = engine.create_execution_context()

    context.set_input_shape(
        "input",
        tuple(inp.shape)
    )

    output = torch.empty(
        (inp.shape[0], 1000),
        device="cuda",
        dtype=torch.float16
    )

    context.set_tensor_address(
        "input",
        inp.data_ptr()
    )

    context.set_tensor_address(
        "logits",
        output.data_ptr()
    )

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        ok = context.execute_async_v3(
            stream.cuda_stream
        )

    if not ok:
        raise RuntimeError("TensorRT inference failed")

    stream.synchronize()

    return output


# ============================================================
# Classify one file
# ============================================================

def classify_file(engine, path):

    cpu_tensor = load_raw_image_pinned(path)

    gpu_uint8, copy_ms = copy_to_gpu(cpu_tensor)

    inp = preprocess_on_gpu(gpu_uint8)

    logits = infer(engine, inp)

    top1 = logits.argmax(dim=1).item()

    return top1, copy_ms, logits


# ============================================================
# Main
# ============================================================

def main():

    runtime, engine = load_engine(ENGINE_PATH)

    total_copy_ms = 0.0

    matched = 0

    print("=" * 90)
    print(
        f"{'Frame':<10}"
        f"{'Original':<15}"
        f"{'Reconstructed':<18}"
        f"{'Match':<10}"
        f"{'Copy Time (ms)':<15}"
    )
    print("=" * 90)

    for frame_id in range(NUM_IMAGES):

        original_path = os.path.join(
            ORIGINAL_DIR,
            f"frame_{frame_id:02d}.bin"
        )

        reconstructed_path = os.path.join(
            RECONSTRUCTED_DIR,
            f"frame_{frame_id:02d}.bin"
        )

        original_top1, _, original_logits = classify_file(
            engine,
            original_path
        )

        reconstructed_top1, copy_ms, reconstructed_logits = classify_file(
            engine,
            reconstructed_path
        )

        same_class = (
            original_top1 == reconstructed_top1
        )

        if same_class:
            matched += 1

        total_copy_ms += copy_ms

        max_diff = (
            original_logits.float()
            - reconstructed_logits.float()
        ).abs().max().item()

        print(
            f"{frame_id:<10}"
            f"{original_top1:<15}"
            f"{reconstructed_top1:<18}"
            f"{str(same_class):<10}"
            f"{copy_ms:<15.6f}"
        )

        print(
            f"           Max logit difference: "
            f"{max_diff:.8f}"
        )

    avg_copy_ms = total_copy_ms / NUM_IMAGES

    print("=" * 90)

    print(
        f"Classification agreement: "
        f"{matched}/{NUM_IMAGES}"
    )

    print(
        f"Average pinned CPU -> GPU copy time: "
        f"{avg_copy_ms:.6f} ms"
    )

    print()

    if matched == NUM_IMAGES:
        print(
            "SUCCESS: Every reconstructed image gets "
            "the same class as its original."
        )
    else:
        print(
            "WARNING: Some reconstructed images "
            "have different predictions."
        )


if __name__ == "__main__":
    main()
