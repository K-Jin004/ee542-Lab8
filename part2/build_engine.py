import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path):
    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, LOGGER)

    if not parser.parse_from_file(onnx_path):
        print(f"Failed to parse {onnx_path}")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return

    config = builder.create_builder_config()

    # Dynamic batch profile: 1 to 32
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(32, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    print(f"Building {engine_path}...")

    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("Engine build failed")
        return

    with open(engine_path, "wb") as f:
        f.write(engine)

    print(f"Saved {engine_path}")


build_engine("resnet18_fp32.onnx", "resnet18_fp32.engine")
build_engine("resnet18_fp16.onnx", "resnet18_fp16.engine")
