"""Offline release-engine build. Colab users download prebuilt engines instead."""
import argparse
from pathlib import Path
import onnx
from onnxconverter_common import float16
import tensorrt as trt
import torch


def build(source, destination):
    logger = trt.Logger(trt.Logger.WARNING)
    graph = onnx.load(str(source))
    onnx.checker.check_model(graph)
    graph = float16.convert_float_to_float16(graph, keep_io_types=True)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(graph.SerializeToString()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
    for portrait in (False, True):
        profile = builder.create_optimization_profile()
        opt = (640, 360) if portrait else (360, 640)
        maximum = (1920, 1080) if portrait else (1080, 1920)
        for i in range(network.num_inputs):
            name = network.get_input(i).name
            factor = 2 if network.num_inputs == 7 and i in (1, 3) else 1
            profile.set_shape(name, (1, 3, 32 * factor, 32 * factor), (1, 3, opt[0] * factor, opt[1] * factor), (1, 3, maximum[0] * factor, maximum[1] * factor))
        config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine construction failed.")
    destination.write_bytes(bytes(engine))
    print("Built", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("Build the release engines on a T4 GPU.")
    if trt.__version__ != "11.0.0.114":
        raise RuntimeError("Release engines require TensorRT 11.0.0.114.")
    for name in ("seed", "recurrent"):
        build(args.model_dir / f"Tmosr2_2x_{name}.onnx", args.model_dir / f"Tmosr2_2x_{name}_T4.engine")


if __name__ == "__main__":
    main()
