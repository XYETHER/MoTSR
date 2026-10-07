"""Export the spatial seed and seven-input recurrent graphs from the checkpoint."""
import argparse
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motsr_arch import MoTSR


class Seed(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, frame):
        return self.model.image_only(frame)


class Recurrent(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, older_lr, older_hr, previous_lr, previous_hr, current_lr, next_lr, later_lr):
        return self.model.forward_step(older_lr, older_hr, previous_lr, previous_hr, current_lr, next_lr, later_lr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = MoTSR().eval()
    model.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    lr = torch.rand(1, 3, 64, 96)
    with torch.inference_mode():
        hr = model.image_only(lr)
    torch.onnx.export(Seed(model), lr, args.output_dir / "MoTSR_2x_seed.onnx", input_names=["input"], output_names=["output"], dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "out_height", 3: "out_width"}}, opset_version=17, dynamo=False)
    names = ["older_lr", "older_hr", "previous_lr", "previous_hr", "current_lr", "next_lr", "later_lr"]
    axes = {name: {2: "hr_height" if i in (1, 3) else "lr_height", 3: "hr_width" if i in (1, 3) else "lr_width"} for i, name in enumerate(names)}
    axes["output"] = {2: "out_height", 3: "out_width"}
    torch.onnx.export(Recurrent(model), (lr, hr, lr, hr, lr, lr, lr), args.output_dir / "MoTSR_2x_recurrent.onnx", input_names=names, output_names=["output"], dynamic_axes=axes, opset_version=17, dynamo=False)
    print("Exported both ONNX graphs.")


if __name__ == "__main__":
    main()
