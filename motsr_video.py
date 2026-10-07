"""Recurrent MoTSR inference. The same runner is used by Colab and the CLI.

Copyright (c) 2026 xyether. MIT; upstream notices are in NOTICE.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import json
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

_TRT_LOGGER = None


def checked(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:] or result.stdout[-6000:])
    return result.stdout


class TorchBackend:
    """Portable reference backend; uses the released safetensors checkpoint."""
    def __init__(self, checkpoint, device=None):
        from safetensors.torch import load_file
        from motsr_arch import MoTSR
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = MoTSR(stage="partial").eval().to(self.device)
        self.model.load_state_dict(load_file(str(checkpoint)), strict=True)

    @torch.inference_mode()
    def seed(self, frame):
        return self.model.image_only(frame)

    @torch.inference_mode()
    def step(self, *frames):
        return self.model.forward_step(*frames)


class TRTEngine:
    def __init__(self, path, height, width):
        import tensorrt as trt
        global _TRT_LOGGER
        self.trt = trt
        if _TRT_LOGGER is None:
            _TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        self.logger = _TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot load the release engine. Select a T4 GPU and rerun setup with TensorRT 11.0.0.114.")
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.current_stream()
        profile = int(height > width)
        if not self.context.set_optimization_profile_async(profile, self.stream.cuda_stream):
            raise RuntimeError("TensorRT could not select the image orientation profile.")
        for index, name in enumerate(self.inputs):
            factor = 2 if len(self.inputs) == 7 and index in (1, 3) else 1
            if not self.context.set_input_shape(name, (1, 3, height * factor, width * factor)):
                raise ValueError("Frame dimensions exceed this engine's profile.")
        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"Unresolved TensorRT shapes: {unresolved}")
        self.output_name = self.outputs[0]
        shape = tuple(self.context.get_tensor_shape(self.output_name))
        dtype = torch.float32 if self.engine.get_tensor_dtype(self.output_name) == trt.float32 else torch.float16
        self.output = torch.empty(shape, dtype=dtype, device="cuda")
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())

    @torch.inference_mode()
    def __call__(self, *args):
        if len(args) != len(self.inputs):
            raise ValueError("Incorrect number of recurrent input tensors.")
        refs = []
        for name, tensor in zip(self.inputs, args):
            dtype = torch.float32 if self.engine.get_tensor_dtype(name) == self.trt.float32 else torch.float16
            tensor = tensor.to(dtype=dtype).contiguous()
            refs.append(tensor)
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed.")
        # Own the prediction: the engine's output buffer is overwritten next step.
        prediction = self.output.clone()
        self.stream.synchronize()
        return prediction


class TRTBackend:
    def __init__(self, model_dir, height, width):
        import tensorrt as trt
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
            raise RuntimeError("Choose Runtime → Change runtime type → T4 GPU, then rerun setup.")
        if trt.__version__ != "11.0.0.114":
            raise RuntimeError("Rerun the setup cell to install TensorRT 11.0.0.114.")
        self.device = torch.device("cuda")
        root = Path(model_dir)
        self.seed_engine = TRTEngine(root / "MoTSR_2x_seed_T4.engine", height, width)
        self.step_engine = TRTEngine(root / "MoTSR_2x_recurrent_T4.engine", height, width)
        if len(self.step_engine.inputs) != 7:
            raise RuntimeError("The recurrent engine must have seven inputs.")

    def seed(self, frame):
        return self.seed_engine(frame)

    def step(self, *frames):
        return self.step_engine(*frames)


def dimensions(width, height, speed_boost=True):
    # Keep full source resolution when it fits; only downscale above the limits.
    max_long, max_short = (1280, 720) if speed_boost else (1920, 1080)
    ratio = min(1.0, max_long / max(width, height), max_short / min(width, height))
    if ratio < 1:
        width, height = max(32, int(width * ratio) // 2 * 2), max(32, int(height * ratio) // 2 * 2)
    if min(width, height) < 32:
        raise ValueError("Use an input with both dimensions at least 32 pixels.")
    return width, height


def tensor_from_bgr(frame, width, height, device):
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_REFLECT_101)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255)


def bgr_from_tensor(prediction, height, width, force_1080p=False):
    rgb = prediction[0, :, :height * 2, :width * 2].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if force_1080p and (bgr.shape[1] > 1920 or bgr.shape[0] > 1080):
        ratio = min(1920 / bgr.shape[1], 1080 / bgr.shape[0])
        size = (int(bgr.shape[1] * ratio) // 2 * 2, int(bgr.shape[0] * ratio) // 2 * 2)
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return bgr


def video_timing(path):
    probe = json.loads(checked(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames", "-show_entries", "stream=width,height,start_time,duration,avg_frame_rate:frame=best_effort_timestamp_time,pkt_duration_time", "-of", "json", str(path)]))
    stream, frames = probe["streams"][0], probe["frames"]
    if not frames:
        raise ValueError("The input contains no video frames.")
    fps = float(Fraction(stream.get("avg_frame_rate", "25/1"))) or 25
    timestamps = [float(f.get("best_effort_timestamp_time", i / fps)) for i, f in enumerate(frames)]
    durations = [timestamps[i + 1] - timestamps[i] for i in range(len(frames) - 1)]
    last = float(frames[-1].get("pkt_duration_time", 1 / fps))
    durations.append(last if last > 0 else 1 / fps)
    if any(d <= 0 for d in durations):
        raise ValueError("Non-increasing video timestamps are unsupported. Remux the clip first.")
    return stream, timestamps, durations


def upscale_video(input_path, output_path, model_dir="models", backend="trt", checkpoint=None,
                  speed_boost=True, force_1080p=False, codec="hevc_nvenc", quality=6,
                  reset_on_cuts=True, keep_frames=None, progress=True):
    if not 0 <= quality <= 10:
        raise ValueError("CRF / quality must be between 0 and 10.")
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output:
        raise ValueError("Choose a different output filename.")
    if codec not in ("h264_nvenc", "hevc_nvenc", "libx264"):
        raise ValueError("Unsupported encoder.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stream, timestamps, durations = video_timing(source)
    width, height = dimensions(int(stream["width"]), int(stream["height"]), speed_boost)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="motsr-") as temp:
        root = Path(temp); inputs = root / "input"; results = root / "output"
        inputs.mkdir(); results.mkdir()
        # No fps filter: decode one image per original frame.
        checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0", "-fps_mode", "passthrough", str(inputs / "%08d.png")])
        paths = sorted(inputs.glob("*.png"))
        if len(paths) != len(timestamps):
            raise RuntimeError(f"Decoded {len(paths)} frames, but timestamps describe {len(timestamps)}.")
        n = len(paths)
        cuts = [0]
        previous = None
        for i, path in enumerate(paths):
            thumbnail = cv2.resize(cv2.imread(str(path)), (64, 36)).astype(np.float32) / 255
            if reset_on_cuts and previous is not None and np.abs(thumbnail - previous).mean() > 0.30:
                cuts.append(i)
            previous = thumbnail
        cuts.append(n)
        runner = TRTBackend(model_dir, height + height % 2, width + width % 2) if backend == "trt" else TorchBackend(checkpoint or Path(model_dir) / "MoTSR_2x.safetensors")
        bar = tqdm(total=n, desc="✨ MoTSR", disable=not progress)
        with torch.inference_mode():
            for left, right in zip(cuts, cuts[1:]):
                cache = {}
                def frame(index):
                    index = min(max(index, left), right - 1)
                    if index not in cache:
                        cache[index] = tensor_from_bgr(cv2.imread(str(paths[index])), width, height, runner.device)
                    return cache[index]
                older_hr = previous_hr = runner.seed(frame(left))
                for i in range(left, right):
                    lr = [frame(i + offset) for offset in (-2, -1, 0, 1, 2)]
                    prediction = runner.step(lr[0], older_hr, lr[1], previous_hr, lr[2], lr[3], lr[4])
                    if not torch.isfinite(prediction).all().item():
                        raise RuntimeError(f"Non-finite prediction at frame {i}; output was not encoded.")
                    # Reuse raw floating-point SR, before clamping, resizing or encoding.
                    older_hr, previous_hr = previous_hr, prediction
                    pixels = bgr_from_tensor(prediction, height, width, force_1080p)
                    if not cv2.imwrite(str(results / f"{i:08d}.png"), pixels):
                        raise RuntimeError("Failed to write an output frame. Check runtime disk space.")
                    cache = {k: v for k, v in cache.items() if k >= i - 1}
                    bar.update(1)
        bar.close()
        manifest = root / "frames.ffconcat"
        rows = ["ffconcat version 1.0"]
        for i, duration in enumerate(durations):
            rows += [f"file '{(results / f'{i:08d}.png').as_posix()}'", "option framerate 90000", f"duration {duration:.9f}"]
        # A terminal packet records the last real frame's hold. Remove it by
        # stream-copy afterwards; no extra model prediction enters the result.
        rows += [f"file '{(results / f'{n - 1:08d}.png').as_posix()}'", "option framerate 90000"]
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        encoded = root / "encoded.mp4"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest), "-map", "0:v:0", "-c:v", codec]
        if codec.endswith("_nvenc"):
            command += ["-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
        else:
            command += ["-preset", "medium", "-crf", str(quality)]
        command += ["-bf", "0", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-enc_time_base", "1:90000", "-video_track_timescale", "90000", "-map_metadata", "-1", str(encoded)]
        if codec == "hevc_nvenc":
            command[-1:-1] = ["-tag:v", "hvc1"]
        checked(command)
        checked(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(encoded), "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-frames:v", str(n), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-video_track_timescale", "90000", "-map_metadata", "-1", str(output)])
        out_stream, out_timestamps, out_durations = video_timing(output)
        if len(out_timestamps) != n:
            raise RuntimeError(f"Encoder changed frame count: {n} → {len(out_timestamps)}.")
        timing_error = max(abs((a - out_timestamps[0]) - (b - timestamps[0])) for a, b in zip(out_timestamps, timestamps))
        if timing_error > 0.002 or abs(sum(out_durations) - sum(durations)) > 0.002:
            raise RuntimeError("Encoded video did not preserve source frame timing within 2 ms.")
        if keep_frames:
            import shutil
            shutil.copytree(results, keep_frames, dirs_exist_ok=True)
        report = {"input_frames": n, "output_frames": len(out_timestamps), "input_size": [int(stream["width"]), int(stream["height"])], "model_input_size": [width, height], "output_size": [int(out_stream["width"]), int(out_stream["height"])], "duration_seconds": sum(out_durations), "max_timestamp_error_seconds": timing_error, "scene_resets": len(cuts) - 1, "codec": codec, "seconds": round(time.monotonic() - started, 3), "backend": backend}
        output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"✅ Saved {output.name} · {n} frames · {out_stream['width']}×{out_stream['height']}")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--backend", choices=["torch", "trt"], default="torch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--full-resolution", action="store_true")
    parser.add_argument("--force-1080p", action="store_true")
    parser.add_argument("--codec", choices=["h264_nvenc", "hevc_nvenc", "libx264"], default="h264_nvenc")
    args = parser.parse_args()
    upscale_video(args.input, args.output, args.model_dir, args.backend, args.checkpoint, not args.full_resolution, args.force_1080p, args.codec)


if __name__ == "__main__":
    main()
