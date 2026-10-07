# 🎥 MoTSR

**2× temporal upscaling for real-life videos — made by [xyether](https://github.com/XYETHER) 🤍**

MoTSR is a 2× model for IRL videos. It uses nearby frames and feedback from previous results to upscale each frame.

Built on **[MoSRv2 by umzi2](https://github.com/umzi2/MoSRV2)**, with temporal fusion and recurrent feedback added by xyether.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XYETHER/MoTSR/blob/main/MoTSR.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> 🧪 Still being developed. Results aren't perfect and will vary depending on the video.

## 👀 See it in action

**720p → 1440p**

![Input and MoTSR output](examples/comparison.png)

### A closer look

The input on the left is resized to match the output's size.

![Detail comparison](examples/detail.png)

| Original clip | MoTSR result |
| :---: | :---: |
| [▶️ Watch / download input](https://github.com/XYETHER/MoTSR/releases/download/v1.0.0/input_preview.mp4) | [▶️ Watch / download 2× output](https://github.com/XYETHER/MoTSR/releases/download/v1.0.0/MoTSR_preview.mp4) |

## 🏋️ Training

I trained MoTSR using **[traiNNer-redux](https://github.com/the-database/traiNNer-redux)** and **[4K-VFHQ-Tiny](https://huggingface.co/datasets/XYETHER/4K-VFHQ-Tiny)**, a dataset I created.

## 🚀 Try it in Colab

1. Open the **[Colab notebook](https://colab.research.google.com/github/XYETHER/MoTSR/blob/main/MoTSR.ipynb)**.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run **✨ Setup Environment**, then **🚀 Upscale**.
4. Upload your video and download the result.

### 🎛️ What do the settings mean?

| Setting | What it does |
| --- | --- |
| `video_path` | Leave it empty to upload, or paste the path of a video already in Colab. |
| `Speed_Boost` | Keeps the model input within 720p. Turn it off to use input up to 1080p. Larger inputs are resized first. |
| `Force_1080p` | Resizes the finished video to fit 1920×1080. |
| `codec` | H.264 works in more players; H.265 is another compression option. Both use GPU encoding. |
| `crf_value` | Default **6**, maximum **10**. Lower values mean less compression and bigger files. This controls NVENC's constant-quality setting. |
| `auto_download` | Starts downloading the finished video automatically. |

💡 The model produces **2× the resolution it receives**. Speed Boost can reduce large inputs first. Audio is kept as AAC when present. Processing uses temporary PNG frames, so long videos need enough runtime disk space.

## 📦 Model downloads

Get the **[weights, ONNX models and T4 engines from the release](https://github.com/XYETHER/MoTSR/releases/tag/v1.0.0)**.

| File | For |
| --- | --- |
| `MoTSR_2x.safetensors` | Original EMA checkpoint for the PyTorch architecture. |
| `MoTSR_2x_recurrent.onnx` | Seven-input recurrent model. |
| `MoTSR_2x_*_T4.engine` | Prebuilt FP16 engines for T4 + TensorRT 11.0.0.114. |

<details>
<summary><strong>🔬 How the temporal architecture works</strong></summary>

Each output uses five RGB input frames and two earlier, floating-point SR predictions:

```mermaid
flowchart LR
    A["LR t−2 + SR t−2"] --> M[MoTSR]
    B["LR t−1 + SR t−1"] --> M
    C["LR t"] --> M
    D["LR t+1"] --> M
    E["LR t+2"] --> M
    M --> O["2× SR t"]
    O --> R[Recurrent memory for later frames]
```

- MoSRv2 supplies the spatial stem, 24 body blocks and pixel-shuffle head.
- Temporal fusion combines center-frame features with neighbor differences. Three residual depthwise mixing blocks use dilations 1, 2 and 3.
- Two feedback branches use earlier SR residuals relative to bilinear-upscaled LR, with learned gates and motion gates.
- The spatial `image_only` path seeds memory at the start of a clip or detected scene. At boundaries, LR neighbors are replicated.
- Raw SR tensors feed the next step **before** clamping, resizing or encoding. State resets at detected cuts, and neighbor windows stay inside that segment.
- The cut detector is a simple thumbnail-difference heuristic; it can miss cuts or trigger on fast motion.
- The model needs two future frames. It is an offline video model, with two frames of look-ahead.

**Tensor contract:** RGB float32 in `[0,1]`, NCHW, batch 1 for the release engines. LR tensors share `[1,3,H,W]`; feedback tensors are `[1,3,2H,2W]`.

| ONNX input order | Tensor |
| --- | --- |
| 1 | `older_lr` = LR[t−2] |
| 2 | `older_hr` = SR[t−2] |
| 3 | `previous_lr` = LR[t−1] |
| 4 | `previous_hr` = SR[t−1] |
| 5 | `current_lr` = LR[t] |
| 6 | `next_lr` = LR[t+1] |
| 7 | `later_lr` = LR[t+2] |

The release engines have landscape and portrait profiles: minimum 32 pixels per LR dimension, maximum long side 1920 and short side 1080. Odd dimensions are reflect-padded during inference and cropped back at 2×.

Checkpoint: **phase-4 EMA, iteration 3829**.

</details>

<details>
<summary><strong>🛠️ Run locally or export the model</strong></summary>

Install PyTorch for your hardware, the packages in `requirements.txt`, and FFmpeg. Download the checkpoint into `models/`.

```bash
pip install -r requirements.txt
python motsr_video.py input.mp4 output.mp4 --backend torch --codec libx264
```

The portable PyTorch backend uses CUDA when available and otherwise CPU. For GPU encoding, use `--codec h264_nvenc` or `--codec hevc_nvenc`. Use `--full-resolution` to disable the 720p speed limit, or `--force-1080p` to cap the finished output.

Export both ONNX graphs:

```bash
pip install onnx
python tools/export_onnx.py models/MoTSR_2x.safetensors --output-dir models
```

Rebuild the T4 engines offline on a T4 with TensorRT 11.0.0.114:

```bash
pip install tensorrt==11.0.0.114 onnx onnxconverter-common
python tools/build_t4_engines.py --model-dir models
python motsr_video.py input.mp4 output.mp4 --backend trt
```

Test results and runtime details are in [validation.json](validation.json).

</details>

## 🤍 Credits & license

- **[xyether](https://github.com/XYETHER)** — temporal architecture, model training, [4K-VFHQ-Tiny dataset](https://huggingface.co/datasets/XYETHER/4K-VFHQ-Tiny), notebook and release.
- **[umzi2](https://github.com/umzi2)** — the base **[MoSRv2 non-temporal architecture](https://github.com/umzi2/MoSRV2)**, licensed under MIT.
- **[traiNNer-redux](https://github.com/the-database/traiNNer-redux)** — the framework used to train MoTSR.
- The notebook layout is adapted from **[Xyether Anime Upscaler](https://github.com/XYETHER/Xyether-Anime-Upscaler)**.

MoTSR's temporal code, weights and notebook are released under **[MIT](LICENSE)**. Upstream notices are preserved in [NOTICE](NOTICE) and [licenses/](licenses/). Example footage is separate from the code/model license.
