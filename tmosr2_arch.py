"""Tmosr2: 2x recurrent real-life video super-resolution.

Temporal extensions and model weights: xyether, 2026, MIT.
MoSRv2 spatial backbone: umzi2, MIT (see licenses/MoSRv2-MIT.txt).
Adapted from the supplied traiNNer-redux checkout; retain its Apache-2.0 notice.
The public name is Tmosr2; checkpoint parameter names are unchanged.
"""
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.module import _IncompatibleKeys



class UniUpsampleV3(nn.Sequential):
    """Checkpoint-compatible direct pixel-shuffle head (the released configuration)."""
    def __init__(self, upsample, scale, in_dim, out_dim, mid_dim):
        if upsample != "pixelshuffledirect":
            raise ValueError("This standalone release supports pixelshuffledirect.")
        super().__init__(nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale))
        self.register_buffer("MetaUpsample", torch.tensor([3, 1, scale, in_dim, out_dim, mid_dim, 4], dtype=torch.uint8))

class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self,
        in_channels: int = 64,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.offset = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed + self.offset


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self, dim: int = 64, expansion_ratio: float = 8 / 3, rms_norm: bool = True
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim) if rms_norm else LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = dim
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = InceptionDWConv2d(conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x * self.gamma + shortcut


class MoSRv2(nn.Module):
    """Mamba Out Super-Resolution"""

    def __init__(
        self,
        in_ch: int = 3,
        scale: int = 4,
        n_block: int = 24,
        dim: int = 64,
        upsampler: str = "pixelshuffledirect",
        expansion_ratio: float = 1.5,
        mid_dim: int = 32,
        unshuffle_mod: bool = True,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        self.short = nn.Upsample(scale_factor=scale, mode="bilinear")
        self.scale = scale
        self.pad = 1
        if unshuffle_mod and scale < 3:
            unshuffle = 4 // scale
            in_to_dim = [
                nn.PixelUnshuffle(unshuffle),
                nn.Conv2d(in_ch * unshuffle**2, dim, 3, 1, 1),
            ]
            self.pad = unshuffle
            scale = 4
        else:
            in_to_dim = [nn.Conv2d(in_ch, dim, 3, 1, 1)]

        self._dim = dim
        self._tail_start = len(in_to_dim) + n_block

        self.gblocks = nn.Sequential(
            *in_to_dim
            + [
                GatedCNNBlock(
                    dim=dim, expansion_ratio=expansion_ratio, rms_norm=rms_norm
                )
                for _ in range(n_block)
            ]
            + [
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.Mish(True),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Mish(True),
                nn.Conv2d(dim, dim, 1, 1),
            ]
        )
        self.to_img = UniUpsampleV3(upsampler, scale, dim, in_ch, mid_dim)

    def prepare_load_state_dict(self, state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Move a differently-sized MoSRV2 tail while retaining shared body blocks."""
        source_tail = next(
            (
                int(key.split(".")[1])
                for key, value in state_dict.items()
                if key.startswith("gblocks.")
                and key.endswith(".weight")
                and tuple(value.shape) == (self._dim * 2, self._dim, 3, 3)
            ),
            self._tail_start,
        )
        if source_tail == self._tail_start:
            return dict(state_dict)

        adapted = dict(state_dict)
        mapped_tail: dict[str, Tensor] = {}
        for offset in (0, 2, 4):
            old_prefix = f"gblocks.{source_tail + offset}."
            new_prefix = f"gblocks.{self._tail_start + offset}."
            for key, value in state_dict.items():
                if key.startswith(old_prefix):
                    mapped_tail[new_prefix + key.removeprefix(old_prefix)] = value

        cutoff = min(source_tail, self._tail_start)
        for key in tuple(adapted):
            parts = key.split(".")
            if parts[0] == "gblocks" and parts[1].isdigit() and int(parts[1]) >= cutoff:
                del adapted[key]
        adapted.update(mapped_tail)
        return adapted

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        assert isinstance(state_dict, dict)
        state_dict["to_img.MetaUpsample"] = self.to_img.MetaUpsample
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        mod_pad_h = (self.pad - h % self.pad) % self.pad
        mod_pad_w = (self.pad - w % self.pad) % self.pad
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x: Tensor) -> Tensor:
        _b, _c, h, w = x.shape
        x = self.check_img_size(x, h, w)
        x = self.to_img(self.gblocks(x)) + self.short(x)
        return x[:, :, : h * self.scale, : w * self.scale]


"""Checkpoint-preserving, centered-window MoSRV2. Independent of legacy TMoSR models."""

from typing import Any

import torch
from torch import Tensor, nn



class TemporalMixBlock(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim),
            nn.Mish(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mix(x)


class TemporalResidualFusion(nn.Module):
    def __init__(self, dim: int, temporal_dim: int, clip_size: int) -> None:
        super().__init__()
        self.compress = nn.Conv2d(dim, temporal_dim, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(temporal_dim * clip_size, temporal_dim, 1),
            nn.Mish(),
            *(TemporalMixBlock(temporal_dim, d) for d in (1, 2, 3)),
        )
        self.project = nn.Conv2d(temporal_dim, dim, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features: list[Tensor], center: int) -> Tensor:
        compressed = [self.compress(feature) for feature in features]
        reference = compressed[center]
        evidence = [reference] + [
            feature - reference for i, feature in enumerate(compressed) if i != center
        ]
        return self.project(self.fuse(torch.cat(evidence, dim=1)))


class _TemporalBase(MoSRv2):
    """Five RGB frames -> center RGB at 2x; all image checkpoint keys are retained."""

    def __init__(
        self,
        scale: int = 2,
        clip_size: int = 5,
        temporal_dim: int = 32,
        stage: str = "bootstrap",
        unfreeze_last_blocks: int = 6,
        **kwargs: Any,
    ) -> None:
        if scale != 2 or clip_size not in (3, 5, 7, 9):
            raise ValueError("centered-window design supports scale=2 and clip_size=3/5/7/9.")
        if kwargs.get("in_ch", 3) != 3 or not kwargs.get("unshuffle_mod", True):
            raise ValueError("V1 requires RGB and unshuffle_mod=true.")
        if kwargs.get("upsampler", "pixelshuffledirect") != "pixelshuffledirect":
            raise ValueError("V1 supports the pixelshuffledirect checkpoint head.")
        super().__init__(scale=scale, **kwargs)
        self.clip_size = clip_size
        self.center = clip_size // 2
        self.temporal = TemporalResidualFusion(self._dim, temporal_dim, clip_size)
        self.configure_stage(stage, unfreeze_last_blocks)

    def configure_stage(self, stage: str, last_blocks: int = 6) -> None:
        if stage not in ("bootstrap", "partial", "joint"):
            raise ValueError(f"Unknown training stage: {stage}")
        n_blocks = self._tail_start - 2
        if not 0 <= last_blocks <= n_blocks:
            raise ValueError("Invalid number of blocks to unfreeze")
        self.requires_grad_(False)
        self.temporal.requires_grad_(True)
        if stage == "partial":
            for module in list(self.gblocks)[self._tail_start - last_blocks :]:
                module.requires_grad_(True)
            self.to_img.requires_grad_(True)
        elif stage == "joint":
            self.requires_grad_(True)
        self.stage = stage

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != self.clip_size or x.shape[2] != 3:
            raise ValueError(
                f"Expected [B,{self.clip_size},3,H,W], got {tuple(x.shape)}"
            )
        h, w = x.shape[-2:]
        # Separate stem calls retain the image model's operation/batch shape exactly.
        frames = [
            self.check_img_size(x[:, i].contiguous(), h, w)
            for i in range(self.clip_size)
        ]
        features = [self.gblocks[1](self.gblocks[0](frame)) for frame in frames]
        out = features[self.center] + self.temporal(features, self.center)
        # Frozen weights still propagate gradients back to the temporal injection.
        for module in list(self.gblocks)[2:]:
            out = module(out)
        out = self.to_img(out) + self.short(frames[self.center])
        return out[:, :, : h * self.scale, : w * self.scale]


"""Recurrent earlier recurrent design: previous LR/SR, current LR, and next LR -> current SR.

The three-frame temporal fusion is a folded form of the V1 five-frame fusion.
The previous-HR path starts at zero, so conversion preserves the V1 prediction
for [previous, previous, current, next, next] at initialization.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn



class _RecurrentBase(_TemporalBase):
    def __init__(
        self,
        scale: int = 2,
        *,
        temporal_dim: int = 32,
        feedback_dim: int = 32,
        stage: str = "bootstrap",
        unfreeze_last_blocks: int = 6,
        teacher_forcing_prob: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            scale=scale, clip_size=5, temporal_dim=temporal_dim,
            stage=stage, unfreeze_last_blocks=unfreeze_last_blocks, **kwargs,
        )
        self.temporal = TemporalResidualFusion(self._dim, temporal_dim, 3)
        self.feedback = nn.Sequential(
            nn.Conv2d(12, feedback_dim, 3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(feedback_dim, self._dim, 1),
        )
        self.feedback_gate = nn.Conv2d(6, 1, 3, padding=1)
        nn.init.zeros_(self.feedback[-1].weight)
        nn.init.zeros_(self.feedback[-1].bias)
        nn.init.zeros_(self.feedback_gate.weight)
        nn.init.constant_(self.feedback_gate.bias, -2.0)
        self.clip_size = 3
        self.center = 1
        self.teacher_forcing_prob = teacher_forcing_prob
        self.rollout_outputs: list[Tensor] = []
        self.rollout_grad_steps = 3
        self.configure_stage(stage, unfreeze_last_blocks)
        self.feedback.requires_grad_(True)
        self.feedback_gate.requires_grad_(True)

    @staticmethod
    def pack_inputs(
        previous_lr: Tensor, previous_hr: Tensor, current_lr: Tensor, next_lr: Tensor
    ) -> Tensor:
        """Pack four physical inputs as seven RGB LR-sized planes for traiNNer."""
        hr_planes = F.pixel_unshuffle(previous_hr, 2).chunk(4, dim=1)
        return torch.stack((previous_lr, current_lr, next_lr, *hr_planes), dim=1)

    @staticmethod
    def unpack_inputs(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 5 or x.shape[1] != 7 or x.shape[2] != 3:
            raise ValueError("Expected packed [B,7,3,H,W] input")
        previous_lr, current_lr, next_lr = x[:, 0], x[:, 1], x[:, 2]
        previous_hr = F.pixel_shuffle(torch.cat(tuple(x[:, i] for i in range(3, 7)), dim=1), 2)
        return previous_lr, previous_hr, current_lr, next_lr

    def image_only(self, lr: Tensor) -> Tensor:
        """Spatial predecessor used for scheduled sampling, without HR leakage."""
        h, w = lr.shape[-2:]
        padded = self.check_img_size(lr, h, w)
        sr = self.to_img(self.gblocks(padded)) + self.short(padded)
        return sr[:, :, : h * 2, : w * 2]

    def forward(self, packed: Tensor) -> Tensor:
        if packed.ndim == 5 and packed.shape[1] != 7:
            # [B, T+2, 3, H, W]: previous, T current frames, final look-ahead.
            if packed.shape[1] < 4 or packed.shape[2] != 3:
                raise ValueError("Expected [B,T+2,3,H,W] rollout input")
            with torch.no_grad():
                previous_hr = self.image_only(packed[:, 0])
            outputs = []
            total_steps = packed.shape[1] - 2
            burn_in = max(0, total_steps - self.rollout_grad_steps)
            for step in range(1, packed.shape[1] - 1):
                if step <= burn_in:
                    with torch.no_grad():
                        current_hr = self.forward_step(
                            packed[:, step - 1], previous_hr,
                            packed[:, step], packed[:, step + 1],
                        )
                else:
                    current_hr = self.forward_step(
                        packed[:, step - 1], previous_hr,
                        packed[:, step], packed[:, step + 1],
                    )
                outputs.append(current_hr)
                previous_hr = current_hr
            self.rollout_outputs = outputs
            return outputs[-1]
        self.rollout_outputs = []
        previous_lr, previous_hr, current_lr, next_lr = self.unpack_inputs(packed)
        if self.training and self.teacher_forcing_prob < 1.0:
            with torch.no_grad():
                predicted_previous = self.image_only(previous_lr)
            if self.teacher_forcing_prob <= 0.0:
                previous_hr = predicted_previous
            else:
                use_teacher = (torch.rand(previous_lr.shape[0], 1, 1, 1, device=previous_lr.device)
                               < self.teacher_forcing_prob)
                previous_hr = torch.where(use_teacher, previous_hr, predicted_previous)
        return self.forward_step(previous_lr, previous_hr, current_lr, next_lr)

    def forward_step(
        self, previous_lr: Tensor, previous_hr: Tensor,
        current_lr: Tensor, next_lr: Tensor,
    ) -> Tensor:
        h, w = current_lr.shape[-2:]
        if (previous_lr.shape != current_lr.shape or next_lr.shape != current_lr.shape
                or previous_hr.shape != (current_lr.shape[0], 3, h * 2, w * 2)):
            raise ValueError("Expected three matching RGB LR frames and one matching 2x previous SR")
        p = self.check_img_size(previous_lr, h, w)
        c = self.check_img_size(current_lr, h, w)
        n = self.check_img_size(next_lr, h, w)
        pad_h, pad_w = p.shape[-2] - h, p.shape[-1] - w
        prev_hr = F.pad(previous_hr, (0, pad_w * 2, 0, pad_h * 2), mode="reflect")
        baseline = F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False)
        hr_residual = F.pixel_unshuffle(prev_hr - baseline, 2)
        prev_feature = self.gblocks[1](self.gblocks[0](p))
        current_feature = self.gblocks[1](self.gblocks[0](c))
        next_feature = self.gblocks[1](self.gblocks[0](n))
        p_small = F.avg_pool2d(p, 2)
        c_small = F.avg_pool2d(c, 2)
        learned_gate = torch.sigmoid(self.feedback_gate(torch.cat((p_small, c_small), dim=1)))
        motion_gate = torch.exp(-12.0 * (p_small - c_small).abs().mean(dim=1, keepdim=True))
        prev_feature = prev_feature + self.feedback(hr_residual) * learned_gate * motion_gate
        out = current_feature + self.temporal(
            [prev_feature, current_feature, next_feature], 1
        )
        for module in list(self.gblocks)[2:]:
            out = module(out)
        out = self.to_img(out) + self.short(c)
        return out[:, :, : h * 2, : w * 2]


"""Tmosr2: five LR frames and two recurrent HR feedback frames.

Inputs at t: LR[t-2], SR[t-2], LR[t-1], SR[t-1], LR[t], LR[t+1], LR[t+2].
The new outer temporal positions and older-HR feedback start at zero so a
converted V2.1 checkpoint initially predicts the same output for matched
inner frames and previous-HR input.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn



class Tmosr2(_RecurrentBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        temporal_dim = self.temporal.compress.out_channels
        feedback_dim = self.feedback[0].out_channels
        self.temporal = TemporalResidualFusion(self._dim, temporal_dim, 5)
        self.feedback_older = nn.Sequential(
            nn.Conv2d(12, feedback_dim, 3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(feedback_dim, self._dim, 1),
        )
        self.feedback_gate_older = nn.Conv2d(6, 1, 3, padding=1)
        nn.init.zeros_(self.feedback_older[-1].weight)
        nn.init.zeros_(self.feedback_older[-1].bias)
        nn.init.zeros_(self.feedback_gate_older.weight)
        nn.init.constant_(self.feedback_gate_older.bias, -2.0)
        self.feedback_older.requires_grad_(True)
        self.feedback_gate_older.requires_grad_(True)

    def forward_step(
        self, older_lr: Tensor, older_hr: Tensor,
        previous_lr: Tensor, previous_hr: Tensor,
        current_lr: Tensor, next_lr: Tensor, later_lr: Tensor,
    ) -> Tensor:
        h, w = current_lr.shape[-2:]
        lr_frames = (older_lr, previous_lr, current_lr, next_lr, later_lr)
        if any(frame.shape != current_lr.shape for frame in lr_frames):
            raise ValueError("All five RGB LR frames must have the same shape")
        hr_shape = (current_lr.shape[0], 3, h * 2, w * 2)
        if older_hr.shape != hr_shape or previous_hr.shape != hr_shape:
            raise ValueError("Both previous SR frames must be RGB at 2x resolution")
        padded = [self.check_img_size(frame, h, w) for frame in lr_frames]
        pad_h, pad_w = padded[2].shape[-2] - h, padded[2].shape[-1] - w
        older_p, previous_p, current_p, next_p, later_p = padded
        features = [self.gblocks[1](self.gblocks[0](frame)) for frame in padded]

        def feedback_feature(lr: Tensor, hr: Tensor, feedback: nn.Module,
                             gate: nn.Module) -> Tensor:
            hr = F.pad(hr, (0, pad_w * 2, 0, pad_h * 2), mode="reflect")
            baseline = F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False)
            residual = F.pixel_unshuffle(hr - baseline, 2)
            reference = F.avg_pool2d(lr, 2)
            current = F.avg_pool2d(current_p, 2)
            learned = torch.sigmoid(gate(torch.cat((reference, current), dim=1)))
            motion = torch.exp(-12.0 * (reference - current).abs().mean(1, keepdim=True))
            return feedback(residual) * learned * motion

        features[2] = features[2] + feedback_feature(
            older_p, older_hr, self.feedback_older, self.feedback_gate_older
        )
        features[1] = features[1] + feedback_feature(
            previous_p, previous_hr, self.feedback, self.feedback_gate
        )
        out = features[2] + self.temporal(features, 2)
        for module in list(self.gblocks)[2:]:
            out = module(out)
        out = self.to_img(out) + self.short(current_p)
        return out[:, :, :h * 2, :w * 2]

    def forward(self, lr_window: Tensor) -> Tensor:
        if lr_window.ndim != 5 or lr_window.shape[2] != 3 or lr_window.shape[1] < 5:
            raise ValueError("Expected [B,5,3,H,W] or [B,T+4,3,H,W]")
        if lr_window.shape[1] == 5:
            self.rollout_outputs = []
            with torch.no_grad():
                older_hr = self.image_only(lr_window[:, 0])
                previous_hr = self.image_only(lr_window[:, 1])
            return self.forward_step(
                lr_window[:, 0], older_hr, lr_window[:, 1], previous_hr,
                lr_window[:, 2], lr_window[:, 3], lr_window[:, 4],
            )
        total_steps = lr_window.shape[1] - 4
        burn_in = max(0, total_steps - self.rollout_grad_steps)
        with torch.no_grad():
            older_hr = self.image_only(lr_window[:, 0])
            previous_hr = self.image_only(lr_window[:, 1])
        outputs = []
        for step in range(total_steps):
            args = (
                lr_window[:, step], older_hr,
                lr_window[:, step + 1], previous_hr,
                lr_window[:, step + 2], lr_window[:, step + 3],
                lr_window[:, step + 4],
            )
            if step < burn_in:
                with torch.no_grad():
                    current_hr = self.forward_step(*args)
            else:
                current_hr = self.forward_step(*args)
            outputs.append(current_hr)
            older_hr, previous_hr = previous_hr, current_hr
        self.rollout_outputs = outputs
        return outputs[-1]

