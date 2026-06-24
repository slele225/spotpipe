"""Prediction heads for the two-channel spot detector (build stage 3).

Four head GROUPS, all at full input resolution, consuming the backbone's
full-resolution feature map ``[B, C, H, W]``:

1. **heatmap** ``[B, 1, H, W]`` -- spot-center detection logits. Trained against
   a tight Gaussian-blob target (see ``training.targets`` / ``losses.detection``).
   Sigmoid is applied downstream (in the loss and at inference), so this head
   outputs raw logits; its final bias is initialised negative (CenterNet prior)
   so the initial foreground probability is small and the focal loss is stable.
2. **offset** ``[B, 2, H, W]`` -- sub-pixel centre correction ``(frac_x, frac_y)``.
   Meaningful only at spot centres (the localization loss masks it there).
3. **intensity means** -- ``logI1 [B, 1, H, W]`` and ``logI2 [B, 1, H, W]``, the
   per-spot TOTAL INTEGRATED log-intensity (photon-proportional units) regressed
   directly at the spot centre. NOT a pixel sum -- the network learns to deblend
   overlapping spots. Meaningful only at centres.
4. **intensity uncertainty** -- ``logvar1 [B, 1, H, W]`` and ``logvar2``, the
   predicted per-channel log-variance. These feed the Gaussian NLL and populate
   the schema ``uncertainty1`` / ``uncertainty2`` columns at inference.

The two-channel intensity and log-variance maps are produced by single 2-channel
convs and split, so channel counts stay unambiguous: forward() returns a dict
with one tensor per named output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SpotHeads", "build_heads"]


class _ConvHead(nn.Module):
    """3x3 conv -> ReLU -> 1x1 conv producing ``out_channels`` at full resolution."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    @property
    def final_conv(self) -> nn.Conv2d:
        return self.body[-1]


class SpotHeads(nn.Module):
    """The four head groups; ``forward`` returns a dict of named output tensors."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 64,
        heatmap_bias: float = -2.19,
    ) -> None:
        super().__init__()
        self.heatmap = _ConvHead(in_channels, 1, mid_channels)
        self.offset = _ConvHead(in_channels, 2, mid_channels)
        self.intensity = _ConvHead(in_channels, 2, mid_channels)  # logI1, logI2
        self.logvar = _ConvHead(in_channels, 2, mid_channels)     # logvar1, logvar2

        # CenterNet-style focal-loss prior: start with low foreground probability.
        nn.init.constant_(self.heatmap.final_conv.bias, float(heatmap_bias))

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        intensity = self.intensity(feat)
        logvar = self.logvar(feat)
        return {
            "heatmap": self.heatmap(feat),           # logits [B,1,H,W]
            "offset": self.offset(feat),             # [B,2,H,W] (frac_x, frac_y)
            "logI1": intensity[:, 0:1],              # [B,1,H,W]
            "logI2": intensity[:, 1:2],              # [B,1,H,W]
            "logvar1": logvar[:, 0:1],               # [B,1,H,W]
            "logvar2": logvar[:, 1:2],               # [B,1,H,W]
        }


def build_heads(in_channels: int, config: dict | None = None) -> SpotHeads:
    """Construct the head stack from a ``model:`` config block."""
    cfg = config or {}
    return SpotHeads(
        in_channels=in_channels,
        mid_channels=int(cfg.get("head_mid_channels", 64)),
        heatmap_bias=float(cfg.get("heatmap_bias", -2.19)),
    )
