"""Target-map construction: ground-truth spot rows -> dense training targets.

Ported UNCHANGED from the old repo (``spotpipe.training.targets``); this is the
boundary where dense / incorrect supervision could creep in, so the rules are
explicit and narrow (see CLAUDE.md).

For each ground-truth spot at sub-pixel ``(x, y)`` (``x`` = column, ``y`` = row;
the image is indexed ``[row, col]`` throughout, matching the simulator's PSF):

* **Centre pixel** = ``(floor(x), floor(y))`` -> stored at array index
  ``[row=floor(y), col=floor(x)]``.
* **Offset target** at that centre pixel = ``(x - floor(x), y - floor(y))`` --
  the fractional sub-pixel part, in ``[0, 1)``. Zero elsewhere (and masked).
* **Intensity / log-variance losses are masked to that single centre pixel
  only** via ``center_mask`` -- never computed over any other pixel. The
  intensity target is the simulator's TRUE integrated log-intensity
  (``logI1``/``logI2``), the pre-detector photon count, NOT a pixel readout.
* **Heatmap** = a tight Gaussian blob rendered at the integer centre pixel
  (peak exactly 1.0 there), blobs combined across spots by element-wise max.

Tie-break when two spots share a centre pixel (rare): the heatmap takes the max
of their blobs (both peak 1.0, so the pixel stays 1.0); ``center_mask`` is 1
regardless; the offset / intensity targets are written **last-spot-wins**.

Returned target dict (per image, channel-first tensors):
  ``heatmap``     [1, H, W]  Gaussian-blob detection target in [0, 1]
  ``center_mask`` [1, H, W]  1.0 at integer centre pixels, else 0.0
  ``offset``      [2, H, W]  (frac_x, frac_y) at centres, else 0.0
  ``logI1``       [1, H, W]  true log-intensity ch1 at centres, else 0.0
  ``logI2``       [1, H, W]  true log-intensity ch2 at centres, else 0.0
"""

from __future__ import annotations

import math

import pandas as pd
import torch

__all__ = ["build_targets", "TARGET_KEYS"]

TARGET_KEYS = ("heatmap", "center_mask", "offset", "logI1", "logI2")


def _draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, sigma: float, radius: int) -> None:
    """Max-combine a unit-peak Gaussian centred at integer pixel ``(cx, cy)``."""
    h, w = heatmap.shape
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = torch.arange(x0, x1, dtype=torch.float32)
    ys = torch.arange(y0, y1, dtype=torch.float32)
    gx = torch.exp(-((xs - cx) ** 2) / (2.0 * sigma * sigma))
    gy = torch.exp(-((ys - cy) ** 2) / (2.0 * sigma * sigma))
    blob = gy[:, None] * gx[None, :]  # peak 1.0 at (cy, cx)
    region = heatmap[y0:y1, x0:x1]
    heatmap[y0:y1, x0:x1] = torch.maximum(region, blob)


def build_targets(
    spots: pd.DataFrame,
    shape: tuple[int, int],
    heatmap_sigma: float,
) -> dict[str, torch.Tensor]:
    """Build the dense training-target maps for one image's ground-truth spots.

    Parameters
    ----------
    spots : ground-truth spot table in the canonical schema (the simulator's
        ``SimulatedImage.spots``). Only ``x``, ``y``, ``logI1``, ``logI2`` are
        read; prediction-only columns are ignored.
    shape : ``(H, W)``.
    heatmap_sigma : Gaussian-blob sigma (pixels) for the detection target. Keep
        tight (high-overlap regime).
    """
    h, w = shape
    heatmap = torch.zeros(1, h, w, dtype=torch.float32)
    center_mask = torch.zeros(1, h, w, dtype=torch.float32)
    offset = torch.zeros(2, h, w, dtype=torch.float32)
    logI1 = torch.zeros(1, h, w, dtype=torch.float32)
    logI2 = torch.zeros(1, h, w, dtype=torch.float32)

    if len(spots) == 0:
        return {
            "heatmap": heatmap,
            "center_mask": center_mask,
            "offset": offset,
            "logI1": logI1,
            "logI2": logI2,
        }

    radius = max(int(math.ceil(3.0 * heatmap_sigma)), 1)
    xs = spots["x"].to_numpy()
    ys = spots["y"].to_numpy()
    li1 = spots["logI1"].to_numpy()
    li2 = spots["logI2"].to_numpy()

    for x, y, l1, l2 in zip(xs, ys, li1, li2):
        x = float(x)
        y = float(y)
        cx = min(max(int(math.floor(x)), 0), w - 1)
        cy = min(max(int(math.floor(y)), 0), h - 1)
        _draw_gaussian(heatmap[0], cx, cy, heatmap_sigma, radius)
        center_mask[0, cy, cx] = 1.0
        offset[0, cy, cx] = x - math.floor(x)   # frac_x in [0, 1)
        offset[1, cy, cx] = y - math.floor(y)   # frac_y in [0, 1)
        logI1[0, cy, cx] = float(l1)            # last-spot-wins on collision
        logI2[0, cy, cx] = float(l2)

    return {
        "heatmap": heatmap,
        "center_mask": center_mask,
        "offset": offset,
        "logI1": logI1,
        "logI2": logI2,
    }
