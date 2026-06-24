"""End-to-end spot model: HRNet backbone + heads, plus the inference path that
emits the canonical :mod:`spotpipe.schema` (build stage 3).

At inference we: run the model, peak-find the heatmap (local-max NMS +
threshold), apply the offset head for the sub-pixel centre, read
``logI1``/``logI2``/``logvar`` at those centres, and emit canonical schema rows.
The intensity read happens at the *detected* centre (detection + offset); during
training it happens at the *ground-truth* centre. This train/inference coupling
is accepted and intended (see CLAUDE.md and ``losses.intensity``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from spotpipe.models.backbone import HRNetBackbone, build_backbone
from spotpipe.models.heads import SpotHeads, build_heads
from spotpipe.schema import SpotRecord, records_to_dataframe

__all__ = [
    "SpotModel",
    "build_spot_model",
    "normalize_counts",
    "predict_spots",
]


def normalize_counts(image: torch.Tensor, adc_max: float) -> torch.Tensor:
    """Scale raw 12-bit counts to ~[0, 1] by a FIXED divisor (the ADC ceiling).

    This is a deterministic, per-dataset constant scaling -- NOT a per-image
    normalisation. We deliberately preserve absolute contrast / pedestal so the
    network stays gain-aware (we do not want gain-invariance; see CLAUDE.md).
    """
    return image.to(torch.float32) / float(adc_max)


class SpotModel(nn.Module):
    """Backbone + heads. ``forward`` returns the heads' dict of output maps."""

    def __init__(self, backbone: HRNetBackbone, heads: SpotHeads) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.backbone(x))


def build_spot_model(config: dict | None = None) -> SpotModel:
    """Assemble the full spot model from a ``model:`` config block."""
    cfg = config or {}
    backbone = build_backbone(cfg)
    heads = build_heads(backbone.out_channels, cfg)
    return SpotModel(backbone, heads)


@torch.no_grad()
def predict_spots(
    model: SpotModel,
    image,
    *,
    image_id: str,
    adc_max: float = 4095.0,
    peak_threshold: float = 0.3,
    nms_kernel: int = 3,
    max_spots: int | None = None,
    logvar_min: float = -10.0,
    logvar_max: float = 6.0,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Run the model on one two-channel image and emit canonical-schema rows.

    Parameters
    ----------
    model : a :class:`SpotModel`.
    image : ``[2, H, W]`` (or ``[1, 2, H, W]``) array/tensor of raw counts.
    image_id : identifier written into every emitted row.
    adc_max : ADC ceiling used for the same fixed input scaling as training.
    peak_threshold : minimum heatmap probability for a detection.
    nms_kernel : window (px) for local-max non-maximum suppression.
    max_spots : optional cap, keeping the highest-confidence peaks.
    logvar_min, logvar_max : clamp range applied to the raw log-variance *only*
        when deriving the ``uncertainty1`` / ``uncertainty2`` schema columns. It
        is the SAME range the intensity NLL clamps to (default ``[-10, 6]``), so
        a single extreme logvar pixel cannot poison downstream calibration /
        uncertainty-weighting. The head's raw logvar output stays unclamped; only
        the derived ``uncertainty`` column is bounded.

    Returns a DataFrame with exactly the canonical columns. ``sigma1_hat`` /
    ``sigma2_hat`` are NaN (this model does not estimate per-spot PSF width);
    ``uncertainty1`` / ``uncertainty2`` are the predicted log-intensity standard
    deviations ``exp(0.5 * clip(logvar, logvar_min, logvar_max))``.
    """
    model.eval()
    model.to(device)

    img = torch.as_tensor(np.asarray(image))
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4 or img.shape[1] != 2:
        raise ValueError(f"expected image [2,H,W] or [1,2,H,W], got shape {tuple(img.shape)}")
    img = normalize_counts(img, adc_max).to(device)

    preds = model(img)
    heat = torch.sigmoid(preds["heatmap"])[0, 0]  # [H, W]

    # Local-max NMS: keep pixels that equal their max-pooled neighbourhood and
    # clear the detection threshold.
    pad = nms_kernel // 2
    pooled = F.max_pool2d(heat[None, None], nms_kernel, stride=1, padding=pad)[0, 0]
    keep = (heat >= pooled) & (heat > peak_threshold)
    ys, xs = torch.where(keep)
    scores = heat[ys, xs]

    if max_spots is not None and scores.numel() > max_spots:
        top = torch.topk(scores, int(max_spots))
        sel = top.indices
        ys, xs, scores = ys[sel], xs[sel], scores[sel]

    offset = preds["offset"][0]   # [2, H, W]
    logI1 = preds["logI1"][0, 0]
    logI2 = preds["logI2"][0, 0]
    logvar1 = preds["logvar1"][0, 0]
    logvar2 = preds["logvar2"][0, 0]

    def _uncertainty(raw_logvar: float) -> float:
        # Bound the DERIVED uncertainty only, using the NLL's clamp range, so one
        # extreme logvar pixel can't poison downstream calibration/weighting. The
        # raw head output is left unclamped (see ``losses.intensity``).
        clamped = min(max(raw_logvar, logvar_min), logvar_max)
        return math.exp(0.5 * clamped)

    records: list[SpotRecord] = []
    for k in range(scores.numel()):
        r = int(ys[k])
        c = int(xs[k])
        dx = float(offset[0, r, c])
        dy = float(offset[1, r, c])
        records.append(
            SpotRecord.from_logs(
                image_id=image_id,
                spot_id=k,
                x=float(c) + dx,            # col + frac_x
                y=float(r) + dy,            # row + frac_y
                p_detect=float(scores[k]),
                logI1=float(logI1[r, c]),
                logI2=float(logI2[r, c]),
                sigma1_hat=math.nan,        # not estimated by this model
                sigma2_hat=math.nan,
                uncertainty1=_uncertainty(float(logvar1[r, c])),
                uncertainty2=_uncertainty(float(logvar2[r, c])),
                flags="",
            )
        )
    return records_to_dataframe(records)
