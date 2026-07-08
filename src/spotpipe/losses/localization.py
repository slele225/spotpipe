"""Sub-pixel localization (offset) loss (build stage 3).

Smooth-L1 on the predicted fractional offset ``(frac_x, frac_y)``, **masked to
ground-truth centre pixels only** -- there is no gradient off-centre. The
offset, together with the integer centre pixel, reconstructs the sub-pixel spot
location at inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["localization_loss"]


def localization_loss(
    offset_pred: torch.Tensor,
    offset_target: torch.Tensor,
    center_mask: torch.Tensor,
    *,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    """Masked smooth-L1 offset loss.

    Parameters
    ----------
    offset_pred : ``[B, 2, H, W]`` predicted ``(frac_x, frac_y)``.
    offset_target : ``[B, 2, H, W]`` target fractional offsets.
    center_mask : ``[B, 1, H, W]`` 1.0 at centre pixels; the loss is computed
        only there.
    """
    per_elem = F.smooth_l1_loss(offset_pred, offset_target, beta=smooth_l1_beta, reduction="none")
    per_pixel = per_elem.sum(dim=1, keepdim=True)  # sum over (x, y) -> [B, 1, H, W]
    masked = per_pixel * center_mask
    return masked.sum() / torch.clamp(center_mask.sum(), min=1.0)
