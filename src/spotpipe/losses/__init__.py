"""Training losses + combiner (build stage 3).

The ENTIRE phase-1 loss is::

    L = w_heatmap * detection
      + w_offset  * localization
      + w_intensity * (intensity_NLL_ch1 + intensity_NLL_ch2)

Phase 1 has NO slope loss and NO ratio loss: the ratio-law slope beta is
computed explicitly DOWNSTREAM of inference, and the per-spot log-ratio is
derived downstream as ``logI2 - logI1`` -- never trained on. A ratio term is
redundant with the per-channel intensity losses and can compete with them; an
in-batch slope is a biased (attenuated) estimator that would distort the very
per-spot intensities the project must keep unbiased (see CLAUDE.md). Hence this
module imports ONLY detection + localization + intensity, and the downstream
``losses.ratio`` slope helper is never wired into training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spotpipe.losses.detection import detection_loss
from spotpipe.losses.intensity import gaussian_nll_masked, intensity_nll
from spotpipe.losses.localization import localization_loss

__all__ = [
    "detection_loss",
    "localization_loss",
    "intensity_nll",
    "gaussian_nll_masked",
    "SpotLoss",
    "DEFAULT_LOSS_WEIGHTS",
]

DEFAULT_LOSS_WEIGHTS = {"heatmap": 1.0, "offset": 1.0, "intensity": 1.0}


class SpotLoss(nn.Module):
    """Weighted sum of detection + localization + per-channel intensity NLL.

    ``forward(preds, targets)`` returns ``(total, components)`` where
    ``components`` is a dict of the individual (unweighted) loss scalars for
    logging: ``heatmap``, ``offset``, ``intensity1``, ``intensity2``, plus the
    weighted ``total``.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        logvar_min: float = -10.0,
        logvar_max: float = 6.0,
    ) -> None:
        super().__init__()
        w = dict(DEFAULT_LOSS_WEIGHTS)
        if weights:
            w.update(weights)
        self.weights = w
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        intensity_use_logvar: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``intensity_use_logvar`` is the variance-warmup switch: pass ``False``
        during the warmup phase so the intensity loss fits the mean under fixed
        variance, then ``True`` to enable the predicted-variance NLL."""
        heatmap = detection_loss(
            preds["heatmap"], targets["heatmap"], targets["center_mask"],
            alpha=self.focal_alpha, beta=self.focal_beta,
        )
        offset = localization_loss(preds["offset"], targets["offset"], targets["center_mask"])
        nll = intensity_nll(
            preds, targets, logvar_min=self.logvar_min, logvar_max=self.logvar_max,
            use_logvar=intensity_use_logvar,
        )

        total = (
            self.weights["heatmap"] * heatmap
            + self.weights["offset"] * offset
            + self.weights["intensity"] * (nll["intensity1"] + nll["intensity2"])
        )
        components = {
            "heatmap": heatmap.detach(),
            "offset": offset.detach(),
            "intensity1": nll["intensity1"].detach(),
            "intensity2": nll["intensity2"].detach(),
            "total": total.detach(),
        }
        return total, components
