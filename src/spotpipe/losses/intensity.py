"""Per-spot intensity loss -- the core, heteroscedastic Gaussian NLL (build stage 3).

For each channel ``k`` the network predicts a log-intensity mean ``logI_k`` and a
log-variance ``logvar_k`` at every pixel; this loss is **masked to ground-truth
spot centres only** and evaluates a Gaussian negative log-likelihood there:

    NLL_k = 0.5 * ( exp(-logvar_k) * (logI_k_pred - logI_k_true)^2 + logvar_k )

* The supervision target is the simulator's TRUE TOTAL INTEGRATED intensity in
  photon-proportional units, in natural-log space (``logI1``/``logI2`` from the
  ground-truth table) -- the pre-detector photon count, never a pixel/window
  readout (which would be contaminated by overlapping neighbours).
* The loss is computed ONLY at integer centre pixels (``center_mask``); it is
  never evaluated densely over background.
* Predicting ``logvar`` lets the network express uncertainty: heavily
  overlapped / dim spots, which are intrinsically harder to deblend, can lower
  their NLL by reporting a LARGER variance, so they come back with larger
  predicted uncertainty (the down-weighting the downstream slope fit wants).

The read happens at the GROUND-TRUTH centre during training; at inference the
centre comes from detection + offset. This coupling is accepted and intended
(see CLAUDE.md). ``logvar`` is clamped for numerical stability.

NO ratio / log-ratio / slope term is computed here or anywhere in training: the
ratio is derived downstream as ``logI2 - logI1`` from the per-channel estimates,
and unbiased per-channel intensities give an unbiased ratio for free.
"""

from __future__ import annotations

import torch

__all__ = ["gaussian_nll_masked", "intensity_nll"]


def gaussian_nll_masked(
    mean_pred: torch.Tensor,
    logvar_pred: torch.Tensor,
    target: torch.Tensor,
    center_mask: torch.Tensor,
    *,
    logvar_min: float = -10.0,
    logvar_max: float = 6.0,
    use_logvar: bool = True,
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL for one channel, masked to centre pixels.

    All tensors are ``[B, 1, H, W]`` (``center_mask`` is the 1.0-at-centres map).

    The predicted log-variance is clamped to ``[logvar_min, logvar_max]``
    (default ``[-10, 6]``) ONLY here, inside the loss, purely for numerical
    stability: the lower bound caps ``exp(-logvar)`` so a confident spot cannot
    blow the precision term up, and the upper bound keeps the variance finite.
    The head's stored ``logvar`` output is left unclamped (see ``heads`` /
    ``predict_spots``); this clamp affects the loss gradient only.

    ``use_logvar`` is the variance-warmup switch. With ``use_logvar=False`` the
    predicted log-variance is ignored and the loss reduces to ``0.5 * (logI_pred
    - logI_true)**2`` -- the SAME Gaussian NLL with variance fixed at 1. Training
    the mean under fixed variance first avoids two well-known NLL pathologies:
    (1) an enormous early NLL (variance=1 with large initial errors) whose
    gradient would swamp the shared backbone, and (2) the variance head inflating
    to "explain away" error so the mean never fits. Once the mean is good, the
    full predicted-variance NLL is enabled to calibrate uncertainty.
    """
    sq_err = (mean_pred - target) ** 2
    if use_logvar:
        logvar = logvar_pred.clamp(logvar_min, logvar_max)
        nll = 0.5 * (torch.exp(-logvar) * sq_err + logvar)  # [B, 1, H, W]
    else:
        nll = 0.5 * sq_err  # fixed unit variance (logvar := 0)
    masked = nll * center_mask
    return masked.sum() / torch.clamp(center_mask.sum(), min=1.0)


def intensity_nll(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    logvar_min: float = -10.0,
    logvar_max: float = 6.0,
    use_logvar: bool = True,
) -> dict[str, torch.Tensor]:
    """Per-channel masked Gaussian NLL on ``logI1`` and ``logI2``.

    Returns ``{"intensity1": NLL_ch1, "intensity2": NLL_ch2}`` so the combiner
    can log and weight the channels (here, equally). ``use_logvar`` is the
    variance-warmup switch (see :func:`gaussian_nll_masked`). NO ratio/slope term.
    """
    mask = targets["center_mask"]
    nll1 = gaussian_nll_masked(
        preds["logI1"], preds["logvar1"], targets["logI1"], mask,
        logvar_min=logvar_min, logvar_max=logvar_max, use_logvar=use_logvar,
    )
    nll2 = gaussian_nll_masked(
        preds["logI2"], preds["logvar2"], targets["logI2"], mask,
        logvar_min=logvar_min, logvar_max=logvar_max, use_logvar=use_logvar,
    )
    return {"intensity1": nll1, "intensity2": nll2}
