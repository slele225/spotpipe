"""Heatmap detection loss (build stage 3).

CenterNet-style penalty-reduced focal loss for the spot/background imbalance.
The target is a tight Gaussian blob rendered at each ground-truth centre (see
``training.targets``); the blob gives graded "you are near a centre" gradients,
while the actual positives are the integer centre pixels (``center_mask``).

For predicted probability ``p = sigmoid(logits)``:

* **Positives** (the centre pixels): ``-(1 - p)**alpha * log(p)`` -- the usual
  focal up-weighting of hard/under-confident centres.
* **Negatives** (everything else): ``-(1 - y)**beta * p**alpha * log(1 - p)``,
  where ``y`` is the Gaussian-blob target. The ``(1 - y)**beta`` factor reduces
  the penalty for near-centre pixels (where ``y`` is close to 1), so the loss
  does not fight the unavoidable spread of the blob.

The loss is normalised by the number of positives (CenterNet convention); with
no spots in the image it falls back to the negative term alone.
"""

from __future__ import annotations

import torch

__all__ = ["detection_loss"]


def detection_loss(
    heatmap_logits: torch.Tensor,
    heatmap_target: torch.Tensor,
    center_mask: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalty-reduced focal loss on heatmap logits.

    Parameters
    ----------
    heatmap_logits : ``[B, 1, H, W]`` raw logits from the heatmap head.
    heatmap_target : ``[B, 1, H, W]`` Gaussian-blob target in ``[0, 1]``.
    center_mask : ``[B, 1, H, W]`` 1.0 at integer centre pixels (the positives).
    alpha, beta : focal-loss exponents (CenterNet defaults 2 and 4).
    """
    p = torch.sigmoid(heatmap_logits).clamp(eps, 1.0 - eps)
    pos = (center_mask > 0.5).to(p.dtype)
    neg = 1.0 - pos

    pos_loss = ((1.0 - p) ** alpha) * torch.log(p) * pos
    neg_weights = (1.0 - heatmap_target) ** beta
    neg_loss = neg_weights * (p ** alpha) * torch.log(1.0 - p) * neg

    n_pos = pos.sum()
    total = -(pos_loss.sum() + neg_loss.sum())
    return total / torch.clamp(n_pos, min=1.0)
