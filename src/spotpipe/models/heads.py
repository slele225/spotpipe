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
3. **intensity means** -- per-spot TOTAL INTEGRATED log-intensity
   (photon-proportional units) regressed directly at the spot centre. NOT a pixel
   sum -- the network learns to deblend overlapping spots. Meaningful only at
   centres.
4. **intensity uncertainty** -- the predicted per-channel log-variance. These feed
   the Gaussian NLL and populate the schema ``uncertainty1`` / ``uncertainty2``
   columns at inference.

Head parameterisation (``head_parameterisation`` in the ``model:`` config)
--------------------------------------------------------------------------
Two ways to parameterise the intensity means. **The schema-facing outputs
(``logI1``, ``logI2``, ``logvar1``, ``logvar2``) are IDENTICAL in both modes**, so
``predict_spots`` and every downstream consumer are unchanged -- only what the
network natively predicts (and therefore what the loss trains on) differs.

* ``"independent"`` (DEFAULT -- the original behaviour; every existing checkpoint
  uses this): predict ``logI1`` and ``logI2`` as two independent channels. The
  log-ratio ``logI2 - logI1`` is a DIFFERENCE of two independent estimates.

* ``"delta"``: predict ``logI1`` and ``delta := logI2 - logI1`` directly, each with
  its own log-variance. ``logI2`` is DERIVED as ``logI1 + delta``. This is the fix
  motivated by ``docs/shrinkage_probe_findings.md``: the intensity head does
  conditional-mean regression that shrinks the WIDER-PSF protein channel under
  crowding (dense ch2 slope 0.70 vs ~1.0 elsewhere), and because the two channels
  shrink by DIFFERENT amounts (s1 - s2 = +0.28) that difference lands entirely on
  the ratio. Predicting ``delta`` directly makes the ratio the model's OWN
  estimand, so its error is no longer the residual of two separately-shrunk
  channels. ``delta`` is also intrinsically more identifiable under overlap: a
  contaminating neighbour bleeds into the SAME pixel in both co-located channels
  and largely cancels in the difference.

  Under ``"delta"`` the derived per-channel variance assumes ``logI1`` and
  ``delta`` errors are independent (which is the modelling assumption the whole
  reparameterisation rests on), giving ``var2 = var1 + var_delta`` i.e.
  ``logvar2 = logaddexp(logvar1, logvar_delta)``.

**IMPORTANT (frozen-rule note):** ``delta`` is a PER-SPOT supervised target,
identical in kind to ``logI1``/``logI2``. It is NOT the forbidden slope/alpha/beta
loss (CLAUDE.md Durable Rule 3): there is no cross-spot regression, no in-batch
slope, no size-dependent weighting, and the model still never sees alpha. See
``docs/intensity_head_fix_proposal.md`` Sec.4.

The state_dict is byte-identical between the two modes (both use a 2-channel
intensity conv and a 2-channel logvar conv); only the INTERPRETATION of the two
output channels differs. So a checkpoint trained in either mode loads without
surgery -- the mode travels in the saved ``model:`` config, not the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SpotHeads", "build_heads", "HEAD_PARAMETERISATIONS"]

HEAD_PARAMETERISATIONS = ("independent", "delta")


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
    """The four head groups; ``forward`` returns a dict of named output tensors.

    ``parameterisation`` selects how the two intensity-mean channels (and their two
    log-variance channels) are interpreted -- see the module docstring. The module
    structure is identical either way; only ``forward``'s bookkeeping differs.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 64,
        heatmap_bias: float = -2.19,
        parameterisation: str = "independent",
    ) -> None:
        super().__init__()
        if parameterisation not in HEAD_PARAMETERISATIONS:
            raise ValueError(
                f"head_parameterisation must be one of {HEAD_PARAMETERISATIONS}, "
                f"got {parameterisation!r}")
        self.parameterisation = parameterisation
        self.heatmap = _ConvHead(in_channels, 1, mid_channels)
        self.offset = _ConvHead(in_channels, 2, mid_channels)
        # Two channels either way: [logI1, logI2] (independent) or [logI1, delta] (delta).
        self.intensity = _ConvHead(in_channels, 2, mid_channels)
        self.logvar = _ConvHead(in_channels, 2, mid_channels)

        # CenterNet-style focal-loss prior: start with low foreground probability.
        nn.init.constant_(self.heatmap.final_conv.bias, float(heatmap_bias))

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        intensity = self.intensity(feat)
        logvar = self.logvar(feat)
        out = {
            "heatmap": self.heatmap(feat),           # logits [B,1,H,W]
            "offset": self.offset(feat),             # [B,2,H,W] (frac_x, frac_y)
        }

        if self.parameterisation == "independent":
            out["logI1"] = intensity[:, 0:1]
            out["logI2"] = intensity[:, 1:2]
            out["logvar1"] = logvar[:, 0:1]
            out["logvar2"] = logvar[:, 1:2]
            return out

        # "delta": native predictions are (logI1, delta); logI2 / logvar2 are DERIVED
        # so the schema-facing interface is unchanged. The loss picks up the native
        # (logI1, delta) via the extra keys below (intensity_nll auto-detects "delta").
        logI1 = intensity[:, 0:1]
        delta = intensity[:, 1:2]
        logvar1 = logvar[:, 0:1]
        logvar_delta = logvar[:, 1:2]

        out["logI1"] = logI1
        out["logI2"] = logI1 + delta                                  # DERIVED
        out["logvar1"] = logvar1
        # var2 = var1 + var_delta under the independence assumption -> logaddexp.
        out["logvar2"] = torch.logaddexp(logvar1, logvar_delta)        # DERIVED
        # Native training targets (ignored by predict_spots; used by intensity_nll):
        out["delta"] = delta
        out["logvar_delta"] = logvar_delta
        return out


def build_heads(in_channels: int, config: dict | None = None) -> SpotHeads:
    """Construct the head stack from a ``model:`` config block."""
    cfg = config or {}
    return SpotHeads(
        in_channels=in_channels,
        mid_channels=int(cfg.get("head_mid_channels", 64)),
        heatmap_bias=float(cfg.get("heatmap_bias", -2.19)),
        parameterisation=str(cfg.get("head_parameterisation", "independent")),
    )
