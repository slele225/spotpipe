"""Prediction heads (STUB -- implemented in build stage 3).

Heads include: detection, sub-pixel localization, per-channel log-intensity
(logI1/logI2), and per-channel log-variance for heteroscedastic uncertainty.
The intensity heads regress each spot's TOTAL INTEGRATED log-intensity directly
at the spot center; the log-variance heads feed a Gaussian NLL and populate the
uncertainty1/uncertainty2 schema columns.
"""

from __future__ import annotations


def build_heads(*args, **kwargs):
    """Construct the prediction heads. STUB."""
    raise NotImplementedError("heads.build_heads is implemented in build stage 3.")
