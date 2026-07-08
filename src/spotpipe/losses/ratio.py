"""Ratio / slope analysis helpers (STUB -- implemented later).

IMPORTANT (see CLAUDE.md): there is NO slope loss in phase 1. The ratio-law
slope beta is computed EXPLICITLY downstream of inference from per-spot
logI1/logI2 -- never trained on. An in-batch slope would be a biased
(attenuated) estimator because the regressor (predicted logI1) carries error,
and training on it could distort the per-spot intensities whose unbiasedness
the project must demonstrate. This module is for the downstream, post-inference
slope computation only; slope supervision may be revisited only after per-spot
estimates are shown unbiased on their own.
"""

from __future__ import annotations


def fit_slope(*args, **kwargs):
    """Downstream, post-inference ratio-law slope fit (NOT a training loss). STUB."""
    raise NotImplementedError("ratio.fit_slope is implemented in a later build stage.")
