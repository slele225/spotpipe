"""Background generation for the FV3000 forward model.

Background is a SCENE parameter, randomised widely as domain randomisation
(see CLAUDE.md). It is produced in **photon-proportional units** and added to
the clean spot signal *before* the detector chain (shot noise onward), so the
background also gets shot-noise, gain, and offset like any other photon flux.

Two sources are supported:

* :func:`make_parametric_background` -- flat level + linear gradient +
  low-frequency smooth structure. This is what phase 1 uses. Always
  non-negative (a photon rate cannot be negative).
* :func:`sample_real_background_crop` -- a *defined but stubbed* hook to later
  sample real background crops, which carry the line/row-correlated noise of
  the scanning detector. Implemented in the real-data phase; raises
  ``NotImplementedError`` for now.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["make_parametric_background", "sample_real_background_crop", "make_background"]


def make_parametric_background(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    level: float,
    gradient_frac: float = 0.0,
    structure_frac: float = 0.0,
    structure_scale_px: float = 32.0,
) -> np.ndarray:
    """Parametric background field in photon-proportional units (>= 0).

    Parameters
    ----------
    shape : (H, W).
    rng : per-image random generator.
    level : mean flat background (photons).
    gradient_frac : peak-to-peak amplitude of a linear gradient, as a fraction
        of ``level``; the gradient direction is random.
    structure_frac : amplitude of smooth low-frequency structure, as a fraction
        of ``level`` (std of the structure field ~ ``structure_frac * level``).
    structure_scale_px : correlation length of the low-frequency structure
        (larger == smoother / lower spatial frequency).

    The three components model, respectively, a uniform offset, uneven
    illumination, and out-of-focus / autofluorescence structure. The result is
    clipped at 0.
    """
    height, width = shape
    bg = np.full(shape, float(level), dtype=np.float64)

    # Linear gradient with random orientation, peak-to-peak == gradient_frac*level.
    if gradient_frac > 0.0:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        ramp = np.cos(theta) * (xx / max(width - 1, 1)) + np.sin(theta) * (yy / max(height - 1, 1))
        ramp -= ramp.mean()  # zero-mean so it tilts about `level`, not shifts it
        span = ramp.max() - ramp.min()
        if span > 0:
            bg += (gradient_frac * level) * (ramp / span)

    # Low-frequency structure: smooth a white field to the requested scale.
    if structure_frac > 0.0:
        noise = rng.standard_normal(shape)
        smooth = gaussian_filter(noise, sigma=float(structure_scale_px), mode="reflect")
        std = smooth.std()
        if std > 0:
            smooth /= std  # unit std, then scale to the requested amplitude
        bg += (structure_frac * level) * smooth

    np.clip(bg, 0.0, None, out=bg)
    return bg


def sample_real_background_crop(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    crop_library: str | None = None,
    **kwargs,
) -> np.ndarray:
    """Sample a real background crop carrying scanner line/row-correlated noise.

    DEFINED-BUT-STUBBED hook for the real-data phase. Real FV3000 backgrounds
    carry row-correlated noise from the line-scanning detector that parametric
    fields do not reproduce; phase-2 calibration will sample crops from a
    library of empty real fields. Interface is fixed now so the forward model
    can switch sources without changing call sites.
    """
    raise NotImplementedError(
        "Real background crops are implemented in the real-data phase. "
        "Use make_parametric_background for phase 1."
    )


def make_background(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    source: str = "parametric",
    **params,
) -> np.ndarray:
    """Dispatch to a background source (``parametric`` for phase 1)."""
    if source == "parametric":
        return make_parametric_background(shape, rng, **params)
    if source == "real_crop":
        return sample_real_background_crop(shape, rng, **params)
    raise ValueError(f"unknown background source: {source!r}")
