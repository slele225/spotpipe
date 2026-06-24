"""Point-spread function model for the FV3000 forward model.

PSF width (``sigma``) is a SCENE parameter, randomised widely (see
``forward_model.sample_scene_params``), including a deliberate
channel-1-vs-channel-2 width mismatch and a small per-channel registration
shift. Those are decided by the caller; this module only knows how to *render*
one spot given its sub-pixel centre, total integrated intensity, and sigma.

Model and conventions
----------------------
* The PSF is an isotropic 2-D Gaussian, **area-normalised** so that summing the
  rendered spot over pixels recovers the spot's total integrated intensity
  ``A`` (the photon-proportional ground-truth label the network later regresses
  -- see CLAUDE.md). ``A`` itself is stored exactly and is never recovered from
  the rendered pixels; the rendering only needs to *conserve* it.
* We do NOT sample the Gaussian at pixel centres. For a diffraction-limited PSF
  barely ~1 px wide, point-sampling badly mis-counts flux and is sigma-biased.
  Instead we **integrate** the continuous Gaussian over each pixel's extent
  using the error function -- exact, and it conserves total intensity (up to a
  ~1e-4 truncation at the rendering window edge; see ``truncate``).
* Pixel ``(px, py)`` covers the unit square ``[px-0.5, px+0.5) x
  [py-0.5, py+0.5)``. Image axis order is ``[row=y, col=x]`` throughout.

The Gaussian kernel is factored behind :class:`GaussianPSF` / :func:`render_psf`
so a *measured* PSF (from bead images) can replace it later without touching the
forward model: ``sigma`` will eventually come from bead calibration rather than
being randomised. A future ``MeasuredPSF`` implementing the same
``render_into`` contract would drop straight in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

__all__ = ["GaussianPSF", "render_psf", "render_channel", "gaussian_peak_fraction"]

_SQRT2 = np.sqrt(2.0)


def _integrated_gaussian_1d(center: float, lo: int, hi: int, sigma: float) -> np.ndarray:
    """Exact per-pixel integral of a unit-area 1-D Gaussian over pixels ``[lo, hi)``.

    Pixel ``i`` covers ``[i-0.5, i+0.5)``; the returned weight is the mass of a
    ``Normal(center, sigma)`` falling inside that pixel, computed from the Gaussian
    CDF at the *shared* pixel edges (one erf per edge, not per pixel)::

        w_i = Phi(i+0.5) - Phi(i-0.5),  Phi(z) = 0.5*(1 + erf((z-center)/(sqrt(2) sigma)))

    Returns an array of length ``hi - lo``. The weights sum to ~1 over an
    infinite range; over the finite ``[lo, hi)`` window they sum to the fraction
    of the Gaussian captured by the window.
    """
    edges = np.arange(lo, hi + 1, dtype=np.float64) - 0.5  # hi-lo+1 shared edges
    cdf = 0.5 * (1.0 + erf((edges - center) / (_SQRT2 * sigma)))
    return np.diff(cdf)  # length hi-lo


def gaussian_peak_fraction(sigma: float) -> float:
    """Peak value of a unit-*area* 2-D Gaussian: ``1 / (2*pi*sigma^2)``.

    The on-axis pixel value of a rendered spot of intensity ``A`` is
    approximately ``A * gaussian_peak_fraction(sigma)`` (exact in the
    continuous limit; the integrated kernel is within sub-percent of it). Used
    to estimate per-spot peak signal for saturation flagging.
    """
    return 1.0 / (2.0 * np.pi * sigma * sigma)


@dataclass(frozen=True)
class GaussianPSF:
    """An isotropic, area-normalised Gaussian PSF for one channel.

    ``sigma`` is in pixels. ``truncate`` sets the half-window in units of sigma
    (4 sigma captures ~0.99987 of the 2-D mass; the lost tail is negligible and
    the exact ``A`` label is stored separately regardless).
    """

    sigma: float
    truncate: float = 4.0

    def render_into(self, canvas: np.ndarray, x: float, y: float, amplitude: float) -> None:
        """Add one area-normalised spot of total intensity ``amplitude`` at sub-pixel ``(x, y)``.

        ``canvas`` is modified in place (added to). Spots whose window falls
        entirely off-canvas contribute nothing.
        """
        render_psf(canvas, x, y, amplitude, self.sigma, truncate=self.truncate)


def render_psf(
    canvas: np.ndarray,
    x: float,
    y: float,
    amplitude: float,
    sigma: float,
    *,
    truncate: float = 4.0,
) -> None:
    """Add one area-normalised Gaussian spot into ``canvas`` (in place).

    Parameters
    ----------
    canvas : 2-D float array, indexed ``[row=y, col=x]``; modified in place.
    x, y : sub-pixel spot centre (column, row), in pixels.
    amplitude : total integrated intensity ``A`` (photon-proportional units).
        The rendered window sums to ``A`` up to the ``truncate`` tail loss.
    sigma : Gaussian width in pixels.
    truncate : rendering half-window in units of sigma.
    """
    height, width = canvas.shape
    radius = max(int(np.ceil(truncate * sigma)), 1)

    cx, cy = int(round(x)), int(round(y))
    xlo, xhi = max(cx - radius, 0), min(cx + radius + 1, width)
    ylo, yhi = max(cy - radius, 0), min(cy + radius + 1, height)
    if xlo >= xhi or ylo >= yhi:
        return  # spot window entirely off-canvas

    wx = _integrated_gaussian_1d(x, xlo, xhi, sigma)  # length xhi-xlo
    wy = _integrated_gaussian_1d(y, ylo, yhi, sigma)  # length yhi-ylo
    canvas[ylo:yhi, xlo:xhi] += amplitude * np.outer(wy, wx)


def render_channel(
    shape: tuple[int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    amplitudes: np.ndarray,
    sigma: float,
    *,
    shift: tuple[float, float] = (0.0, 0.0),
    truncate: float = 4.0,
) -> np.ndarray:
    """Render every spot of one channel onto a fresh zero canvas and return it.

    ``shift = (dx, dy)`` is the per-channel registration offset added to every
    spot centre (channel misregistration). The returned array is the clean
    photon-proportional spot signal for the channel (background added by the
    caller).
    """
    canvas = np.zeros(shape, dtype=np.float64)
    dx, dy = shift
    for x, y, a in zip(xs, ys, amplitudes):
        render_psf(canvas, float(x) + dx, float(y) + dy, float(a), sigma, truncate=truncate)
    return canvas
