"""Intensity extraction: the shared measurement instrument.

Given an image, a list of spot positions, and a PSF ``sigma``, return an
integrated intensity per spot per channel. This is the *one* instrument every
baseline and our own model's "shared" intensity numbers flow through, so a bug
here biases every tool at once -- hence the golden test in
``tests/test_intensity_extraction.py`` that this module must satisfy.

This is called *intensity extraction* in the SMLM literature (not
"photometry"). Two standard methods, both provided here:

* ``aperture``  -- sum pixels in a fixed-radius ROI, minus a robust local
  background (aka ROI / summed intensity).
* ``gaussian``  -- fit a 2-D Gaussian with **fixed** sigma; the intensity is the
  fitted volume (== ``2*pi*sigma^2 * peak``; here fitted directly as the volume
  of an area-normalised kernel, which is the same quantity).

Design rules (enforced, see the module prompt / CLAUDE.md):

1. ``sigma`` is a REQUIRED input with NO default. On simulated benchmark data it
   is the simulator's true sigma; on real data it is the bead-calibrated sigma.
   This module never invents or hardcodes a sigma.
2. ONE source of truth for PSF width. From the single ``sigma`` we DERIVE the
   aperture radius (``3*sigma``) and the background annulus (inner ``4*sigma``,
   outer ``6*sigma``). Those multipliers live in :class:`ExtractionConfig`; the
   gaussian method fits with the SAME sigma, so the two methods can never
   disagree about PSF width.
3. Local background is the MEDIAN of pixels in the annulus (robust to
   neighbouring spots). Aperture intensity = ``sum(aperture) - median * n_ap``.
4. Gaussian fit: sigma FIXED; only volume, background, and the sub-pixel centre
   are free. A fit that fails to converge records NaN + a ``fit_failed`` flag,
   never a crash.
5. Per channel: channels are extracted independently (each with its own sigma),
   both returned.
6. Edge handling: a spot whose annulus/crop extends past the image border
   records NaN + an ``edge`` flag rather than silently truncating (truncation
   biases the intensity low).
7. Vectorised: the aperture method is fully vectorised across spots; the
   gaussian method loops over spots but only over small per-spot crops.

Units: intensities are returned in LINEAR photon-proportional units. The log is
taken downstream to fill the schema's ``logI1``/``logI2`` fields. This module
never writes schema columns; the caller decides which file the numbers land in
(the native-vs-shared distinction is handled by separate files downstream).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

__all__ = [
    "ExtractionConfig",
    "ChannelIntensities",
    "TwoChannelIntensities",
    "extract_channel",
    "extract_intensities",
]

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Config: PSF-width multipliers (ONE sigma is the source of truth)            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtractionConfig:
    """Aperture/annulus/crop geometry, all as multiples of the single ``sigma``.

    These multipliers are the only knobs; every pixel radius is
    ``multiplier * sigma`` so the aperture and gaussian methods share one PSF
    width by construction.
    """

    aperture_radius_sigma: float = 3.0   # ROI radius
    annulus_inner_sigma: float = 4.0     # background annulus inner radius
    annulus_outer_sigma: float = 6.0     # background annulus outer radius
    crop_halfwidth_sigma: float = 6.0    # gaussian-fit crop half-width
    fit_max_iter: int = 12               # Gauss-Newton iterations per spot (centre)

    def __post_init__(self) -> None:
        if not (0 < self.aperture_radius_sigma <= self.annulus_inner_sigma):
            raise ValueError("need 0 < aperture_radius_sigma <= annulus_inner_sigma")
        if not (self.annulus_inner_sigma < self.annulus_outer_sigma):
            raise ValueError("need annulus_inner_sigma < annulus_outer_sigma")
        if self.crop_halfwidth_sigma < self.annulus_outer_sigma:
            raise ValueError("crop_halfwidth_sigma should cover the annulus")


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ChannelIntensities:
    """Per-spot extracted intensities for a single channel.

    ``intensity`` is linear photon-proportional flux, NaN where extraction was
    refused (edge) or failed (fit_failed). ``flags[i]`` is a comma-joined string
    of status flags for spot ``i`` (``""`` when clean).
    """

    intensity: np.ndarray   # (N,) float, linear units; NaN on edge/fit_failed
    flags: list[str]        # (N,) per-spot flags
    method: str
    sigma: float


@dataclass
class TwoChannelIntensities:
    """Extracted intensities for both channels of one image."""

    ch1: ChannelIntensities
    ch2: ChannelIntensities

    @property
    def I1(self) -> np.ndarray:  # noqa: E743 - matches schema field name
        return self.ch1.intensity

    @property
    def I2(self) -> np.ndarray:  # noqa: E743
        return self.ch2.intensity


# --------------------------------------------------------------------------- #
# Integrated-Gaussian kernel (matches the vendored forward model's rendering) #
# --------------------------------------------------------------------------- #
def _integrated_gaussian_1d(center: float, edges: np.ndarray, sigma: float) -> np.ndarray:
    """Per-pixel mass of a unit-area 1-D Gaussian over pixels defined by ``edges``.

    ``edges`` are the shared pixel boundaries (length ``M+1`` for ``M`` pixels;
    pixel ``i`` spans ``[edges[i], edges[i+1])``). Returns the ``M`` per-pixel
    weights ``Phi(edges[i+1]) - Phi(edges[i])``. This mirrors the simulator's
    exact erf-integrated rendering, so fitting recovers the true volume without
    the sigma-dependent bias that point-sampling a ~1 px PSF would introduce.
    """
    cdf = 0.5 * (1.0 + erf((edges - center) / (_SQRT2 * sigma)))
    return np.diff(cdf)


def _integrated_gaussian_1d_grad(center: float, edges: np.ndarray, sigma: float) -> np.ndarray:
    """d/d(center) of :func:`_integrated_gaussian_1d`.

    ``d/dc [Phi(e) - Phi(e')] = N(e'; c, sigma) - N(e; c, sigma)`` where ``N`` is
    the Normal pdf; used to give ``curve_fit`` an analytic Jacobian for the
    sub-pixel centre so it converges in a handful of iterations.
    """
    pdf = (_INV_SQRT_2PI / sigma) * np.exp(-0.5 * ((edges - center) / sigma) ** 2)
    return pdf[:-1] - pdf[1:]


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #
def _validate(xs: np.ndarray, ys: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a positive finite number, got {sigma!r}")
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"xs and ys must have the same length, got {xs.shape} vs {ys.shape}")
    return xs, ys


def _append_flag(flags: list[str], i: int, tag: str) -> None:
    flags[i] = tag if not flags[i] else f"{flags[i]},{tag}"


# --------------------------------------------------------------------------- #
# Aperture method (fully vectorised across spots)                             #
# --------------------------------------------------------------------------- #
def _extract_aperture(
    plane: np.ndarray, xs: np.ndarray, ys: np.ndarray, sigma: float, cfg: ExtractionConfig
) -> tuple[np.ndarray, list[str]]:
    height, width = plane.shape
    n = xs.size
    intensity = np.full(n, np.nan, dtype=np.float64)
    flags = [""] * n
    if n == 0:
        return intensity, flags

    r_ap = cfg.aperture_radius_sigma * sigma
    r_in = cfg.annulus_inner_sigma * sigma
    r_out = cfg.annulus_outer_sigma * sigma
    radius = int(np.ceil(r_out))

    cx = np.round(xs).astype(np.int64)
    cy = np.round(ys).astype(np.int64)

    # Edge = the crop box that must contain the whole annulus falls off-image.
    edge = (cx - radius < 0) | (cx + radius >= width) | (cy - radius < 0) | (cy + radius >= height)
    for i in np.nonzero(edge)[0]:
        _append_flag(flags, int(i), "edge")

    good = np.nonzero(~edge)[0]
    if good.size == 0:
        return intensity, flags

    off = np.arange(-radius, radius + 1)
    oy, ox = np.meshgrid(off, off, indexing="ij")  # (K, K)

    gcx = cx[good][:, None, None]
    gcy = cy[good][:, None, None]
    col = gcx + ox[None]          # (M, K, K) absolute column of each crop pixel
    row = gcy + oy[None]          # (M, K, K) absolute row
    vals = plane[row, col].astype(np.float64)  # in-bounds by construction

    dx = col - xs[good][:, None, None]
    dy = row - ys[good][:, None, None]
    d2 = dx * dx + dy * dy

    ap_mask = d2 <= r_ap * r_ap
    an_mask = (d2 > r_in * r_in) & (d2 <= r_out * r_out)

    n_ap = ap_mask.sum(axis=(1, 2))
    ap_sum = np.where(ap_mask, vals, 0.0).sum(axis=(1, 2))
    # Median over the annulus per spot (robust to a neighbour leaking into it).
    an_vals = np.where(an_mask, vals, np.nan)
    bg_med = np.nanmedian(an_vals, axis=(1, 2))

    intensity[good] = ap_sum - bg_med * n_ap
    return intensity, flags


# --------------------------------------------------------------------------- #
# Gaussian method (loops over small per-spot crops)                           #
# --------------------------------------------------------------------------- #
def _solve_volume_bg(kernel: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form linear least squares of ``y ~ V*kernel + bg``.

    With sigma FIXED and the centre held, the model is linear in ``(V, bg)`` --
    a 2x2 normal-equation solve, no iteration. Returns ``(V, bg, det)``; a
    non-positive ``det`` signals a degenerate crop (caller treats as failure).
    """
    n = y.size
    skk = float(kernel @ kernel)
    sk1 = float(kernel.sum())
    sky = float(kernel @ y)
    s1y = float(y.sum())
    det = skk * n - sk1 * sk1
    if det <= 0:
        return np.nan, np.nan, det
    volume = (n * sky - sk1 * s1y) / det
    bg = (skk * s1y - sk1 * sky) / det
    return volume, bg, det


def _fit_one_gaussian(
    crop: np.ndarray,
    col_edges: np.ndarray,
    row_edges: np.ndarray,
    x0: float,
    y0: float,
    sigma: float,
    max_iter: int,
) -> float:
    """Fit ``bg + V * G(., sigma)`` to one crop; return fitted volume V (or NaN).

    Free parameters: ``V`` (volume == integrated intensity), ``bg`` (flat
    background), and the sub-pixel centre ``(cx, cy)``. ``sigma`` is FIXED.

    Because sigma is fixed the model is *linear* in ``(V, bg)`` once the centre
    is held, and only the centre is nonlinear. We exploit that with a small
    Gauss-Newton loop: each iteration solves ``(V, bg)`` in closed form at the
    current centre (:func:`_solve_volume_bg`), then takes one GN step on the
    centre using the analytic kernel gradients. This converges in a handful of
    iterations from the detection centre and costs microseconds -- vastly faster
    than a general nonlinear optimiser per spot, with the same fixed-sigma model.

    The kernel is the area-normalised erf-integrated Gaussian, so V is the
    integrated intensity directly (equal to ``2*pi*sigma^2 * peak``). NaN on a
    degenerate solve, a centre driven out of the crop, or a non-finite/negative
    volume.
    """
    y = crop.ravel()
    cx, cy = x0, y0
    volume = np.nan

    for _ in range(max_iter):
        ix = _integrated_gaussian_1d(cx, col_edges, sigma)
        iy = _integrated_gaussian_1d(cy, row_edges, sigma)
        kernel = np.outer(iy, ix).ravel()

        volume, bg, det = _solve_volume_bg(kernel, y)
        if not np.isfinite(det) or det <= 0 or not np.isfinite(volume):
            return np.nan

        # Gauss-Newton step on the sub-pixel centre (2x2 normal equations).
        dix = _integrated_gaussian_1d_grad(cx, col_edges, sigma)
        diy = _integrated_gaussian_1d_grad(cy, row_edges, sigma)
        jx = (volume * np.outer(iy, dix)).ravel()
        jy = (volume * np.outer(diy, ix)).ravel()
        resid = y - (bg + volume * kernel)

        jxx = float(jx @ jx)
        jyy = float(jy @ jy)
        jxy = float(jx @ jy)
        bx = float(jx @ resid)
        by = float(jy @ resid)
        d = jxx * jyy - jxy * jxy
        if d <= 0:
            break  # centre unidentifiable (e.g. V~0); keep the linear solve
        dcx = (jyy * bx - jxy * by) / d
        dcy = (jxx * by - jxy * bx) / d

        # Clamp the step to <= 1 px so a bad crop can't fling the centre away.
        step = np.hypot(dcx, dcy)
        if step > 1.0:
            dcx /= step
            dcy /= step
        cx += dcx
        cy += dcy
        if abs(dcx) + abs(dcy) < 1e-3:
            break
        # Bail immediately once the centre leaves the crop: it is diverging onto a
        # neighbour/artefact and will be flagged failed -- no point iterating on.
        if not (col_edges[0] <= cx <= col_edges[-1] and row_edges[0] <= cy <= row_edges[-1]):
            return np.nan

    # Final linear solve at the converged centre for a consistent (V, bg).
    ix = _integrated_gaussian_1d(cx, col_edges, sigma)
    iy = _integrated_gaussian_1d(cy, row_edges, sigma)
    volume, _bg, det = _solve_volume_bg(np.outer(iy, ix).ravel(), y)

    if not np.isfinite(det) or det <= 0 or not np.isfinite(volume) or volume < 0:
        return np.nan
    # A centre that fled the crop means the fit latched onto a neighbour/artefact.
    if not (col_edges[0] <= cx <= col_edges[-1] and row_edges[0] <= cy <= row_edges[-1]):
        return np.nan
    return float(volume)


def _extract_gaussian(
    plane: np.ndarray, xs: np.ndarray, ys: np.ndarray, sigma: float, cfg: ExtractionConfig
) -> tuple[np.ndarray, list[str]]:
    height, width = plane.shape
    n = xs.size
    intensity = np.full(n, np.nan, dtype=np.float64)
    flags = [""] * n
    if n == 0:
        return intensity, flags

    crop_r = int(np.ceil(cfg.crop_halfwidth_sigma * sigma))

    cx = np.round(xs).astype(np.int64)
    cy = np.round(ys).astype(np.int64)
    edge = (cx - crop_r < 0) | (cx + crop_r >= width) | (cy - crop_r < 0) | (cy + crop_r >= height)

    off = np.arange(-crop_r, crop_r + 1)

    for i in range(n):
        if edge[i]:
            _append_flag(flags, i, "edge")
            continue

        rows = cy[i] + off
        cols = cx[i] + off
        crop = plane[np.ix_(rows, cols)].astype(np.float64)

        # Pixel i spans [i-0.5, i+0.5): shared edges are the integer coords ± 0.5.
        # (V, bg) are solved in closed form inside the fitter, so no init guess is
        # needed; the sub-pixel centre just starts at the detection position.
        col_edges = np.concatenate([cols - 0.5, [cols[-1] + 0.5]])
        row_edges = np.concatenate([rows - 0.5, [rows[-1] + 0.5]])

        vol = _fit_one_gaussian(
            crop, col_edges, row_edges, float(xs[i]), float(ys[i]),
            sigma, cfg.fit_max_iter,
        )
        if np.isnan(vol):
            _append_flag(flags, i, "fit_failed")
        intensity[i] = vol

    return intensity, flags


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
_METHODS = {"aperture": _extract_aperture, "gaussian": _extract_gaussian}


def extract_channel(
    plane: np.ndarray,
    xs,
    ys,
    sigma: float,
    method: str,
    config: ExtractionConfig | None = None,
) -> ChannelIntensities:
    """Extract per-spot integrated intensity from ONE 2-D image plane.

    Parameters
    ----------
    plane : 2-D array indexed ``[row=y, col=x]`` (a single channel).
    xs, ys : sub-pixel spot centres (columns / rows), in pixels.
    sigma : PSF width in pixels. REQUIRED, no default (design rule 1).
    method : ``"aperture"`` or ``"gaussian"``.
    config : geometry multipliers; defaults to :class:`ExtractionConfig`.
    """
    plane = np.asarray(plane)
    if plane.ndim != 2:
        raise ValueError(f"plane must be 2-D [row, col], got shape {plane.shape}")
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")
    cfg = config or ExtractionConfig()
    xs, ys = _validate(xs, ys, sigma)
    intensity, flags = _METHODS[method](plane, xs, ys, float(sigma), cfg)
    return ChannelIntensities(intensity=intensity, flags=flags, method=method, sigma=float(sigma))


def extract_intensities(
    image: np.ndarray,
    xs,
    ys,
    sigma1: float,
    sigma2: float,
    method: str,
    config: ExtractionConfig | None = None,
) -> TwoChannelIntensities:
    """Extract per-spot intensities from a two-channel image.

    ``image`` is channel-first ``[2, H, W]`` (the forward model's layout). The
    two channels are extracted INDEPENDENTLY, each with its own PSF ``sigma``
    (channels have a deliberate width mismatch upstream), and both are returned.
    Both sigmas are REQUIRED. The same spot positions ``xs``/``ys`` are used for
    both channels.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[0] != 2:
        raise ValueError(f"image must be channel-first [2, H, W], got shape {image.shape}")
    ch1 = extract_channel(image[0], xs, ys, sigma1, method, config)
    ch2 = extract_channel(image[1], xs, ys, sigma2, method, config)
    return TwoChannelIntensities(ch1=ch1, ch2=ch2)
