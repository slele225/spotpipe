"""Naive in-repo baselines that run TODAY and emit the canonical schema.

We do not yet have DECODE / Spotiflow installed (those are stubbed adapters), so
the harness is validated against real -- if simple -- competitors rather than
stubs. Both baselines estimate each channel INDEPENDENTLY and then DIVIDE, which
is exactly the approach the project expects to beat in the dim x high-overlap
corner: a per-channel window readout is contaminated by overlapping neighbours
and biased by per-channel saturation, and dividing two noisy channel estimates
inflates and biases the log-ratio at low SNR. The network, by contrast, regresses
each spot's total integrated intensity directly and learns to deblend.

* ``classical_per_channel_aperture`` -- classical detection (Laplacian-of-Gaussian
  local maxima) run on EACH channel independently and merged, then APERTURE
  photometry in each channel independently, then ratio = I2 / I1. The honest naive
  baseline. (Aperture/naive photometry -- NOT a PSF fit.)
* ``oracle_center_aperture_divide`` -- the same APERTURE photometry but read at the
  GROUND-TRUTH centres (perfect detection + localization), isolating the divide
  step alone: even with perfect centres, dividing two noisy per-channel aperture
  reads biases the ratio at low SNR. (Oracle CENTRES only -- the intensities are
  still naive aperture reads, NOT oracle intensities.)

Both convert counts to photon-proportional units with the KNOWN per-channel gain
and subtract a local background (which removes the per-channel offset pedestal,
per CLAUDE.md, "offset subtracted before any ratio or log is taken"). The exact
chain per spot per channel is:

    observed counts -> subtract per-channel pedestal+background (annulus median,
    which contains the per-channel offset) -> divide by per-channel gain ->
    aperture-integrated photon-proportional intensity -> log / ratio.

Raw detector counts are NEVER divided directly. Using the known detector
constants is fair here: this is a synthetic benchmark whose point is the divide /
contamination failure, not a units mismatch. Neither baseline fills
``uncertainty*`` or ``sigma*_hat`` (left NaN): baselines structurally cannot offer
the per-spot uncertainty our model does.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

from spotpipe.benchmark.features import axis_params_from_meta
from spotpipe.schema import SpotRecord, records_to_dataframe

__all__ = [
    "classical_per_channel_aperture",
    "oracle_center_aperture_divide",
    "aperture_photometry",
]

_EPS = 1e-6


def _channel_gains(meta: dict) -> tuple[float, float]:
    det = meta.get("detector", {})
    g1 = float(det.get("ch1", {}).get("gain", 1.0))
    g2 = float(det.get("ch2", {}).get("gain", 1.0))
    return g1, g2


def aperture_photometry(
    channel: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    r_ap: float,
    r_in: float,
    r_out: float,
    gain: float,
) -> np.ndarray:
    """Aperture photometry: integrated photon-proportional intensity per spot.

    For each centre, sum the counts in a disc of radius ``r_ap``, subtract a local
    background estimated as the MEDIAN of an annulus ``[r_in, r_out)`` (this
    removes the per-channel OFFSET pedestal + background + gradient robustly),
    then divide by the per-channel ``gain`` to land in photon-proportional units.
    Clipped at a tiny positive floor so ``log`` is always defined.

    Audit (requirement #2): the chain is observed counts -> subtract per-channel
    offset/background -> divide by per-channel gain -> integrated intensity. Raw
    detector counts are never divided directly (the ``/ gain`` acts on the
    background-subtracted aperture signal, not on raw counts).
    """
    channel = np.asarray(channel, dtype=float)
    h, w = channel.shape
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    rad = int(math.ceil(r_out)) + 1
    out = np.empty(xs.size, dtype=float)

    for k in range(xs.size):
        cx, cy = xs[k], ys[k]
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = max(ix - rad, 0), min(ix + rad + 1, w)
        y0, y1 = max(iy - rad, 0), min(iy + rad + 1, h)
        patch = channel[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        ap = dist <= r_ap
        ann = (dist >= r_in) & (dist < r_out)
        bg = np.median(patch[ann]) if np.any(ann) else 0.0
        signal_counts = float(patch[ap].sum() - bg * np.count_nonzero(ap))
        out[k] = max(signal_counts, _EPS) / max(gain, _EPS)
    return out


def _emit(image_id, xs, ys, I1, I2, p_detect, flags) -> pd.DataFrame:
    """Build canonical-schema rows from per-spot intensities (uncertainty NaN)."""
    records = []
    p_detect = np.atleast_1d(p_detect) if np.ndim(p_detect) else np.full(len(xs), float(p_detect))
    for k in range(len(xs)):
        records.append(
            SpotRecord.from_logs(
                image_id=image_id,
                spot_id=k,
                x=float(xs[k]),
                y=float(ys[k]),
                p_detect=float(p_detect[k]),
                logI1=float(math.log(max(I1[k], _EPS))),
                logI2=float(math.log(max(I2[k], _EPS))),
                sigma1_hat=math.nan,           # baseline does not estimate PSF width
                sigma2_hat=math.nan,
                uncertainty1=math.nan,         # baselines emit no uncertainty
                uncertainty2=math.nan,
                flags=flags,
            )
        )
    return records_to_dataframe(records)


def _aperture_radii(cfg: dict) -> tuple[float, float, float]:
    r_ap = float(cfg.get("window_radius_px", 3.0))
    r_in = float(cfg.get("bg_inner_px", r_ap + 1.0))
    r_out = float(cfg.get("bg_outer_px", r_ap + 4.0))
    return r_ap, r_in, r_out


# --------------------------------------------------------------------------- #
# oracle_center_aperture_divide                                                #
# --------------------------------------------------------------------------- #
def oracle_center_aperture_divide(image: np.ndarray, gt: pd.DataFrame, meta: dict, *, image_id: str, cfg: dict | None = None) -> pd.DataFrame:
    """Aperture photometry at the GROUND-TRUTH (oracle) centres, then divide.

    Perfect detection + localization by construction (one prediction per GT spot,
    at its true centre), so this isolates the divide step: any ratio bias /
    variance it shows comes purely from reading and dividing two noisy per-channel
    APERTURE intensities (plus per-channel saturation), NOT from detection or
    localization. Only the CENTRES are oracle; the intensities are naive aperture
    reads (offset/background-subtracted, gain-divided), not oracle intensities.
    """
    cfg = cfg or {}
    image = np.asarray(image, dtype=float)
    g1, g2 = _channel_gains(meta)
    r_ap, r_in, r_out = _aperture_radii(cfg)

    xs = gt["x"].to_numpy(float)
    ys = gt["y"].to_numpy(float)
    if xs.size == 0:
        return records_to_dataframe([])

    I1 = aperture_photometry(image[0], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=g1)
    I2 = aperture_photometry(image[1], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=g2)
    return _emit(image_id, xs, ys, I1, I2, p_detect=1.0, flags="baseline,oracle_center_aperture_divide")


# --------------------------------------------------------------------------- #
# classical_per_channel_aperture                                               #
# --------------------------------------------------------------------------- #
def _detect_channel(channel: np.ndarray, *, smooth_sigma: float, threshold_rel: float, footprint: int):
    """LoG-style local maxima on one channel: (xs, ys, response) above threshold."""
    sm = gaussian_filter(channel, smooth_sigma)
    # Robust background + noise from the smoothed image (MAD-based).
    med = float(np.median(sm))
    mad = float(np.median(np.abs(sm - med))) + _EPS
    robust_std = 1.4826 * mad
    threshold = med + threshold_rel * robust_std

    local_max = maximum_filter(sm, size=footprint) == sm
    keep = local_max & (sm > threshold)
    ys, xs = np.where(keep)
    return xs.astype(float), ys.astype(float), sm[ys, xs] - med


def _merge_detections(cands: list[tuple[np.ndarray, np.ndarray, np.ndarray]], min_sep: float):
    """Greedy proximity merge of per-channel detections (highest response wins)."""
    if not cands:
        return np.empty(0), np.empty(0), np.empty(0)
    xs = np.concatenate([c[0] for c in cands]) if cands else np.empty(0)
    ys = np.concatenate([c[1] for c in cands])
    resp = np.concatenate([c[2] for c in cands])
    if xs.size == 0:
        return xs, ys, resp

    order = np.argsort(resp)[::-1]
    kept_x, kept_y, kept_r = [], [], []
    for i in order:
        x, y = xs[i], ys[i]
        if kept_x:
            d2 = (np.array(kept_x) - x) ** 2 + (np.array(kept_y) - y) ** 2
            if np.any(d2 < min_sep * min_sep):
                continue
        kept_x.append(x); kept_y.append(y); kept_r.append(resp[i])
    return np.array(kept_x), np.array(kept_y), np.array(kept_r)


def classical_per_channel_aperture(image: np.ndarray, meta: dict, *, image_id: str, cfg: dict | None = None) -> pd.DataFrame:
    """Classical detection per channel (merged) + per-channel aperture divide.

    Detection runs INDEPENDENTLY on each channel (LoG local maxima above a
    robust threshold); the two candidate sets are merged by proximity. Intensity
    is then read INDEPENDENTLY per channel by aperture photometry at the merged
    centres, and the ratio is the quotient -- the naive "estimate each channel,
    then divide" pipeline.
    """
    cfg = cfg or {}
    image = np.asarray(image, dtype=float)
    g1, g2 = _channel_gains(meta)
    r_ap, r_in, r_out = _aperture_radii(cfg)

    smooth_sigma = float(cfg.get("detect_smooth_sigma", 1.2))
    threshold_rel = float(cfg.get("detect_threshold_rel", 4.0))
    footprint = int(cfg.get("detect_footprint_px", 3))
    min_sep = float(cfg.get("min_separation_px", 2.0))

    d1 = _detect_channel(image[0], smooth_sigma=smooth_sigma, threshold_rel=threshold_rel, footprint=footprint)
    d2 = _detect_channel(image[1], smooth_sigma=smooth_sigma, threshold_rel=threshold_rel, footprint=footprint)
    xs, ys, resp = _merge_detections([d1, d2], min_sep)
    if xs.size == 0:
        return records_to_dataframe([])

    I1 = aperture_photometry(image[0], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=g1)
    I2 = aperture_photometry(image[1], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=g2)

    # Bounded detection confidence: response relative to the strongest peak.
    p_detect = np.clip(resp / (resp.max() + _EPS), 0.0, 1.0) if resp.size else resp
    return _emit(image_id, xs, ys, I1, I2, p_detect=p_detect, flags="baseline,classical_per_channel_aperture")
