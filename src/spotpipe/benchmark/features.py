"""Per-spot difficulty axes for binning: SNR and local density (build stage 4).

The benchmark bins every metric by two axes (see CLAUDE.md): **SNR** and **spot
density**. Both are defined here, from simulator metadata only, so the
definitions live in exactly one documented place and apply identically to ground
truth and to any method's predictions.

SNR -- per-spot, photon-domain *peak* signal-to-noise
------------------------------------------------------
Computed per channel ``k`` from the spot's total integrated intensity
``A_k = exp(logI_k)`` (photon-proportional units), the image's flat background
level ``B_k`` (photons), and the detector constants, accounting for the
``n_frames`` frame averaging (which divides added-noise variance by ``n_frames``;
CLAUDE.md):

    peak_k  = A_k / (2*pi*sigma_k^2)              # own-contribution peak (photons)
    read_k  = noise_floor_sigma_k / gain_k        # read noise in photon-equiv units
    noise_k = sqrt( ((peak_k + B_k) + read_k^2) / n_frames )
    snr_k   = peak_k / noise_k

The per-spot scalar is ``snr = min(snr_1, snr_2)`` -- the **limiting channel**.
The ratio ``I2/I1`` (the headline quantity) can be no better measured than its
worse-measured channel, so the lower-SNR channel is what makes a spot hard; we
bin by it. (``snr_1`` / ``snr_2`` are kept too, for inspection.) Background here
is the flat ``level`` only; gradient/structure add a little more and are ignored
for this scalar -- it is a binning axis, not a calibration.

Density -- per-spot local crowding
-----------------------------------
``n_neighbors`` = the number of OTHER spots whose centre lies within
``density_radius_px`` of this spot. This is the local-overlap measure the
project's "high-overlap corner" is about: overlapping PSFs are what a
window-readout baseline cannot deblend. The image-level spots-per-pixel density
is also available (``image_density``) for image-level summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "DetectorAxisParams",
    "axis_params_from_meta",
    "peak_snr",
    "local_neighbor_count",
    "attach_features",
]


class DetectorAxisParams:
    """The handful of per-image scalars the SNR axis needs, pulled from meta."""

    __slots__ = ("sigma1", "sigma2", "bg1", "bg2", "gain1", "gain2", "floor1", "floor2", "n_frames")

    def __init__(self, *, sigma1, sigma2, bg1, bg2, gain1, gain2, floor1, floor2, n_frames):
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.bg1 = float(bg1)
        self.bg2 = float(bg2)
        self.gain1 = float(gain1)
        self.gain2 = float(gain2)
        self.floor1 = float(floor1)
        self.floor2 = float(floor2)
        self.n_frames = max(int(n_frames), 1)


def axis_params_from_meta(meta: dict) -> DetectorAxisParams:
    """Extract the SNR-axis scalars from a per-image ``meta`` dict.

    Reads the true PSF widths (``meta['ground_truth_sigma']``), the flat
    background level per channel (``meta['scene']['background{1,2}']['level']``),
    and the per-channel detector gain / read-noise floor plus ``n_frames``
    (``meta['detector']``). Missing fields fall back to sensible defaults so a
    sparser meta still yields a usable (if rougher) axis.
    """
    gts = meta.get("ground_truth_sigma", {})
    scene = meta.get("scene", {})
    det = meta.get("detector", {})
    ch1 = det.get("ch1", {})
    ch2 = det.get("ch2", {})
    return DetectorAxisParams(
        sigma1=gts.get("sigma1", 1.3),
        sigma2=gts.get("sigma2", 1.5),
        bg1=scene.get("background1", {}).get("level", 5.0),
        bg2=scene.get("background2", {}).get("level", 5.0),
        gain1=ch1.get("gain", 1.0),
        gain2=ch2.get("gain", 1.0),
        floor1=ch1.get("noise_floor_sigma", 0.0),
        floor2=ch2.get("noise_floor_sigma", 0.0),
        n_frames=det.get("n_frames", 3),
    )


def _channel_snr(A, sigma, bg, gain, floor, n_frames):
    A = np.asarray(A, dtype=float)
    peak = A / (2.0 * np.pi * sigma * sigma)
    read = (floor / gain) if gain > 0 else 0.0
    noise = np.sqrt(np.clip((peak + bg) + read * read, 1e-12, None) / n_frames)
    return peak / noise


def peak_snr(logI1, logI2, params: DetectorAxisParams) -> dict[str, np.ndarray]:
    """Per-spot peak SNR per channel and combined (``min``), from log-intensities.

    ``logI1`` / ``logI2`` are natural-log total integrated intensities (photon
    units), as stored in the schema. Returns ``{'snr1', 'snr2', 'snr'}`` arrays.
    """
    A1 = np.exp(np.asarray(logI1, dtype=float))
    A2 = np.exp(np.asarray(logI2, dtype=float))
    snr1 = _channel_snr(A1, params.sigma1, params.bg1, params.gain1, params.floor1, params.n_frames)
    snr2 = _channel_snr(A2, params.sigma2, params.bg2, params.gain2, params.floor2, params.n_frames)
    return {"snr1": snr1, "snr2": snr2, "snr": np.minimum(snr1, snr2)}


def local_neighbor_count(x, y, radius: float) -> np.ndarray:
    """Number of OTHER points within ``radius`` of each point (per image).

    A brute-force ``O(n^2)`` count -- eval images hold at most a few hundred
    spots, so this is cheap and dependency-free.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n <= 1:
        return np.zeros(n, dtype=float)
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    within = d2 <= (radius * radius)
    np.fill_diagonal(within, False)  # exclude self
    return within.sum(axis=1).astype(float)


def attach_features(
    df: pd.DataFrame,
    meta_by_image: dict[str, dict],
    *,
    density_radius_px: float,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``snr1``/``snr2``/``snr``/``n_neighbors`` added.

    SNR is computed from each row's ``logI1``/``logI2`` and the row's image meta;
    ``n_neighbors`` is the local crowding of the *same* table (so GT rows get true
    crowding and prediction rows get predicted crowding). Rows whose ``image_id``
    is absent from ``meta_by_image`` get NaN SNR (no axis info) but a valid
    neighbour count.
    """
    df = df.copy().reset_index(drop=True)
    for col in ("snr1", "snr2", "snr", "n_neighbors"):
        df[col] = np.nan
    if len(df) == 0:
        return df

    for image_id, idx in df.groupby("image_id").groups.items():
        idx = np.asarray(idx)
        sub = df.loc[idx]
        df.loc[idx, "n_neighbors"] = local_neighbor_count(
            sub["x"].to_numpy(), sub["y"].to_numpy(), density_radius_px
        )
        meta = meta_by_image.get(str(image_id))
        if meta is None:
            continue
        snr = peak_snr(sub["logI1"].to_numpy(), sub["logI2"].to_numpy(), axis_params_from_meta(meta))
        df.loc[idx, "snr1"] = snr["snr1"]
        df.loc[idx, "snr2"] = snr["snr2"]
        df.loc[idx, "snr"] = snr["snr"]
    return df
