"""CMEAnalysis + aperture adapter helpers (external-detector method).

This is the in-repo side of the ``cmeanalysis_plus_aperture`` method. CMEAnalysis
itself is an EXTERNAL, local-only MATLAB package -- it is never vendored into this
repo and its source is never modified. The only contract between CMEAnalysis and
spotpipe is a **normalized detections CSV** (see below); everything here consumes
that CSV and nothing else.

Role split (the fair first version):

* CMEAnalysis is the **detector / localizer ONLY**. It runs on the RAW per-channel
  images and reports sub-pixel spot centres on its master channel.
* The CME native Gaussian amplitudes ``A`` / ``slave_A`` are **NOT** used as the
  canonical ``I1`` / ``I2`` (they are peak amplitudes, not integrated intensities,
  and using them would be a different method that must be renamed + documented).
* Canonical ``I1`` / ``I2`` are extracted HERE, by the same **aperture + annulus
  background** estimator the in-repo aperture baseline uses (so the only
  difference between ``cmeanalysis_plus_aperture`` and
  ``classical_per_channel_aperture`` is the *detection source*), read from the
  **photon-proportional** images. Raw counts are never divided; the ``audit/``
  true background is never read.

Normalized detections CSV contract (the primary interface)
----------------------------------------------------------
Required columns:

    image_id, x, y

    * ``image_id`` matches the eval-set image id (the frozen-set ``<id>``).
    * ``x`` = sub-pixel column, ``y`` = sub-pixel row, in spotpipe's **0-indexed**
      pixel convention. The external MATLAB wrapper is responsible for converting
      MATLAB's 1-indexed coordinates (``x - 1``, ``y - 1``).

Optional columns (carried for provenance / scoring; never used as canonical I):

    score        -- a CME confidence / p-value-like quantity (see p_detect modes)
    A            -- CME master-channel fitted amplitude (peak, not integrated)
    slave_A      -- CME slave-channel fitted amplitude
    channel      -- detect (master) channel index, if the producer records it
    native_I1    -- optional native CME intensity, channel 1 (provenance only)
    native_I2    -- optional native CME intensity, channel 2 (provenance only)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark import baselines
from spotpipe.schema import records_to_dataframe

__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "load_normalized_detections",
    "compute_p_detect",
    "cme_plus_aperture",
]

REQUIRED_COLUMNS: tuple[str, ...] = ("image_id", "x", "y")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "score", "A", "slave_A", "channel", "native_I1", "native_I2",
)

_EPS = 1e-6


def load_normalized_detections(path: str | Path) -> pd.DataFrame:
    """Load + validate a normalized CME detections CSV.

    Raises ``ValueError`` if any required column is missing. ``image_id`` is
    coerced to ``str`` and ``x`` / ``y`` to ``float`` so grouping / photometry are
    type-stable; optional columns are passed through untouched if present.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"normalized CME detections CSV {str(path)!r} is missing required "
            f"columns {missing}; required = {list(REQUIRED_COLUMNS)}"
        )
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def compute_p_detect(det_df: pd.DataFrame, source: str = "constant"):
    """Map detection columns to a bounded ``p_detect`` in ``[0, 1]``.

    The safe default is ``constant`` -> ``1.0`` (CME does not natively provide a
    higher-is-more-confident probability; ``pval``-like fields are smaller-is-more-
    confident, so they must be transformed, not used directly). Supported modes:

    * ``constant``        -- every detection gets ``1.0`` (default).
    * ``A``               -- master amplitude, normalised by its per-call max.
    * ``score``           -- the ``score`` column used directly, clipped to [0, 1].
    * ``one_minus_pval``  -- ``1 - score`` (treats ``score`` as a p-value).
    * ``neg_log10_pval``  -- ``-log10(score)``, normalised by its per-call max.

    Returns a scalar ``1.0`` for ``constant`` (``baselines._emit`` broadcasts it),
    or a per-detection array otherwise. Missing source columns fall back to ``1.0``
    with no error so a minimal CSV (``image_id,x,y`` only) still works.
    """
    n = len(det_df)
    src = (source or "constant").lower()
    if src == "constant":
        return 1.0
    if src == "a":
        if "A" not in det_df.columns:
            return 1.0
        a = det_df["A"].to_numpy(dtype=float)
        m = float(np.nanmax(a)) if a.size else 0.0
        return np.clip(a / (m + _EPS), 0.0, 1.0) if m > 0 else np.ones(n)
    if src == "score":
        if "score" not in det_df.columns:
            return 1.0
        return np.clip(det_df["score"].to_numpy(dtype=float), 0.0, 1.0)
    if src == "one_minus_pval":
        if "score" not in det_df.columns:
            return 1.0
        return np.clip(1.0 - det_df["score"].to_numpy(dtype=float), 0.0, 1.0)
    if src == "neg_log10_pval":
        if "score" not in det_df.columns:
            return 1.0
        s = np.clip(det_df["score"].to_numpy(dtype=float), 1e-300, 1.0)
        v = -np.log10(s)
        m = float(np.nanmax(v)) if v.size else 0.0
        return np.clip(v / (m + _EPS), 0.0, 1.0) if m > 0 else np.ones(n)
    raise ValueError(
        f"unknown p_detect_source {source!r}; use one of "
        "{'constant','A','score','one_minus_pval','neg_log10_pval'}"
    )


def cme_plus_aperture(
    photon: np.ndarray,
    det_df: pd.DataFrame,
    *,
    image_id: str,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """CME centres + aperture photometry on the photon images -> canonical schema.

    ``photon`` is the two-channel ``[2, H, W]`` photon-proportional image (already
    offset-subtracted + gain-corrected). ``det_df`` are the normalized detection
    rows for THIS image. Intensities are aperture + annulus reads at the CME
    centres in each channel (``gain=1.0`` because the photon image is already
    gain-corrected), reusing :func:`spotpipe.benchmark.baselines.aperture_photometry`
    so the estimator is identical to the aperture baseline. ``sigma*_hat`` and
    ``uncertainty*`` are left NaN (this method emits no PSF width or uncertainty).
    """
    cfg = cfg or {}
    photon = np.asarray(photon, dtype=float)
    if photon.ndim != 3 or photon.shape[0] < 2:
        raise ValueError(f"photon image must be [2, H, W]; got shape {photon.shape}")

    xs = det_df["x"].to_numpy(dtype=float)
    ys = det_df["y"].to_numpy(dtype=float)
    if xs.size == 0:
        return records_to_dataframe([])

    r_ap, r_in, r_out = baselines._aperture_radii(cfg)
    I1 = baselines.aperture_photometry(photon[0], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=1.0)
    I2 = baselines.aperture_photometry(photon[1], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=1.0)

    p_detect = compute_p_detect(det_df, cfg.get("p_detect_source", "constant"))
    return baselines._emit(
        image_id, xs, ys, I1, I2, p_detect=p_detect, flags="cmeanalysis_plus_aperture",
    )
