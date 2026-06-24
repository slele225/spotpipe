"""Spotiflow + aperture adapter helpers (external-detector method).

This is the in-repo side of the two ``spotiflow_*_plus_aperture`` methods. It
mirrors :mod:`spotpipe.benchmark.cmeanalysis` exactly: an EXTERNAL detector
(Spotiflow) localizes spots and reaches us ONLY as a normalized detections CSV;
everything here consumes that CSV and nothing else. **This module never imports
the external ``spotiflow`` package** -- importing ``spotpipe`` must not require
Spotiflow installed (the lazy external dependency lives only in
``scripts/run_spotiflow_predict.py``).

Role split (the fair version, identical in spirit to ``cmeanalysis_plus_aperture``):

* Spotiflow is the **detector / localizer ONLY**. It runs on the RAW detector
  image (e.g. the pixelwise ``raw_max`` of the two channels) and reports sub-pixel
  spot centres. See ``scripts/run_spotiflow_predict.py``.
* Spotiflow's native heatmap probability is carried only as ``p_detect`` (detection
  confidence) -- it is **never** used as an intensity.
* Canonical ``I1`` / ``I2`` are extracted HERE, by the same **aperture + annulus
  background** estimator the in-repo aperture baseline uses
  (:func:`spotpipe.benchmark.baselines.aperture_photometry`), read from the
  **photon-proportional** images. Raw counts are never divided; the simulator's
  true-background files (the non-fair oracle background) are never read.

Two methods share this one adapter, differing only by ``model_variant`` (which
detector produced the detections):

* ``spotiflow_general_plus_aperture``                  -- Spotiflow pretrained "general".
* ``spotiflow_finetuned_spotpipe_synth_plus_aperture`` -- Spotiflow fine-tuned on
  spotpipe synthetic training data (NOT the frozen test / fixed-eval sets).

Normalized detections CSV contract (the primary interface)
----------------------------------------------------------
Required columns::

    image_id, x, y

    * ``image_id`` matches the eval-set image id (the frozen-set ``<id>``).
    * ``x`` = sub-pixel column, ``y`` = sub-pixel row, in spotpipe's 0-indexed
      pixel convention. ``scripts/run_spotiflow_predict.py`` is responsible for
      converting Spotiflow's ``(y, x)`` point order to spotpipe ``(x, y)``.

Optional columns (provenance only; never used as canonical intensity)::

    p_detect      -- Spotiflow heatmap probability / confidence in [0, 1]
    source        -- "spotiflow"
    model_variant -- "general" | "finetuned_spotpipe_synth"
    detect_image  -- "raw_max" | "master_ch1" | ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark import baselines
from spotpipe.benchmark.adapters import Adapter, CmeAnalysisPlusApertureAdapter
from spotpipe.schema import records_to_dataframe

__all__ = [
    "SPOTIFLOW_METHOD_GENERAL",
    "SPOTIFLOW_METHOD_FINETUNED",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "load_normalized_spotiflow_detections",
    "spots_yx_to_xy",
    "SpotiflowPlusApertureAdapter",
]

SPOTIFLOW_METHOD_GENERAL = "spotiflow_general_plus_aperture"
SPOTIFLOW_METHOD_FINETUNED = "spotiflow_finetuned_spotpipe_synth_plus_aperture"

REQUIRED_COLUMNS: tuple[str, ...] = ("image_id", "x", "y")
OPTIONAL_COLUMNS: tuple[str, ...] = ("p_detect", "source", "model_variant", "detect_image")

_EPS = 1e-6


def spots_yx_to_xy(spots) -> tuple[np.ndarray, np.ndarray]:
    """Convert Spotiflow's ``(row, col) = (y, x)`` points to spotpipe ``(x, y)``.

    Spotiflow's ``model.predict`` returns an ``(n_spots, 2)`` array of point
    coordinates in **array / image order** -- axis 0 (rows = ``y``) then axis 1
    (columns = ``x``). spotpipe's canonical convention is ``x = column``,
    ``y = row`` (origin top-left). This single helper is the one place that
    encodes that swap, so the assumption is unit-tested rather than buried in the
    predict script. Returns ``(xs, ys)`` float arrays.
    """
    arr = np.asarray(spots, dtype=float).reshape(-1, 2)
    ys = arr[:, 0]
    xs = arr[:, 1]
    return xs, ys


def load_normalized_spotiflow_detections(path: str | Path) -> pd.DataFrame:
    """Load + validate a normalized Spotiflow detections CSV.

    Raises ``ValueError`` if any required column is missing. ``image_id`` is
    coerced to ``str`` and ``x`` / ``y`` to ``float`` so grouping / photometry are
    type-stable; ``p_detect`` is coerced to float when present (non-numeric -> NaN);
    other optional columns pass through untouched.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"normalized Spotiflow detections CSV {str(path)!r} is missing required "
            f"columns {missing}; required = {list(REQUIRED_COLUMNS)}. Produce it with "
            "scripts/run_spotiflow_predict.py."
        )
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    if "p_detect" in df.columns:
        df["p_detect"] = pd.to_numeric(df["p_detect"], errors="coerce").astype(float)
    return df


def _p_detect_array(det_df: pd.DataFrame, n: int) -> np.ndarray:
    """Bounded per-detection ``p_detect`` in ``[0, 1]``; missing/NaN -> 1.0.

    Spotiflow's confidence is carried through as detection probability only (never
    as intensity). When the producer omitted it (or wrote NaN), fall back to a
    constant ``1.0`` so matching/metrics still have a usable confidence.
    """
    if "p_detect" not in det_df.columns:
        return np.ones(n, dtype=float)
    p = det_df["p_detect"].to_numpy(dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    return np.clip(p, 0.0, 1.0)


class SpotiflowPlusApertureAdapter(Adapter):
    """Spotiflow centres + aperture photometry on the PHOTON images -> canonical schema.

    Construct via the registry factories (so ``method_name`` / ``model_variant`` are
    bound): ``get_adapter('spotiflow_general_plus_aperture')`` or
    ``get_adapter('spotiflow_finetuned_spotpipe_synth_plus_aperture')``.

    ``predict`` reads the normalized detections CSV at
    ``config['spotiflow'][<model_variant>]['detections_csv']`` (or the top-level
    ``config['spotiflow']['detections_csv']`` fallback), then for each eval item
    extracts ``I1`` / ``I2`` by aperture + annulus photometry at the Spotiflow
    centres on the photon image (``item.photon`` when attached -- the frozen set's
    ``images_ch{1,2}_photon`` TIFFs -- else derived from raw counts + detector meta
    via the shared CMEAnalysis photon helper). ``gain=1.0`` because the photon
    image is already gain-corrected. ``sigma*_hat`` / ``uncertainty*`` are left NaN.
    """

    def __init__(self, *, method_name: str, model_variant: str, **kwargs):
        # Instance-level name so the harness/registry report the honest method name.
        self.name = method_name
        self.method_name = method_name
        self.model_variant = model_variant

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        scfg = dict(config.get("spotiflow", {}))
        vcfg = dict(scfg.get(self.model_variant, {}))
        det_path = vcfg.get("detections_csv") or scfg.get("detections_csv")
        if not det_path:
            raise ValueError(
                f"{self.method_name} requires "
                f"config['spotiflow']['{self.model_variant}']['detections_csv'] (the "
                "normalized Spotiflow detections CSV). Produce it by running Spotiflow in "
                "its own env: scripts/run_spotiflow_predict.py "
                f"--frozen-dir <dir> --model-variant {self.model_variant} --out <csv>."
            )
        det_path = Path(det_path)
        if not det_path.exists():
            raise FileNotFoundError(
                f"{self.method_name}: detections CSV {str(det_path)!r} not found. Run the "
                "detector first (in the .venvs/spotiflow env): "
                "scripts/run_spotiflow_predict.py "
                f"--frozen-dir <dir> --model-variant {self.model_variant} "
                f"--out {str(det_path)!r}."
            )

        det = load_normalized_spotiflow_detections(det_path)
        by_image = {str(k): v for k, v in det.groupby("image_id")}

        detect_image = str(scfg.get("detect_image", vcfg.get("detect_image", "raw_max")))
        flags = (
            f"source=spotiflow;model_variant={self.model_variant};"
            f"detect_image={detect_image};intensity=aperture_photon"
        )
        r_ap, r_in, r_out = baselines._aperture_radii(scfg)

        frames = []
        for item in eval_set:
            sub = by_image.get(str(item.image_id))
            if sub is None or len(sub) == 0:
                # No detections for this image -> no rows (never fabricate spots).
                continue
            photon = CmeAnalysisPlusApertureAdapter._photon_for(item)
            xs = sub["x"].to_numpy(dtype=float)
            ys = sub["y"].to_numpy(dtype=float)
            I1 = baselines.aperture_photometry(photon[0], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=1.0)
            I2 = baselines.aperture_photometry(photon[1], xs, ys, r_ap=r_ap, r_in=r_in, r_out=r_out, gain=1.0)
            p_detect = _p_detect_array(sub, xs.size)
            frames.append(
                baselines._emit(item.image_id, xs, ys, I1, I2, p_detect=p_detect, flags=flags)
            )
        if not frames:
            return records_to_dataframe([])
        return self._concat(frames)
