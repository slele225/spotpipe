"""Legacy ``v0_hrnet`` adapter helpers (an OLD, externally-trained detector).

``v0_hrnet`` wraps a **legacy** two-channel HRNet detector archived in a SEPARATE
repository (``liposome-detect/models/hrnet_v1``). It is labelled ``v0`` honestly:
it is an old model from a long time ago, trained on a DIFFERENT forward model with
DIFFERENT channel semantics (lipid / protein) and DIFFERENT normalization, NOT on
spotpipe's simulator. We benchmark it as-is, cross-domain, to have an honest
historical baseline -- not because its absolute intensities are calibrated to
spotpipe's photon-proportional units (they are not; see the units caveat below).

Unlike the ``*_plus_aperture`` methods (CMEAnalysis / Spotiflow / SpotMAX, which
are detector-only and get their canonical ``I1`` / ``I2`` from in-repo aperture
photometry on the PHOTON images), the legacy HRNet **predicts intensities and
per-spot uncertainties directly** -- its decode head emits, per detection::

    x, y, detection_score,
    lipid_intensity, lipid_intensity_logvar,
    protein_intensity, protein_intensity_logvar

so this is ``v0_hrnet`` (model-native intensities), NOT ``v0_hrnet_plus_aperture``.
There is no aperture step and no photon image is read here.

Two-stage contract (mirrors the other external adapters)
--------------------------------------------------------
1. ``scripts/run_v0_hrnet_predict.py`` is the ONE place that imports torch / timm
   and the legacy repo's code. It runs in a SEPARATE environment, runs the legacy
   model over the frozen set, and writes a **canonical-schema predictions CSV**
   (exactly :data:`spotpipe.schema.SCHEMA_COLUMNS`) via the shared converter
   :func:`detections_to_canonical`. **This module never imports torch / timm or
   the legacy repo** -- importing ``spotpipe`` must not require any of them.
2. :class:`V0HrnetAdapter` (the harness side) just LOADS that canonical CSV,
   validates it, and subsets it to the eval set's image ids. It fabricates
   nothing: an image with no predicted rows contributes no rows.

Units caveat (carried in every row's ``flags`` as ``units=legacy_flux``)
------------------------------------------------------------------------
The legacy model's ``I1`` / ``I2`` are in the OLD simulator's flux units, not
spotpipe photon-proportional units, so absolute-intensity bias is expected to be
large / meaningless. Detection metrics and the log-ratio SLOPE (beta) remain
meaningful: a constant per-channel scale is an additive constant in log space, so
it shifts the ratio intercept, not the slope.

Channel mapping
---------------
spotpipe's channels are generic (ch1 -> ``logI1``, ch2 -> ``logI2``); the legacy
model's are lipid / protein. The mapping is explicit and recorded in ``flags``
(``ch1=<lipid|protein>;ch2=<...>``). The default is ``ch1=lipid, ch2=protein``
(so ``log_ratio = logI2 - logI1`` corresponds to the legacy log(protein/lipid),
whose slope is the legacy ``alpha / 2``). The producing script
(:mod:`scripts.run_v0_hrnet_predict`) chooses it via ``--ch1-channel`` and the
converter below encodes the per-channel uncertainty transform once, unit-tested.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.schema import SCHEMA_COLUMNS, SpotRecord, records_to_dataframe

__all__ = [
    "V0_HRNET_METHOD",
    "LEGACY_DETECTION_KEYS",
    "CHANNEL_CHOICES",
    "detections_to_canonical",
    "load_canonical_predictions",
    "V0HrnetAdapter",
]

V0_HRNET_METHOD = "v0_hrnet"

# Per-detection keys the legacy decode emits (see liposome-detect
# src/models/decode.py SCHEMA_KEYS); the converter reads exactly these.
LEGACY_DETECTION_KEYS: tuple[str, ...] = (
    "x", "y", "detection_score",
    "lipid_intensity", "lipid_intensity_logvar",
    "protein_intensity", "protein_intensity_logvar",
)

CHANNEL_CHOICES: tuple[str, ...] = ("lipid", "protein")

# Floor before log: the legacy mean-flux head exp's its output so it is strictly
# positive, but clamp defensively so a hand-made / corrupt row can never log(<=0).
_FLUX_FLOOR = 1e-6


def _logvar_to_uncertainty(logvar: float) -> float:
    """Legacy log-flux-residual log-variance -> spotpipe ``uncertainty`` (std of logI).

    The legacy intensity NLL models ``sigma2 = exp(logvar)`` as the variance of the
    log-flux residual ``log(pred)-log(true)`` (see liposome-detect train/losses.py),
    so the predicted standard deviation in log-intensity -- exactly spotpipe's
    ``uncertainty`` semantics -- is ``exp(0.5 * logvar)``. Non-finite -> NaN.
    """
    if logvar is None or not math.isfinite(float(logvar)):
        return math.nan
    return float(math.exp(0.5 * float(logvar)))


def detections_to_canonical(
    image_id: str,
    detections,
    *,
    ch1_channel: str = "lipid",
    extra_flags: str = "",
) -> pd.DataFrame:
    """Legacy per-image detections -> canonical 16-column schema DataFrame.

    ``detections`` is an iterable of dicts with :data:`LEGACY_DETECTION_KEYS`
    (the legacy decode output). ``ch1_channel`` selects which legacy channel maps
    to spotpipe ch1 (``logI1``); the other maps to ch2. ``logI_k`` is the log of
    the legacy mean flux (floored at :data:`_FLUX_FLOOR`); ``uncertainty_k`` is
    the per-channel log-variance turned into a log-intensity std via
    :func:`_logvar_to_uncertainty`. ``sigma1_hat`` / ``sigma2_hat`` are NaN (the
    legacy model has no PSF-width head). ``I1`` / ``I2`` / ``log_ratio`` / ``ratio``
    are derived so the redundant columns stay consistent.

    The channel mapping + the honest legacy/units provenance are recorded in every
    row's ``flags`` so the choice is never ambiguous downstream.
    """
    ch1_channel = str(ch1_channel).lower()
    if ch1_channel not in CHANNEL_CHOICES:
        raise ValueError(
            f"ch1_channel must be one of {CHANNEL_CHOICES}; got {ch1_channel!r}"
        )
    ch2_channel = "protein" if ch1_channel == "lipid" else "lipid"

    flags = (
        f"{V0_HRNET_METHOD};legacy;intensity=model_native;units=legacy_flux;"
        f"ch1={ch1_channel};ch2={ch2_channel}"
    )
    if extra_flags:
        flags = f"{flags};{extra_flags}"

    records: list[SpotRecord] = []
    for det in detections:
        i1 = float(det[f"{ch1_channel}_intensity"])
        i2 = float(det[f"{ch2_channel}_intensity"])
        records.append(
            SpotRecord.from_logs(
                image_id=str(image_id),
                spot_id=len(records),
                x=float(det["x"]),
                y=float(det["y"]),
                p_detect=float(det["detection_score"]),
                logI1=math.log(max(i1, _FLUX_FLOOR)),
                logI2=math.log(max(i2, _FLUX_FLOOR)),
                sigma1_hat=math.nan,
                sigma2_hat=math.nan,
                uncertainty1=_logvar_to_uncertainty(det[f"{ch1_channel}_intensity_logvar"]),
                uncertainty2=_logvar_to_uncertainty(det[f"{ch2_channel}_intensity_logvar"]),
                flags=flags,
            )
        )
    return records_to_dataframe(records)


def load_canonical_predictions(path: str | Path) -> pd.DataFrame:
    """Load + validate a canonical ``v0_hrnet`` predictions CSV.

    The producing script writes exactly :data:`SCHEMA_COLUMNS`; this asserts that
    contract loudly (``ValueError`` listing any missing column) so a malformed CSV
    is surfaced rather than silently mis-benchmarked. ``image_id`` -> str for
    type-stable subsetting; columns are returned in canonical order.
    """
    df = pd.read_csv(path)
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"v0_hrnet predictions CSV {str(path)!r} is missing canonical columns "
            f"{missing}; expected exactly {list(SCHEMA_COLUMNS)}. Produce it with "
            "scripts/run_v0_hrnet_predict.py."
        )
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    return df[list(SCHEMA_COLUMNS)]


class V0HrnetAdapter:
    """Legacy HRNet model-native predictions -> canonical schema (no aperture step).

    Construct via the registry: ``get_adapter('v0_hrnet')``.

    ``predict`` reads the canonical predictions CSV at
    ``config['v0_hrnet']['predictions_csv']`` (produced by
    ``scripts/run_v0_hrnet_predict.py`` in a SEPARATE torch/timm env), then for
    each eval item returns that image's predicted rows. It fabricates nothing: an
    image absent from the CSV contributes zero rows. The legacy model's intensities
    and uncertainties are used directly (model-native); no photon image is read and
    no aperture photometry is performed -- that is what distinguishes ``v0_hrnet``
    from the ``*_plus_aperture`` detector-only methods.
    """

    name = V0_HRNET_METHOD

    def __init__(self, *, log_fn=print, **kwargs):
        self._log = log_fn

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        # Imported lazily to avoid an import cycle (adapters <-> v0_hrnet).
        from spotpipe.benchmark.adapters import Adapter

        vcfg = dict(config.get("v0_hrnet", {}))
        pred_path = vcfg.get("predictions_csv")
        if not pred_path:
            raise ValueError(
                f"{self.name} requires config['v0_hrnet']['predictions_csv'] (the "
                "canonical predictions CSV). Produce it by running the legacy model "
                "in a SEPARATE torch/timm env: scripts/run_v0_hrnet_predict.py "
                "--frozen-dir <dir> --legacy-repo <liposome-detect> "
                "--legacy-config <hrnet_v1.yaml> --checkpoint <best.pt> --out <csv>."
            )
        pred_path = Path(pred_path)
        if not pred_path.exists():
            raise FileNotFoundError(
                f"{self.name}: predictions CSV {str(pred_path)!r} not found. Run the "
                "legacy model first (separate torch/timm env): "
                "scripts/run_v0_hrnet_predict.py --frozen-dir <dir> "
                "--legacy-repo <liposome-detect> --legacy-config <hrnet_v1.yaml> "
                f"--checkpoint <best.pt> --out {str(pred_path)!r}."
            )

        pred = load_canonical_predictions(pred_path)
        by_image = {str(k): v for k, v in pred.groupby("image_id")}

        n_images = 0
        frames = []
        for item in eval_set:
            sub = by_image.get(str(item.image_id))
            if sub is None or len(sub) == 0:
                continue  # never fabricate spots for an image with no predictions
            n_images += 1
            frames.append(sub)

        self._log(
            f"[v0_hrnet] loaded {len(pred)} legacy predictions; "
            f"{n_images} eval image(s) had detections"
        )
        if not frames:
            return records_to_dataframe([])
        return Adapter._concat([f[list(SCHEMA_COLUMNS)] for f in frames])
