"""Method adapters: one uniform interface over every spot-detection method.

The whole benchmark is method-agnostic because every method is wrapped by an
adapter exposing the SAME contract:

    Adapter.predict(eval_set, config) -> canonical-schema DataFrame

The returned DataFrame has exactly :data:`spotpipe.schema.SCHEMA_COLUMNS`; the
harness persists it to a schema file (``predict_to_file`` is the convenience that
does both). Because the schema IS the interface, **adding a new method is exactly
one new adapter** -- nothing in matching, metrics, or the harness changes.

``eval_set`` is an iterable of items each exposing ``.image`` (``[2, H, W]`` raw
counts), ``.image_id`` (str), ``.meta`` (per-image metadata dict), and ``.gt``
(ground-truth schema DataFrame; used only by the oracle baseline). ``config`` is
the parsed ``benchmark:`` config block.

Provided now (runnable): our model, and the two naive baselines. Stubbed for
later (raise ``NotImplementedError`` describing the contract): DECODE, Spotiflow,
and a generic external placeholder -- to be filled in once those tools are
installed, each by implementing ``predict`` to run the tool per channel, divide,
and emit the canonical schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark import baselines
from spotpipe.schema import SCHEMA_COLUMNS, records_to_dataframe, write_spots

__all__ = [
    "Adapter",
    "OurModelAdapter",
    "ClassicalPerChannelApertureAdapter",
    "OracleCenterApertureDivideAdapter",
    "CmeAnalysisPlusApertureAdapter",
    "DecodeAdapter",
    "SpotiflowAdapter",
    "ExternalPlaceholderAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
]


class Adapter:
    """Common base. Subclasses implement :meth:`predict`; the schema is the API."""

    name: str = "adapter"

    def predict(self, eval_set, config: dict) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_to_file(self, eval_set, config: dict, path: str | Path) -> Path:
        """Run :meth:`predict` and persist the canonical schema to ``path``."""
        df = self.predict(eval_set, config)
        return write_spots(df, path)

    @staticmethod
    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return records_to_dataframe([])
        out = pd.concat(frames, ignore_index=True)
        return out[list(SCHEMA_COLUMNS)]


# --------------------------------------------------------------------------- #
# Working adapters                                                             #
# --------------------------------------------------------------------------- #
class OurModelAdapter(Adapter):
    """Wraps :func:`spotpipe.models.spot_model.predict_spots` (our model)."""

    name = "our_model"

    def __init__(self, model, *, device: str = "cpu"):
        self.model = model
        self.device = device

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        from spotpipe.models.spot_model import predict_spots

        mcfg = config.get("our_model", {})
        adc_max = float(config.get("adc_max", 4095.0))
        kwargs = dict(
            peak_threshold=float(mcfg.get("peak_threshold", 0.3)),
            nms_kernel=int(mcfg.get("nms_kernel", 3)),
            max_spots=mcfg.get("max_spots", None),
            logvar_min=float(mcfg.get("logvar_min", -10.0)),
            logvar_max=float(mcfg.get("logvar_max", 6.0)),
        )
        frames = [
            predict_spots(self.model, item.image, image_id=item.image_id,
                          adc_max=adc_max, device=self.device, **kwargs)
            for item in eval_set
        ]
        return self._concat(frames)


class ClassicalPerChannelApertureAdapter(Adapter):
    """Wraps the classical LoG + per-channel APERTURE photometry + divide baseline."""

    name = "classical_per_channel_aperture"

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        bcfg = config.get("baseline", {})
        frames = [
            baselines.classical_per_channel_aperture(item.image, item.meta, image_id=item.image_id, cfg=bcfg)
            for item in eval_set
        ]
        return self._concat(frames)


class OracleCenterApertureDivideAdapter(Adapter):
    """Wraps the oracle-CENTRE APERTURE-divide baseline (reads GT centres only)."""

    name = "oracle_center_aperture_divide"

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        bcfg = config.get("baseline", {})
        frames = [
            baselines.oracle_center_aperture_divide(item.image, item.gt, item.meta, image_id=item.image_id, cfg=bcfg)
            for item in eval_set
        ]
        return self._concat(frames)


class CmeAnalysisPlusApertureAdapter(Adapter):
    """CMEAnalysis (external detector) + in-repo aperture photometry.

    CMEAnalysis is a LOCAL, EXTERNAL MATLAB package -- never vendored here. Its
    detections reach us only as a normalized CSV (``image_id, x, y`` + optional
    columns; see :mod:`spotpipe.benchmark.cmeanalysis`). This adapter reads that
    CSV (``config['cmeanalysis']['detections_csv']``), then extracts canonical
    ``I1`` / ``I2`` by aperture + annulus photometry on the **photon** images at
    the CME centres -- CME's native amplitudes are NOT used as intensities.

    Photometry uses each item's photon image when the loader attached one
    (``item.photon``, e.g. the frozen set's ``images_ch{1,2}_photon`` TIFFs);
    otherwise it derives the photon image from the raw counts and the known
    per-channel offset/gain in ``item.meta`` -- the same linear correction the
    frozen set records (``photon_k = (raw - offset_k) / gain_k``).
    """

    name = "cmeanalysis_plus_aperture"

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        from spotpipe.benchmark import cmeanalysis

        ccfg = config.get("cmeanalysis", {})
        det_path = ccfg.get("detections_csv")
        if not det_path:
            raise ValueError(
                "cmeanalysis_plus_aperture requires "
                "config['cmeanalysis']['detections_csv'] (the normalized CME "
                "detections CSV); see scripts/run_cmeanalysis.py to produce it."
            )
        det = cmeanalysis.load_normalized_detections(det_path)
        by_image = {str(k): v for k, v in det.groupby("image_id")}

        frames = []
        for item in eval_set:
            sub = by_image.get(str(item.image_id))
            if sub is None or len(sub) == 0:
                continue
            photon = self._photon_for(item)
            frames.append(
                cmeanalysis.cme_plus_aperture(photon, sub, image_id=item.image_id, cfg=ccfg)
            )
        return self._concat(frames)

    @staticmethod
    def _photon_for(item) -> np.ndarray:
        """Return the ``[2, H, W]`` photon image for an eval item.

        Prefers an attached ``item.photon`` (real photon TIFFs); falls back to the
        documented linear correction ``(raw - offset_k) / gain_k`` from the raw
        counts + detector meta when none is attached (e.g. in-memory eval sets).
        """
        photon = getattr(item, "photon", None)
        if photon is not None:
            return np.asarray(photon, dtype=float)
        image = np.asarray(item.image, dtype=float)
        det = item.meta.get("detector", {})
        o1 = float(det.get("ch1", {}).get("offset", 0.0))
        g1 = float(det.get("ch1", {}).get("gain", 1.0))
        o2 = float(det.get("ch2", {}).get("offset", 0.0))
        g2 = float(det.get("ch2", {}).get("gain", 1.0))
        out = np.empty_like(image, dtype=float)
        out[0] = (image[0] - o1) / max(g1, 1e-6)
        out[1] = (image[1] - o2) / max(g2, 1e-6)
        return out


# --------------------------------------------------------------------------- #
# Stubbed external tools (filled in once installed)                            #
# --------------------------------------------------------------------------- #
class _ExternalStub(Adapter):
    """Shared stub: documents the contract every external adapter must fulfil."""

    tool = "external tool"

    def predict(self, eval_set, config: dict) -> pd.DataFrame:
        raise NotImplementedError(
            f"The {self.tool} adapter is a stub. To implement it, run {self.tool} on each "
            "two-channel image -- INDEPENDENTLY per channel (channel 1 and channel 2 each "
            "get their own detection + per-spot intensity), pair the per-channel detections, "
            "DIVIDE the two channel intensities to get the ratio, and emit the canonical "
            "spotpipe.schema (one row per spot; fill p_detect/logI1/logI2/x/y; leave "
            "uncertainty1/2 NaN unless the tool provides them). It must return exactly "
            "SCHEMA_COLUMNS so matching/metrics/harness work unchanged. Install the tool, "
            "then implement predict() here -- nothing else in the benchmark changes."
        )


class DecodeAdapter(_ExternalStub):
    """STUB adapter for DECODE (deep-learning emitter detection). Not yet installed."""

    name = "decode"
    tool = "DECODE"


class SpotiflowAdapter(_ExternalStub):
    """STUB adapter for Spotiflow (deep-learning spot detection). Not yet installed."""

    name = "spotiflow"
    tool = "Spotiflow"


class ExternalPlaceholderAdapter(_ExternalStub):
    """STUB generic placeholder for any other external method. Not yet installed."""

    name = "external_placeholder"
    tool = "the external method"


# --------------------------------------------------------------------------- #
# Lazy factories for the Spotiflow + aperture methods                          #
# --------------------------------------------------------------------------- #
# The two honest Spotiflow methods share one adapter
# (:class:`spotpipe.benchmark.spotiflow.SpotiflowPlusApertureAdapter`). It is
# registered through these factory functions, lazily importing the spotiflow
# module so (a) there is no import cycle (the spotiflow module imports `Adapter`
# from here) and (b) `import spotpipe.benchmark.adapters` never pulls in the
# external `spotiflow` package (the in-repo adapter module does not import it
# either; only scripts/run_spotiflow_predict.py does). These are distinct from
# the legacy `spotiflow` stub above, which has the wrong (per-channel-divide)
# contract and is NOT used for the new methods.
def _spotiflow_general_adapter(**kwargs) -> Adapter:
    from spotpipe.benchmark.spotiflow import (
        SPOTIFLOW_METHOD_GENERAL,
        SpotiflowPlusApertureAdapter,
    )

    return SpotiflowPlusApertureAdapter(
        method_name=SPOTIFLOW_METHOD_GENERAL, model_variant="general", **kwargs
    )


def _spotiflow_finetuned_adapter(**kwargs) -> Adapter:
    from spotpipe.benchmark.spotiflow import (
        SPOTIFLOW_METHOD_FINETUNED,
        SpotiflowPlusApertureAdapter,
    )

    return SpotiflowPlusApertureAdapter(
        method_name=SPOTIFLOW_METHOD_FINETUNED,
        model_variant="finetuned_spotpipe_synth",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
# Values are constructors: an Adapter subclass, or a factory returning one (the
# Spotiflow methods). `get_adapter` calls the value with kwargs either way.
ADAPTER_REGISTRY: dict[str, object] = {
    OurModelAdapter.name: OurModelAdapter,
    ClassicalPerChannelApertureAdapter.name: ClassicalPerChannelApertureAdapter,
    OracleCenterApertureDivideAdapter.name: OracleCenterApertureDivideAdapter,
    CmeAnalysisPlusApertureAdapter.name: CmeAnalysisPlusApertureAdapter,
    DecodeAdapter.name: DecodeAdapter,
    SpotiflowAdapter.name: SpotiflowAdapter,
    ExternalPlaceholderAdapter.name: ExternalPlaceholderAdapter,
    "spotiflow_general_plus_aperture": _spotiflow_general_adapter,
    "spotiflow_finetuned_spotpipe_synth_plus_aperture": _spotiflow_finetuned_adapter,
}

# Backward-compatible aliases for the pre-rename method names. Deprecated: they
# resolve to the honestly-named adapters (whose `.name` is the new label, so all
# outputs use the new names). Kept only so older configs/checkpoints still load.
_DEPRECATED_ALIASES: dict[str, str] = {
    "classical_per_channel": "classical_per_channel_aperture",
    "gt_center_divide": "oracle_center_aperture_divide",
}


def get_adapter(name: str, **kwargs) -> Adapter:
    """Construct an adapter by canonical name.

    ``our_model`` requires a trained ``model=...`` keyword; the baselines and the
    external stubs take no construction arguments. Deprecated pre-rename names are
    accepted as aliases (they construct the honestly-named adapter).
    """
    canonical = _DEPRECATED_ALIASES.get(name, name)
    if canonical not in ADAPTER_REGISTRY:
        raise KeyError(
            f"unknown method {name!r}; known methods: {sorted(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[canonical](**kwargs)
