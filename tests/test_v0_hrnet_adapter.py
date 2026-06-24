"""Tests for the legacy ``v0_hrnet`` adapter (an OLD, externally-trained detector).

No torch / timm / legacy-repo install needed: the adapter's only contract is the
canonical predictions CSV the (separate-env) script writes, so we fabricate legacy
decode dicts and a tiny eval set and check the in-repo side end-to-end. The whole
point of ``v0_hrnet`` is that importing ``spotpipe`` requires NONE of the heavy
legacy dependencies -- so that is tested first.

Runnable two ways (like the other stage tests)::

    uv run python tests/test_v0_hrnet_adapter.py     (standalone)
    uv run pytest tests/test_v0_hrnet_adapter.py

Covered:
  1. Importing spotpipe / the registry does NOT require torch / timm / the legacy repo.
  2. The in-repo module never imports torch / timm at source.
  3. ``detections_to_canonical`` emits EXACTLY the canonical 16 columns, uses the
     legacy model-native intensities directly, maps log-variance -> uncertainty
     (std of logI = exp(0.5*logvar)), leaves sigma*_hat NaN, and keeps I/ratio
     derived-consistent.
  4. The channel mapping (ch1=lipid default vs ch1=protein) is honoured and recorded
     in flags (incl. the honest legacy/units provenance).
  5. ``load_canonical_predictions`` validates the full canonical schema.
  6. The adapter loads the CSV, subsets to the eval images, FABRICATES NOTHING for
     an image with no predictions, and returns exactly the schema columns.
  7. The adapter raises useful errors when the predictions CSV is unset / missing.
"""

from __future__ import annotations

import builtins
import importlib
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _fake_legacy_dets():
    """Two legacy decode dicts (the keys liposome-detect src/models/decode.py emits)."""
    return [
        {
            "x": 12.5, "y": 7.0, "detection_score": 0.9,
            "lipid_intensity": 5000.0, "lipid_intensity_logvar": 0.0,
            "protein_intensity": 3000.0, "protein_intensity_logvar": math.log(4.0),
        },
        {
            "x": 40.0, "y": 33.2, "detection_score": 0.4,
            "lipid_intensity": 800.0, "lipid_intensity_logvar": math.log(9.0),
            "protein_intensity": 1600.0, "protein_intensity_logvar": 0.0,
        },
    ]


def _eval_item(image_id: str):
    """Minimal eval item: the adapter only reads ``.image_id``."""
    return SimpleNamespace(image_id=image_id)


# --------------------------------------------------------------------------- #
# 1-2. Importing must not require torch / timm / the legacy repo               #
# --------------------------------------------------------------------------- #
def test_import_does_not_require_torch_or_timm(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in {"torch", "timm"}:
            raise ImportError(f"{top} blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("spotpipe.benchmark.v0_hrnet", None)
    mod = importlib.import_module("spotpipe.benchmark.v0_hrnet")
    assert mod.V0_HRNET_METHOD == "v0_hrnet"

    from spotpipe.benchmark.adapters import ADAPTER_REGISTRY, get_adapter

    assert "v0_hrnet" in ADAPTER_REGISTRY
    adapter = get_adapter("v0_hrnet")
    assert adapter.name == "v0_hrnet"


def test_module_does_not_import_torch_at_source() -> None:
    src = (REPO_ROOT / "src" / "spotpipe" / "benchmark" / "v0_hrnet.py").read_text(encoding="utf-8")
    assert "import torch" not in src, "in-repo module must never import torch"
    assert "import timm" not in src, "in-repo module must never import timm"


# --------------------------------------------------------------------------- #
# 3-4. detections_to_canonical: schema, model-native intensities, mapping      #
# --------------------------------------------------------------------------- #
def test_detections_to_canonical_default_mapping() -> None:
    from spotpipe.benchmark.v0_hrnet import detections_to_canonical
    from spotpipe.schema import SCHEMA_COLUMNS

    df = detections_to_canonical("img_00000", _fake_legacy_dets(), ch1_channel="lipid")

    assert list(df.columns) == list(SCHEMA_COLUMNS)
    assert len(df) == 2
    assert df["image_id"].tolist() == ["img_00000", "img_00000"]
    assert df["spot_id"].tolist() == [0, 1]
    # ch1 = lipid, ch2 = protein -> model-native intensities used DIRECTLY.
    r0 = df.iloc[0]
    assert r0["I1"] == pytest.approx(5000.0)
    assert r0["I2"] == pytest.approx(3000.0)
    assert r0["logI1"] == pytest.approx(math.log(5000.0))
    assert r0["logI2"] == pytest.approx(math.log(3000.0))
    # log_ratio / ratio stay derived-consistent.
    assert r0["log_ratio"] == pytest.approx(r0["logI2"] - r0["logI1"])
    assert r0["ratio"] == pytest.approx(3000.0 / 5000.0)
    assert r0["p_detect"] == pytest.approx(0.9)
    # uncertainty = std of logI = exp(0.5*logvar). lipid logvar=0 -> 1.0;
    # protein logvar=log(4) -> exp(0.5*log4) = 2.0.
    assert r0["uncertainty1"] == pytest.approx(1.0)
    assert r0["uncertainty2"] == pytest.approx(2.0)
    # No PSF-width head -> sigma*_hat NaN.
    assert math.isnan(r0["sigma1_hat"]) and math.isnan(r0["sigma2_hat"])
    # Honest provenance recorded in flags.
    assert "v0_hrnet" in r0["flags"] and "legacy" in r0["flags"]
    assert "units=legacy_flux" in r0["flags"]
    assert "ch1=lipid" in r0["flags"] and "ch2=protein" in r0["flags"]


def test_detections_to_canonical_reversed_mapping() -> None:
    from spotpipe.benchmark.v0_hrnet import detections_to_canonical

    df = detections_to_canonical("imgB", _fake_legacy_dets(), ch1_channel="protein")
    r0 = df.iloc[0]
    # ch1 = protein now -> I1 is the protein flux, I2 the lipid flux.
    assert r0["I1"] == pytest.approx(3000.0)
    assert r0["I2"] == pytest.approx(5000.0)
    assert r0["uncertainty1"] == pytest.approx(2.0)   # protein logvar=log(4)
    assert r0["uncertainty2"] == pytest.approx(1.0)   # lipid logvar=0
    assert "ch1=protein" in r0["flags"] and "ch2=lipid" in r0["flags"]


def test_detections_to_canonical_rejects_bad_channel() -> None:
    from spotpipe.benchmark.v0_hrnet import detections_to_canonical

    with pytest.raises(ValueError):
        detections_to_canonical("imgB", _fake_legacy_dets(), ch1_channel="rgb")


# --------------------------------------------------------------------------- #
# 5. load_canonical_predictions validates the full schema                      #
# --------------------------------------------------------------------------- #
def test_load_canonical_predictions_validates_schema() -> None:
    from spotpipe.benchmark.v0_hrnet import load_canonical_predictions

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.csv"
        pd.DataFrame({"image_id": ["a"], "x": [1.0], "y": [2.0]}).to_csv(bad, index=False)
        with pytest.raises(ValueError):
            load_canonical_predictions(bad)


# --------------------------------------------------------------------------- #
# 6. Adapter: load CSV, subset to eval images, fabricate nothing               #
# --------------------------------------------------------------------------- #
def test_adapter_subsets_and_fabricates_nothing() -> None:
    from spotpipe.benchmark.adapters import get_adapter
    from spotpipe.benchmark.v0_hrnet import detections_to_canonical
    from spotpipe.schema import SCHEMA_COLUMNS, write_spots

    with tempfile.TemporaryDirectory() as tmp:
        # Canonical predictions for TWO images.
        pred = pd.concat(
            [
                detections_to_canonical("img_00000", _fake_legacy_dets()),
                detections_to_canonical("img_00001", _fake_legacy_dets()[:1]),
            ],
            ignore_index=True,
        )
        csv = Path(tmp) / "predictions.csv"
        write_spots(pred, csv)

        # Eval set asks for img_00000 (present) and img_00002 (absent).
        eval_set = [_eval_item("img_00000"), _eval_item("img_00002")]
        cfg = {"v0_hrnet": {"predictions_csv": str(csv)}}
        adapter = get_adapter("v0_hrnet", log_fn=lambda *_: None)
        out = adapter.predict(eval_set, cfg)

    assert list(out.columns) == list(SCHEMA_COLUMNS)
    # Only img_00000's 2 rows: img_00001 not requested, img_00002 not predicted.
    assert sorted(out["image_id"].unique()) == ["img_00000"]
    assert len(out) == 2


def test_adapter_requires_predictions_csv() -> None:
    from spotpipe.benchmark.adapters import get_adapter

    adapter = get_adapter("v0_hrnet", log_fn=lambda *_: None)
    with pytest.raises(ValueError):
        adapter.predict([_eval_item("img_00000")], {"v0_hrnet": {}})


def test_adapter_missing_csv_path_errors() -> None:
    from spotpipe.benchmark.adapters import get_adapter

    adapter = get_adapter("v0_hrnet", log_fn=lambda *_: None)
    cfg = {"v0_hrnet": {"predictions_csv": "does/not/exist.csv"}}
    with pytest.raises(FileNotFoundError):
        adapter.predict([_eval_item("img_00000")], cfg)


# --------------------------------------------------------------------------- #
# Standalone runner (mirrors the other stage tests)                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
