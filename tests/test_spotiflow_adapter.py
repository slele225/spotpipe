"""Tests for the Spotiflow + aperture adapter (fair external-detector method).

No Spotiflow install needed for the core tests: the adapter's only contract is the
normalized detections CSV, so we fabricate one (at the simulator's own ground-truth
centres, standing in for Spotiflow's localizations) and check the in-repo side
end-to-end. An optional smoke test runs real Spotiflow only if it is installed.

Runnable two ways (like the other stage tests)::

    uv run python tests/test_spotiflow_adapter.py     (standalone)
    uv run pytest tests/test_spotiflow_adapter.py

Covered:
  1. Importing the adapter registry does NOT require the external ``spotiflow``.
  2. ``load_normalized_spotiflow_detections`` validates the required columns.
  3. ``spots_yx_to_xy`` preserves the x=col / y=row convention (the (y,x)->(x,y) swap).
  4. The adapter consumes a normalized CSV + an in-memory eval set, derives the
     photon image from raw counts + meta, extracts aperture intensities (from the
     PHOTON image, not raw), and emits EXACTLY the canonical schema.
  5. Spotiflow's native ``p_detect`` is detection confidence only -- never intensity.
  6. The adapter module never references ``audit/``.
  7. A missing detections CSV fails with a useful message.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "train_smoke.yaml"


def _sim_config() -> dict:
    with open(SMOKE_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _tiny_eval_set(n_images: int = 3, shape=(64, 64), seed: int = 13):
    from spotpipe.benchmark.harness import build_eval_set

    return build_eval_set(_sim_config(), n_images=n_images, seed=seed, shape=shape)


def _fake_detections_from_gt(eval_set, *, with_p_detect: bool = True) -> pd.DataFrame:
    """One normalized detection per GT spot, at its true centre (stand-in for Spotiflow).

    Optionally attaches a bogus ``p_detect`` to confirm it flows to detection
    confidence ONLY and never contaminates the aperture intensities.
    """
    rows = []
    for item in eval_set:
        for _, g in item.gt.iterrows():
            row = {
                "image_id": item.image_id,
                "x": float(g["x"]),
                "y": float(g["y"]),
                "source": "spotiflow",
                "model_variant": "general",
                "detect_image": "raw_max",
            }
            if with_p_detect:
                row["p_detect"] = 0.42
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. Importing the registry must not require external spotiflow               #
# --------------------------------------------------------------------------- #
def test_import_does_not_require_spotiflow(monkeypatch) -> None:
    """Block the external ``spotiflow`` import, then load our module + build the adapter."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spotiflow" or name.startswith("spotiflow."):
            raise ImportError("external spotiflow blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Force a fresh import of the in-repo module under the block.
    sys.modules.pop("spotpipe.benchmark.spotiflow", None)
    mod = importlib.import_module("spotpipe.benchmark.spotiflow")
    assert mod.SPOTIFLOW_METHOD_GENERAL == "spotiflow_general_plus_aperture"

    from spotpipe.benchmark.adapters import get_adapter

    adapter = get_adapter("spotiflow_general_plus_aperture")
    assert adapter.model_variant == "general"
    ft = get_adapter("spotiflow_finetuned_spotpipe_synth_plus_aperture")
    assert ft.model_variant == "finetuned_spotpipe_synth"
    assert ft.name == "spotiflow_finetuned_spotpipe_synth_plus_aperture"


# --------------------------------------------------------------------------- #
# 2. CSV contract validation                                                  #
# --------------------------------------------------------------------------- #
def test_load_normalized_detections_validates_required_columns() -> None:
    from spotpipe.benchmark import spotiflow

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.csv"
        pd.DataFrame({"image_id": ["a"], "x": [1.0]}).to_csv(bad, index=False)  # no 'y'
        with pytest.raises(ValueError):
            spotiflow.load_normalized_spotiflow_detections(bad)

        good = Path(tmp) / "good.csv"
        pd.DataFrame({"image_id": [1, 2], "x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(good, index=False)
        df = spotiflow.load_normalized_spotiflow_detections(good)
        assert list(spotiflow.REQUIRED_COLUMNS) == ["image_id", "x", "y"]
        assert df["image_id"].tolist() == ["1", "2"]   # coerced to str
        assert df["x"].dtype == float and df["y"].dtype == float


# --------------------------------------------------------------------------- #
# 3. Coordinate convention: (y, x) image order -> spotpipe (x, y)             #
# --------------------------------------------------------------------------- #
def test_spots_yx_to_xy_convention() -> None:
    from spotpipe.benchmark.spotiflow import spots_yx_to_xy

    # Spotiflow returns rows of (row, col) = (y, x). Spotpipe wants x=col, y=row.
    spots = np.array([[5.0, 9.0], [1.0, 2.0]])   # (y=5,x=9), (y=1,x=2)
    xs, ys = spots_yx_to_xy(spots)
    assert np.allclose(xs, [9.0, 2.0])
    assert np.allclose(ys, [5.0, 1.0])
    # Empty input stays well-defined.
    xs0, ys0 = spots_yx_to_xy(np.empty((0, 2)))
    assert xs0.size == 0 and ys0.size == 0


# --------------------------------------------------------------------------- #
# 4. Adapter end-to-end on a fake normalized CSV (exact schema, photon I)     #
# --------------------------------------------------------------------------- #
def test_adapter_end_to_end_fake_detections() -> None:
    from spotpipe.benchmark.adapters import (
        ADAPTER_REGISTRY,
        CmeAnalysisPlusApertureAdapter,
        get_adapter,
    )
    from spotpipe.benchmark import baselines
    from spotpipe.schema import SCHEMA_COLUMNS

    assert "spotiflow_general_plus_aperture" in ADAPTER_REGISTRY
    adapter = get_adapter("spotiflow_general_plus_aperture")

    eval_set = _tiny_eval_set()
    n_gt = sum(len(item.gt) for item in eval_set)
    assert n_gt > 0, "tiny eval set produced no spots"

    with tempfile.TemporaryDirectory() as tmp:
        det_csv = Path(tmp) / "detections.csv"
        _fake_detections_from_gt(eval_set).to_csv(det_csv, index=False)

        cfg = {"spotiflow": {
            "detect_image": "raw_max",
            "window_radius_px": 3.0, "bg_inner_px": 5.0, "bg_outer_px": 8.0,
            "general": {"detections_csv": str(det_csv)},
        }}
        pred = adapter.predict(eval_set, cfg)

    # exactly the canonical schema, in order; one row per detection.
    assert list(pred.columns) == list(SCHEMA_COLUMNS)
    assert len(pred) == n_gt

    # intensities finite & positive, and come from the PHOTON image (gain=1.0),
    # NOT from raw counts with the per-channel gain divide.
    assert np.all(np.isfinite(pred["logI1"].to_numpy()))
    assert np.all(pred["I1"].to_numpy() > 0) and np.all(pred["I2"].to_numpy() > 0)

    item0 = eval_set[0]
    photon0 = CmeAnalysisPlusApertureAdapter._photon_for(item0)
    xs = item0.gt["x"].to_numpy(float)
    ys = item0.gt["y"].to_numpy(float)
    exp_I1 = baselines.aperture_photometry(photon0[0], xs, ys, r_ap=3.0, r_in=5.0, r_out=8.0, gain=1.0)
    got_I1 = pred[pred["image_id"] == item0.image_id]["I1"].to_numpy()
    assert np.allclose(got_I1, exp_I1, rtol=1e-6, atol=1e-6), "I1 must be the photon-image aperture read"

    # method provenance flag; Spotiflow emits no PSF width / uncertainty.
    assert (pred["flags"] == "source=spotiflow;model_variant=general;detect_image=raw_max;intensity=aperture_photon").all()
    assert pred["uncertainty1"].isna().all() and pred["sigma1_hat"].isna().all()


# --------------------------------------------------------------------------- #
# 5. Native p_detect is confidence only -- never intensity                    #
# --------------------------------------------------------------------------- #
def test_p_detect_is_confidence_not_intensity() -> None:
    from spotpipe.benchmark.adapters import get_adapter

    eval_set = _tiny_eval_set(n_images=2)
    adapter = get_adapter("spotiflow_general_plus_aperture")

    with tempfile.TemporaryDirectory() as tmp:
        with_p = Path(tmp) / "with_p.csv"
        without_p = Path(tmp) / "without_p.csv"
        _fake_detections_from_gt(eval_set, with_p_detect=True).to_csv(with_p, index=False)
        _fake_detections_from_gt(eval_set, with_p_detect=False).to_csv(without_p, index=False)

        base = {"detect_image": "raw_max", "window_radius_px": 3.0, "bg_inner_px": 5.0, "bg_outer_px": 8.0}
        pred_p = adapter.predict(eval_set, {"spotiflow": {**base, "general": {"detections_csv": str(with_p)}}})
        pred_np = adapter.predict(eval_set, {"spotiflow": {**base, "general": {"detections_csv": str(without_p)}}})

    # p_detect carried through as confidence (0.42), and defaults to 1.0 when absent.
    assert np.allclose(pred_p["p_detect"].to_numpy(), 0.42)
    assert np.allclose(pred_np["p_detect"].to_numpy(), 1.0)
    # Intensities are identical regardless of p_detect -> p_detect never feeds I.
    assert np.allclose(pred_p["I1"].to_numpy(), pred_np["I1"].to_numpy())
    assert np.allclose(pred_p["I2"].to_numpy(), pred_np["I2"].to_numpy())


# --------------------------------------------------------------------------- #
# 6. The adapter module never reads audit/ true background                    #
# --------------------------------------------------------------------------- #
def test_adapter_module_never_references_audit() -> None:
    src = (REPO_ROOT / "src" / "spotpipe" / "benchmark" / "spotiflow.py").read_text(encoding="utf-8")
    assert "audit" not in src.lower(), "fair adapter must never read audit/ true background"


# --------------------------------------------------------------------------- #
# 7. Missing detections CSV -> useful error                                   #
# --------------------------------------------------------------------------- #
def test_adapter_requires_detections_csv() -> None:
    from spotpipe.benchmark.adapters import get_adapter

    adapter = get_adapter("spotiflow_general_plus_aperture")
    # No CSV configured at all.
    with pytest.raises(ValueError, match="run_spotiflow_predict"):
        adapter.predict(_tiny_eval_set(n_images=1), {"spotiflow": {"general": {}}})
    # Configured but missing on disk.
    with pytest.raises(FileNotFoundError, match="run_spotiflow_predict"):
        adapter.predict(
            _tiny_eval_set(n_images=1),
            {"spotiflow": {"general": {"detections_csv": "does/not/exist.csv"}}},
        )


# --------------------------------------------------------------------------- #
# 8. Optional real-Spotiflow smoke (skipped unless installed)                 #
# --------------------------------------------------------------------------- #
def test_real_spotiflow_one_image_smoke() -> None:
    pytest.importorskip("spotiflow", reason="external spotiflow not installed in this env")
    from spotiflow.model import Spotiflow

    from spotpipe.benchmark.adapters import get_adapter
    from spotpipe.benchmark.spotiflow import spots_yx_to_xy
    from spotpipe.schema import SCHEMA_COLUMNS

    # One in-memory image; build raw_max exactly as the predict script does.
    item = _tiny_eval_set(n_images=1)[0]
    raw_max = np.maximum(item.image[0], item.image[1]).astype(np.float32)

    model = Spotiflow.from_pretrained("general")
    spots, _details = model.predict(raw_max)
    xs, ys = spots_yx_to_xy(spots)
    assert xs.shape == ys.shape

    # Feed Spotiflow's real detections through the in-repo adapter -> canonical schema.
    with tempfile.TemporaryDirectory() as tmp:
        det_csv = Path(tmp) / "detections.csv"
        pd.DataFrame({
            "image_id": [item.image_id] * xs.size,
            "x": xs, "y": ys, "p_detect": [np.nan] * xs.size,
            "source": "spotiflow", "model_variant": "general", "detect_image": "raw_max",
        }).to_csv(det_csv, index=False)
        adapter = get_adapter("spotiflow_general_plus_aperture")
        pred = adapter.predict(
            [item],
            {"spotiflow": {"detect_image": "raw_max", "window_radius_px": 3.0,
                           "bg_inner_px": 5.0, "bg_outer_px": 8.0,
                           "general": {"detections_csv": str(det_csv)}}},
        )
    assert list(pred.columns) == list(SCHEMA_COLUMNS)
    assert len(pred) == xs.size


if __name__ == "__main__":
    print("=" * 70)
    print("SPOTIFLOW ADAPTER TEST 1: registry builds the two methods")
    print("=" * 70)
    # The monkeypatch fixture is pytest-only; standalone, just check the registry
    # constructs both honest methods (the in-repo module imports no external dep).
    import spotpipe.benchmark.spotiflow  # noqa: F401
    from spotpipe.benchmark.adapters import get_adapter as _ga
    assert _ga("spotiflow_general_plus_aperture").model_variant == "general"
    assert _ga("spotiflow_finetuned_spotpipe_synth_plus_aperture").model_variant == "finetuned_spotpipe_synth"
    print("ok")

    print("\n" + "=" * 70)
    print("SPOTIFLOW ADAPTER TEST 2: normalized-CSV contract validation")
    print("=" * 70)
    test_load_normalized_detections_validates_required_columns()
    print("ok")

    print("\n" + "=" * 70)
    print("SPOTIFLOW ADAPTER TEST 3: (y,x)->(x,y) coordinate convention")
    print("=" * 70)
    test_spots_yx_to_xy_convention()
    print("ok")

    print("\n" + "=" * 70)
    print("SPOTIFLOW ADAPTER TEST 4: adapter end-to-end on fake detections")
    print("=" * 70)
    test_adapter_end_to_end_fake_detections()
    print("ok")

    print("\n" + "=" * 70)
    print("SPOTIFLOW ADAPTER TEST 5: p_detect is confidence, not intensity")
    print("=" * 70)
    test_p_detect_is_confidence_not_intensity()
    print("ok")

    print("\n" + "=" * 70)
    print("SPOTIFLOW ADAPTER TEST 6/7: no audit/ + missing-CSV errors")
    print("=" * 70)
    test_adapter_module_never_references_audit()
    test_adapter_requires_detections_csv()
    print("ok")

    print("\nAll Spotiflow adapter tests passed.")
