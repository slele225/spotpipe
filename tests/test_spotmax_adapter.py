"""Tests for the SpotMAX + aperture adapter (fair external-detector method).

No SpotMAX install needed for the core tests: the adapter's only contract is the
normalized/neutral detections CSV, so we fabricate SpotMAX-style output tables and
tiny photon images and check the in-repo side end-to-end. An optional smoke test
runs the real ``spotmax`` CLI only if it is on PATH.

Runnable two ways (like the other stage tests)::

    uv run python tests/test_spotmax_adapter.py     (standalone)
    uv run pytest tests/test_spotmax_adapter.py

Covered:
  1. Importing spotpipe / the adapter registry does NOT require external ``spotmax``.
  2. A fake SpotMAX output table parses into the neutral detections CSV, mapping
     position/confidence to spotpipe convention and preserving native columns JSON.
  3. ``resolve_xy_columns`` enforces x=column / y=row (candidates, override, error).
  4. Fake neutral detections + tiny photon images -> EXACTLY the canonical 16
     columns, with I1/I2 read from the PHOTON image (gain=1.0), p_detect=NaN kept.
  5. Coordinate convention is preserved (x indexes column, y indexes row).
  6. Non-positive extracted intensities are handled EXPLICITLY (clamp+flag / reject),
     with transparent counts and no silent drops.
  7. ``load_normalized_spotmax_detections`` validates the required columns.
  8. The adapter requires a detections CSV (useful errors) and the module never
     reads ``audit/`` true background.
"""

from __future__ import annotations

import builtins
import importlib
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _write_fake_spotmax_run(root: Path, *, table_name="1_valid_spots.csv", columns=None):
    """Build a fake SpotMAX run tree with one Position + output table.

    Returns ``(run_dir, id_map)``. The table uses SpotMAX-flavoured (z,y,x) columns
    plus a native intensity column we must NOT use as canonical I.
    """
    if columns is None:
        # x = column, y = row (SpotMAX/scikit-image convention).
        columns = pd.DataFrame({
            "Cell_ID": [1, 1, 2],
            "z": [0, 0, 0],
            "y": [10.0, 20.0, 30.0],     # row
            "x": [40.0, 50.0, 60.0],     # column
            "spot_vs_backgr_effect_size": [3.1, 2.0, 5.5],   # native, never used as I
        })
    position = "Position_000001"
    sm_dir = root / position / "SpotMAX_output"
    sm_dir.mkdir(parents=True, exist_ok=True)
    columns.to_csv(sm_dir / table_name, index=False)
    return root, {position: "img_00007"}


def _bright_photon(shape=(64, 64), centres=((40, 10), (50, 20)), amp=500.0, bg=5.0):
    """Two-channel photon image with bright Gaussian blobs at ``(col,row)`` centres."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), bg, dtype=float)
    for cx, cy in centres:
        img = img + amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.2 ** 2))
    return np.stack([img, img * 0.5], axis=0)


# --------------------------------------------------------------------------- #
# 1. Importing must not require external spotmax                               #
# --------------------------------------------------------------------------- #
def test_import_does_not_require_spotmax(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spotmax" or name.startswith("spotmax."):
            raise ImportError("external spotmax blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("spotpipe.benchmark.spotmax", None)
    mod = importlib.import_module("spotpipe.benchmark.spotmax")
    assert mod.SPOTMAX_METHOD == "spotmax_ai_plus_aperture"

    from spotpipe.benchmark.adapters import ADAPTER_REGISTRY, get_adapter

    assert "spotmax_ai_plus_aperture" in ADAPTER_REGISTRY
    adapter = get_adapter("spotmax_ai_plus_aperture")
    assert adapter.name == "spotmax_ai_plus_aperture"


def test_module_does_not_import_spotmax_at_source() -> None:
    src = (REPO_ROOT / "src" / "spotpipe" / "benchmark" / "spotmax.py").read_text(encoding="utf-8")
    assert "import spotmax" not in src, "in-repo module must never import the external spotmax"


# --------------------------------------------------------------------------- #
# 2. Parse fake SpotMAX output -> neutral detections                          #
# --------------------------------------------------------------------------- #
def test_parse_spotmax_output_to_neutral() -> None:
    from spotpipe.benchmark import spotmax as smx

    with tempfile.TemporaryDirectory() as tmp:
        run_dir, id_map = _write_fake_spotmax_run(Path(tmp))
        neutral = smx.parse_spotmax_output(run_dir, id_map)

    assert list(neutral.columns) == list(smx.NEUTRAL_COLUMNS)
    assert len(neutral) == 3
    assert (neutral["image_id"] == "img_00007").all()
    # x = column, y = row (NOT swapped).
    assert neutral["x"].tolist() == [40.0, 50.0, 60.0]
    assert neutral["y"].tolist() == [10.0, 20.0, 30.0]
    # No native confidence column present -> p_detect NaN (never fabricated).
    assert neutral["p_detect"].isna().all()
    # native row preserved as JSON for auditability.
    native = json.loads(neutral["native_columns_json"].iloc[0])
    assert native["spot_vs_backgr_effect_size"] == 3.1
    assert native["x"] == 40.0 and native["y"] == 10.0
    assert neutral["native_source_file"].iloc[0] == "1_valid_spots.csv"


def test_parse_prefers_valid_over_detected() -> None:
    from spotpipe.benchmark import spotmax as smx

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sm = root / "Position_000001" / "SpotMAX_output"
        sm.mkdir(parents=True)
        pd.DataFrame({"x": [1.0], "y": [2.0]}).to_csv(sm / "0_detected_spots.csv", index=False)
        pd.DataFrame({"x": [9.0], "y": [8.0]}).to_csv(sm / "1_valid_spots.csv", index=False)
        neutral = smx.parse_spotmax_output(root, {"Position_000001": "imgA"})
    # valid spots win over detected spots.
    assert neutral["x"].tolist() == [9.0] and neutral["y"].tolist() == [8.0]
    assert neutral["native_source_file"].iloc[0] == "1_valid_spots.csv"


# --------------------------------------------------------------------------- #
# 3. Coordinate-column resolution (x=col, y=row)                              #
# --------------------------------------------------------------------------- #
def test_resolve_xy_columns() -> None:
    from spotpipe.benchmark import spotmax as smx

    assert smx.resolve_xy_columns(["z", "y", "x"]) == ("x", "y")
    assert smx.resolve_xy_columns(["x_local", "y_local"]) == ("x_local", "y_local")
    # explicit override wins
    assert smx.resolve_xy_columns(["a", "b", "x", "y"], x_col="a", y_col="b") == ("a", "b")
    # no coordinate columns -> loud error (not a silent guess)
    with pytest.raises(ValueError):
        smx.resolve_xy_columns(["foo", "bar"])
    # explicit but absent -> error
    with pytest.raises(ValueError):
        smx.resolve_xy_columns(["x", "y"], x_col="nope")


def test_resolve_p_detect_column() -> None:
    from spotpipe.benchmark import spotmax as smx

    assert smx.resolve_p_detect_column(["x", "y"]) is None         # honest default
    assert smx.resolve_p_detect_column(["x", "y", "score"]) == "score"
    assert smx.resolve_p_detect_column(["x", "y", "conf"], p_col="conf") == "conf"


# --------------------------------------------------------------------------- #
# 4. Neutral detections + photon images -> canonical 16 columns               #
# --------------------------------------------------------------------------- #
def test_spotmax_plus_aperture_canonical_schema() -> None:
    from spotpipe.benchmark import baselines, spotmax as smx
    from spotpipe.schema import SCHEMA_COLUMNS

    photon = _bright_photon(centres=((40, 10), (50, 20)))
    det = pd.DataFrame({"image_id": ["imgA", "imgA"], "x": [40.0, 50.0], "y": [10.0, 20.0],
                        "p_detect": [np.nan, np.nan]})
    cfg = {"window_radius_px": 3.0, "bg_inner_px": 5.0, "bg_outer_px": 8.0}
    pred, stats = smx.spotmax_plus_aperture(photon, det, image_id="imgA", cfg=cfg)

    assert list(pred.columns) == list(SCHEMA_COLUMNS)
    assert len(pred) == 2
    assert stats == {"n_in": 2, "n_out": 2, "n_nonpositive": 0}
    # intensities are the PHOTON-image aperture reads (gain=1.0), positive & finite.
    exp_I1 = baselines.aperture_photometry(photon[0], np.array([40.0, 50.0]), np.array([10.0, 20.0]),
                                           r_ap=3.0, r_in=5.0, r_out=8.0, gain=1.0)
    assert np.allclose(pred["I1"].to_numpy(), exp_I1, rtol=1e-6, atol=1e-6)
    assert np.all(pred["I1"].to_numpy() > 0) and np.all(pred["I2"].to_numpy() > 0)
    # log_ratio / ratio consistency
    assert np.allclose(pred["log_ratio"], pred["logI2"] - pred["logI1"])
    # SpotMAX emits no PSF width / uncertainty; p_detect kept NaN (not fabricated).
    assert pred["sigma1_hat"].isna().all() and pred["uncertainty1"].isna().all()
    assert pred["p_detect"].isna().all()
    assert (pred["flags"] == "spotmax_ai_plus_aperture;detect_image=raw_max;photometry=aperture_annulus").all()


# --------------------------------------------------------------------------- #
# 5. Coordinate convention preserved (x=col, y=row) through photometry         #
# --------------------------------------------------------------------------- #
def test_coordinate_convention_x_is_column() -> None:
    from spotpipe.benchmark import spotmax as smx

    # One bright blob at column=40, row=10.
    photon = _bright_photon(centres=((40, 10),), amp=500.0, bg=1.0)
    cfg = {"window_radius_px": 3.0, "bg_inner_px": 5.0, "bg_outer_px": 8.0}

    # Correct (x=col=40, y=row=10) lands on the blob -> bright.
    hit, _ = smx.spotmax_plus_aperture(photon, pd.DataFrame({"x": [40.0], "y": [10.0]}),
                                       image_id="i", cfg=cfg)
    # Swapped (x=10, y=40) lands on background -> dim.
    miss, _ = smx.spotmax_plus_aperture(photon, pd.DataFrame({"x": [10.0], "y": [40.0]}),
                                        image_id="i", cfg=cfg)
    assert float(hit["I1"].iloc[0]) > 10.0 * float(miss["I1"].iloc[0]), \
        "x must index column and y must index row (bright blob recovered only at (40,10))"


# --------------------------------------------------------------------------- #
# 6. Non-positive intensity handling is explicit                              #
# --------------------------------------------------------------------------- #
def test_nonpositive_intensity_clamp_and_reject() -> None:
    from spotpipe.benchmark import spotmax as smx

    # Flat background so the annulus median ~ aperture mean -> background-subtracted
    # signal is ~0 -> clamped to the floor. One real blob, one flat-region detection.
    photon = _bright_photon(centres=((40, 10),), amp=500.0, bg=20.0)
    det = pd.DataFrame({"x": [40.0, 5.0], "y": [10.0, 5.0]})   # blob, then flat region
    cfg = {"window_radius_px": 3.0, "bg_inner_px": 5.0, "bg_outer_px": 8.0}

    clamp, s_clamp = smx.spotmax_plus_aperture(photon, det, image_id="i", cfg={**cfg, "nonpositive": "clamp"})
    assert s_clamp["n_in"] == 2 and s_clamp["n_out"] == 2
    assert s_clamp["n_nonpositive"] >= 1
    # the clamped row keeps its place AND is flagged (no silent drop).
    flagged = clamp["flags"].str.contains("nonpos_clamped")
    assert int(flagged.sum()) == s_clamp["n_nonpositive"]

    reject, s_rej = smx.spotmax_plus_aperture(photon, det, image_id="i", cfg={**cfg, "nonpositive": "reject"})
    assert s_rej["n_in"] == 2
    assert s_rej["n_out"] == 2 - s_rej["n_nonpositive"]   # rejected rows dropped, and counted
    assert len(reject) == s_rej["n_out"]


# --------------------------------------------------------------------------- #
# 7. Normalized-CSV contract validation                                       #
# --------------------------------------------------------------------------- #
def test_load_normalized_validates_required_columns() -> None:
    from spotpipe.benchmark import spotmax as smx

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.csv"
        pd.DataFrame({"image_id": ["a"], "x": [1.0]}).to_csv(bad, index=False)  # no 'y'
        with pytest.raises(ValueError):
            smx.load_normalized_spotmax_detections(bad)

        good = Path(tmp) / "good.csv"
        pd.DataFrame({"image_id": [1, 2], "x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(good, index=False)
        df = smx.load_normalized_spotmax_detections(good)
        assert df["image_id"].tolist() == ["1", "2"]    # coerced to str
        assert df["x"].dtype == float and df["y"].dtype == float


# --------------------------------------------------------------------------- #
# 8. Adapter requires detections CSV + never reads audit/                     #
# --------------------------------------------------------------------------- #
def test_adapter_requires_detections_csv() -> None:
    from spotpipe.benchmark.adapters import get_adapter

    adapter = get_adapter("spotmax_ai_plus_aperture")
    with pytest.raises(ValueError, match="detections_csv"):
        adapter.predict([], {"spotmax": {}})
    with pytest.raises(FileNotFoundError):
        adapter.predict([], {"spotmax": {"detections_csv": "does/not/exist.csv"}})


def test_adapter_module_never_references_audit() -> None:
    src = (REPO_ROOT / "src" / "spotpipe" / "benchmark" / "spotmax.py").read_text(encoding="utf-8")
    assert "audit" not in src.lower(), "fair adapter must never read audit/ true background"


def test_adapter_end_to_end_with_stub_eval_item() -> None:
    """Adapter reads a neutral CSV + an eval item with attached photon -> canonical."""
    from spotpipe.benchmark.adapters import get_adapter
    from spotpipe.schema import SCHEMA_COLUMNS

    photon = _bright_photon(centres=((40, 10), (50, 20)))
    item = SimpleNamespace(image_id="imgA", photon=photon, image=None, meta={})
    with tempfile.TemporaryDirectory() as tmp:
        det_csv = Path(tmp) / "neutral.csv"
        pd.DataFrame({"image_id": ["imgA", "imgA"], "x": [40.0, 50.0], "y": [10.0, 20.0],
                      "p_detect": [np.nan, 0.9]}).to_csv(det_csv, index=False)
        adapter = get_adapter("spotmax_ai_plus_aperture")
        pred = adapter.predict([item], {"spotmax": {"detections_csv": str(det_csv),
                                                    "window_radius_px": 3.0,
                                                    "bg_inner_px": 5.0, "bg_outer_px": 8.0}})
    assert list(pred.columns) == list(SCHEMA_COLUMNS)
    assert len(pred) == 2
    # p_detect carried through (NaN stays NaN, 0.9 preserved) -- never used as intensity.
    assert math.isnan(float(pred["p_detect"].iloc[0])) and float(pred["p_detect"].iloc[1]) == 0.9


# --------------------------------------------------------------------------- #
# 9. Optional real-SpotMAX CLI smoke (skipped unless installed)               #
# --------------------------------------------------------------------------- #
def test_real_spotmax_cli_available_smoke() -> None:
    import shutil

    if shutil.which("spotmax") is None:
        pytest.skip("spotmax CLI not on PATH (run the real smoke in its own env)")
    # We do not drive a full headless run here (needs models/INI for the installed
    # version); just confirm the export tree + parser glue is importable alongside a
    # real install. The full run is documented in the script docstrings.
    from spotpipe.benchmark import spotmax as smx
    assert smx.SPOTMAX_METHOD == "spotmax_ai_plus_aperture"


if __name__ == "__main__":
    print("=" * 70 + "\nSPOTMAX ADAPTER TESTS (standalone)\n" + "=" * 70)
    test_module_does_not_import_spotmax_at_source(); print("1. no spotmax import ....... ok")
    test_parse_spotmax_output_to_neutral(); print("2. parse -> neutral ........ ok")
    test_parse_prefers_valid_over_detected(); print("   valid>detected ......... ok")
    test_resolve_xy_columns(); test_resolve_p_detect_column(); print("3. column resolution ....... ok")
    test_spotmax_plus_aperture_canonical_schema(); print("4. canonical 16 columns .... ok")
    test_coordinate_convention_x_is_column(); print("5. x=col, y=row ............ ok")
    test_nonpositive_intensity_clamp_and_reject(); print("6. nonpositive handling .... ok")
    test_load_normalized_validates_required_columns(); print("7. CSV contract ............ ok")
    test_adapter_requires_detections_csv(); test_adapter_module_never_references_audit()
    test_adapter_end_to_end_with_stub_eval_item(); print("8. adapter end-to-end ...... ok")
    print("\nAll SpotMAX adapter tests passed.")
