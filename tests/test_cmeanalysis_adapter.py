"""Tests for the CMEAnalysis + aperture adapter (fair external-detector method).

No MATLAB / CMEAnalysis needed: the adapter's only contract is the normalized
detections CSV, so we fabricate one (at the simulator's own ground-truth centres,
which stands in for CME's localizations) and check the in-repo side end-to-end.

Runnable two ways (like the other stage tests)::

    uv run python tests/test_cmeanalysis_adapter.py     (standalone)
    uv run pytest tests/test_cmeanalysis_adapter.py

Covered:
  1. ``load_normalized_detections`` validates the required columns.
  2. ``compute_p_detect`` honours the documented modes (default constant 1.0).
  3. The adapter consumes a normalized CSV + an in-memory eval set, derives the
     photon image from raw counts + meta, extracts aperture intensities, and emits
     EXACTLY the canonical schema. CME native ``A``/``slave_A`` are ignored for I.
"""

from __future__ import annotations

import math
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


def _tiny_eval_set(n_images: int = 3, shape=(64, 64), seed: int = 11):
    from spotpipe.benchmark.harness import build_eval_set

    return build_eval_set(_sim_config(), n_images=n_images, seed=seed, shape=shape)


def _fake_detections_from_gt(eval_set) -> pd.DataFrame:
    """One normalized detection per GT spot, at its true centre (stand-in for CME).

    Adds optional ``score`` / ``A`` / ``slave_A`` / ``channel`` columns so the
    optional-column handling is exercised; intensities must still come from the
    photon image, NOT from ``A``.
    """
    rows = []
    for item in eval_set:
        for _, g in item.gt.iterrows():
            rows.append({
                "image_id": item.image_id,
                "x": float(g["x"]),
                "y": float(g["y"]),
                "score": 0.01,          # p-value-like (small = confident)
                "A": 1234.0,            # bogus native amplitude; must be ignored for I
                "slave_A": 567.0,
                "channel": 2,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. CSV contract validation                                                  #
# --------------------------------------------------------------------------- #
def test_load_normalized_detections_validates_required_columns() -> None:
    from spotpipe.benchmark import cmeanalysis

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.csv"
        pd.DataFrame({"image_id": ["a"], "x": [1.0]}).to_csv(bad, index=False)  # no 'y'
        with pytest.raises(ValueError):
            cmeanalysis.load_normalized_detections(bad)

        good = Path(tmp) / "good.csv"
        pd.DataFrame({"image_id": [1, 2], "x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(good, index=False)
        df = cmeanalysis.load_normalized_detections(good)
        assert list(cmeanalysis.REQUIRED_COLUMNS) == ["image_id", "x", "y"]
        assert df["image_id"].tolist() == ["1", "2"]   # coerced to str
        assert df["x"].dtype == float and df["y"].dtype == float


# --------------------------------------------------------------------------- #
# 2. p_detect modes                                                           #
# --------------------------------------------------------------------------- #
def test_compute_p_detect_modes() -> None:
    from spotpipe.benchmark import cmeanalysis

    df = pd.DataFrame({"A": [10.0, 5.0, 0.0], "score": [0.0, 0.5, 1.0]})

    assert cmeanalysis.compute_p_detect(df, "constant") == 1.0
    assert cmeanalysis.compute_p_detect(pd.DataFrame({"x": [1]}), "A") == 1.0  # missing col -> 1.0

    a = np.asarray(cmeanalysis.compute_p_detect(df, "A"))
    assert np.isclose(a[0], 1.0) and a[1] < 1.0 and a.min() >= 0.0

    omp = np.asarray(cmeanalysis.compute_p_detect(df, "one_minus_pval"))
    assert np.allclose(omp, [1.0, 0.5, 0.0])

    nlp = np.asarray(cmeanalysis.compute_p_detect(df, "neg_log10_pval"))
    assert nlp.min() >= 0.0 and nlp.max() <= 1.0

    with pytest.raises(ValueError):
        cmeanalysis.compute_p_detect(df, "bogus_mode")


# --------------------------------------------------------------------------- #
# 3. Adapter end-to-end on a fake normalized CSV                              #
# --------------------------------------------------------------------------- #
def test_adapter_end_to_end_fake_detections() -> None:
    from spotpipe.benchmark.adapters import (
        ADAPTER_REGISTRY,
        CmeAnalysisPlusApertureAdapter,
        get_adapter,
    )
    from spotpipe.schema import SCHEMA_COLUMNS

    # registered + constructible with no kwargs
    assert ADAPTER_REGISTRY["cmeanalysis_plus_aperture"] is CmeAnalysisPlusApertureAdapter
    adapter = get_adapter("cmeanalysis_plus_aperture")

    eval_set = _tiny_eval_set()
    n_gt = sum(len(item.gt) for item in eval_set)
    assert n_gt > 0, "tiny eval set produced no spots"

    with tempfile.TemporaryDirectory() as tmp:
        det_csv = Path(tmp) / "detections.csv"
        _fake_detections_from_gt(eval_set).to_csv(det_csv, index=False)

        cfg = {"cmeanalysis": {
            "detections_csv": str(det_csv),
            "p_detect_source": "constant",
            "window_radius_px": 3.0, "bg_inner_px": 4.0, "bg_outer_px": 7.0,
        }}
        pred = adapter.predict(eval_set, cfg)

    # exactly the canonical schema, in order
    assert list(pred.columns) == list(SCHEMA_COLUMNS)
    assert len(pred) == n_gt, "one prediction per fake detection expected"

    # constant p_detect; intensities finite & positive (from photon image, not A)
    assert np.allclose(pred["p_detect"].to_numpy(), 1.0)
    assert np.all(np.isfinite(pred["logI1"].to_numpy()))
    assert np.all(np.isfinite(pred["logI2"].to_numpy()))
    assert np.all(pred["I1"].to_numpy() > 0) and np.all(pred["I2"].to_numpy() > 0)
    # the bogus native A=1234 was NOT used: aperture intensities differ from it.
    assert not np.allclose(pred["I1"].to_numpy(), 1234.0)

    # method provenance flag; CME emits no PSF width / uncertainty.
    assert (pred["flags"] == "cmeanalysis_plus_aperture").all()
    assert pred["uncertainty1"].isna().all() and pred["sigma1_hat"].isna().all()


def test_adapter_requires_detections_csv() -> None:
    from spotpipe.benchmark.adapters import get_adapter

    adapter = get_adapter("cmeanalysis_plus_aperture")
    with pytest.raises(ValueError):
        adapter.predict(_tiny_eval_set(n_images=1), {"cmeanalysis": {}})


# --------------------------------------------------------------------------- #
# 4. --limit image selection helper (scripts/run_cmeanalysis.py)              #
# --------------------------------------------------------------------------- #
def _runner_module():
    """Import scripts/run_cmeanalysis.py as a module (it lives outside any package)."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "run_cmeanalysis.py"
    spec = importlib.util.spec_from_file_location("run_cmeanalysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_image_entries() -> None:
    runner = _runner_module()
    entries = [{"image_id": f"img_{i:05d}"} for i in range(5)]

    # None -> all, in stable manifest order
    assert runner.select_image_entries(entries, None) == entries

    # first N, stable order
    sel = runner.select_image_entries(entries, 3)
    assert [e["image_id"] for e in sel] == ["img_00000", "img_00001", "img_00002"]

    # limit larger than the set -> all; limit 0 -> empty
    assert runner.select_image_entries(entries, 99) == entries
    assert runner.select_image_entries(entries, 0) == []

    # negative is an error
    with pytest.raises(ValueError):
        runner.select_image_entries(entries, -1)


def test_make_batches() -> None:
    runner = _runner_module()
    entries = [{"image_id": f"img_{i:05d}"} for i in range(139)]

    # exact batch sizes: 139 -> 64 + 64 + 11 (the recommended --batch-size 64)
    b = runner.make_batches(entries, 64)
    assert [len(x) for x in b] == [64, 64, 11]

    # stable order + exact image_ids preserved (no renaming, no reordering)
    assert b[0][0]["image_id"] == "img_00000"
    assert b[0][-1]["image_id"] == "img_00063"
    assert b[1][0]["image_id"] == "img_00064"
    assert b[2][-1]["image_id"] == "img_00138"
    assert [e for batch in b for e in batch] == entries  # flatten round-trips

    # limit + batch_size interaction: select first 100, then batch by 64 -> 64 + 36
    sel = runner.select_image_entries(entries, 100)
    b2 = runner.make_batches(sel, 64)
    assert [len(x) for x in b2] == [64, 36]
    assert b2[-1][-1]["image_id"] == "img_00099"

    # even division
    assert [len(x) for x in runner.make_batches(entries[:128], 64)] == [64, 64]

    # empty -> no batches; invalid sizes -> error
    assert runner.make_batches([], 64) == []
    with pytest.raises(ValueError):
        runner.make_batches(entries, 0)
    with pytest.raises(ValueError):
        runner.make_batches(entries, -3)
    with pytest.raises(ValueError):
        runner.make_batches(entries, None)


def test_diag_sweep_parsers() -> None:
    runner = _runner_module()

    # sweep-limit spec parsing ('full'/'all'/'none' -> None)
    assert runner._parse_sweep_limits("4,8,16,32,64,full") == [4, 8, 16, 32, 64, None]
    assert runner._parse_sweep_limits(" 2 , full , 5 ") == [2, None, 5]

    # sigma line parsing: takes the last printed value; flags NaN
    ok_txt = "noise\nGaussian PSF s.d. values:  1.40 1.40\nRunning detection ..."
    sigma, ok = runner._parse_sigma(ok_txt)
    assert sigma == "1.40 1.40" and ok is True

    nan_txt = "Gaussian PSF s.d. values:  NaN NaN\n"
    sigma, ok = runner._parse_sigma(nan_txt)
    assert "NaN" in sigma and ok is False

    sigma, ok = runner._parse_sigma("no sigma line here")
    assert sigma == "(not printed)" and ok is False


if __name__ == "__main__":
    print("=" * 70)
    print("CME ADAPTER TEST 1: normalized-CSV contract validation")
    print("=" * 70)
    test_load_normalized_detections_validates_required_columns()
    print("ok")

    print("\n" + "=" * 70)
    print("CME ADAPTER TEST 2: p_detect modes")
    print("=" * 70)
    test_compute_p_detect_modes()
    print("ok")

    print("\n" + "=" * 70)
    print("CME ADAPTER TEST 3: adapter end-to-end on fake detections")
    print("=" * 70)
    test_adapter_end_to_end_fake_detections()
    test_adapter_requires_detections_csv()
    print("ok")

    print("\n" + "=" * 70)
    print("CME ADAPTER TEST 4: --limit image selection helper")
    print("=" * 70)
    test_select_image_entries()
    test_make_batches()
    test_diag_sweep_parsers()
    print("ok")

    print("\nAll CME adapter tests passed.")
