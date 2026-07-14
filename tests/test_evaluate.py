"""Tests for the shared blind evaluator (spotpipe.benchmark.evaluate).

Covers the four validation gates from docs/evaluator_convention.md plus the
pure-function contracts:

* Gate A -- identity/oracle: GT-vs-GT gives perfect detection, zero intensity/
  ratio bias, and recovers each set's injected true_alpha within its OLS SE.
* Gate B -- known-alpha recovery across the curvature sweep, incl. the alpha=0
  null control recovering ~0.
* Gate C -- the factor of 2: the fit uses log(sqrt(A1)); switching to log(A1)
  halves the slope.
* Gate D -- precision is a number (never NaN/`--`) for any cell with predictions.

Gate A/B use a small REAL subset (the curvature null control + two sets) so they
run in seconds; the full-benchmark oracle report is produced by
`spotpipe evaluate --oracle`. Everything else is tiny synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotpipe.benchmark import evaluate as ev
from spotpipe.benchmark.evaluate import (
    ConditionSpec,
    detection_metrics,
    evaluate_condition,
    fit_alpha,
    load_benchmark_info,
    load_ground_truth,
)
from spotpipe.paths import get_paths
from spotpipe.schema import SCHEMA_COLUMNS

BENCH = get_paths().dataset("benchmark")
_HAS_BENCH = (BENCH / "BENCH_MANIFEST.json").exists()
_needs_bench = pytest.mark.skipif(not _HAS_BENCH, reason="benchmark not generated")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _schema_df(rows: list[dict]) -> pd.DataFrame:
    """Build a schema-conforming frame, filling unspecified columns with 0/''."""
    out = {c: [] for c in SCHEMA_COLUMNS}
    for r in rows:
        for c in SCHEMA_COLUMNS:
            out[c].append(r.get(c, "" if c == "flags" else 0.0))
    return pd.DataFrame(out)


def _curved_gt(alpha_true: float, n: int = 500, seed: int = 0, scatter: float = 0.0):
    """Synthetic GT where log(A2/A1) = alpha_true * log(sqrt(A1)) (+ scatter)."""
    rng = np.random.default_rng(seed)
    logI1 = rng.uniform(3.0, 7.0, size=n)          # wide A1 spread
    log_ratio = alpha_true * (0.5 * logI1) + rng.normal(0.0, scatter, size=n)
    logI2 = logI1 + log_ratio
    rows = []
    for i in range(n):
        rows.append(dict(image_id="img0", spot_id=i,
                         x=float(rng.uniform(10, 240)), y=float(rng.uniform(10, 240)),
                         logI1=float(logI1[i]), logI2=float(logI2[i])))
    return _schema_df(rows)


# --------------------------------------------------------------------------- #
# Pure functions                                                               #
# --------------------------------------------------------------------------- #
def test_detection_metrics_basic():
    r, p, f1 = detection_metrics(n_gt=10, n_pred=8, n_matched=6)
    assert r == pytest.approx(0.6)
    assert p == pytest.approx(0.75)
    assert f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    # No predictions -> precision undefined (NaN), recall still defined.
    r0, p0, f10 = detection_metrics(n_gt=5, n_pred=0, n_matched=0)
    assert r0 == 0.0 and np.isnan(p0) and np.isnan(f10)


def test_fit_alpha_exact_recovers_slope():
    """With zero scatter the OLS slope is the injected alpha to float tolerance."""
    for alpha_true in (-1.2, -0.3, 0.0, 0.6, 1.2):
        gt = _curved_gt(alpha_true, n=400, seed=1, scatter=0.0)
        fit = fit_alpha(gt["logI1"].to_numpy(), gt["logI2"].to_numpy())
        assert fit.alpha == pytest.approx(alpha_true, abs=1e-9)
        assert fit.n == 400


def test_gate_c_factor_of_two():
    """Gate C: the fit uses log(sqrt(A1)); regressing on log(A1) halves the slope.

    A silent switch to log(A1) would make every recovered alpha off by exactly 2x.
    """
    gt = _curved_gt(0.8, n=600, seed=2, scatter=0.05)
    logI1 = gt["logI1"].to_numpy()
    logI2 = gt["logI2"].to_numpy()

    fit = fit_alpha(logI1, logI2)          # x = log(sqrt(A1)) = 0.5*logI1

    # Manual OLS of the SAME y on x = log(A1) (the WRONG axis).
    y = logI2 - logI1
    x_wrong = logI1
    slope_wrong = np.polyfit(x_wrong, y, 1)[0]

    assert fit.alpha == pytest.approx(2.0 * slope_wrong, rel=1e-9)
    # And the correct fit recovers the injected slope (0.8), not 0.4.
    assert fit.alpha == pytest.approx(0.8, abs=0.03)
    assert slope_wrong == pytest.approx(0.4, abs=0.02)


def test_gate_d_precision_defined_with_predictions():
    """Gate D: a cell WITH predictions always has a numeric precision (never --)."""
    gt = _schema_df([dict(image_id="i", spot_id=k, x=10.0 + 5 * k, y=20.0, logI1=5.0, logI2=5.0)
                     for k in range(5)])
    # 3 predictions: 2 near GT (match), 1 far away (an unmatched FALSE POSITIVE).
    pred = _schema_df([
        dict(image_id="i", spot_id=0, x=10.0, y=20.0, logI1=5.0, logI2=5.0),
        dict(image_id="i", spot_id=1, x=15.0, y=20.0, logI1=5.0, logI2=5.0),
        dict(image_id="i", spot_id=2, x=200.0, y=200.0, logI1=5.0, logI2=5.0),
    ])
    cond = ConditionSpec("snr_density", "snr=5_density=0.006", {})
    row = evaluate_condition(gt, pred, cond, match_distance_px=1.68)
    assert row["n_fp"] == 1
    assert np.isfinite(row["precision"])           # defined despite the FP
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(2 / 5)


def test_missing_predictions_reported_not_crashed():
    gt = _schema_df([dict(image_id="i", spot_id=0, x=10.0, y=10.0, logI1=5.0, logI2=5.0)])
    cond = ConditionSpec("snr_density", "snr=2_density=0.0006", {})
    row = evaluate_condition(gt, None, cond, match_distance_px=1.68)
    assert row["status"] == "missing"
    assert row["recall"] == 0.0 and np.isnan(row["precision"])


def test_oracle_condition_is_perfect_synthetic():
    """GT-vs-GT on a synthetic condition: perfect detection, zero bias."""
    gt = _curved_gt(0.5, n=200, seed=3, scatter=0.05)
    cond = ConditionSpec("curvature", "alpha=0.5", {"true_alpha": 0.5})
    row = evaluate_condition(gt, gt.copy(), cond, match_distance_px=1.68)
    assert row["recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    assert abs(row["logI1_bias"]) < 1e-9 and abs(row["logI2_bias"]) < 1e-9
    assert abs(row["log_ratio_bias"]) < 1e-9
    # Recovered alpha is the GT's own OLS slope, within its SE of the injected 0.5.
    assert abs(row["alpha_hat"] - 0.5) <= 4 * row["alpha_se"] + 1e-9


# --------------------------------------------------------------------------- #
# Gate A / B on a small REAL subset                                            #
# --------------------------------------------------------------------------- #
@_needs_bench
def test_gate_a_oracle_real_curvature_null_control():
    """Gate A on the real alpha=0 null control: perfect scores, alpha ~ 0."""
    info = load_benchmark_info(BENCH)
    cond = next(c for c in info.conditions if c.label == "alpha=0")
    assert cond.meta.get("null_control") is True
    gt = load_ground_truth(BENCH, cond)
    row = evaluate_condition(gt, gt.copy(), cond,
                             match_distance_px=info.match_distance_px())
    assert row["recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)
    assert abs(row["logI1_bias"]) < 1e-9 and abs(row["logI2_bias"]) < 1e-9
    assert abs(row["log_ratio_bias"]) < 1e-9
    # Null control: recovered alpha must be ~0 (within a few SE).
    assert abs(row["alpha_hat"]) <= 4 * row["alpha_se"] + 1e-6
    assert abs(row["alpha_hat"]) < 0.03


@_needs_bench
def test_gate_b_known_alpha_recovery_real():
    """Gate B: GT recovers injected alpha across a few real sets (within SE)."""
    info = load_benchmark_info(BENCH)
    for label in ("alpha=-1.2", "alpha=0.6", "alpha=1.2"):
        cond = next(c for c in info.conditions if c.label == label)
        gt = load_ground_truth(BENCH, cond)
        row = evaluate_condition(gt, gt.copy(), cond,
                                 match_distance_px=info.match_distance_px())
        assert abs(row["alpha_hat"] - cond.true_alpha) <= 5 * row["alpha_se"] + 1e-6
        assert abs(row["alpha_bias"]) < 0.03


@_needs_bench
def test_match_gate_from_manifest_not_hardcoded():
    """The match gate is 1.0 x max(sigma1, sigma2), read from the manifest."""
    info = load_benchmark_info(BENCH)
    assert info.match_distance_px() == pytest.approx(max(info.sigma1, info.sigma2))
    assert info.match_distance_px(2.0) == pytest.approx(2.0 * max(info.sigma1, info.sigma2))
