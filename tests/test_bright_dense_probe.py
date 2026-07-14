"""Golden tests for the bright x dense intensity probe.

The probe is the instrument that will decide whether to spend 40k A100 steps on a
different intensity sampler. An instrument that lies is worse than no instrument, so it
gets tested like one:

* a PERFECT predictor (ground truth fed back in as predictions) must report EXACTLY zero
  bias and recall 1.0 -- otherwise every bias it reports carries an unknown offset;
* a predictor with a KNOWN injected offset must report exactly that offset back;
* the rendered cells must actually be the thing we asked for: constant intensity, constant
  density, zero registration shift (the shift default is 1.0 and has silently poisoned
  ground truth once already -- see VENDORED_NOTES.md).

These run without a GPU and without a trained model; the model forward pass is the only
part that is not covered here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# NOTE: deliberately NO torch stub here. `bright_dense_probe` imports torch lazily (inside
# main()), so its rendering + matching functions import clean. Installing a global stub from
# a test module would poison every OTHER test in the same pytest session -- the tests that
# genuinely need a real torch would then fail for the wrong reason.

from spotpipe.benchmark.generate import (  # noqa: E402
    _bench_axis_params,
    _solve_intensity_for_snr,
    load_benchmark_config,
)
from spotpipe.paths import get_paths  # noqa: E402

bdp = pytest.importorskip("bright_dense_probe")


@pytest.fixture(scope="module")
def base_config():
    cfg, _ = load_benchmark_config(get_paths().configs / "benchmark.yaml")
    return cfg


@pytest.fixture(scope="module")
def cell(base_config):
    """One bright, dense cell -- the corner the whole probe exists to measure."""
    axis = _bench_axis_params(base_config)
    A = float(_solve_intensity_for_snr(10.0, axis)["intensity"])
    sims, det = bdp.render_cell(base_config, A, 0.012, n_images=2, seed=0)
    return sims, A


def test_perfect_predictor_has_exactly_zero_bias(cell):
    # Ground truth in, ground truth out. Any nonzero bias here is an artefact of the
    # probe's own matching/column handling and would contaminate every number it reports.
    sims, _A = cell
    gt = sims[0].spots
    res = bdp.match_and_bias(gt.copy(), gt, radius=1.68)
    assert res["recall"] == pytest.approx(1.0)
    assert res["n_matched"] == len(gt)
    for col in ("logI1", "logI2", "log_ratio"):
        assert res[f"{col}_bias"] == pytest.approx(0.0, abs=1e-12)
        assert res[f"{col}_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_known_offset_is_recovered(cell):
    # Inject a -1.10 log offset into ch2 -- the exact size of the defect being chased --
    # and require the probe to hand it back. This is what makes a reported -1.10 mean
    # "the model under-reads ch2 by 3x" rather than "the probe has a bug".
    sims, _A = cell
    gt = sims[0].spots
    pred = gt.copy()
    pred["logI2"] = pred["logI2"] - 1.10
    pred["log_ratio"] = pred["logI2"] - pred["logI1"]
    res = bdp.match_and_bias(pred, gt, radius=1.68)
    assert res["logI1_bias"] == pytest.approx(0.0, abs=1e-12)
    assert res["logI2_bias"] == pytest.approx(-1.10, abs=1e-9)
    assert res["log_ratio_bias"] == pytest.approx(-1.10, abs=1e-9)


def test_matching_does_not_reuse_a_prediction(cell):
    # One prediction cannot satisfy two ground-truth spots. If it could, a single lucky
    # detection in a crowded field would inflate recall and quietly bias the intensity mean.
    sims, _A = cell
    gt = sims[0].spots.iloc[:5].copy()
    single = gt.iloc[[0]].copy()
    res = bdp.match_and_bias(single, gt, radius=1e6)   # radius so wide everything is "close"
    assert res["n_matched"] == 1


def test_rendered_cell_is_constant_intensity_and_density(cell):
    sims, A = cell
    for sim in sims:
        i1 = sim.spots["I1"].to_numpy(float)
        i2 = sim.spots["I2"].to_numpy(float)
        # neutral zero-scatter ratio law => A2 == A1 == A exactly, no jitter
        assert np.allclose(i1, A, rtol=1e-9), f"intensity not pinned: {np.unique(i1)[:3]}"
        assert np.allclose(i2, i1, rtol=1e-9), "A2 != A1: the neutral ratio law is not applied"
    counts = {len(s.spots) for s in sims}
    assert len(counts) == 1, f"n_spots varies across a constant-density cell: {counts}"


def test_registration_shift_is_zero(cell):
    # The forward model's registration_shift default is 1.0 px. If _ZERO_REGISTRATION were
    # dropped, ground-truth positions would mark a point the photons are NOT centred on,
    # and the probe's "bias" would silently absorb a localization error instead.
    sims, _A = cell
    for sim in sims:
        meta = sim.meta
        shifts = [meta.get("shift1"), meta.get("shift2")]
        for s in shifts:
            if s is None:
                continue
            assert np.allclose(np.asarray(s, float), 0.0, atol=1e-12), \
                f"non-zero registration shift leaked into the probe: {s}"


def test_diag_grid_spans_the_defect_and_the_new_grid_top():
    # The probe must straddle both regimes or it cannot separate them: SNR 10/15 are where
    # the defect was characterised, SNR 2/3 are the top of the v3 benchmark.
    assert 10.0 in bdp.DIAG_SNR and 15.0 in bdp.DIAG_SNR
    assert 2.0 in bdp.DIAG_SNR and 3.0 in bdp.DIAG_SNR
    assert min(bdp.DIAG_DENSITY) <= 0.0006 and max(bdp.DIAG_DENSITY) >= 0.012
