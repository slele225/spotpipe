"""Tests for the measured-detector retrain training layer.

Covers the two prompt-mandated per-image departures (gain randomisation +
saturation-safe intensity solving), the curriculum, the multi-worker loader, and
the overfit / smoke self-checks. All run on CPU with tiny sizes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spotpipe.simulator import forward_model
from spotpipe.training.dataset import (
    IntensityWindowConfig,
    build_eval_examples,
    curriculum_scene_config,
    generate_examples,
    make_loader,
    summarize_solved_windows,
)
from spotpipe.training.intensity_window import (
    DetectorConstants,
    clean_gained_peak,
    sample_image_detector,
    solve_a1_ceiling,
)
from spotpipe.training.targets import TARGET_KEYS, build_targets

DET_CFG = {
    "n_frames": 3, "poisson_gaussian_threshold": 20, "adc_max": 4095,
    "ch1": {"gain_range": [3, 30], "offset": 154.0, "read_var": 3.1,
            "saturation_knee": 3941.0, "excess_noise_factor": 1.0},
    "ch2": {"gain_range": [20, 150], "offset": 154.0, "read_var": 4.4,
            "saturation_knee": 3941.0, "excess_noise_factor": 1.0},
}

SCENE_CFG = {
    "density": {"min": 0.0006, "max": 0.012}, "oversample_dense_fraction": 0.3,
    "ratio_law": {"alpha": {"min": -0.7, "max": 0.7}, "beta": {"min": -0.6, "max": 0.6},
                  "scatter_std": {"min": 0.03, "max": 0.25}},
    "psf": {"sigma1": {"min": 1.0, "max": 1.8}, "c2_sigma_mismatch": {"min": 1.05, "max": 1.35}},
    "registration_shift": {"max_px": 1.0},
    "background": {"level": {"min": 1.0, "max": 4.0}, "gradient_frac": {"min": 0.0, "max": 0.0},
                   "structure_frac": {"min": 0.0, "max": 0.0}},
    "clustering": {"cluster_prob": 0.4, "n_clusters": {"min": 2, "max": 8},
                   "cluster_sigma_px": {"min": 6.0, "max": 24.0}},
}


def _consts():
    return DetectorConstants.from_config(DET_CFG)


# --------------------------------------------------------------------------- #
# CHANGE 1 -- per-image gain randomisation                                      #
# --------------------------------------------------------------------------- #
def test_detector_constants_from_config():
    c = _consts()
    assert c.gain1_range == (3.0, 30.0) and c.gain2_range == (20.0, 150.0)
    assert math.isclose(c.floor1, math.sqrt(3.1)) and math.isclose(c.floor2, math.sqrt(4.4))


def test_sample_image_detector_in_range_and_varies():
    c = _consts()
    rng = np.random.default_rng(0)
    g1s, g2s = [], []
    for _ in range(200):
        det, g1, g2 = sample_image_detector(rng, c)
        assert 3.0 <= g1 <= 30.0 and 20.0 <= g2 <= 150.0
        assert det.ch1.offset == 154.0 and det.ch2.noise_floor_sigma == pytest.approx(math.sqrt(4.4))
        g1s.append(g1); g2s.append(g2)
    # gains vary independently -> the RATIO varies (the model can't lean on it).
    ratios = np.array(g2s) / np.array(g1s)
    assert ratios.std() > 0.5


# --------------------------------------------------------------------------- #
# CHANGE 2 -- intensity range solved per image, keeps BOTH channels unclipped   #
# --------------------------------------------------------------------------- #
def test_solve_ceiling_keeps_both_channels_below_knee():
    """The brightest spot at the solved ceiling must clear the knee in BOTH channels."""
    c = _consts()
    rng = np.random.default_rng(1)
    knee = 3941.0
    for _ in range(500):
        g1 = float(rng.uniform(3, 30)); g2 = float(rng.uniform(20, 150))
        sigma1 = float(rng.uniform(1.0, 1.8)); sigma2 = sigma1 * float(rng.uniform(1.05, 1.35))
        intercept = float(rng.uniform(-0.7, 0.7)); slope = float(rng.uniform(-0.6, 0.6))
        scatter = float(rng.uniform(0.03, 0.25))
        sol = solve_a1_ceiling(
            gain1=g1, gain2=g2, sigma1=sigma1, sigma2=sigma2, sim_intercept=intercept,
            sim_log_slope=slope, scatter_std=scatter, background=2.0, knee1=knee, knee2=knee,
            target_frac=0.85, scatter_sigmas=3.5, floor_a1_photons=10.0,
        )
        if sol["degenerate"]:
            continue
        a1 = sol["a1_cap_photons"]
        a2 = math.exp(intercept) * a1 ** (1.0 + slope)   # median partner (no scatter)
        assert clean_gained_peak(a1, g1, sigma1, 2.0) <= knee
        # ch2 checked at the median partner; the solve reserves scatter headroom on top.
        assert clean_gained_peak(a2, g2, sigma2, 2.0) <= knee


def test_measured_detector_images_do_not_saturate():
    """Rendered training images should carry ~zero saturated spots (CHANGE 2 goal)."""
    c = _consts()
    exs = generate_examples(SCENE_CFG, c, IntensityWindowConfig(), n_images=40, seed=7,
                            shape=(96, 96), heatmap_sigma=1.5, t=1.0)
    total = sum(e.meta["n_spots"] for e in exs)
    sat = sum(e.meta["n_saturated"] for e in exs)
    assert total > 0
    assert sat / total < 0.005, f"saturated fraction {sat}/{total} too high"


def test_ceiling_solve_below_old_impossible_range():
    """At the measured protein gain the ceiling is far below the old A1 max (7943 ph)."""
    rep = summarize_solved_windows(SCENE_CFG, _consts(), IntensityWindowConfig(),
                                   shape=(256, 256), n_samples=1000, seed=0, t=1.0)
    assert rep["a1_cap_photons"]["median"] < 7943.0
    assert rep["degenerate_fraction"] < 0.05


# --------------------------------------------------------------------------- #
# Curriculum: intensity dim tail widens with t; scene difficulty ramps          #
# --------------------------------------------------------------------------- #
def test_curriculum_widens_intensity_window():
    c = _consts()
    r0 = summarize_solved_windows(SCENE_CFG, c, IntensityWindowConfig(), shape=(128, 128),
                                  n_samples=800, seed=0, t=0.0)
    r1 = summarize_solved_windows(SCENE_CFG, c, IntensityWindowConfig(), shape=(128, 128),
                                  n_samples=800, seed=0, t=1.0)
    assert r0["window_decades"]["median"] == pytest.approx(0.0, abs=1e-9)   # bright-only at t=0
    assert r1["window_decades"]["median"] > 0.8                            # full dim tail at t=1


def test_curriculum_scene_config_ramps_density_only():
    easy = curriculum_scene_config(SCENE_CFG, 0.0)
    hard = curriculum_scene_config(SCENE_CFG, 1.0)
    # density max grows toward full; slope/PSF untouched (never a learnable prior).
    assert easy["density"]["max"] < hard["density"]["max"]
    assert easy["ratio_law"]["beta"] == SCENE_CFG["ratio_law"]["beta"]
    assert easy["psf"] == SCENE_CFG["psf"]


# --------------------------------------------------------------------------- #
# Target maps                                                                   #
# --------------------------------------------------------------------------- #
def test_build_targets_shapes_and_centers():
    c = _consts()
    ex = generate_examples(SCENE_CFG, c, IntensityWindowConfig(), n_images=1, seed=3,
                           shape=(64, 64), heatmap_sigma=1.5, t=1.0)[0]
    t = ex.targets
    assert set(t.keys()) == set(TARGET_KEYS)
    assert t["heatmap"].shape == (1, 64, 64) and t["offset"].shape == (2, 64, 64)
    # one center per spot, minus any integer-center collisions (last-spot-wins).
    n_spots = ex.meta["n_spots"]
    assert 0 <= int(t["center_mask"].sum()) <= n_spots
    if n_spots:
        assert int(t["center_mask"].sum()) > 0
        assert math.isclose(float(t["heatmap"].max()), 1.0, abs_tol=1e-5)


def test_build_targets_empty_image():
    import pandas as pd
    from spotpipe.schema import SCHEMA_COLUMNS
    empty = pd.DataFrame(columns=SCHEMA_COLUMNS)
    t = build_targets(empty, (32, 32), 1.5)
    assert int(t["center_mask"].sum()) == 0 and float(t["heatmap"].sum()) == 0.0


# --------------------------------------------------------------------------- #
# CHANGE 5 -- multi-worker loader yields one batch per step, correct shapes      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stream_loader_yields_all_steps(num_workers):
    c = _consts()
    n_steps, bs = 12, 3
    loader = make_loader(SCENE_CFG, c, IntensityWindowConfig(), shape=(48, 48),
                         heatmap_sigma=1.5, batch_size=bs, n_steps=n_steps, adc_max=4095,
                         seed=0, ramp_steps=6, use_curriculum=True, num_workers=num_workers,
                         pin_memory=False)
    seen_steps = []
    for images, targets, step, t in loader:
        assert tuple(images.shape) == (bs, 2, 48, 48)
        assert 0.0 <= t <= 1.0
        seen_steps.append(int(step))
    assert sorted(seen_steps) == list(range(1, n_steps + 1))


def test_eval_set_has_hard_corner_images():
    c = _consts()
    exs = build_eval_examples(SCENE_CFG, c, IntensityWindowConfig(), n_images=16, seed=999,
                              shape=(96, 96), heatmap_sigma=1.5, n_hard_corner=5)
    kinds = [e.meta["eval_kind"] for e in exs]
    assert kinds.count("hard_corner") == 5
    assert "beta0" in kinds and "bright_sparse" in kinds


# --------------------------------------------------------------------------- #
# Overfit + smoke self-checks                                                   #
# --------------------------------------------------------------------------- #
def _smoke_config():
    return {
        "seed": 0,
        "model": {"in_channels": 2, "base_channels": 16, "num_branches": 2,
                  "blocks_per_branch": 1, "head_mid_channels": 16, "heatmap_bias": -2.19},
        "simulator": {"image": {"height": 64, "width": 64}, "detector": DET_CFG, "scene": SCENE_CFG},
        "training": {"train_steps": 8, "batch_size": 3, "lr": 0.002, "heatmap_sigma": 1.5,
                     "lr_warmup_steps": 2, "variance_warmup_steps": 3, "eval_every": 4,
                     "checkpoint_every": 4,
                     "best_checkpoint": {"enabled": True, "hard_corner_min_pairs": 2},
                     "curriculum": {"enabled": True, "curriculum_ramp_steps": 4},
                     "intensity_window": {"full_decades": 2.0},
                     "val": {"n_images": 6, "n_hard_corner": 2}},
        "benchmark": {"match_radius_px": 3.0},
    }


def test_overfit_collapses():
    from spotpipe.training.train import overfit
    res = overfit(_smoke_config(), n_images=4, steps=150, device="cpu")
    final = res["final_eval"]
    # target-map + loss + decode are correct iff a tiny set can be memorised.
    assert final["logI1_mae"] < 0.3 and final["logI2_mae"] < 0.3
    assert res["loss_curve"][-1][1] < res["loss_curve"][0][1]


def test_smoke_train_runs_and_writes(tmp_path):
    from spotpipe.training.train import train
    res = train(_smoke_config(), device="cpu", out_dir=tmp_path, num_workers=0, resume=False)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "best_checkpoint.pt").exists()
    import json
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["model_label"].startswith("measured_detector")
    assert m["detector_constants"]["gain1_range"] == [3.0, 30.0]
