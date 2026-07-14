"""Sanity tests for the two-family benchmark generator (generation only).

Uses the tiny smoke profile so the whole module generates in a couple of seconds.
Covers the checklist from the build prompt:

* a snr_density cell reloads; GT is schema-valid; n_images correct;
* a curvature set's recorded true_alpha == 2 * sim_log_slope (convention wiring);
* a curvature set's A1 spread clears the minimum threshold;
* the alpha=0 set exists, is flagged null control, and has the larger image count;
* determinism: regenerating one small set with the same seed yields identical GT.
"""

import json
import math

import pandas as pd
import pytest

from spotpipe.benchmark.alpha import alpha_to_sim_slope, sim_slope_to_alpha
from spotpipe.benchmark.generate import (
    BenchmarkConfig,
    _edge_label,
    generate_benchmark,
    load_benchmark_config,
)
from spotpipe.paths import get_paths
from spotpipe.schema import SCHEMA_COLUMNS, read_spots


@pytest.fixture(scope="module")
def smoke_bench(tmp_path_factory):
    base_config, cfg = load_benchmark_config(get_paths().configs / "benchmark_smoke.yaml")
    out = tmp_path_factory.mktemp("bench")
    manifest = generate_benchmark(base_config, cfg, out, log_fn=lambda *_: None)
    return out, manifest, cfg


# --------------------------------------------------------------------------- #
# Convention wiring (pure, no generation)                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [-1.2, -0.3, 0.0, 0.075, 0.6, 1.2])
def test_alpha_convention_roundtrip(alpha):
    # The factor of 2 lives in exactly one place; slope->alpha and back are exact.
    assert sim_slope_to_alpha(alpha_to_sim_slope(alpha)) == pytest.approx(alpha)
    assert sim_slope_to_alpha(0.5) == pytest.approx(1.0)  # 2 * 0.5


# --------------------------------------------------------------------------- #
# Family 1: a cell reloads, GT is schema-valid, n_images correct              #
# --------------------------------------------------------------------------- #
def test_snr_density_cell_reloads_and_is_schema_valid(smoke_bench):
    out, _manifest, cfg = smoke_bench
    # Pick the cell FROM THE CONFIG rather than hardcoding a label. The grid is a measured,
    # revisable thing (v2 -> v3 moved SNR from [2..15] down to [0.75..3.0]); a hardcoded
    # "snr=5_density=0.006" turns any legitimate grid change into a spurious test failure,
    # which trains people to edit the test until it passes. Mid-grid keeps it representative.
    snr = cfg.snr_targets[len(cfg.snr_targets) // 2]
    dens = cfg.density_levels[len(cfg.density_levels) // 2]
    cell = out / "snr_density" / f"snr={_edge_label(snr)}_density={_edge_label(dens)}"
    meta = json.loads((cell / "meta.json").read_text())

    assert meta["family"] == "snr_density"
    assert meta["n_images"] == cfg.images_per_cell
    assert meta["true_alpha"] == 0.0  # neutral zero-scatter ratio law
    # TRUE constant-SNR: target recorded, intensity solved, ch1 == ch2.
    assert meta["target_snr"] == pytest.approx(snr)
    assert meta["solved_intensity_photons"] > 0
    assert meta["solved_A1_photons"] == meta["solved_A2_photons"]
    # density axis is a constant area density in spots/px, used as the label.
    assert meta["area_density_spots_per_px"] == pytest.approx(dens)
    assert meta["condition"]["area_density_spots_per_px"] == pytest.approx(dens)

    gt_files = sorted((cell / "ground_truth").glob("gt_*.csv"))
    img_files = sorted((cell / "images").glob("image_*.tif"))
    assert len(gt_files) == cfg.images_per_cell
    assert len(img_files) == cfg.images_per_cell
    # every image record carries per-image, per-channel ground_truth_sigma
    for rec in meta["images"]:
        assert set(rec["ground_truth_sigma"]) == {"sigma1", "sigma2"}
        assert rec["ground_truth_sigma"]["sigma1"] > 0
        assert rec["ground_truth_sigma"]["sigma2"] > 0

    for gt in gt_files:
        df = pd.read_csv(gt)
        assert list(df.columns) == list(SCHEMA_COLUMNS)
        read_spots(gt)  # parses into records (frozen-schema roundtrip)


def test_snr_density_grid_is_complete(smoke_bench):
    out, manifest, cfg = smoke_bench
    n_snr = len(cfg.snr_targets)
    n_cells = n_snr * len(cfg.density_levels)
    assert manifest["families"]["snr_density"]["n_cells"] == n_cells
    assert len(list((out / "snr_density").glob("snr=*_density=*"))) == n_cells


def test_snr_density_grid_is_orthogonal(smoke_bench):
    # Every density level appears at every SNR level (full SNR x density grid).
    out, _manifest, cfg = smoke_bench
    n_snr = len(cfg.snr_targets)
    got = {(c["snr_index"], c["area_density_spots_per_px"])
           for c in _manifest_cells(out)}
    expected = {(si, float(d)) for si in range(n_snr) for d in cfg.density_levels}
    assert got == expected


def test_snr_density_is_true_constant_snr(smoke_bench):
    # Change 4: within a cell every spot has identical SNR (~zero spread) and the
    # realised SNR equals the target. Intensity is solved by inversion, no jitter.
    out, _manifest, _cfg = smoke_bench
    for meta_path in (out / "snr_density").glob("snr=*_density=*/meta.json"):
        meta = json.loads(meta_path.read_text())
        target = meta["target_snr"]
        assert meta["realised_snr_spread"] == pytest.approx(0.0, abs=1e-6)
        rs = meta["realised_snr"]
        if rs["n"]:
            assert rs["min"] == pytest.approx(target, rel=1e-6, abs=1e-6)
            assert rs["max"] == pytest.approx(target, rel=1e-6, abs=1e-6)
        # ch2 (protein) is the limiting channel at this fixed PSF / background.
        assert meta["limiting_channel"] in (1, 2)


def test_snr_density_solved_intensity_table_and_no_clip(smoke_bench):
    # The manifest carries the solved-intensity-per-SNR-cell table and EVERY cell must be
    # UNCLIPPED on both channels (positive ADC headroom). That invariant is permanent.
    #
    # What is NOT permanent: whether a cell sits inside the LEGACY checkpoints' [20, 7943]
    # photon range. The v3 grid deliberately reaches DOWN to 12.7 photons (SNR 0.75) --
    # below the legacy floor -- because that is where detection actually discriminates
    # (docs/benchmark_grid_requirements.md). Those cells are correctly FLAGGED as
    # out-of-range for the LEGACY checkpoints. The measured-detector model trains down to
    # 3 photons and covers them (docs/coverage_probe_findings.md), so the flag is a
    # statement about the legacy checkpoints, not about the grid being wrong.
    #
    # So: assert the flag fires EXACTLY when the cell leaves the legacy range -- never
    # assert it is always absent.
    out, manifest, cfg = smoke_bench
    table = manifest["solved_intensity_table"]
    assert {row["target_snr"] for row in table} == {float(s) for s in cfg.snr_targets}
    for row in table:
        A = row["solved_intensity_photons"]
        assert row["ch1_saturates"] is False and row["ch2_saturates"] is False
        assert row["ch1_headroom_adu"] > 0 and row["ch2_headroom_adu"] > 0

        in_legacy = 20.0 <= A <= 7943.0
        assert row["in_legacy_training_distribution"] is in_legacy
        # flag <=> outside the legacy range. Both directions, so neither a missing flag nor
        # a spurious one can slip through.
        assert (row["flag"] is None) is in_legacy


def test_snr_density_clipping_target_fails_loud():
    # Decision 1: a target that would clip a channel must FAIL generation, not ship.
    base_config, _ = load_benchmark_config(get_paths().configs / "benchmark_smoke.yaml")
    cfg = BenchmarkConfig(seed=1, height=32, width=32, images_per_cell=1,
                          snr_targets=(50.0,), density_levels=(0.001,),
                          alpha_values=(0.0,), images_per_alpha=1, null_control_multiplier=1)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="CLIPS a channel"):
            generate_benchmark(base_config, cfg, td, log_fn=lambda *_: None)


def _manifest_cells(out):
    return json.loads((out / "BENCH_MANIFEST.json").read_text())["snr_density_cells"]


def test_snr_density_is_constant_area_density_no_clustering(smoke_bench):
    # Change 1: each cell has one constant area density across all its images, spots
    # are placed uniformly (no clustering), and the label matches the density.
    out, _manifest, _cfg = smoke_bench
    for meta_path in (out / "snr_density").glob("snr=*_density=*/meta.json"):
        meta = json.loads(meta_path.read_text())
        target = meta["area_density_spots_per_px"]
        assert meta["area_density_constant_per_cell"] is True
        # label encodes the density in spots/px
        assert f"density={target:g}" in meta["label"]
        # every image realises the identical area density == the target
        dens = {rec["area_density"] for rec in meta["images"]}
        assert len(dens) == 1
        assert dens.pop() == pytest.approx(target)
        # no clustering anywhere: uniform placement is forced off
        assert meta["scene_config"]["clustering"]["cluster_prob"] == 0.0
        assert meta["scene_config"]["oversample_dense_fraction"] == 0.0


# --------------------------------------------------------------------------- #
# Family 2: convention wiring recorded, A1 spread, null control               #
# --------------------------------------------------------------------------- #
def test_curvature_true_alpha_is_twice_sim_log_slope(smoke_bench):
    out, _manifest, _cfg = smoke_bench
    for meta_path in (out / "curvature").glob("alpha=*/meta.json"):
        meta = json.loads(meta_path.read_text())
        assert meta["true_alpha"] == pytest.approx(2.0 * meta["sim_log_slope"])
        assert meta["sim_intercept"] == 0.0


def test_curvature_a1_spread_exceeds_threshold(smoke_bench):
    out, _manifest, cfg = smoke_bench
    for meta_path in (out / "curvature").glob("alpha=*/meta.json"):
        meta = json.loads(meta_path.read_text())
        assert meta["a1_spread_decades"] >= cfg.min_alpha_decades
        assert meta["a1_spread_ok"] is True


def test_curvature_no_ch2_saturation(smoke_bench):
    # Change 2: per-set A1-window sizing keeps ch2 below the knee -> ~0 saturation
    # on every curvature set (incl. the steep +alpha set that used to run ~0.49).
    out, _manifest, _cfg = smoke_bench
    for meta_path in (out / "curvature").glob("alpha=*/meta.json"):
        meta = json.loads(meta_path.read_text())
        assert meta["ch2_saturated_fraction"] == pytest.approx(0.0, abs=0.01), meta["label"]
        # the window is recorded and clears the spread threshold by construction
        assert meta["intensity_window"]["decades"] >= _cfg.min_alpha_decades


def test_null_control_exists_flagged_and_larger(smoke_bench):
    out, _manifest, cfg = smoke_bench
    null_dir = out / "curvature" / "alpha=0"
    assert null_dir.exists()
    meta = json.loads((null_dir / "meta.json").read_text())
    assert meta["true_alpha"] == 0.0
    assert meta["null_control"] is True

    # strictly more images than a non-null set
    other = json.loads((out / "curvature" / "alpha=0.6" / "meta.json").read_text())
    assert other["null_control"] is False
    assert meta["n_images"] == other["n_images"] * cfg.null_control_multiplier
    assert meta["n_images"] > other["n_images"]


# --------------------------------------------------------------------------- #
# Determinism: same seed -> identical GT                                       #
# --------------------------------------------------------------------------- #
def test_regeneration_is_deterministic(tmp_path):
    # A minimal 1-cell / 1-alpha config, generated twice, must be byte-identical GT.
    base_config, _ = load_benchmark_config(get_paths().configs / "benchmark_smoke.yaml")
    cfg = BenchmarkConfig(
        seed=7, height=32, width=32,
        images_per_cell=1, alpha_values=(0.6,), images_per_alpha=1,
        null_control_multiplier=1,
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_benchmark(base_config, cfg, a, log_fn=lambda *_: None)
    generate_benchmark(base_config, cfg, b, log_fn=lambda *_: None)

    for sub in ("snr_density/snr=2_density=0.0006/ground_truth",
                "curvature/alpha=0.6/ground_truth"):
        fa = sorted((a / sub).glob("gt_*.csv"))
        fb = sorted((b / sub).glob("gt_*.csv"))
        assert fa and len(fa) == len(fb)
        for pa, pb in zip(fa, fb):
            assert pa.read_text() == pb.read_text()
