"""The benchmark density ramp must stay INSIDE the training density range.

This is the guard for the 2026-07-14 change that extended the family-1 density ramp
to 0.02 spots/px (it used to taper off at 0.015) and raised the training max from
0.012 to 0.024 to cover it.

The failure mode this prevents is silent and expensive: `configs/train.yaml` sets the
distribution the model actually SEES, while `configs/benchmark.yaml` sets the densities
it is MEASURED at, and `generate._TRAINED_DENSITY_MAX` is the constant that decides
which cells get flagged out-of-distribution in the manifest. If those three drift apart,
the benchmark quietly measures the model outside its training range and the degradation
gets read as a method difference instead of a coverage artifact -- which is exactly the
mistake PROJECT_STATE Sec.4 warns about. Any future change to the density ramp must move
all three together, or fail here.
"""

import yaml

from spotpipe.benchmark.generate import (
    _LEGACY_DENSITY_MAX,
    _TRAINED_DENSITY_MAX,
    load_benchmark_config,
)
from spotpipe.paths import get_paths


def _train_density_max(config_name: str = "train.yaml") -> float:
    with open(get_paths().configs / config_name) as fh:
        cfg = yaml.safe_load(fh)
    return float(cfg["simulator"]["scene"]["density"]["max"])


def test_trained_density_max_constant_matches_train_config():
    # The generator hardcodes the training max to flag stress cells; it must equal
    # what train.yaml actually trains on.
    assert _TRAINED_DENSITY_MAX == _train_density_max()


def test_benchmark_density_levels_inside_training_range():
    _, cfg = load_benchmark_config(get_paths().configs / "benchmark.yaml")
    train_max = _train_density_max()
    assert max(cfg.density_levels) <= train_max, (
        f"benchmark density levels {cfg.density_levels} exceed the training max "
        f"{train_max} -- the model would be measured out-of-distribution")


def test_top_density_level_has_headroom_below_training_max():
    # The top cell must be INSIDE the trained range, not ON its boundary: models are
    # weakest at the edge of their distribution, so a benchmark point sitting exactly
    # at the ceiling measures edge degradation, not crowding performance.
    _, cfg = load_benchmark_config(get_paths().configs / "benchmark.yaml")
    assert max(cfg.density_levels) < _train_density_max()


def test_density_ramp_reaches_the_crowded_regime():
    # The point of the 2026-07-14 change: the ramp must not taper off early.
    _, cfg = load_benchmark_config(get_paths().configs / "benchmark.yaml")
    assert max(cfg.density_levels) >= 0.02


def test_smoke_benchmark_ramp_matches_full_benchmark():
    # The smoke config is the only thing the tests actually generate, so it has to
    # exercise the same density axis as the real benchmark or the guard is theatre.
    _, full = load_benchmark_config(get_paths().configs / "benchmark.yaml")
    _, smoke = load_benchmark_config(get_paths().configs / "benchmark_smoke.yaml")
    assert tuple(smoke.density_levels) == tuple(full.density_levels)


def test_legacy_max_is_still_recorded_and_below_current():
    # Legacy checkpoints saw <=0.012 and are genuinely OOD at the top two levels;
    # that distinction must survive, not get overwritten by the new max.
    assert _LEGACY_DENSITY_MAX < _TRAINED_DENSITY_MAX
