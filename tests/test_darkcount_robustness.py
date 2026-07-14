"""Sanity for the protein-channel dark-count robustness check (Change 8).

Tiny and self-contained: builds a small spike field with the MEASURED detector,
runs a tiny random-weight model detector (no checkpoint needed for the wiring
test), and asserts the probe reports coherent counts. The single-pixel spike
model and its ch2-only placement are what we verify here; the actual legacy
checkpoint rate is reported by scripts/darkcount_robustness.py, not asserted.
"""

import numpy as np

from spotpipe.benchmark.darkcount_check import (
    DarkCountConfig,
    make_darkcount_image,
    run_darkcount_check,
)
from spotpipe.benchmark.generate import load_benchmark_config
from spotpipe.benchmark.infer import InferenceParams
from spotpipe.models import build_spot_model
from spotpipe.paths import get_paths
from spotpipe.simulator import noise


def _measured_detector():
    base_config, _ = load_benchmark_config(get_paths().configs / "benchmark_smoke.yaml")
    return noise.sample_detector_params(base_config.get("detector", {}), np.random.default_rng(0))


def test_spikes_are_ch2_only_and_single_pixel():
    det = _measured_detector()
    cfg = DarkCountConfig(height=64, width=64, n_images=1, rate=0.02)
    image, spike_xy = make_darkcount_image((64, 64), det, cfg, np.random.default_rng(1))
    assert image.shape == (2, 64, 64)
    assert image.dtype == np.uint16
    assert spike_xy.ndim == 2 and spike_xy.shape[1] == 2
    assert len(spike_xy) > 0  # rate 0.02 over 4096 px -> ~80 spikes

    # ch1 (lipid) carries NO spikes: its pixels are just the empty field, so its
    # spread is tiny next to ch2 (which has one-gain-step spikes ~124 ADU up).
    ch1, ch2 = image[0].astype(float), image[1].astype(float)
    assert ch2.max() - ch2.min() > ch1.max() - ch1.min()
    # every spike pixel in ch2 sits at least ~half a gain step above the pedestal.
    xs = spike_xy[:, 0].astype(int)
    ys = spike_xy[:, 1].astype(int)
    assert np.all(ch2[ys, xs] > det.ch2.offset + 0.5 * det.ch2.gain)


def test_run_darkcount_check_reports_coherent_counts():
    det = _measured_detector()
    # A tiny random-weight model just to exercise the detector wiring end-to-end.
    model = build_spot_model({"in_channels": 2, "base_channels": 8, "num_branches": 2,
                              "blocks_per_branch": 1, "head_mid_channels": 8})
    params = InferenceParams()
    cfg = DarkCountConfig(height=64, width=64, n_images=2, rate=0.0057)
    stats = run_darkcount_check(model, params, det, cfg, seed=3, log_fn=lambda *_: None)

    assert stats["n_images"] == 2
    assert stats["n_spikes_total"] > 0
    # spikes-detected can never exceed spikes injected, and the rate is in [0, 1].
    assert 0 <= stats["n_spikes_detected_as_spots"] <= stats["n_spikes_total"]
    assert 0.0 <= stats["spike_detection_rate"] <= 1.0
    assert stats["n_detections_matched_to_spike"] <= stats["n_detections_total"]
