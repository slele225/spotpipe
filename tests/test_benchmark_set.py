"""Tests for the FROZEN benchmark/test-set generator (stage 4.5).

Runnable two ways (like the other smoke tests)::

    uv run python tests/test_benchmark_set.py    (standalone; prints the report)
    uv run pytest tests/test_benchmark_set.py

A tiny, fast generation (small images, low per-cell target) that exercises the
real :func:`spotpipe.simulator.benchmark_set.generate_benchmark_set` end to end
and asserts the freeze invariants the prompt requires:

* every SNR x density cell reaches the (small) target -> nothing underpopulated;
* all beta groups and the four channel-imbalance stress classes are non-empty;
* GT round-trips through ``spotpipe.schema`` with the ``ground_truth`` flag;
* raw per-channel TIFFs are uint16 in ``[0, adc_max]``; photon TIFFs are float and
  satisfy ``photon == (raw - offset) / gain`` exactly;
* the manifest carries the frozen edges + checksums, and ``verify_benchmark_set``
  passes on the freshly written set (and fails if a byte is flipped).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import tifffile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "simulator.yaml"
BENCH_SET_CONFIG = REPO_ROOT / "configs" / "benchmark_test_set.yaml"


def _configs() -> tuple[dict, dict]:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    bench = yaml.safe_load(BENCH_SET_CONFIG.read_text(encoding="utf-8"))["benchmark_set"]
    # Shrink to a fast smoke size; keep the stratification machinery identical.
    bench.update({
        "image": {"height": 96, "width": 96},
        "min_gt_per_cell": 20,
        "min_images_per_beta_group": 2,
        "min_images_per_stress_flag": 2,
        "base_images": 8,
        "max_attempts": 400,
    })
    return base, bench


def test_generate_and_freeze() -> None:
    from spotpipe.schema import dataframe_to_records, read_spots, records_to_dataframe
    from spotpipe.simulator.benchmark_set import (
        generate_benchmark_set,
        stratification_report,
        verify_benchmark_set,
    )

    base, bench = _configs()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "benchmark_test_v1"
        manifest = generate_benchmark_set(base, bench, out, log_fn=lambda _s: None)

        # --- coverage: nothing underpopulated; cells, betas, stress non-empty ---
        assert manifest["underpopulated_cells"] == [], "some SNR x density cell underpopulated"
        assert manifest["n_gt_spots"] == sum(c["count"] for c in manifest["counts_snr_x_density"]) \
            or manifest["n_gt_spots"] >= sum(c["count"] for c in manifest["counts_snr_x_density"])
        for b, n in manifest["counts_per_beta_group"].items():
            assert n > 0, f"beta group {b} empty"
        ci = manifest["counts_per_stress_flag"]["channel_imbalance"]
        for cls in ("ch1_bright_ch2_dim", "ch1_dim_ch2_bright", "both_dim", "both_bright"):
            assert ci.get(cls, 0) > 0, f"channel-imbalance class {cls} empty"
        sat = manifest["counts_per_stress_flag"]["saturation"]
        assert sum(sat.values()) == manifest["n_images"]

        # --- frozen binning is recorded verbatim -----------------------------
        assert manifest["snr_bin_edges"] == [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, "inf"]
        assert manifest["density_bin_edges"] == [0.0, 2.0, 5.0, 10.0, "inf"]
        assert manifest["beta_groups"] == [-0.6, -0.3, 0.0, 0.3, 0.6]

        # --- GT schema round-trip --------------------------------------------
        recs = read_spots(out / "ground_truth.csv")
        recs2 = dataframe_to_records(records_to_dataframe(recs))
        assert len(recs) == len(recs2) == manifest["n_gt_spots"]
        assert all("ground_truth" in r.flags for r in recs[:50])

        # --- raw export: uint16 in [0, adc_max] ------------------------------
        adc = manifest["detector"]["adc_max"]
        img0 = manifest["images"][0]["image_id"]
        raw1 = tifffile.imread(out / "images_ch1_raw" / f"{img0}.tif")
        assert raw1.dtype == np.uint16 and raw1.min() >= 0 and raw1.max() <= adc

        # --- photon export: float, exactly (raw - offset)/gain ---------------
        conv = manifest["raw_vs_photon_convention"]
        off1, g1 = conv["offsets"]["ch1"], conv["gains"]["ch1"]
        pho1 = tifffile.imread(out / "images_ch1_photon" / f"{img0}.tif")
        assert np.issubdtype(pho1.dtype, np.floating)
        assert np.abs(pho1 - (raw1.astype(np.float32) - np.float32(off1)) / np.float32(g1)).max() == 0.0

        # --- canonical two-channel npy ---------------------------------------
        canon = np.load(out / "images" / f"{img0}.npy")
        assert canon.shape[0] == 2 and canon.dtype == np.uint16

        # --- freeze: verify passes, and flipping a byte makes it fail ---------
        ok, problems = verify_benchmark_set(out)
        assert ok, f"fresh set failed verification: {problems}"
        gt_path = out / "ground_truth.csv"
        gt_path.write_bytes(gt_path.read_bytes() + b"\n# tampered\n")
        ok2, problems2 = verify_benchmark_set(out)
        assert not ok2 and any("ground_truth.csv" in p for p in problems2), \
            "checksum verification did not catch a tampered ground_truth.csv"

        print(stratification_report(manifest))
        print(f"\n[freeze] verify OK on fresh set; tamper detected -> {problems2}")


if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK-SET TEST: generate, stratify, export, freeze")
    print("=" * 70)
    test_generate_and_freeze()
    print("\nbenchmark-set test passed.")
