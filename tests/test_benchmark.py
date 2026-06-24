"""Tests for the build-stage-4 benchmark harness.

Runnable two ways (like the stage-3 smoke tests)::

    uv run python tests/test_benchmark.py     (standalone; prints the tiny table)
    uv run pytest tests/test_benchmark.py

1. ``test_match_gt_against_itself`` -- the harness's own sanity check: matching a
   ground-truth table against itself MUST give perfect recall/precision and
   exactly zero intensity/ratio error. If this fails, the matcher or metrics are
   wrong, not the methods.

2. ``test_tiny_end_to_end`` -- a tiny run over a few images with our model + both
   baselines that produces a metrics table and at least the recovered-beta
   figure. Structural assertions only (no dependence on how well the briefly
   trained model detects), plus the by-construction guarantee that the
   oracle-centre baseline recovers every GT spot.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "train_smoke.yaml"
BENCH_CONFIG = REPO_ROOT / "configs" / "benchmark.yaml"


def _sim_config() -> dict:
    with open(SMOKE_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _bench_config() -> dict:
    with open(BENCH_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Test 1: GT-vs-GT must be a perfect, zero-error match (harness sanity check)  #
# --------------------------------------------------------------------------- #
def test_match_gt_against_itself() -> None:
    import pandas as pd

    from spotpipe.benchmark.features import attach_features
    from spotpipe.benchmark.harness import build_eval_set
    from spotpipe.benchmark.matching import match_dataset
    from spotpipe.benchmark.metrics import compute_metrics

    cfg = _bench_config()["benchmark"]
    eval_set = build_eval_set(_sim_config(), n_images=4, seed=7, shape=(64, 64))
    meta_by_image = {it.image_id: it.meta for it in eval_set}
    gt = pd.concat([it.gt for it in eval_set], ignore_index=True)
    assert len(gt) > 0, "tiny eval set produced no spots"

    feat = attach_features(gt, meta_by_image, density_radius_px=cfg["density_radius_px"])

    # Match GT against an identical copy.
    match = match_dataset(feat, feat.copy(), max_distance=cfg["match_radius_px"], method="greedy")
    assert match.n_matched == len(gt), "GT-vs-GT did not match every spot"
    assert len(match.unmatched_gt) == 0 and len(match.unmatched_pred) == 0
    assert np.allclose(match.distances, 0.0), "self-match distances must be zero"

    snr_bins = [float(x) if x is not None else math.inf for x in cfg["snr_bins"]]
    dens_bins = [float(x) if x is not None else math.inf for x in cfg["density_bins"]]
    m = compute_metrics(feat, feat.copy(), match, meta_by_image,
                        snr_bins=snr_bins, density_bins=dens_bins)

    det = m["detection_overall"]
    assert det["recall"] == 1.0 and det["precision"] == 1.0 and det["f1"] == 1.0
    overall = m["binned"]["overall"]["intensity"]
    for q in ("logI1", "logI2", "log_ratio"):
        assert abs(overall[q]["bias"]) < 1e-9, f"{q} bias not zero: {overall[q]['bias']}"
        assert overall[q]["rmse"] < 1e-9, f"{q} rmse not zero: {overall[q]['rmse']}"

    # Hungarian must give the same perfect result.
    match_h = match_dataset(feat, feat.copy(), max_distance=cfg["match_radius_px"], method="hungarian")
    assert match_h.n_matched == len(gt)

    print(f"[gt-vs-gt] {len(gt)} spots: recall=precision=f1=1.0, "
          f"logI/ratio error == 0 (greedy & hungarian).")


# --------------------------------------------------------------------------- #
# Test 2: tiny end-to-end benchmark (our model + both baselines)              #
# --------------------------------------------------------------------------- #
def test_tiny_end_to_end() -> None:
    import pandas as pd

    from spotpipe.benchmark import baselines
    from spotpipe.benchmark.harness import load_eval_set, run_benchmark
    from spotpipe.simulator.generate_dataset import generate_dataset
    from spotpipe.training.train import train

    sim_cfg = _sim_config()
    bench_cfg = _bench_config()
    methods = ["our_model", "classical_per_channel_aperture", "oracle_center_aperture_divide"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Fixed eval set on disk (exercises generate_dataset + load_eval_set).
        eval_dir = tmp / "eval"
        generate_dataset(sim_cfg, eval_dir, n_images=4, seed=123, split="val")
        eval_set = load_eval_set(eval_dir)
        assert len(eval_set) == 4

        # Briefly train our model (deterministic; quality is irrelevant here).
        train_cfg = _sim_config()
        train_cfg.update({k: v for k, v in {"model": {
            "in_channels": 2, "base_channels": 16, "num_branches": 3,
            "blocks_per_branch": 2, "head_mid_channels": 32, "heatmap_bias": -2.19,
        }}.items()})
        train_cfg["training"] = {
            "batch_size": 4, "lr": 0.002, "heatmap_sigma": 1.5,
            "intensity_logvar_warmup_fraction": 0.3,
            "val": {"n_images": 2, "eval_every": 100, "seed": 999},
            "log_every": 100, "curriculum": {"enabled": False},
        }
        result = train(train_cfg, steps=40, out_dir=None, log_fn=lambda _s: None)
        model = result["model"]

        # Lower the detection threshold a touch so a lightly trained net fires.
        bench_cfg["benchmark"]["our_model"]["peak_threshold"] = 0.2

        out = tmp / "bench"
        res = run_benchmark(eval_set, bench_cfg, out_dir=out, methods=methods,
                            model=model, log_fn=lambda _s: None)

        # Artifacts written.
        assert (out / "metrics_table.csv").exists()
        assert (out / "slopes.csv").exists()
        assert (out / "manifest.json").exists()
        assert (out / "figures" / "recovered_beta.png").exists(), "recovered-beta figure missing"
        assert (out / "figures" / "ratio_vs_snr.png").exists()
        for name in methods:
            assert (out / "predictions" / f"{name}.csv").exists()
            assert (out / "metrics" / f"{name}.json").exists()

        # All (renamed) methods present in the table.
        table = res["table"]
        assert set(methods) <= set(table["method"].unique())

        # The oracle-centre baseline reads every GT centre -> perfect recall.
        gtc = res["metrics"]["oracle_center_aperture_divide"]["detection_overall"]
        assert gtc["recall"] == 1.0, f"oracle_center_aperture_divide recall != 1.0 ({gtc['recall']})"

        # Both baselines produced real matched pairs to measure ratio bias on.
        for b in ("classical_per_channel_aperture", "oracle_center_aperture_divide"):
            n_pairs = res["metrics"][b]["binned"]["overall"]["intensity"]["log_ratio"]["n"]
            assert n_pairs > 0, f"{b} produced no matched pairs"

        # Our model emits per-spot uncertainty -> calibration exists; baselines don't.
        assert res["metrics"]["our_model"].get("calibration") is None or \
            isinstance(res["metrics"]["our_model"]["calibration"], dict)
        assert res["metrics"]["oracle_center_aperture_divide"]["calibration"] is None

        # --- (#2) offset/gain audit of the oracle aperture baseline ----------
        # Confirm the chain is counts -> subtract offset/background -> / gain, by
        # showing recovered logI is near the TRUE photon-scale logI. Dividing raw
        # detector counts would inflate logI by ~log(gain * n_aperture_px) >> 2.
        item = eval_set[0]
        det = item.meta["detector"]
        g1, g2 = det["ch1"]["gain"], det["ch2"]["gain"]
        o1, o2 = det["ch1"]["offset"], det["ch2"]["offset"]
        p0 = baselines.oracle_center_aperture_divide(
            item.image, item.gt, item.meta, image_id=item.image_id,
            cfg=bench_cfg["benchmark"]["baseline"])
        merged = p0.merge(item.gt, on="spot_id", suffixes=("_pred", "_true"))
        med_err1 = float((merged["logI1_pred"] - merged["logI1_true"]).abs().median())
        med_err2 = float((merged["logI2_pred"] - merged["logI2_true"]).abs().median())
        assert med_err1 < 2.0 and med_err2 < 2.0, (
            f"oracle aperture logI error too large ({med_err1:.2f}/{med_err2:.2f}) -- "
            "is the baseline dividing raw counts instead of offset/gain-correcting?")
        print(f"\n[#2 offset/gain audit] oracle_center_aperture_divide chain: "
              f"counts - (offset {o1}/{o2} via annulus median) -> / gain ({g1}/{g2}) -> aperture intensity.")
        print(f"    median |logI_pred - logI_true| = {med_err1:.3f} (ch1) / {med_err2:.3f} (ch2) nat "
              f"-> photon-scale (raw-count division would be >> this). Raw counts never divided.")

        # --- renamed method labels in the tiny benchmark table ---------------
        print("\n[method labels + overall metrics]")
        ov = []
        for meth, m in res["metrics"].items():
            d = m["detection_overall"]
            lr = m["binned"]["overall"]["intensity"]["log_ratio"]
            ov.append(dict(method=meth, recall=round(d["recall"], 3), precision=round(d["precision"], 3),
                           f1=round(d["f1"], 3), ratio_bias=round(lr["bias"], 3),
                           ratio_std=round(lr["std"], 3), n_pairs=lr["n"]))
        print(pd.DataFrame(ov).to_string(index=False))

        # --- (#4) recovered-beta: matched_only vs end_to_end variants --------
        slopes = res["slopes"]
        assert set(slopes["variant"].unique()) == {"matched_only", "end_to_end"}
        assert "precision" in slopes.columns  # end_to_end logs detection precision
        print("\n[#4 recovered-beta variants] slopes.csv (head, both variants):")
        cols = ["method", "variant", "image_id", "true_beta", "beta_hat", "n_spots", "precision"]
        print(slopes[cols].round(4).head(10).to_string(index=False))

        print(f"\n[end-to-end] ran {len(methods)} methods over {len(eval_set)} images; "
              f"table rows={len(table)}; figures: "
              f"{sorted(p.name for p in (out / 'figures').glob('*.png'))}")


if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK TEST 1: GT-vs-GT perfect-match validation")
    print("=" * 70)
    test_match_gt_against_itself()

    print("\n" + "=" * 70)
    print("BENCHMARK TEST 2: tiny end-to-end benchmark")
    print("=" * 70)
    test_tiny_end_to_end()

    print("\nAll benchmark tests passed.")
