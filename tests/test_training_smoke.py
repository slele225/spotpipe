"""Smoke tests for the build-stage-3 model / losses / training stack.

Two tests, both runnable via ``uv run``:

  * ``uv run python tests/test_training_smoke.py``   (standalone; no pytest needed)
  * ``uv run pytest tests/test_training_smoke.py``    (if pytest is installed)

1. ``test_end_to_end_loop_and_schema`` -- trains a handful of steps on a few
   on-the-fly images and proves the whole loop runs and the inference path emits
   a valid, round-trippable canonical ``spotpipe.schema`` file.

2. ``test_overfit_tiny_set`` (the critical one) -- overfits a tiny FIXED set for
   ~200 steps and asserts the loss collapses and the predicted logI1/logI2 at GT
   centres closely match the truth. If the loss or the target-map masking is
   broken, this is the fastest way to catch it: the intensity MAE would stay
   large instead of going to ~0.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "train_smoke.yaml"


def _load_config() -> dict:
    from spotpipe.training.train import load_train_config

    return load_train_config(CONFIG)


# --------------------------------------------------------------------------- #
# Test 1: end-to-end loop + canonical-schema inference                         #
# --------------------------------------------------------------------------- #
def test_end_to_end_loop_and_schema() -> None:
    from spotpipe.schema import SCHEMA_COLUMNS, read_spots, records_to_dataframe, write_spots
    from spotpipe.training.dataset import build_fixed_val_examples
    from spotpipe.training.train import predict_dataset, resolve_blocks, train

    cfg = _load_config()
    cfg["training"]["batch_size"] = 2
    cfg["training"]["val"]["n_images"] = 2

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        result = train(cfg, steps=4, out_dir=run_dir, log_fn=lambda _s: None)

        # The loop ran and produced finite losses + the expected run artifacts.
        assert result["history"], "no training history recorded"
        for row in result["history"]:
            assert math.isfinite(row["train_total"]), f"non-finite train loss: {row}"
        for name in ("checkpoint.pt", "config.yaml", "manifest.json", "metrics.jsonl"):
            assert (Path(result["run_dir"]) / name).exists(), f"missing run artifact: {name}"

        # Inference path -> canonical schema, written and read back round-trip.
        shape, _det_cfg, scene_cfg, adc = resolve_blocks(cfg)
        val = build_fixed_val_examples(
            scene_cfg, result["detector"], n_images=2, seed=999,
            shape=shape, heatmap_sigma=cfg["training"]["heatmap_sigma"],
        )
        df = predict_dataset(result["model"], val, adc_max=adc, peak_threshold=0.0, max_spots=15)
        assert list(df.columns) == list(SCHEMA_COLUMNS), "prediction columns are not the schema"

        csv_path = Path(result["run_dir"]) / "predictions.csv"
        write_spots(df, csv_path)
        back = records_to_dataframe(read_spots(csv_path))

        assert list(back.columns) == list(SCHEMA_COLUMNS)
        assert len(back) == len(df), "row count changed across write/read"
        numeric = [
            "x", "y", "p_detect", "logI1", "logI2", "I1", "I2",
            "log_ratio", "ratio", "uncertainty1", "uncertainty2",
        ]
        assert np.allclose(
            df[numeric].to_numpy(float), back[numeric].to_numpy(float),
            equal_nan=True, rtol=1e-5, atol=1e-8,
        ), "schema CSV did not round-trip numerically"
        # This model does not estimate PSF width: those columns are NA.
        assert df["sigma1_hat"].isna().all() and df["sigma2_hat"].isna().all()

    print(f"[end-to-end] loop ran 4 steps; emitted {len(df)} schema rows; "
          f"round-trip OK; artifacts written.")


# --------------------------------------------------------------------------- #
# Test 2: overfit a tiny fixed set (critical)                                  #
# --------------------------------------------------------------------------- #
def test_overfit_tiny_set() -> None:
    from spotpipe.training.train import overfit

    cfg = _load_config()
    n_images, steps = 6, 200
    result = overfit(cfg, n_images=n_images, steps=steps, log_fn=lambda _s: None)

    curve = result["loss_curve"]
    initial_total = curve[0][1]
    final_total = curve[-1][1]
    fe = result["final_eval"]

    # Print a compact loss curve + final intensity match (what the prompt asks to see).
    print(f"\n[overfit] {n_images} images, {steps} steps")
    print("  loss curve (step: total):")
    marks = {1, steps // 4, steps // 2, (3 * steps) // 4, steps}
    for step, total in curve:
        if step in marks:
            print(f"    {step:>4d}: {total:+.4f}")
    print(f"  final heatmap={fe['loss_heatmap']:.4f}  offset={fe['loss_offset']:.4f}  "
          f"intensity1={fe['loss_intensity1']:+.4f}  intensity2={fe['loss_intensity2']:+.4f}")
    print(f"  final intensity match at GT centres: "
          f"logI1_mae={fe['logI1_mae']:.4f}  logI2_mae={fe['logI2_mae']:.4f}  "
          f"(median {fe['logI1_median_ae']:.4f} / {fe['logI2_median_ae']:.4f}) over {int(fe['n_centers'])} centres")

    # The loss must collapse. (The total goes negative because the heteroscedastic
    # NLL becomes confident once the mean is fit -- so we check a large drop, not
    # literally ~0, but detection + offset DO go toward 0.)
    assert final_total < 0.5 * initial_total, (
        f"loss did not collapse: initial={initial_total:.3f} final={final_total:.3f}"
    )
    assert fe["loss_heatmap"] < 0.6, f"detection head did not learn (heatmap={fe['loss_heatmap']:.3f})"
    assert fe["loss_offset"] < 0.25, f"offset head did not learn (offset={fe['loss_offset']:.3f})"

    # The headline: predicted logI at GT centres must closely match the truth.
    assert fe["logI1_mae"] < 0.3, f"logI1 did not overfit (mae={fe['logI1_mae']:.3f})"
    assert fe["logI2_mae"] < 0.3, f"logI2 did not overfit (mae={fe['logI2_mae']:.3f})"


if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST 1: end-to-end loop + canonical-schema inference")
    print("=" * 70)
    test_end_to_end_loop_and_schema()

    print("\n" + "=" * 70)
    print("SMOKE TEST 2: overfit a tiny fixed set (critical)")
    print("=" * 70)
    test_overfit_tiny_set()

    print("\nAll smoke tests passed.")
