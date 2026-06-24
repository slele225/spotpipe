"""Tests for the phase-5b additions: hard-corner selection, three data roles,
provisional-vs-final benchmark labelling, and crash-resume.

Fast by construction -- small images / a handful of steps -- so the new control
flow is exercised without a real training run. Run standalone or under pytest::

    uv run python tests/test_training_5b.py
    uv run pytest tests/test_training_5b.py
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "train_smoke.yaml"


def _smoke_config() -> dict:
    from spotpipe.training.train import load_train_config

    return load_train_config(CONFIG)


# --------------------------------------------------------------------------- #
# 1. Selection key: hard-corner-primary, robust fallback, tie-breaks          #
# --------------------------------------------------------------------------- #
def test_selection_key_tiers() -> None:
    from spotpipe.training.train import _SELECT_TIERS, _selection_key

    MIN = 50
    hard_ok = {"val_hard_logratio_mae": 0.20, "val_hard_n_pairs": 60,
               "val_logratio_mae": 0.50, "val_det_f1": 0.8, "val_loss_total": 1.0}
    hard_better = {**hard_ok, "val_hard_logratio_mae": 0.10}
    hard_too_few = {**hard_ok, "val_hard_n_pairs": 10}     # below MIN -> falls back to overall
    overall_only = {"val_hard_logratio_mae": float("nan"), "val_hard_n_pairs": 0,
                    "val_logratio_mae": 0.4, "val_det_f1": 0.5, "val_loss_total": 1.2}
    loss_only = {"val_hard_logratio_mae": float("nan"), "val_hard_n_pairs": 0,
                 "val_logratio_mae": float("nan"), "val_det_f1": float("nan"), "val_loss_total": 2.0}

    k_hard = _selection_key(hard_ok, min_hard_pairs=MIN)
    k_hard_better = _selection_key(hard_better, min_hard_pairs=MIN)
    k_fallback = _selection_key(hard_too_few, min_hard_pairs=MIN)
    k_overall = _selection_key(overall_only, min_hard_pairs=MIN)
    k_loss = _selection_key(loss_only, min_hard_pairs=MIN)

    # Tier ordering: a hard-corner-eligible eval outranks overall, which outranks loss.
    assert k_hard[0] == 0 and _SELECT_TIERS[0].startswith("hard_corner")
    assert k_fallback[0] == 1 and k_overall[0] == 1
    assert k_loss[0] == 2
    assert k_hard < k_fallback < k_loss
    # Within the hard tier, lower hard-corner MAE wins.
    assert k_hard_better < k_hard
    # Too-few hard pairs => selection uses the OVERALL MAE, not the (smaller) hard MAE.
    assert math.isclose(k_fallback[1], hard_too_few["val_logratio_mae"])

    # Tie-break: same tier + same overall MAE -> higher detection F1 preferred.
    a = {"val_hard_logratio_mae": float("nan"), "val_hard_n_pairs": 0,
         "val_logratio_mae": 0.4, "val_det_f1": 0.5, "val_loss_total": 1.0}
    b = {**a, "val_det_f1": 0.9}
    assert _selection_key(b, min_hard_pairs=MIN) < _selection_key(a, min_hard_pairs=MIN)
    print("[5b] selection key: hard-corner-primary + min-pairs fallback + tie-breaks OK")


# --------------------------------------------------------------------------- #
# 2. Eval-dir round-trip: build -> write -> load (training + harness readers)  #
# --------------------------------------------------------------------------- #
def test_eval_dir_roundtrip() -> None:
    from spotpipe.benchmark.harness import load_eval_set
    from spotpipe.training.dataset import build_eval_examples, load_eval_examples, write_eval_dir
    from spotpipe.training.train import _build_detector, resolve_blocks

    cfg = _smoke_config()
    shape, det_cfg, scene_cfg, _adc = resolve_blocks(cfg)
    detector = _build_detector(det_cfg, 0)
    hsig = float(cfg["training"]["heatmap_sigma"])

    built = build_eval_examples(
        scene_cfg, detector, n_images=4, seed=70001, shape=shape,
        heatmap_sigma=hsig, n_hard_corner=2, id_prefix="val",
    )
    assert [e.meta["eval_kind"] for e in built][:2] == ["hard_corner", "hard_corner"]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "val"
        manifest = write_eval_dir(
            built, out, detector=detector, split="val", seed=70001,
            shape=shape, heatmap_sigma=hsig, scene_config=scene_cfg,
        )
        assert manifest["n_images"] == 4
        assert (out / "manifest.json").exists()

        # Training-side loader rebuilds dense targets and matches the originals.
        loaded = load_eval_examples(out, heatmap_sigma=hsig)
        assert len(loaded) == 4
        a, b = built[0], loaded[0]
        assert a.meta["image_id"] == b.meta["image_id"]
        assert np.allclose(a.image.numpy(), b.image.numpy(), atol=1e-4)
        assert len(a.spots) == len(b.spots)
        assert set(["center_mask", "logI1", "logI2"]).issubset(b.targets.keys())

        # The SAME directory is also readable by the benchmark harness.
        eval_images = load_eval_set(out)
        assert len(eval_images) == 4
        assert eval_images[0].image.shape[0] == 2
    print("[5b] eval-dir round-trip: training loader + harness loader both read it OK")


# --------------------------------------------------------------------------- #
# 3. Hard-corner val metric + provisional (val-set) benchmark labelling        #
# --------------------------------------------------------------------------- #
def _tiny_real_config(tmp: Path) -> dict:
    """A few-step real-run config on tiny images, val loaded from a built-on-disk set."""
    from spotpipe.training.dataset import build_eval_examples, write_eval_dir
    from spotpipe.training.train import _build_detector, resolve_blocks

    cfg = _smoke_config()
    shape, det_cfg, scene_cfg, _adc = resolve_blocks(cfg)
    detector = _build_detector(det_cfg, 0)
    hsig = float(cfg["training"]["heatmap_sigma"])
    val = build_eval_examples(scene_cfg, detector, n_images=4, seed=70001, shape=shape,
                              heatmap_sigma=hsig, n_hard_corner=2, id_prefix="val")
    val_dir = tmp / "val"
    write_eval_dir(val, val_dir, detector=detector, split="val", seed=70001,
                   shape=shape, heatmap_sigma=hsig, scene_config=scene_cfg)

    cfg["training"]["train_steps"] = 6
    cfg["training"]["batch_size"] = 2
    cfg["training"]["eval_every"] = 3
    cfg["training"]["checkpoint_every"] = 3
    cfg["training"]["variance_warmup_steps"] = 2
    cfg["training"]["lr_warmup_steps"] = 1
    cfg["training"]["curriculum"] = {"enabled": True, "curriculum_ramp_steps": 3}
    cfg["training"]["val"] = {"path": str(val_dir)}
    cfg["training"]["best_checkpoint"] = {"enabled": True, "hard_corner_min_pairs": 1}
    cfg["auto_benchmark"] = True
    cfg["benchmark"] = {
        "match_radius_px": 3.0, "match_method": "greedy",
        "snr_bins": [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")],
        "density_bins": [0.0, 1.0, 3.0, 6.0, float("inf")],
        "density_radius_px": 4.0,
        # NOTE: no test_set_dir -> auto-benchmark must fall back to a PROVISIONAL val run.
        "our_model": {"peak_threshold": 0.0, "nms_kernel": 3, "max_spots": 50},
        "uncertainty": {"n_sigma_bins": 4},
    }
    return cfg


def test_hard_corner_metric_and_provisional_benchmark() -> None:
    from spotpipe.training.train import train

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _tiny_real_config(tmp)
        run_dir = tmp / "run"
        result = train(cfg, out_dir=run_dir, resume=False, log_fn=lambda _s: None)

        # The validation metric carries BOTH overall and hard-corner fields.
        fe = result["final_eval"]
        for key in ("val_logratio_mae", "val_n_pairs",
                    "val_hard_logratio_mae", "val_hard_n_pairs", "val_det_f1"):
            assert key in fe, f"missing validation field {key}"

        # Resumable state + a selected best checkpoint were written.
        assert (run_dir / "train_state.pt").exists()
        assert (run_dir / "best_checkpoint.pt").exists()

        # The auto-benchmark fell back to a PROVISIONAL val run, clearly labelled, and
        # did NOT write anything that looks like a final test result.
        prov = run_dir / "benchmark_provisional_val"
        assert prov.exists(), "provisional val benchmark dir missing"
        assert not (run_dir / "benchmark_test").exists(), "must not produce a test-labelled dir"
        role = json.loads((prov / "data_role.json").read_text(encoding="utf-8"))
        assert role["provisional"] is True and role["is_final_test_result"] is False
        assert result["benchmark"]["provisional"] is True

        # val_curve persisted the hard-corner columns for inspection.
        import pandas as pd
        vc = pd.read_csv(run_dir / "val_curve.csv")
        assert "val_hard_logratio_mae" in vc.columns and "val_hard_n_pairs" in vc.columns
    print("[5b] hard-corner metric present; benchmark correctly labelled PROVISIONAL (val)")


# --------------------------------------------------------------------------- #
# 4. Crash resume: restore from a partial train_state and finish              #
# --------------------------------------------------------------------------- #
def test_resume_restores() -> None:
    import torch

    from spotpipe.training.train import train

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = _tiny_real_config(tmp)
        run_dir = tmp / "run"
        train(cfg, out_dir=run_dir, resume=False, log_fn=lambda _s: None)

        # Rewind the saved state to simulate a crash partway through, then resume.
        state = torch.load(run_dir / "train_state.pt", map_location="cpu", weights_only=False)
        state["step"] = 3
        torch.save(state, run_dir / "train_state.pt")

        msgs: list[str] = []
        result = train(cfg, out_dir=run_dir, resume=True, log_fn=msgs.append)
        assert any("[resume] restored train_state at step 3" in m for m in msgs), \
            "resume did not restore from the partial state"
        assert result["best"]["path"], "no best checkpoint after resume"
    print("[5b] crash resume: restored a partial train_state and finished the run")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("=" * 70)
            print(f"RUNNING {name}")
            print("=" * 70)
            fn()
    print("\nAll 5b tests passed.")
