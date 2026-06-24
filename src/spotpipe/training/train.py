"""Training driver for the two-channel spot model (build stage 3).

Builds the simulator-backed dataset, the HRNet spot model, the phase-1 loss
combiner (detection + localization + per-channel intensity NLL -- NO ratio/slope
loss), and an AdamW + cosine schedule, then runs a curriculum that ramps SCENE
difficulty only (never detector constants). Bias/variance is always measured on
a FIXED held-out evaluation set spanning the full final difficulty range
(``training.dataset.build_fixed_val_examples``), so curriculum progress never
confounds the metric.

Outputs (under the run dir): a copy of the config, a manifest pinning the git
commit + detector, a per-step metrics log, and a model checkpoint. An inference
helper (:func:`predict_dataset`) bridges to the canonical schema via
``spotpipe.models.predict_spots``.

Run a quick self-check::

    uv run python -m spotpipe.training.train --smoke
    uv run python -m spotpipe.training.train --overfit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from spotpipe.losses import SpotLoss
from spotpipe.models.spot_model import build_spot_model, predict_spots
from spotpipe.schema import records_to_dataframe
from spotpipe.simulator import forward_model, noise
from spotpipe.simulator.generate_dataset import _git_commit, load_simulator_config
from spotpipe.training.dataset import (
    Example,
    build_fixed_val_examples,
    collate,
    curriculum_scene_config,
    generate_examples,
    load_eval_examples,
)

__all__ = [
    "load_train_config",
    "resolve_blocks",
    "intensity_match_metrics",
    "evaluate",
    "validation_logratio_mae",
    "train",
    "overfit",
    "predict_dataset",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
def load_train_config(path: str | Path) -> dict:
    """Load a training YAML config into a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_blocks(config: dict) -> tuple[tuple[int, int], dict, dict, int]:
    """Resolve (shape, detector_cfg, scene_cfg, adc_max).

    A config may inline ``image`` / ``detector`` / ``scene`` blocks, and/or point
    at a simulator config via ``simulator_config: <path>`` whose blocks fill in
    anything not inlined. Inline blocks win whole (no deep merge).
    """
    base: dict = {}
    sim_ref = config.get("simulator_config")
    if sim_ref:
        sim_path = Path(sim_ref)
        if not sim_path.is_absolute():
            sim_path = REPO_ROOT / sim_path
        base = load_simulator_config(sim_path)

    image = config.get("image", base.get("image", {"height": 256, "width": 256}))
    detector_cfg = config.get("detector", base.get("detector", {}))
    scene_cfg = config.get("scene", base.get("scene", {}))
    shape = (int(image.get("height", 256)), int(image.get("width", 256)))
    adc_max = int(detector_cfg.get("adc_max", 4095))
    return shape, detector_cfg, scene_cfg, adc_max


def _build_detector(detector_cfg: dict, seed: int) -> noise.DetectorParams:
    """Sample the FIXED detector once, from the seed alone (as the simulator does)."""
    det_seed = np.random.SeedSequence(int(seed)).spawn(1)[0]
    return noise.sample_detector_params(detector_cfg, np.random.default_rng(det_seed))


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def intensity_match_metrics(
    preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Mean / median |logI_pred - logI_true| at GT centres, per channel.

    This is the headline overfit diagnostic: it reads predictions at exactly the
    masked centre pixels, so it measures the same quantity the intensity loss
    supervises.
    """
    mask = targets["center_mask"] > 0.5  # [B, 1, H, W]
    n = int(mask.sum())
    out: dict[str, float] = {"n_centers": float(n)}
    for ch, key in ((1, "logI1"), (2, "logI2")):
        if n > 0:
            err = (preds[key][mask] - targets[key][mask]).abs()
            out[f"logI{ch}_mae"] = float(err.mean())
            out[f"logI{ch}_median_ae"] = float(err.median())
        else:
            out[f"logI{ch}_mae"] = float("nan")
            out[f"logI{ch}_median_ae"] = float("nan")
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, dict[str, torch.Tensor]],
    loss_fn: SpotLoss,
) -> dict[str, float]:
    """Loss components + intensity-match metrics on a (already-collated) batch."""
    was_training = model.training
    model.eval()
    images, targets = batch
    preds = model(images)
    _, comps = loss_fn(preds, targets)
    metrics = {f"loss_{k}": float(v) for k, v in comps.items()}
    metrics.update(intensity_match_metrics(preds, targets))
    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def validation_logratio_mae(
    model: torch.nn.Module,
    eval_examples: list[Example],
    *,
    adc_max: float,
    meta_by_image: dict[str, dict] | None = None,
    match_radius_px: float = 3.0,
    method: str = "greedy",
    predict_kwargs: dict | None = None,
    device: str | torch.device = "cpu",
    density_radius_px: float = 4.0,
    hard_snr_hi: float = 2.0,
    hard_density_lo: float = 6.0,
) -> dict[str, float]:
    """Log-ratio MAE on matched pairs over the eval set, OVERALL and in the HARD CORNER.

    This is the headline-tracking validation metric the best-checkpoint selector
    uses. It runs the SAME inference -> canonical-schema -> location-matching path
    the benchmark uses (``predict_spots`` + ``benchmark.matching.match_dataset``),
    so the checkpoint chosen "best" mid-training is selected on exactly the
    quantity (and the exact eval set) the end-of-run benchmark reports on. It does
    NOT recompute or alter any benchmark metric -- it only reuses the matcher, the
    schema's ``log_ratio`` column, and (for the hard corner) the benchmark's own
    ``attach_features`` SNR / local-density axes.

    The **hard corner** is the lowest-SNR x highest-density cell, defined identically
    everywhere (prompt 5b / CLAUDE.md): true (GT) ``snr in [0, hard_snr_hi)`` AND true
    ``n_neighbors >= hard_density_lo`` -- i.e. ``snr_bins[0]`` x the top ``density_bins``
    cell. Features come from the GROUND-TRUTH matched spot (same choice the benchmark
    makes), so this number equals that 2-D cell's ``log_ratio.mae``.

    Returns ``val_logratio_mae`` / ``val_n_pairs`` (overall), ``val_hard_logratio_mae``
    / ``val_hard_n_pairs`` (hard corner), and ``val_recall`` / ``val_precision`` /
    ``val_det_f1`` (overall detection, for the selection tie-break). Any quantity that
    has no support (no detections / no matched pairs / hard corner empty) is ``nan``.
    """
    from spotpipe.benchmark.features import attach_features
    from spotpipe.benchmark.matching import match_dataset

    out = {
        "val_logratio_mae": float("nan"), "val_n_pairs": 0,
        "val_hard_logratio_mae": float("nan"), "val_hard_n_pairs": 0,
        "val_recall": float("nan"), "val_precision": float("nan"), "val_det_f1": float("nan"),
    }

    pk = predict_kwargs or {}
    pred = predict_dataset(model, eval_examples, adc_max=adc_max, device=device, **pk)
    gt = (
        pd.concat([ex.spots for ex in eval_examples], ignore_index=True)
        if eval_examples else pred.iloc[0:0]
    )
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 or n_pred == 0:
        return out

    # Attach the SNR / local-density axes so the hard-corner subset can be sliced
    # from the GT side of each matched pair (same definition as the benchmark).
    if meta_by_image is not None:
        gt = attach_features(gt, meta_by_image, density_radius_px=density_radius_px)
        pred = attach_features(pred, meta_by_image, density_radius_px=density_radius_px)

    match = match_dataset(gt, pred, max_distance=float(match_radius_px), method=method)
    n_matched = match.n_matched
    recall = n_matched / n_gt if n_gt else float("nan")
    precision = n_matched / n_pred if n_pred else float("nan")
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else float("nan")
    out.update(val_n_pairs=int(n_matched), val_recall=recall, val_precision=precision, val_det_f1=f1)
    if n_matched == 0:
        return out

    gt_m = match.gt_matched.reset_index(drop=True)
    pred_m = match.pred_matched.reset_index(drop=True)
    resid = pred_m["log_ratio"].to_numpy(float) - gt_m["log_ratio"].to_numpy(float)
    finite = np.isfinite(resid)
    out["val_logratio_mae"] = float(np.abs(resid[finite]).mean()) if finite.any() else float("nan")

    if "snr" in gt_m.columns and "n_neighbors" in gt_m.columns:
        snr = gt_m["snr"].to_numpy(float)
        nbr = gt_m["n_neighbors"].to_numpy(float)
        hard = finite & (snr >= 0.0) & (snr < float(hard_snr_hi)) & (nbr >= float(hard_density_lo))
        out["val_hard_n_pairs"] = int(hard.sum())
        if hard.any():
            out["val_hard_logratio_mae"] = float(np.abs(resid[hard]).mean())
    return out


# --------------------------------------------------------------------------- #
# Schedules (three independent knobs over one timeline)                        #
# --------------------------------------------------------------------------- #
def _lr_factor(step: int, *, warmup: int, total: int) -> float:
    """LR multiplier: linear warmup to 1.0 over ``warmup`` steps, then cosine
    decay to ~0 by ``total``.

    ``step`` is 1-indexed. At ``step == warmup`` the factor peaks at 1.0; the
    cosine then runs over ``[warmup, total]`` reaching 0 exactly at ``total``.
    """
    if warmup > 0 and step <= warmup:
        return step / float(warmup)
    denom = max(total - warmup, 1)
    progress = min(max((step - warmup) / float(denom), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _resolve_variance_warmup_steps(tcfg: dict, n_steps: int) -> int:
    """Fixed STEP COUNT of fixed-unit-variance fitting before the logvar NLL.

    Prefers the explicit ``variance_warmup_steps`` (a count); falls back to the
    legacy ``intensity_logvar_warmup_fraction`` * steps for the smoke config.
    """
    if tcfg.get("variance_warmup_steps") is not None:
        return int(tcfg["variance_warmup_steps"])
    return int(float(tcfg.get("intensity_logvar_warmup_fraction", 0.3)) * n_steps)


def _resolve_ramp_steps(tcfg: dict, cur_cfg: dict, n_steps: int) -> int:
    """Curriculum ramp length in steps (ramp easy->hard, then HOLD at full).

    Prefers the explicit ``curriculum_ramp_steps`` (under ``training`` or
    ``training.curriculum``); falls back to the legacy ``ramp_fraction`` * steps.
    """
    for src in (cur_cfg, tcfg):
        if src.get("curriculum_ramp_steps") is not None:
            return max(int(src["curriculum_ramp_steps"]), 1)
    return max(int(float(cur_cfg.get("ramp_fraction", 0.6)) * n_steps), 1)


# --------------------------------------------------------------------------- #
# Best-checkpoint selection                                                    #
# --------------------------------------------------------------------------- #
def _save_checkpoint(path: Path, model: torch.nn.Module, config: dict) -> None:
    torch.save({"model_state": model.state_dict(), "config": config}, path)


# Tier labels for the selection key (smaller tier == strictly preferred).
_SELECT_TIERS = {0: "hard_corner_val_logratio_mae", 1: "overall_val_logratio_mae", 2: "val_total_loss"}


def _selection_key(row: dict, *, min_hard_pairs: int) -> tuple:
    """Best-checkpoint sort key (SMALLER is better) -- robust hard-corner selection.

    The whole low-bias / low-variance claim lives in the dim x high-overlap corner, so
    we select PRIMARILY on the hard-corner ``val_logratio_mae`` -- but only when that
    corner has enough matched pairs to be stable (``hard_corner_min_pairs``, default
    50; prompt 5b clarification #2). Otherwise we fall back to the OVERALL
    ``val_logratio_mae``, and only if even that is unavailable (no matched pairs yet,
    common very early) to ``val_total_loss``. Ties within a tier break by overall
    log-ratio MAE then by higher detection F1.

    The leading tier index makes the ordering total and monotone: a hard-corner-eligible
    eval always outranks an overall-only one, which always outranks a loss-only one, so
    selection can never oscillate down a tier once a higher tier becomes available.
    NOTE: every input here is a VALIDATION quantity -- benchmark/test outputs are never
    consulted for selection (CLAUDE.md / prompt 5b).
    """
    hard = row.get("val_hard_logratio_mae")
    hard_n = int(row.get("val_hard_n_pairs", 0) or 0)
    overall = row.get("val_logratio_mae")
    f1 = row.get("val_det_f1")
    loss = row.get("val_loss_total")

    overall_tb = float(overall) if (overall is not None and math.isfinite(overall)) else math.inf
    f1_tb = float(f1) if (f1 is not None and math.isfinite(f1)) else 0.0

    if hard is not None and math.isfinite(hard) and hard_n >= int(min_hard_pairs):
        return (0, float(hard), overall_tb, -f1_tb)
    if overall is not None and math.isfinite(overall):
        return (1, float(overall), overall_tb, -f1_tb)
    loss_v = float(loss) if (loss is not None and math.isfinite(loss)) else math.inf
    return (2, loss_v, overall_tb, -f1_tb)


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device spec; ``"auto"``/None picks CUDA (the A100) when available."""
    if isinstance(spec, torch.device):
        return spec
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _coerce_bin_edges(edges) -> list[float]:
    """Turn YAML bin edges into floats; ``null`` (YAML for a missing top) -> +inf."""
    return [math.inf if e is None else float(e) for e in edges]


def _save_train_state(
    run_dir: Path, *, model, optimizer, step, n_steps, best, history, loss_curve,
    val_curve, saved_ckpts, logvar_on,
) -> None:
    """Atomically write a resumable training state (weights + optimizer + bookkeeping).

    Written every checkpoint so an overnight crash resumes mid-stream. The write is
    staged to a temp file and renamed so a crash DURING the write can't corrupt the
    state a resume would read.
    """
    tmp = run_dir / "train_state.pt.tmp"
    torch.save(
        {
            "step": int(step), "n_steps": int(n_steps),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best": best, "history": history,
            "loss_curve": [list(x) for x in loss_curve],
            "val_curve": val_curve, "saved_ckpts": saved_ckpts,
            "logvar_on": bool(logvar_on),
        },
        tmp,
    )
    tmp.replace(run_dir / "train_state.pt")


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train(
    config: dict,
    *,
    device: str | torch.device = "cpu",
    out_dir: str | Path | None = None,
    fixed_train_examples: list[Example] | None = None,
    eval_examples: list[Example] | None = None,
    steps: int | None = None,
    curriculum_enabled: bool | None = None,
    resume: bool = True,
    log_fn=print,
) -> dict:
    """Run training and return a summary dict.

    Parameters
    ----------
    config : parsed training config.
    out_dir : if given, run artifacts (config copy, manifest, metrics, checkpoint)
        are written here; if None, nothing is written to disk.
    fixed_train_examples : if given, this exact set is used as the train batch on
        EVERY step (the overfit harness); otherwise fresh images are generated
        per step under the curriculum.
    eval_examples : the held-out evaluation set; if None, a fixed val set is built
        from the config (``training.val``).
    steps : overrides ``training.steps``.
    curriculum_enabled : overrides ``training.curriculum.enabled``.
    """
    device = resolve_device(device)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    tcfg = config.get("training", {})
    model_cfg = config.get("model", {})
    shape, detector_cfg, scene_cfg, adc_max = resolve_blocks(config)

    # ``train_steps`` is the real-run knob; ``steps`` is the legacy smoke key.
    n_steps = int(steps if steps is not None else tcfg.get("train_steps", tcfg.get("steps", 1000)))
    batch_size = int(tcfg.get("batch_size", 8))
    lr = float(tcfg.get("lr", 2e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 1.0))  # 0 disables
    heatmap_sigma = float(tcfg.get("heatmap_sigma", 1.5))
    loss_weights = tcfg.get("loss_weights", None)
    log_every = int(tcfg.get("log_every", max(n_steps // 10, 1)))

    # --- three schedules over one timeline (all configurable step-count knobs) --
    # (1) LR: linear warmup over ``lr_warmup_steps`` then cosine decay to ~0.
    lr_warmup_steps = int(tcfg.get("lr_warmup_steps", 0))
    # (2) Variance warmup: a fixed STEP COUNT of fixed-unit-variance fitting before
    #     the heteroscedastic logvar NLL switches on.
    logvar_warmup_steps = _resolve_variance_warmup_steps(tcfg, n_steps)
    # (3) Scene curriculum: ramp easy->hard over ``curriculum_ramp_steps``, HOLD.
    cur_cfg = tcfg.get("curriculum", {})
    use_curriculum = (
        cur_cfg.get("enabled", True) if curriculum_enabled is None else bool(curriculum_enabled)
    )
    ramp_steps = _resolve_ramp_steps(tcfg, cur_cfg, n_steps)

    # Eval / checkpoint cadence (real-run knobs; legacy val.eval_every honoured).
    eval_every = int(tcfg.get("eval_every", tcfg.get("val", {}).get("eval_every", max(n_steps // 5, 1))))
    checkpoint_every = int(tcfg.get("checkpoint_every", 0))

    # In-loss logvar clamp (numerical stability only; head output stays unclamped).
    logvar_min = float(tcfg.get("logvar_min", -10.0))
    logvar_max = float(tcfg.get("logvar_max", 6.0))

    # Best-checkpoint selection: hard-corner val_logratio_mae primarily, with the
    # robust fallback to overall MAE / val_total_loss (see ``_selection_key``).
    best_cfg = tcfg.get("best_checkpoint", {})
    track_best = bool(best_cfg.get("enabled", checkpoint_every > 0)) and out_dir is not None
    # Minimum matched pairs for the hard corner to drive selection (else fall back).
    hard_corner_min_pairs = int(
        best_cfg.get("hard_corner_min_pairs", tcfg.get("hard_corner_min_pairs", 50))
    )

    # Our-model inference settings, reused for BOTH the mid-training val metric and
    # the end-of-run benchmark (one definition -> selection matches what's reported).
    bench_cfg = config.get("benchmark", {})
    om_cfg = bench_cfg.get("our_model", {})
    match_radius_px = float(bench_cfg.get("match_radius_px", 3.0))
    match_method = str(bench_cfg.get("match_method", "greedy"))
    om_predict_kwargs = dict(
        peak_threshold=float(om_cfg.get("peak_threshold", 0.3)),
        nms_kernel=int(om_cfg.get("nms_kernel", 3)),
        max_spots=om_cfg.get("max_spots", 2000),
        logvar_min=logvar_min, logvar_max=logvar_max,
    )
    # The ONE agreed binning, used to define the hard corner identically everywhere:
    # lowest SNR bin x highest density bin (prompt 5b clarification #3).
    snr_bins = _coerce_bin_edges(bench_cfg.get("snr_bins", [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")]))
    density_bins = _coerce_bin_edges(bench_cfg.get("density_bins", [0.0, 1.0, 3.0, 6.0, float("inf")]))
    density_radius_px = float(bench_cfg.get("density_radius_px", 4.0))
    hard_snr_hi = float(snr_bins[1])            # upper edge of the lowest SNR bin
    hard_density_lo = float(density_bins[-2])   # lower edge of the highest density bin
    val_metric_kwargs = dict(
        adc_max=adc_max, match_radius_px=match_radius_px, method=match_method,
        predict_kwargs=om_predict_kwargs, device=device,
        density_radius_px=density_radius_px,
        hard_snr_hi=hard_snr_hi, hard_density_lo=hard_density_lo,
    )

    detector = _build_detector(detector_cfg, seed)
    model = build_spot_model(model_cfg).to(device)
    loss_fn = SpotLoss(loss_weights, logvar_min=logvar_min, logvar_max=logvar_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # The FIXED validation set, used ONLY for best-checkpoint selection (prompt 5b:
    # training / validation / test are three distinct data roles). Prefer a set built
    # ONCE and persisted to ``val.path`` so BOTH runs select on byte-identical data;
    # fall back to building one in memory from the config.
    vcfg = tcfg.get("val", {})
    eval_seed = int(vcfg.get("seed", 12345))
    val_path = vcfg.get("path")
    if eval_examples is None and val_path:
        val_dir = Path(val_path)
        if not val_dir.is_absolute():
            val_dir = REPO_ROOT / val_dir
        if not (val_dir / "manifest.json").exists():
            raise FileNotFoundError(
                f"validation set not found at {val_dir} -- build it once with "
                f"scripts/build_fixed_eval.py before launching the run"
            )
        eval_examples = load_eval_examples(val_dir, heatmap_sigma=heatmap_sigma)
        log_fn(f"[val] loaded FIXED validation set ({len(eval_examples)} images) from {val_dir}")
    if eval_examples is None:
        eval_examples = build_fixed_val_examples(
            scene_cfg, detector,
            n_images=int(vcfg.get("n_images", 8)),
            seed=eval_seed,
            shape=shape, heatmap_sigma=heatmap_sigma,
        )
    meta_by_image = {ex.meta["image_id"]: ex.meta for ex in eval_examples}
    eval_batch = _to_device(collate(eval_examples, adc_max), device)

    fixed_train_batch = None
    if fixed_train_examples is not None:
        fixed_train_batch = _to_device(collate(fixed_train_examples, adc_max), device)

    # Create the run dir up-front so checkpoints can be written during the loop.
    run_dir = Path(out_dir) if out_dir is not None else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        _write_eval_manifest(
            run_dir, eval_examples, seed=eval_seed, shape=shape,
            detector=detector, match_radius_px=match_radius_px,
        )

    history: list[dict] = []
    loss_curve: list[tuple[int, float]] = []
    val_curve: list[dict] = []
    best = {"step": None, "metric": None, "value": math.inf, "key": None, "path": None}
    saved_ckpts: list[str] = []
    logvar_on = False
    start_step = 1

    # --- crash resume: continue from the last saved train_state if present ------ #
    # Datagen is per-step-seeded and the forward/backward path is deterministic, so
    # restoring (weights, optimizer, step, best, curves) is sufficient to resume an
    # overnight run mid-stream rather than restart from zero (prompt 5b robustness).
    if resume and run_dir is not None and (run_dir / "train_state.pt").exists():
        state = torch.load(run_dir / "train_state.pt", map_location=device, weights_only=False)
        if int(state.get("n_steps", n_steps)) == n_steps and int(state.get("step", 0)) < n_steps:
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            start_step = int(state["step"]) + 1
            best = state.get("best", best)
            history = state.get("history", [])
            loss_curve = [tuple(x) for x in state.get("loss_curve", [])]
            val_curve = state.get("val_curve", [])
            saved_ckpts = state.get("saved_ckpts", [])
            logvar_on = bool(state.get("logvar_on", False))
            log_fn(f"[resume] restored train_state at step {state['step']}; continuing from {start_step}")

    for step in range(start_step, n_steps + 1):
        # --- (1) LR schedule: warmup -> cosine decay ------------------------- #
        cur_lr = lr * _lr_factor(step, warmup=lr_warmup_steps, total=n_steps)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        # --- (3) scene curriculum: ramp then hold at full difficulty --------- #
        if fixed_train_batch is not None:
            images, targets = fixed_train_batch
            cur_t = 1.0
        else:
            cur_t = min(step / ramp_steps, 1.0) if use_curriculum else 1.0
            scene_t = curriculum_scene_config(scene_cfg, cur_t) if use_curriculum else scene_cfg
            step_seed = np.random.SeedSequence([seed, step])
            examples = generate_examples(
                scene_t, detector, n_images=batch_size, seed=step_seed,
                shape=shape, heatmap_sigma=heatmap_sigma,
            )
            images, targets = _to_device(collate(examples, adc_max), device)

        # --- (2) variance warmup: fixed unit variance, then learned logvar --- #
        use_logvar = step > logvar_warmup_steps
        if use_logvar and not logvar_on:
            log_fn(
                f"[schedule] step {step}: variance warmup complete -- enabling "
                f"learned-logvar NLL (steps 1-{logvar_warmup_steps} fit the mean "
                f"under fixed unit variance)"
            )
            logvar_on = True

        model.train()
        preds = model(images)
        total, comps = loss_fn(preds, targets, intensity_use_logvar=use_logvar)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        loss_curve.append((step, float(total.detach())))

        is_final = step == n_steps
        is_eval = (eval_every > 0 and step % eval_every == 0) or is_final
        is_ckpt = checkpoint_every > 0 and ((step % checkpoint_every == 0) or is_final)
        do_eval = is_eval or is_ckpt

        row = {"step": step, "lr": cur_lr, "curriculum_t": cur_t, "use_logvar": use_logvar}
        row.update({f"train_{k}": float(v) for k, v in comps.items()})

        if do_eval:
            row.update({f"val_{k}": v for k, v in evaluate(model, eval_batch, loss_fn).items()})
            row.update(validation_logratio_mae(
                model, eval_examples, meta_by_image=meta_by_image, **val_metric_kwargs,
            ))
            val_curve.append({
                k: row.get(k) for k in (
                    "step", "lr", "curriculum_t", "val_loss_total",
                    "val_logratio_mae", "val_n_pairs",
                    "val_hard_logratio_mae", "val_hard_n_pairs",
                    "val_recall", "val_precision", "val_det_f1",
                    "val_logI1_mae", "val_logI2_mae",
                )
            })

        if (step % log_every == 0) or step == 1 or do_eval:
            history.append(row)
            log_fn(_format_row(row))

        # --- checkpoint + best-checkpoint selection (hard-corner val_logratio_mae) - #
        if is_ckpt and run_dir is not None:
            ckpt_path = run_dir / f"checkpoint_step{step:05d}.pt"
            _save_checkpoint(ckpt_path, model, config)
            saved_ckpts.append(str(ckpt_path))
            if track_best:
                cand_key = _selection_key(row, min_hard_pairs=hard_corner_min_pairs)
                if best["key"] is None or cand_key < best["key"]:
                    best_path = run_dir / "best_checkpoint.pt"
                    _save_checkpoint(best_path, model, config)
                    best.update(
                        step=step, path=str(best_path), key=cand_key,
                        metric=_SELECT_TIERS[cand_key[0]], value=float(cand_key[1]),
                        hard_n_pairs=int(row.get("val_hard_n_pairs", 0) or 0),
                        overall_val_logratio_mae=row.get("val_logratio_mae"),
                        det_f1=row.get("val_det_f1"),
                    )
                    log_fn(
                        f"[best] step {step}: new best by {best['metric']}={best['value']:.4f} "
                        f"(hard_n_pairs={best['hard_n_pairs']}, overall_mae="
                        f"{_fmt(best['overall_val_logratio_mae'])}, f1={_fmt(best['det_f1'])})"
                    )
            _save_train_state(
                run_dir, model=model, optimizer=optimizer, step=step, n_steps=n_steps,
                best=best, history=history, loss_curve=loss_curve, val_curve=val_curve,
                saved_ckpts=saved_ckpts, logvar_on=logvar_on,
            )

    final_eval = evaluate(model, eval_batch, loss_fn)
    final_eval.update(validation_logratio_mae(
        model, eval_examples, meta_by_image=meta_by_image, **val_metric_kwargs,
    ))

    # If checkpointing was on but nothing was selected (e.g. eval never matched a
    # pair), fall back to the final weights so a best checkpoint always exists.
    if track_best and best["path"] is None and run_dir is not None:
        best_path = run_dir / "best_checkpoint.pt"
        _save_checkpoint(best_path, model, config)
        best.update(step=n_steps, path=str(best_path), metric="final", value=float("nan"))

    benchmark_summary = None
    if run_dir is not None:
        _write_run_outputs(
            run_dir, config=config, history=history, loss_curve=loss_curve,
            val_curve=val_curve, final_eval=final_eval, model=model, detector=detector,
            shape=shape, n_steps=n_steps, best=best, saved_ckpts=saved_ckpts,
        )
        # Auto-benchmark the BEST checkpoint over the SAME fixed eval set.
        if config.get("auto_benchmark", False):
            benchmark_summary = _run_auto_benchmark(
                run_dir, config, eval_examples, best, device=device, log_fn=log_fn,
            )

    return {
        "run_dir": str(run_dir) if run_dir else None,
        "history": history,
        "loss_curve": loss_curve,
        "val_curve": val_curve,
        "final_eval": final_eval,
        "best": best,
        "model": model,
        "detector": detector,
        "adc_max": adc_max,
        "config": config,
        "benchmark": benchmark_summary,
    }


def overfit(
    config: dict,
    *,
    n_images: int = 6,
    steps: int = 200,
    device: str | torch.device = "cpu",
    log_fn=print,
) -> dict:
    """Overfit a tiny FIXED set of images: the fastest masking/target sanity check.

    Builds ``n_images`` images once (curriculum off, full-range scene), then
    trains on that exact batch every step. Returns the standard ``train`` summary
    plus a fixed set whose loss should collapse and whose predicted logI1/logI2
    at GT centres should closely match the truth -- if it cannot, the loss or the
    target-map construction is broken.
    """
    seed = int(config.get("seed", 0))
    shape, detector_cfg, scene_cfg, _ = resolve_blocks(config)
    heatmap_sigma = float(config.get("training", {}).get("heatmap_sigma", 1.5))
    detector = _build_detector(detector_cfg, seed)
    fixed = generate_examples(
        scene_cfg, detector, n_images=n_images, seed=np.random.SeedSequence([seed, 777]),
        shape=shape, heatmap_sigma=heatmap_sigma, id_prefix="overfit",
    )
    return train(
        config, device=device, out_dir=None,
        fixed_train_examples=fixed, eval_examples=fixed,
        steps=steps, curriculum_enabled=False, log_fn=log_fn,
    )


# --------------------------------------------------------------------------- #
# Inference bridge to the canonical schema                                     #
# --------------------------------------------------------------------------- #
def predict_dataset(
    model: torch.nn.Module,
    examples: list[Example],
    *,
    adc_max: float = 4095.0,
    device: str | torch.device = "cpu",
    **predict_kwargs,
) -> pd.DataFrame:
    """Run inference over a list of examples and concatenate canonical-schema rows.

    Images that produced no detections (common early in training) contribute an
    empty frame; those are dropped before concatenation so they neither add rows
    nor trigger pandas' all-NA-concat FutureWarning.
    """
    frames = [
        predict_spots(
            model, ex.image, image_id=ex.meta["image_id"], adc_max=adc_max,
            device=device, **predict_kwargs,
        )
        for ex in examples
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        return records_to_dataframe([])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _to_device(batch, device):
    images, targets = batch
    return images.to(device), {k: v.to(device) for k, v in targets.items()}


def _fmt(v) -> str:
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _format_row(row: dict) -> str:
    parts = [f"step {row['step']:>5d}", f"lr={row.get('lr', 0.0):.2e}", f"t={row.get('curriculum_t', 1.0):.2f}"]
    for k in ("train_total", "train_heatmap", "train_offset", "train_intensity1", "train_intensity2"):
        if k in row:
            parts.append(f"{k.replace('train_', '')}={row[k]:.4f}")
    if "val_loss_total" in row:
        parts.append(f"| val_total={row['val_loss_total']:.4f}")
    if "val_logratio_mae" in row:
        parts.append(f"val_logratio_mae={_fmt(row['val_logratio_mae'])}(n={row.get('val_n_pairs', 0)})")
    if "val_hard_logratio_mae" in row:
        parts.append(
            f"hard_mae={_fmt(row['val_hard_logratio_mae'])}(n={row.get('val_hard_n_pairs', 0)})"
        )
    if "val_det_f1" in row:
        parts.append(f"f1={_fmt(row['val_det_f1'])}")
    if "val_logI1_mae" in row:
        parts.append(f"val_logI1_mae={_fmt(row['val_logI1_mae'])}")
        parts.append(f"val_logI2_mae={_fmt(row['val_logI2_mae'])}")
    return "  ".join(parts)


def _write_eval_manifest(
    run_dir: Path, eval_examples, *, seed, shape, detector, match_radius_px
) -> dict:
    """Record the ONE fixed eval set so both uses provably refer to it.

    This is the single manifest the prompt requires: the same set (one seed, the
    image-ids listed here, with the forced hard corners) feeds periodic validation
    AND the end-of-run auto-benchmark.
    """
    manifest = {
        "purpose": (
            "The single fixed evaluation set for this experiment. Built ONCE and "
            "reused for BOTH (a) periodic validation during training and (b) the "
            "end-of-run auto-benchmark, so the best checkpoint is selected on the "
            "same data the benchmark reports on."
        ),
        "git_commit": _git_commit(),
        "seed": int(seed),
        "n_images": len(eval_examples),
        "shape": [shape[0], shape[1]],
        "match_radius_px": float(match_radius_px),
        "hard_corners": "beta=0, +beta extreme, -beta extreme, dim x high-overlap, bright x sparse",
        "detector": forward_model._detector_to_meta(detector),
        "images": [
            {
                "image_id": ex.meta["image_id"],
                "n_spots": int(ex.meta.get("n_spots", len(ex.spots))),
                "beta": ex.meta.get("scene", {}).get("beta"),
            }
            for ex in eval_examples
        ],
    }
    with open(run_dir / "eval_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _write_run_outputs(
    out_dir, *, config, history, loss_curve, val_curve, final_eval, model, detector,
    shape, n_steps, best, saved_ckpts,
) -> Path:
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    tcfg = config.get("training", {})
    manifest = {
        "git_commit": _git_commit(),
        "seed": int(config.get("seed", 0)),
        "steps": int(n_steps),
        "shape": [shape[0], shape[1]],
        "schedules": {
            "lr_warmup_steps": int(tcfg.get("lr_warmup_steps", 0)),
            "variance_warmup_steps": _resolve_variance_warmup_steps(tcfg, n_steps),
            "curriculum_ramp_steps": _resolve_ramp_steps(tcfg, tcfg.get("curriculum", {}), n_steps),
            "eval_every": int(tcfg.get("eval_every", tcfg.get("val", {}).get("eval_every", 0))),
            "checkpoint_every": int(tcfg.get("checkpoint_every", 0)),
        },
        "best_checkpoint": {
            "selection_metric": (
                "hard-corner val_logratio_mae if hard_n_pairs >= hard_corner_min_pairs; "
                "else overall val_logratio_mae; else val_total_loss "
                "(tie-break: overall val_logratio_mae, then detection F1). "
                "Selection NEVER uses benchmark/test outputs."
            ),
            "hard_corner_min_pairs": int(
                tcfg.get("best_checkpoint", {}).get(
                    "hard_corner_min_pairs", tcfg.get("hard_corner_min_pairs", 50)
                )
            ),
            "hard_corner_def": "true SNR in [0, snr_bins[1]) AND true n_neighbors >= density_bins[-2]",
            "selected_step": best.get("step"),
            "selected_by": best.get("metric"),
            "selected_value": best.get("value"),
            "selected_hard_n_pairs": best.get("hard_n_pairs"),
            "selected_overall_val_logratio_mae": best.get("overall_val_logratio_mae"),
            "selected_det_f1": best.get("det_f1"),
            "path": Path(best["path"]).name if best.get("path") else None,
        },
        "checkpoints": [Path(p).name for p in saved_ckpts],
        "detector": forward_model._detector_to_meta(detector),
        "final_eval": final_eval,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as fh:
        for row in history:
            fh.write(json.dumps(row) + "\n")

    pd.DataFrame(loss_curve, columns=["step", "total"]).to_csv(run_dir / "loss_curve.csv", index=False)
    if val_curve:
        pd.DataFrame(val_curve).to_csv(run_dir / "val_curve.csv", index=False)

    _save_checkpoint(run_dir / "checkpoint.pt", model, config)
    return run_dir


def _resolve_test_set_dir(config: dict) -> Path | None:
    """Return the frozen benchmark/test-set dir IFF it exists on disk, else None.

    The test set is a SEPARATE frozen set (different seed/manifest from the val set)
    built by a different prompt and used ONLY for final reporting -- never for
    checkpoint selection. If it isn't there yet, the caller falls back to a clearly
    labelled provisional run on the validation set (prompt 5b).
    """
    test_dir = config.get("benchmark", {}).get("test_set_dir")
    if not test_dir:
        return None
    path = Path(test_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if (path / "manifest.json").exists() else None


def _run_auto_benchmark(run_dir: Path, config: dict, eval_examples, best: dict, *,
                        device="cpu", log_fn=print) -> dict | None:
    """Auto-benchmark the BEST checkpoint with the two runnable baselines.

    Reports on the FROZEN TEST set when it exists on disk (``benchmark.test_set_dir``)
    -- the only set valid for final numbers, since the checkpoint was selected on the
    val set and reporting on the *same* set would overstate the result. If the test
    set does not exist yet, it falls back to the VALIDATION set and labels the output
    loudly as provisional, writing it to ``benchmark_provisional_val/`` (never a path
    that looks like a final test result; prompt 5b clarification #4).
    """
    from spotpipe.benchmark.harness import (
        EvalImage, load_eval_set, load_model_from_checkpoint, run_benchmark,
    )

    if not best.get("path"):
        log_fn("[benchmark] no best checkpoint available; skipping auto-benchmark.")
        return None

    test_dir = _resolve_test_set_dir(config)
    eval_images = None
    if test_dir is not None:
        try:
            eval_images = load_eval_set(test_dir)
        except Exception as exc:  # incompatible/partial test-set layout: don't crash the run
            log_fn(
                f"[benchmark] WARNING: found a test set at {test_dir} but could not load it "
                f"({type(exc).__name__}: {exc}); falling back to a PROVISIONAL val benchmark."
            )
            eval_images = None
    if eval_images is not None:
        provisional, data_role = False, "test"
        bench_out = run_dir / "benchmark_test"
        source = str(test_dir)
    else:
        eval_images = [
            EvalImage(
                image_id=ex.meta["image_id"], image=np.asarray(ex.image),
                meta=ex.meta, gt=ex.spots,
            )
            for ex in eval_examples
        ]
        provisional, data_role = True, "provisional_val"
        bench_out = run_dir / "benchmark_provisional_val"
        source = "FIXED validation set (no frozen test set on disk yet)"
        log_fn(
            "[benchmark] WARNING: no frozen test set found -- running a PROVISIONAL "
            "benchmark on the VALIDATION set (the same set used for selection). These "
            "are NOT final test numbers; re-run once the frozen test set exists."
        )

    best_model = load_model_from_checkpoint(best["path"], device=device)
    methods = ["our_model", "classical_per_channel_aperture", "oracle_center_aperture_divide"]
    log_fn(
        f"[benchmark] auto-benchmark on BEST checkpoint (step {best.get('step')}, "
        f"{best.get('metric')}={best.get('value')}) over the {data_role} set "
        f"({len(eval_images)} images) -> {bench_out}"
    )
    result = run_benchmark(
        eval_images, config, out_dir=bench_out, methods=methods,
        model=best_model, device=device, log_fn=log_fn,
    )

    # Stamp the data role next to the metrics so an output can never be mistaken for
    # a final test result when it was actually a provisional val read.
    role = {
        "data_role": data_role,
        "provisional": provisional,
        "is_final_test_result": (not provisional),
        "source": source,
        "selected_checkpoint": Path(best["path"]).name,
        "note": (
            "PROVISIONAL -- benchmarked on the validation set used for selection; "
            "NOT a final test result." if provisional else
            "Final benchmark on the frozen test set (separate from val selection)."
        ),
    }
    with open(Path(bench_out) / "data_role.json", "w", encoding="utf-8") as fh:
        json.dump(role, fh, indent=2)

    return {
        "out_dir": result["out_dir"],
        "data_role": data_role,
        "provisional": provisional,
        "methods_run": list(result["metrics"]),
        "skipped": result["skipped"],
        "figures": [Path(f).name for f in result["figures"]],
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the spot model. Default is a REAL run (all three schedules, "
        "best-checkpoint selection, optional auto-benchmark); --smoke / --overfit are "
        "the quick self-checks."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "train_smoke.yaml"))
    parser.add_argument("--out", default=None, help="run output dir (default: experiment outputs/, or none)")
    parser.add_argument("--steps", type=int, default=None, help="override the configured step count")
    parser.add_argument("--device", default="auto",
                        help="'auto' (CUDA/A100 if available, else CPU), or 'cuda' / 'cpu'")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any train_state.pt and start fresh")
    parser.add_argument("--smoke", action="store_true", help="quick end-to-end loop + inference (5 steps)")
    parser.add_argument("--overfit", action="store_true", help="overfit a tiny fixed set")
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    dev = resolve_device(args.device)
    print(f"[train] device={dev} (cuda_available={torch.cuda.is_available()})")

    if args.overfit:
        result = overfit(config, steps=args.steps or 200, device=args.device)
        final = result["final_eval"]
        print(f"[overfit] final logI1_mae={final['logI1_mae']:.4f} logI2_mae={final['logI2_mae']:.4f}")
        return 0

    if args.smoke:
        steps = args.steps or 5
        result = train(config, device=args.device, out_dir=args.out, steps=steps)
        print(f"[smoke] trained {steps} steps; run_dir={result['run_dir']}")
        return 0

    # --- Real run -------------------------------------------------------------
    out = args.out
    if out is None and config.get("experiment"):
        # Stamped experiment config: default outputs to the experiment's outputs/.
        out = str(Path(args.config).resolve().parent / "outputs")
    if out is None:
        parser.error(
            "a real run needs --out (or an experiment config.yaml carrying an 'experiment' block)"
        )

    result = train(config, device=dev, out_dir=out, steps=args.steps, resume=not args.no_resume)
    best = result["best"]
    print(f"[train] done: run_dir={result['run_dir']}")
    print(
        f"[train] best checkpoint: step={best.get('step')} "
        f"selected_by={best.get('metric')} value={best.get('value')} "
        f"(hard_n_pairs={best.get('hard_n_pairs')}, "
        f"overall_mae={best.get('overall_val_logratio_mae')}, f1={best.get('det_f1')})"
    )
    bench = result.get("benchmark")
    if bench:
        label = "PROVISIONAL (val set)" if bench.get("provisional") else "FINAL (test set)"
        print(
            f"[train] auto-benchmark [{label}]: methods={bench['methods_run']} "
            f"skipped={bench['skipped']} figures={bench['figures']} -> {bench['out_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
