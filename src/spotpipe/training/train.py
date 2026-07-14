"""Training driver for the measured-detector hrnet_large retrain.

Ported/adapted from the old repo's ``spotpipe.training.train``. It builds the
FROZEN HRNet spot model + phase-1 loss (detection + localization + per-channel
intensity NLL -- NO ratio/slope loss), an AdamW + warmup/cosine schedule, and runs
the three-stage schedule the old run used (LR warmup < variance warmup < curriculum
full), selecting the best checkpoint on the FIXED-val ``val_logratio_mae``.

Departures from the old driver (measured-detector retrain):

* Data comes from a MULTI-WORKER :class:`~spotpipe.training.dataset.SpotStreamDataset`
  DataLoader hoisted ONCE before the loop (CHANGE 5), not inline single-process
  generation. ``--profile`` reports dataload-vs-compute time and fails the <20% gate.
* Detector gains are randomised per image and the intensity range is solved per image
  (CHANGE 1 + 2), inside the dataset; the driver only wires the config through.
* GPU is asserted (fail loud) when requested; a silent CPU fallback would take days.
* The old auto-benchmark path is dropped (its harness modules are not ported here);
  benchmark-v2 smoke inference uses the existing ``spotpipe infer`` adapter instead.

Quick self-checks::

    spotpipe train --overfit          # tiny fixed set; loss must collapse
    spotpipe train --profile          # dataload-vs-compute timing gate
    spotpipe train --smoke            # short end-to-end loop
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from spotpipe.losses import SpotLoss
from spotpipe.models.spot_model import build_spot_model, predict_spots
from spotpipe.schema import records_to_dataframe
from spotpipe.simulator._features import attach_features
from spotpipe.simulator.generate_dataset import _git_commit
from spotpipe.training.dataset import (
    Example,
    IntensityWindowConfig,
    build_eval_examples,
    collate,
    generate_examples,
    load_eval_examples,
    make_loader,
    summarize_solved_windows,
)
from spotpipe.training.intensity_window import DetectorConstants

__all__ = [
    "load_train_config",
    "resolve_blocks",
    "resolve_device",
    "intensity_match_metrics",
    "evaluate",
    "validation_logratio_mae",
    "predict_dataset",
    "profile_dataload",
    "train",
    "overfit",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
# Pinned vendored simulator SHA (VENDORED_NOTES.md / benchmark manifest); recorded
# in provenance so the checkpoint names the forward model it trained on.
VENDORED_SIMULATOR_SHA = "7b9a0b85ee527afeb73d9e68f9bdb30960775083"


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
def load_train_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_blocks(config: dict) -> tuple[tuple[int, int], dict, dict, int]:
    """Resolve ``(shape, detector_cfg, scene_cfg, adc_max)`` from inline blocks.

    Unlike the old resolver there is no ``simulator_config:`` file indirection --
    the measured-detector training config inlines ``image`` / ``detector`` / ``scene``
    (or nests them under ``simulator:``). Detector carries per-channel ``gain_range``.
    """
    sim = config.get("simulator", config)
    image = sim.get("image", {"height": 256, "width": 256})
    detector_cfg = sim.get("detector", {})
    scene_cfg = sim.get("scene", {})
    shape = (int(image.get("height", 256)), int(image.get("width", 256)))
    adc_max = int(detector_cfg.get("adc_max", 4095))
    return shape, detector_cfg, scene_cfg, adc_max


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _assert_gpu(device: torch.device, require_gpu: bool, log_fn) -> None:
    """Fail loud on a silent CPU fallback when a GPU run was intended (CHANGE 5)."""
    log_fn(f"[device] using {device} (cuda_available={torch.cuda.is_available()}, "
           f"cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0})")
    if require_gpu and device.type != "cuda":
        raise RuntimeError(
            "require_gpu=True but the resolved device is not CUDA. Refusing to run on "
            "CPU -- a 40k-step run would take days. Check the GPU box / torch install."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False.")


# --------------------------------------------------------------------------- #
# Metrics + evaluator (the shared, blind log-ratio evaluator)                  #
# --------------------------------------------------------------------------- #
def intensity_match_metrics(
    preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Mean / median |logI_pred - logI_true| at GT centres, per channel (overfit diag)."""
    mask = targets["center_mask"] > 0.5
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
def evaluate(model, batch, loss_fn: SpotLoss) -> dict[str, float]:
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
def predict_dataset(model, examples: list[Example], *, adc_max=4095.0, device="cpu",
                    **predict_kwargs) -> pd.DataFrame:
    frames = [
        predict_spots(model, ex.image, image_id=ex.meta["image_id"], adc_max=adc_max,
                      device=device, **predict_kwargs)
        for ex in examples
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        return records_to_dataframe([])
    return pd.concat(frames, ignore_index=True)


@torch.no_grad()
def validation_logratio_mae(
    model, eval_examples: list[Example], *, adc_max: float,
    meta_by_image: dict[str, dict] | None = None, match_radius_px: float = 3.0,
    method: str = "greedy", predict_kwargs: dict | None = None, device="cpu",
    density_radius_px: float = 4.0, hard_snr_hi: float = 2.0, hard_density_lo: float = 6.0,
) -> dict[str, float]:
    """Log-ratio MAE on matched pairs, OVERALL and in the HARD CORNER (dim x overlap).

    Runs the SAME inference -> canonical-schema -> location-matching path the
    benchmark uses (``predict_spots`` + ``matching.match_dataset``), and (for the
    hard corner) the SAME frozen SNR / local-density features (``_features``), so the
    checkpoint chosen "best" mid-training is selected on exactly the quantity the
    benchmark reports. The hard corner: true GT ``snr in [0, hard_snr_hi)`` AND true
    ``n_neighbors >= hard_density_lo``. Any quantity with no support is ``nan``.
    """
    from spotpipe.benchmark.matching import match_dataset

    out = {
        "val_logratio_mae": float("nan"), "val_n_pairs": 0,
        "val_hard_logratio_mae": float("nan"), "val_hard_n_pairs": 0,
        "val_recall": float("nan"), "val_precision": float("nan"), "val_det_f1": float("nan"),
    }
    pred = predict_dataset(model, eval_examples, adc_max=adc_max, device=device,
                           **(predict_kwargs or {}))
    gt = (pd.concat([ex.spots for ex in eval_examples], ignore_index=True)
          if eval_examples else pred.iloc[0:0])
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 or n_pred == 0:
        return out

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
    """LR multiplier: linear warmup to 1.0 over ``warmup`` then cosine decay to ~0."""
    if warmup > 0 and step <= warmup:
        return step / float(warmup)
    denom = max(total - warmup, 1)
    progress = min(max((step - warmup) / float(denom), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _resolve_variance_warmup_steps(tcfg: dict, n_steps: int) -> int:
    if tcfg.get("variance_warmup_steps") is not None:
        return int(tcfg["variance_warmup_steps"])
    return int(float(tcfg.get("intensity_logvar_warmup_fraction", 0.3)) * n_steps)


def _resolve_ramp_steps(tcfg: dict, cur_cfg: dict, n_steps: int) -> int:
    for src in (cur_cfg, tcfg):
        if src.get("curriculum_ramp_steps") is not None:
            return max(int(src["curriculum_ramp_steps"]), 1)
    return max(int(float(cur_cfg.get("ramp_fraction", 0.6)) * n_steps), 1)


# --------------------------------------------------------------------------- #
# Selection + checkpoints                                                       #
# --------------------------------------------------------------------------- #
def _save_checkpoint(path: Path, model, config: dict) -> None:
    torch.save({"model_state": model.state_dict(), "config": config}, path)


_SELECT_TIERS = {0: "hard_corner_val_logratio_mae", 1: "overall_val_logratio_mae", 2: "val_total_loss"}


def _selection_key(row: dict, *, min_hard_pairs: int) -> tuple:
    """Best-checkpoint sort key (SMALLER is better): hard-corner MAE, then overall, then loss."""
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


def _coerce_bin_edges(edges) -> list[float]:
    return [math.inf if e is None else float(e) for e in edges]


def _save_train_state(run_dir: Path, *, model, optimizer, step, n_steps, best, history,
                      loss_curve, val_curve, saved_ckpts, logvar_on) -> None:
    tmp = run_dir / "train_state.pt.tmp"
    torch.save({
        "step": int(step), "n_steps": int(n_steps),
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "best": best, "history": history, "loss_curve": [list(x) for x in loss_curve],
        "val_curve": val_curve, "saved_ckpts": saved_ckpts, "logvar_on": bool(logvar_on),
    }, tmp)
    tmp.replace(run_dir / "train_state.pt")


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
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
        parts.append(f"hard_mae={_fmt(row['val_hard_logratio_mae'])}(n={row.get('val_hard_n_pairs', 0)})")
    if "val_det_f1" in row:
        parts.append(f"f1={_fmt(row['val_det_f1'])}")
    if "val_logI1_mae" in row:
        parts.append(f"val_logI1_mae={_fmt(row['val_logI1_mae'])}")
        parts.append(f"val_logI2_mae={_fmt(row['val_logI2_mae'])}")
    return "  ".join(parts)


def _default_num_workers() -> int:
    return max((os.cpu_count() or 2) - 2, 0)


# --------------------------------------------------------------------------- #
# Dataload-vs-compute profiler (CHANGE 5 gate: dataload must be < 20% of step)  #
# --------------------------------------------------------------------------- #
def profile_dataload(
    config: dict, *, device="cpu", n_steps: int = 60, warmup: int = 10,
    num_workers: int | None = None, log_fn=print,
) -> dict:
    """Time dataload-wait vs forward/backward compute per step; report the fraction.

    Runs a real forward+backward with the real loader so the numbers reflect the
    actual pipeline. Returns a dict incl. ``dataload_fraction``; the caller enforces
    the < 0.20 gate (STOP if not met -- do NOT launch a data-starved 40k run).
    """
    device = resolve_device(device)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    tcfg = config.get("training", {})
    shape, detector_cfg, scene_cfg, adc_max = resolve_blocks(config)
    consts = DetectorConstants.from_config(detector_cfg)
    wcfg = IntensityWindowConfig.from_config(tcfg)
    batch_size = int(tcfg.get("batch_size", 16))
    heatmap_sigma = float(tcfg.get("heatmap_sigma", 1.5))
    ramp_steps = _resolve_ramp_steps(tcfg, tcfg.get("curriculum", {}), int(tcfg.get("train_steps", 40000)))
    nw = _default_num_workers() if num_workers is None else int(num_workers)

    model = build_spot_model(config.get("model", {})).to(device)
    loss_fn = SpotLoss(tcfg.get("loss_weights"), logvar_min=float(tcfg.get("logvar_min", -10.0)),
                       logvar_max=float(tcfg.get("logvar_max", 6.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 2e-3)))
    loader = make_loader(
        scene_cfg, consts, wcfg, shape=shape, heatmap_sigma=heatmap_sigma,
        batch_size=batch_size, n_steps=n_steps, adc_max=adc_max, seed=seed,
        ramp_steps=ramp_steps, use_curriculum=True, num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(loader)
    model.train()
    dl_times, cp_times = [], []
    log_fn(f"[profile] device={device} num_workers={nw} batch_size={batch_size} "
           f"shape={shape} -- {n_steps} steps ({warmup} warmup)")
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        images, targets, _bs, _t = next(data_iter)
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        preds = model(images)
        total, _ = loss_fn(preds, targets, intensity_use_logvar=False)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if step > warmup:
            dl_times.append(t1 - t0)
            cp_times.append(t2 - t1)
    del data_iter, loader
    dl = float(np.mean(dl_times)) if dl_times else float("nan")
    cp = float(np.mean(cp_times)) if cp_times else float("nan")
    frac = dl / (dl + cp) if (dl + cp) > 0 else float("nan")
    res = {
        "device": str(device), "num_workers": nw, "batch_size": batch_size, "shape": list(shape),
        "dataload_s_per_step": dl, "compute_s_per_step": cp,
        "step_s": dl + cp, "dataload_fraction": frac,
        "steps_per_s": (1.0 / (dl + cp)) if (dl + cp) > 0 else float("nan"),
        "gate_pass": bool(frac < 0.20) if math.isfinite(frac) else False,
    }
    log_fn(f"[profile] dataload={dl*1e3:.1f} ms/step  compute={cp*1e3:.1f} ms/step  "
           f"dataload_fraction={frac*100:.1f}%  ({'PASS <20%' if res['gate_pass'] else 'FAIL >=20%'})")
    return res


# --------------------------------------------------------------------------- #
# Provenance                                                                    #
# --------------------------------------------------------------------------- #
def _detector_constants_to_meta(consts: DetectorConstants) -> dict:
    d = asdict(consts)
    d["note"] = (
        "MEASURED-detector retrain. Gains are RANDOMISED per image over gain{1,2}_range "
        "(CHANGE 1: deliberate reversal of the vendored gain-aware design, for robustness "
        "to a future PMT-voltage change). Offset / read_var / saturation_knee / "
        "excess_noise_factor / n_frames / adc_max are held at the measured values. "
        "noise_floor_sigma = sqrt(read_var)."
    )
    return d


# --------------------------------------------------------------------------- #
# Training                                                                      #
# --------------------------------------------------------------------------- #
def train(
    config: dict, *, device="cpu", out_dir: str | Path | None = None,
    fixed_train_examples: list[Example] | None = None,
    eval_examples: list[Example] | None = None, steps: int | None = None,
    curriculum_enabled: bool | None = None, resume: bool = True,
    num_workers: int | None = None, require_gpu: bool = False, log_fn=print,
) -> dict:
    """Run training and return a summary dict. See module docstring for the pipeline."""
    device = resolve_device(device)
    _assert_gpu(device, require_gpu, log_fn)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    tcfg = config.get("training", {})
    model_cfg = config.get("model", {})
    shape, detector_cfg, scene_cfg, adc_max = resolve_blocks(config)
    consts = DetectorConstants.from_config(detector_cfg)
    wcfg = IntensityWindowConfig.from_config(tcfg)

    n_steps = int(steps if steps is not None else tcfg.get("train_steps", tcfg.get("steps", 1000)))
    batch_size = int(tcfg.get("batch_size", 16))
    lr = float(tcfg.get("lr", 2e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 1.0))
    heatmap_sigma = float(tcfg.get("heatmap_sigma", 1.5))
    loss_weights = tcfg.get("loss_weights", None)
    log_every = int(tcfg.get("log_every", max(n_steps // 10, 1)))

    lr_warmup_steps = int(tcfg.get("lr_warmup_steps", 0))
    logvar_warmup_steps = _resolve_variance_warmup_steps(tcfg, n_steps)
    cur_cfg = tcfg.get("curriculum", {})
    use_curriculum = (cur_cfg.get("enabled", True) if curriculum_enabled is None
                      else bool(curriculum_enabled))
    ramp_steps = _resolve_ramp_steps(tcfg, cur_cfg, n_steps)

    eval_every = int(tcfg.get("eval_every", max(n_steps // 5, 1)))
    checkpoint_every = int(tcfg.get("checkpoint_every", 0))
    logvar_min = float(tcfg.get("logvar_min", -10.0))
    logvar_max = float(tcfg.get("logvar_max", 6.0))

    best_cfg = tcfg.get("best_checkpoint", {})
    track_best = bool(best_cfg.get("enabled", checkpoint_every > 0)) and out_dir is not None
    hard_corner_min_pairs = int(best_cfg.get("hard_corner_min_pairs",
                                             tcfg.get("hard_corner_min_pairs", 50)))

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
    snr_bins = _coerce_bin_edges(bench_cfg.get("snr_bins", [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, None]))
    density_bins = _coerce_bin_edges(bench_cfg.get("density_bins", [0.0, 1.0, 3.0, 6.0, None]))
    density_radius_px = float(bench_cfg.get("density_radius_px", 4.0))
    hard_snr_hi = float(snr_bins[1])
    hard_density_lo = float(density_bins[-2])
    val_metric_kwargs = dict(
        adc_max=adc_max, match_radius_px=match_radius_px, method=match_method,
        predict_kwargs=om_predict_kwargs, device=device, density_radius_px=density_radius_px,
        hard_snr_hi=hard_snr_hi, hard_density_lo=hard_density_lo,
    )

    model = build_spot_model(model_cfg).to(device)
    loss_fn = SpotLoss(loss_weights, logvar_min=logvar_min, logvar_max=logvar_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"[model] built ({n_params/1e6:.2f}M params)  steps={n_steps} batch={batch_size} "
           f"lr={lr} warmups(lr={lr_warmup_steps},var={logvar_warmup_steps}) ramp={ramp_steps}")

    # Fixed validation set (selection only). Prefer a persisted set for byte-identical
    # selection across runs; else build one in memory (full difficulty, hard corners).
    vcfg = tcfg.get("val", {})
    eval_seed = int(vcfg.get("seed", 12345))
    val_path = vcfg.get("path")
    if eval_examples is None and val_path:
        val_dir = Path(val_path)
        if not val_dir.is_absolute():
            val_dir = REPO_ROOT / val_dir
        if (val_dir / "manifest.json").exists():
            eval_examples = load_eval_examples(val_dir, heatmap_sigma=heatmap_sigma)
            log_fn(f"[val] loaded FIXED validation set ({len(eval_examples)} images) from {val_dir}")
    if eval_examples is None:
        eval_examples = build_eval_examples(
            scene_cfg, consts, wcfg, n_images=int(vcfg.get("n_images", 24)),
            seed=eval_seed, shape=shape, heatmap_sigma=heatmap_sigma,
            n_hard_corner=int(vcfg.get("n_hard_corner", 10)),
        )
        log_fn(f"[val] built in-memory validation set ({len(eval_examples)} images)")
    meta_by_image = {ex.meta["image_id"]: ex.meta for ex in eval_examples}
    eval_batch = _to_device(collate(eval_examples, adc_max), device)

    fixed_train_batch = None
    if fixed_train_examples is not None:
        fixed_train_batch = _to_device(collate(fixed_train_examples, adc_max), device)

    run_dir = Path(out_dir) if out_dir is not None else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)

    history: list[dict] = []
    loss_curve: list[tuple[int, float]] = []
    val_curve: list[dict] = []
    best = {"step": None, "metric": None, "value": math.inf, "key": None, "path": None}
    saved_ckpts: list[str] = []
    logvar_on = False
    start_step = 1

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
            log_fn(f"[resume] restored at step {state['step']}; continuing from {start_step}")

    # Hoist the multi-worker loader ONCE (CHANGE 5): workers persist for the whole
    # continuous pass, never respawned per phase. Skipped for the overfit path.
    loader = data_iter = None
    if fixed_train_batch is None:
        nw = _default_num_workers() if num_workers is None else int(num_workers)
        loader = make_loader(
            scene_cfg, consts, wcfg, shape=shape, heatmap_sigma=heatmap_sigma,
            batch_size=batch_size, n_steps=n_steps, adc_max=adc_max, seed=seed,
            ramp_steps=ramp_steps, use_curriculum=use_curriculum, start_step=start_step,
            num_workers=nw, pin_memory=(device.type == "cuda"),
        )
        data_iter = iter(loader)
        log_fn(f"[data] streaming loader: num_workers={nw} "
               f"(persistent={nw > 0}, pin_memory={device.type == 'cuda'})")

    dl_time_acc = cp_time_acc = 0.0
    for step in range(start_step, n_steps + 1):
        cur_lr = lr * _lr_factor(step, warmup=lr_warmup_steps, total=n_steps)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        t0 = time.perf_counter()
        if fixed_train_batch is not None:
            images, targets = fixed_train_batch
            cur_t = 1.0
        else:
            images, targets, _bstep, cur_t = next(data_iter)
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        t1 = time.perf_counter()

        use_logvar = step > logvar_warmup_steps
        if use_logvar and not logvar_on:
            log_fn(f"[schedule] step {step}: variance warmup complete -- enabling learned-logvar NLL")
            logvar_on = True

        model.train()
        preds = model(images)
        total, comps = loss_fn(preds, targets, intensity_use_logvar=use_logvar)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        t2 = time.perf_counter()
        dl_time_acc += t1 - t0
        cp_time_acc += t2 - t1

        loss_curve.append((step, float(total.detach())))
        is_final = step == n_steps
        is_eval = (eval_every > 0 and step % eval_every == 0) or is_final
        is_ckpt = checkpoint_every > 0 and ((step % checkpoint_every == 0) or is_final)
        do_eval = is_eval or is_ckpt

        row = {"step": step, "lr": cur_lr, "curriculum_t": cur_t, "use_logvar": use_logvar}
        row.update({f"train_{k}": float(v) for k, v in comps.items()})

        if do_eval:
            row.update({f"val_{k}": v for k, v in evaluate(model, eval_batch, loss_fn).items()})
            row.update(validation_logratio_mae(model, eval_examples, meta_by_image=meta_by_image,
                                                **val_metric_kwargs))
            val_curve.append({k: row.get(k) for k in (
                "step", "lr", "curriculum_t", "val_loss_total", "val_logratio_mae", "val_n_pairs",
                "val_hard_logratio_mae", "val_hard_n_pairs", "val_recall", "val_precision",
                "val_det_f1", "val_logI1_mae", "val_logI2_mae")})

        if (step % log_every == 0) or step == 1 or do_eval:
            if step > start_step:
                row["dataload_frac_sofar"] = dl_time_acc / max(dl_time_acc + cp_time_acc, 1e-9)
            history.append(row)
            log_fn(_format_row(row))

        if is_ckpt and run_dir is not None:
            ckpt_path = run_dir / f"checkpoint_step{step:05d}.pt"
            _save_checkpoint(ckpt_path, model, config)
            saved_ckpts.append(str(ckpt_path))
            if track_best:
                cand_key = _selection_key(row, min_hard_pairs=hard_corner_min_pairs)
                if best["key"] is None or cand_key < best["key"]:
                    best_path = run_dir / "best_checkpoint.pt"
                    _save_checkpoint(best_path, model, config)
                    best.update(step=step, path=str(best_path), key=cand_key,
                                metric=_SELECT_TIERS[cand_key[0]], value=float(cand_key[1]),
                                hard_n_pairs=int(row.get("val_hard_n_pairs", 0) or 0),
                                overall_val_logratio_mae=row.get("val_logratio_mae"),
                                det_f1=row.get("val_det_f1"))
                    log_fn(f"[best] step {step}: new best by {best['metric']}={best['value']:.4f} "
                           f"(hard_n={best['hard_n_pairs']}, overall={_fmt(best['overall_val_logratio_mae'])}, "
                           f"f1={_fmt(best['det_f1'])})")
            _save_train_state(run_dir, model=model, optimizer=optimizer, step=step, n_steps=n_steps,
                              best=best, history=history, loss_curve=loss_curve, val_curve=val_curve,
                              saved_ckpts=saved_ckpts, logvar_on=logvar_on)

    if loader is not None:
        del data_iter, loader

    final_eval = evaluate(model, eval_batch, loss_fn)
    final_eval.update(validation_logratio_mae(model, eval_examples, meta_by_image=meta_by_image,
                                              **val_metric_kwargs))
    if track_best and best["path"] is None and run_dir is not None:
        best_path = run_dir / "best_checkpoint.pt"
        _save_checkpoint(best_path, model, config)
        best.update(step=n_steps, path=str(best_path), metric="final", value=float("nan"))

    if run_dir is not None:
        _write_run_outputs(run_dir, config=config, history=history, loss_curve=loss_curve,
                           val_curve=val_curve, final_eval=final_eval, model=model, consts=consts,
                           shape=shape, n_steps=n_steps, best=best, saved_ckpts=saved_ckpts,
                           dataload_frac=dl_time_acc / max(dl_time_acc + cp_time_acc, 1e-9))

    return {
        "run_dir": str(run_dir) if run_dir else None, "history": history, "loss_curve": loss_curve,
        "val_curve": val_curve, "final_eval": final_eval, "best": best, "model": model,
        "consts": consts, "adc_max": adc_max, "config": config,
        "dataload_fraction": dl_time_acc / max(dl_time_acc + cp_time_acc, 1e-9),
    }


def overfit(config: dict, *, n_images: int = 6, steps: int = 300, device="cpu",
            num_workers: int = 0, log_fn=print) -> dict:
    """Overfit a tiny FIXED set: the fastest masking / target-map sanity check.

    Builds ``n_images`` images once (curriculum off, full difficulty), trains on that
    exact batch every step. Loss must collapse and predicted logI1/logI2 at GT centres
    must match truth -- if not, the loss or the target-map construction is broken.
    """
    seed = int(config.get("seed", 0))
    tcfg = config.get("training", {})
    shape, detector_cfg, scene_cfg, _ = resolve_blocks(config)
    consts = DetectorConstants.from_config(detector_cfg)
    wcfg = IntensityWindowConfig.from_config(tcfg)
    heatmap_sigma = float(tcfg.get("heatmap_sigma", 1.5))
    fixed = generate_examples(scene_cfg, consts, wcfg, n_images=n_images,
                              seed=np.random.SeedSequence([seed, 777]), shape=shape,
                              heatmap_sigma=heatmap_sigma, t=1.0, id_prefix="overfit")
    return train(config, device=device, out_dir=None, fixed_train_examples=fixed,
                 eval_examples=fixed, steps=steps, curriculum_enabled=False,
                 num_workers=num_workers, log_fn=log_fn)


# --------------------------------------------------------------------------- #
# Run outputs / provenance                                                     #
# --------------------------------------------------------------------------- #
def _write_run_outputs(out_dir, *, config, history, loss_curve, val_curve, final_eval,
                       model, consts, shape, n_steps, best, saved_ckpts, dataload_frac) -> Path:
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    tcfg = config.get("training", {})
    manifest = {
        "git_commit": _git_commit(),
        "vendored_simulator_sha": VENDORED_SIMULATOR_SHA,
        "model_label": "measured_detector_hrnet_large (NOT legacy)",
        "seed": int(config.get("seed", 0)),
        "steps": int(n_steps), "shape": [shape[0], shape[1]],
        "schedules": {
            "lr_warmup_steps": int(tcfg.get("lr_warmup_steps", 0)),
            "variance_warmup_steps": _resolve_variance_warmup_steps(tcfg, n_steps),
            "curriculum_ramp_steps": _resolve_ramp_steps(tcfg, tcfg.get("curriculum", {}), n_steps),
            "eval_every": int(tcfg.get("eval_every", 0)),
            "checkpoint_every": int(tcfg.get("checkpoint_every", 0)),
        },
        "best_checkpoint": {
            "selection_metric": ("hard-corner val_logratio_mae if hard_n_pairs >= "
                                 "hard_corner_min_pairs; else overall val_logratio_mae; else "
                                 "val_total_loss. NEVER uses benchmark/test outputs."),
            "hard_corner_min_pairs": int(tcfg.get("best_checkpoint", {}).get(
                "hard_corner_min_pairs", tcfg.get("hard_corner_min_pairs", 50))),
            "selected_step": best.get("step"), "selected_by": best.get("metric"),
            "selected_value": best.get("value"),
            "selected_hard_n_pairs": best.get("hard_n_pairs"),
            "selected_overall_val_logratio_mae": best.get("overall_val_logratio_mae"),
            "selected_det_f1": best.get("det_f1"),
            "path": Path(best["path"]).name if best.get("path") else None,
        },
        "checkpoints": [Path(p).name for p in saved_ckpts],
        "detector_constants": _detector_constants_to_meta(consts),
        "dataload_fraction": float(dataload_frac),
        "final_eval": final_eval,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as fh:
        for r in history:
            fh.write(json.dumps(r) + "\n")
    pd.DataFrame(loss_curve, columns=["step", "total"]).to_csv(run_dir / "loss_curve.csv", index=False)
    if val_curve:
        pd.DataFrame(val_curve).to_csv(run_dir / "val_curve.csv", index=False)
    _save_checkpoint(run_dir / "checkpoint.pt", model, config)
    return run_dir
