"""Bright x dense intensity probe — does the intensity head under-read BRIGHT spots?

WHY
---
`docs/coverage_probe_findings.md` killed the COVERAGE explanation for the intensity-head
collapse (model log-ratio bias -1.10 in bright+dense, while a perfect detector reading the
same pixels gets -0.03). Detector effects, crowding, soft-knee compression and training
coverage are now ALL ruled out. The last suspect is RARITY, not unreachability:

    `full_dim_bias = 1.6` deliberately over-samples the dim tail of the per-image
    intensity window, so only ~5% of trained spots exceed ~1,000 photons. A Gaussian-NLL
    intensity head fed a dim-biased sampler may simply regress toward its mean at the
    bright end.

That is a HYPOTHESIS. This script is the instrument that tests it. It measures per-spot
log-intensity bias on a pinned (brightness x density) grid, on MATCHED spots only, so
detection quality cannot contaminate the intensity number (a detection failure shows up as
low recall, never as bias).

HOW IT IS USED
--------------
1. Run on the CURRENT checkpoint FIRST. It must REPRODUCE the known defect (log-ratio bias
   ~= -0.92 at snr=10/density=0.012 and ~= -1.10 at snr=15/density=0.012, per
   results/RESULTS_hrnet_large_measured.md). If it does not, the PROBE is wrong -- fix the
   probe, not the model. Do not proceed until it reproduces.
2. Then run on the two short A/B arms (full_dim_bias 1.6 vs 1.0). If the bright+dense bias
   collapses toward the dim+dense value in the flat arm, RARITY IS CONFIRMED and it
   dictates the retrain's sampler. If it does not move, rarity is refuted too and the cause
   is in the loss/head. Either way, do NOT spend 40k steps until this reads.

The grid deliberately spans the OLD (v2) bright cells, which the v3 benchmark no longer
contains. That is the point: this is a DIAGNOSTIC, not a headline benchmark.

Conventions are BORROWED from the frozen benchmark layer, never re-derived here: FIXED PSF
(1.4 / 1.68), CONSTANT 2-photon background, ZERO registration shift, the neutral
zero-scatter ratio law (A2 == A1), and the evaluator's match gate of 1.0 x max(sigma).

Usage:
    python scripts/bright_dense_probe.py --checkpoint hrnet_large_measured --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark.generate import (
    _BENCH_BACKGROUND_PHOTONS,
    _BENCH_SIGMA1,
    _BENCH_SIGMA2,
    _CONSTANT_BACKGROUND,
    _FIXED_PSF,
    _NEUTRAL_RATIO_LAW,
    _ZERO_REGISTRATION,
    _bench_axis_params,
    _constant_density_override,
    _deep_merge,
    _solve_intensity_for_snr,
    load_benchmark_config,
)
from spotpipe.paths import get_paths
from spotpipe.simulator import forward_model, noise

# NOTE: `load_checkpoint` / `predict_spots` are imported LAZILY inside main(). They drag in
# the model stack (and thus a real torch), whereas the rendering + matching functions below
# are pure numpy/pandas. Keeping the import lazy lets the golden tests exercise the
# measurement logic with no GPU and no trained checkpoint -- see
# tests/test_bright_dense_probe.py, which is what makes a number this script prints
# trustworthy in the first place.

# SNR 10 / 15 are the v2 cells where the defect was characterised (625 / 1,365 photons).
# SNR 2 / 3 are the TOP of the v3 grid (43 / 78 photons) and act as the in-range control.
DIAG_SNR = (2.0, 3.0, 5.0, 10.0, 15.0)
# Sparse is the control column: the defect is bright AND DENSE, so sparse must stay clean.
DIAG_DENSITY = (0.0006, 0.012)


def render_cell(base_config: dict, A: float, density: float, n_images: int, seed: int):
    """Render a pinned constant-intensity, constant-density cell (benchmark conventions)."""
    log10_A = float(np.log10(A))
    override = _deep_merge(_FIXED_PSF, _CONSTANT_BACKGROUND)
    override = _deep_merge(override, _ZERO_REGISTRATION)
    override = _deep_merge(override, {
        "intensity": {"log10_min": log10_A, "log10_max": log10_A, "dim_bias": 1.0}})
    override = _deep_merge(override, _constant_density_override(density))
    override = _deep_merge(override, _NEUTRAL_RATIO_LAW)

    scene_cfg = _deep_merge(base_config.get("scene", {}), override)
    shape = (int(base_config.get("image", {}).get("height", 256)),
             int(base_config.get("image", {}).get("width", 256)))
    detector = noise.sample_detector_params(
        base_config.get("detector", {}), np.random.default_rng([seed, 1010]))

    sims = []
    for i, child in enumerate(np.random.SeedSequence([seed, 777]).spawn(n_images)):
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sims.append(forward_model.simulate_image(
            image_id=f"probe_s{seed}_{i}", shape=shape, scene=scene,
            detector=detector, rng=rng))
    return sims, detector


def match_and_bias(pred: pd.DataFrame, gt: pd.DataFrame, radius: float) -> dict:
    """Greedy nearest-neighbour match within ``radius``; log-intensity bias on matches only.

    The schema is FROZEN and shared: ground truth and predictions carry the SAME
    ``logI1`` / ``logI2`` / ``log_ratio`` columns, so bias is a plain difference. Each
    prediction can satisfy at most ONE ground-truth spot -- otherwise a single lucky
    detection in a crowded field would inflate recall and skew the intensity mean.
    """
    if len(pred) == 0 or len(gt) == 0:
        return {"n_matched": 0, "recall": 0.0,
                **{f"{c}_{s}": np.nan
                   for c in ("logI1", "logI2", "log_ratio") for s in ("bias", "rmse")}}

    P = pred[["x", "y"]].to_numpy(float)
    G = gt[["x", "y"]].to_numpy(float)
    d = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=-1)

    used_p: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for gi in np.argsort(d.min(axis=1)):          # easiest ground-truth spots first
        for pi in np.argsort(d[gi]):
            if d[gi, pi] > radius:
                break
            if int(pi) not in used_p:
                used_p.add(int(pi))
                pairs.append((int(gi), int(pi)))
                break
    if not pairs:
        return {"n_matched": 0, "recall": 0.0,
                **{f"{c}_{s}": np.nan
                   for c in ("logI1", "logI2", "log_ratio") for s in ("bias", "rmse")}}

    gi = np.array([p[0] for p in pairs])
    pi = np.array([p[1] for p in pairs])
    out: dict = {"n_matched": len(pairs), "recall": len(pairs) / len(gt)}

    for col in ("logI1", "logI2", "log_ratio"):
        t = gt[col].to_numpy(float)[gi]
        p = pred[col].to_numpy(float)[pi]
        ok = np.isfinite(t) & np.isfinite(p)
        if ok.any():
            out[f"{col}_bias"] = float(np.mean(p[ok] - t[ok]))
            out[f"{col}_rmse"] = float(np.sqrt(np.mean((p[ok] - t[ok]) ** 2)))
        else:
            out[f"{col}_bias"] = np.nan
            out[f"{col}_rmse"] = np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="an installed checkpoint NAME (under models/checkpoints/) OR a path "
                         "to a training run dir (e.g. outputs/train/rarity-probe-A). A run dir "
                         "already holds best_checkpoint.pt + manifest.json, which is exactly "
                         "what the loader needs -- so the probe arms can be read out without "
                         "installing anything as a checkpoint.")
    ap.add_argument("--bench-config", default="benchmark.yaml",
                    help="supplies the FIXED PSF + measured detector (NOT the grid)")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-threshold", type=float, default=None,
                    help="override the checkpoint's threshold (default: the checkpoint's own)")
    ap.add_argument("--out", default=None, help="also write the table as CSV (repo-relative)")
    args = ap.parse_args()

    import torch
    from spotpipe.benchmark.infer import load_checkpoint
    from spotpipe.models.spot_model import predict_spots

    paths = get_paths()
    base_config, _ = load_benchmark_config(paths.configs / Path(args.bench_config).name)
    axis = _bench_axis_params(base_config)

    # Accept either an installed checkpoint name or a run dir. A run dir is the normal case
    # for the probe arms -- they are DIAGNOSTIC models and must never be installed as
    # checkpoints (PROJECT_STATE: a checkpoint is PRECIOUS and carries provenance).
    ckpt_arg = Path(args.checkpoint)
    if ckpt_arg.is_absolute() or len(ckpt_arg.parts) > 1 or (paths.root / ckpt_arg).is_dir():
        ckpt_dir = ckpt_arg if ckpt_arg.is_absolute() else (paths.root / ckpt_arg)
        ckpt_root, ckpt_name = ckpt_dir.parent, ckpt_dir.name
    else:
        ckpt_root, ckpt_name = paths.checkpoints, args.checkpoint

    bundle = load_checkpoint(ckpt_name, checkpoints_root=ckpt_root, repo_root=paths.root)
    thr = args.peak_threshold if args.peak_threshold is not None else bundle.params.peak_threshold
    match_radius = 1.0 * max(_BENCH_SIGMA1, _BENCH_SIGMA2)   # the frozen evaluator gate
    adc_max = float(base_config.get("detector", {}).get("adc_max", 4095))

    print(f"checkpoint     : {bundle.name}  (training sha {bundle.training_git_sha})")
    print(f"device         : {args.device}   cuda_available={torch.cuda.is_available()}")
    print(f"PSF            : sigma1={_BENCH_SIGMA1} sigma2={_BENCH_SIGMA2}   "
          f"bg={_BENCH_BACKGROUND_PHOTONS} ph   match radius={match_radius:.2f} px")
    print(f"decode         : peak_threshold={thr} nms={bundle.params.nms_kernel} "
          f"max_spots={bundle.params.max_spots}")
    print(f"images per cell: {args.n_images}\n")

    rows = []
    for snr in DIAG_SNR:
        A = float(_solve_intensity_for_snr(float(snr), axis)["intensity"])
        for dens in DIAG_DENSITY:
            sims, _det = render_cell(base_config, A, dens, args.n_images, args.seed)
            preds, gts = [], []
            for sim in sims:
                preds.append(predict_spots(
                    bundle.model, torch.from_numpy(sim.image.astype(np.float32)),
                    image_id=sim.meta["image_id"], adc_max=adc_max, device=args.device,
                    peak_threshold=thr, nms_kernel=bundle.params.nms_kernel,
                    max_spots=bundle.params.max_spots))
                gts.append(sim.spots)
            pred = pd.concat(preds, ignore_index=True)
            gt = pd.concat(gts, ignore_index=True)
            rows.append({"snr": snr, "A1_photons": round(A, 1), "density": dens,
                         "n_gt": len(gt), **match_and_bias(pred, gt, match_radius)})

    df = pd.DataFrame(rows)
    print(f"{'SNR':>5} {'A1 ph':>8} {'density':>8} {'recall':>7} "
          f"{'logI1_bias':>11} {'logI2_bias':>11} {'logratio_bias':>14}")
    print("-" * 72)
    for _, r in df.iterrows():
        print(f"{r['snr']:>5} {r['A1_photons']:>8.1f} {r['density']:>8} {r['recall']:>7.3f} "
              f"{r['logI1_bias']:>11.3f} {r['logI2_bias']:>11.3f} {r['log_ratio_bias']:>14.3f}")

    d_hi = max(DIAG_DENSITY)
    dense = df[df["density"] == d_hi]
    bright = float(dense[dense["snr"] >= 10.0]["log_ratio_bias"].mean())
    dim = float(dense[dense["snr"] <= 3.0]["log_ratio_bias"].mean())
    print("\n=== READOUT ===")
    print(f"bright+dense (SNR >= 10, density {d_hi}) mean log-ratio bias : {bright:+.3f}")
    print(f"dim+dense    (SNR <=  3, density {d_hi}) mean log-ratio bias : {dim:+.3f}")
    print(f"gap (this IS the defect)                                    : {bright - dim:+.3f}")
    print("\nCURRENT checkpoint: must reproduce ~-0.9 to -1.1 in bright+dense, else the PROBE")
    print("is wrong -- fix the probe, not the model.")
    print("FLAT arm (dim_bias 1.0): a collapse of the gap toward 0 == RARITY CONFIRMED.")

    if args.out:
        out = paths.root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
