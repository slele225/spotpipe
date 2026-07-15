"""Alpha probe — the tiebreaker the bright x dense bake-off cannot settle.

WHY
---
The bright x dense probe measures per-cell log-ratio BIAS. But the project's headline is
alpha = the SLOPE of log(A2/A1) vs log(sqrt(A1)) across a size range. A constant log-ratio
offset does NOT bias alpha (it is absorbed by the intercept); only a SIZE-DEPENDENT bias
tilts it. So a head change can look worse on point bias yet be BETTER for alpha, or vice
versa. The bake-off left this untested; this probe settles it.

It renders curvature-family conditions (the SAME operating point generate.py uses: low
density, wide A1 spread, fixed PSF, constant background, zero registration, per-set
saturation-safe A1 window) at a sweep of INJECTED alpha, runs a model, matches to ground
truth, and fits alpha with the FROZEN evaluator fit (`benchmark.evaluate.fit_alpha` -- the
one function that owns the factor-of-2; reimplementing it is how you get a silently halved
slope, CLAUDE.md rule). Nothing here re-derives the convention.

Reports, per arm: recovered alpha vs injected alpha across the sweep (the MAE is the
headline number), and the alpha=0 NULL CONTROL (any |alpha| there is manufactured
curvature). Run it on both bake-off arms; the one with lower alpha-MAE and a tighter null
is the better alpha estimator, which is the actual thesis metric.

CPU-fine but GPU faster. Accepts an installed checkpoint NAME or a run-dir path.

    python scripts/alpha_probe.py --checkpoint outputs/train/headfix-INDEP --device cuda
    python scripts/alpha_probe.py --checkpoint outputs/train/headfix-DELTA --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spotpipe.benchmark.alpha import alpha_to_sim_slope
from spotpipe.benchmark.evaluate import fit_alpha
from spotpipe.benchmark.generate import (
    _BENCH_SIGMA1,
    _BENCH_SIGMA2,
    _CONSTANT_BACKGROUND,
    _CURVATURE_OPERATING_POINT,
    _CURVATURE_SCATTER_STD,
    _FIXED_PSF,
    _ZERO_REGISTRATION,
    _curvature_intensity_window,
    _deep_merge,
    load_benchmark_config,
)
from spotpipe.paths import get_paths
from spotpipe.simulator import forward_model, noise

# Injected alpha sweep. 0.0 is the NULL CONTROL (rendered with extra images).
ALPHA_SWEEP = (-0.6, -0.3, 0.0, 0.3, 0.6)
MATCH_RADIUS = 1.0 * max(_BENCH_SIGMA1, _BENCH_SIGMA2)
MIN_ALPHA_DECADES = 1.0


def _match(pred: pd.DataFrame, gt: pd.DataFrame, radius: float):
    if len(pred) == 0 or len(gt) == 0:
        return np.array([], int), np.array([], int)
    P = pred[["x", "y"]].to_numpy(float)
    G = gt[["x", "y"]].to_numpy(float)
    d = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=-1)
    used: set[int] = set()
    pairs = []
    for gi in np.argsort(d.min(axis=1)):
        for pi in np.argsort(d[gi]):
            if d[gi, pi] > radius:
                break
            if int(pi) not in used:
                used.add(int(pi)); pairs.append((int(gi), int(pi))); break
    if not pairs:
        return np.array([], int), np.array([], int)
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def render_curvature_set(base_config: dict, injected_alpha: float, n_images: int, seed: int):
    """Render a curvature set at a pinned injected alpha (same recipe as generate.py)."""
    s = alpha_to_sim_slope(injected_alpha)  # sim_log_slope = alpha / 2 (the frozen factor)
    ratio_law = {"ratio_law": {
        "alpha": {"min": 0.0, "max": 0.0},                       # sim_intercept = 0
        "beta": {"min": s, "max": s},                            # sim_log_slope = alpha/2
        "scatter_std": {"min": _CURVATURE_SCATTER_STD, "max": _CURVATURE_SCATTER_STD},
    }}
    window = _curvature_intensity_window(base_config, s, MIN_ALPHA_DECADES)

    ov = _deep_merge(_FIXED_PSF, _CONSTANT_BACKGROUND)
    ov = _deep_merge(ov, _ZERO_REGISTRATION)
    ov = _deep_merge(ov, dict(_CURVATURE_OPERATING_POINT))
    ov = _deep_merge(ov, ratio_law)
    ov = _deep_merge(ov, {"intensity": window})

    scene_cfg = _deep_merge(base_config.get("scene", {}), ov)
    shape = (int(base_config.get("image", {}).get("height", 256)),
             int(base_config.get("image", {}).get("width", 256)))
    det = noise.sample_detector_params(base_config.get("detector", {}),
                                       np.random.default_rng([seed, 314]))
    sims = []
    # SeedSequence entries must be non-negative; alpha can be negative, so offset it.
    alpha_seed = int(round(injected_alpha * 1000)) + 100000
    for i, child in enumerate(np.random.SeedSequence([seed, 271, alpha_seed]).spawn(n_images)):
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sims.append(forward_model.simulate_image(
            image_id=f"curv_a{injected_alpha}_{i}", shape=shape, scene=scene, detector=det, rng=rng))
    return sims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="installed name OR a run-dir path")
    ap.add_argument("--bench-config", default="benchmark.yaml")
    ap.add_argument("--n-images", type=int, default=15)
    ap.add_argument("--null-multiplier", type=int, default=3, help="extra images for alpha=0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from spotpipe.benchmark.infer import load_checkpoint
    from spotpipe.models.spot_model import predict_spots

    paths = get_paths()
    base_config, _ = load_benchmark_config(paths.configs / Path(args.bench_config).name)

    ck = Path(args.checkpoint)
    if ck.is_absolute() or len(ck.parts) > 1 or (paths.root / ck).is_dir():
        cdir = ck if ck.is_absolute() else (paths.root / ck)
        croot, cname = cdir.parent, cdir.name
    else:
        croot, cname = paths.checkpoints, args.checkpoint
    bundle = load_checkpoint(cname, checkpoints_root=croot, repo_root=paths.root)
    adc_max = float(base_config.get("detector", {}).get("adc_max", 4095))

    print(f"checkpoint : {bundle.name}  (sha {bundle.training_git_sha})")
    print(f"device     : {args.device}  cuda={torch.cuda.is_available()}")
    print(f"fit        : FROZEN benchmark.evaluate.fit_alpha (owns the factor of 2)")
    print(f"match      : {MATCH_RADIUS:.2f} px   scatter={_CURVATURE_SCATTER_STD}\n")

    rows = []
    for a in ALPHA_SWEEP:
        n = args.n_images * (args.null_multiplier if a == 0.0 else 1)
        sims = render_curvature_set(base_config, a, n, args.seed)
        li1, li2 = [], []
        n_gt = n_match = 0
        for sim in sims:
            pred = predict_spots(
                bundle.model, torch.from_numpy(sim.image.astype(np.float32)),
                image_id=sim.meta["image_id"], adc_max=adc_max, device=args.device,
                peak_threshold=bundle.params.peak_threshold,
                nms_kernel=bundle.params.nms_kernel, max_spots=bundle.params.max_spots)
            gi, pi = _match(pred, sim.spots, MATCH_RADIUS)
            n_gt += len(sim.spots); n_match += len(gi)
            if len(gi):
                li1.append(pred["logI1"].to_numpy(float)[pi])
                li2.append(pred["logI2"].to_numpy(float)[pi])
        L1 = np.concatenate(li1) if li1 else np.array([])
        L2 = np.concatenate(li2) if li2 else np.array([])
        ok = np.isfinite(L1) & np.isfinite(L2)
        fit = fit_alpha(L1[ok], L2[ok]) if ok.sum() >= 3 else None
        rows.append({
            "injected_alpha": a,
            "recovered_alpha": (fit.alpha if fit else np.nan),
            "alpha_se": (fit.alpha_se if fit else np.nan),
            "error": (fit.alpha - a if fit else np.nan),
            "n_matched": n_match, "recall": (n_match / n_gt if n_gt else 0.0),
        })

    df = pd.DataFrame(rows)
    print(f"{'injected':>9} {'recovered':>10} {'+/- SE':>8} {'error':>8} {'recall':>7} {'n':>7}")
    print("-" * 56)
    for _, r in df.iterrows():
        print(f"{r['injected_alpha']:>9.3f} {r['recovered_alpha']:>10.3f} {r['alpha_se']:>8.3f} "
              f"{r['error']:>8.3f} {r['recall']:>7.3f} {r['n_matched']:>7}")

    valid = df.dropna(subset=["recovered_alpha"])
    mae = float(valid["error"].abs().mean()) if len(valid) else float("nan")
    null = df[df["injected_alpha"] == 0.0].iloc[0] if (df["injected_alpha"] == 0.0).any() else None
    print("\n=== HEADLINE ===")
    print(f"alpha MAE over the sweep          : {mae:.4f}   (lower = better alpha estimator)")
    if null is not None:
        print(f"alpha=0 NULL CONTROL              : {null['recovered_alpha']:+.4f} "
              f"+/- {null['alpha_se']:.4f}   (any |alpha| here is MANUFACTURED curvature)")
    print("\nCompare arms: the LOWER alpha-MAE with a null closest to 0 is the better alpha")
    print("estimator -- which is the actual thesis metric, not the per-cell ratio bias.")

    if args.out:
        out = paths.root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
