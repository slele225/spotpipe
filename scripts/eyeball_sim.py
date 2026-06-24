#!/usr/bin/env python
"""Render-and-eyeball the FV3000 forward model.

Generates a small handful of images across the difficulty range and writes
figures that let a human visually confirm the physics BEFORE any network exists.
Run::

    uv run python scripts/eyeball_sim.py
    uv run python scripts/eyeball_sim.py --config configs/simulator.yaml --out outputs/eyeball_sim --seed 0

Figures written to ``--out`` (see the printed summary for what each one should
show). Drawn scene params are printed alongside so they can be checked against
the plots. This script ONLY exercises the simulator -- no models/training.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from spotpipe.simulator import forward_model, noise
from spotpipe.simulator.generate_dataset import load_simulator_config
from spotpipe.simulator.noise import transfer_curve

LN10 = np.log(10.0)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def scene_with(base: dict, **ov) -> dict:
    """Copy the scene config block and pin specific params (range min==max)."""
    s = copy.deepcopy(base)
    if ov.get("density") is not None:
        s["density"] = {"min": ov["density"], "max": ov["density"]}
        s["oversample_dense_fraction"] = 0.0
    if ov.get("oversample_dense") is not None:
        s["oversample_dense_fraction"] = ov["oversample_dense"]
    rl = s.setdefault("ratio_law", {})
    if ov.get("beta") is not None:
        rl["beta"] = {"min": ov["beta"], "max": ov["beta"]}
    if ov.get("alpha") is not None:
        rl["alpha"] = {"min": ov["alpha"], "max": ov["alpha"]}
    if ov.get("scatter") is not None:
        rl["scatter_std"] = {"min": ov["scatter"], "max": ov["scatter"]}
    pp = s.setdefault("psf", {})
    if ov.get("sigma1") is not None:
        pp["sigma1"] = {"min": ov["sigma1"], "max": ov["sigma1"]}
    if ov.get("mismatch") is not None:
        pp["c2_sigma_mismatch"] = {"min": ov["mismatch"], "max": ov["mismatch"]}
    ic = s.setdefault("intensity", {})
    for key in ("dim_bias", "log10_min", "log10_max"):
        if ov.get(key) is not None:
            ic[key] = ov[key]
    if ov.get("cluster_prob") is not None:
        s.setdefault("clustering", {})["cluster_prob"] = ov["cluster_prob"]
    return s


def sim_one(scene_cfg, detector, shape, seed, image_id="img", with_diag=True):
    rng = np.random.default_rng(seed)
    scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
    sim = forward_model.simulate_image(
        image_id=image_id, shape=shape, scene=scene, detector=detector,
        rng=rng, with_diagnostics=with_diag,
    )
    return scene, sim


def scene_line(scene) -> str:
    return (
        f"n_spots={scene.n_spots} density={scene.density:.4f} "
        f"alpha={scene.alpha:+.3f} beta={scene.beta:+.3f} scatter={scene.scatter_std:.3f} "
        f"sigma1={scene.sigma1:.2f} sigma2={scene.sigma2:.2f} ({scene.clustering}) "
        f"shift1=({scene.shift1[0]:+.2f},{scene.shift1[1]:+.2f}) "
        f"shift2=({scene.shift2[0]:+.2f},{scene.shift2[1]:+.2f}) "
        f"bg1~{scene.background1['level']:.1f} bg2~{scene.background2['level']:.1f}"
    )


def peak_counts(channel: np.ndarray, xs, ys, half: int = 1) -> np.ndarray:
    """Max observed count in a (2*half+1)^2 window around each spot centre."""
    h, w = channel.shape
    out = np.empty(len(xs))
    for i, (x, y) in enumerate(zip(xs, ys)):
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(cx - half, 0), min(cx + half + 1, w)
        y0, y1 = max(cy - half, 0), min(cy + half + 1, h)
        out[i] = channel[y0:y1, x0:x1].max()
    return out


def densest_window(xs, ys, shape, win=56, radius=6.0):
    """Centre of the most crowded region (for an overlap-showing zoom crop)."""
    h, w = shape
    if len(xs) == 0:
        return w // 2, h // 2
    pts = np.column_stack([xs, ys])
    tree = cKDTree(pts)
    counts = np.array([len(tree.query_ball_point(p, radius)) for p in pts])
    cx, cy = pts[int(np.argmax(counts))]
    cx = int(np.clip(cx, win // 2, w - win // 2))
    cy = int(np.clip(cy, win // 2, h - win // 2))
    return cx, cy


def show_channel(ax, channel, vmax=None, title=""):
    vmax = vmax if vmax is not None else np.percentile(channel, 99.8)
    im = ax.imshow(channel, cmap="magma", vmin=0, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def fig_spots_and_overlap(scene_cfg, detector, shape, seed, out: Path):
    """Spots render at the right places with correct PSF width; overlaps blend."""
    sparse_cfg = scene_with(scene_cfg, density=0.0010, log10_min=2.6, log10_max=3.7,
                            dim_bias=1.0, cluster_prob=0.0)
    dense_cfg = scene_with(scene_cfg, density=0.010, cluster_prob=1.0)
    scenes = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for row, (cfg, sd, label) in enumerate(
        [(sparse_cfg, seed + 1, "sparse"), (dense_cfg, seed + 2, "dense/clustered")]
    ):
        scene, sim = sim_one(cfg, detector, shape, sd, image_id=label)
        scenes.append((label, scene))
        xs, ys = sim.diagnostics["xs"], sim.diagnostics["ys"]
        c1, c2 = sim.image[0], sim.image[1]
        vmax = max(np.percentile(c1, 99.8), np.percentile(c2, 99.8))
        show_channel(axes[row, 0], c1, vmax, f"{label} ch1  (sigma1={scene.sigma1:.2f}px)")
        show_channel(axes[row, 1], c2, vmax, f"{label} ch2  (sigma2={scene.sigma2:.2f}px)")
        for ax in (axes[row, 0], axes[row, 1]):
            ax.scatter(xs, ys, s=14, facecolors="none", edgecolors="cyan", linewidths=0.5)
        # zoom on the most crowded region to show physical blending
        cx, cy = densest_window(xs, ys, shape, win=56)
        crop = c1[cy - 28:cy + 28, cx - 28:cx + 28]
        axc = axes[row, 2]
        axc.imshow(crop, cmap="magma", vmin=0, vmax=np.percentile(crop, 99.8), origin="upper")
        sel = (np.abs(xs - cx) < 28) & (np.abs(ys - cy) < 28)
        axc.scatter(xs[sel] - (cx - 28), ys[sel] - (cy - 28), s=40,
                    facecolors="none", edgecolors="cyan", linewidths=0.8)
        axc.set_title(f"{label} ch1 zoom (overlap)", fontsize=9)
        axc.set_xticks([]); axc.set_yticks([])
    fig.suptitle("Fig 1 - spot placement, PSF width, and physical blending "
                 "(cyan = GT centres)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "01_spots_and_overlap.png", dpi=130)
    plt.close(fig)
    return scenes


def fig_channel_gain(scene_cfg, detector, shape, seed, out: Path):
    """The two channels show the expected per-channel gain difference."""
    cfg = scene_with(scene_cfg, density=0.0016, log10_min=2.3, log10_max=3.7,
                     dim_bias=1.0, cluster_prob=0.0, alpha=0.0, beta=0.0, scatter=0.05)
    scene, sim = sim_one(cfg, detector, shape, seed + 3, image_id="gain")
    c1, c2 = sim.image[0], sim.image[1]
    xs, ys = sim.diagnostics["xs"], sim.diagnostics["ys"]
    pk1 = peak_counts(c1, xs, ys) - detector.ch1.offset
    pk2 = peak_counts(c2, xs, ys) - detector.ch2.offset
    log_a1 = sim.diagnostics["log_a1"] / LN10

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), constrained_layout=True)
    vmax = max(np.percentile(c1, 99.8), np.percentile(c2, 99.8))
    im0 = show_channel(axes[0], c1, vmax, f"ch1  g={detector.ch1.gain:.1f}  (shared scale)")
    show_channel(axes[1], c2, vmax, f"ch2  g={detector.ch2.gain:.1f}  (shared scale)")
    fig.colorbar(im0, ax=axes[:2], fraction=0.025, pad=0.02, label="counts")

    sc = axes[2].scatter(pk1, pk2, c=log_a1, s=16, cmap="viridis")
    lo = max(np.min(pk1[pk1 > 0]) if np.any(pk1 > 0) else 1.0, 1.0)
    hi = max(pk1.max(), 1.0)
    gx = np.linspace(lo, hi, 50)
    ratio = detector.ch2.gain / detector.ch1.gain
    axes[2].plot(gx, ratio * gx, "k--", lw=1, label=f"slope g2/g1={ratio:.2f} (pre-sat.)")
    axes[2].plot(gx, gx, color="gray", ls=":", lw=1, label="identity")
    axes[2].set_xlabel("ch1 peak (counts - offset)")
    axes[2].set_ylabel("ch2 peak (counts - offset)")
    axes[2].set_title("per-spot peak: ch2 vs ch1", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04, label="log10 A1")
    fig.suptitle("Fig 2 - per-channel gain difference (same field; ch2 brighter, "
                 "bends as it saturates)", fontsize=11)
    fig.savefig(out / "02_channel_gain.png", dpi=130)
    plt.close(fig)
    return scene


def fig_saturation(scene_cfg, detector, shape, seed, out: Path):
    """Some bright spots reach the knee (and ch2's brightest hit 4095); some don't."""
    bright = scene_with(scene_cfg, density=0.004, log10_min=2.6, log10_max=4.0,
                        dim_bias=1.0, alpha=0.3, cluster_prob=0.0)
    mp1, mp2 = [], []
    for j in range(6):
        _, sim = sim_one(bright, detector, shape, seed + 10 + j, image_id=f"sat{j}")
        mp1.append(sim.diagnostics["mpeak1"]); mp2.append(sim.diagnostics["mpeak2"])
    mp1 = np.concatenate(mp1); mp2 = np.concatenate(mp2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ch, mp, name in [(axes[0], detector.ch1, mp1, "ch1"), (axes[1], detector.ch2, mp2, "ch2")]:
        knee = ch.saturation_knee
        grid = np.linspace(1.0, max(mp.max(), 2 * knee), 400)
        ax.plot(grid, transfer_curve(ch, grid, detector.adc_max), "k-", lw=1.4,
                label="transfer  knee*tanh(M/knee)+offset")
        sat = mp >= knee
        ax.scatter(mp[~sat], transfer_curve(ch, mp[~sat], detector.adc_max), s=10,
                   color="steelblue", alpha=0.5, label=f"below knee ({np.sum(~sat)})")
        ax.scatter(mp[sat], transfer_curve(ch, mp[sat], detector.adc_max), s=12,
                   color="crimson", alpha=0.7, label=f"reaches knee ({np.sum(sat)})")
        ax.axvline(knee, color="orange", ls="--", lw=1, label=f"knee={knee:.0f}")
        ax.axhline(detector.adc_max, color="gray", ls=":", lw=1, label="4095 hard clip")
        ax.set_xscale("log")
        ax.set_xlabel("clean gained peak signal  M = g * peak photons")
        ax.set_ylabel("output count")
        ax.set_title(f"{name}: g={ch.gain:.1f} knee={knee:.0f} offset={ch.offset:.0f}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Fig 3 - per-channel soft saturation knee (pooled over 6 bright images)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "03_saturation.png", dpi=130)
    plt.close(fig)
    # report a few examples
    print("\n[Fig 3] saturation examples (clean gained peak M vs knee):")
    print(f"  ch1 knee={detector.ch1.saturation_knee:.0f}: "
          f"{np.sum(mp1 >= detector.ch1.saturation_knee)}/{len(mp1)} spot-peaks reach it")
    print(f"  ch2 knee={detector.ch2.saturation_knee:.0f}: "
          f"{np.sum(mp2 >= detector.ch2.saturation_knee)}/{len(mp2)} spot-peaks reach it; "
          f"{np.sum(transfer_curve(detector.ch2, mp2, detector.adc_max) >= detector.adc_max)} hit the 4095 clip")


def fig_ratio_law(scene_cfg, detector, shape, seed, out: Path):
    """Recovered GT log A2 vs log A1 matches the drawn alpha, beta (beta per image)."""
    # Span the configured beta range [-0.6, 0.6] -> slopes (1+beta) ~0.4-1.6.
    betas = [-0.6, -0.3, 0.0, 0.2, 0.4, 0.6]
    alpha = 0.2
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for ax, beta in zip(axes.ravel(), betas):
        cfg = scene_with(scene_cfg, density=0.0045, log10_min=1.3, log10_max=3.9,
                         dim_bias=1.0, alpha=alpha, beta=beta, scatter=0.10, cluster_prob=0.0)
        _, sim = sim_one(cfg, detector, shape, seed + 20 + int((beta + 1) * 100), image_id="rl")
        la1 = sim.diagnostics["log_a1"]; la2 = sim.diagnostics["log_a2"]
        ax.scatter(la1, la2, s=8, alpha=0.4, color="slateblue")
        xline = np.array([la1.min(), la1.max()])
        ax.plot(xline, (1 + beta) * xline + alpha, "r-", lw=1.6,
                label=f"true: slope={1 + beta:.2f}, a={alpha:.2f}")
        ax.set_title(f"beta={beta:+.2f}  (slope 1+beta={1 + beta:.2f})", fontsize=9)
        ax.set_xlabel("log A1 (nat)"); ax.set_ylabel("log A2 (nat)")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Fig 4 - ratio law per image: log A2 = (1+beta) log A1 + alpha + noise "
                 "(incl. beta=0 and negative)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "04_ratio_law.png", dpi=130)
    plt.close(fig)


def fig_population(scene_cfg, detector, shape, seed, out: Path, n_images=30):
    """Dim tail and high-overlap regime are well populated; beta spans the range."""
    log10_a1, nn_dist, betas = [], [], []
    for j in range(n_images):
        _, sim = sim_one(scene_cfg, detector, shape, seed + 100 + j, image_id=f"pop{j}")
        log10_a1.append(sim.diagnostics["log_a1"] / LN10)
        betas.append(sim.meta["scene"]["beta"])
        xs, ys = sim.diagnostics["xs"], sim.diagnostics["ys"]
        if len(xs) > 1:
            d, _ = cKDTree(np.column_stack([xs, ys])).query(np.column_stack([xs, ys]), k=2)
            nn_dist.append(d[:, 1])
    log10_a1 = np.concatenate(log10_a1)
    nn_dist = np.concatenate(nn_dist)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].hist(log10_a1, bins=50, color="teal", alpha=0.85)
    axes[0].set_xlabel("log10 A1 (true integrated photons)")
    axes[0].set_ylabel("spots"); axes[0].set_title("intensity: dim tail over-sampled", fontsize=9)
    axes[0].axvline(scene_cfg["intensity"]["log10_min"], color="k", ls=":", lw=1)
    axes[0].axvline(scene_cfg["intensity"]["log10_max"], color="k", ls=":", lw=1)

    axes[1].hist(nn_dist, bins=np.linspace(0, 30, 60), color="indianred", alpha=0.85)
    axes[1].set_xlabel("nearest-neighbour distance (px)")
    axes[1].set_ylabel("spots")
    axes[1].set_title("overlap: many spots within ~1-3 PSF widths", fontsize=9)
    axes[1].axvline(3.0, color="k", ls=":", lw=1, label="~2 sigma")
    axes[1].legend(fontsize=7)

    axes[2].hist(betas, bins=20, color="slateblue", alpha=0.85)
    axes[2].set_xlabel("beta (per image)")
    axes[2].set_ylabel("images")
    axes[2].set_title("beta varies per image (incl. 0 / negative)", fontsize=9)
    axes[2].axvline(0.0, color="k", ls="-", lw=1)

    fig.suptitle(f"Fig 5 - population over {n_images} images "
                 "(dim tail, high-overlap, beta spread)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "05_population.png", dpi=130)
    plt.close(fig)


def fig_count_histograms(scene_cfg, detector, shape, seed, out: Path, n_images=6):
    """Observed-count histograms look like real microscopy data."""
    px1, px2 = [], []
    for j in range(n_images):
        _, sim = sim_one(scene_cfg, detector, shape, seed + 200 + j,
                         image_id=f"hist{j}", with_diag=False)
        px1.append(sim.image[0].ravel()); px2.append(sim.image[1].ravel())
    px1 = np.concatenate(px1); px2 = np.concatenate(px2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, px, ch, name in [(axes[0], px1, detector.ch1, "ch1"), (axes[1], px2, detector.ch2, "ch2")]:
        ax.hist(px, bins=np.linspace(0, 4095, 256), color="dimgray", alpha=0.9)
        ax.set_yscale("log")
        ax.axvline(ch.offset, color="green", ls="--", lw=1, label=f"offset floor ~{ch.offset:.0f}")
        ax.axvline(4095, color="red", ls=":", lw=1, label="4095 clip")
        ax.set_xlabel("observed count (12-bit)")
        ax.set_ylabel("pixels (log)")
        ax.set_title(f"{name}: offset floor, dim bulk, bright tail", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Fig 6 - observed-count histograms, pooled over {n_images} images", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "06_count_histograms.png", dpi=130)
    plt.close(fig)
    print("\n[Fig 6] count stats:")
    print(f"  ch1: floor~{detector.ch1.offset:.0f}  p50={int(np.percentile(px1,50))}  "
          f"p99.9={int(np.percentile(px1,99.9))}  max={int(px1.max())}  @4095={int((px1==4095).sum())}")
    print(f"  ch2: floor~{detector.ch2.offset:.0f}  p50={int(np.percentile(px2,50))}  "
          f"p99.9={int(np.percentile(px2,99.9))}  max={int(px2.max())}  @4095={int((px2==4095).sum())}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render-and-eyeball the FV3000 forward model.")
    parser.add_argument("--config", default="configs/simulator.yaml")
    parser.add_argument("--out", default="outputs/eyeball_sim")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_simulator_config(args.config)
    shape = (int(cfg["image"]["height"]), int(cfg["image"]["width"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Detector is the fixed instrument: sampled once, from the seed alone.
    detector = noise.sample_detector_params(cfg["detector"], np.random.default_rng(args.seed))
    scene_cfg = cfg["scene"]

    print("=" * 78)
    print(f"FV3000 eyeball  config={args.config}  seed={args.seed}  image={shape[0]}x{shape[1]}")
    print("DETECTOR (fixed instrument constants):")
    for name, ch in [("ch1", detector.ch1), ("ch2", detector.ch2)]:
        print(f"  {name}: gain={ch.gain:.2f} offset={ch.offset:.0f} F={ch.excess_noise_factor:.2f} "
              f"knee={ch.saturation_knee:.0f} floor_sigma={ch.noise_floor_sigma:.1f}")
    print(f"  n_frames={detector.n_frames} poisson_gaussian_threshold="
          f"{detector.poisson_gaussian_threshold:.0f} adc_max={detector.adc_max}")
    print("=" * 78)

    scenes1 = fig_spots_and_overlap(scene_cfg, detector, shape, args.seed, out)
    for label, sc in scenes1:
        print(f"[Fig 1] {label:16s} {scene_line(sc)}")
    gain_scene = fig_channel_gain(scene_cfg, detector, shape, args.seed, out)
    print(f"[Fig 2] gain-demo field   {scene_line(gain_scene)}")
    fig_saturation(scene_cfg, detector, shape, args.seed, out)
    fig_ratio_law(scene_cfg, detector, shape, args.seed, out)
    print("[Fig 4] ratio-law grid    beta in {-0.60,-0.30,0.00,+0.20,+0.40,+0.60}, alpha=+0.20, scatter=0.10")
    fig_population(scene_cfg, detector, shape, args.seed, out)
    fig_count_histograms(scene_cfg, detector, shape, args.seed, out)

    print("\n" + "=" * 78)
    print(f"Figures written to: {out.resolve()}")
    print("  01_spots_and_overlap.png  - spots at GT centres, PSF width, blended overlaps")
    print("  02_channel_gain.png       - same field both channels; ch2 brighter (gain), bends at saturation")
    print("  03_saturation.png         - per-channel soft knee; which spots reach it; ch2 hits 4095")
    print("  04_ratio_law.png          - log A2 vs log A1 follows drawn (1+beta) line, per-image beta")
    print("  05_population.png         - dim tail over-sampled, overlap regime populated, beta spread")
    print("  06_count_histograms.png   - offset floor, dim bulk, bright tail, 4095 clip per channel")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
