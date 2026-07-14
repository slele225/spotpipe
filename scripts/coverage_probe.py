"""Coverage probe: does the TRAINING distribution reach the BENCHMARK's (A1 x density) grid?

This is step 1 of docs/handoff_retrain_intensity_head.md -- "CONFIRM IT BEFORE SPENDING
40k STEPS". The hypothesis under test, quoted from that handoff:

    "The training distribution solves per-image intensity to keep both channels
     unclipped. That COUPLES brightness to density: a dense image is forced dim,
     because many bright spots would clip ch2. So 'bright AND dense' is not merely
     rare in training -- it is structurally unreachable."

The probe samples the training scene + per-image gains + solved intensity window through
the REAL code path (`sample_image_detector` -> `curriculum_scene_config` ->
`sample_scene_params` -> `_resolve_intensity_window` -> the forward model's own
`_sample_intensities`), reports the realised joint (A1, area-density) support, and asks
whether every benchmark cell falls inside it.

Nothing is rendered (no pixels, no model), so it runs in seconds.

Usage:
    python scripts/coverage_probe.py [--train-config configs/train.yaml]
                                     [--bench-config configs/benchmark.yaml]
                                     [--n-images 3000] [--t 1.0]

Exit code 1 if any benchmark cell falls outside the CONFIGURED training range.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _torch_stub  # noqa: F401,E402  (no-op when real torch is installed)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark.generate import (  # noqa: E402
    _bench_axis_params,
    _solve_intensity_for_snr,
    load_benchmark_config,
)
from spotpipe.simulator import forward_model  # noqa: E402
from spotpipe.training.dataset import (  # noqa: E402
    IntensityWindowConfig,
    _resolve_intensity_window,
    curriculum_scene_config,
)
from spotpipe.training.intensity_window import (  # noqa: E402
    DetectorConstants,
    sample_image_detector,
)


def sample_training_images(train_cfg: dict, n_images: int, t: float, seed: int = 0):
    """Per-image (density, gains, window, A1 draws) exactly as the trainer produces them."""
    sim_cfg = train_cfg["simulator"]
    base_scene = sim_cfg["scene"]
    shape = (int(sim_cfg.get("image", {}).get("height", 256)),
             int(sim_cfg.get("image", {}).get("width", 256)))
    consts = DetectorConstants.from_config(sim_cfg["detector"])
    wcfg = IntensityWindowConfig.from_config(train_cfg.get("training", {}))

    rows = []
    for child in np.random.SeedSequence([seed, 20260714]).spawn(n_images):
        rng = np.random.default_rng(child)
        _det, g1, g2 = sample_image_detector(rng, consts)
        scene_cfg_t = curriculum_scene_config(base_scene, t)
        scene = forward_model.sample_scene_params(scene_cfg_t, rng, shape)
        win = _resolve_intensity_window(scene, g1, g2, consts, wcfg, t)
        scene.intensity_log10_max = win["log10_max"]
        scene.intensity_log10_min = win["log10_min"]
        scene.intensity_dim_bias = win["dim_bias"]
        rows.append({
            "density": float(scene.density),
            "gain1": float(g1),
            "gain2": float(g2),
            "log10_min": float(win["log10_min"]),
            "log10_max": float(win["log10_max"]),
            "a1": np.asarray(forward_model._sample_intensities(scene, rng), float),
        })
    return rows, shape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--bench-config", default="configs/benchmark.yaml")
    ap.add_argument("--n-images", type=int, default=3000)
    ap.add_argument("--t", type=float, default=1.0,
                    help="curriculum difficulty (1.0 = full; the steady state after the ramp)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_cfg = yaml.safe_load(open(root / args.train_config, encoding="utf-8"))

    rows, shape = sample_training_images(train_cfg, args.n_images, args.t)
    dens = np.array([r["density"] for r in rows])
    lo = np.array([r["log10_min"] for r in rows])
    hi = np.array([r["log10_max"] for r in rows])
    all_a1 = np.concatenate([r["a1"] for r in rows if r["a1"].size])

    if _torch_stub.STUBBED:
        print("[note] torch not installed -- using the distribution-only stub "
              "(scripts/_torch_stub.py). No model is run; the sampling path is the real one.\n")

    # The CONFIGURED range is the support; the realised min/max of a finite sample sits
    # strictly inside it (a log-uniform draw never exactly hits its endpoint). Comparing
    # benchmark levels against the realised min would false-flag the lowest level as OOD.
    d_cfg = train_cfg["simulator"]["scene"]["density"]
    d_lo_cfg, d_hi_cfg = float(d_cfg["min"]), float(d_cfg["max"])

    print(f"=== TRAINING SUPPORT   (n={len(rows)} images, curriculum t={args.t}, shape={shape}) ===")
    print(f"config density range : [{d_lo_cfg}, {d_hi_cfg}]  spots/px")
    print(f"realised density     : [{dens.min():.5f}, {dens.max():.5f}]   median {np.median(dens):.5f}")
    print(f"realised per-spot A1 : [{all_a1.min():.1f}, {all_a1.max():.1f}] photons  "
          f"(median {np.median(all_a1):.1f}, n={all_a1.size:,})")
    print(f"per-image window     : log10_min in [{lo.min():.2f}, {lo.max():.2f}], "
          f"log10_max in [{hi.min():.2f}, {hi.max():.2f}]")

    # --- the hypothesis, in one number -------------------------------------
    med_a1 = np.array([float(np.median(r["a1"])) for r in rows if r["a1"].size])
    d_of = np.array([r["density"] for r in rows if r["a1"].size])
    corr = float(np.corrcoef(np.log10(d_of), np.log10(med_a1))[0, 1])
    print("\n=== HYPOTHESIS: is 'bright AND dense' structurally unreachable? ===")
    print(f"corr( log10 density , log10 median-A1 ) = {corr:+.4f}")
    print("  strongly NEGATIVE -> dense images are forced dim  -> CONFIRMED")
    print("  ~ZERO             -> density and brightness are drawn independently -> REFUTED")

    print(f"\n{'density bin':>24} | {'n img':>6} | {'A1 p50':>8} | {'A1 p95':>9} | {'A1 max':>9}")
    print("-" * 70)
    edges = np.quantile(dens, np.linspace(0, 1, 6))
    for i in range(len(edges) - 1):
        last = (i == len(edges) - 2)
        m = (dens >= edges[i]) & ((dens <= edges[i + 1]) if last else (dens < edges[i + 1]))
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        a1s = np.concatenate([rows[j]["a1"] for j in idx if rows[j]["a1"].size])
        print(f"{edges[i]:>10.5f}-{edges[i+1]:<12.5f} | {idx.size:>6} | {np.median(a1s):>8.1f} | "
              f"{np.quantile(a1s, 0.95):>9.1f} | {a1s.max():>9.1f}")
    print("A flat A1 profile down this column == brightness is independent of density.")

    # --- benchmark cells vs the training support ---------------------------
    base_config, bcfg = load_benchmark_config(root / args.bench_config)
    axis = _bench_axis_params(base_config)
    a1_floor, a1_ceil = float(all_a1.min()), float(all_a1.max())

    print("\n=== BENCHMARK CELLS vs TRAINING SUPPORT ===")
    print(f"training A1 support : [{a1_floor:.1f}, {a1_ceil:.0f}] photons (realised)")
    print(f"training density    : [{d_lo_cfg}, {d_hi_cfg}] spots/px (configured)")
    print(f"\n{'SNR':>6} | {'A1 (ph)':>9} | {'in A1 support?':>16} | {'% of train spots >= this':>24}")
    print("-" * 74)
    ood_snr = []
    for snr in bcfg.snr_targets:
        A = float(_solve_intensity_for_snr(float(snr), axis)["intensity"])
        ok = a1_floor <= A <= a1_ceil
        note = "OK" if ok else ("BELOW floor" if A < a1_floor else "ABOVE ceiling")
        if not ok:
            ood_snr.append((snr, A, note))
        frac = 100.0 * float((all_a1 >= A).mean())
        print(f"{snr:>6} | {A:>9.1f} | {note:>16} | {frac:>23.1f}%")

    ood_dens = [d for d in bcfg.density_levels
                if not (d_lo_cfg <= float(d) <= d_hi_cfg)]
    print(f"\ndensity levels  : {list(bcfg.density_levels)}")
    print(f"outside support : {ood_dens if ood_dens else 'none'}")

    print("\n=== VERDICT ===")
    if ood_snr or ood_dens:
        print("OUT-OF-DISTRIBUTION cells exist. Widen the TRAINING support before this grid")
        print("becomes a headline -- otherwise those cells measure coverage, not method:")
        for snr, A, note in ood_snr:
            print(f"  - SNR {snr}: needs A1 = {A:.1f} ph  ({note})")
        if ood_dens:
            print(f"  - density {ood_dens}: outside the configured training range "
                  f"[{d_lo_cfg}, {d_hi_cfg}]")
        return 1
    print("Every benchmark cell falls inside the training support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
