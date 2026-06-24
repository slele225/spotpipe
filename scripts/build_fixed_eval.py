#!/usr/bin/env python
"""Build a FIXED evaluation split ONCE and persist it for reuse (prompt 5b).

Phase 5b uses THREE distinct data roles:

  * training   -- on-the-fly synthetic, infinite, seeded (built each step in the loop).
  * validation -- a FIXED set used ONLY for best-checkpoint selection. Built ONCE here
                  and reused by BOTH the small and large runs, so the two models are
                  selected on byte-identical data.
  * test       -- a SEPARATE frozen set (different seed) used ONLY for final reporting.
                  Built by a different prompt; this script can build it too via
                  ``--split test --seed <different>`` if asked.

The set is written in the ``generate_dataset`` on-disk layout (manifest + images/ +
spots/ + meta/), readable by BOTH the training-side loader and the benchmark harness.

Crucially, the detector is sampled the SAME way the training driver samples it
(``_build_detector(detector_cfg, detector_seed)``) so the eval images share the run's
ONE fixed instrument (the same PMT gains / offsets / saturation). Keep ``--detector-seed``
equal to the runs' ``seed`` (both runs use 0).

Usage::

    uv run python scripts/build_fixed_eval.py \
        --config experiments/2026-06-23_hrnet_small/config.yaml \
        --out data/fixed_eval/val --split val --seed 70001 --n-images 32 --n-hard-corner 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable straight from a fresh checkout too (no sys.path hacks for shared code).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark.features import attach_features
from spotpipe.training.dataset import build_eval_examples, write_eval_dir
from spotpipe.training.train import _build_detector, load_train_config, resolve_blocks

REPO_ROOT = Path(__file__).resolve().parents[1]


def _hard_corner_report(examples, *, density_radius_px, hard_snr_hi, hard_density_lo) -> str:
    """Count GT spots in the hard corner so we can sanity-check selection stability."""
    import pandas as pd

    meta_by_image = {ex.meta["image_id"]: ex.meta for ex in examples}
    gt = pd.concat([ex.spots for ex in examples], ignore_index=True)
    gt = attach_features(gt, meta_by_image, density_radius_px=density_radius_px)
    snr = gt["snr"].to_numpy(float)
    nbr = gt["n_neighbors"].to_numpy(float)
    hard = (snr >= 0) & (snr < hard_snr_hi) & (nbr >= hard_density_lo)
    return (
        f"{int(hard.sum())} GT spots in the hard corner "
        f"(SNR in [0,{hard_snr_hi:g}) AND n_neighbors >= {hard_density_lo:g}) "
        f"out of {len(gt)} total GT spots"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a FIXED eval split (val or test) once.")
    p.add_argument("--config", default=str(REPO_ROOT / "experiments" / "2026-06-23_hrnet_small" / "config.yaml"),
                   help="a training/experiment config (for scene/detector/image + heatmap_sigma)")
    p.add_argument("--out", required=True, help="output directory for the eval split")
    p.add_argument("--split", default="val", choices=["val", "test"], help="data role label")
    p.add_argument("--seed", type=int, default=70001, help="DATA seed for this split (val != test!)")
    p.add_argument("--detector-seed", type=int, default=None,
                   help="detector seed; defaults to the config's run seed so the eval "
                        "instrument matches training")
    p.add_argument("--n-images", type=int, default=32)
    p.add_argument("--n-hard-corner", type=int, default=10,
                   help="independent dim x high-overlap images to over-sample the hard corner")
    args = p.parse_args(argv)

    config = load_train_config(args.config)
    shape, detector_cfg, scene_cfg, _adc = resolve_blocks(config)
    tcfg = config.get("training", {})
    heatmap_sigma = float(tcfg.get("heatmap_sigma", 1.5))
    detector_seed = args.detector_seed if args.detector_seed is not None else int(config.get("seed", 0))
    detector = _build_detector(detector_cfg, detector_seed)

    bench_cfg = config.get("benchmark", {})
    snr_bins = bench_cfg.get("snr_bins", [0.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf")])
    density_bins = bench_cfg.get("density_bins", [0.0, 1.0, 3.0, 6.0, float("inf")])
    density_radius_px = float(bench_cfg.get("density_radius_px", 4.0))
    hard_snr_hi = float(snr_bins[1])
    hard_density_lo = float(density_bins[-2])

    print(f"[build-eval] split={args.split} seed={args.seed} detector_seed={detector_seed} "
          f"shape={shape} n_images={args.n_images} n_hard_corner={args.n_hard_corner}")

    examples = build_eval_examples(
        scene_cfg, detector, n_images=args.n_images, seed=args.seed, shape=shape,
        heatmap_sigma=heatmap_sigma, n_hard_corner=args.n_hard_corner, id_prefix=args.split,
    )

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    manifest = write_eval_dir(
        examples, out_dir, detector=detector, split=args.split, seed=args.seed,
        shape=shape, heatmap_sigma=heatmap_sigma, scene_config=scene_cfg,
        extra_manifest={
            "data_role": args.split,
            "detector_seed": int(detector_seed),
            "purpose": (
                "FIXED validation set for best-checkpoint selection (prompt 5b)."
                if args.split == "val" else
                "FROZEN test set for final reporting only (never used for selection)."
            ),
            "hard_corner_def": f"true SNR in [0,{hard_snr_hi:g}) AND true n_neighbors >= {hard_density_lo:g}",
        },
    )

    print(f"[build-eval] wrote {manifest['n_images']} images -> {out_dir}")
    print(f"[build-eval] git_commit={manifest['git_commit']}")
    print(f"[build-eval] {_hard_corner_report(examples, density_radius_px=density_radius_px, hard_snr_hi=hard_snr_hi, hard_density_lo=hard_density_lo)}")
    if args.split == "val":
        print("[build-eval] NOTE: the frozen TEST set must use a DIFFERENT --seed than this "
              "val set so selection-on-val / reporting-on-test do not leak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
