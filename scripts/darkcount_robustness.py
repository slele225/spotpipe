"""Protein-channel dark-count robustness check (Change 8) -- SEPARATE from the
main benchmark. Generates empty two-channel fields with sparse single-pixel dark
spikes injected into ch2 (rate ~0.57% of pixels, ~one gain step above offset),
runs a vendored checkpoint's detector, and reports how many spikes are detected
as spots. Modifies nothing vendored and touches nothing in the benchmark
generator; it only measures. All paths resolve through ``spotpipe.paths``.

Usage (from repo root, in the venv)::

    python scripts/darkcount_robustness.py                    # hrnet_large, measured detector
    python scripts/darkcount_robustness.py --checkpoint hrnet_small --n-images 8
    python scripts/darkcount_robustness.py --out outputs/darkcount_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spotpipe.benchmark.darkcount_check import DarkCountConfig, run_darkcount_check
from spotpipe.benchmark.generate import load_benchmark_config
from spotpipe.benchmark.infer import is_legacy_checkpoint, load_checkpoint
from spotpipe.paths import get_paths
from spotpipe.simulator import noise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="protein-channel dark-count robustness check")
    ap.add_argument("--checkpoint", default="hrnet_large",
                    help="checkpoint name under models/checkpoints (default: hrnet_large)")
    ap.add_argument("--bench-config", default="benchmark.yaml",
                    help="benchmark yaml whose measured detector to use (default: benchmark.yaml)")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--rate", type=float, default=0.0057, help="ch2 spike rate (fraction of pixels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    paths = get_paths()
    base_config, _cfg = load_benchmark_config(paths.configs / args.bench_config)
    detector = noise.sample_detector_params(base_config.get("detector", {}), np.random.default_rng(0))
    print(f"[darkcount] measured detector: ch1 gain={detector.ch1.gain} ch2 gain={detector.ch2.gain} "
          f"offset={detector.ch2.offset} adc_max={detector.adc_max}")

    bundle = load_checkpoint(args.checkpoint, checkpoints_root=paths.checkpoints, repo_root=paths.root)
    tag = ("LEGACY reference" if is_legacy_checkpoint(bundle.training_git_sha)
           else "clean retrain / headline")
    print(f"[darkcount] checkpoint={args.checkpoint} ({tag}, "
          f"training_git_sha={bundle.training_git_sha}) "
          f"peak_threshold={bundle.params.peak_threshold} nms_kernel={bundle.params.nms_kernel}")

    dc_cfg = DarkCountConfig(height=args.height, width=args.width, n_images=args.n_images, rate=args.rate)
    stats = run_darkcount_check(bundle.model, bundle.params, detector, dc_cfg, seed=args.seed)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"checkpoint": args.checkpoint, **stats}, indent=2), encoding="utf-8")
        print(f"[darkcount] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
