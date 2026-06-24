#!/usr/bin/env python
"""Export SYNTHETIC train/val data for Spotiflow fine-tuning (detector only).

Fine-tuning Spotiflow on spotpipe synthetic data must use a SEPARATE, freshly
generated synthetic set -- NEVER the frozen benchmark/test set and NEVER the
HRNet fixed-eval/validation set (using either would leak the test set or entangle
model selection). This script generates such a set from a simulator config and
exports it in the directory/CSV form Spotiflow training consumes.

Guards (hard-fail): the train/val output paths may not be, or live under,
``data/benchmark_test_*`` or ``data/fixed_eval``. The script refuses to write there.

Export format (per split, one pair per image)::

    <out>/images/<id>.tif    float32 detect image (default raw_max = max(ch1,ch2))
    <out>/annotations/<id>.csv   point annotations, columns: y,x  (image/array order)
    <out>/manifest.json      seed, generator config, counts, source-safety statement

NOTE ON FORMAT: Spotiflow's exact expected training layout/column names should be
confirmed against the installed version (``spotiflow-train --help`` and the docs)
before the real fine-tune run -- this writes a clear, conventional ``y,x`` point
CSV per image plus a manifest, and is intentionally a thin scaffold (the transfer
context says not to overbuild this until the training format is verified). The
forbidden-path guards and the synthetic-only generation are the load-bearing parts.

This script does NOT import or run Spotiflow; it only produces training inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark.harness import build_eval_set
from spotpipe.simulator.generate_dataset import _git_commit

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUBSTRINGS = ("benchmark_test", "fixed_eval")


def _assert_allowed_output(path: Path, label: str) -> None:
    """Refuse output paths that are (or live under) the frozen/fixed-eval sets."""
    resolved = path.resolve()
    parts = {p.lower() for p in resolved.parts}
    for bad in FORBIDDEN_SUBSTRINGS:
        if any(bad in part for part in parts):
            raise SystemExit(
                f"REFUSING to write {label} to {resolved}: path contains "
                f"'{bad}'. Fine-tune data must be a SEPARATE synthetic set, never the "
                "frozen benchmark/test set or the HRNet fixed-eval set (that would leak "
                "the test set / entangle model selection)."
            )


def _detect_image(image: np.ndarray, detect_image: str) -> np.ndarray:
    """Match the detection-time collapse so training images look like inference ones."""
    ch1 = np.asarray(image[0], dtype=np.float32)
    ch2 = np.asarray(image[1], dtype=np.float32)
    if detect_image == "raw_max":
        return np.maximum(ch1, ch2)
    if detect_image == "raw_sum":
        return ch1 + ch2
    if detect_image == "master_ch1":
        return ch1
    if detect_image == "master_ch2":
        return ch2
    raise ValueError(f"unknown detect_image {detect_image!r}")


def _export_split(eval_set, out_dir: Path, detect_image: str) -> int:
    """Write <id>.tif + <id>.csv (y,x) pairs for one split; return spot count."""
    import tifffile

    img_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    n_spots = 0
    for item in eval_set:
        detect = _detect_image(item.image, detect_image)
        tifffile.imwrite(img_dir / f"{item.image_id}.tif", detect)
        # Spotiflow point annotations are in image/array order (y = row, x = col).
        ann = pd.DataFrame({
            "y": item.gt["y"].to_numpy(dtype=float),
            "x": item.gt["x"].to_numpy(dtype=float),
        })
        ann.to_csv(ann_dir / f"{item.image_id}.csv", index=False)
        n_spots += len(ann)
    return n_spots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export synthetic Spotiflow fine-tune data.")
    parser.add_argument("--train-out", default="data/external/spotiflow_finetune_train")
    parser.add_argument("--val-out", default="data/external/spotiflow_finetune_val")
    parser.add_argument("--simulator-config", default=str(REPO_ROOT / "configs" / "simulator.yaml"))
    parser.add_argument("--detect-image", default="raw_max",
                        choices=["raw_max", "raw_sum", "master_ch1", "master_ch2"])
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args(argv)

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    _assert_allowed_output(train_out, "train set")
    _assert_allowed_output(val_out, "val set")

    with open(args.simulator_config, "r", encoding="utf-8") as fh:
        sim_cfg = yaml.safe_load(fh)

    # Distinct seeds for train vs val so they never share images. Neither shares a
    # seed with the frozen test set or the HRNet fixed-eval set.
    train_set = build_eval_set(sim_cfg, n_images=args.n_train, seed=args.seed, id_prefix="sfft_train")
    val_set = build_eval_set(sim_cfg, n_images=args.n_val, seed=args.seed + 1, id_prefix="sfft_val")

    n_train_spots = _export_split(train_set, train_out, args.detect_image)
    n_val_spots = _export_split(val_set, val_out, args.detect_image)

    manifest = {
        "purpose": "Spotiflow fine-tuning (detector only) on spotpipe SYNTHETIC data.",
        "git_commit": _git_commit(),
        "seed_train": int(args.seed),
        "seed_val": int(args.seed + 1),
        "simulator_config": str(args.simulator_config),
        "detect_image": args.detect_image,
        "n_train_images": len(train_set),
        "n_val_images": len(val_set),
        "n_train_spots": int(n_train_spots),
        "n_val_spots": int(n_val_spots),
        "annotation_format": "per-image CSV, columns y,x (image/array order)",
        "source_safety": (
            "Generated fresh from the simulator config; the frozen benchmark/test set "
            "(data/benchmark_test_*) and the HRNet fixed-eval set (data/fixed_eval) were "
            "NOT read or used in any way."
        ),
    }
    for out_dir in (train_out, val_out):
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    print(f"[finetune-data] train: {len(train_set)} images / {n_train_spots} spots -> {train_out}")
    print(f"[finetune-data] val:   {len(val_set)} images / {n_val_spots} spots -> {val_out}")
    print("[finetune-data] NOTE: verify Spotiflow's expected training layout "
          "(spotiflow-train --help) before the real fine-tune run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
