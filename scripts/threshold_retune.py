"""Off-benchmark peak_threshold retune — the HONEST way to pick the detection threshold.

WHY THIS EXISTS (and why NOT threshold_sweep.py)
------------------------------------------------
`peak_threshold` is an inference parameter and the benchmark is the TEST set. Picking the
threshold because it wins on benchmark cells is contamination -- and it would be OUR method
getting tuning the baselines don't. `scripts/threshold_sweep.py` sweeps ON benchmark
conditions; it is a false-positive DIAGNOSIS tool, not the retune.

This script tunes on a VALIDATION set drawn from the TRAINING distribution
(`build_eval_examples`, the same generator the trainer's own val split uses, at full
difficulty with hard-corner coverage) at a DIFFERENT seed so nothing leaks from selection.
It never reads `data/benchmark`.

Protocol (must be applied IDENTICALLY to every method later, or the comparison is rigged):
  render training-dist val -> forward once per (image, threshold) -> match with the FROZEN
  matcher at the evaluator's gate -> aggregate recall/precision/F1 -> pick argmax F1
  (report precision at recall floors too). Write the chosen value into the checkpoint config
  with a note; do NOT silently apply it.

    python scripts/threshold_retune.py --checkpoint outputs/train/headfix40k-DELTA \
        --train-config configs/train_40k_DELTA.yaml --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spotpipe.benchmark.evaluate import detection_metrics
from spotpipe.benchmark.generate import _BENCH_SIGMA1, _BENCH_SIGMA2
from spotpipe.benchmark.matching import match_dataset
from spotpipe.paths import get_paths

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# Evaluator's own detection gate. Training val has variable PSF, but the gate is the SAME
# for every threshold, so the threshold-to-threshold comparison is valid; we report it.
MATCH_GATE_PX = 1.0 * max(_BENCH_SIGMA1, _BENCH_SIGMA2)
VAL_SEED = 999_777  # distinct from the trainer's val seed (12345) -> no selection leak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="installed name OR a run-dir path")
    ap.add_argument("--train-config", default="configs/train_40k_DELTA.yaml",
                    help="training config whose distribution the val set is drawn from")
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--recall-floor", type=float, default=0.90,
                    help="report the highest-precision threshold that still clears this recall")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from spotpipe.benchmark.infer import load_checkpoint
    from spotpipe.models.spot_model import predict_spots
    from spotpipe.training.dataset import IntensityWindowConfig, build_eval_examples
    from spotpipe.training.intensity_window import DetectorConstants

    paths = get_paths()
    train_cfg = yaml.safe_load(open(paths.root / args.train_config, encoding="utf-8"))
    sim = train_cfg["simulator"]
    shape = (int(sim.get("image", {}).get("height", 256)), int(sim.get("image", {}).get("width", 256)))
    consts = DetectorConstants.from_config(sim["detector"])
    wcfg = IntensityWindowConfig.from_config(train_cfg.get("training", {}))
    hsig = float(train_cfg["training"].get("heatmap_sigma", 1.5))

    ck = Path(args.checkpoint)
    if ck.is_absolute() or len(ck.parts) > 1 or (paths.root / ck).is_dir():
        cdir = ck if ck.is_absolute() else (paths.root / ck)
        croot, cname = cdir.parent, cdir.name
    else:
        croot, cname = paths.checkpoints, args.checkpoint
    bundle = load_checkpoint(cname, checkpoints_root=croot, repo_root=paths.root)
    adc_max = float(sim["detector"].get("adc_max", 4095))

    print(f"checkpoint : {bundle.name}  (sha {bundle.training_git_sha})")
    print(f"val set    : {args.n_images} imgs from {args.train_config} (seed {VAL_SEED}, OFF-benchmark)")
    print(f"match gate : {MATCH_GATE_PX:.2f} px (evaluator's)   shipped threshold: {bundle.params.peak_threshold}\n")

    examples = build_eval_examples(sim["scene"], consts, wcfg, n_images=args.n_images,
                                   seed=VAL_SEED, shape=shape, heatmap_sigma=hsig, id_prefix="retune")

    rows = []
    for thr in THRESHOLDS:
        gts, preds = [], []
        for ex in examples:
            p = predict_spots(bundle.model, ex.image, image_id=ex.meta["image_id"],
                              adc_max=adc_max, device=args.device, peak_threshold=float(thr),
                              nms_kernel=bundle.params.nms_kernel, max_spots=bundle.params.max_spots)
            preds.append(p)
            gts.append(ex.spots)
        gt = pd.concat(gts, ignore_index=True)
        pred = pd.concat(preds, ignore_index=True) if any(len(p) for p in preds) else pd.DataFrame(columns=gt.columns)
        dm = match_dataset(gt, pred, max_distance=MATCH_GATE_PX, method="hungarian")
        recall, precision, f1 = detection_metrics(dm.n_gt, dm.n_pred, dm.n_matched)
        rows.append({"peak_threshold": thr, "recall": recall, "precision": precision,
                     "f1": f1, "n_gt": dm.n_gt, "n_pred": dm.n_pred, "n_fp": len(dm.unmatched_pred)})

    df = pd.DataFrame(rows)
    print(f"{'thr':>5} {'recall':>8} {'precision':>10} {'f1':>8} {'n_fp':>8}")
    print("-" * 44)
    for _, r in df.iterrows():
        print(f"{r['peak_threshold']:>5.2f} {r['recall']:>8.3f} {r['precision']:>10.3f} "
              f"{r['f1']:>8.3f} {int(r['n_fp']):>8}")

    best_f1 = df.loc[df["f1"].idxmax()]
    floored = df[df["recall"] >= args.recall_floor]
    best_prec = floored.loc[floored["precision"].idxmax()] if len(floored) else None

    print("\n=== RECOMMENDATION ===")
    print(f"max-F1        : peak_threshold = {best_f1['peak_threshold']:.2f}  "
          f"(F1 {best_f1['f1']:.3f}, recall {best_f1['recall']:.3f}, prec {best_f1['precision']:.3f})")
    if best_prec is not None:
        print(f"max-prec @ recall>={args.recall_floor}: peak_threshold = {best_prec['peak_threshold']:.2f}  "
              f"(recall {best_prec['recall']:.3f}, prec {best_prec['precision']:.3f})")
    chosen = float(best_f1["peak_threshold"])
    print(f"\nCHOSEN (max-F1): {chosen:.2f}")
    print(f"APPLY it at eval time (does NOT mutate the checkpoint):")
    print(f"    spotpipe infer --checkpoint <name> --benchmark data/benchmark --peak-threshold {chosen:.2f}")
    print("Apply the SAME protocol (this script's recipe) to every baseline, or the comparison is rigged.")

    if args.out:
        out = paths.root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
