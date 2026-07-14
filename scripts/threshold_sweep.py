"""Precision diagnosis: sweep peak_threshold x nms_kernel and CLASSIFY the false positives.

RESULTS_hrnet_large_measured.md sec.2: on sparse fields the model over-detects badly --
at snr=5 / density=0.002 it emits 11,769 predictions for 6,550 true spots (5,321 FPs,
81% of the true count). Precision is worst in the MIDDLE of the SNR range and at LOW
density, which is not a "dim spots are hard" story.

The dark-count hypothesis is DEAD before we start: BENCH_MANIFEST lists
`protein_pmt_dark_counts` under `known_unmodeled_features` -- those spikes are not in
these pixels. So the FPs are the model or the decode parameters. Two remaining causes,
and they want opposite fixes:

  (1) DUPLICATES / peak splitting. One true spot fires two peaks that survive NMS. The
      Hungarian matcher pairs 1:1, so the second copy is scored as a false positive.
      SIGNATURE: FPs sit CLOSE to a true spot -- just outside the 1.68 px match gate.
      FIX: bigger nms_kernel. Raising the threshold would NOT help (the duplicate is as
      confident as the spot it duplicates) and would cost recall.

  (2) HALLUCINATIONS on empty background. The peak threshold (0.3) is too permissive and
      the model fires on background noise -- there is a lot of empty field on a sparse
      image to fire into, which is exactly why this is worst at LOW density.
      SIGNATURE: FPs sit FAR from any true spot, and they are LOW-CONFIDENCE.
      FIX: higher peak_threshold. A bigger NMS kernel would do nothing.

The `fp_near_frac` column (fraction of FPs within 2x the match gate of a true spot) is
the discriminator. The forward pass is run ONCE and every (threshold, nms) pair is
decoded from the cached predictions -- the decode is cheap, the conv is not.

NOTE the tuning-honesty rule: peak_threshold is an INFERENCE parameter carried in the
checkpoint config, and the benchmark is a TEST set. Anything learned here must be applied
to ALL methods' tuning policy, and the retuned value must be recorded in the checkpoint
config with a note -- not silently applied to a favourable cell.

Usage:
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --condition snr=5_density=0.002 --n-images 25
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

from spotpipe.benchmark import infer
from spotpipe.benchmark.evaluate import load_benchmark_info
from spotpipe.benchmark.matching import match_dataset
from spotpipe.models.spot_model import normalize_counts
from spotpipe.paths import get_paths
from spotpipe.schema import SCHEMA_COLUMNS

_DEFAULT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_DEFAULT_NMS = [3, 5, 7]


def _load_condition(cdir: Path, n_images: int):
    with open(cdir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    images, gts = [], []
    for im in meta.get("images", [])[:n_images]:
        images.append((str(im["image_id"]),
                       tifffile.imread(cdir / im["image_file"]).astype(np.float32)))
        gt = pd.read_csv(cdir / im["ground_truth_file"])
        gt["image_id"] = str(im["image_id"])
        gts.append(gt)
    return images, pd.concat(gts, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="hrnet_large_measured")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--family", default="snr_density")
    ap.add_argument("--condition", default="snr=5_density=0.002",
                    help="the worst-precision cell (default)")
    ap.add_argument("--also", nargs="*", default=["snr=15_density=0.0006"],
                    help="extra cells to confirm the fix generalises")
    ap.add_argument("--n-images", type=int, default=25)
    ap.add_argument("--thresholds", type=float, nargs="*", default=_DEFAULT_THRESHOLDS)
    ap.add_argument("--nms", type=int, nargs="*", default=_DEFAULT_NMS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    paths = get_paths()
    bench_root = Path(args.benchmark) if args.benchmark else paths.root / "data" / "benchmark"
    info = load_benchmark_info(bench_root)
    gate = info.match_distance_px()          # the SAME 1.68 px gate the evaluator uses

    bundle = infer.load_checkpoint(args.checkpoint, checkpoints_root=paths.checkpoints,
                                   repo_root=paths.root)
    base = bundle.params
    print(f"[sweep] checkpoint={args.checkpoint} "
          f"(shipped: peak_threshold={base.peak_threshold}, nms_kernel={base.nms_kernel})")
    print(f"[sweep] match gate = {gate:.3f} px (evaluator's own)\n")

    rows = []
    for cond in [args.condition, *args.also]:
        cdir = bench_root / args.family / cond
        if not cdir.is_dir():
            print(f"[sweep] skip (missing): {cond}")
            continue
        images, gt = _load_condition(cdir, args.n_images)
        print(f"[sweep] {cond}: {len(images)} images, {len(gt)} GT spots -- forward pass...")

        # ONE forward pass; every (threshold, nms) pair is decoded from the cache.
        cache = []      # (preds, batch_ids)
        for i in range(0, len(images), args.batch_size):
            chunk = images[i: i + args.batch_size]
            batch = torch.stack([torch.from_numpy(a) for _, a in chunk], dim=0)
            with torch.no_grad():
                preds = bundle.model(normalize_counts(batch, base.adc_max))
            cache.append((preds, [iid for iid, _ in chunk]))

        for thr in args.thresholds:
            for nms in args.nms:
                p = replace(base, peak_threshold=float(thr), nms_kernel=int(nms))
                recs = []
                for preds, ids in cache:
                    for j, image_id in enumerate(ids):
                        recs.extend(infer._decode_image(preds, j, image_id, p))
                pred = pd.DataFrame([r.__dict__ for r in recs], columns=list(SCHEMA_COLUMNS))

                dm = match_dataset(gt, pred, max_distance=gate, method="hungarian")
                n_tp = len(dm.gt_matched)
                n_fp = len(dm.unmatched_pred)
                n_fn = len(dm.unmatched_gt)
                recall = n_tp / max(n_tp + n_fn, 1)
                precision = n_tp / max(n_tp + n_fp, 1)
                f1 = 2 * recall * precision / max(recall + precision, 1e-9)

                # THE DISCRIMINATOR: how far are the FPs from the nearest TRUE spot?
                # Close  => duplicates (NMS). Far => hallucinations (threshold).
                near = np.nan
                fp_conf = np.nan
                if n_fp:
                    fp = dm.unmatched_pred
                    d_min = []
                    for image_id, g in fp.groupby("image_id"):
                        tgt = gt[gt["image_id"] == image_id]
                        if tgt.empty:
                            d_min.extend([np.inf] * len(g))
                            continue
                        d = np.hypot(g["x"].to_numpy()[:, None] - tgt["x"].to_numpy()[None, :],
                                     g["y"].to_numpy()[:, None] - tgt["y"].to_numpy()[None, :])
                        d_min.extend(d.min(axis=1).tolist())
                    d_min = np.asarray(d_min, float)
                    near = float((d_min < 2.0 * gate).mean())
                    fp_conf = float(fp["p_detect"].mean()) if "p_detect" in fp else np.nan

                rows.append({
                    "condition": cond, "peak_threshold": thr, "nms_kernel": nms,
                    "n_gt": len(gt), "n_pred": len(pred), "n_tp": n_tp, "n_fp": n_fp,
                    "n_fn": n_fn, "recall": recall, "precision": precision, "f1": f1,
                    "fp_near_frac": near, "fp_mean_p_detect": fp_conf,
                    "shipped": (thr == base.peak_threshold and nms == base.nms_kernel),
                })
                flag = "  <- shipped" if rows[-1]["shipped"] else ""
                print(f"  thr={thr:.2f} nms={nms}  R={recall:.3f} P={precision:.3f} "
                      f"F1={f1:.3f}  FP={n_fp:>5}  fp_near={near:.2f}{flag}")
        print()

    if not rows:
        return 2
    out = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else paths.root / "results" / "threshold_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("=" * 96)
    print("BEST F1 PER CONDITION")
    print("=" * 96)
    for cond, sub in out.groupby("condition"):
        best = sub.loc[sub["f1"].idxmax()]
        ship = sub[sub["shipped"]]
        print(f"\n{cond}")
        if len(ship):
            s = ship.iloc[0]
            print(f"  shipped (thr={s.peak_threshold}, nms={s.nms_kernel}): "
                  f"R={s.recall:.3f} P={s.precision:.3f} F1={s.f1:.3f}  FP={int(s.n_fp)}")
        print(f"  best    (thr={best.peak_threshold}, nms={best.nms_kernel}): "
              f"R={best.recall:.3f} P={best.precision:.3f} F1={best.f1:.3f}  FP={int(best.n_fp)}")

    print("\nREAD IT LIKE THIS:")
    print("  * fp_near_frac HIGH (>~0.5) and FPs fall with bigger nms_kernel")
    print("      -> DUPLICATE PEAKS. Fix nms_kernel. Raising the threshold just costs recall.")
    print("  * fp_near_frac LOW and FPs fall with higher peak_threshold at little recall cost")
    print("      -> BACKGROUND HALLUCINATIONS. Raise peak_threshold (retune, honestly, for all).")
    print(f"\n[sweep] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
