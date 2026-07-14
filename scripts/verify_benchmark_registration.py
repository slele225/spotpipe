"""Verify ON DISK that the generated benchmark has ZERO registration shift.

`tests/test_benchmark_registration.py` guards the *override* (scene-level). This
script checks the *actual generated pixels*: for a bright, sparse condition it
fits each GT spot's photometric centre in BOTH channels and reports the residual
(fit centre - GT coordinate).

Expected outcome:

* FIXED (max_px = 0.0): residual mean ~ 0 and sd well under ~0.15 px, and the
  per-image mean residual is ~0 for every image (photon noise only).
* BROKEN (max_px = 1.0, the forward-model default): per-image mean residual is a
  CONSTANT nonzero offset drawn from U(-1, 1) -- so the pooled sd approaches
  1/sqrt(3) = 0.577 px per axis and the per-image means scatter over [-1, 1].
  ch1 and ch2 shift independently, so their per-image offsets also disagree.

The discriminator is the PER-IMAGE MEAN, not the pooled sd: a constant offset per
image is the signature of a registration shift; photon-limited localization error
averages to zero within an image.

Usage:
    python scripts/verify_benchmark_registration.py
    python scripts/verify_benchmark_registration.py --condition snr=20_density=0.0006 --n-images 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from spotpipe.paths import get_paths

# Discriminating thresholds (px).
_SD_FAIL = 0.30           # pooled per-axis sd above this => something is shifting
_IMAGE_MEAN_FAIL = 0.20   # |per-image mean residual| above this => constant offset
_U_SD = 1.0 / np.sqrt(3)  # 0.577 -- the sd of U(-1,1), the broken-case signature


def _centroid(img: np.ndarray, x: float, y: float, half: int) -> tuple[float, float, float]:
    """Background-subtracted intensity centroid in a (2*half+1)^2 window at (x, y).

    Returns (cx, cy, flux). Background = the window's median (robust to the spot).
    """
    h, w = img.shape
    x0, x1 = int(round(x)) - half, int(round(x)) + half + 1
    y0, y1 = int(round(y)) - half, int(round(y)) + half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return np.nan, np.nan, np.nan          # too close to the edge; skip

    win = img[y0:y1, x0:x1].astype(np.float64)
    win = win - np.median(win)
    win[win < 0.0] = 0.0
    flux = win.sum()
    if flux <= 0.0:
        return np.nan, np.nan, np.nan

    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    cx = float((win.sum(axis=0) * xs).sum() / flux)
    cy = float((win.sum(axis=1) * ys).sum() / flux)
    return cx, cy, float(flux)


def _isolated(gt: pd.DataFrame, min_sep: float) -> pd.DataFrame:
    """Keep only spots whose nearest neighbour is >= min_sep px away.

    A neighbour inside the centroid window drags the centroid and would masquerade
    as a shift, so crowded spots are excluded -- this measures registration, not
    crowding.
    """
    keep = []
    for _, img_gt in gt.groupby("image_id"):
        pts = img_gt[["x", "y"]].to_numpy()
        if len(pts) == 1:
            keep.append(img_gt)
            continue
        d = np.hypot(pts[:, 0][:, None] - pts[:, 0][None, :],
                     pts[:, 1][:, None] - pts[:, 1][None, :])
        np.fill_diagonal(d, np.inf)
        keep.append(img_gt[d.min(axis=1) >= min_sep])
    return pd.concat(keep) if keep else gt.iloc[:0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=None, help="benchmark root (default: data/benchmark/)")
    ap.add_argument("--family", default="snr_density")
    ap.add_argument("--condition", default=None,
                    help="condition dir name (default: the brightest, sparsest one)")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--half-window", type=int, default=4, help="centroid half-window (px)")
    ap.add_argument("--min-sep", type=float, default=12.0, help="isolation radius (px)")
    args = ap.parse_args(argv)

    paths = get_paths()
    bench_root = Path(args.benchmark) if args.benchmark else paths.root / "data" / "benchmark"
    fam_root = bench_root / args.family
    if not fam_root.is_dir():
        print(f"[verify] no such family dir: {fam_root}", file=sys.stderr)
        return 2

    # Default: brightest + sparsest condition -- the cleanest place to see a shift,
    # since photon-limited localization error is smallest there.
    if args.condition:
        cond_dir = fam_root / args.condition
    else:
        cands = sorted(d for d in fam_root.iterdir() if d.is_dir())
        def _key(d: Path) -> tuple[float, float]:
            snr = float(d.name.split("snr=")[1].split("_")[0])
            den = float(d.name.split("density=")[1])
            return (-snr, den)          # highest SNR, then lowest density
        cond_dir = sorted(cands, key=_key)[0]
    if not cond_dir.is_dir():
        print(f"[verify] no such condition: {cond_dir}", file=sys.stderr)
        return 2

    with open(cond_dir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    images = meta.get("images", [])[: args.n_images]
    print(f"[verify] benchmark = {bench_root}")
    print(f"[verify] condition = {args.family}/{cond_dir.name}  ({len(images)} images)")
    print(f"[verify] centroid half-window = {args.half_window} px, "
          f"isolation radius = {args.min_sep} px\n")

    rows = []
    for rec in images:
        image_id = str(rec["image_id"])
        img = tifffile.imread(cond_dir / rec["image_file"])          # (2, H, W)
        gt = pd.read_csv(cond_dir / rec["ground_truth_file"])
        gt = _isolated(gt, args.min_sep)
        for _, s in gt.iterrows():
            for ch in (0, 1):
                cx, cy, flux = _centroid(img[ch], float(s["x"]), float(s["y"]),
                                         args.half_window)
                if np.isnan(cx):
                    continue
                rows.append({"image_id": image_id, "channel": ch + 1,
                             "dx": cx - float(s["x"]), "dy": cy - float(s["y"])})

    if not rows:
        print("[verify] no isolated, in-bounds spots found -- try a sparser condition.",
              file=sys.stderr)
        return 2

    df = pd.DataFrame(rows)
    print(f"[verify] {len(df)} isolated spot-channel measurements\n")

    ok = True
    print(f"{'ch':<4} {'n':>5} {'mean dx':>9} {'mean dy':>9} {'sd dx':>8} {'sd dy':>8} "
          f"{'max |per-image mean|':>22}")
    print("-" * 72)
    for ch, sub in df.groupby("channel"):
        per_image = sub.groupby("image_id")[["dx", "dy"]].mean()
        worst = float(np.abs(per_image.to_numpy()).max())
        sd_x, sd_y = float(sub["dx"].std()), float(sub["dy"].std())
        print(f"{int(ch):<4} {len(sub):>5} {sub['dx'].mean():>9.4f} {sub['dy'].mean():>9.4f} "
              f"{sd_x:>8.4f} {sd_y:>8.4f} {worst:>22.4f}")
        if max(sd_x, sd_y) > _SD_FAIL or worst > _IMAGE_MEAN_FAIL:
            ok = False

    # ch1-vs-ch2 disagreement: the shifts are drawn INDEPENDENTLY per channel, so a
    # live shift shows up as per-image channel-to-channel misregistration too.
    piv = df.pivot_table(index="image_id", columns="channel", values=["dx", "dy"],
                         aggfunc="mean")
    if 1 in piv["dx"].columns and 2 in piv["dx"].columns:
        d = np.hypot(piv["dx"][1] - piv["dx"][2], piv["dy"][1] - piv["dy"][2])
        print(f"\n[verify] per-image ch1-vs-ch2 centre disagreement: "
              f"median {float(d.median()):.4f} px, max {float(d.max()):.4f} px")

    print(f"\n[verify] reference: a live U(-1,1) shift would give per-axis sd ~ "
          f"{_U_SD:.3f} px and per-image means scattered over [-1, 1].")
    if ok:
        print("[verify] PASS -- residuals are photon-limited and per-image means are ~0. "
              "No registration shift in the ground truth.")
        return 0
    print("[verify] FAIL -- a constant per-image offset is present. The benchmark still "
          "carries a registration shift; do NOT trust localization or alpha from it.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
