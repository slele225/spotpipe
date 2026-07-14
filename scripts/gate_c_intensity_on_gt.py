"""GATE C -- is the bright/dense log-ratio bias the DETECTOR or the MODEL?

RESULTS_hrnet_large_measured.md sec.3: the model's log-ratio bias reaches -1.1 (a factor-3
error in A2/A1) in the bright, dense cells, and grows with density at fixed SNR.

This script removes the model entirely. It runs the SHARED measurement instrument
(`benchmark.intensity_extraction`) at the GROUND-TRUTH positions -- i.e. a perfect
detector with perfect localization -- and asks whether the intensities are still
under-read. Whatever bias survives here is NOT the model's fault: it is in the pixels,
and it will hit every baseline identically.

Three candidate causes, and the columns that separate them:

  (a) DETECTOR COMPRESSION. ch2 gain is 40 vs ch1's 6.63, and the tanh soft knee
      compresses smoothly WELL BELOW the 3941 ADU hard knee -- so it never trips the
      hard-saturation flag (`n_saturated_total = 0` in every cell). Where overlapping
      spots pile up, ch2 loses light.
      => `ch2_frac_px_over_50%knee` climbs with density, and the bias is concentrated
         in spots whose local peak is near the knee.

  (b) CROWDING IN THE EXTRACTION. A neighbour inside the 3-sigma aperture inflates the
      measured flux; a neighbour in the background annulus inflates the background and
      DEFLATES it. Either way it is a measurement artifact, not the detector.
      => the bias appears in the CROWDED isolation bin but not the ISOLATED one, and
         `ch2_frac_px_over_knee` stays flat.

  (c) THE MODEL. If GT-position extraction recovers I2 cleanly in the very cells where
      the model reported -1.1, then the pixels are fine and the intensity head is the
      problem.
      => every bias below is ~0 and sec.3 of the results doc is WRONG.

(a) and (b) both mean "do not quote intensity metrics in these cells, for ANY method".
Only (c) is a model defect.

Units: extraction returns flux in ADU-proportional units; ground truth is in photons.
The conversion is I_photons = V_ADU / gain (the background subtraction already removes
the ADU offset). Gains are read from BENCH_MANIFEST.json -- never hardcoded.

Usage:
    python scripts/gate_c_intensity_on_gt.py
    python scripts/gate_c_intensity_on_gt.py --method aperture --n-images 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from spotpipe.benchmark.evaluate import load_benchmark_info
from spotpipe.benchmark.intensity_extraction import extract_intensities
from spotpipe.paths import get_paths

# The cells that matter: the two catastrophic ones, their sparse controls at the SAME
# SNR (isolates density), and a dim control (isolates brightness).
_DEFAULT_CONDITIONS = [
    "snr=15_density=0.012",    # log_ratio_bias -1.100  <- the worst
    "snr=15_density=0.006",    # log_ratio_bias -0.732
    "snr=15_density=0.0006",   # log_ratio_bias -0.136  <- same SNR, sparse: DENSITY control
    "snr=10_density=0.012",    # log_ratio_bias -0.920
    "snr=2_density=0.015",     # log_ratio_bias -0.016  <- same density, dim: BRIGHTNESS control
]

_ISOLATION_SIGMA = 6.0     # a spot is "isolated" if its nearest neighbour is >= 6*sigma away
                           # (the extraction crop half-width -- nothing else can reach into it)


def _detector(bench_root: Path) -> dict:
    """The measured detector block, straight from the manifest (never hardcoded)."""
    with open(bench_root / "BENCH_MANIFEST.json", "r", encoding="utf-8") as fh:
        return json.load(fh)["detector"]


def _nn_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.full(len(x), np.inf)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _local_peak(plane: np.ndarray, x: float, y: float, half: int = 2) -> float:
    h, w = plane.shape
    x0, x1 = max(0, int(round(x)) - half), min(w, int(round(x)) + half + 1)
    y0, y1 = max(0, int(round(y)) - half), min(h, int(round(y)) + half + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    return float(plane[y0:y1, x0:x1].max())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--family", default="snr_density")
    ap.add_argument("--conditions", nargs="*", default=_DEFAULT_CONDITIONS)
    ap.add_argument("--method", default="gaussian", choices=("gaussian", "aperture"))
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--out", default=None, help="CSV path (default: results/gate_c_intensity_on_gt.csv)")
    args = ap.parse_args(argv)

    paths = get_paths()
    bench_root = Path(args.benchmark) if args.benchmark else paths.root / "data" / "benchmark"
    # Same manifest reader the evaluator uses -- sigma cannot drift between the two.
    info = load_benchmark_info(bench_root)
    sigma1, sigma2 = info.sigma1, info.sigma2
    det = _detector(bench_root)
    g1, g2 = float(det["ch1"]["gain"]), float(det["ch2"]["gain"])
    knee = float(det["ch1"]["saturation_knee"])
    offset = float(det["ch1"]["offset"])

    print(f"[gate-c] benchmark = {bench_root}")
    print(f"[gate-c] instrument = intensity_extraction({args.method}) at GROUND-TRUTH positions")
    print(f"[gate-c] sigma1={sigma1} sigma2={sigma2}  gain1={g1} gain2={g2}  "
          f"knee={knee} ADU  offset={offset} ADU")
    print(f"[gate-c] a perfect detector: any bias below is in the PIXELS, not the model\n")

    rows = []
    for cond in args.conditions:
        cdir = bench_root / args.family / cond
        if not cdir.is_dir():
            print(f"[gate-c] skip (missing): {cond}")
            continue
        with open(cdir / "meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        recs = []
        px_over_50, px_over_90, px_total = 0, 0, 0
        for im in meta.get("images", [])[: args.n_images]:
            img = tifffile.imread(cdir / im["image_file"]).astype(np.float64)   # [2,H,W] ADU
            gt = pd.read_csv(cdir / im["ground_truth_file"])
            if gt.empty:
                continue

            ch2 = img[1] - offset
            px_over_50 += int((ch2 > 0.50 * knee).sum())
            px_over_90 += int((ch2 > 0.90 * knee).sum())
            px_total += ch2.size

            xs = gt["x"].to_numpy(float)
            ys = gt["y"].to_numpy(float)
            ext = extract_intensities(img, xs, ys, sigma1=sigma1, sigma2=sigma2,
                                      method=args.method)
            # ADU flux -> photons (background subtraction already removed the offset).
            i1 = ext.I1 / g1
            i2 = ext.I2 / g2
            with np.errstate(divide="ignore", invalid="ignore"):
                lr_hat = np.log(i2) - np.log(i1)
                l1_hat, l2_hat = np.log(i1), np.log(i2)

            recs.append(pd.DataFrame({
                "nn_px": _nn_distance(xs, ys),
                "logI1_err": l1_hat - gt["logI1"].to_numpy(float),
                "logI2_err": l2_hat - gt["logI2"].to_numpy(float),
                "log_ratio_err": lr_hat - gt["log_ratio"].to_numpy(float),
                "ch2_peak_adu": [_local_peak(ch2, x, y) for x, y in zip(xs, ys)],
            }))

        if not recs:
            continue
        df = pd.concat(recs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        clean = df.dropna(subset=["log_ratio_err"])
        iso = clean[clean["nn_px"] >= _ISOLATION_SIGMA * max(sigma1, sigma2)]
        crowded = clean[clean["nn_px"] < 3.0 * max(sigma1, sigma2)]

        rows.append({
            "condition": cond,
            "n_spots": len(clean),
            # THE headline: does a perfect detector still under-read the ratio?
            "gt_pos_log_ratio_bias": clean["log_ratio_err"].mean(),
            "gt_pos_logI1_bias": clean["logI1_err"].mean(),
            "gt_pos_logI2_bias": clean["logI2_err"].mean(),
            # (b) crowding: does the bias live only in the crowded spots?
            "n_isolated": len(iso),
            "isolated_log_ratio_bias": iso["log_ratio_err"].mean() if len(iso) else np.nan,
            "n_crowded": len(crowded),
            "crowded_log_ratio_bias": crowded["log_ratio_err"].mean() if len(crowded) else np.nan,
            # (a) compression: how close to the knee is ch2 actually getting?
            "ch2_median_peak_adu": clean["ch2_peak_adu"].median(),
            "ch2_p99_peak_adu": clean["ch2_peak_adu"].quantile(0.99),
            "ch2_frac_px_over_50pct_knee": px_over_50 / max(px_total, 1),
            "ch2_frac_px_over_90pct_knee": px_over_90 / max(px_total, 1),
        })
        print(f"  [done] {cond:<24} n={len(clean):>6}  "
              f"log-ratio bias {clean['log_ratio_err'].mean():+.4f}  "
              f"(isolated {iso['log_ratio_err'].mean() if len(iso) else float('nan'):+.4f} / "
              f"crowded {crowded['log_ratio_err'].mean() if len(crowded) else float('nan'):+.4f})")

    if not rows:
        print("[gate-c] no conditions evaluated")
        return 2

    out = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else paths.root / "results" / "gate_c_intensity_on_gt.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n" + "=" * 100)
    print("GATE C -- shared instrument at GROUND-TRUTH positions (no model involved)")
    print("=" * 100)
    print(out[["condition", "n_spots", "gt_pos_log_ratio_bias", "gt_pos_logI1_bias",
               "gt_pos_logI2_bias", "isolated_log_ratio_bias", "crowded_log_ratio_bias",
               "ch2_median_peak_adu", "ch2_frac_px_over_50pct_knee"]].to_string(index=False))
    print(f"\n[gate-c] knee = {knee} ADU above offset. ch2 gain {g2} vs ch1 {g1} "
          f"(ch2 reaches the knee {g2 / g1:.1f}x sooner).")
    print("\nREAD IT LIKE THIS:")
    print("  * bias survives at GT positions, grows with density, ch2 >> ch1, peaks near knee")
    print("      -> DETECTOR COMPRESSION. Benchmark artifact. Fix the gain/SNR grid, not the model.")
    print("  * bias lives in the CROWDED bin only, isolated bin ~0, ch2 far from knee")
    print("      -> EXTRACTION CROWDING. Measurement artifact. Still not the model.")
    print("  * bias ~0 everywhere at GT positions")
    print("      -> THE MODEL's intensity head. sec.3 of RESULTS is wrong; the pixels are fine.")
    print(f"\n[gate-c] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
