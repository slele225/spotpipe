#!/usr/bin/env python
"""Convert SpotMAX output tables -> neutral detections -> canonical predictions.

Stage 5+6 of the SpotMAX adapter. Two outputs from one pass:

  1. a NEUTRAL detections CSV (``image_id,x,y,p_detect,native_source_file,
     native_row,native_columns_json``) parsed from the SpotMAX output tables --
     positions/confidence mapped to spotpipe convention, full native row kept as
     JSON for auditability (native intensities are NOT used);
  2. the CANONICAL 16-column spotpipe predictions CSV for
     ``spotmax_ai_plus_aperture``: SpotMAX centres + aperture/annulus photometry
     on the PHOTON images (the same estimator as the aperture baseline). Raw
     counts are never divided; ``audit/`` is never read.

The neutral CSV is ALSO the harness adapter's input (point
``config['spotmax']['detections_csv']`` at it). This script writes the canonical
predictions CSV directly too, so you have the artifact named in the plan; the
harness adapter reproduces the identical photometry from the neutral CSV.

This script does NOT import SpotMAX. It reads the SpotMAX *output* under the run
dir and the frozen set's photon TIFFs. SpotMAX output COLUMN NAMES are not
assumed -- the parser auto-detects coordinate columns (override with
``--x-col``/``--y-col``/``--p-col`` after inspecting a real run).

Typical run::

    uv run python scripts/convert_spotmax_output.py \
        --spotmax-run external_runs/spotmax/smoke \
        --benchmark data/benchmark_test_v1 \
        --out external_runs/spotmax/smoke/predictions/spotmax_ai_plus_aperture_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable from a fresh checkout too (no sys.path hacks for shared code).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark import spotmax as smx
from spotpipe.schema import SCHEMA_COLUMNS, write_spots


def _load_id_map(run_dir: Path) -> dict[str, str]:
    """position -> image_id from the exporter's id_map.csv."""
    id_map_path = run_dir / "id_map.csv"
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"id_map.csv not found under {run_dir}; run scripts/export_spotmax_input.py first."
        )
    df = pd.read_csv(id_map_path)
    return {str(r["position"]): str(r["image_id"]) for _, r in df.iterrows()}


def _photon_for_images(bench: Path, image_ids) -> dict[str, np.ndarray]:
    """Load the ``[2,H,W]`` photon image for each needed image_id (fair photometry)."""
    import tifffile

    with open(bench / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    by_id = {str(e["image_id"]): e for e in manifest["images"]}
    wanted = set(map(str, image_ids))
    out: dict[str, np.ndarray] = {}
    for image_id in sorted(wanted):
        entry = by_id.get(image_id)
        if entry is None:
            continue
        p1 = tifffile.imread(bench / entry["ch1_photon"])
        p2 = tifffile.imread(bench / entry["ch2_photon"])
        out[image_id] = np.stack([p1, p2], axis=0).astype(float)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert SpotMAX output -> canonical predictions.")
    parser.add_argument("--spotmax-run", required=True, help="run dir (holds id_map.csv + SpotMAX_output/)")
    parser.add_argument("--benchmark", required=True, help="frozen benchmark/test set dir (photon images)")
    parser.add_argument("--out", required=True, help="canonical predictions CSV path")
    parser.add_argument("--neutral-out", default=None,
                        help="neutral detections CSV path (default: <out dir>/neutral_detections.csv)")
    parser.add_argument("--x-col", default=None, help="override SpotMAX x (column) source column")
    parser.add_argument("--y-col", default=None, help="override SpotMAX y (row) source column")
    parser.add_argument("--p-col", default=None, help="override SpotMAX p_detect source column")
    parser.add_argument("--detect-image", default="raw_max", help="provenance flag value")
    parser.add_argument("--nonpositive", default="clamp", choices=["clamp", "reject"],
                        help="non-positive intensity policy")
    parser.add_argument("--window-radius-px", type=float, default=3.0)
    parser.add_argument("--bg-inner-px", type=float, default=5.0)
    parser.add_argument("--bg-outer-px", type=float, default=8.0)
    args = parser.parse_args(argv)

    run_dir = Path(args.spotmax_run)
    bench = Path(args.benchmark)

    # 1. Parse SpotMAX output tables -> neutral detections.
    id_map = _load_id_map(run_dir)
    tables = smx.find_spotmax_tables(run_dir)
    if not tables:
        print(
            f"[convert] no SpotMAX_output tables found under {run_dir}. Has SpotMAX run yet?\n"
            f"          expected: <Position>/.../SpotMAX_output/<n>_(valid|detected)_spots*.csv"
        )
        return 1
    print(f"[convert] found {len(tables)} SpotMAX output table(s):")
    for pos, tbl in sorted(tables.items()):
        cols = list(smx._read_table(tbl).columns)
        print(f"          {pos}: {tbl.name}  columns={cols}")

    neutral = smx.parse_spotmax_output(
        run_dir, id_map, x_col=args.x_col, y_col=args.y_col, p_col=args.p_col,
    )
    n_parsed = int(len(neutral))
    print(f"[convert] parsed {n_parsed} detection(s) across {neutral['image_id'].nunique()} image(s)")

    neutral_out = Path(args.neutral_out) if args.neutral_out else Path(args.out).parent / "neutral_detections.csv"
    neutral_out.parent.mkdir(parents=True, exist_ok=True)
    neutral.to_csv(neutral_out, index=False)
    print(f"[convert] wrote neutral detections: {neutral_out}")

    if n_parsed == 0:
        print("[convert] no detections parsed; writing empty canonical CSV.")
        write_spots(pd.DataFrame(columns=list(SCHEMA_COLUMNS)), args.out)
        return 0

    # 2. Aperture/annulus photometry on the PHOTON images -> canonical schema.
    photon_by_image = _photon_for_images(bench, neutral["image_id"].unique())
    cfg = {
        "detect_image": args.detect_image,
        "nonpositive": args.nonpositive,
        "window_radius_px": args.window_radius_px,
        "bg_inner_px": args.bg_inner_px,
        "bg_outer_px": args.bg_outer_px,
    }
    total = {"n_in": 0, "n_out": 0, "n_nonpositive": 0}
    frames = []
    for image_id, sub in neutral.groupby("image_id"):
        photon = photon_by_image.get(str(image_id))
        if photon is None:
            print(f"[convert] WARNING: no photon image for {image_id}; skipping its {len(sub)} detection(s)")
            continue
        df, stats = smx.spotmax_plus_aperture(photon, sub, image_id=str(image_id), cfg=cfg)
        for k in total:
            total[k] += stats[k]
        frames.append(df)

    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(SCHEMA_COLUMNS))
    write_spots(pred, args.out)

    print(
        f"[convert] photometry: {total['n_in']} detections in -> {total['n_out']} emitted "
        f"({total['n_nonpositive']} non-positive, policy={args.nonpositive})"
    )
    print(f"[convert] wrote canonical predictions: {args.out} ({len(pred)} rows)")
    print(
        "[convert] benchmark with:\n"
        f"           uv run python scripts/run_benchmark.py --frozen-dir {bench} "
        f"--limit {len(id_map)} --methods spotmax_ai_plus_aperture "
        f"--config <yaml with spotmax.detections_csv={neutral_out}> --out {run_dir / 'benchmark'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
