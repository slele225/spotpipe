#!/usr/bin/env python
"""Run the FULL frozen SpotMAX benchmark in small batches (memory-bounded).

A 5-image SpotMAX run already peaked >95% RAM, so the full frozen set is processed
in independent batches (default 5 images), each a FRESH ``spotmax`` process that
releases memory before the next. This script orchestrates the whole flow but
**never imports SpotMAX** -- SpotMAX is invoked only as an external CLI subprocess
(``spotmax -p config.ini``), exactly as a separate-env install requires.

Stages (``--stages``, comma list; default all, runnable independently / resumable):

  prepare    split the frozen set into batches of N; for each batch write a
             Cell-ACDC ``Position_*/Images`` tree + ``id_map.csv`` + a per-batch
             ``config.ini`` derived from your working GUI-saved INI (only the
             "Folder path" line is repointed; everything else is preserved).
  run        for each batch with no SpotMAX output yet, run
             ``<--spotmax-cmd> -p <batch>/config.ini`` (external CLI). Resumable:
             batches that already have ``SpotMAX_output`` are skipped.
  merge      parse each batch's output tables into per-batch neutral detections,
             then MERGE into one neutral CSV (image-disjoint batches -> concat).
  convert    merged neutral + photon images -> canonical predictions (aperture/
             annulus photometry; honest ``--method`` flags). Reports counts.
  benchmark  run the harness with the chosen method over the frozen set.

Fairness / hygiene (unchanged): SpotMAX detects on RAW ``raw_max`` TIFFs; canonical
I1/I2 come from aperture/annulus photometry on the PHOTON images; the simulator
true-background is never read; all outputs live under git-ignored ``external_runs/``.

Typical (threshold detector, full set, you run SpotMAX yourself per batch)::

    # 1) prepare batches + per-batch INIs from your working GUI-saved INI
    uv run python scripts/run_spotmax_batches.py --stages prepare \
        --benchmark data/benchmark_test_v1 --out external_runs/spotmax/full \
        --template-ini external_runs/spotmax/working_threshold.ini --batch-size 5

    # 2) in your SpotMAX env, run each external_runs/spotmax/full/batches/batch_*/config.ini
    #    (or let this script drive the CLI:  --stages run --spotmax-cmd spotmax)

    # 3) merge + convert + benchmark
    uv run python scripts/run_spotmax_batches.py --stages merge,convert,benchmark \
        --benchmark data/benchmark_test_v1 --out external_runs/spotmax/full \
        --method spotmax_threshold_plus_aperture
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable from a fresh checkout too (no sys.path hacks for shared code).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark import spotmax as smx
from spotpipe.schema import write_spots

ALL_STAGES = ("prepare", "run", "merge", "convert", "benchmark")


def _photometry_cfg(args) -> dict:
    return {
        "detect_image": args.detect_image,
        "nonpositive": args.nonpositive,
        "window_radius_px": args.window_radius_px,
        "bg_inner_px": args.bg_inner_px,
        "bg_outer_px": args.bg_outer_px,
    }


def _batch_dirs(batches_root: Path) -> list[Path]:
    return sorted(p for p in batches_root.glob("batch_*") if p.is_dir())


def _load_batch_id_map(batch_dir: Path) -> dict[str, str]:
    df = pd.read_csv(batch_dir / "id_map.csv")
    return {str(r["position"]): str(r["image_id"]) for _, r in df.iterrows()}


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_prepare(args, bench: Path, batches_root: Path) -> None:
    if not args.template_ini:
        raise ValueError("--template-ini (your working GUI-saved SpotMAX INI) is required for 'prepare'")
    template_text = Path(args.template_ini).read_text(encoding="utf-8")

    with open(bench / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = manifest["images"]
    if args.limit is not None:
        entries = entries[: int(args.limit)]
    if not entries:
        raise ValueError("no images to batch (empty manifest / --limit 0)")

    size = int(args.batch_size)
    batches_root.mkdir(parents=True, exist_ok=True)
    n_batches = (len(entries) + size - 1) // size
    for b in range(n_batches):
        chunk = entries[b * size:(b + 1) * size]
        batch_dir = batches_root / f"batch_{b:03d}"
        input_root = batch_dir / "input"
        rows = smx.export_positions(bench, chunk, input_root, args.detect_image)
        pd.DataFrame(rows).to_csv(batch_dir / "id_map.csv", index=False)
        ini_text = smx.rewrite_ini_folder_path(
            template_text, str(input_root.resolve()), key=args.ini_folder_key,
        )
        (batch_dir / "config.ini").write_text(ini_text, encoding="utf-8")
        print(f"[batches] {batch_dir.name}: {len(chunk)} images -> {input_root} (+ config.ini)")
    print(f"[batches] prepared {n_batches} batch(es) of up to {size} image(s) under {batches_root}")


def stage_run(args, batches_root: Path) -> None:
    batch_dirs = _batch_dirs(batches_root)
    if not batch_dirs:
        raise FileNotFoundError(f"no batches under {batches_root}; run --stages prepare first")
    for batch_dir in batch_dirs:
        ini = batch_dir / "config.ini"
        if not ini.exists():
            raise FileNotFoundError(f"{batch_dir.name}: missing config.ini (re-run 'prepare')")
        if any((batch_dir / "input").rglob("SpotMAX_output")):
            print(f"[batches] {batch_dir.name}: SpotMAX_output present -> skip (resumable)")
            continue
        cmd = [args.spotmax_cmd, "-p", str(ini)]
        print(f"[batches] {batch_dir.name}: running {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"SpotMAX exited {result.returncode} for {batch_dir.name}. Run it manually in "
                f"your SpotMAX env: spotmax -p {ini}"
            )


def stage_merge(args, batches_root: Path, merged_neutral: Path) -> pd.DataFrame:
    batch_dirs = _batch_dirs(batches_root)
    if not batch_dirs:
        raise FileNotFoundError(f"no batches under {batches_root}; run --stages prepare first")
    per_batch_paths = []
    for batch_dir in batch_dirs:
        id_map = _load_batch_id_map(batch_dir)
        neutral = smx.parse_spotmax_output(
            batch_dir / "input", id_map, x_col=args.x_col, y_col=args.y_col, p_col=args.p_col,
        )
        bpath = batch_dir / "neutral_detections.csv"
        neutral.to_csv(bpath, index=False)
        per_batch_paths.append(bpath)
        print(f"[batches] {batch_dir.name}: parsed {len(neutral)} detection(s)")
    merged = smx.merge_neutral_detections(per_batch_paths)
    merged_neutral.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_neutral, index=False)
    print(f"[batches] merged {len(merged)} detection(s) across {merged['image_id'].nunique()} "
          f"image(s) -> {merged_neutral}")
    return merged


def stage_convert(args, bench: Path, merged_neutral: Path, canonical_out: Path) -> None:
    if not merged_neutral.exists():
        raise FileNotFoundError(f"{merged_neutral} not found; run --stages merge first")
    neutral = pd.read_csv(merged_neutral)
    pred, total = smx.neutral_to_canonical(neutral, bench, cfg=_photometry_cfg(args), method_name=args.method)
    write_spots(pred, canonical_out)
    if total["n_missing_image"]:
        print(f"[batches] WARNING: {total['n_missing_image']} detection(s) had no photon image (skipped)")
    print(f"[batches] convert ({args.method}): {total['n_in']} in -> {total['n_out']} emitted "
          f"({total['n_nonpositive']} non-positive, policy={args.nonpositive})")
    print(f"[batches] wrote canonical predictions: {canonical_out} ({len(pred)} rows)")


def stage_benchmark(args, bench: Path, merged_neutral: Path, bench_out: Path) -> None:
    if not merged_neutral.exists():
        raise FileNotFoundError(f"{merged_neutral} not found; run --stages merge first")
    from spotpipe.benchmark.harness import load_frozen_benchmark_set, run_benchmark

    config: dict = {}
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    block = config.setdefault("benchmark", {})
    sm = block.setdefault("spotmax", {})
    sm.update(_photometry_cfg(args))
    sm["detections_csv"] = str(merged_neutral)

    eval_set = load_frozen_benchmark_set(bench, limit=args.limit)
    result = run_benchmark(eval_set, config, out_dir=bench_out, methods=[args.method])
    print(f"[batches] benchmark wrote {len(result['metrics'])} method(s) to {result['out_dir']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full SpotMAX benchmark in memory-bounded batches.")
    parser.add_argument("--benchmark", required=True, help="frozen benchmark/test set dir")
    parser.add_argument("--out", required=True, help="run dir (e.g. external_runs/spotmax/full)")
    parser.add_argument("--stages", default=",".join(ALL_STAGES),
                        help=f"comma list of stages to run (any of {','.join(ALL_STAGES)})")
    parser.add_argument("--batch-size", type=int, default=5, help="images per SpotMAX batch (default 5)")
    parser.add_argument("--limit", type=int, default=None, help="cap total images (for testing)")
    parser.add_argument("--method", default=smx.SPOTMAX_METHOD_THRESHOLD, choices=list(smx.SPOTMAX_METHODS),
                        help="honest method name (default: the threshold detector path)")
    parser.add_argument("--template-ini", default=None, help="working GUI-saved SpotMAX INI (for 'prepare')")
    parser.add_argument("--ini-folder-key", default="Folder path", help="INI key whose value is the data folder")
    parser.add_argument("--spotmax-cmd", default="spotmax", help="external SpotMAX CLI for 'run'")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"),
                        help="base benchmark config (for 'benchmark')")
    parser.add_argument("--detect-image", default="raw_max", help="detection image protocol / provenance flag")
    parser.add_argument("--nonpositive", default="clamp", choices=["clamp", "reject"])
    parser.add_argument("--window-radius-px", type=float, default=3.0)
    parser.add_argument("--bg-inner-px", type=float, default=5.0)
    parser.add_argument("--bg-outer-px", type=float, default=8.0)
    parser.add_argument("--x-col", default=None, help="override SpotMAX x (column) source column")
    parser.add_argument("--y-col", default=None, help="override SpotMAX y (row) source column")
    parser.add_argument("--p-col", default=None, help="override SpotMAX p_detect source column")
    args = parser.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; valid: {list(ALL_STAGES)}")

    bench = Path(args.benchmark)
    out = Path(args.out)
    batches_root = out / "batches"
    merged_neutral = out / "predictions" / "neutral_detections.csv"
    canonical_out = out / "predictions" / f"{args.method}_predictions.csv"
    bench_out = out / "benchmark"

    if "prepare" in stages:
        stage_prepare(args, bench, batches_root)
    if "run" in stages:
        stage_run(args, batches_root)
    if "merge" in stages:
        stage_merge(args, batches_root, merged_neutral)
    if "convert" in stages:
        stage_convert(args, bench, merged_neutral, canonical_out)
    if "benchmark" in stages:
        stage_benchmark(args, bench, merged_neutral, bench_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
