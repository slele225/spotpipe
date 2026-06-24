#!/usr/bin/env python
"""Produce / validate the normalized CME detections CSV for cmeanalysis_plus_aperture.

CMEAnalysis is a LOCAL, EXTERNAL MATLAB package. This repo never vendors it and
never contains the MATLAB wrapper -- the wrapper lives outside the repo (and
outside the CMEAnalysis source tree). This script only:

  * validates an already-exported normalized detections CSV (``normalized_csv``),
    OR
  * lays out a clean 2-channel "condition" folder from the frozen set's RAW
    per-channel TIFFs and invokes the EXTERNAL MATLAB wrapper to produce the
    normalized CSV (``external_matlab``).

It NEVER prepares the legacy 3-page TIFF convention, and NEVER reads ``audit/``.

Normalized detections CSV contract (the only spotpipe<->CME interface)
----------------------------------------------------------------------
Required columns: ``image_id, x, y``  (x = column, y = row, 0-indexed sub-pixel)
Optional columns: ``score, A, slave_A, channel, native_I1, native_I2``

Expected EXTERNAL MATLAB wrapper signature (you maintain this file outside the repo)
-----------------------------------------------------------------------------------
::

    spotpipe_cme_detect(condDir, masterChName, slaveChName, outCsv, NA, M, pixelSizeM, Alpha)

It should: loadConditionData(condDir, {masterChName, slaveChName}, {markers}, ...)
headlessly, runDetection(data, 'Master', 1), then read each per-image
``<masterCh>/Detection/detection_v2.mat``, take ``frameInfo(1)``, convert MATLAB
1-indexed master coords to spotpipe 0-indexed (``x-1``, ``y-1``), and write ONE
normalized CSV with ``image_id`` = the per-image subfolder name.

Examples
--------
Validate a pre-exported normalized CSV::

    uv run python scripts/run_cmeanalysis.py \\
        --input-format normalized_csv \\
        --detections-csv data/benchmark_test_v1/cme_detections/detections.csv

Diagnose CMEAnalysis auto-sigma over image-count prefixes (no fixed sigma; finds
when sigma estimation degrades to NaN)::

    uv run python scripts/run_cmeanalysis.py \\
        --input-format diagnose_sigma \\
        --frozen-dir data/benchmark_test_v1 \\
        --sweep-limits 4,8,16,32,64,full \\
        --detect-channel 2 \\
        --cme-software-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/cmeAnalysis-master/software" \\
        --matlab-wrapper-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/spotpipe_external" \\
        --matlab-entrypoint spotpipe_cme_detect

Lay out inputs + run the external MATLAB wrapper over the frozen set::

    uv run python scripts/run_cmeanalysis.py \\
        --input-format external_matlab \\
        --frozen-dir data/benchmark_test_v1 \\
        --detect-channel 2 \\
        --cme-software-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/cmeAnalysis-master/software" \\
        --matlab-wrapper-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/spotpipe_external" \\
        --matlab-entrypoint spotpipe_cme_detect \\
        --na 1.49 --magnification 108 --pixel-size-m 6.5e-6 \\
        --out data/benchmark_test_v1/cme_detections/detections.csv

Full 139-image set in batches (RECOMMENDED: --batch-size 64 -> 64+64+11), so
CMEAnalysis auto-estimates sigma independently per batch (avoids nf=round(40/nd)->0
on the whole condition folder)::

    uv run python scripts/run_cmeanalysis.py \\
        --input-format external_matlab \\
        --frozen-dir data/benchmark_test_v1 \\
        --batch-size 64 \\
        --detect-channel 2 \\
        --cme-software-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/cmeAnalysis-master/software" \\
        --matlab-wrapper-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/spotpipe_external" \\
        --matlab-entrypoint spotpipe_cme_detect \\
        --na 1.49 --magnification 108 --pixel-size-m 6.5e-6 \\
        --out data/benchmark_test_v1/cme_detections/detections.csv
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable straight from a fresh checkout too (no sys.path hacks for shared code).
sys.path.insert(0, str(REPO_ROOT / "src"))


def _validate_csv(path: Path) -> int:
    """Validate a normalized CSV against the contract; return the row count."""
    from spotpipe.benchmark.cmeanalysis import OPTIONAL_COLUMNS, load_normalized_detections

    df = load_normalized_detections(path)
    n_images = df["image_id"].nunique()
    present_opt = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    print(f"[ok] {path}: {len(df)} detections over {n_images} image(s); "
          f"optional columns present: {present_opt or 'none'}")
    return len(df)


def select_image_entries(entries: list[dict], limit: int | None) -> list[dict]:
    """First ``limit`` manifest image entries, in manifest order (stable).

    Mirrors ``harness.load_frozen_benchmark_set``'s slicing exactly so the smoke
    subset prepared here matches what the benchmark will then load: ``limit=None``
    keeps all images; otherwise keep ``entries[:limit]``. Order is the manifest
    order (same as the frozen-set loader / metadata). Negative ``limit`` is an error.
    """
    if limit is None:
        return list(entries)
    if int(limit) < 0:
        raise ValueError(f"--limit must be >= 0; got {limit}")
    return list(entries[: int(limit)])


def make_batches(entries: list[dict], batch_size: int | None) -> list[list[dict]]:
    """Split entries into stable-order batches of at most ``batch_size`` each.

    Order is preserved exactly (manifest order); the last batch may be shorter.
    ``batch_size`` must be a positive int (it is required in batch mode). Empty
    ``entries`` yields ``[]``. This is the pure helper behind ``--batch-size``: it
    lets CMEAnalysis auto-estimate sigma per (small) batch, sidestepping the
    ``nf = round(40/nd) -> 0`` collapse on large condition folders.
    """
    if batch_size is None:
        raise ValueError("--batch-size is required for batching (got None)")
    if int(batch_size) < 1:
        raise ValueError(f"--batch-size must be >= 1; got {batch_size}")
    bs = int(batch_size)
    return [list(entries[i:i + bs]) for i in range(0, len(entries), bs)]


def _copy_condition(frozen_dir: Path, cond: Path, entries: list[dict], detect_channel: int) -> tuple[str, str]:
    """Copy the given images' RAW TIFFs into a clean 2-channel condition folder.

    Layout (what loadConditionData expects): ``<cond>/<image_id>/ch1/<id>.tif`` and
    ``<cond>/<image_id>/ch2/<id>.tif``. ``image_id`` values are preserved verbatim;
    both raw channels are copied unchanged (no 3-page convention, no photometric
    changes). Returns (master_name, slave_name).
    """
    master = f"ch{int(detect_channel)}"
    slave = "ch2" if master == "ch1" else "ch1"
    cond.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        image_id = str(entry["image_id"])
        for ch_key, ch_name in (("ch1_raw", "ch1"), ("ch2_raw", "ch2")):
            src = frozen_dir / entry[ch_key]
            dst_dir = cond / image_id / ch_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst_dir / f"{image_id}.tif")
    return master, slave


def _layout_condition_folder(
    frozen_dir: Path, work_dir: Path, detect_channel: int, limit: int | None = None,
) -> tuple[Path, str, str]:
    """Build a clean 2-channel condition folder from the frozen RAW TIFFs.

    When ``limit`` is set, only the first N images (manifest order) are prepared --
    for a tiny CME smoke run without touching the full frozen set. Returns
    (cond_dir, master_name, slave_name).
    """
    with open(frozen_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    cond = work_dir / "condition"
    entries = select_image_entries(manifest["images"], limit)
    image_ids = [str(e["image_id"]) for e in entries]
    scope = f"--limit {limit}" if limit is not None else "full set"
    print(f"[layout] selected {len(image_ids)} of {len(manifest['images'])} image(s) ({scope}).")
    print(f"[layout] image_ids: {image_ids}")

    master, slave = _copy_condition(frozen_dir, cond, entries, detect_channel)
    print(f"[layout] wrote a {len(image_ids)}-image 2-channel condition folder to {cond} "
          f"(master={master}, slave={slave})")
    return cond, master, slave


def _build_matlab_invocation(args, cond_dir: Path, master: str, slave: str, out_csv: Path):
    """Build the MATLAB ``-batch`` invocation, resolving every path to absolute.

    MATLAB's path utilities (getDirFromPath) assume absolute paths -- a relative
    condDir was the cause of the "Array indices must be positive integers" failure
    in getDirFromPath. Returns (argv, sw, wrap, cond_dir, out_csv).
    """
    sw = Path(args.cme_software_folder).resolve()
    wrap = Path(args.matlab_wrapper_folder).resolve()
    cond_dir = cond_dir.resolve()
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Both folders are added to the path; the wrapper lives OUTSIDE the repo and
    # outside the CMEAnalysis source tree.
    def _ml(s: str) -> str:
        return str(s).replace("\\", "/").replace("'", "''")

    matlab_cmd = (
        f"addpath(genpath('{_ml(sw)}')); addpath('{_ml(wrap)}'); "
        f"{args.matlab_entrypoint}("
        f"'{_ml(cond_dir)}', '{master}', '{slave}', '{_ml(out_csv)}', "
        f"{args.na}, {args.magnification}, {args.pixel_size_m}, {args.alpha}); "
        f"exit;"
    )
    return [args.matlab_exe, "-batch", matlab_cmd], sw, wrap, cond_dir, out_csv


def _run_matlab(args, cond_dir: Path, master: str, slave: str, out_csv: Path) -> int:
    """Invoke the external MATLAB wrapper, with BOTH folders on the MATLAB path."""
    if not args.cme_software_folder or not args.matlab_wrapper_folder:
        print("[error] external_matlab requires --cme-software-folder and "
              "--matlab-wrapper-folder.", file=sys.stderr)
        return 2

    argv, sw, wrap, cond_dir, out_csv = _build_matlab_invocation(args, cond_dir, master, slave, out_csv)
    print(f"[matlab] NA={args.na}  M={args.magnification}  pixelSize_m={args.pixel_size_m}  "
          f"Alpha={args.alpha}")
    print(f"[matlab] entrypoint={args.matlab_entrypoint}")
    print(f"[matlab] software folder : {sw}")
    print(f"[matlab] wrapper  folder : {wrap}")
    print(f"[matlab] command: matlab -batch \"{argv[-1]}\"")

    proc = subprocess.run(argv)
    if proc.returncode != 0:
        print(f"[error] MATLAB exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode
    if not out_csv.exists():
        print(f"[error] MATLAB did not produce {out_csv}", file=sys.stderr)
        return 1
    _validate_csv(out_csv)
    return 0


# --------------------------------------------------------------------------- #
# Diagnostic: auto-sigma prefix sweep                                          #
# --------------------------------------------------------------------------- #
_SIGMA_RE = re.compile(r"Gaussian PSF s\.d\. values:\s*(.+)")


def _parse_sigma(text: str) -> tuple[str, bool]:
    """Extract the CMEAnalysis-printed PSF sigma line; flag whether it is NaN.

    Returns (sigma_str, ok). ``sigma_str`` is '(not printed)' if CMEAnalysis never
    reached the sigma print. ``ok`` is False if sigma is missing or contains NaN.
    """
    m = None
    for m in _SIGMA_RE.finditer(text):
        pass  # keep the last occurrence
    if m is None:
        return "(not printed)", False
    sigma = m.group(1).strip()
    ok = "nan" not in sigma.lower()
    return sigma, ok


def _parse_sweep_limits(spec: str) -> list[int | None]:
    """Parse '4,8,16,32,64,full' -> [4,8,16,32,64,None] (None = whole set)."""
    out: list[int | None] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("full", "all", "none"):
            out.append(None)
        else:
            out.append(int(tok))
    return out


def _diag_one(args, frozen_dir: Path, diag_root: Path, total: int, limit: int | None) -> dict:
    """Run one auto-sigma attempt over the first ``limit`` images, clean scratch."""
    label = "full" if limit is None else str(limit)
    n_images = total if limit is None else min(int(limit), total)
    work_dir = diag_root / f"work_{label}"
    out_csv = (diag_root / f"detections_{label}.csv").resolve()

    # clean scratch + any prior CSV so "csv produced?" is meaningful
    if work_dir.exists():
        shutil.rmtree(work_dir)
    if out_csv.exists():
        out_csv.unlink()

    cond, master, slave = _layout_condition_folder(frozen_dir, work_dir, args.detect_channel, limit=limit)
    argv, _sw, _wrap, _cond, out_csv = _build_matlab_invocation(args, cond, master, slave, out_csv)

    print(f"[diag] limit={label}  n_images={n_images}  running auto-sigma CMEAnalysis ...")
    proc = subprocess.run(argv, capture_output=True, text=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")

    sigma_str, sigma_ok = _parse_sigma(text)
    reached_detection = "Running detection for" in text
    csv_made = out_csv.exists()
    rows = None
    if csv_made:
        try:
            from spotpipe.benchmark.cmeanalysis import load_normalized_detections
            rows = len(load_normalized_detections(out_csv))
        except Exception:
            rows = -1

    insufficient = "Could not determine distribution" in text
    print(f"[diag] limit={label}: rc={proc.returncode}  sigma=[{sigma_str}]  "
          f"reached_detection={reached_detection}  csv={csv_made}"
          + (f" rows={rows}" if csv_made else "")
          + (f"  (insufficient-samples warning)" if insufficient else ""))

    return {
        "limit": label, "n_images": n_images, "returncode": proc.returncode,
        "sigma": sigma_str, "sigma_ok": sigma_ok, "reached_detection": reached_detection,
        "csv_produced": csv_made, "rows": rows, "insufficient_samples": insufficient,
        "passed": bool(sigma_ok and reached_detection and csv_made),
    }


def _diagnose_sigma_sweep(args) -> int:
    """Sweep auto-sigma CMEAnalysis over image-count prefixes; report failures.

    DIAGNOSTIC ONLY -- it never applies a fixed sigma; it uses CMEAnalysis' own
    data-driven sigma estimation exactly as a normal run would, just over the first
    N images, to reveal when/whether sigma estimation degrades to NaN. Each prefix
    runs with a freshly cleaned scratch folder so runs are independent.
    """
    if not args.frozen_dir:
        print("[error] diagnose_sigma requires --frozen-dir", file=sys.stderr)
        return 2
    if not args.cme_software_folder or not args.matlab_wrapper_folder:
        print("[error] diagnose_sigma requires --cme-software-folder and "
              "--matlab-wrapper-folder.", file=sys.stderr)
        return 2

    frozen_dir = Path(args.frozen_dir).resolve()
    with open(frozen_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    all_ids = [str(e["image_id"]) for e in manifest["images"]]
    total = len(all_ids)

    diag_root = (Path(args.out).resolve() if args.out
                 else (frozen_dir / "cme_detections" / "sigma_diag"))
    diag_root.mkdir(parents=True, exist_ok=True)

    limits = _parse_sweep_limits(args.sweep_limits)
    limits = [l for l in limits if l is None or l <= total]  # drop prefixes beyond the set
    print(f"[diag] frozen set has {total} images; sweeping limits {[ 'full' if l is None else l for l in limits ]}")
    print(f"[diag] scratch + per-limit CSVs under {diag_root} (git-ignored)")

    results = [_diag_one(args, frozen_dir, diag_root, total, lim) for lim in limits]

    # --- summary table ---
    print("\n" + "=" * 78)
    print("AUTO-SIGMA PREFIX SWEEP")
    print("=" * 78)
    print(f"{'limit':>6} {'n_img':>6} {'rc':>4} {'sigma':>16} {'detect':>7} {'csv':>5} {'rows':>6}  result")
    for r in results:
        print(f"{r['limit']:>6} {r['n_images']:>6} {r['returncode']:>4} {r['sigma']:>16} "
              f"{str(r['reached_detection']):>7} {str(r['csv_produced']):>5} "
              f"{str(r['rows']) if r['rows'] is not None else '-':>6}  "
              f"{'PASS' if r['passed'] else 'FAIL'}")

    # --- boundary analysis ---
    by_n = sorted(results, key=lambda r: r["n_images"])
    last_pass = max((r["n_images"] for r in by_n if r["passed"]), default=None)
    first_fail = min((r["n_images"] for r in by_n if not r["passed"]), default=None)
    print("-" * 78)
    if first_fail is None:
        print("[diag] all swept prefixes PASSED auto-sigma estimation.")
    elif last_pass is None:
        print(f"[diag] FAILED even at the smallest swept prefix (n={first_fail}). "
              "Try a smaller --sweep-limits (e.g. 1,2,3,4) to localize.")
    else:
        print(f"[diag] boundary: PASS up to n={last_pass}, FAIL by n={first_fail}.")
        print(f"[diag] the failure first appears within images "
              f"[{all_ids[last_pass]} .. {all_ids[first_fail-1]}] "
              f"(0-based index {last_pass}..{first_fail-1}).")
        print("[diag] to pinpoint the exact image, re-run with finer --sweep-limits in that "
              f"range, e.g. --sweep-limits {','.join(str(n) for n in range(last_pass+1, first_fail+1))}")
        print("[diag] NOTE: if the boundary tracks IMAGE COUNT (not a specific id), the cause is "
              "likely CMEAnalysis' frame-sampling for sigma (nf=round(40/nd) -> 0 for large nd), "
              "not a single bad image.")
    return 0 if first_fail is None else 1


def _run_external_batched(args, frozen_dir: Path, work_dir: Path, out_csv: Path) -> int:
    """Run CMEAnalysis in stable-order batches, then concatenate the per-batch CSVs.

    Each batch is a clean, independent condition folder + MATLAB call, so CMEAnalysis
    auto-estimates sigma on a small enough set that ``nf = round(40/nd)`` stays >= 1.
    Per-batch artifacts (condition folder, detections.csv) are kept under
    ``cme_work/batch_###/`` for debugging; a failing batch aborts loudly with its
    artifacts intact. ``image_id`` values are preserved verbatim end to end.
    """
    import pandas as pd

    from spotpipe.benchmark.cmeanalysis import load_normalized_detections

    if not args.cme_software_folder or not args.matlab_wrapper_folder:
        print("[error] external_matlab requires --cme-software-folder and "
              "--matlab-wrapper-folder.", file=sys.stderr)
        return 2

    with open(frozen_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = select_image_entries(manifest["images"], args.limit)
    batches = make_batches(entries, args.batch_size)
    if not batches:
        print("[error] no images selected to batch (check --limit / frozen set).", file=sys.stderr)
        return 2

    sizes = ", ".join(str(len(b)) for b in batches)
    print(f"[batch] {len(entries)} image(s) -> {len(batches)} batch(es) of size {args.batch_size} "
          f"(actual: {sizes})")

    batch_csvs: list[Path] = []
    for i, batch in enumerate(batches):
        label = f"batch_{i:03d}"
        bdir = work_dir / label
        if bdir.exists():
            shutil.rmtree(bdir)
        cond = bdir / "condition"
        master, slave = _copy_condition(frozen_dir, cond, batch, args.detect_channel)
        bcsv = (bdir / "detections.csv").resolve()
        ids = [str(e["image_id"]) for e in batch]
        print(f"\n[batch {i:03d}] {len(batch)} image(s): {ids[0]} .. {ids[-1]}  -> {bcsv}")

        rc = _run_matlab(args, cond, master, slave, bcsv)
        if rc != 0 or not bcsv.exists():
            print(f"[error] batch {i:03d} ({master} master) FAILED (rc={rc}); "
                  f"artifacts kept for debugging at {bdir}", file=sys.stderr)
            return rc or 1
        n_rows = len(load_normalized_detections(bcsv))
        print(f"[batch {i:03d}] OK: {len(batch)} images ({ids[0]} .. {ids[-1]}), {n_rows} detections")
        batch_csvs.append(bcsv)

    # concatenate per-batch normalized CSVs, preserving image_id + column order
    frames = [load_normalized_detections(c) for c in batch_csvs]
    combined = pd.concat(frames, ignore_index=True)
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"\n[ok] concatenated {len(batch_csvs)} batch CSV(s) -> {out_csv} "
          f"({len(combined)} detections over {combined['image_id'].nunique()} image(s))")
    _validate_csv(out_csv)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-format",
                   choices=["normalized_csv", "external_matlab", "diagnose_sigma"],
                   default="normalized_csv")
    p.add_argument("--sweep-limits", default="4,8,16,32,64,full",
                   help="diagnose_sigma: comma-separated image-count prefixes to sweep "
                        "(use 'full' for the whole set)")
    p.add_argument("--detections-csv", default=None,
                   help="normalized CSV to validate (normalized_csv mode)")
    p.add_argument("--out", default=None, help="normalized CSV to write (external_matlab mode)")

    p.add_argument("--frozen-dir", default=None, help="frozen benchmark/test set (source RAW TIFFs)")
    p.add_argument("--work-dir", default=None,
                   help="scratch dir for the condition folder (default: <out>/../cme_work)")
    p.add_argument("--detect-channel", type=int, choices=[1, 2], default=2,
                   help="CME master/detect channel")
    p.add_argument("--limit", type=int, default=None,
                   help="external_matlab: prepare/detect only the first N frozen images "
                        "(manifest order) for a tiny smoke run; omit for the full set")
    p.add_argument("--batch-size", type=int, default=None,
                   help="external_matlab: split images into stable-order batches of N and run "
                        "CMEAnalysis once per batch (each auto-estimates sigma independently), "
                        "then concatenate per-batch CSVs into --out. Recommended: 64. Omit to "
                        "run the whole condition folder in one CMEAnalysis call (unchanged).")

    # external MATLAB wrapper location (OUTSIDE the repo and the CME source tree)
    p.add_argument("--cme-software-folder", default=None, help="CMEAnalysis source folder")
    p.add_argument("--matlab-wrapper-folder", default=None, help="folder holding the external wrapper")
    p.add_argument("--matlab-entrypoint", default="spotpipe_cme_detect")
    p.add_argument("--matlab-exe", default="matlab", help="MATLAB executable (default: matlab)")

    # microscope / PSF parameters -- NOT silently baked in; logged on every run.
    p.add_argument("--na", type=float, default=1.49, help="numerical aperture")
    p.add_argument("--magnification", type=float, default=108.0, help="objective magnification")
    p.add_argument("--pixel-size-m", type=float, default=6.5e-6, help="camera pixel size (m)")
    p.add_argument("--alpha", type=float, default=0.05, help="CME detection significance level")

    args = p.parse_args(argv)
    print(f"[params] NA={args.na}  magnification={args.magnification}  "
          f"pixel_size_m={args.pixel_size_m}  detect_channel={args.detect_channel}")

    if args.input_format == "diagnose_sigma":
        return _diagnose_sigma_sweep(args)

    if args.input_format == "normalized_csv":
        if not args.detections_csv:
            print("[error] normalized_csv mode requires --detections-csv", file=sys.stderr)
            return 2
        src = Path(args.detections_csv)
        _validate_csv(src)
        if args.out and Path(args.out) != src:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, args.out)
            print(f"[ok] copied validated CSV to {args.out}")
        return 0

    # external_matlab
    if not args.frozen_dir or not args.out:
        print("[error] external_matlab mode requires --frozen-dir and --out", file=sys.stderr)
        return 2
    frozen_dir = Path(args.frozen_dir)
    out_csv = Path(args.out)
    work_dir = Path(args.work_dir) if args.work_dir else out_csv.parent / "cme_work"

    if args.batch_size is not None:
        return _run_external_batched(args, frozen_dir, work_dir, out_csv)

    cond, master, slave = _layout_condition_folder(
        frozen_dir, work_dir, args.detect_channel, limit=args.limit,
    )
    return _run_matlab(args, cond, master, slave, out_csv)


if __name__ == "__main__":
    raise SystemExit(main())
