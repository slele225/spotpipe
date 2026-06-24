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
"""

from __future__ import annotations

import argparse
import json
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


def _layout_condition_folder(
    frozen_dir: Path, work_dir: Path, detect_channel: int, limit: int | None = None,
) -> tuple[Path, str, str]:
    """Build a clean 2-channel condition folder from the frozen RAW TIFFs.

    Layout (what loadConditionData expects): ``<cond>/<image_id>/ch1/<id>.tif`` and
    ``<cond>/<image_id>/ch2/<id>.tif``. The master channel is whichever the
    ``detect_channel`` selects; both raw channels are copied verbatim (no 3-page
    convention, no photometric changes). When ``limit`` is set, only the first N
    images (manifest order) are prepared -- for a tiny CME smoke run without
    touching the full frozen set. Returns (cond_dir, master_name, slave_name).
    """
    with open(frozen_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    master = f"ch{int(detect_channel)}"
    slave = "ch2" if master == "ch1" else "ch1"
    cond = work_dir / "condition"
    cond.mkdir(parents=True, exist_ok=True)

    entries = select_image_entries(manifest["images"], limit)
    image_ids = [str(e["image_id"]) for e in entries]
    scope = f"--limit {limit}" if limit is not None else "full set"
    print(f"[layout] selected {len(image_ids)} of {len(manifest['images'])} image(s) ({scope}).")
    print(f"[layout] image_ids: {image_ids}")

    for entry in entries:
        image_id = str(entry["image_id"])
        for ch_key, ch_name in (("ch1_raw", "ch1"), ("ch2_raw", "ch2")):
            src = frozen_dir / entry[ch_key]
            dst_dir = cond / image_id / ch_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst_dir / f"{image_id}.tif")
    print(f"[layout] wrote a {len(image_ids)}-image 2-channel condition folder to {cond} "
          f"(master={master}, slave={slave})")
    return cond, master, slave


def _run_matlab(args, cond_dir: Path, master: str, slave: str, out_csv: Path) -> int:
    """Invoke the external MATLAB wrapper, with BOTH folders on the MATLAB path."""
    if not args.cme_software_folder or not args.matlab_wrapper_folder:
        print("[error] external_matlab requires --cme-software-folder and "
              "--matlab-wrapper-folder.", file=sys.stderr)
        return 2

    # MATLAB's path utilities (getDirFromPath) assume absolute paths, so resolve
    # everything we hand off to absolute form (a relative condDir was the cause of
    # the "Array indices must be positive integers" failure in getDirFromPath).
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
    print(f"[matlab] NA={args.na}  M={args.magnification}  pixelSize_m={args.pixel_size_m}  "
          f"Alpha={args.alpha}")
    print(f"[matlab] entrypoint={args.matlab_entrypoint}")
    print(f"[matlab] software folder : {sw}")
    print(f"[matlab] wrapper  folder : {wrap}")
    print(f"[matlab] command: matlab -batch \"{matlab_cmd}\"")

    proc = subprocess.run([args.matlab_exe, "-batch", matlab_cmd])
    if proc.returncode != 0:
        print(f"[error] MATLAB exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode
    if not out_csv.exists():
        print(f"[error] MATLAB did not produce {out_csv}", file=sys.stderr)
        return 1
    _validate_csv(out_csv)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-format", choices=["normalized_csv", "external_matlab"],
                   default="normalized_csv")
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
    cond, master, slave = _layout_condition_folder(
        frozen_dir, work_dir, args.detect_channel, limit=args.limit,
    )
    return _run_matlab(args, cond, master, slave, out_csv)


if __name__ == "__main__":
    raise SystemExit(main())
