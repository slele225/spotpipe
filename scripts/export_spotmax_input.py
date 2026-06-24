#!/usr/bin/env python
"""Export a SpotMAX-compatible input tree from the frozen benchmark set.

Stage 2 of the SpotMAX adapter. This does NOT import or run SpotMAX -- it only
prepares the on-disk inputs SpotMAX will later detect on (in its own env, via
``spotmax -p config.ini``). It is a DETECTOR INPUT step only: it writes RAW-derived
detection TIFFs, never photon images and never the canonical schema.

What it does, per image of the frozen benchmark/test set (first ``--n-images``):

  * read the two RAW channels (``images_ch{1,2}_raw/<id>.tif``);
  * build the detection image (default ``--detect-image raw_max``: the pixelwise
    max of the two raw channels, as float32) -- detection may use raw images, and
    raw_max lets SpotMAX find a spot bright in EITHER channel;
  * write it into a Cell-ACDC / SpotMAX Position tree::

        <out>/input/Position_000001/Images/Position_000001_spots.tif
        <out>/input/Position_000002/Images/Position_000002_spots.tif
        ...

  * write ``<out>/id_map.csv`` (position <-> benchmark image_id <-> source files),
    which ``scripts/convert_spotmax_output.py`` uses to map SpotMAX outputs back
    to canonical image ids;
  * write a best-effort headless ``<out>/config.ini`` template.

It NEVER reads ``audit/`` and never touches the photon images (those are for fair
intensity extraction downstream, in the harness adapter, not detection).

The INI is a STARTING TEMPLATE: SpotMAX's INI schema is version-specific, so the
generated file documents the key fields and you may need to regenerate it from
the SpotMAX GUI for your installed version. The parser
(``scripts/convert_spotmax_output.py``) does NOT assume SpotMAX's output column
names -- inspect ``SpotMAX_output/`` after the smoke run.

Typical run::

    uv run python scripts/export_spotmax_input.py \
        --benchmark data/benchmark_test_v1 \
        --out external_runs/spotmax/smoke --n-images 5 --detect-image raw_max
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

# Endname (suffix before the extension) of the detection channel file. The INI's
# spots-channel endname must match this. Single source of truth.
SPOTS_ENDNAME = "spots"


def build_detect_image(ch1: np.ndarray, ch2: np.ndarray, detect_image: str) -> np.ndarray:
    """Collapse the two raw channels into the single 2-D image SpotMAX detects on.

    ``raw_max`` (default + recommended first protocol) is the pixelwise max so a
    spot bright in EITHER channel can be found. Other protocols are wired for
    later ablations. Output is float32 (SpotMAX reads standard image dtypes).
    """
    a = np.asarray(ch1, dtype=np.float32)
    b = np.asarray(ch2, dtype=np.float32)
    if detect_image == "raw_max":
        return np.maximum(a, b)
    if detect_image == "raw_sum":
        return a + b
    if detect_image == "master_ch1":
        return a
    if detect_image == "master_ch2":
        return b
    raise ValueError(
        f"unknown --detect-image {detect_image!r}; supported: "
        "raw_max | raw_sum | master_ch1 | master_ch2"
    )


def _position_name(index: int) -> str:
    """1-based SpotMAX Position folder name (``Position_000001`` ...)."""
    return f"Position_{index + 1:06d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export SpotMAX detection inputs from the frozen set.")
    parser.add_argument("--benchmark", required=True, help="frozen benchmark/test set dir")
    parser.add_argument("--out", required=True, help="output run dir (e.g. external_runs/spotmax/smoke)")
    parser.add_argument("--n-images", type=int, default=5, help="number of images to export (smoke subset)")
    parser.add_argument("--detect-image", default="raw_max",
                        choices=["raw_max", "raw_sum", "master_ch1", "master_ch2"])
    args = parser.parse_args(argv)

    import tifffile

    bench = Path(args.benchmark)
    out = Path(args.out)
    input_root = out / "input"
    input_root.mkdir(parents=True, exist_ok=True)

    with open(bench / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = manifest["images"][: int(args.n_images)]
    if not entries:
        print("[export] no images to export (empty manifest / n-images=0)")
        return 1

    id_rows: list[dict] = []
    for i, entry in enumerate(entries):
        image_id = str(entry["image_id"])
        position = _position_name(i)
        ch1 = tifffile.imread(bench / entry["ch1_raw"])
        ch2 = tifffile.imread(bench / entry["ch2_raw"])
        detect = build_detect_image(ch1, ch2, args.detect_image)

        images_dir = input_root / position / "Images"
        images_dir.mkdir(parents=True, exist_ok=True)
        tif_path = images_dir / f"{position}_{SPOTS_ENDNAME}.tif"
        tifffile.imwrite(tif_path, detect.astype(np.float32))

        id_rows.append({
            "position": position,
            "image_id": image_id,
            "detect_image": args.detect_image,
            "spots_tif": str(tif_path.relative_to(out)),
            "src_ch1_raw": entry["ch1_raw"],
            "src_ch2_raw": entry["ch2_raw"],
            "height": int(detect.shape[0]),
            "width": int(detect.shape[1]),
        })
        print(f"[export] {position} <- {image_id}: wrote {tif_path.name} {detect.shape}")

    id_map_path = out / "id_map.csv"
    pd.DataFrame(id_rows).to_csv(id_map_path, index=False)
    print(f"[export] wrote id_map: {id_map_path} ({len(id_rows)} positions)")

    ini_path = out / "config.ini"
    _write_ini_template(ini_path, input_root, args.detect_image)
    print(f"[export] wrote headless INI template: {ini_path}")
    print(
        "[export] NEXT: in a SEPARATE SpotMAX env run\n"
        f"           spotmax -p {ini_path}\n"
        "         then inspect the SpotMAX_output/ tables and run "
        "scripts/convert_spotmax_output.py."
    )
    return 0


def _write_ini_template(path: Path, input_root: Path, detect_image: str) -> None:
    """Write a conservative headless SpotMAX INI *template*.

    SpotMAX's INI schema is version-specific; treat this as a starting point and,
    if SpotMAX rejects it, regenerate from the SpotMAX GUI for your installed
    version (the data tree under ``input/`` is already SpotMAX-compatible). The
    key fields are the spots-channel endname (must match the exported TIFFs), 2-D
    single-frame metadata, and CSV output.
    """
    folder = str(input_root.resolve())
    text = f"""# SpotMAX headless workflow -- TEMPLATE (auto-generated by export_spotmax_input.py)
# Detection input: {detect_image} of the two raw channels, exported as
#   <Position>/Images/<Position>_{SPOTS_ENDNAME}.tif
# SpotMAX is used as a DETECTOR/LOCALIZER ONLY; native intensities are not used.
# NOTE: section/parameter names are version-specific -- if SpotMAX errors, open the
# data folder in the SpotMAX GUI, set parameters, and export a fresh INI.

[File paths and channels]
Folder path = {folder}
Spots channel end name = {SPOTS_ENDNAME}
Segmentation end name =
Reference channel end name =

[METADATA]
SizeT = 1
SizeZ = 1
Analyse a single frame = True

[Pre-processing]
Aggregate cells prior detection = False

[Spots channel]
Spots detection method = spotMAX AI

[SpotFIT]
Compute spots size (fit gaussian peak(s)) = False

[Configuration]
Use GUI = False
Stop analysis on critical error = True
Save output files = True
Text file extension for output tables = .csv
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
