#!/usr/bin/env python
"""Run Spotiflow detection over the frozen set and write a normalized CSV.

This is the ONE place that imports the external ``spotiflow`` package. It runs in
its own environment (``.venvs/spotiflow``), NOT the core spotpipe env -- nothing in
``spotpipe`` imports Spotiflow. It is a detector/localizer pass ONLY: it writes spot
COORDINATES, never intensities and never the canonical schema. Intensity extraction
and canonical emission happen later, in the harness, via
``spotiflow_*_plus_aperture`` (see ``src/spotpipe/benchmark/spotiflow.py``).

What it does, per image of the frozen benchmark/test set:

  * build the RAW detector image (default ``detect_image=raw_max``: the pixelwise
    max of the two raw channels, as float32) -- detection may use raw images;
  * run Spotiflow on that single 2D image;
  * convert Spotiflow's ``(y, x)`` point order to spotpipe ``(x, y)`` (via the
    shared, unit-tested ``spotpipe.benchmark.spotiflow.spots_yx_to_xy`` helper);
  * record ``p_detect`` (heatmap probability) when Spotiflow exposes it, else NaN.

It NEVER reads ``audit/`` and never touches the photon images (those are for fair
intensity extraction downstream, not detection).

Fairness note: the frozen set must not be used for model selection / threshold
tuning / fine-tuning. This script only RUNS a fixed detector over it; it tunes
nothing. The fine-tuned model is trained separately, on synthetic data only
(``scripts/prepare_spotiflow_finetune_data.py``).

Environment + examples (Windows PowerShell / POSIX): see the module docstring of
``src/spotpipe/benchmark/spotiflow.py`` and the transfer context. Typical run::

    .venvs/spotiflow/bin/python scripts/run_spotiflow_predict.py \
        --frozen-dir data/benchmark_test_v1 \
        --out outputs/external/spotiflow/general_raw_max/detections.csv \
        --model-variant general --pretrained-model general --detect-image raw_max
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable from the separate spotiflow venv too (no sys.path hacks for shared code).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark.harness import load_frozen_benchmark_set
from spotpipe.benchmark.spotiflow import spots_yx_to_xy

# Normalized detections CSV columns (the adapter's input contract).
CSV_COLUMNS = ["image_id", "x", "y", "p_detect", "source", "model_variant", "detect_image"]


def build_detect_image(image: np.ndarray, detect_image: str) -> np.ndarray:
    """Collapse the two raw channels into the single 2D image Spotiflow detects on.

    ``image`` is ``[2, H, W]`` raw detector counts. ``raw_max`` (the default and
    recommended first protocol) is the pixelwise max so a spot bright in EITHER
    channel can be found. Other protocols are wired for later experiments.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[0] < 2:
        raise ValueError(f"expected a [2, H, W] raw image; got shape {image.shape}")
    ch1 = image[0].astype(np.float32)
    ch2 = image[1].astype(np.float32)
    if detect_image == "raw_max":
        return np.maximum(ch1, ch2).astype(np.float32)
    if detect_image == "raw_sum":
        return (ch1 + ch2).astype(np.float32)
    if detect_image == "master_ch1":
        return ch1
    if detect_image == "master_ch2":
        return ch2
    raise ValueError(
        f"unknown --detect-image {detect_image!r}; supported: "
        "raw_max | raw_sum | master_ch1 | master_ch2"
    )


def _load_model(model_variant: str, pretrained_model: str, model_dir: str | None):
    """Lazily import Spotiflow and load the requested detector.

    ``general`` -> ``Spotiflow.from_pretrained(pretrained_model)``;
    ``finetuned_spotpipe_synth`` -> ``Spotiflow.from_folder(model_dir)``.
    """
    from spotiflow.model import Spotiflow  # external dependency; only imported here

    if model_variant == "general":
        return Spotiflow.from_pretrained(pretrained_model)
    if model_variant == "finetuned_spotpipe_synth":
        if not model_dir:
            raise ValueError(
                "--model-dir is required for --model-variant finetuned_spotpipe_synth "
                "(the folder produced by spotiflow-train)."
            )
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"fine-tuned model dir {model_dir!r} not found; train it first with "
                "scripts/prepare_spotiflow_finetune_data.py + spotiflow-train."
            )
        return Spotiflow.from_folder(model_dir)
    raise ValueError(
        f"unknown --model-variant {model_variant!r}; "
        "use 'general' or 'finetuned_spotpipe_synth'."
    )


def _extract_probs(details, n: int) -> np.ndarray:
    """Best-effort per-spot probability from Spotiflow's ``details``; NaN if absent.

    Spotiflow's prediction ``details`` object is not part of a stable public
    contract for per-point confidence, so probe a few likely attributes and only
    use one whose length matches the spot count. Otherwise return NaN -- the
    downstream adapter treats missing p_detect as a constant confidence.
    """
    for attr in ("prob", "probs", "probabilities", "intens", "intensities"):
        v = getattr(details, attr, None)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size == n:
            return arr
    return np.full(n, np.nan, dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Spotiflow detection -> normalized CSV.")
    parser.add_argument("--frozen-dir", required=True, help="frozen benchmark/test set dir")
    parser.add_argument("--out", required=True, help="output normalized detections CSV path")
    parser.add_argument("--model-variant", default="general",
                        choices=["general", "finetuned_spotpipe_synth"])
    parser.add_argument("--pretrained-model", default="general",
                        help="pretrained model name for --model-variant general")
    parser.add_argument("--model-dir", default=None,
                        help="custom model folder for --model-variant finetuned_spotpipe_synth")
    parser.add_argument("--detect-image", default="raw_max",
                        choices=["raw_max", "raw_sum", "master_ch1", "master_ch2"])
    parser.add_argument("--limit", type=int, default=None, help="first N images only (smoke)")
    args = parser.parse_args(argv)

    # Detection uses the RAW image only; never load photon TIFFs here.
    eval_set = load_frozen_benchmark_set(args.frozen_dir, limit=args.limit, with_photon=False)
    print(f"[spotiflow] loaded {len(eval_set)} image(s) from {args.frozen_dir}")

    model = _load_model(args.model_variant, args.pretrained_model, args.model_dir)
    print(f"[spotiflow] model loaded: variant={args.model_variant}")

    rows: list[dict] = []
    for item in eval_set:
        detect = build_detect_image(item.image, args.detect_image)
        spots, details = model.predict(detect)
        xs, ys = spots_yx_to_xy(spots)            # (y,x) image order -> spotpipe (x,y)
        probs = _extract_probs(details, xs.size)
        for x, y, p in zip(xs, ys, probs):
            rows.append({
                "image_id": str(item.image_id),
                "x": float(x),
                "y": float(y),
                "p_detect": float(p),
                "source": "spotiflow",
                "model_variant": args.model_variant,
                "detect_image": args.detect_image,
            })
        print(f"[spotiflow] {item.image_id}: {xs.size} detections")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(out_path, index=False)
    print(f"[spotiflow] wrote {len(df)} detections to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
