#!/usr/bin/env python
"""Run the LEGACY ``v0_hrnet`` model over the frozen set -> canonical predictions CSV.

This is the ONE place that imports torch / timm and the legacy ``liposome-detect``
repository. It runs in its OWN environment (e.g. that repo's ``.venv`` with
spotpipe installed editable, or a dedicated ``.venvs/v0_hrnet``), NOT the core
spotpipe env -- nothing in ``spotpipe`` imports torch/timm or the legacy repo.

The legacy HRNet is an OLD model (hence ``v0``), trained on a DIFFERENT forward
model with DIFFERENT channel semantics (lipid/protein) and normalization. We run
it cross-domain, as-is, for an honest historical baseline. Unlike the
``*_plus_aperture`` detector-only methods, this model PREDICTS intensities and
per-spot uncertainties directly, so this script writes the FULL canonical schema
(:data:`spotpipe.schema.SCHEMA_COLUMNS`) -- there is no downstream aperture step.

Per image of the frozen benchmark/test set it:

  * loads the RAW two-channel image (``[2, H, W]``, ch1/ch2; never the photon TIFFs);
  * reorders channels to the legacy model's expected ``[protein, lipid]`` order,
    per ``--ch1-channel`` (default ``lipid`` -> ch1; the other channel -> ch2);
  * reuses the legacy repo's own ``load_model`` + ``decode_image_array`` (which
    normalize with the legacy config's ``norm_mean`` / ``norm_std`` and decode the
    heatmap with the config's decode params) -- so the architecture, normalization,
    and decode are byte-for-byte the legacy ones, not reimplemented here;
  * converts the legacy per-detection dicts to the canonical schema via the shared,
    unit-tested :func:`spotpipe.benchmark.v0_hrnet.detections_to_canonical`.

Units caveat: the legacy intensities are in the OLD simulator's flux units, not
spotpipe photon-proportional units (recorded as ``units=legacy_flux`` in every
row's flags). Detection metrics and the log-ratio slope stay meaningful; absolute
intensity bias does not. See ``src/spotpipe/benchmark/v0_hrnet.py``.

Fairness: this only RUNS a fixed legacy model over the frozen set; it tunes
nothing and never reads ``audit/`` or the photon/true-background files.

Typical run (Windows PowerShell)::

    python scripts/run_v0_hrnet_predict.py \
        --frozen-dir data/benchmark_test_v1 \
        --legacy-repo "C:\\Users\\shivl\\Music\\liposome-detect" \
        --legacy-config "C:\\Users\\shivl\\Music\\liposome-detect\\configs\\train\\hrnet_v1.yaml" \
        --checkpoint  "C:\\Users\\shivl\\Music\\liposome-detect\\models\\hrnet_v1\\best.pt" \
        --out outputs/external/v0_hrnet/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable from a separate legacy venv too (no sys.path hacks for shared code).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark.harness import load_frozen_benchmark_set
from spotpipe.benchmark.v0_hrnet import (
    CHANNEL_CHOICES,
    V0_HRNET_METHOD,
    detections_to_canonical,
)
from spotpipe.schema import SCHEMA_COLUMNS, write_spots


def _load_legacy(legacy_repo: str, legacy_config: str, checkpoint: str, device: str):
    """Put the legacy repo on ``sys.path`` and load its model via its own loader.

    Imports are LAZY and local so importing this script (e.g. for ``--help`` or in
    the smoke test) does not require torch / timm / the legacy repo to be present.
    Returns ``(model, cfg, device, decode_image_array)``.
    """
    repo = Path(legacy_repo)
    if not repo.exists():
        raise FileNotFoundError(f"--legacy-repo {legacy_repo!r} not found")
    # The legacy package is imported as ``src.*`` from its repo root.
    sys.path.insert(0, str(repo))
    try:
        from src.eval.matching import decode_image_array, load_model  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"could not import the legacy model from {legacy_repo!r}; this script "
            "must run in an environment with torch + timm installed and the legacy "
            "liposome-detect repo importable. Original error: " + repr(exc)
        ) from exc

    model, cfg, device = load_model(legacy_config, checkpoint, device=device)
    return model, cfg, device, decode_image_array


def _legacy_order_image(image: np.ndarray, ch1_channel: str) -> np.ndarray:
    """Reorder a spotpipe ``[ch1, ch2]`` raw image to the legacy ``[protein, lipid]``.

    The legacy model's input channel order is (0=protein, 1=lipid). With
    ``ch1_channel='lipid'`` spotpipe ch1 IS lipid, so ``[protein, lipid] =
    [ch2, ch1] = [image[1], image[0]]``; with ``ch1_channel='protein'`` it is
    ``[image[0], image[1]]``. Returned as float32 (what the legacy normalizer
    expects).
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] < 2:
        raise ValueError(f"expected a [2, H, W] raw image; got shape {image.shape}")
    ch1, ch2 = image[0], image[1]
    lipid = ch1 if ch1_channel == "lipid" else ch2
    protein = ch2 if ch1_channel == "lipid" else ch1
    return np.stack([protein, lipid], axis=0).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the legacy v0_hrnet model -> canonical predictions CSV."
    )
    parser.add_argument("--frozen-dir", required=True, help="frozen benchmark/test set dir")
    parser.add_argument("--out", required=True, help="output canonical predictions CSV path")
    parser.add_argument("--legacy-repo", required=True,
                        help="path to the legacy liposome-detect repo (added to sys.path)")
    parser.add_argument("--legacy-config", required=True,
                        help="legacy training config YAML (e.g. configs/train/hrnet_v1.yaml)")
    parser.add_argument("--checkpoint", required=True,
                        help="legacy checkpoint (best.pt) -- NEVER copied into this repo")
    parser.add_argument("--ch1-channel", default="lipid", choices=list(CHANNEL_CHOICES),
                        help="which legacy channel maps to spotpipe ch1/logI1 (default lipid)")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="override the legacy config decode score threshold")
    parser.add_argument("--nms-kernel", type=int, default=None,
                        help="override the legacy config decode NMS kernel")
    parser.add_argument("--max-detections", type=int, default=None,
                        help="override the legacy config decode max detections")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="first N images only (smoke)")
    args = parser.parse_args(argv)

    # Detection uses the RAW image only; never load photon TIFFs here.
    eval_set = load_frozen_benchmark_set(args.frozen_dir, limit=args.limit, with_photon=False)
    print(f"[v0_hrnet] loaded {len(eval_set)} image(s) from {args.frozen_dir}")

    model, cfg, device, decode_image_array = _load_legacy(
        args.legacy_repo, args.legacy_config, args.checkpoint, args.device
    )
    print(f"[v0_hrnet] legacy model loaded on {device}; ch1_channel={args.ch1_channel}")

    # Apply optional decode overrides onto the legacy cfg the loader returned, so
    # decode_image_array (which reads cfg['decode']) honours them.
    dec = dict(cfg.get("decode", {}))
    if args.score_threshold is not None:
        dec["score_threshold"] = float(args.score_threshold)
    if args.nms_kernel is not None:
        dec["nms_kernel"] = int(args.nms_kernel)
    if args.max_detections is not None:
        dec["max_detections"] = int(args.max_detections)
    cfg["decode"] = dec

    frames = []
    n_dets = 0
    for item in eval_set:
        arr = _legacy_order_image(item.image, args.ch1_channel)
        dets = decode_image_array(model, cfg, device, arr)
        df = detections_to_canonical(item.image_id, dets, ch1_channel=args.ch1_channel)
        frames.append(df)
        n_dets += len(df)
        print(f"[v0_hrnet] {item.image_id}: {len(df)} detections")

    pred = (
        pd.concat(frames, ignore_index=True)[list(SCHEMA_COLUMNS)]
        if frames else pd.DataFrame(columns=list(SCHEMA_COLUMNS))
    )
    out_path = write_spots(pred, args.out)
    print(f"[v0_hrnet] wrote {n_dets} {V0_HRNET_METHOD} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
