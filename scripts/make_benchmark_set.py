#!/usr/bin/env python
"""Generate (or verify) the FROZEN, stratified benchmark/test dataset.

Thin CLI over :mod:`spotpipe.simulator.benchmark_set`. Produces the SEPARATE,
FROZEN, externally-ingestible benchmark/test set used ONLY for final reporting
and external-method comparison -- NEVER for checkpoint selection. It is the
gating artifact every method (our HRNet, the aperture/PSF-fit baselines, DECODE,
Spotiflow, ...) runs on, identically.

This does NOT run any model -- it only produces and verifies the dataset.

Examples
--------
Generate the default frozen set and print the stratification report::

    uv run python scripts/make_benchmark_set.py --config configs/benchmark_test_set.yaml

Re-verify an already-generated set against its manifest checksums::

    uv run python scripts/make_benchmark_set.py --verify --out data/benchmark_test_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable straight from a fresh checkout too (no sys.path hacks for shared code).
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _post_generation_checks(out_dir: Path, manifest: dict) -> None:
    """Round-trip + export + checksum confirmations required by the prompt."""
    from spotpipe.schema import dataframe_to_records, read_spots, records_to_dataframe
    from spotpipe.simulator.benchmark_set import verify_benchmark_set

    print("\n" + "=" * 72)
    print("ROUND-TRIP / EXPORT / CHECKSUM CONFIRMATIONS")
    print("=" * 72)

    # 1. GT schema round-trips through spotpipe.schema
    recs = read_spots(out_dir / "ground_truth.csv")
    df = records_to_dataframe(recs)
    recs2 = dataframe_to_records(df)
    assert len(recs) == len(recs2) == manifest["n_gt_spots"], "GT schema round-trip count mismatch"
    assert all("ground_truth" in r.flags for r in recs[:50]), "GT rows missing 'ground_truth' flag"
    print(f"[schema]  ground_truth.csv round-trips through spotpipe.schema: "
          f"{len(recs)} rows, canonical columns OK, flags include 'ground_truth'.")

    # 2/3. Open one raw + one photon file; check shape / dtype / range
    img0 = manifest["images"][0]["image_id"]
    raw = tifffile.imread(out_dir / "images_ch1_raw" / f"{img0}.tif")
    print(f"[raw]     images_ch1_raw/{img0}.tif: shape={raw.shape}, dtype={raw.dtype}, "
          f"range=[{int(raw.min())},{int(raw.max())}]")
    assert raw.dtype == np.uint16, "raw export must be uint16"
    assert raw.min() >= 0 and raw.max() <= manifest["detector"]["adc_max"], "raw out of [0, adc_max]"

    pho = tifffile.imread(out_dir / "images_ch1_photon" / f"{img0}.tif")
    print(f"[photon]  images_ch1_photon/{img0}.tif: shape={pho.shape}, dtype={pho.dtype}, "
          f"range=[{pho.min():.2f},{pho.max():.2f}]")
    assert np.issubdtype(pho.dtype, np.floating), "photon export must be float"

    canon = np.load(out_dir / "images" / f"{img0}.npy")
    print(f"[canon]   images/{img0}.npy: shape={canon.shape}, dtype={canon.dtype} (two-channel)")
    assert canon.shape[0] == 2 and canon.dtype == np.uint16

    # 4. Manifest checksums written + verify
    assert (out_dir / "checksums.sha256").exists(), "checksums file missing"
    ok, problems = verify_benchmark_set(out_dir)
    print(f"[freeze]  checksums.sha256 written; verify_benchmark_set -> "
          f"{'OK (matches manifest)' if ok else 'PROBLEMS: ' + '; '.join(problems)}")
    assert ok, "freshly generated set failed its own checksum verification"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/benchmark_test_set.yaml",
                        help="benchmark-set generation config (default: configs/benchmark_test_set.yaml)")
    parser.add_argument("--out", default=None,
                        help="output dir (overrides out_dir in the config)")
    parser.add_argument("--verify", action="store_true",
                        help="verify an already-generated set against its manifest checksums and exit")
    args = parser.parse_args(argv)

    from spotpipe.simulator.benchmark_set import (
        generate_benchmark_set,
        stratification_report,
        verify_benchmark_set,
    )

    if args.verify:
        if args.out:
            out_dir = _resolve(args.out)
        else:
            cfg = _load_yaml(_resolve(args.config))
            out_dir = _resolve(cfg.get("out_dir", "data/benchmark_test_v1"))
        ok, problems = verify_benchmark_set(out_dir)
        print(f"verify {out_dir}: {'OK -- matches manifest checksums' if ok else 'FAILED'}")
        for p in problems:
            print(f"  - {p}")
        return 0 if ok else 1

    cfg = _load_yaml(_resolve(args.config))
    base_config = _load_yaml(_resolve(cfg["base_config"]))
    bench_cfg = cfg["benchmark_set"]
    out_dir = _resolve(args.out) if args.out else _resolve(cfg.get("out_dir", "data/benchmark_test_v1"))

    manifest = generate_benchmark_set(base_config, bench_cfg, out_dir, log_fn=print)

    # Stratification report (required output).
    print("\n" + stratification_report(manifest))

    # Post-generation round-trip / export / checksum confirmations.
    _post_generation_checks(out_dir, manifest)

    if manifest["underpopulated_cells"]:
        print("\nFAILURE: benchmark set is underpopulated (see report above). "
              "Increase max_attempts or adjust profiles; the set is NOT valid to freeze.",
              file=sys.stderr)
        return 1

    print(f"\nFrozen benchmark/test set ready: {out_dir}")
    print("  (Do NOT wire into checkpoint selection. No model has been run on it.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
