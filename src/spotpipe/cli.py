"""Command-line entry point. ``spotpipe smoke`` now; train/bench in later stages."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

__all__ = ["main", "run_smoke", "run_bench_gen", "run_infer"]


def run_smoke(config_path: str | Path | None = None, out_dir: str | Path | None = None) -> Path:
    """End-to-end smoke: simulate → GT schema CSVs → model forward → prediction schema CSVs.

    Returns the output directory. Everything under it:
      dataset/   vendored generate_dataset output (images/, spots/ = GT schema, meta/, manifest.json)
      predictions/pred_<image_id>.csv   canonical-schema model outputs (random weights)
    """
    # Imports deferred so `spotpipe --help` stays instant.
    import numpy as np
    import torch

    from spotpipe.config import load_config
    from spotpipe.models import build_spot_model, predict_spots
    from spotpipe.paths import get_paths
    from spotpipe.schema import SCHEMA_COLUMNS, write_spots
    from spotpipe.simulator.generate_dataset import generate_dataset

    paths = get_paths()
    config_path = Path(config_path) if config_path else paths.configs / "smoke.yaml"
    cfg = load_config(config_path)
    out = Path(out_dir) if out_dir else paths.output(cfg.run.split)
    dataset_dir = out / "dataset"
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    manifest = generate_dataset(
        cfg.simulator, dataset_dir,
        n_images=cfg.run.n_images, seed=cfg.run.seed, split=cfg.run.split,
    )
    t_sim = time.perf_counter() - t0
    n_gt = sum(e["n_spots"] for e in manifest["images"])
    print(f"[sim]   {cfg.run.n_images} images {manifest['shape']} -> {dataset_dir}")
    print(f"[sim]   {n_gt} ground-truth spots, {t_sim:.1f}s ({t_sim / cfg.run.n_images * 1e3:.0f} ms/img)")

    torch.manual_seed(cfg.run.seed)
    model = build_spot_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] built ({n_params / 1e6:.2f}M params, random weights)")

    t1 = time.perf_counter()
    n_pred = 0
    shape_printed = False
    for entry in manifest["images"]:
        image = np.load(dataset_dir / entry["image_file"])["image"]  # uint16 [2,H,W]
        if not shape_printed:
            print(f"[model] input image shape {image.shape} dtype {image.dtype}")
            shape_printed = True
        df = predict_spots(
            model, image,
            image_id=entry["image_id"],
            adc_max=cfg.inference.adc_max,
            peak_threshold=cfg.inference.peak_threshold,
            nms_kernel=cfg.inference.nms_kernel,
            max_spots=cfg.inference.max_spots,
        )
        assert list(df.columns) == list(SCHEMA_COLUMNS)
        write_spots(df, pred_dir / f"pred_{entry['image_id']}.csv")
        n_pred += len(df)
    t_fwd = time.perf_counter() - t1

    total = time.perf_counter() - t0
    print(f"[model] forward+emit on {cfg.run.n_images} images: {t_fwd:.1f}s "
          f"({t_fwd / cfg.run.n_images * 1e3:.0f} ms/img), {n_pred} predicted spots")
    print(f"[out]   GT: {dataset_dir / 'spots'}  predictions: {pred_dir}")
    print(f"[total] {total:.1f}s")
    return out


def run_bench_gen(config_path: str | Path | None = None, out_dir: str | Path | None = None) -> Path:
    """Generate the two-family benchmark image sets + ground truth.

    Generation ONLY: no method runs, no slope is fit, no metric is computed. Writes
    the portable ``snr_density/`` + ``curvature/`` directory artifact plus
    ``BENCH_MANIFEST.json``. Returns the benchmark root directory.
    """
    from spotpipe.benchmark.generate import generate_benchmark, load_benchmark_config
    from spotpipe.paths import get_paths

    paths = get_paths()
    config_path = Path(config_path) if config_path else paths.configs / "benchmark.yaml"
    base_config, cfg = load_benchmark_config(config_path)
    # A benchmark set is a portable, syncable directory artifact -> under data/
    # (moved by scripts/sync_to_remote.sh; CLAUDE.md rule 6), never a machine path.
    out = Path(out_dir) if out_dir else paths.dataset("benchmark")
    generate_benchmark(base_config, cfg, out)
    return out


def run_infer(
    checkpoint: str = "all",
    benchmark: str | Path | None = None,
    out: str | Path | None = None,
    *,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int | None = None,
    smoke: bool = False,
) -> Path:
    """Run a trained checkpoint (or ``all``) over the benchmark -> prediction CSVs.

    Emits one schema-conforming ``predictions.csv`` per condition per method under
    ``<out>/<method>/`` plus a ``RUN_MANIFEST.json``. Resumable (skip-if-exists),
    incremental, and parallel-loaded. Modifies nothing vendored. Returns the
    results root. See :mod:`spotpipe.benchmark.infer` for the full contract.
    """
    from spotpipe.benchmark.infer import run_inference
    from spotpipe.paths import get_paths

    paths = get_paths()
    bench_root = Path(benchmark) if benchmark else paths.dataset("benchmark")
    results_root = Path(out) if out else paths.output("predictions")
    run_inference(
        checkpoint,
        bench_root=bench_root,
        results_root=results_root,
        repo_root=paths.root,
        checkpoints_root=paths.checkpoints,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        smoke=smoke,
    )
    return results_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spotpipe")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="tiny end-to-end run: simulate -> forward -> schema CSVs")
    smoke.add_argument("--config", default=None, help="yaml config (default: configs/smoke.yaml)")
    smoke.add_argument("--out", default=None, help="output dir (default: outputs/<split>/)")

    bench = sub.add_parser("bench-gen",
                           help="generate benchmark image sets + GT (no method / no fit / no metric)")
    bench.add_argument("--config", default=None,
                       help="benchmark yaml (default: configs/benchmark.yaml; smoke: configs/benchmark_smoke.yaml)")
    bench.add_argument("--out", default=None, help="benchmark root dir (default: data/benchmark/)")

    infer = sub.add_parser("infer",
                           help="run trained checkpoint(s) over the benchmark -> prediction CSVs")
    infer.add_argument("--checkpoint", default="all",
                       help="checkpoint name (e.g. hrnet_large) or 'all' for both (default: all)")
    infer.add_argument("--benchmark", default=None,
                       help="benchmark root dir (default: data/benchmark/)")
    infer.add_argument("--out", default=None,
                       help="results root dir (default: outputs/predictions/)")
    infer.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                       help="compute device; 'cuda' aborts if no GPU (default: auto)")
    infer.add_argument("--batch-size", type=int, default=8, help="forward-pass batch size")
    infer.add_argument("--num-workers", type=int, default=None,
                       help="DataLoader workers (default: min(cpu_count, 8))")
    infer.add_argument("--smoke", action="store_true",
                       help="tiny subset (few conditions x few images) for a fast correctness check")

    args = parser.parse_args(argv)

    if args.command == "smoke":
        run_smoke(args.config, args.out)
        return 0
    if args.command == "bench-gen":
        run_bench_gen(args.config, args.out)
        return 0
    if args.command == "infer":
        run_infer(
            args.checkpoint, args.benchmark, args.out,
            device=args.device, batch_size=args.batch_size,
            num_workers=args.num_workers, smoke=args.smoke,
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
