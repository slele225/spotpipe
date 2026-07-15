"""Command-line entry point. ``spotpipe smoke`` now; train/bench in later stages."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

__all__ = ["main", "run_smoke", "run_bench_gen", "run_infer", "run_train", "run_evaluate"]


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
    peak_threshold: float | None = None,
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
        peak_threshold=peak_threshold,
    )
    return results_root


def run_evaluate(
    results: str | Path | None = None,
    benchmark: str | Path | None = None,
    out: str | Path | None = None,
    *,
    methods: list[str] | None = None,
    match_radius_sigma: float = 1.0,
    oracle: bool = False,
) -> Path:
    """Run the ONE shared, blind evaluator over a results root -> metrics CSVs.

    Ingests ``<results>/<method>/{snr_density,curvature}/.../predictions.csv``
    (whatever ``spotpipe infer`` and the baselines emit) plus the benchmark GT and
    writes ``<out>/<method>/metrics_by_condition.csv`` + ``alpha_recovery.csv``,
    the combined cross-method tables, and ``summary_by_method.csv``. Tool-agnostic:
    the same code path scores every method. With ``--oracle`` it instead scores the
    GROUND TRUTH as its own predictions (Gate A calibration) and writes an
    ``oracle_gt/`` folder. See :mod:`spotpipe.benchmark.evaluate`.
    """
    import time

    from spotpipe.benchmark.evaluate import evaluate_all, ground_truth_as_predictions
    from spotpipe.paths import get_paths

    paths = get_paths()
    bench_root = Path(benchmark) if benchmark else paths.dataset("benchmark")
    out_dir = Path(out) if out else paths.root / "results"

    t0 = time.perf_counter()
    if oracle:
        metrics, alpha = ground_truth_as_predictions(bench_root, match_radius_sigma=match_radius_sigma)
        mdir = out_dir / "oracle_gt"
        mdir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(mdir / "metrics_by_condition.csv", index=False)
        alpha.to_csv(mdir / "alpha_recovery.csv", index=False)
        print(f"[eval] oracle GT-as-predictions -> {mdir}  ({time.perf_counter() - t0:.1f}s)")
        return out_dir

    results_root = Path(results) if results else paths.output("predictions")
    result = evaluate_all(bench_root, results_root, out_dir,
                          methods=methods, match_radius_sigma=match_radius_sigma)
    print(f"[eval] {len(result['methods'])} method(s) evaluated in "
          f"{time.perf_counter() - t0:.1f}s -> {out_dir}")
    return out_dir


def run_train(
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    *,
    mode: str = "run",
    device: str = "auto",
    steps: int | None = None,
    num_workers: int | None = None,
    require_gpu: bool = False,
    resume: bool = True,
) -> int:
    """Train the measured-detector hrnet_large model, or run one of its self-checks.

    ``mode`` is one of: ``run`` (the real staged run; needs ``--out``), ``overfit``
    (tiny fixed set; loss must collapse), ``smoke`` (short end-to-end loop),
    ``profile`` (dataload-vs-compute timing gate), ``solved-windows`` (print the
    CHANGE-2 solved-A1-range distribution). See src/spotpipe/training/train.py.
    """
    import json

    import torch

    from spotpipe.paths import get_paths
    from spotpipe.training.dataset import (
        DetectorConstants,
        IntensityWindowConfig,
        summarize_solved_windows,
    )
    from spotpipe.training.train import (
        load_train_config,
        overfit,
        profile_dataload,
        resolve_blocks,
        resolve_device,
        train,
    )

    paths = get_paths()
    default = "train_smoke.yaml" if mode in ("overfit", "smoke", "profile") else "train.yaml"
    config_path = Path(config_path) if config_path else paths.configs / default
    config = load_train_config(config_path)
    dev = resolve_device(device)
    print(f"[train] config={config_path} mode={mode} device={dev} "
          f"(cuda_available={torch.cuda.is_available()})")

    if mode == "solved-windows":
        shape, det_cfg, scene_cfg, _ = resolve_blocks(config)
        consts = DetectorConstants.from_config(det_cfg)
        wcfg = IntensityWindowConfig.from_config(config.get("training", {}))
        for t in (0.0, 0.5, 1.0):
            rep = summarize_solved_windows(scene_cfg, consts, wcfg, shape=shape,
                                           n_samples=3000, seed=int(config.get("seed", 0)), t=t)
            print(f"[solved-windows t={t}] " + json.dumps(rep, indent=2))
        return 0

    if mode == "profile":
        res = profile_dataload(config, device=device, num_workers=num_workers)
        print("[profile] " + json.dumps(res, indent=2))
        return 0 if res["gate_pass"] else 2

    if mode == "overfit":
        result = overfit(config, steps=steps or 300, device=device)
        final = result["final_eval"]
        print(f"[overfit] final logI1_mae={final['logI1_mae']:.4f} "
              f"logI2_mae={final['logI2_mae']:.4f} "
              f"loss_total={final.get('loss_total')}")
        return 0

    if mode == "smoke":
        result = train(config, device=device, out_dir=out_dir, steps=steps or 20,
                       num_workers=num_workers if num_workers is not None else 0,
                       resume=False)
        print(f"[smoke] trained {steps or 20} steps; dataload_fraction="
              f"{result['dataload_fraction']:.3f}")
        return 0

    # Real run.
    out = out_dir
    if out is None and config.get("experiment"):
        out = str(paths.output(f"train/{config['experiment'].get('name', 'run')}"))
    if out is None:
        print("[train] a real run needs --out (or an experiment.name in the config)")
        return 1
    result = train(config, device=dev, out_dir=out, steps=steps, num_workers=num_workers,
                   require_gpu=require_gpu, resume=resume)
    best = result["best"]
    print(f"[train] done: run_dir={result['run_dir']}  dataload_fraction="
          f"{result['dataload_fraction']:.3f}")
    print(f"[train] best: step={best.get('step')} by={best.get('metric')} "
          f"value={best.get('value')}")
    return 0


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
    infer.add_argument("--peak-threshold", type=float, default=None,
                       help="override the checkpoint's peak_threshold (retuned OFF-benchmark by "
                            "scripts/threshold_retune.py). Recorded in the run manifest.")

    ev = sub.add_parser("evaluate",
                        help="run the shared blind evaluator over prediction CSVs -> metrics")
    ev.add_argument("--results", default=None,
                    help="results root: <root>/<method>/... (default: outputs/predictions/)")
    ev.add_argument("--benchmark", default=None,
                    help="benchmark root dir with GT + BENCH_MANIFEST.json (default: data/benchmark/)")
    ev.add_argument("--out", default=None, help="metrics output dir (default: results/)")
    ev.add_argument("--method", action="append", default=None, dest="methods",
                    help="evaluate only this method folder (repeatable; default: all found)")
    ev.add_argument("--match-radius-sigma", type=float, default=1.0,
                    help="match gate in PSF sigma (default: 1.0 x max(sigma1,sigma2))")
    ev.add_argument("--oracle", action="store_true",
                    help="Gate A: score ground truth as its own predictions (calibration)")

    train_p = sub.add_parser("train",
                             help="train the measured-detector hrnet_large model (or a self-check)")
    train_p.add_argument("--config", default=None,
                         help="training yaml (default: configs/train.yaml; self-checks: train_smoke.yaml)")
    train_p.add_argument("--out", default=None, help="run output dir (default: outputs/train/<name>/)")
    train_p.add_argument("--mode", default="run",
                         choices=("run", "overfit", "smoke", "profile", "solved-windows"),
                         help="run (real staged run) | overfit | smoke | profile | solved-windows")
    train_p.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    train_p.add_argument("--steps", type=int, default=None, help="override the configured step count")
    train_p.add_argument("--num-workers", type=int, default=None,
                         help="DataLoader workers (default: cpu_count-2; 0 = inline)")
    train_p.add_argument("--require-gpu", action="store_true",
                         help="fail loud if the resolved device is not CUDA (for the GPU box)")
    train_p.add_argument("--no-resume", action="store_true",
                         help="ignore any train_state.pt and start fresh")

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
            peak_threshold=args.peak_threshold,
        )
        return 0
    if args.command == "evaluate":
        run_evaluate(
            args.results, args.benchmark, args.out,
            methods=args.methods, match_radius_sigma=args.match_radius_sigma,
            oracle=args.oracle,
        )
        return 0
    if args.command == "train":
        return run_train(
            args.config, args.out, mode=args.mode, device=args.device, steps=args.steps,
            num_workers=args.num_workers, require_gpu=args.require_gpu, resume=not args.no_resume,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
