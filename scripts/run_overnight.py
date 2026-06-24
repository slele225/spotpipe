#!/usr/bin/env python
"""Chain the two phase-5b overnight training runs (small -> large), unattended.

Runs `hrnet-small` then `hrnet-large` sequentially in-process, each with the FULL
40k-step schedule, best-checkpoint selection on the FIXED shared validation set, and
an end-of-run auto-benchmark of the selected checkpoint. Designed to be launched once
on the A100 and left alone overnight.

Robustness (prompt 5b):
  * device auto-detects CUDA (the A100); falls back to CPU.
  * each run logs to ``<exp>/outputs/train.log`` (tee'd to stdout) AND checkpoints +
    a resumable ``train_state.pt`` every ``checkpoint_every`` steps -- a crash resumes
    mid-stream instead of restarting from zero (re-run this script to resume).
  * a finished run drops ``<exp>/outputs/RUN_COMPLETE.json``; re-running is a no-op
    for it (unless ``--force``), so the chain is safe to relaunch.
  * by default a crash in the small run still lets the large run proceed
    (``--stop-on-failure`` to gate instead). The chained order is preserved.
  * param counts for BOTH models are printed up front, before any training.

Usage::

    # one-shot: confirm setup without training
    uv run python scripts/run_overnight.py --dry-run

    # the overnight launch (small -> large)
    uv run python scripts/run_overnight.py
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from spotpipe.models.spot_model import build_spot_model
from spotpipe.simulator.generate_dataset import _git_commit
from spotpipe.training.train import load_train_config, resolve_blocks, resolve_device, train

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGES = [
    REPO_ROOT / "experiments" / "2026-06-23_hrnet-small" / "config.yaml",
    REPO_ROOT / "experiments" / "2026-06-23_hrnet-large" / "config.yaml",
]


def _param_count(model_cfg: dict) -> int:
    return sum(p.numel() for p in build_spot_model(model_cfg).parameters())


class _TeeLogger:
    """Minimal tee: write each line to stdout AND a per-run log file (append)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self._fh.write(str(msg) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _check_val_set(config: dict) -> str | None:
    """Return the resolved val-set dir if configured & present, else None."""
    val_path = config.get("training", {}).get("val", {}).get("path")
    if not val_path:
        return None
    d = Path(val_path)
    if not d.is_absolute():
        d = REPO_ROOT / d
    return str(d) if (d / "manifest.json").exists() else None


def run_stage(config_path: Path, *, device, resume: bool, force: bool, dry_run: bool) -> dict:
    config = load_train_config(config_path)
    exp_dir = config_path.resolve().parent
    out_dir = exp_dir / "outputs"
    name = config.get("experiment", {}).get("name", exp_dir.name)
    model_cfg = config.get("model", {})
    nparams = _param_count(model_cfg)
    done_marker = out_dir / "RUN_COMPLETE.json"

    print("=" * 78)
    print(f"STAGE: {name}")
    print(f"  config       : {config_path}")
    print(f"  params       : {nparams:,}  (base_channels={model_cfg.get('base_channels')})")
    print(f"  train_steps  : {config.get('training', {}).get('train_steps')}")
    print(f"  out_dir      : {out_dir}")
    val_dir = _check_val_set(config)
    print(f"  val set      : {val_dir or 'MISSING -- build with scripts/build_fixed_eval.py'}")
    print(f"  device       : {device}")
    print("=" * 78)

    if dry_run:
        return {"name": name, "params": nparams, "status": "dry-run", "val_set": val_dir}

    if val_dir is None:
        raise FileNotFoundError(
            f"{name}: validation set not found. Build it once first:\n"
            f"  uv run python scripts/build_fixed_eval.py --config {config_path} "
            f"--out data/fixed_eval/val --split val --seed 70001"
        )

    if done_marker.exists() and not force:
        print(f"[{name}] already complete ({done_marker}); skipping (use --force to rerun).")
        return {"name": name, "params": nparams, "status": "skipped-complete",
                **json.loads(done_marker.read_text(encoding="utf-8"))}

    log = _TeeLogger(out_dir / "train.log")
    t0 = time.time()
    try:
        log(f"[run] {name}: {nparams:,} params on {device}; git_commit={_git_commit()}; "
            f"start_unix={t0:.0f}")
        result = train(config, device=device, out_dir=out_dir, resume=resume, log_fn=log)
        best = result["best"]
        bench = result.get("benchmark") or {}
        summary = {
            "name": name,
            "params": nparams,
            "status": "complete",
            "elapsed_sec": round(time.time() - t0, 1),
            "git_commit": _git_commit(),
            "best_step": best.get("step"),
            "best_selected_by": best.get("metric"),
            "best_value": best.get("value"),
            "best_hard_n_pairs": best.get("hard_n_pairs"),
            "best_overall_val_logratio_mae": best.get("overall_val_logratio_mae"),
            "best_det_f1": best.get("det_f1"),
            "benchmark_out": bench.get("out_dir"),
            "benchmark_provisional": bench.get("provisional"),
            "benchmark_data_role": bench.get("data_role"),
        }
        done_marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"[run] {name}: COMPLETE in {summary['elapsed_sec']}s -- best step {summary['best_step']} "
            f"by {summary['best_selected_by']}={summary['best_value']}; "
            f"benchmark[{'PROVISIONAL-val' if bench.get('provisional') else 'TEST'}] -> {bench.get('out_dir')}")
        return summary
    finally:
        log.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the phase-5b overnight chain (small -> large).")
    p.add_argument("--device", default="auto", help="'auto' (CUDA/A100 if present), 'cuda', or 'cpu'")
    p.add_argument("--no-resume", action="store_true", help="ignore any train_state.pt and start fresh")
    p.add_argument("--force", action="store_true", help="rerun stages already marked complete")
    p.add_argument("--stop-on-failure", action="store_true",
                   help="abort the chain if a stage crashes (default: continue to the next stage)")
    p.add_argument("--dry-run", action="store_true", help="print param counts + setup checks, do not train")
    p.add_argument("--stages", nargs="*", default=None, help="override stage config paths")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    print(f"[overnight] device={device} (cuda_available={torch.cuda.is_available()})  "
          f"git_commit={_git_commit()}")
    if device.type != "cuda":
        print("[overnight] WARNING: not running on CUDA -- the 40k-step schedule is only "
              "practical on a GPU (A100). On CPU this will take many hours per run.")

    stages = [Path(s) for s in args.stages] if args.stages else DEFAULT_STAGES

    results = []
    for cfg_path in stages:
        try:
            res = run_stage(cfg_path, device=device, resume=not args.no_resume,
                            force=args.force, dry_run=args.dry_run)
            results.append(res)
            if res.get("status") not in ("complete", "skipped-complete", "dry-run") and args.stop_on_failure:
                print(f"[overnight] stopping: stage {cfg_path.name} did not complete.")
                break
        except Exception:
            print(f"[overnight] STAGE FAILED: {cfg_path}")
            traceback.print_exc()
            results.append({"name": str(cfg_path), "status": "failed"})
            if args.stop_on_failure:
                print("[overnight] --stop-on-failure set; aborting chain.")
                break
            print("[overnight] continuing to next stage (crash isolated to this run).")

    print("\n" + "=" * 78)
    print("[overnight] SUMMARY")
    for r in results:
        print(f"  {r.get('name'):<40s} {r.get('status'):<18s} "
              f"params={r.get('params', '?')} best_step={r.get('best_step', '-')}")
    print("=" * 78)
    n_ok = sum(1 for r in results if r.get("status") in ("complete", "skipped-complete", "dry-run"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
