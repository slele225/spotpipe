# Experiment: 2026-06-23_hrnet-validation

- **Git commit (pinned):** `ce697f485b7191529a1ea4fae2d2f36aebe5c09a-dirty`
- **Config:** [`config.yaml`](config.yaml)
- **Outputs:** [`outputs/`](outputs/)

> **Note on the pinned commit.** The stamper pinned `…ce697f4-dirty` because the
> working tree was dirty when this experiment was stamped (build stages 2–4 are
> implemented but not yet committed). The `-dirty` suffix is the stamper's
> honest marker that the pin is **not** a clean, reproducible commit. This is
> acceptable for a throwaway plumbing run, but **before any real/overnight run
> the shared code under `src/spotpipe/` should be committed** so the experiment
> pins a clean hash.

## Motivation

Very-short **plumbing validation** of the real training + auto-benchmark
pipeline; **not a science run**. The goal is to confirm the whole real-training
pipeline fires end-to-end before any GPU-hours are committed:

- on-the-fly synthetic data generation (seeded/reproducible),
- the **three independent schedules** over one timeline,
- evaluation against the **fixed** hard-corner eval set,
- checkpointing + **best-checkpoint selection**,
- the auto-triggered **benchmark** on the best checkpoint.

At 300 steps the numbers are **meaningless** and must not be read as results.
The only claim is that every stage runs and writes its artifacts.

## What this run exercises (the three schedules + selection)

All three are explicit, configurable **step-count** knobs in `config.yaml`:

1. **LR schedule** — linear warmup over `lr_warmup_steps: 15` (~5% of 300), then
   cosine decay to ~0 over the full run.
2. **Variance warmup** — a fixed **step count** `variance_warmup_steps: 40`: fit
   the intensity *mean* under fixed unit variance for steps 1–40, then enable the
   heteroscedastic `logvar` NLL from step 41.
3. **Scene curriculum** — ramp scene difficulty (density / overlap / low-SNR /
   background) easy→hard over `curriculum_ramp_steps: 150`, then **HOLD** at full
   difficulty. Scene difficulty only — detector constants are never ramped; β and
   PSF/registration stay at full range throughout.

**Best-checkpoint selection.** Validation runs on the **fixed** eval set every
`eval_every: 75` steps; checkpoints are written every `checkpoint_every: 75`.
The best checkpoint is selected by **`val_logratio_mae`** — the mean absolute
error of `log_ratio_pred − log_ratio_true` on location-matched pairs over the
fixed eval set, computed by the same matcher (`benchmark.matching`) and the same
schema `log_ratio` the benchmark uses. If no eval has produced a finite
log-ratio MAE yet (e.g. the briefly-trained net detects nothing), selection
falls back to **`val_total_loss`** for that comparison. Selection never uses any
benchmark figure or post-hoc plot — that would be circular.

## One fixed eval set, both uses

The fixed validation/eval set is built **once** (`val.seed: 12345`, the
hard-corner specification: β=0, ±β extremes, dim×high-overlap, bright×sparse) and
reused for **both** periodic validation **and** the end-of-run auto-benchmark.
Its identity is recorded in [`outputs/eval_manifest.json`](outputs/eval_manifest.json)
(seed, image-ids, per-image β). The benchmark is **not** given a separately
seeded set — so the checkpoint chosen "best" is chosen on exactly the data the
benchmark reports on.

## Config diff from baseline

No baseline. Differs from `configs/train_smoke.yaml` by being the **real**
(non-smoke) entry point at full input size: 256×256 / 2 channels, the same
~450k-param HRNet, full FV3000 detector + full scene ranges
(`configs/simulator.yaml`), and the three real schedules wired as step-count
knobs with best-checkpoint selection and `auto_benchmark: true`.

## Results

Run with:

```
uv run python -m spotpipe.training.train --config experiments/2026-06-23_hrnet-validation/config.yaml
```

Artifacts land in `outputs/`:

- `config.yaml`, `manifest.json` (git commit pinned, resolved schedule steps,
  best-checkpoint record), `eval_manifest.json` (the one fixed eval set).
- `metrics.jsonl`, `loss_curve.csv`, `val_curve.csv` (training + validation curves).
- `checkpoint_step*.pt`, `best_checkpoint.pt`, `checkpoint.pt` (final).
- `benchmark/` — metrics table + figures (recovered-β both variants, ratio
  bias/variance vs SNR and density, uncertainty calibration) on the best
  checkpoint over the fixed eval set.

_Plumbing-only: the metrics here are not to be interpreted as scientific results._

## Decision

Pipeline validation only. If all stages fire and write their artifacts, the
real-training + auto-benchmark plumbing is sound and the (committed-clean) real
runs can proceed. Otherwise, fix whatever errored or silently no-op'd first.
