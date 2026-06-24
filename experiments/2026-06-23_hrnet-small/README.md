# Experiment: 2026-06-23_hrnet-small

- **Git commit (pinned):** `ce697f485b7191529a1ea4fae2d2f36aebe5c09a-dirty` (update to the 5b commit before launch)
- **Config:** [`config.yaml`](config.yaml)
- **Outputs:** [`outputs/`](outputs/)
- **Baseline:** `2026-06-23_hrnet-validation` (the plumbing run)
- **Paired with:** [`2026-06-23_hrnet-large`](../2026-06-23_hrnet-large) (capacity comparison)

## Motivation

The first **serious** training run of the two-channel spot model: baseline-capacity
HRNet (~449k params), full 40k-step schedule, real best-checkpoint selection and an
auto-benchmark of the chosen checkpoint. This is the reference point the wider
`hrnet-large` run is compared against. It is **not** the final paper result (that
waits on the frozen test set + all baselines) -- it is the first real read on whether
the approach works.

## Config diff from baseline

Versus the `hrnet-validation` plumbing run:

- **Schedule:** the FULL long-run schedule, not the 300-step plumbing one --
  `train_steps: 40000`, `lr_warmup_steps: 1500`, `variance_warmup_steps: 4000`,
  `curriculum_ramp_steps: 20000`, `eval_every`/`checkpoint_every: 1000`,
  cosine LR. Ordering holds: LR warmup < variance warmup < curriculum-full (50%),
  then HOLD at full difficulty for the back half while cosine LR is still meaningful.
- **`batch_size: 16`** (was 4).
- **Three data roles** (was one set for both val + benchmark): training is on-the-fly;
  a FIXED shared validation set (`data/fixed_eval/val`, separate seed) drives selection;
  the auto-benchmark reports on the frozen test set (`data/fixed_eval/test`) when it
  exists, else a clearly labelled PROVISIONAL benchmark on the val set.
- **Hard-corner selection:** best checkpoint chosen primarily on the **hard corner**
  (`SNR in [0,2) x density top bin`) `val_logratio_mae`, with `hard_corner_min_pairs: 50`
  fallback to overall MAE, then `val_total_loss`.

Model is unchanged from the validated baseline width (`base_channels: 16`).

## Results

_Filled in after the run. Key artifacts under `outputs/`:_

- `train.log` -- full per-run log (also stdout).
- `loss_curve.csv`, `val_curve.csv` -- training/validation curves (val curve carries
  both overall and hard-corner `val_logratio_mae` + matched-pair counts + detection F1).
- `manifest.json` -- pinned git commit, full schedule, detector, and the selected
  best checkpoint (step, selection tier, hard-corner pair count).
- `best_checkpoint.pt` -- the selected checkpoint.
- `benchmark_provisional_val/` (or `benchmark_test/` if the frozen test set exists) --
  binned metrics table (6 SNR x 4 density), recovered-beta (matched-only + end-to-end),
  ratio bias/variance vs SNR and vs density, uncertainty calibration, and a
  `data_role.json` stating whether the numbers are provisional (val) or final (test).

## Decision

_What we concluded and what happens next (keep / discard / follow-up)._
