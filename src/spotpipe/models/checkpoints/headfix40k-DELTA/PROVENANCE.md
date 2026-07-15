# PROVENANCE — headfix40k-DELTA (the chosen headline model, 2026-07-15)

**This is the reproducible v3 headline model.** It supersedes `hrnet_large_measured`, which
was trained at density max 0.012 and is out-of-distribution in the v3 benchmark's crowded
cells (up to 0.025) — exactly where the win over the baselines lives.

## What it is
* **Architecture:** HRNet-large (base_channels 32, ~1.53M params) + the **delta intensity
  head** (`model.head_parameterisation: delta`): predicts `(logI1, Δ)` with `Δ = logI2 − logI1`
  and derives `logI2 = logI1 + Δ`. State-dict-identical to the independent head; the schema
  (`logI1`, `logI2`, `log_ratio`) is unchanged. See `docs/intensity_head_fix_proposal.md`.
* **Why the delta head:** the independent head does conditional-mean shrinkage that hits the
  wider-PSF protein channel harder under crowding (`docs/shrinkage_probe_findings.md`); the
  unequal shrinkage (`s₁ − s₂`) lands entirely on the ratio and manufactures curvature.
  Predicting Δ directly removes that residual. See `docs/head_fix_results.md` for the head-to-head.

## Reproducibility
| field | value |
|---|---|
| training git commit | `bd9b2a2694702df3197cfffc4e29f68dad4102c9` |
| vendored simulator SHA | `7b9a0b85ee527afeb73d9e68f9bdb30960775083` |
| training config | `configs/train_40k_DELTA.yaml` (gen by `scripts/make_40k_headfix_configs.py`) |
| seed | 0 |
| steps | 40,000 |
| best checkpoint | **step 23,000**, selected by hard-corner val_logratio_mae = 0.4067 |
| overall val_logratio_mae @ best | 0.2883 |
| training distribution | measured detector, gains randomised per image, per-image saturation-safe intensity window, **density max 0.030** (v3-covering), constant ~2-photon background |

**Selection never touches benchmark/test outputs** (hard-corner val only) — the benchmark stays a
blind test set.

## Headline numbers (α probe, 13-point sweep, `results/headfix40k/`)
* **α-MAE 0.055**, **α=0 null +0.024 ± 0.002** — both far past the baselines (α 1.0–2.8, old oracle 1.9).
* Crowded (density 0.012) ratio bias ≈ 0 with ~2× tighter spread than the independent head.

## ⚠️ Still pending before this is the FINAL headline
1. **Full stratified v3 benchmark eval was NOT completed** — the overnight infer hit a device bug
   (fixed in `infer.py`), and the re-run scored only the trivial GT-passthrough oracle. Run
   `spotpipe infer --checkpoint headfix40k-DELTA` + `spotpipe evaluate` (CPU is fine, ~20 min) to
   get per-cell detection + intensity metrics, and confirm detection holds at the low-SNR cells
   (snr 0.75–1.25) the α probe does not cover.
2. **peak_threshold is still 0.3** (too permissive; 0.547 precision vs 0.995 at 0.7). Retune
   OFF-benchmark before the headline, and apply the same protocol to every baseline.
3. **Real baselines** (cmeAnalysis / SpotMAX / Spotiflow) not yet run on the v3 grid — each needs
   its own env. The oracle in `results/headfix40k/eval40k` is the trivial GT-passthrough, not the
   informative GT-center+shared-extraction oracle.

The **`headfix40k-INDEP`** checkpoint is carried alongside as the control (matched 40k steps,
independent head; α-MAE 0.061, null +0.009). It is the safe fallback if the stratified eval
surfaces a low-SNR detection regression in DELTA.
