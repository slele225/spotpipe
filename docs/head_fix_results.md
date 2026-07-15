# Head-fix bake-off — RESULT: ship the delta head (2026-07-15)

The intensity-head investigation is closed. **The `(logI1, Δ)` reparameterisation wins and is
the v3 headline model (`headfix40k-DELTA`).** This doc records the numbers and the reasoning so
the decision is auditable. Raw data: `results/headfix40k/`.

## The chain that got here (each probe killed a hypothesis cheaply)
1. `coverage_probe_findings.md` — the "bright+dense is structurally unreachable" hypothesis is
   FALSE (density and brightness are drawn independently; corr +0.02).
2. `rarity_probe_findings.md` — flattening the sampler does NOT fix it; and the tell surfaced:
   arm B improved both channels yet made the RATIO worse, because per-channel accuracy ≠ ratio
   accuracy.
3. `shrinkage_probe_findings.md` — the mechanism: conditional-mean shrinkage localised to the
   wider-PSF protein channel under crowding (dense ch2 slope 0.70 vs ~1.0 elsewhere); the unequal
   shrinkage (`s₁ − s₂`) lands entirely on the ratio.
4. `intensity_head_fix_proposal.md` — the fix: predict Δ directly so the ratio is the model's own
   estimand and the `s₁ − s₂` residual cannot exist.

## The 40k bake-off (two arms, identical but for `model.head_parameterisation`)

Both arms: HRNet-large, 40k steps, v3-covering distribution (density max 0.030), git `bd9b2a2`.

| metric | INDEP (independent) | DELTA (logI1, Δ) | winner |
|---|---|---|---|
| **α-MAE** (13-pt sweep) | 0.0607 | **0.0550** | DELTA |
| **α=0 null control** | **+0.0093** ± 0.0024 | +0.0243 ± 0.0022 | INDEP (both clean) |
| hard-corner val (dim×overlap — the claim regime) | 0.423 | **0.407** | DELTA |
| crowded (d=0.012) ratio bias, snr 2/3 | +0.225 / +0.164 | **−0.009 / −0.044** | DELTA |
| crowded (d=0.012) ratio RMSE, snr 2/3 | 0.251 / 0.190 | **0.098 / 0.090** | DELTA (2–2.6× tighter) |
| best step | 21,000 | 23,000 | — |

For scale: **both crush the baselines** (α-MAE 1.0–2.8, old-repo oracle 1.9). The head choice is
about which of two already-excellent models to ship.

## Why DELTA, despite the marginally-worse null
DELTA wins α-MAE, the hard corner (the dim×high-overlap regime where the low-bias claim lives),
and the crowded-intensity regime (ratio bias ≈ 0 with ~2× tighter spread) — which is exactly where
the headroom over the baselines is (PROJECT_STATE §8: "the headroom is entirely in crowding").
Tighter ratio spread in crowded data means tighter α error bars from the cells that matter.

Its one soft spot is the α=0 null: +0.024 vs INDEP's +0.009. Both PASS the manufactured-curvature
control (0.024 is ~1% of the α range); the crowding win outweighs a 0.015 gap on an already-clean
null.

## What the 8k → 40k comparison taught us (don't repeat these misreads)
* At **8k**, DELTA's null (−0.02) crushed INDEP's (+0.22) — that INDEP null was UNDERTRAINING and
  cleared to +0.009 by 40k. An 8k null is not a verdict.
* At **8k**, DELTA looked WORSE in dim-dense — also undertraining (logI1 not settled); at 40k it
  reversed to clearly better. Short-schedule intensity metrics are unreliable.
* **Per-channel MAE is not the objective; the log-ratio is.** A change can improve both channels and
  worsen the ratio (rarity probe). Always judge on `log_ratio_bias` / α, never logI1/logI2 MAE.

## Not done — the path to the FINAL headline
1. **Full stratified v3 benchmark eval of DELTA** (and INDEP as control). The overnight run's infer
   hit a device bug (now fixed in `infer.py`); the re-run scored only the trivial GT-passthrough
   oracle. Inference is CPU-fine (~20 min) — run it locally:
   `spotpipe infer --checkpoint headfix40k-DELTA --benchmark data/benchmark --device cpu` then
   `spotpipe evaluate`. Confirm detection holds at the low-SNR cells (snr 0.75–1.25) the α probe
   doesn't cover. INDEP is the fallback if a low-SNR detection regression shows up.
2. **Retune `peak_threshold`** off-benchmark (0.3 → ~0.7; precision 0.547 → 0.995 at no recall cost),
   same protocol extended to every baseline.
3. **Real baselines** on the v3 grid: cmeAnalysis → SpotMAX → Spotiflow, each in its own env, plus
   the informative GT-center + shared-extraction oracle (the `--oracle` flag currently emits only
   the trivial GT passthrough).
