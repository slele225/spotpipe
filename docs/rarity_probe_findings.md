# Rarity probe — REFUTED. The intensity-head defect is not a data problem.

**Date:** 2026-07-14 · **Hardware:** rented A100 80GB (destroyed after the run)
**Raw data:** `results/rarity_probe/bright_dense_probe_{CURRENT,ARM_A,ARM_B}.csv` (gitignored;
the numbers that matter are reproduced below)
**Instrument:** `scripts/bright_dense_probe.py` · **Runner:** `scripts/run_rarity_probe.sh`

---

## The question

After `docs/coverage_probe_findings.md` killed the COVERAGE explanation for the intensity head's
bright+dense collapse, one suspect remained: **rarity**. `full_dim_bias = 1.6` over-samples the dim
tail, so only ~5% of trained spots exceed ~1,000 photons, and the joint bright+dense corner is
~1% of training spots. Maybe a Gaussian-NLL head simply regresses toward its mean where it has
little data.

Two 8k-step arms, **identical except one knob** (`make_probe_configs.py` asserts this):

* **Arm A** — `full_dim_bias = 1.6` (the shipped sampler). The control.
* **Arm B** — `full_dim_bias = 1.0` (log-uniform in log A₁). Populates the bright end.

## Gate: the probe reproduces the defect

Before spending anything, the probe was run on `hrnet_large_measured` and reproduced the known
defect to three decimals — so its numbers mean something:

| cell | expected (RESULTS doc) | probe measured |
|---|---|---|
| snr=10, density=0.012 | ≈ −0.92 | **−0.918** |
| snr=15, density=0.012 | ≈ −1.10 | **−1.098** |
| logI2_bias @ snr15 dense | ≈ −1.66 | **−1.676** |

## Result: REFUTED

| | bright+dense (SNR≥10, d=0.012) | dim+dense (SNR≤3, d=0.012) | gap |
|---|---|---|---|
| **Arm A** (dim_bias 1.6) | **−0.524** | **+0.253** | −0.777 |
| **Arm B** (dim_bias 1.0) | **−0.832** | **−0.327** | −0.504 |

The gap shrank (0.78 → 0.50) — but **for the wrong reason**. The bright corner did not improve; it
got **worse** (−0.524 → −0.832). The gap narrowed only because the **dim end collapsed** (+0.253 →
−0.327).

**That is not a fix. It is the summary statistic being gamed by degrading the baseline** — and the
dim tail is where the project's entire low-bias claim lives. Flattening the sampler is a downgrade.

⚠️ **Compare arms to each other, never to CURRENT.** CURRENT is a 40k model; both arms are 8k.
Arm B looks bad everywhere (even sparse cells sit at ~−0.5 where CURRENT is ~+0.03) — that is
undertraining, not the sampler.

## The finding that actually matters: per-channel accuracy ≠ ratio accuracy

At snr=15, density=0.012:

| arm | logI1_bias | logI2_bias | **log-ratio bias** |
|---|---|---|---|
| A (1.6) | −1.184 | −1.854 | **−0.669** |
| B (1.0) | **−0.382** | **−1.282** | **−0.900** |

**Arm B improved BOTH channels individually and made the RATIO WORSE.**

In arm A both channels are under-read by similar amounts, so the errors partially **cancel** in the
difference. Flattening the sampler fixed ch1 more than ch2, **decorrelated** the errors, and the
ratio bias grew.

α depends **only on the ratio**. Therefore:

> **Per-channel intensity accuracy is NOT the objective, and any "fix" evaluated on logI1/logI2 MAE
> alone will mislead you in exactly this direction.** What the model needs is *correlated* channel
> errors, not small ones. Watch `log_ratio_bias`; treat `logI1_mae` / `logI2_mae` as diagnostics only.

This also complicates PROJECT_STATE §2's "the ratio comes for free" reasoning (*"if logI1 and logI2
are each unbiased, log(A2/A1) = logI2 − logI1 at inference"*). True at the optimum — but off the
optimum, the ratio's bias depends on how the two channels' errors **co-vary**, and a change that
improves both marginals can still make the ratio worse. Which is what happened here.

## Secondary clue: the defect appears to GROW with training

40k CURRENT bright+dense ≈ **−1.01**, but 8k arm A ≈ **−0.52**. If real, the head is *progressively*
regressing toward the conditional mean in a corner it rarely sees — which is a
**loss/optimisation** signature, not a data-coverage one.

⚠️ **Not airtight:** the arms trained at the NEW density max (0.030) and CURRENT did not, so the
distributions differ. Worth a clean test (checkpoint the same run at 8k and 40k) before relying on it.

## Where that leaves the cause

Ruled out: **coverage** (probe, corr = +0.02), **rarity** (this doc), **crowding** (GT-position bias
is worse for isolated spots), **detector effects**, **soft-knee compression** (tested + refuted —
do not re-litigate), **NMS**, **dark counts**.

Remaining suspects — all in the **loss/head**, not the data:

1. **The Gaussian-NLL intensity term.** An over-confident (or over-cowardly) `logvar` in the crowded
   regime lets the predicted mean drift with little penalty: `½[exp(−s)·(logÎ − logI)² + s]` pays
   almost nothing for a biased mean if `s` inflates.
2. **The `logvar` clamp `[−10, 6]`.** Check whether it is *saturating* in the bright+dense corner —
   if it is pinned at a bound there, the NLL's self-calibration is switched off exactly where the
   defect lives.
3. **Whatever couples the two channels' errors** — the ratio rides on that coupling, and nothing in
   the loss currently encourages it (per §3's NO-slope-loss rule, correctly; but a *correlation* term
   is not a slope term and would not leak α into training).

## Do NOT

* Do not retrain with `full_dim_bias = 1.0`. It trades the dim tail for a bright corner it does not
  even fix.
* Do not evaluate the next fix on per-channel MAE. Evaluate on `log_ratio_bias`, stratified by
  density.
* Do not re-litigate coverage, crowding, or soft-knee compression.

## Cost

~1 GPU-hour. It prevented a 40k-step retrain (plus a checkpoint, plus a full benchmark + evaluation
cycle) built on a hypothesis that turned out to be false — and it surfaced the error-correlation
finding above, which changes what "fixing the intensity head" even means.
