# Shrinkage probe — the defect is LOCALISED to protein-channel-under-crowding

**Date:** 2026-07-14 · **Hardware:** CPU, dev box (no GPU) · **~2 min**
**Instrument:** `scripts/shrinkage_probe.py` · **Raw:** `results/shrinkage_probe.csv`
**Checkpoint probed:** `hrnet_large_measured`

---

## Honest header: a prediction failed, and the failure is the finding

`docs/intensity_head_fix_proposal.md` §2 made a falsifiable prediction: the shrinkage slope
`s < 1` **everywhere**. **That failed.** Three of four cells came back at `s ≈ 1` (unbiased). Only
ONE cell shrinks. So the diagnosis is NOT "general conditional-mean shrinkage toward a global
intensity prior" — that specific claim is dead, and distance-from-the-prior is NOT the driver
(sparse-bright spots are just as far from the ~29-photon median and they do NOT shrink).

What the failure buys: a **sharper, more localised** diagnosis that two *pre-registered*
predictions nailed.

## The measured slopes

Regress predicted logI on TRUE logI across a wide intensity sweep (10 → 1,280 photons):

| density | ch | slope `s` | reading |
|---|---|---|---|
| 0.0006 | 1 (lipid) | **1.059** | sparse — unbiased |
| 0.0006 | 2 (protein) | **1.009** | sparse — unbiased |
| 0.012 | 1 (lipid) | **0.983** | dense — unbiased |
| 0.012 | **2 (protein)** | **0.700** | dense — **the ONLY broken cell** |

The defect is **entirely protein-channel-under-crowding.** Lipid (ch1) is unbiased at every density;
protein (ch2) is unbiased when sparse. Only dense + protein breaks.

## Why this is a refinement, not a rescue

The proposal doc pre-registered four predictions. Scoring them honestly:

1. `s < 1` everywhere — **FAILED** (only dense ch2).
2. `s(dense) < s(sparse)`, both channels — **PASSED** (ch1: 0.983<1.059; ch2: 0.700<1.009).
3. `s(ch2) < s(ch1)`, both densities — **PASSED** (sparse 1.009<1.059; dense **0.700<0.983**).
4. fixed point near the ~29-photon prior — dense ch2 fixed point = **48.5 ph** (same order); the
   other cells don't shrink so their "fixed point" is a meaningless extrapolation.

Predictions **2 and 3 were stated before seeing the data, and both passed.** Their conjunction *is*
"dense-ch2 is the worst cell," which is what happened. So the localisation is a product of two
correct directional forecasts, not a post-hoc story fitted to one number.

**The refined mechanism:** the NLL's optimum for the mean is the conditional mean, which regresses
toward the prior **only where the image evidence cannot identify the spot.** Evidence is lost
exactly when a wide PSF meets overlap: σ₂ = 1.68 px spots cannot be deblended at density 0.012, so
ch2 shrinks; σ₁ = 1.4 px still resolves them, so ch1 does not; sparse fields resolve both. "Ambiguous
evidence → shrink" was the right half of the original claim; "toward a global brightness prior" was
the wrong half.

## The ratio driver (unchanged, and quantitatively confirmed)

```
log-ratio bias  ≈  (s₁ − s₂) · (logI − fixed_point)
```

At dense: **s₁ − s₂ = 0.983 − 0.700 = +0.283.** At the bright-dense corner (A₁ ≈ 1,365 ph,
logI ≈ 7.22, fp ≈ 48.5 ph → log 3.88):

```
ratio bias ≈ 0.283 × (7.22 − 3.88) ≈ +0.28 × 3.34 ≈ ... (sign: pred logI2 falls, so ratio bias ≈ −0.9)
```

which reproduces the **−0.9 to −1.1** measured directly by the rarity probe. The ratio bias is
carried by the **difference** in channel shrinkage, exactly as claimed — and this is why arm B, which
shrank both per-channel biases but changed `s₁` and `s₂` unequally, made the ratio *worse*.

## The logvar head is fine

0% of spots at either clamp bound (`[−10, 6]`) in any cell — the `logvar` saturation hypothesis is
**dead**. And the head correctly widens its variance where it shrinks: dense-ch2 logvar mean −1.51 vs
sparse-ch2 −2.87. It is doing textbook conditional-mean behaviour: return the shrunk mean *and* flag
the larger uncertainty. Calibration works; the mean is just optimally biased under overlap.

## What this does to the fix

It makes the **`(logI1, Δ)` reparameterisation** (`intensity_head_fix_proposal.md` §3) the clearly
correct move, and better-motivated than when it was proposed:

* **ch1 is a reliable anchor at every density** (slope 0.98–1.06). The only broken quantity is ch2's
  *absolute* intensity under overlap.
* `Δ = logI2 − logI1` **shares the overlap contamination**: spots are co-located in both channels, so
  a contaminating neighbour bleeds into the same pixel in both, and much of it cancels in the
  difference. `Δ` stays identifiable exactly where absolute ch2 does not. Predicting it directly
  removes the `s₁ − s₂` term that carries the entire ratio bias.
* The cheaper alternative (a channel-error-correlation penalty, proposal §3) *equalises* `s₁` and
  `s₂` rather than removing the shrinkage; worth keeping as the B-arm of the fix bake-off.

**Sharper test the next probe should add:** the shrinkage is overlap-driven, so it should scale with
the ch2 PSF width. If a future run sweeps σ₂, `s₂` should fall as σ₂ rises at fixed density — a
clean, independent check that identifiability (not something else about the protein channel) is the
cause.

## Still true, still the gate

`s` varies with density and channel, and √A₁ is the size ruler — so a density-varying `s` is a
size-dependent bias, which is the mechanism that tilts α (PROJECT_STATE §1). Today α MAE = 0.036 only
because family 2 runs sparse (`s ≈ 1`). **Any fix must be gated on the α=0 null control and the
known-α sets**, not on per-channel MAE.
