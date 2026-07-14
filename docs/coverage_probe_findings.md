# Coverage probe findings — the intensity-head retrain hypothesis is REFUTED

**Date:** 2026-07-14
**Probe:** `scripts/coverage_probe.py` (3,000 sampled training images, curriculum `t = 1.0`)
**Trigger:** `docs/handoff_retrain_intensity_head.md` §2 — *"Cheap check first (~20 min, do NOT
skip): sample the training config's simulator and plot the realised joint (A₁, area-density)
coverage against the benchmark grid. … If it does NOT [fall outside the training support], stop —
the cause is something else and this handoff's §3 is wrong."*

It does not fall outside. **§3 is wrong. Stop.**

---

## 1. The hypothesis, and why it is false

> *"The training distribution solves per-image intensity to keep both channels unclipped. That
> **couples brightness to density**: a dense image is forced dim, because many bright spots would
> clip ch2. So 'bright AND dense' is not merely rare in training — it is **structurally
> unreachable**."*

**Measured:**

```
corr( log10 density , log10 median-A1 ) = +0.0207
```

and the A₁ profile is flat across every density quintile — the exact opposite of a distribution
where dense images are forced dim:

| density bin (spots/px) | n images | A₁ p50 | A₁ p95 | A₁ max |
|---|---|---|---|---|
| 0.00060 – 0.00179 | 600 | 27.6 | 992.0 | 13,118 |
| 0.00179 – 0.00455 | 600 | 28.5 | 1,015.6 | 15,543 |
| 0.00455 – 0.00797 | 600 | 29.5 | 1,047.5 | 18,388 |
| 0.00797 – 0.01360 | 600 | 30.2 | 1,050.9 | 17,816 |
| 0.01360 – 0.02399 | 600 | 29.3 | 1,036.5 | 13,445 |

**Why it was never going to be true — read the code.** `intensity_window.solve_a1_ceiling()` takes
`gain1, gain2, sigma1, sigma2, sim_intercept, sim_log_slope, scatter_std, background, knee1, knee2`.
It **does not take density**. It cannot: it solves the ceiling for a *single* spot's clean gained
peak. Density is drawn separately in `sample_scene_params`. The two axes are independent **by
construction**, and no amount of retraining will "break" a coupling that does not exist.

The ceiling *is* real and *is* binding — it is just bound by **gain**, not density. A per-image ch2
gain drawn near 150 forces a dim ceiling; a gain near 20 permits a bright one. That is a
brightness↔**gain** coupling, and it is intentional (CHANGE 1).

## 2. Therefore

* **None of options (a) / (b) / (c) in `handoff_retrain_intensity_head.md` §3 is warranted.** Do not
  let images clip; do not lower the simulated ch2 gain; do not declare bright+dense out of scope on
  coverage grounds. All three are answers to a question that turned out to be false.
* **The intensity-head defect is real but its cause is unknown.** The evidence in §1 of that handoff
  still stands (model log-ratio bias −1.10 in bright+dense, `logI2_bias` −1.66/−2.09, while a
  perfect detector reading the same pixels gets −0.032). Detector effects, crowding, soft-knee
  compression and coverage are now **all** ruled out. Whatever is left is in the model or the loss.
* Rarity is *not* the same as unreachability, and is the remaining suspect worth testing: only
  ~5% of trained spots exceed ~1,000 photons (`full_dim_bias = 1.6` deliberately over-samples the
  dim tail). A dim-biased sampler plus a Gaussian-NLL intensity head is a plausible route to a
  bright-end regression-to-the-mean — but that is a **hypothesis, not a finding**, and it should be
  tested (e.g. re-weight or flatten `full_dim_bias` and watch the bright-end bias) before it justifies
  another 40k steps.

## 3. The finding that actually changes the plan

**The benchmark-v3 grid removes the bright cells where the defect lives.**

| | v2 grid (where the defect was diagnosed) | v3 grid (the new benchmark) |
|---|---|---|
| SNR targets | 2, 3, 5, 8, 10, 15 | 0.75, 1.0, 1.25, 1.5, 2.0, 3.0 |
| A₁ per spot | 43 → **1,365 photons** | 12.7 → **77.6 photons** |

The intensity collapse was measured at `snr=10` (625 ph) and `snr=15` (1,365 ph). **The v3 grid's
brightest cell is 77.6 photons** — 18× dimmer than where the defect was characterised, and sitting
at the *median* of the training A₁ distribution (29.5 ph) rather than out in its 5% tail.

So on the new grid, the family-1 headline numbers are **not exposed to this defect at all**. That
does not make the defect harmless — it means the decision about whether to chase it is now a
**scope** decision, not a blocker:

* **Family 2 (curvature)** keeps a wide A₁ spread, but runs at **low density**, and the defect is
  specifically bright **AND** dense. α recovery is already good (MAE 0.036; null +0.021 ± 0.002).
* **Real data** is the live risk. If real liposome images are brighter than SNR 3, the model will
  under-read the protein channel there, and no benchmark cell will warn us.

## 4. What the probe DID find that needs fixing

One gap, and it is the one the grid change created:

```
density level 0.025  >  trained density max 0.024   -> OUT OF DISTRIBUTION
```

Fixed by raising `configs/train.yaml` `scene.density.max` → **0.030** (headroom, so the top cell sits
inside the support rather than on its boundary). After that: *"Every benchmark cell falls inside the
training support."*

## 5. Stale claim corrected

`docs/benchmark_grid_requirements.md` §1 carries a **BLOCKER**: *"the retrain's intensity range must
reach down to ~8 photons before the new SNR axis can be used as a headline"*, because SNR ≤ 1.0
(8–18 ph) falls below the training range of **[20, 7943]** photons.

**That range is the LEGACY checkpoints'.** The measured-detector training config solves the window
per image with `floor_a1_photons = 3.0`, and the probe measures a realised support of
**[3.0, 18,388] photons**. Every new SNR level is covered, with 31–68% of trained spots at or above
each one:

| SNR | A₁ (photons) | in support? | % of trained spots ≥ this |
|---|---|---|---|
| 0.75 | 12.7 | ✅ | 68.0% |
| 1.0 | 17.7 | ✅ | 61.1% |
| 1.25 | 23.3 | ✅ | 55.2% |
| 1.5 | 29.4 | ✅ | 50.1% |
| 2.0 | 43.1 | ✅ | 42.0% |
| 3.0 | 77.6 | ✅ | 31.3% |

**The low-SNR blocker is already lifted.** No intensity-range change is needed for the SNR move.

---

## Reproduce

```bash
python scripts/coverage_probe.py --n-images 3000
```

Exit code 0 == every benchmark cell is inside the training support. Runs in seconds; renders no
pixels and runs no model (`scripts/_torch_stub.py` lets it import the real sampling code path on a
box with no torch, rather than re-implementing the distribution and getting a plot of a
distribution the trainer does not sample).
