# Benchmark grid requirements — SNR and density axes

> ## ✅ IMPLEMENTED 2026-07-14 — with one correction
>
> Both axes are now live in `configs/benchmark.yaml` (and mirrored in `benchmark_smoke.yaml`):
> `snr_targets: [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]`, `density_levels: [0.0006, 0.002, 0.006,
> 0.012, 0.015, 0.02, 0.025]` → **6 × 7 = 42 cells**. Generation verified: 42 cells, no cell
> clips either channel, constant-SNR spread 0.0e+00.
>
> Training support was widened to contain the grid: `configs/train.yaml`
> `scene.density.max` **0.012 → 0.030** (headroom above the 0.025 top cell).
> `tests/test_density_coverage.py` now pins train.yaml ↔ benchmark.yaml ↔ the generator
> constant together so they cannot silently drift apart again.
>
> **CORRECTION to §1's BLOCKER.** The claim that *"the retrain's intensity range must reach
> down to ~8 photons before the new SNR axis can be used as a headline"* was measured against
> the **LEGACY checkpoints'** [20, 7943] photon range. The measured-detector training config
> solves its window per image with `floor_a1_photons = 3.0`; the realised support is
> **[3.0, 18,388] photons**. Every new SNR level is already covered, with **31–68% of trained
> spots at or above each one** (`scripts/coverage_probe.py`). **The blocker is already lifted** —
> the low-SNR cells are NOT a cmeAnalysis-only diagnostic and need no intensity-range change.
>
> The generator still *flags* SNR 0.75 / 1.0 as out-of-range — correctly, because those cells
> are OOD **for the legacy checkpoints**. That flag is a statement about the legacy models, not
> about the grid.
>
> Still outstanding from this doc: the **≥6 px border margin** (§4) — an evaluator change, not
> done yet.

**Status:** ~~proposed~~ **IMPLEMENTED** change to `configs/benchmark.yaml`. Written 2026-07-14
from measurements on the clean (zero-registration-shift) benchmark, git `5e9ee11`.

**Why this exists:** the current SNR axis measures nothing. All six levels
`[2, 3, 5, 8, 10, 15]` sit on a saturated plateau, so 30 of the 43 conditions are
spending their variance separating methods that are already indistinguishable
there. The density axis stops right where the interesting physics *starts*. Both
were fixed by measurement, not by taste — the evidence is below.

---

## 1. The SNR axis is inert and must be moved DOWN

### Evidence

`cmeAnalysis`, given σ (1.4 / 1.68), density held at 0.002, border ≥6 px excluded:

| SNR | photons/spot | recall | precision | F1 |
|---|---|---|---|---|
| 0.5  |  8 | **0.011** | 0.741 | 0.022 |
| 0.75 | 13 | 0.102 | 0.931 | 0.184 |
| 1.0  | 18 | 0.335 | 0.968 | 0.498 |
| 1.25 | 23 | 0.645 | 0.982 | 0.779 |
| 1.5  | 29 | 0.869 | 0.989 | 0.925 |
| 2.0  | 43 | 0.950 | 0.993 | 0.971 |
| 3.0  | 78 | 0.953 | 0.995 | 0.974 |

And on the CURRENT grid (same tool, same settings), F1 by SNR:

```
SNR   2     3     5     8     10    15
F1  .919  .929  .932  .934  .934  .933      <- 1.5 points of spread across 7.5x
```

**The entire dynamic range is SNR 0.75 → 2.0** (13 → 43 photons). Below 0.75 the
detector is blind; above 2.0 it is flat forever. The current grid begins at the
top of the plateau.

**50% recall point: SNR ≈ 1.1, ≈ 20 photons/spot.** That is the number that
characterises a detector's noise limit on this benchmark.

Note the failure *mode*: precision holds ≥0.93 all the way down to SNR 0.75.
cmeAnalysis fails by going **blind**, not by hallucinating. So a learned
detector's win at low SNR must be a **recall** win — and there is a lot on the
table (at SNR 1.0, 2,750 of 3,930 real spots are missed).

### REQUIRED

```yaml
snr_targets: [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
```

Six levels, same count as today, repositioned onto the live range. `3.0` is
retained as an "everyone is fine here" anchor so the plateau is still visible.

Dropping `0.5` is deliberate: at 1.1% recall the cell yields almost no matched
spots, so it produces a metric with no statistical power and an undefined-ish
alpha fit. It is a floor demonstration, not a measurement. Add it back only if a
"nobody can do this" anchor is wanted.

### BLOCKER — the retrain must cover this range first

The generator flags SNR ≤ 1.0 as **`[OOD]`**: 8–18 photons falls **below the
legacy checkpoints' training range of [20, 7943] photons**.

On the re-centred axis, our model would be operating **outside its training
distribution exactly in the cells that finally discriminate**, while cmeAnalysis
— which has no training distribution — would not. That hands the *baseline* an
unearned win, and it is the exact mirror image of the registration-shift bug we
just removed.

**The retrain's intensity range must reach down to ~8 photons before the new SNR
axis can be used as a headline.** Until then the low-SNR cells are a
cmeAnalysis-only diagnostic, and must be labelled as such.

---

## 2. The density axis must go UP, to the point where PSFs merge

### Evidence

The density axis is really sweeping **mean nearest-neighbour distance against PSF
width**. PSF FWHM: ch1 = 3.30 px, ch2 = 3.96 px.

| density | spots / 256² | mean NN dist | NN ÷ FWHM(ch2) | recall (measured) |
|---|---|---|---|---|
| 0.0006 |   39 | 20.4 px | 5.16 | 0.983 |
| 0.002  |  131 | 11.2 px | 2.83 | 0.961 |
| 0.006  |  393 |  6.5 px | 1.63 | 0.894 |
| 0.012  |  786 |  4.6 px | 1.15 | 0.801 |
| 0.015  |  983 |  4.1 px | 1.03 | 0.761 |
| 0.02   | 1311 |  3.5 px | 0.89 | — (PSFs overlap) |
| 0.025  | 1638 |  3.2 px | 0.80 | — (PSFs overlap) |

The current ceiling (0.015) sits **exactly at NN = 1.0 × FWHM** — the *onset* of
overlap. Everything genuinely hard is above it, and recall is still falling
steeply there (−0.04 per +0.003 density), i.e. nowhere near a floor.

At density 0.025 the mean neighbour separation (3.2 px) is **below the ch2 PSF
FWHM (3.96 px)**: neighbouring spots are physically merged. A local-max detector
cannot separate them even in principle. That is the right place to stop.

### REQUIRED

```yaml
density_levels: [0.0006, 0.002, 0.006, 0.012, 0.015, 0.02, 0.025]
```

Seven levels. Extrapolating the measured trend, recall at 0.025 should land
around 0.6–0.7 — **still not saturated**, so the axis stays informative all the
way to the top. If a hard floor is wanted, add `0.03` (NN = 2.9 px, 0.73 × FWHM);
that is optional, not required.

**This is where the model's claim lives.** cmeAnalysis is at ceiling in sparse
fields (recall 0.983, precision 0.990 at density 0.0006) — there is nothing to
beat there, and any "win" claimed in that regime is noise. The headroom is
entirely in crowding: ~24 recall points at density ≥ 0.012, against a precision
that is already 0.99.

---

## 3. Grid size and cost

| | current | proposed |
|---|---|---|
| snr_density cells | 6 × 5 = 30 | 6 × 7 = **42** |
| images per cell | 50 | 50 |
| snr_density images | 1,500 | **2,100** |
| curvature sets | 13 (unchanged) | 13 (unchanged) |

Generation: ~6 min → ~8 min. A full cmeAnalysis sweep: ~50 min → ~70 min. Both
acceptable; neither is a reason to trim the grid.

---

## 4. What must NOT change (or the comparison stops meaning anything)

* **`registration_shift.max_px = 0`.** Non-negotiable. The default is 1.0 and it
  silently poisoned the ground truth once already (see `VENDORED_NOTES.md` and
  `tests/test_benchmark_registration.py`). Set it EXPLICITLY; omission
  re-inherits the 1.0 default.
* **PSF σ constant and published in `BENCH_MANIFEST.json`** (1.4 / 1.68) and
  handed to classical tools. Rationale: a real user bead-calibrates σ; making
  them guess it from crowded data measures their σ-estimator, not their detector.
  (Measured: cmeAnalysis's own σ-fit blows up to 3.478 px in the bright+dense
  corner and takes recall from 0.57 to 0.065. That is a calibration failure, not
  a detection failure, and it should not be the headline.)
* **Channel mapping: ch1 = LIPID (561), ch2 = PROTEIN (488)** — opposite the
  acquisition order.
* **Evaluator frozen**: Hungarian, gate `1.0 × max(σ1, σ2)` = 1.68 px from the
  manifest, unweighted OLS α fit. One blind evaluator for every method.

### NEW REQUIREMENT — border margin

**Exclude a ≥6 px border margin, applied identically to ground truth and
predictions, for EVERY method.**

cmeAnalysis's `fitGaussians2D` uses a 4σ window and structurally cannot fit a spot
within ~6 px of the edge. Measured: **88% of its missed spots lie within 6 px of
the border**, against 9.2% expected by chance. That is worth ~5 F1 points, handed
free to any padded CNN. Without the margin, our model wins on an edge convention.

This is an evaluator change, not a generator change.

---

## 5. Summary of the config diff

```yaml
benchmark:
  snr_targets:    [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]      # was [2, 3, 5, 8, 10, 15]
  density_levels: [0.0006, 0.002, 0.006, 0.012, 0.015, 0.02, 0.025]   # was [..., 0.015]
  images_per_cell: 50                                    # unchanged
  # curvature family unchanged
```

The old ADC-clip cap on `snr_targets` (protein clips near SNR ≈ 16.8 at gain 40)
is now moot — the new grid tops out at 3.0. Generation still asserts no cell
clips; that assertion should stay.

---

## 6. Provenance

Every number above is measured, not assumed:

* SNR curve: `cme_analysis/snr_probe/` — 7 cells × 30 images at density 0.002,
  cmeAnalysis given σ, border-corrected.
* Density curve: the main 43-condition sweep, `cme_analysis/work/eval/`.
* Border effect: nearest-neighbour analysis of unmatched GT in
  `snr=5_density=0.0006`.
* σ-estimator failure: `sigma_fit_m` column of the raw detection CSVs.
