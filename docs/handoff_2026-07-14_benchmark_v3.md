# HANDOFF — 2026-07-14: benchmark v3, a refuted retrain hypothesis, and a live probe

*Written for OTHER CHATS / future context. If you are picking up this project today, read this
first — it supersedes parts of `handoff_retrain_intensity_head.md` and
`benchmark_grid_requirements.md`, and both of those now carry amendment banners pointing here.*

**Repo:** `C:\Users\shivl\Music\summer_research_26`, branch **`main`** (the OLD read-only repo is
branch `master` — see the GOTCHAS section, this bit us today).
**All changes below are committed and pushed.** Commit `e69a248` + a `.gitignore` fix.

---

## 1. TL;DR — what changed

| | before (v2) | after (v3) |
|---|---|---|
| `snr_targets` | `[2, 3, 5, 8, 10, 15]` | **`[0.75, 1.0, 1.25, 1.5, 2.0, 3.0]`** |
| per-spot A₁ | 43 → 1,365 photons | **12.7 → 77.6 photons** |
| `density_levels` | `[0.0006, 0.002, 0.006, 0.012, 0.015]` | **`[…, 0.02, 0.025]`** (7 levels) |
| family-1 grid | 6 × 5 = 30 cells | **6 × 7 = 42 cells** (2,100 images) |
| training `scene.density.max` | 0.012 | **0.030** |
| `max_spots` (decode) | 2000 | **3000** |

Curvature family (family 2) is **unchanged**. The evaluator is **unchanged**.

**Why the axes moved** (all measured, `docs/benchmark_grid_requirements.md`):

* The v2 **SNR axis was inert** — cmeAnalysis scored F1 .919/.929/.932/.934/.934/.933 across SNR
  2→15. That is 1.5 points of spread across a 7.5× range: it measured nothing. The entire live
  range is **SNR 0.75 → 2.0** (13 → 43 photons). 50% recall point ≈ SNR 1.1.
* The v2 **density axis stopped where the physics starts** — its ceiling (0.015) sits at mean
  nearest-neighbour = 1.0 × PSF FWHM, the *onset* of overlap. 0.025 is where NN distance (3.2 px)
  drops **below** the ch2 PSF FWHM (3.96 px) and neighbouring spots are physically merged. That is
  where the model's claim lives; cmeAnalysis is at ceiling in sparse fields (recall 0.983) and
  there is nothing to beat there.

## 2. ⛔ The retrain hypothesis was FALSE — do not implement it

`handoff_retrain_intensity_head.md` §3 offered a menu (a/b/c) premised on:

> *"solving intensity per image COUPLES brightness to density — a dense image is forced dim — so
> bright+dense is structurally unreachable."*

**This is false.** `scripts/coverage_probe.py` (new) sampled the real training code path over
3,000 images and found:

* `corr(log density, log median-A1) = **+0.02**`
* A₁ p50/p95/max **flat across every density quintile** (27.6/992/13k → 29.3/1036/13k)
* `solve_a1_ceiling()` takes gains, PSF, slope, background — **it never takes density**. The two
  axes are independent *by construction*.

→ **None of options (a)/(b)/(c) should be implemented.** Full writeup:
**`docs/coverage_probe_findings.md`**.

**Also corrected:** the "retrain must reach down to ~8 photons" BLOCKER in
`benchmark_grid_requirements.md` cites the **LEGACY** [20, 7943] range. The measured-detector
config already trains down to **3 photons** (realised support [3.0, 18,388]); all six new SNR
levels are covered, with 31–68% of trained spots at or above each. **That blocker is lifted.**

## 3. The intensity-head defect is REAL, and its cause is still unknown

Reproduced today on `hrnet_large_measured` (`results/bright_dense_probe_CURRENT.csv`):

| SNR | A₁ (ph) | density | recall | logI2_bias | log-ratio bias |
|---|---|---|---|---|---|
| 15 | 1,364.6 | 0.0006 | 0.994 | −0.185 | **−0.132** |
| 15 | 1,364.6 | 0.012 | 0.899 | −1.676 | **−1.098** |
| 10 | 624.7 | 0.012 | 0.893 | −0.821 | **−0.918** |
| 3 | 77.6 | 0.012 | 0.907 | +0.114 | −0.065 |

**Key structure: it needs bright AND dense.** At the same brightness (SNR 15), sparse is −0.132
but dense is −1.098 — 8× worse. Pure "bright spots are rare" would degrade bright spots
*everywhere*, including sparse fields. It does not.

Ruled out so far: **coverage** (§2), **crowding** (GT-position bias is *worse* for isolated spots),
**detector effects**, **soft-knee compression** (tested and refuted — do not re-litigate), **NMS**,
**dark counts**.

Remaining suspect: **rarity of the joint corner**. The axes are independent, but independence does
not populate a corner: P(bright ≈ 5%) × P(dense ≈ 20%) ≈ **1%** of trained spots. `full_dim_bias =
1.6` over-samples the dim tail, so bright is rare on its own.

## 4. ✅ DONE — the rarity probe: **REFUTED** (A100 destroyed)

**The probe ran. Rarity is dead too.** Full writeup: **`docs/rarity_probe_findings.md`**. Headlines:

* Flattening the sampler (`full_dim_bias` 1.6 → 1.0) did **not** fix the bright+dense corner — it made
  it **worse** (−0.524 → −0.832). The "gap" shrank only because the **dim end collapsed**
  (+0.253 → −0.327), which is the summary statistic being gamed by degrading the baseline.
* **The big finding:** arm B improved **both channels individually** (logI1 −1.18 → −0.38, logI2
  −1.85 → −1.28) and made the **RATIO WORSE** (−0.669 → −0.900). The two channels' errors partly
  **cancel** in the ratio; the flat sampler decorrelated them. **α depends only on the ratio, so
  per-channel accuracy is NOT the objective.** Any future fix judged on logI1/logI2 MAE will mislead.
* Cause is now believed to be in the **loss/head** (Gaussian-NLL intensity term; the `logvar` clamp
  `[-10, 6]` possibly saturating in the crowded corner), **not the data**.
* **NO 40k retrain was run.** Do not launch one on a sampler change.

*(Historical detail of how the probe was set up, retained below.)*

### Original plan (as run)

`scripts/run_rarity_probe.sh` is running on the rented A100 (`ssh ubuntu@216.81.245.244`, in tmux
session `probe`, logging to `~/probe.log`). It is a **DIAGNOSTIC, not the retrain**: two 8k-step
arms differing in exactly ONE knob (`training.intensity_window.full_dim_bias`, 1.6 vs 1.0),
generated by `scripts/make_probe_configs.py` which *asserts* the single-knob difference.

* **STEP 2 gate PASSED** — the probe reproduced the known defect to 3 decimal places (−0.918 /
  −1.098 vs the expected −0.92 / −1.10), so its numbers are trustworthy.
* **Read the DENSE column, not the overall bright-end bias.** If arm B fixes bright-sparse but
  leaves bright-dense at ~−1.0, rarity is refuted for the corner that matters.
* **Check the DIM rows too.** Flattening `dim_bias` trades bright accuracy against dim accuracy,
  and *dim is where the whole low-bias claim lives*. A fix that wins bright and loses dim is a
  downgrade.
* If **arm A** (the control, dim_bias 1.6) shows no defect, 8k steps is too short and arm B is
  uninterpretable — lengthen the arms, do not read them.

**No 40k retrain has been launched.** That decision waits on this readout.

## 5. What is next (in order), once the arms read out

1. **40k retrain** on the chosen sampler → new checkpoint dir + `PROVENANCE.md`.
2. **Retune `peak_threshold`** — 0.3 is far too permissive (precision 0.547 → **0.995** at 0.7 with
   **zero** recall cost). Tune on a val set from the TRAINING distribution, **never on the
   benchmark**, and extend the same protocol to every baseline or the comparison is rigged.
3. **`spotpipe bench-gen`** the v3 grid (42 cells, ~2,100 images, ~8 min).
4. **`spotpipe infer` + `spotpipe evaluate`** — Gate A (known-α recovery) and the α=0 null control
   FIRST, before any baseline.
5. **Re-run cmeAnalysis** — its old results are VOID (the benchmark tree changed: new cells, and
   every cell regenerates). Runner lives at
   `C:\Users\shivl\OneDrive\Desktop\matlab\cmeAnalysis-master\software`. ~70 min for the 42-cell grid.
6. **⚠️ NOT DONE — the ≥6 px border margin** (`benchmark_grid_requirements.md` §4). cmeAnalysis's
   `fitGaussians2D` structurally cannot fit a spot within ~6 px of the edge; **88% of its missed
   spots are within 6 px of the border** vs 9.2% expected by chance. That is ~5 F1 points handed
   free to any padded CNN — *us*. It is an EVALUATOR change and it must land **before** the headline.

## 6. New files (all committed)

| file | what |
|---|---|
| `scripts/coverage_probe.py` | Is the benchmark grid inside the training support? Exit 1 if not. |
| `scripts/bright_dense_probe.py` | Measures intensity bias on a pinned (brightness × density) grid. Golden-tested. |
| `scripts/make_probe_configs.py` | Generates the two probe arms; asserts they differ in one knob. |
| `scripts/run_rarity_probe.sh` | The A100 runner; self-gating (preflight → defect reproduction → arms). |
| `scripts/_torch_stub.py` | Lets distribution-only tools import the REAL sampling path on a box with no torch. |
| `tests/test_density_coverage.py` | Pins train.yaml ↔ benchmark.yaml ↔ the generator constant together. |
| `tests/test_bright_dense_probe.py` | Golden tests: perfect predictor → 0.000 bias; injected −1.10 → −1.10 back. |
| `docs/coverage_probe_findings.md` | Why the retrain hypothesis is dead. |
| `configs/train_probe_{A_status_quo,B_flat}.yaml` | The two arms (auto-generated). |

## 7. GOTCHAS discovered today — these will bite you again

* **The GitHub default branch is `master`, which is the OLD repo.** A fresh
  `git clone https://github.com/slele225/spotpipe.git` checks out `master` and gives you the old
  tree (old checkpoints, no new scripts) with **no error**. Always `git checkout main`. Better: change
  the default branch on GitHub.
* **`.gitignore` had `data/` unanchored**, which matches a dir named `data` at ANY depth — it was
  silently swallowing **`src/spotpipe/data/__init__.py`**, so `import spotpipe.data` failed on every
  fresh clone while working fine on the dev box (git does not track empty dirs). Fixed → `/data/`.
  Caught by `test_import` in the preflight, which is the argument for running the full suite before
  spending GPU time.
* `tests/test_benchmark_generation.py` hardcoded `snr=5_density=0.006`. Now derived from the config,
  so a legitimate grid change no longer looks like a test failure.
* The generator still FLAGS SNR 0.75 / 1.0 as out-of-range. That is **correct** — those cells are OOD
  *for the LEGACY checkpoints*. It is a statement about the legacy models, not about the grid.
