# Measured-detector hrnet_large retrain — design + change log

Scope: the DISPOSABLE training layer (`src/spotpipe/training/`) + `configs/train*.yaml`
+ `src/spotpipe/benchmark/matching.py`. **Nothing vendored/frozen is modified.** The
per-image gain randomisation and per-image intensity solving are done in a new driver
that composes the frozen simulator primitives (`forward_model.simulate_image`,
`noise.DetectorParams`, `psf`) — the same "config-overrides, not code-edits" pattern
the benchmark-v2 layer used.

This retrain replaces the **legacy** checkpoints, which were trained on a detector
wrong by ~20× (old ch2 gain 6.0 vs measured 124.3) and fire hundreds of spurious
detections on measured-detector images.

## Why this was a build, not a config change
The training harness was **deliberately not ported** when this repo was vendored
(CLAUDE.md). So there was no training loop, dataloader, target-map construction,
curriculum, or `val_logratio_mae` evaluator here. All of that was ported/adapted from
the old repo (`C:\Users\shivl\Videos\spotpipe`, READ-ONLY) with the departures below.

## The changes
| # | Change | Where |
|---|---|---|
| 1 | **Per-image gain randomisation** — ch1∈[3,30], ch2∈[20,150] ADU/photon, independent (gain *ratio* varies too). Everything else held at the measured value. | `training/intensity_window.py::sample_image_detector` |
| 2 | **Intensity range solved per image** from the sampled gains + PSF + ratio law, so no spot clips EITHER channel (protein ceiling via the ratio law; lipid ceiling at `adc_max/gain1`). | `training/intensity_window.py::solve_a1_ceiling` |
| 3 | Background → realistic constant ~2 photons (jitter [1,4]); old wide [2,25] + gradient/structure removed. Density/PSF/ratio/registration/clustering kept. | `configs/train.yaml` scene block |
| 4 | Same model/heads/losses + staged schedule (LR warmup 1500 < variance warmup 4000 < curriculum 20000; 40k steps, bs16, lr0.002). NO slope/alpha loss. | `configs/train.yaml`, `training/train.py` |
| 5 | **Real multi-worker DataLoader** replacing the old inline single-process generation (the ~50-hour data-starve bug). | `training/dataset.py::SpotStreamDataset`, `make_loader` |
| 6 | Provenance records the measured detector + gain ranges + sim SHA + seeds; labelled measured-detector (NOT legacy). | `training/train.py::_write_run_outputs` |

## Design decisions made here (open to veto)
- **Gain-randomisation reverses a documented frozen-module intent.** `simulator/noise.py`
  forbids broad gain jitter by design ("the network must remain gain-aware, not
  gain-invariant"), and `normalize_counts` preserves absolute scale for the same reason.
  CHANGE 1 deliberately reverses this for robustness to a future PMT-voltage change. We
  do NOT edit the vendored module — we construct `noise.ChannelDetector`/`DetectorParams`
  directly per image. The reversal is recorded in every run's `manifest.json`.
- **Curriculum × solve composition.** The per-image solve sets the intensity *ceiling*;
  the curriculum ramps the *dim tail* below it — bright-only at `t=0` (0-decade window)
  → full dim-biased tail at `t=1`. So CHANGE 2 (ceiling) and CHANGE 4 (curriculum in the
  dim×overlap regime) compose without fighting. `curriculum_scene_config` therefore ramps
  density/overlap/background/scatter only; intensity is handled in `_resolve_intensity_window`.
- **`saturation_knee = adc_max − offset = 3941`** for the measured detector (benchmark-v2
  precedent): the soft-tanh knee's asymptote sits at the hard 12-bit clip.
- **Dropped the old auto-benchmark** from the training driver (its harness modules aren't
  ported). STOP #2's benchmark-v2 smoke uses the existing `spotpipe infer` adapter.

## CPU sanity gates (all pass here)
- **Overfit** (4–8 images, ≤300 steps): loss collapses; `logI{1,2}_mae` ≈ 0.005–0.02
  (the fixed set now includes ~3-photon spots, which are intrinsically noisier to fit exactly).
- **Solved-A1 window** @ full difficulty (256×256, 5000 samples): ceiling median **468 ph**
  (max 20 792), window median **2.0 decades**, **floor 3 ph** (dim end reaches exactly 3.0 ph;
  46% of images cover ≤3.5 ph, 57% ≤8 ph — spanning the benchmark-v2 dim curvature spots
  2.9–7.9 ph), **0% degenerate**, protein channel binds **68%**. Ceiling ≪ old impossible 7943 ph.
- **3-photon render**: pinned 3-ph spots render as valid uint16 with finite, non-degenerate
  logI targets and zero saturation (no all-zero / NaN maps).
- **Dataload profiling**: the multi-worker loader works (Windows spawn OK). CPU fraction is
  ~1% but not the real gate — **the <20% gate must be re-run on the GPU box** (compute is
  far faster there, so the fraction rises).
- **Tests**: `tests/test_training.py` (14) + full suite (53) green.

## Running the real training (on the Linux GPU box)
```
# from the GPU box, after scripts/sync_to_remote.sh <user@host> <remote_root>
spotpipe train --mode profile --config configs/train.yaml --require-gpu    # MUST print <20%
spotpipe train --config configs/train.yaml --device cuda --require-gpu     # 40k-step run
```
The best checkpoint (`best_checkpoint.pt`) is selected on hard-corner `val_logratio_mae`
against the fixed in-memory val set. Provenance is written to the run dir's `manifest.json`.
The trained checkpoint should be installed under
`src/spotpipe/models/checkpoints/hrnet_large_measured/` with a `PROVENANCE.md`.
