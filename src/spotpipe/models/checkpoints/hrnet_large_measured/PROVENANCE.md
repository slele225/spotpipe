# Checkpoint provenance — hrnet-large (MEASURED detector)

* **Label**: measured_detector_hrnet_large (NOT legacy) — reproducible headline model. NOT legacy.
* **Checkpoint**: best_checkpoint.pt, selected **step 25000** of 40000 by
  **hard_corner_val_logratio_mae** = 0.4600 (hard_n=188,
  overall val_logratio_mae=0.2245, det_f1=0.5975).
  Selection rule: hard-corner val_logratio_mae if hard_n_pairs >= hard_corner_min_pairs; else overall val_logratio_mae; else val_total_loss. NEVER uses benchmark/test outputs.
* **Final eval @ step 40000**: val_logratio_mae=0.2157 (n=4579),
  hard=0.4745 (n=192), recall=0.437,
  precision=0.954, det_f1=0.599,
  logI1_mae=0.532, logI2_mae=0.551.
* **Experiment**: measured-detector-hrnet-large — A100-SXM4-80GB, 2026-07-14.
* **Training git commit**: 26b0d487d16d6516082f20cf03a5a56a1e8e3f9b-dirty
* **Vendored simulator SHA**: 7b9a0b85ee527afeb73d9e68f9bdb30960775083
* **Seed**: 0   **Image**: 256x256   **Steps**: 40000, batch 16, lr 0.002
* **Staged schedule**: LR warmup 1500 < variance warmup 4000
  < curriculum ramp 20000; eval_every 1000. NO slope/alpha loss term.
* **Dataload fraction** (run avg): 2.3%  (Gate-1 profile: 1.3%). GPU not starved.

## Measured detector + per-image gain randomisation (CHANGE 1)
| ch | dye / lambda | PMT | measured gain (ADU/ph)* | gain RANDOMISED/image | offset | read_var | sat_knee | ENF |
|----|--------------|-----|-------------------------|-----------------------|--------|----------|----------|-----|
| ch1 = LIPID | 561 nm | 500 V | 6.63 | [3.0, 30.0] | 154.0 | 3.1 | 3941.0 | 1.0 |
| ch2 = PROTEIN | 488 nm | 750 V | 124.3 | [20.0, 150.0] | 154.0 | 4.4 | 3941.0 | 1.0 |

*measured single-gain values from config.yaml comments (photon-transfer curve).
n_frames=3, pg_threshold=20.0, adc_max=4095, noise_floor_sigma=sqrt(read_var).
CHANNEL MAPPING: ch1=LIPID(561), ch2=PROTEIN(488) — pipeline order, OPPOSITE acquisition order; a swap inverts A2/A1.

## Gain-aware-reversal note
MEASURED-detector retrain. Gains are RANDOMISED per image over gain{1,2}_range (CHANGE 1: deliberate reversal of the vendored gain-aware design, for robustness to a future PMT-voltage change). Offset / read_var / saturation_knee / excess_noise_factor / n_frames / adc_max are held at the measured values. noise_floor_sigma = sqrt(read_var).
