# HANDOFF — retrain to fix the intensity head + widen the benchmark to match real data

> ## ⚠️ AMENDED 2026-07-14 (later the same day) — §2 IS REFUTED, §3 IS MOOT
>
> The §2 "cheap check" was run (`scripts/coverage_probe.py`) and it **killed the hypothesis**,
> exactly as §2 said it might. Read **`docs/coverage_probe_findings.md`** before acting on
> anything below.
>
> * **Brightness is NOT coupled to density.** `solve_a1_ceiling()` takes gains, PSF, slope and
>   background — it **never takes density**. Measured over 3,000 sampled training images:
>   `corr(log density, log median-A1) = +0.02`, with a **flat** A₁ p50/p95/max profile across
>   every density quintile. "Bright AND dense" was never structurally unreachable.
> * **Therefore §3's options (a) / (b) / (c) are all answers to a false question.** Do not let
>   images clip, do not lower the simulated ch2 gain, do not declare bright+dense out of scope
>   *on coverage grounds*.
> * **§1's evidence still stands** — the defect is real. Its cause is now *unknown*: coverage,
>   crowding, detector and soft-knee are ALL ruled out. The last suspect is **rarity** (only ~5%
>   of trained spots exceed ~1,000 photons, because `full_dim_bias = 1.6` over-samples the dim
>   tail). That is being tested by `scripts/run_rarity_probe.sh` — two short A/B arms differing
>   in exactly one knob — **before** any 40k-step run.
> * **The §1 defect lives in cells the new grid no longer has.** It was characterised at SNR 10
>   and 15 → **625 and 1,365 photons**. The v3 grid's brightest cell is **77.6 photons**.
> * **The §5/BLOCKER low-SNR claim below is STALE**: it cites the *legacy* [20, 7943] photon
>   range. The measured-detector config already trains down to 3 photons, so all six new SNR
>   levels are covered. No intensity-range change is needed.
>
> What survives from this handoff: **§1 (the evidence)** and **§4 (the threshold retune)**.

*Written 2026-07-14, after the full benchmark-v2 evaluation of `hrnet_large_measured`.
Read this + `results/RESULTS_hrnet_large_measured.md` + `docs/PROJECT_STATE.md` before starting.*

## Where things stand

The full benchmark-v2 run is **done** and the model is characterised
(`results/RESULTS_hrnet_large_measured.md`). Two model defects were found and diagnosed:

1. **The intensity head collapses in bright AND dense fields.** ← the reason for this retrain
2. `peak_threshold = 0.3` is far too permissive (free fix, see §4 — independent of the retrain).

α recovery is **fine** and is unaffected by either (α MAE 0.036; α=0 null returns +0.021 ± 0.002).
The evaluator passed Gate A (recovers all 12 injected α to ≤0.005; null +0.0016 ± 0.0019). The
benchmark's registration shift is fixed and verified on-disk. None of that needs redoing.

---

## 1. The defect, and the evidence it is the MODEL

In the bright + dense cells the model's log-ratio bias reaches **−1.10** (a factor-3 error in
A2/A1). `logI2_bias` hits −1.66 / −2.09 while `logI1_bias` stays near −0.5: the **protein channel is
under-read ~5×**, specifically where the field is bright AND dense.

**Gate C** (`scripts/gate_c_intensity_on_gt.py`) removes the model — it runs the shared intensity
instrument at the GROUND-TRUTH positions, i.e. a perfect detector with perfect localization — and
proves the pixels are fine:

| condition | model bias | GT-position bias (perfect detector) |
|---|---|---|
| snr=10_density=0.012 | **−0.920** | **−0.032** |
| snr=15_density=0.012 | **−1.100** | **−0.168** |
| snr=15_density=0.0006 | −0.136 | −0.098 |
| snr=2_density=0.015 | −0.016 | +0.035 |

No detector effect explains a factor of 3. **Crowding is ruled out** as well: the GT-position bias is
*worse* for isolated spots (−0.249) than crowded ones (−0.159) — the opposite of an
aperture-contamination signature. An earlier soft-knee-compression hypothesis was **tested and
refuted** by exactly this table; don't re-run it.

(Minor, separate: the shared extraction instrument itself under-reads ch2 by ~0.10–0.17 in log at
SNR=15, tracking the 10.7% of ch2 pixels above half the knee. Mild compression is real but accounts
for ~15% of the model's error, not the whole. Quantify before quoting any method's intensity numbers
to 2 dp.)

## 2. The hypothesis — CONFIRM IT BEFORE SPENDING 40k STEPS

**Suspected cause: a training-coverage gap that is structural, not accidental.**

The training distribution **solves per-image intensity to keep both channels unclipped**. That
*couples brightness to density*: a dense image is forced dim, because many bright spots would clip
ch2 (gain 40, knee 3941 ADU). So "bright AND dense" is not merely rare in training — it is
**structurally unreachable**. Landing there at test time, the intensity head appears to regress
toward its mean.

**Cheap check first (~20 min, do NOT skip):** sample the training config's simulator and plot the
realised joint (A₁, area-density) coverage against the benchmark grid. If the bright+dense corner
falls outside the training support, the hypothesis is confirmed and it dictates the retrain config.
If it does NOT, stop — the cause is something else and this handoff's §3 is wrong.

## 3. The design decision (Shiv's call — do not guess)

If the gap is confirmed, the retrain must break the brightness↔density coupling. The options are not
free, and they encode what you want the model to be good at:

* **(a) Sample intensity independently of density and let some images clip**, flagging them. Closest
  to the real instrument, where clipping *does* happen. The model learns to cope with saturated
  pixels instead of never seeing them.
* **(b) Lower the simulated ch2 gain** so bright+dense stays unclipped. Keeps the clean no-clip
  invariant, but walks away from the measured detector — and the benchmark's protein gain of 40 is
  *already* a "planned acquisition setting", not a measurement (see `configs/benchmark.yaml`).
  Changing it again needs a reason you're willing to defend.
* **(c) Declare bright+dense out of scope** and label those benchmark cells as out-of-distribution.
  Legitimate, and cheap, but it forecloses the dense real-data regime.

Note (a) and (b) interact with the plan to **widen the benchmark to match real data** (lower SNR,
higher density): whatever training range you pick must *cover* the new benchmark grid, or you will
simply relocate the same coverage gap. Fix the training distribution and the benchmark grid
**together, in that order** — training support first, benchmark grid inside it (except where an
out-of-distribution stress cell is deliberate and flagged).

Physical ceiling to respect: the generator asserts no cell clips, and at ch2 gain 40 that caps SNR
near **16.8**. *Lower* SNR and *higher* density are both free; *brighter* is what's constrained.

## 4. Independent of all the above: the threshold retune

`peak_threshold = 0.3` (carried in the checkpoint config) is far too permissive. From
`scripts/threshold_sweep.py` on `snr=5_density=0.002`:

| peak_threshold | recall | precision | F1 | FP |
|---|---|---|---|---|
| 0.3 (shipped) | 0.986 | 0.547 | 0.703 | 2677 |
| 0.7 | 0.986 | 0.995 | 0.990 | 16 |

**Recall does not move** (0.9860 → 0.9860). The FPs are background hallucinations, not duplicate
peaks (`fp_near_frac` ≈ 0.1; mean p_detect 0.42). **NMS is exonerated — leave `nms_kernel = 3`**
(raising it only costs recall). The bright/sparse control bounds the top end: at 0.8 recall collapses
to 0.921. Usable window ≈ **0.6–0.7**.

⚠️ **Retune on a VALIDATION set from the training distribution, never on the benchmark.** The
benchmark is the test set; picking 0.7 because it wins there is contamination, and it would be *our*
method getting tuning that the baselines don't. Record the chosen value in the checkpoint config with
a note, and extend the same tuning protocol (same procedure, same data, same effort) to every
baseline — SpotMAX, Spotiflow, cmeAnalysis — or the comparison is rigged.

## 5. Suggested order

1. Coverage plot → confirm or kill the §2 hypothesis.
2. Decide §3 (a / b / c). This determines both the training config AND the new benchmark grid.
3. Retrain → new checkpoint dir + `PROVENANCE.md` (record simulator SHA, config, seeds).
   `infer.is_legacy_checkpoint()` derives the LEGACY/HEADLINE label from the training SHA, so a clean
   retrain is auto-labelled correctly — nothing to edit there.
4. Retune `peak_threshold` off-benchmark; write it into the checkpoint config.
5. Widen `configs/benchmark.yaml` (lower SNR, higher density) → `bench-gen` → `infer` → `evaluate`.
   Also update the manifest's `in_legacy_training_distribution` flags to compare against the NEW
   model's actual training range (they currently reference the legacy checkpoints').
6. Re-verify: `scripts/verify_benchmark_registration.py` (registration shift stays at 0 — the
   forward model's default is 1.0, so `_ZERO_REGISTRATION` must be applied to any new set), Gate A
   (`spotpipe evaluate --oracle`), then Gate C on the new grid.
7. Only then: baselines.

## Do not re-litigate

* Soft-knee compression as the cause of the intensity collapse — **tested, refuted** (§1).
* Dark counts as the cause of the false positives — **impossible**: `BENCH_MANIFEST` lists
  `protein_pmt_dark_counts` under `known_unmodeled_features`; they are not in the benchmark pixels.
* NMS as the cause of the false positives — **exonerated** (§4).
* The registration shift — **fixed and verified**; just don't drop `_ZERO_REGISTRATION` from any new
  benchmark family (the forward model's default `max_px` is 1.0, so omission silently re-enables it).
