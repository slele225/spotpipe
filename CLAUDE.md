# CLAUDE.md — spotpipe

Durable rules for this repo. Later prompts rely on these being here. Read before
making changes.

## What this is

A two-channel microscopy spot-detection pipeline. A neural network detects
diffraction-limited spots in two-channel confocal images and estimates per-spot
intensities in each channel, so a downstream log-ratio / slope analysis can
recover a biological relationship. Training is on synthetic data from a forward
model of an Olympus FV3000 (analog-integration PMT detector). Real-data
calibration comes later.

## Repo structure & the shared-code rule

- **Shared code lives ONLY in `src/spotpipe/`.** There is exactly one copy of
  any shared code. It is imported as `from spotpipe.simulator import ...` from
  anywhere — never via `sys.path` manipulation or path appending (the package is
  installed editable).
- **Experiments hold config + README + outputs ONLY — never code.** An
  experiment is defined as *"shared code at a pinned git commit + this config."*
  Copying code into an experiment folder is never necessary and never allowed.
- New experiments are stamped with `scripts/new_experiment.py <slug>`, which
  records the current git commit into the experiment's `config.yaml`.

## Forward model & detector physics

- The forward model simulates an FV3000 analog-integration PMT chain. The two
  channels are imaged at **different PMT voltages**, so they have different
  per-channel gains and saturation behavior.
- **Detector-physics parameters** — per-channel gain, offset, excess-noise
  factor, saturation knee, noise floor, frame-averaging factor — are **FIXED or
  narrowly randomized**. They are real instrument constants to be measured
  later, not scene variables. Do **not** broadly randomize them; we do **not**
  want the network to be gain-invariant.
- **Scene parameters** — spot density, spot intensity, the biological ratio law
  including its slope β, per-spot ratio scatter, background, PSF width and
  C1-vs-C2 mismatch, channel registration shift — are randomized **WIDELY** as
  domain randomization.
- **β (the ratio-law slope) is varied per image, including β = 0.** If β were
  fixed, the network would learn it as a prior and bias the very quantity we
  measure.
- All bookkeeping is in **photon-proportional units**; observed counts are
  derived from them. **Per-channel offset is subtracted before any ratio or log
  is taken**, in both simulation bookkeeping and inference.
- **Frame averaging is `integration count = 3`** (mean of 3 scans): simulate by
  **reducing added-noise variance** accordingly, **not** by scaling the signal.

## Intensities & supervision

- `logI1`/`logI2` are the per-spot **TOTAL INTEGRATED** intensity (in
  photon-proportional units), regressed directly by the network as a scalar at
  each spot center — **NOT** a sum or read of image pixels. The network learns
  to deblend overlapping spots; a pixel/window readout would be contaminated by
  neighboring spots, so it is never used.
- Intensity losses are **masked to ground-truth spot centers**, and the
  supervision target is each spot's **true integrated photon count from the
  simulator (pre-detector)**, before spots are summed into the image.

## Uncertainty (part of phase 1, not deferred)

- Per-spot **heteroscedastic uncertainty** is part of phase 1. The intensity
  heads predict **log-variance per channel**; the loss is a **Gaussian NLL** on
  `logI1`/`logI2`. These populate the `uncertainty1`/`uncertainty2` schema
  columns and let the downstream slope fit be uncertainty-weighted.
- Heavily overlapped / dim spots should come back with **larger** predicted
  uncertainty.

## No slope loss in phase 1

- The ratio-law slope **β is computed explicitly DOWNSTREAM of inference** from
  per-spot `logI1`/`logI2` — **never trained on**.
- An in-batch slope would be a **biased (attenuated)** estimator because the
  regressor (predicted `logI1`) carries error, and training on it could distort
  per-spot intensities — exactly the per-spot unbiasedness the project must
  demonstrate.
- Slope supervision may be revisited **only after** per-spot estimates are shown
  unbiased on their own.

## Evaluation & curriculum

- The **validation/evaluation set is FIXED** across the whole training
  curriculum. The training set difficulty ramps (density, overlap, noise,
  background), but bias/variance is always measured on the **same held-out set**
  spanning the **full final difficulty range** — including the **dim ×
  high-overlap corner** — so curriculum progress never confounds the metric.
- The curriculum varies **scene difficulty only, never detector constants**.
- During data generation, the **dim-spot tail and high-overlap regime are
  over-sampled** relative to uniform, because that is the regime the project's
  central low-bias / low-variance claim targets.

## Real-data domain adaptation (phase 2, optional)

- Phase 2 is **optional and never part of the phase-1 baseline**. It uses
  **geometric-transform consistency** (flips / rotations / shifts with
  inverse-transform agreement) on **unlabeled real images ONLY**.
- **No photometric/intensity transforms** — demanding intensity-invariance would
  corrupt the measured quantity (same logic as not randomizing detector gain).
- Synthetic supervised batches are mixed in throughout.

## Output schema

- Every spot-detection method (our model and external baselines) emits the
  **canonical schema** in `spotpipe.schema`, identically. All analysis and
  benchmarking depend on this.

## Build order (staged across prompts — do not jump ahead)

1. Skeleton (this prompt).
2. Forward model + noise in isolation.
3. Model + losses + training.
4. Benchmark harness + method adapters.
5. Then experiments.

Scientific logic that belongs to a later stage stays as a stub
(`raise NotImplementedError`) until its stage.
