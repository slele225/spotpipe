# `data/` — generated artifacts, not committed data

Benchmark/test datasets are **generated artifacts, not committed data.** Nothing
under `data/` except this README (and other `*.md` docs) is tracked by git — the
generated images, ground-truth tables, per-image metadata, the per-channel
TIFFs, the checksums, and the dataset manifest are all git-ignored (see
`.gitignore`: `data/benchmark_test_*/` ignores the whole frozen-set directory).

## Why generation, not git

The dataset is large binary data and is fully reproducible from

```text
seed + config + git commit + code version
```

Because generation is seeded and deterministic, the same code/config/seed
reproduces the same frozen dataset. There is no need to SCP the dataset through
git; **only the generation script and config are versioned.** The manifest's
per-artifact SHA-256 checksums let any checkout confirm a regenerated set matches
(`make_benchmark_set.py --verify`).

The intended workflow is:

```text
author code locally → commit / push code → pull repo on the GPU instance
→ generate the frozen benchmark/test dataset ON the instance → train + benchmark
```

## Phase-5b validation set (`data/fixed_eval/val`)

Distinct from the frozen test set below. The **validation set** is used ONLY for
best-checkpoint selection and is SHARED by the `hrnet-small` and `hrnet-large` runs so
both models are selected on byte-identical data. It is hard-corner-rich (over-sampled
dim × high-overlap images) so the hard-corner selection metric has enough matched pairs
to be stable. Build it ONCE on the training box (rebuild here, don't copy across
machines — the eval detector must match this box's training detector):

```bash
uv run python scripts/build_fixed_eval.py \
    --config experiments/2026-06-23_hrnet-small/config.yaml \
    --out data/fixed_eval/val --split val --seed 70001 --n-images 32 --n-hard-corner 10
```

It uses a **different seed** than the frozen test set, so selecting on val and reporting
on test do not leak. Layout is the `generate_dataset` form (`manifest.json` + `images/`
+ `spots/` + `meta/`), read by both the training loader and the benchmark harness.

## Create the FROZEN, stratified benchmark/test set

```bash
uv run python scripts/make_benchmark_set.py --config configs/benchmark_test_set.yaml
```

This is the SEPARATE, FROZEN, externally-ingestible benchmark/test set used ONLY
for final reporting and external-method comparison — **never** for checkpoint
selection (that is the training-validation set, a *different* seed/manifest;
keeping them distinct prevents test-set leakage). It is **stratified** (targeted
/ rejection generation populates every SNR × density × β cell, oversampling the
dim × high-overlap corner) — not a random draw. See
`src/spotpipe/simulator/benchmark_set.py` and CLAUDE.md.

Generation prints a stratification report and round-trip / export / checksum
confirmations, and **fails loudly** if any SNR × density cell is underpopulated
after `max_attempts` rather than silently shipping a thin set.

### Layout

```text
data/benchmark_test_v1/
  images/<id>.npy             # canonical two-channel array, uint16 [2,H,W]
  images_ch1_raw/<id>.tif     # observed detector counts, ch1, uint16 [0, adc_max]
  images_ch2_raw/<id>.tif     # observed detector counts, ch2, uint16 [0, adc_max]
  images_ch1_photon/<id>.tif  # offset-subtracted, gain-corrected photon-prop., ch1, float32
  images_ch2_photon/<id>.tif  # offset-subtracted, gain-corrected photon-prop., ch2, float32
  meta/meta_<id>.json         # full per-image simulator metadata (feature parity)
  audit/background_<id>.npy   # TRUE simulator background [2,H,W] — NON-FAIR, debug only
  ground_truth.csv            # canonical spotpipe.schema GT table, all images
  beta_per_image.csv          # true alpha / beta / beta-group per image
  metadata.csv                # per-spot binning/filtering metadata (SNR, density, bins, flags)
  checksums.sha256            # per-file sha256 of every artifact
  manifest.json               # frozen bin edges, full config, git commit, counts, checksums
```

The whole directory is git-ignored (`data/benchmark_test_*/`).

### Raw vs photon-proportional images (the fairness rule)

- **Raw per-channel TIFFs** are observed detector counts. External single-channel
  tools (Spotiflow / DECODE) **detect / localize** on these — that is what real
  microscopy hands them.
- **Photon-proportional per-channel TIFFs** are the *only* images adapters use to
  **extract intensity** `I1`/`I2`, via their own declared local-background /
  aperture / PSF-fit estimator. The exact per-channel correction (recorded in
  `manifest.json`) is

  ```text
  photon_k = (raw_counts - offset_k) / gain_k
  ```

  a linear pedestal+gain correction only — it deliberately does **not** invert the
  saturation knee (a fair method cannot know it). Adapters must never divide raw
  counts, and never read `audit/` (true) background unless explicitly labeled
  oracle.

## Verifying integrity (freeze means freeze)

```bash
uv run python scripts/make_benchmark_set.py --verify --out data/benchmark_test_v1
```

recomputes every per-file and per-directory SHA-256 and compares them to
`manifest.json`, so a later run can confirm it is operating on the unmodified,
frozen dataset before reporting numbers on it.

## Rules

- **Do not commit the generated dataset.** It is reproduced on the instance, not
  carried in git.
- **Do not use this benchmark/test set for checkpoint selection.** Use it only for
  final reporting and external-method comparisons. The eval set is fixed and held
  out across the whole training curriculum (see `CLAUDE.md`).
- Once external methods start running on it, **it must not change** — re-freeze a
  new version (`benchmark_test_v2`) instead of mutating `v1`.
