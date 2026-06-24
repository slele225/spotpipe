# `cmeanalysis_plus_aperture` — CMEAnalysis (external detector) + aperture photometry

A benchmark method that uses **CMEAnalysis as detector/localizer only** and
extracts canonical intensities in-repo by aperture + annulus photometry on the
photon-proportional images. It is opt-in (not in the default `methods:` list);
enable with `--methods cmeanalysis_plus_aperture`.

## Architecture / boundaries

- **CMEAnalysis is external and local-only.** It is **never vendored** into this
  repo, and its source is **never modified**. The MATLAB wrapper that drives it
  also lives **outside** the repo (and outside the CMEAnalysis source tree).
- **The only spotpipe ↔ CME interface is a normalized detections CSV** (below).
  The Python adapter (`spotpipe.benchmark.cmeanalysis` +
  `CmeAnalysisPlusApertureAdapter`) depends on that CSV and nothing else — it
  never parses CME `.mat` files and never sees MATLAB.
- **Fairness rules** (same as every external adapter):
  - CME detects/localizes on the **raw** per-channel images.
  - Canonical `I1`/`I2` come from **aperture + annulus** photometry on the
    **photon-proportional** images (`images_ch{1,2}_photon/`), using the *same*
    estimator as `classical_per_channel_aperture` — so the only difference between
    the two methods is the detection source.
  - CME native amplitudes (`A`, `slave_A`) are **not** used as `I1`/`I2` (they are
    peak amplitudes, not integrated intensities). Using them would be a different
    method that must be renamed and documented.
  - Raw counts are never divided; the `audit/` true background is never read.

## Normalized detections CSV contract (primary interface)

A single CSV across all images.

**Required columns**

```
image_id, x, y
```

- `image_id` matches the eval-set image id (the frozen-set `<id>`).
- `x` = sub-pixel **column**, `y` = sub-pixel **row**, in spotpipe's **0-indexed**
  pixel convention. The external MATLAB wrapper converts MATLAB's 1-indexed
  coordinates (`x - 1`, `y - 1`).

**Optional columns** (provenance / scoring; never used as canonical `I`)

```
score, A, slave_A, channel, native_I1, native_I2
```

- `score` — a CME confidence / p-value-like quantity (see `p_detect` modes).
- `A` / `slave_A` — CME master / slave fitted amplitudes.
- `channel` — detect (master) channel index, if recorded.
- `native_I1` / `native_I2` — optional native CME intensities (provenance only).

### `p_detect` modes (`cmeanalysis.p_detect_source`)

CMEAnalysis has no native higher-is-more-confident probability (`pval`-like fields
are *smaller*-is-more-confident), so the safe default is **`constant` → 1.0**.

| mode | meaning |
|------|---------|
| `constant` (default) | every detection gets `p_detect = 1.0` |
| `A` | master amplitude, normalized by its per-call max |
| `score` | the `score` column, clipped to `[0, 1]` |
| `one_minus_pval` | `1 - score` (treats `score` as a p-value) |
| `neg_log10_pval` | `-log10(score)`, normalized by its per-call max |

Missing source columns fall back to `1.0` (a minimal `image_id,x,y` CSV works).

## Config block (`configs/benchmark.yaml` → `benchmark.cmeanalysis`)

```yaml
cmeanalysis:
  detections_csv: data/benchmark_test_v1/cme_detections/detections.csv
  detect_channel: 2             # CME master channel (layout/provenance only)
  p_detect_source: constant     # constant | A | score | one_minus_pval | neg_log10_pval
  window_radius_px: 3.0         # aperture radius (mirrors the aperture baseline)
  bg_inner_px: 4.0              # local-background annulus inner radius (px)
  bg_outer_px: 7.0              # local-background annulus outer radius (px)
  use_photon_images: true
```

`detect_channel` selects the master channel for layout/provenance only; canonical
`I1`/`I2` are always aperture reads on `images_ch1_photon` → `I1` and
`images_ch2_photon` → `I2`.

## Expected external MATLAB wrapper

You maintain this file outside the repo, e.g.
`…/cme analysis stuff/spotpipe_external/spotpipe_cme_detect.m`. Expected signature:

```matlab
spotpipe_cme_detect(condDir, masterChName, slaveChName, outCsv, NA, M, pixelSizeM, Alpha)
```

It should run CMEAnalysis headlessly and emit the normalized CSV:

1. `loadConditionData(condDir, {masterChName, slaveChName}, {markers}, 'Parameters', [NA M pixelSizeM])`
   — no GUI prompts (channels/markers/parameters passed in). The **master channel
   is `chNames{1}`**.
2. `runDetection(data, 'Master', 1)` — writes `<masterCh>/Detection/detection_v2.mat`
   per image.
3. For each image, load `detection_v2.mat`, take `frameInfo(1)`, and for the master
   channel `mCh`:
   - `x = x(mCh,:) - 1; y = y(mCh,:) - 1;` (1-indexed → 0-indexed, col/row)
   - optionally `A = A(mCh,:)`, `slave_A = A(slaveCh,:)`, `score = pval_Ar(mCh,:)`
   - `image_id` = the per-image subfolder name
4. `writetable` → `outCsv` with header `image_id,x,y[,score,A,slave_A,channel]`.

The repo-side runner (`scripts/run_cmeanalysis.py`) prepares a clean 2-channel
condition folder from the frozen raw TIFFs
(`<work>/condition/<image_id>/ch{1,2}/<id>.tif`) and invokes this wrapper with
**both** the CMEAnalysis software folder and the wrapper folder on the MATLAB path.

## Commands

Validate a pre-exported normalized CSV (no MATLAB):

```bash
uv run python scripts/run_cmeanalysis.py \
  --input-format normalized_csv \
  --detections-csv data/benchmark_test_v1/cme_detections/detections.csv
```

Lay out inputs + run the external MATLAB wrapper over the frozen set:

```bash
uv run python scripts/run_cmeanalysis.py \
  --input-format external_matlab \
  --frozen-dir data/benchmark_test_v1 \
  --detect-channel 2 \
  --cme-software-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/cmeAnalysis-master/software" \
  --matlab-wrapper-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/spotpipe_external" \
  --matlab-entrypoint spotpipe_cme_detect \
  --na 1.49 --magnification 108 --pixel-size-m 6.5e-6 \
  --out data/benchmark_test_v1/cme_detections/detections.csv
```

> Microscope/PSF values (`--na`, `--magnification`, `--pixel-size-m`) are **not**
> silently baked in — the runner logs the exact values used on every run. Known
> approximate imaging values: NA ≈ 1.49, sample pixel size ≈ 0.07 µm.

**Full frozen set: run in batches (recommended).** CMEAnalysis auto-estimates the
PSF sigma by sampling `nf = round(40/nd)` frames per movie; for a large condition
folder (`nd` ≈ 139) this rounds to **0**, so sigma becomes `NaN` and detection
aborts. Run in stable-order batches so each batch auto-estimates sigma independently
(per-batch CSVs are concatenated into `--out`, preserving `image_id`):

```bash
uv run python scripts/run_cmeanalysis.py \
  --input-format external_matlab \
  --frozen-dir data/benchmark_test_v1 \
  --batch-size 64 \
  --detect-channel 2 \
  --cme-software-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/cmeAnalysis-master/software" \
  --matlab-wrapper-folder "C:/Users/shivl/OneDrive/Desktop/matlab/cme analysis stuff/spotpipe_external" \
  --matlab-entrypoint spotpipe_cme_detect \
  --na 1.49 --magnification 108 --pixel-size-m 6.5e-6 \
  --out data/benchmark_test_v1/cme_detections/detections.csv
```

`--batch-size 64` splits the 139-image set into 64 + 64 + 11. To diagnose the
sigma boundary directly, use `--input-format diagnose_sigma --sweep-limits 4,8,16,32,64,full`.

Run the benchmark with the CME method (on the frozen set):

```bash
uv run python scripts/run_benchmark.py \
  --frozen-dir data/benchmark_test_v1 \
  --config configs/benchmark.yaml \
  --methods cmeanalysis_plus_aperture,classical_per_channel_aperture,oracle_center_aperture_divide \
  --out outputs/bench_cme_full
```

All generated artifacts (detections CSV, benchmark outputs) live under git-ignored
paths (`data/**`, `outputs/`).
