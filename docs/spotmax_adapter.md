# SpotMAX (external detector) + aperture photometry

Two opt-in benchmark methods that use **SpotMAX as detector/localizer only** and
extract canonical intensities in-repo by aperture + annulus photometry on the
photon-proportional images. They **share one adapter** and differ ONLY in how
SpotMAX was configured to detect (carried honestly in the method name + `flags`):

| method | SpotMAX detector |
|--------|------------------|
| `spotmax_ai_plus_aperture` | `spotMAX AI` |
| `spotmax_threshold_plus_aperture` | `Thresholding` + `peak_local_max` (classical, non-AI) |

Both are opt-in (not in the default `methods:` list); enable with e.g.
`--methods spotmax_threshold_plus_aperture`. Pick the name that matches the
detector your working INI actually used — never label a thresholding run as `ai`.

## Architecture / boundaries

- **SpotMAX is external and isolated.** It is **never vendored** into this repo,
  and importing `spotpipe` never imports `spotmax`. SpotMAX runs **headless via
  its CLI in a separate environment** (`spotmax -p config.ini`); only its output
  *tables* come back to us.
- **The only spotpipe ↔ SpotMAX interface is a detections CSV.** The Python side
  (`spotpipe.benchmark.spotmax` + `SpotmaxPlusApertureAdapter`) depends on a
  normalized/neutral CSV and nothing else — it never imports SpotMAX and never
  drives the GUI.
- **Fairness rules** (same as every external adapter):
  - SpotMAX detects/localizes on the **raw** detector image — here
    `detect_image = raw_max` (pixelwise max of the two raw channels), exported as
    SpotMAX-compatible TIFFs.
  - Canonical `I1`/`I2` come from **aperture + annulus** photometry on the
    **photon-proportional** images (`images_ch{1,2}_photon/`), using the *same*
    estimator as `classical_per_channel_aperture` — so the only difference from
    the aperture baseline is the detection source.
  - SpotMAX **native intensities are not used** as `I1`/`I2` (they would be a
    different, separately-named method whose units must be verified first).
  - Raw counts are never divided; the simulator's true-background files are never
    read.

## Two-stage CSV contract

SpotMAX output column names are **version-specific and not assumed**. The parser
auto-detects coordinate columns (override with `--x-col`/`--y-col`/`--p-col`).

### 1. Neutral detections CSV (audit-friendly intermediate)

```
image_id, x, y, p_detect, native_source_file, native_row, native_columns_json
```

- `x` = sub-pixel **column**, `y` = sub-pixel **row**, 0-indexed, origin top-left.
  SpotMAX is scikit-image flavoured (`(z, y, x)`, x=col, y=row) — already
  spotpipe's convention — but the chosen source columns are recorded so the
  mapping is **traceable, not assumed**.
- `p_detect` = a SpotMAX confidence column if one is present, else **`NaN`**
  (never fabricated — SpotMAX has no native higher-is-better detection probability).
- `native_columns_json` preserves the full native row for traceability.

### 2. Normalized detections CSV (the adapter's input)

The neutral CSV **is** a valid input — it has the required `image_id, x, y` (plus
optional `p_detect`). The adapter ignores the provenance columns.

## Output table preference

When several SpotMAX tables are present, the parser prefers (substring match, so
the run-number prefix need not be known):

```
1_valid_spots  >  0_detected_spots  >  2_spotfit
```

SpotMAX is used as a localizer, so filtered "valid" spots first, then all
"detected" spots; `2_spotfit` is only a positional fallback (its native
intensities are never used).

## Non-positive intensity policy (`spotmax.nonpositive`)

After background subtraction a detection on flat background can have ≤ 0 aperture
signal. Handled **explicitly**, never silently dropped, with transparent counts:

| policy | behaviour |
|--------|-----------|
| `clamp` (default) | keep the spot, clamp `I` at a tiny floor, append `nonpos_clamped` to its `flags` |
| `reject` | drop the spot (counted in the reported stats) |

Both the converter and the harness adapter print `n_in / n_out / n_nonpositive`.

## Config block (`configs/benchmark.yaml` → `benchmark.spotmax`)

```yaml
spotmax:
  detections_csv: external_runs/spotmax/smoke/predictions/neutral_detections.csv
  detect_image: raw_max         # pixelwise max(ch1_raw, ch2_raw) fed to SpotMAX
  nonpositive: clamp            # clamp (keep+flag) | reject (drop+count)
  window_radius_px: 3.0         # aperture radius (mirrors the aperture baseline)
  bg_inner_px: 5.0              # local-background annulus inner radius (px)
  bg_outer_px: 8.0              # local-background annulus outer radius (px)
```

## Commands

**1. Export SpotMAX inputs (in the spotpipe env)** — writes a Cell-ACDC Position
tree, an `id_map.csv`, and a headless INI *template*:

```bash
uv run python scripts/export_spotmax_input.py \
  --benchmark data/benchmark_test_v1 \
  --out external_runs/spotmax/smoke --n-images 5 --detect-image raw_max
```

**2. Run SpotMAX (in a SEPARATE SpotMAX env)** — the INI is a starting template;
SpotMAX's INI schema is version-specific, so if it errors, open the exported
`input/` folder in the SpotMAX GUI, set parameters, and export a fresh INI:

```bash
spotmax -p external_runs/spotmax/smoke/config.ini
```

Then inspect the produced `…/SpotMAX_output/` tables and their **column names**.

**3. Convert SpotMAX output → neutral + canonical CSVs (in the spotpipe env)** —
reports detections parsed and non-positive counts:

```bash
uv run python scripts/convert_spotmax_output.py \
  --spotmax-run external_runs/spotmax/smoke \
  --benchmark data/benchmark_test_v1 \
  --out external_runs/spotmax/smoke/predictions/spotmax_ai_plus_aperture_predictions.csv
# add --x-col/--y-col/--p-col if the auto-detected columns are wrong
```

**4. Benchmark via the harness adapter** (reads the neutral CSV from config; the
adapter reproduces the identical photometry):

```bash
uv run python scripts/run_benchmark.py \
  --frozen-dir data/benchmark_test_v1 --limit 5 \
  --methods spotmax_ai_plus_aperture \
  --config configs/benchmark.yaml \
  --out external_runs/spotmax/smoke/benchmark
```

Run the full set (drop `--limit`) only after the smoke run succeeds. For the
canonical CSV of the threshold method, pass `--method spotmax_threshold_plus_aperture`
to `convert_spotmax_output.py`.

## Full set in memory-bounded batches (`scripts/run_spotmax_batches.py`)

A 5-image SpotMAX run peaked >95% RAM, so the full frozen set is processed in
independent batches (default 5 images), each a **fresh** `spotmax` process that
releases memory before the next. The helper never imports SpotMAX — it drives the
external CLI as a subprocess. Stages (`--stages`, comma list, runnable separately
and resumable):

| stage | does |
|-------|------|
| `prepare` | split into batches; per batch write a `Position_*/Images` tree + `id_map.csv` + a `config.ini` derived from your **working GUI-saved INI** (only the `Folder path` line is repointed; everything else preserved) |
| `run` | for each batch lacking output, run `<--spotmax-cmd> -p <batch>/config.ini` (skips done batches) |
| `merge` | parse each batch's tables → per-batch neutral, then **merge** into one neutral CSV |
| `convert` | merged neutral + photon images → canonical predictions (honest `--method` flags) |
| `benchmark` | run the harness with `--method` over the frozen set |

```bash
# 1) prepare batches + per-batch INIs from your working GUI-saved INI
uv run python scripts/run_spotmax_batches.py --stages prepare \
  --benchmark data/benchmark_test_v1 --out external_runs/spotmax/full \
  --template-ini external_runs/spotmax/working_threshold.ini --batch-size 5

# 2) in your SpotMAX env, run each batches/batch_*/config.ini
#    (or let the helper drive it:  --stages run --spotmax-cmd spotmax)

# 3) merge + convert + benchmark with the honest threshold name
uv run python scripts/run_spotmax_batches.py --stages merge,convert,benchmark \
  --benchmark data/benchmark_test_v1 --out external_runs/spotmax/full \
  --method spotmax_threshold_plus_aperture
```

If SpotMAX's INI uses a key other than `Folder path` for the data folder, pass
`--ini-folder-key "<that key>"`. Coordinate-column overrides (`--x-col/--y-col/--p-col`)
flow through to the parser.

All generated artifacts (exported TIFFs, INIs, SpotMAX output, prediction CSVs,
benchmark figures) live under git-ignored paths (`external_runs/`, `SpotMAX_output/`,
`data/**`, `*.tif`).

## Real-run output notes (SpotMAX v1.3.1)

Confirmed from a real Thresholding + `peak_local_max` smoke run:

- The output directory is `spotMAX_output` (lower-`s`) — matched **case-insensitively**.
- Per-spot tables are `1_1_valid_spots_<basename>.csv` / `1_0_detected_spots_<basename>.csv`;
  there are also `*_aggregated.csv` siblings (one row per segmented object) which
  are **excluded** (wrong granularity).
- Tables carry **both** global `x,y` and object-local `x_local,y_local`. The
  parser prefers the **global** coordinate (it aligns with the whole photon image;
  local would be bbox-relative under segmentation).
- There is no higher-is-better confidence column (only `*_pvalue` / `*_effect_size`),
  so `p_detect` is left `NaN`.

## Tests

`tests/test_spotmax_adapter.py` runs without SpotMAX installed: it fabricates
SpotMAX-style output tables + tiny photon images and checks the parse → neutral →
canonical chain, the `x=column / y=row` convention (incl. global-over-local
preference and the real lowercase-dir / `_aggregated` layout), the non-positive
policy, the two honest method names + flags (AI vs threshold), merged neutral
detections, the INI folder-path rewrite, and that importing `spotpipe` never
imports `spotmax`. An optional real-CLI smoke is skipped when `spotmax` is not on PATH.
