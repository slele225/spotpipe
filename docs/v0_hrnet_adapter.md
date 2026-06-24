# `v0_hrnet` — legacy HRNet detector (model-native intensities)

A benchmark method that wraps an **old, externally-trained HRNet** as an honest
historical baseline. Unlike the `*_plus_aperture` methods (CMEAnalysis / Spotiflow
/ SpotMAX, which are detector-only and get intensities from in-repo aperture
photometry), the legacy HRNet **predicts intensities AND per-spot uncertainties
directly**, so this method uses those predictions as-is — there is **no aperture
step and no photon image is read**. Hence the name `v0_hrnet`, not
`v0_hrnet_plus_aperture`.

It is opt-in (not in the default `methods:` list); enable with `--methods v0_hrnet`.

## Honest naming: why `v0`

The model is archived in a **separate** repo
(`liposome-detect/models/hrnet_v1`). It is an OLD model from a long time ago,
trained on a **different forward model** with **different channel semantics**
(lipid / protein) and **different normalization** — *not* on spotpipe's simulator.
We label it `v0` to make that legacy/cross-domain status explicit (the source repo
calls it `hrnet_v1`; here it is the *zeroth*, pre-spotpipe model).

## Architecture / boundaries

- **The legacy model and its checkpoint are never vendored** into spotpipe. The
  54 MB `best.pt` stays in the other repo (`*.pt` is gitignored repo-wide anyway)
  and is referenced only by path.
- **Importing `spotpipe` never imports torch / timm or the legacy repo.** The
  in-repo adapter module (`spotpipe.benchmark.v0_hrnet`) depends only on the
  canonical predictions CSV below. The one place that touches torch / timm and the
  legacy code is `scripts/run_v0_hrnet_predict.py`, run in a **separate
  environment**.
- **The only spotpipe ↔ legacy-model interface is a canonical predictions CSV**
  (exactly `spotpipe.schema.SCHEMA_COLUMNS`). The harness adapter
  (`V0HrnetAdapter`) loads it, validates it, subsets it to the eval-set image ids,
  and fabricates nothing (an image with no predicted rows contributes no rows).

## Units caveat (carried in every row's `flags` as `units=legacy_flux`)

The legacy `I1` / `I2` are in the **old simulator's flux units**, not spotpipe
photon-proportional units. So:

- **Absolute-intensity bias is expected to be large / meaningless.** (In a 3-image
  smoke run the log-ratio bias was ~4.8 — by design, not a bug.)
- **Detection metrics and the log-ratio SLOPE (β) stay meaningful**: a constant
  per-channel scale is an additive constant in log space, so it shifts the ratio
  intercept, not the slope.

The simulator's `audit/` true background and the photon/true-background files are
never read; this method only *runs* a fixed legacy model and emits its native
outputs.

## Channel mapping

spotpipe's channels are generic (ch1 → `logI1`, ch2 → `logI2`); the legacy model's
are lipid / protein. The mapping is explicit and recorded in `flags`
(`ch1=<lipid|protein>;ch2=<…>`):

- Default **`ch1 = lipid`, `ch2 = protein`**, so `log_ratio = logI2 − logI1`
  corresponds to the legacy `log(protein / lipid)` — whose slope is the legacy
  `alpha / 2`.
- The producing script chooses it via `--ch1-channel`; the legacy model's input is
  reordered to its expected `[protein, lipid]` order accordingly.

The legacy log-variance head models the variance of the **log-flux residual**, so
spotpipe `uncertainty_k = exp(0.5 · logvar_k)` (a std in log-intensity). There is
no PSF-width head, so `sigma1_hat` / `sigma2_hat` are `NaN`.

## Canonical predictions CSV contract (primary interface)

A single CSV across all images, with **exactly** the canonical 16 columns:

```
image_id, spot_id, x, y, p_detect, logI1, logI2, I1, I2,
log_ratio, ratio, sigma1_hat, sigma2_hat, uncertainty1, uncertainty2, flags
```

- `x` = sub-pixel **column**, `y` = sub-pixel **row** (spotpipe 0-indexed, from the
  legacy decode, already in full-resolution pixels).
- `p_detect` = the legacy heatmap detection score.
- `logI1` / `logI2` = log of the legacy model-native mean flux for the mapped
  channels; `I1` / `I2` / `log_ratio` / `ratio` are derived consistently.
- `sigma1_hat` / `sigma2_hat` = `NaN`; `uncertainty1` / `uncertainty2` as above.

Produce it with `scripts/run_v0_hrnet_predict.py` (see below).

## Config (`configs/benchmark.yaml`, opt-in)

```yaml
benchmark:
  v0_hrnet:
    predictions_csv: outputs/external/v0_hrnet/predictions.csv
    ch1_channel: lipid            # which legacy channel -> spotpipe ch1/logI1
    # Provenance for the producing script (not read by the harness adapter):
    legacy_repo:   C:\Users\shivl\Music\liposome-detect
    legacy_config: C:\Users\shivl\Music\liposome-detect\configs\train\hrnet_v1.yaml
    checkpoint:    C:\Users\shivl\Music\liposome-detect\models\hrnet_v1\best.pt
```

## Running it (two stages)

**1. Run the legacy model (separate torch/timm env) → canonical CSV.** Needs an
environment with `torch`, `timm`, `tifffile`, `numpy`, `pyyaml`, `pandas` and the
legacy repo importable (e.g. that repo's `.venv`, or a dedicated `.venvs/v0_hrnet`;
all gitignored). Detection runs on the **raw** image.

```
# smoke (first 3 images)
python scripts/run_v0_hrnet_predict.py \
  --frozen-dir data/benchmark_test_v1 \
  --legacy-repo   "C:\Users\shivl\Music\liposome-detect" \
  --legacy-config "C:\Users\shivl\Music\liposome-detect\configs\train\hrnet_v1.yaml" \
  --checkpoint    "C:\Users\shivl\Music\liposome-detect\models\hrnet_v1\best.pt" \
  --out outputs/external/v0_hrnet/predictions.csv --limit 3
# full: drop --limit
```

**2. Benchmark it with the harness** (core spotpipe env; no torch needed):

```
python -m spotpipe.benchmark.harness --frozen-dir data/benchmark_test_v1 \
  --methods v0_hrnet --out outputs/external/v0_hrnet/bench_full
```

All generated predictions/results live under `outputs/external/v0_hrnet/`
(gitignored) — never committed.

## Tests

`tests/test_v0_hrnet_adapter.py` covers the in-repo side without torch/timm:
import isolation, the schema/mapping/uncertainty conversion, CSV validation, and
the adapter's subset-and-never-fabricate behavior.
