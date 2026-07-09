# Training distribution — recon report

**Scope:** READ-ONLY. Reports the distribution the frozen HRNet checkpoints were
trained on, so benchmark grid zones can be placed with real numbers. Nothing was
modified; the simulator is untouched.

**Sources consulted (in the order CLAUDE.md prescribes):**

1. `configs/default.yaml` — the vendored "real sizes" simulator block. Header
   claims it "reproduces the old repo's `configs/simulator.yaml` (@ 7b9a0b8)".
2. `configs/smoke.yaml` — tiny sizes, same scene block.
3. `src/spotpipe/models/checkpoints/{hrnet_small,hrnet_large}/config.yaml`,
   `manifest.json`, `PROVENANCE.md` — the carried A100 experiment configs.
4. Simulator defaults baked into `src/spotpipe/simulator/forward_model.py`,
   `psf.py`, `noise.py`, `_features.py`.

---

## 0. Provenance caveat — READ FIRST

There is a real gap between what we *can* read and the bytes the checkpoints
were actually trained on. State it plainly rather than paper over it:

| Fact | Value | Source |
|---|---|---|
| Vendored simulator SHA (this repo) | `7b9a0b8` | `VENDORED_NOTES.md` |
| **Training-data simulator SHA** | **`93fc0aa8…-dirty`** | `hrnet_large/manifest.json`, `PROVENANCE.md` |
| Experiment code SHA | `ce697f48…-dirty` | checkpoints' `config.yaml` |
| Scene ranges actually used at train time | **NOT in this repo** | see below |

* The checkpoints' `config.yaml` does **not** embed the scene ranges. It only
  references `simulator_config: configs/simulator.yaml` — and **that file does
  not exist in this repo** (only `smoke.yaml` / `default.yaml` are present).
* So the *only* scene-range numbers we have are in `default.yaml`, which is a
  **reproduction claim** ("reproduces … @ 7b9a0b8"), not a byte-verified copy of
  the training config. The training tree was **dirty** at a *different* SHA
  (`93fc0aa8`), and `PROVENANCE.md` explicitly warns that bit-exact regeneration
  would require reconstructing that dirty tree from the old repo.

**Do `default.yaml` and `smoke.yaml` disagree?** No — their `scene:` blocks are
byte-identical (only `image` size, `n_images`, and model width differ). There is
therefore **no internal disagreement to arbitrate**; the single caveat is that
neither can be cross-checked against the missing `configs/simulator.yaml`.

**Bottom line for grid placement:** treat the `default.yaml` scene ranges below
as the best available proxy for the training distribution, flagged as
*reproduction, not verified identical to the A100 run*.

---

## 1. Training distribution (from `default.yaml` scene block)

All scene axes are drawn **per image** (domain randomization). Detector
constants are fixed per dataset (sampled once from the seed).

| Axis | Range / distribution | Units | Notes |
|---|---|---|---|
| Spot **density** | log-uniform `[0.0006, 0.012]` | spots **per pixel** | `oversample_dense_fraction: 0.3` re-draws 30% of images from the upper octave `[0.00268, 0.012]` |
| Per-spot **intensity A₁** (ch1) | `log10 A₁ = 1.3 + 2.6·u^1.6`, `u~U(0,1)` | log10 photons | range `[10^1.3, 10^3.9] ≈ [20, 7943]`; `dim_bias=1.6` oversamples the dim tail |
| Ratio-law **alpha** (intercept) | uniform `[-0.7, 0.7]` | log units | sets ch2 vs ch1 offset — see §3 |
| Ratio-law **beta** (slope) | uniform `[-0.6, 0.6]` | dimensionless | incl. 0 and negative; **never a training signal** (CLAUDE.md rule 3) |
| Ratio **scatter_std** | uniform `[0.03, 0.25]` | nat-log | per-spot log-ratio noise |
| **PSF sigma1** (ch1) | uniform `[1.0, 1.8]` | pixels | see §4 |
| **PSF sigma2** (ch2) | `sigma1 × mismatch`, mismatch `[1.05, 1.35]` → `[1.05, 2.43]` | pixels | deliberate C1↔C2 width mismatch |
| Registration shift | `±1.0` per channel, per axis | pixels | sub-pixel misregistration |
| Background **level** | uniform `[2.0, 25.0]` | photons | flat pedestal; +gradient/+structure below |
| Background gradient_frac | `[0.0, 0.6]` | frac of level | random-orientation linear tilt |
| Background structure_frac | `[0.0, 0.5]` | frac of level | low-freq smooth field |
| Background structure_scale | `[16, 64]` | pixels | correlation length |
| Clustering | `cluster_prob=0.4` | — | else uniform; clustered → `n_clusters [2,8]`, `cluster_sigma [6,24] px` |

**Spot counts** (density × H×W). Default images are 256×256 = 65 536 px:

* min density `0.0006` → ~**39** spots/image
* max density `0.012` → ~**786** spots/image
* oversampled upper octave → ~**176–786** spots/image

---

## 2. SNR — definition and range

**SNR is not a sampled axis.** The simulator never draws an "SNR"; it draws
intensity, background, and PSF sigma, and SNR is a *derived per-spot* quantity
computed downstream. The exact formula lives in one place —
`src/spotpipe/simulator/_features.py::_channel_snr` / `peak_snr`:

```
peak_k  = A_k / (2π · sigma_k²)                       # own-contribution PEAK (photons)
read_k  = noise_floor_sigma_k / gain_k                # read noise in photon-equiv units
noise_k = sqrt( ((peak_k + B_k) + read_k²) / n_frames )
snr_k   = peak_k / noise_k
snr     = min(snr1, snr2)                             # the LIMITING channel
```

Key properties:

* It is a **peak** (not integrated) photon-domain SNR. `A_k` is the total
  integrated intensity; the peak pixel is `A_k/(2πσ_k²)`.
* Background `B_k` is the **flat `level` only** (gradient/structure ignored for
  this scalar — it's a binning axis, not a calibration).
* `n_frames = 3` divides the added-noise variance (frame averaging).
* The per-spot scalar is `min(snr1, snr2)`: the ratio I2/I1 can be no better
  measured than its worse channel.
* Detector constants used: ch1 `gain 2.5, floor 8.0`; ch2 `gain 6.0, floor 11.0`
  ⇒ read1 ≈ 3.2, read2 ≈ 1.83 photon-equiv.

**Illustrative range** (σ=1.4 ⇒ 2πσ²≈12.3, ch1):

| A₁ | peak | B=2 | B=25 |
|---|---|---|---|
| dim `10^1.3 ≈ 20` | 1.6 | snr ≈ 0.76 | snr ≈ 0.47 |
| bright `10^3.9 ≈ 7943` | 646 | snr ≈ 45 | snr ≈ 43 |

So training SNR spans roughly **<1 (dim × high-bg) to ~50+ (bright)**. This
matches the **benchmark SNR bin edges** the checkpoints' `config.yaml` declares:
`[0, 2, 5, 10, 20, 50, ∞]`. Use those edges for grid SNR zones.

---

## 3. Alpha / curvature test — CRITICAL FINDING

**Question:** does the simulator support imposing a protein-density-vs-radius
law `ρ ∝ r^(−alpha)` with a *specified* alpha?

**Answer: No — not as posed, because the simulator has no radius variable at
all** (see §5). What it *does* have is a **ratio law on intensities**, and the
word "alpha" in the simulator means a different thing than the "alpha" in
`ρ ∝ r^(−alpha)`. Do not conflate them.

### What the simulator actually sets for per-spot ch2 intensity A₂

Per image it draws `(alpha, beta, scatter_std)`; per spot it draws `A₁`, then
(`forward_model.py:264-270`):

```
log A₂ = log A₁ + alpha + beta·log A₁ + Normal(0, scatter_std)
       = (1 + beta)·log A₁ + alpha + Normal(0, scatter_std)
```

* **alpha** = the *intercept* of `log A₂` vs `log A₁` (a channel offset), NOT a
  radius power-law exponent.
* **beta** = the *slope* of `log(A₂/A₁)` vs `log A₁`. Regressing `log A₂` on
  `log A₁` gives slope `1+beta`.
* **PI-convention note** (`VENDORED_NOTES.md`): downstream biological plots use
  x-axis `log(√A₁) = 0.5·log A₁`, against which the reported slope is
  `beta_pi = 2·beta`. Pure change of x-variable; generated intensities unchanged.

### Can you pin a specific value?

Yes for the ratio-law parameters, and `benchmark_set.py` already does it: to fix
alpha, set `ratio_law.alpha: {min: X, max: X}`; to fix beta, set
`ratio_law.beta: {min: X, max: X}` (it pins `min==max` per image and records the
value in `beta_per_image.csv`). So a **known-slope (beta) benchmark on A₁ is
directly supported today**.

### What is NOT supported (what we'd need to add downstream)

A genuine `ρ ∝ r^(−alpha)` curvature benchmark needs a *radius* axis and a
mapping from radius to both channel intensities. The simulator provides neither:

* No radius is sampled; `A₁` is drawn **directly** from the log10 distribution
  (`_sample_intensities`), not derived from any size.
* To build a known-alpha-vs-radius set downstream (without touching the frozen
  simulator) we would need to, per spot: (1) draw a radius `r`; (2) map `r → A₁`
  (e.g. lipid signal ∝ surface area ∝ `r²`); (3) impose protein density
  `ρ ∝ r^(−alpha)` and set `A₂ = ρ · (lipid amount)` ∝ `r^(2−alpha)`; then feed
  the resulting `(A₁, A₂)` pair in. That radius→intensity model is **not** in the
  simulator and must be authored in the benchmark layer.

**Summary:** the ratio-law `alpha`/`beta` let you pin a log-linear `A₂`–`A₁`
relationship per image, but they are intensity-space parameters, not a
radius power law. The curvature/`ρ ∝ r^(−alpha)` test as framed requires a new
downstream radius→(A₁, A₂) generator.

---

## 4. PSF sigma (for the intensity-extraction σ input + aperture geometry)

**There is no single fixed sigma — it is randomized per image**, and the true
value per image is stored in `meta['ground_truth_sigma']`.

| Quantity | Value |
|---|---|
| sigma1 (ch1) | uniform `[1.0, 1.8]` px |
| sigma2 (ch2) | `sigma1 × [1.05, 1.35]` → `[1.05, 2.43]` px |
| representative midpoint | sigma1 ≈ **1.4 px** |
| `_features.py` fallback (meta missing) | sigma1 = 1.3, sigma2 = 1.5 |
| PSF model | isotropic 2-D Gaussian, **area-normalized** (sums to A), erf-integrated per pixel |

Benchmark stress profiles (`benchmark_set.py`) push sigma1 to `[1.0, 1.15]`
(narrow) or `[1.5, 1.9]` (wide) for the extreme SNR bins.

**For the intensity-extraction module:** it already takes sigma per call
(`extract_channel(image, xs, ys, sigma1, …)`) and derives geometry from it —
aperture radius `3σ`, background annulus inner `4σ`, outer `6σ`
(`intensity_extraction.py:78-80`). Feed it the **per-image ground-truth sigma**
from meta rather than a constant; if a single default is needed, use **σ ≈ 1.4**
(ch1) / **σ ≈ 1.7** (ch2, at mid mismatch).

---

## 5. "Liposome radius / diameter" — DOES NOT EXIST in the simulator

The recon asked how radius is parameterized and how it maps to A₁. It doesn't:

* **No radius/diameter parameter exists anywhere** in the simulator (grep for
  `radius|liposome|diameter` finds only PSF-render window radii and
  aperture/annulus radii in the *extraction* code — nothing generative).
* Every spot is a **point source** rendered as an area-normalized Gaussian whose
  width is the **PSF sigma** (an optical constant per image), independent of the
  spot's intensity. A brighter spot is not a bigger spot — it is a taller
  Gaussian of the same width.
* A spot's **A₁ is set directly** by `_sample_intensities`
  (`log10 A₁ = 1.3 + 2.6·u^1.6`), continuously, with **no radius in the loop**.
* Consequence for the alpha/curvature work: any notion of "liposome radius" and
  its mapping to lipid intensity A₁ (and thus to a `ρ ∝ r^(−alpha)` protein law)
  is **not present** and would have to be added in the downstream benchmark
  generator (see §3).

---

## 6. Numbers to carry into grid-zone placement

* **SNR zones:** bin edges `[0, 2, 5, 10, 20, 50, ∞]` (checkpoint config);
  training spans ~<1 to ~50+. SNR = `min(snr1,snr2)`, peak-domain, formula in §2.
* **Density (local) zones:** `n_neighbors` within `density_radius_px = 4.0`;
  bin edges `[0, 1, 3, 6, ∞]` (checkpoint config). Image-level density is
  log-uniform `[0.0006, 0.012]` spots/px (~39–786 spots/256² image).
* **Intensity:** `log10 A₁ ∈ [1.3, 3.9]`, dim-biased (`u^1.6`).
* **Ratio law:** alpha `[-0.7, 0.7]`, beta `[-0.6, 0.6]` (report `beta_pi=2β`);
  to pin, set `{min:X, max:X}`.
* **PSF sigma:** sigma1 `[1.0, 1.8]` (mid 1.4), sigma2 = ×`[1.05, 1.35]`.
* **Detector (fixed):** ch1 gain 2.5 / offset 100 / knee 2600 / floor 8;
  ch2 gain 6.0 / offset 150 / knee 4200 / floor 11; n_frames 3; adc_max 4095.
* **Caveat:** ranges are `default.yaml`'s reproduction of the training config,
  not byte-verified against the (missing) `configs/simulator.yaml` used by the
  `93fc0aa8-dirty` A100 run.
</content>
</invoke>
