# Baseline comparison — master reference (2026-07-15)

**Purpose.** One place a fresh chat can use to run each baseline, score it through the ONE shared
blind evaluator, and assemble the cross-method comparison tables. The headline model
(`headfix40k-DELTA`) is already evaluated; its numbers are the reference row every baseline is
compared against. This doc is the contract + the table skeletons + per-baseline status.

**How to use.** For each baseline: (1) run it in its OWN env (never import into the main repo),
(2) emit schema-conformant `predictions.csv` per condition into a method folder, (3) run
`spotpipe evaluate`, (4) drop its row into the tables in §6. The evaluator is tool-agnostic — the
SAME code scores every method, which is the fairness guarantee. No per-tool eval code, ever.

---

## 1. The benchmark (v3 grid) — the test set

Deterministic (`seed 0`); regenerate with `spotpipe bench-gen` into `data/benchmark/`
(gitignored — does NOT travel via git; each machine regenerates or syncs it). The generator
self-cleans stale cells (fixed 2026-07-15). `data/benchmark/BENCH_MANIFEST.json` is authoritative.

* **Family 1 — SNR × density** (detection + intensity vs difficulty): 6 SNR × 7 density = **42
  cells**, 50 images each.
  * `snr_targets: [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]` — TRUE constant-SNR (intensity solved per
    cell). These are 12.7 → 77.6 photons/spot. The live detection range; the v2 grid `[2..15]`
    was a saturated plateau that measured nothing.
  * `density_levels: [0.0006, 0.002, 0.006, 0.012, 0.015, 0.02, 0.025]` spots/px. 0.025 is where
    mean nearest-neighbour distance drops below the ch2 PSF FWHM (spots physically merge). **This
    is where the win over the baselines lives** — sparse fields are at ceiling for everyone.
* **Family 2 — curvature** (α recovery): 13 injected α ∈ [−1.2, 1.2] incl. dense-near-zero, 50
  images each; **α=0 gets 3× images** (the ⭐ null control). Low density, wide A₁ spread.
* Held constant everywhere: PSF σ₁=1.4 / σ₂=1.68 px, background 2 photons, zero registration shift,
  256×256, measured detector (ch1 gain 6.63, ch2 gain 40.0 *chosen*, offset 154, read var 3.1/4.4).
* **Channel mapping (a silent swap destroys α):** ch1 = LIPID (561 nm), ch2 = PROTEIN (488 nm) —
  OPPOSITE the acquisition order.

## 2. The frozen evaluator — how every method is scored identically

`spotpipe evaluate --results <root> --benchmark data/benchmark --out <outdir>`
(implemented in `src/spotpipe/benchmark/evaluate.py`; see `docs/evaluator_convention.md`).

* **Matching:** Hungarian, predicted↔true, gated at **1.0 × max(σ) = 1.68 px** (read from the
  manifest — every tool gets the same gate).
* **α fit:** unweighted OLS of `log(A2/A1)` vs `log(sqrt(A1))` = `logI2 − logI1` on `0.5·logI1`.
  Unweighted ON PURPOSE (size-correlated weights could manufacture curvature). The factor-of-2
  lives in one tested function (`benchmark.evaluate.fit_alpha`); never reimplement it.
* **Detection:** matched → recall + intensity; unmatched **prediction** → FP, **binned
  per-stratum** (so per-cell precision is always defined — a fixed bug from the old repo);
  unmatched GT → FN.
* **Per method it writes:** `<outdir>/<method>/metrics_by_condition.csv` (all 55 conditions:
  recall/precision/f1, logI1/logI2 bias+rmse, log_ratio bias+rmse, α_hat per curvature set) +
  `alpha_recovery.csv` (the 13-point α table) + combined cross-method tables + `summary_by_method.csv`.
* **`--oracle` is a SEPARATE, EXCLUSIVE mode** (scores GT-as-predictions into `oracle_gt/` and
  returns — it does NOT also score methods). ⚠️ This oracle is the TRIVIAL GT passthrough (perfect
  by construction), NOT the informative GT-center + shared-extraction oracle. Building that
  informative oracle (perfect centres, intensity via the shared instrument → still fails α) is an
  OPEN TODO and is the ⭐ killer control.

## 3. Prediction-CSV schema — what every baseline must emit

One `predictions.csv` per condition, mirroring the benchmark tree
(`<method>/{snr_density,curvature}/<condition>/predictions.csv`). Columns (source of truth:
`src/spotpipe/schema/schema.py`):

```
image_id, spot_id, x, y, p_detect, logI1, logI2, I1, I2, log_ratio, ratio,
sigma1_hat, sigma2_hat, uncertainty1, uncertainty2, flags
```

* `x, y` sub-pixel centres (x=col, y=row); `logI1/logI2` = natural-log integrated intensity in
  **photon-proportional** units (convert ADU→photons with the BENCH_MANIFEST gains, NOT the real
  measured gains — the benchmark uses ch2 gain 40.0). `_hat` = estimate.
* `uncertainty1/2` = predicted per-channel log-intensity SD, or **NaN** if the tool has none
  (cmeAnalysis, classical). `sigma*_hat` NaN if not estimated.
* Emit correct **positions** even for native-intensity tools, so the shared-extraction re-read can
  run at those positions (decomposes detection vs intensity error).

## 4. ⭐ REFERENCE ROW — headline model `headfix40k-DELTA` (validated 2026-07-15)

The `(logI1, Δ)` HRNet, 40k steps, v3-covering distribution. See `docs/head_fix_results.md` +
`PROVENANCE.md`. Scored through the evaluator above (peak_threshold 0.3, retuned off-benchmark).

**α recovery (family 2), per injected α:**

| α | recovered | bias | recall | precision |
|---|---|---|---|---|
| −1.2 | −1.080 | +0.120 | 0.99 | 1.00 |
| −0.9 | −0.796 | +0.104 | 0.99 | 1.00 |
| −0.6 | −0.575 | +0.025 | 0.99 | 1.00 |
| −0.3 | −0.284 | +0.016 | 0.99 | 0.99 |
| −0.15 | −0.143 | +0.007 | 0.99 | 0.94 |
| −0.075 | −0.057 | +0.018 | 0.99 | 0.91 |
| **0 (null)** | **+0.024** ± 0.002 | +0.024 | 0.99 | 0.90 |
| +0.075 | +0.104 | +0.029 | 0.99 | 0.89 |
| +0.15 | +0.178 | +0.028 | 0.99 | 0.89 |
| +0.3 | +0.348 | +0.048 | 0.98 | 0.91 |
| +0.6 | +0.682 | +0.082 | 0.95 | 0.94 |
| +0.9 | +0.936 | +0.036 | 0.86 | 0.94 |
| +1.2 | +1.085 | −0.115 | 0.76 | 0.95 |

**α-MAE ≈ 0.050, α=0 null +0.024 ± 0.002.** Old-repo baselines were α-MAE 1.0–2.8 (oracle 1.9) →
**20–50× better.**

**Detection (family 1), recall by SNR (density-averaged — dragged down by the 0.02/0.025 crowding cells):**

| SNR | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| recall | 0.17 | 0.47 | 0.74 | 0.80 | 0.84 | 0.87 |
| precision | 0.91 | 0.94 | 0.97 | 0.99 | 0.96 | 0.87 |

Raw: `results/eval_v3/our_model_headfix40k-DELTA/` (+ `INDEP` control not yet re-evaluated).

## 5. Baseline roster (from `docs/benchmark_baselines.md`)

The claim lives on three axes: **detection**, **unbiased intensity/log-ratio vs size**,
**calibrated uncertainty**. The sharp claim: *even tools that natively match two channels produce a
size-dependent-biased ratio* (their per-channel estimators are biased for small/dim/crowded spots),
so their α is tilted.

| tool | axis | native 2-ch? | env | status |
|---|---|---|---|---|
| **cmeAnalysis** | intensity (head-to-head) | YES (master/slave) | MATLAB | runner handoff exists; rebuild for v3 grid |
| **ComDet** | intensity (head-to-head) | YES | Fiji/ImageJ | not started |
| **DECODE** | ⭐ uncertainty (only one) | no (per-channel) | own venv | not started; takes measured gain/offset/read |
| **Big-FISH (FQ v2)** | intensity | no | pip | not started |
| **Spotiflow** | detection control | no | own venv | not started |
| **classical LoG + aperture** | honest floor | (shared) | main repo | not started |
| **GT-center oracle (informative)** | ⭐ killer control | (shared) | main repo | **evaluator TODO** (not the trivial `--oracle`) |

Run order: cmeAnalysis → ComDet → DECODE → Big-FISH → Spotiflow. Each in its own env; each vendors a
copy of `schema.py`; the main repo only ingests CSVs. Fine-tune learned tools on the vendored
simulator; run classical tools with the matched σ (1.4/1.68).

## 6. COMPARISON TABLES TO FILL (skeletons for the other chat)

Fill one row per method by reading `<method>/summary_by_method.csv` +
`<method>/alpha_recovery.csv`. `headfix40k-DELTA` row is filled from §4.

### 6a. Headline — α recovery
| method | α-MAE | α=0 null | notes |
|---|---|---|---|
| **headfix40k-DELTA (ours)** | **0.050** | **+0.024** | reproducible v3 headline |
| cmeAnalysis-native | | | |
| cmeAnalysis-shared | | | |
| ComDet-native | | | |
| DECODE | | | + uncertainty axis |
| Big-FISH | | | |
| Spotiflow (shared) | | | detection control |
| classical LoG (shared) | | | naive floor |
| GT-center oracle (informative) | | | ⭐ perfect detection, expected to still fail α |

### 6b. Detection — F1 stratified (fill recall/precision/F1 per SNR and per density)
| method | recall@lowSNR | recall@crowded(≥0.012) | precision | F1 |
|---|---|---|---|---|
| headfix40k-DELTA (ours) | see §4 | | | |
| ... | | | | |

### 6c. Intensity / log-ratio bias, stratified by density (the crowding claim)
| method | log_ratio_bias @ sparse | log_ratio_bias @ crowded | log_ratio_rmse @ crowded |
|---|---|---|---|
| headfix40k-DELTA (ours) | | | |
| ... | | | |

⭐ **The money comparison is 6c at high density** — sparse fields are at ceiling for everyone; the
thesis is that our ratio stays unbiased with a tight spread where the classical estimators tilt.

## 7. Open evaluator work before the FINAL headline
1. **≥6 px border margin** (`benchmark_grid_requirements.md` §4) — cmeAnalysis structurally can't
   fit spots within ~6 px of the edge (88% of its misses are there); without the margin our padded
   CNN wins ~5 F1 points on an edge convention. **Evaluator change; apply to all methods. Do FIRST.**
2. **Informative GT-center + shared-extraction oracle** — the `--oracle` flag currently emits only
   the trivial GT passthrough. Build the real one (perfect centres, intensity via the shared
   instrument) — it is the killer control (perfect detection, still fails α).
3. **peak_threshold protocol parity** — ours retuned OFF-benchmark by max-F1 → 0.3
   (`scripts/threshold_retune.py`). Run the SAME protocol for every tunable baseline or the
   comparison is rigged.
