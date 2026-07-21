# Presentation facts — HRNet-DELTA, all key numbers (single source for slides/LaTeX)

Consolidates results that were previously only in a working session. Every number here is
either read from a file (path given) or computed from files in `results/eval_v3_r2/`. A
fresh chat building slides/LaTeX should treat THIS as the authority and not re-derive.

---

## 0. One-paragraph summary
We built a two-channel HRNet spot detector with a `(logI1, Δ)` intensity head that predicts
the protein/lipid log-ratio directly (Δ = logI2 − logI1), plus a calibrated-uncertainty
head. On a controlled 55-condition simulated benchmark it beats the classical standard
(cmeAnalysis) on detection recall (+0.18 F1), on curvature-exponent accuracy (α-MAE 0.050
vs 0.072), and on per-spot ratio precision (~2.8× tighter). On real NTA-LUV data it
reproduces expected biology (endophilin, a BAR-domain protein, shows positive curvature
sensing). Remaining gaps: the uncertainty is under-confident (fixable by recalibration),
α accuracy degrades at the edge of the trained range, and a fair-comparison border margin is
not yet applied.

## 1. The model
- **Architecture:** HRNet-large, 3 branches, base_channels 32, ~1.53M params. Full-res,
  fully-convolutional (input H,W divisible by 4).
- **Four heads:** heatmap (detection), offset (sub-pixel), intensity, log-variance.
- **Intensity head parameterisation = `delta`:** predicts `logI1` and `Δ = logI2 − logI1`;
  derives `logI2 = logI1 + Δ` and `logvar2 = logaddexp(logvar1, logvar_delta)`. State-dict
  identical to the independent head; the schema (`logI1`,`logI2`,`log_ratio`) is unchanged.
- **Loss:** `L = heatmap + offset + (NLL_logI1 + NLL_Δ)`, all weights 1.0.
  - heatmap: CenterNet penalty-reduced focal (α=2, β=4) on a Gaussian-blob target.
  - offset: smooth-L1, masked to spot centres.
  - intensity: heteroscedastic Gaussian NLL `½[exp(−s)(pred−true)² + s]`, s=logvar∈[−10,6].
  - **logvar = predicted variance of the log-INTENSITY estimate** (per channel / on Δ).
    Full derivation: `docs/loss_and_outputs_brief.md`.
- **Provenance:** git `bd9b2a2`, sim SHA `7b9a0b8`, seed 0, 40k steps, best step 23,000,
  density max 0.030, peak_threshold 0.3 (retuned OFF-benchmark).
  `src/spotpipe/models/checkpoints/headfix40k-DELTA/PROVENANCE.md`.

## 2. Why the delta head (the investigation, one line each)
1. Coverage hypothesis (bright+dense unreachable) — **REFUTED** (`coverage_probe_findings.md`).
2. Rarity hypothesis (dim-biased sampler) — **REFUTED** (`rarity_probe_findings.md`); tell:
   improving both channels made the RATIO worse → per-channel accuracy ≠ ratio accuracy.
3. Mechanism found: conditional-mean **shrinkage** localised to the wider-PSF protein channel
   under crowding (dense-ch2 slope 0.70 vs ~1.0 elsewhere); unequal shrinkage s1−s2 lands on
   the ratio (`shrinkage_probe_findings.md`).
4. Fix: predict Δ directly so the ratio is the model's own estimand → the s1−s2 residual
   cannot exist (`intensity_head_fix_proposal.md`, `head_fix_results.md`).

## 3. Benchmark headline (macro-avg over 55 conditions) — `results/eval_v3_r2/summary_by_method.csv`
cmeAnalysis shown as its strongest config (given-σ, native photometry).

| metric | HRNet-DELTA (ours) | cmeAnalysis | winner |
|---|---|---|---|
| mean F1 | **0.781** | 0.655 | ours (+0.13) |
| mean recall | **0.721** | 0.538 | ours (+0.18) |
| mean precision | 0.939 | **0.978** | cme |
| curvature F1 | **0.948** | 0.854 | ours |
| **α-MAE** | **0.050** | 0.072 | ours |
| α=0 null | +0.024 ± 0.002 | **+0.011 ± 0.005** | cme (both clean) |
| log-ratio RMSE | **0.269** | 0.460 | ours |

## 4. Precision on α (the cleanest single win) — computed from `eval_v3_r2` per-method alpha_recovery
`alpha_rse` = per-spot scatter of log(A2/A1) about the fit (slope-independent);
`alpha_se` = error bar on α itself.

| | HRNet-DELTA | cmeAnalysis | ratio |
|---|---|---|---|
| mean residual SE (RSE) | **0.090** | 0.249 | **2.8× tighter** |
| mean α standard error | **0.0037** | 0.0163 | **4.4× tighter** |

RSE is flat across α for ours (0.077–0.118) but degrades for cme (0.16→0.43). **R² is
confounded with |α| (→0 at the null even for a perfect fit) — compare it only BETWEEN methods
at the SAME α, never across α, and never present the null's R² as a defect.**

## 5. Where α accuracy is weak, and why (computed this session)
- **Error scales with |α|:** corr(|true α|, |error|) = **+0.842**.
  mean |err| for |α|≤0.3 = **0.024**; for |α|≥0.9 = **0.094** (~4× worse).
- **Cause:** the NLL optimum is the posterior mean → shrinks toward the training prior
  (mean α=0). Training ratio-law β ∈ [−0.6,+0.6] → **trained α ∈ [−1.2,+1.2]**, and the
  benchmark's extremes ±1.2 sit EXACTLY on that boundary, where shrinkage is maximal.
- **EIV decomposition** (recovered = 0.946·true + 0.032): a 5.4% multiplicative
  attenuation + a +0.032 additive offset + 0.042 random scatter. cmeAnalysis instead
  AMPLIFIES (slope 1.105) because its channel errors are correlated (σ12 > σ1²); the delta
  head has σ12 = σ1² by construction, so it suffers only clean, correctable attenuation.
- **Deming/TLS is NOT the fix:** attenuation-correcting fixes |α|≥0.9 but worsens α≈0 (net
  ~0.003). The additive offset (manufactured curvature) is the residual problem near zero.
- **How to improve, ranked:** (1) widen training β to ±0.8 so the benchmark sits inside the
  prior — but check the solved A1-window first, since A2=A1^(1+β) shrinks the saturation-safe
  ceiling; (2) reduce σ(logI1)≈0.22, likely via train/infer centre-jitter; (3) attack the
  additive offset; (4) off-benchmark 2-parameter (slope+offset) calibration as a band-aid.
  Do NOT add a slope/α term to the loss (would manufacture curvature).

## 6. Uncertainty calibration (computed this session; curvature family, 40,095 matched spots)
The unique axis — no classical baseline emits per-spot uncertainty at all.
- **std(z) of the log-ratio = 0.51** (z = (pred−true)/pred_sigma; 1.0 = perfect).
  → **~2× UNDER-confident** (error bars ~2× wider than needed). Per-channel: logI1 0.42, logI2 0.39.
- **Coverage:** nominal 68% interval captures **86%**; nominal 50% captures 64% (over-covers).
- **corr(predicted σ, |actual error|) = +0.35** → the head DOES know where it is less certain.
- **Honest claim:** "per-spot uncertainty that tracks error and is conservative" — NOT yet
  "calibrated". Fixable with off-benchmark temperature-scaling of logvar. Likely worse (more
  under-confident) on single-scan real data (training assumed n_frames=3).

## 7. Real data (this session; NOT ground-truth-validated — a measurement, not an eval)
**⚠️ SIGN CONVENTION on real-data figures:** reported α = −(OLS slope), the field-standard
`ρ ∝ r^(−α)` (positive α = positive curvature sensing). The BENCHMARK tables (§3–5) report the
RAW slope. Magnitudes match; signs are opposite — never mix them in one table.

### 7a. His-mEGFP concentration series (`D:\data_for_matlab\2.28.26 images`, 4 conditions × 20 cells)
Reported α (field convention, from `alpha_comparison_summary.csv`):

| condition | cmeAnalysis α | HRNet α |
|---|---|---|
| 20nM EGFP | −0.92 | **+0.23** |
| 50nM EGFP | −0.39 | +0.13 |
| 100nM EGFP | −0.23 | −0.03 |
| 300nM EGFP | −0.07 | −0.07 |

⚠️ **CAVEAT — do not over-claim this series.** The lipid channel median intensity ramps 3.9×
monotonically with EGFP concentration (28→110), while protein stays ~flat. If the conditions
differ only in protein concentration, that lipid ramp is likely EGFP bleed-through into the
561 channel — which would make the α trend partly an artefact. Resolve (bleed-through vs real)
before presenting this as a result.

### 7b. Endophilin (`D:\ethan images\2026\4.14.26`, 300nM, 55 fields) — the clean positive control
BAR-domain protein → expected POSITIVE curvature sensing.
- **HRNet α = +0.83**, cmeAnalysis α = +0.68 (field convention). Both positive → expected biology.
- Channel map: page0=protein(488), page1=lipid(561), page2=unused; script swaps to pipeline
  order ch1=lipid, ch2=protein.
- Caveat: this dataset's protein channel is background-dominated (soluble endophilin), outside
  the training background range — trust absolute intensities less than the EGFP set.

## 8. Known gaps / caveats to state honestly on slides
1. **No ≥6 px border margin applied** in the evaluator (TODO). cmeAnalysis structurally can't
   fit edge spots → favours our padded CNN by ~5 F1 points. The F1 lead (0.781 vs 0.655)
   survives it, but the number will move. Fix before publishing detection.
2. **Sign convention** differs between benchmark tables and real-data figures (see §7).
3. **Uncertainty is under-confident**, not yet calibrated (§6).
4. **α weak at |α|≥0.9** (training-boundary shrinkage, §5).
5. **The `--oracle` scored so far is the trivial GT passthrough** (perfect by construction),
   NOT the informative GT-centre + shared-extraction oracle — that comparison isn't done.
6. **Real-data results are measurements, not validations** (no ground truth).
7. Only cmeAnalysis run as a baseline; DECODE/Spotiflow/ComDet not yet.

## 9. Where the raw numbers live
- Benchmark tables (all 55 cells, every method): `docs/BENCHMARK_RESULTS_v3.md`
- Per-method CSVs (authoritative; the `combined_*` files truncate on some mounts):
  `results/eval_v3_r2/<method>/{metrics_by_condition,alpha_recovery}.csv`
- Loss/head details: `docs/loss_and_outputs_brief.md`
- Decision + 8k→40k history: `docs/head_fix_results.md`
- Baseline plan: `docs/baseline_comparison_masterdoc.md`
- Real-data plot scripts: `scripts/plot_alpha_side_by_side.py`, `scripts/plot_alpha_real_data.py`
- Real-data figure summaries: `alpha_comparison_summary.csv` in each image folder
