# Evaluator convention — CANONICAL DEFINITION (freeze this)

> Definitional spec, referenced from CLAUDE.md in summer_research_26. It fixes how the ONE shared,
> blind evaluator turns schema-conformant prediction CSVs + benchmark ground truth into the
> project's RESULTS: detection metrics, per-spot log-ratio bias/spread, and the recovered curvature
> slope `alpha`. Like `alpha_convention.md` and `snr_convention.md`, this must survive across chats
> and prompts, because a future prompt silently changing the matcher or the alpha fit would make
> every cross-method comparison an artifact of the evaluation code rather than the methods.
> Governs the DISPOSABLE benchmark layer only; touches nothing frozen (simulator / model / losses /
> schema). Implemented in `src/spotpipe/benchmark/evaluate.py`.

## ⭐ The fairness guarantee (why this is built exactly once)

**ONE shared, blind evaluator for EVERY tool.** It ingests schema-conformant CSVs
(`spotpipe.schema.SCHEMA_COLUMNS`) and does **not** know or care which tool produced them — our
model, SpotMAX, Spotiflow, cmeanalysis, or a classical baseline are scored by the *same code path*.
There is **no per-tool evaluation code** and there is **no `if method == ...`** anywhere. If our
model were matched or fit with even slightly different logic than a baseline, any performance
difference could be an artifact of the evaluator, and the whole comparison would be contestable.
This is non-negotiable.

## Input contract

A results root with one folder per method, each mirroring the benchmark tree (exactly what
`spotpipe infer` emits; baselines conform to the same contract):

```
<results_root>/<method_name>/
  snr_density/snr={S}_density={D}/predictions.csv
  curvature/alpha={A}/predictions.csv
```

Ground truth lives under the benchmark root (`data/benchmark/` by default), one schema CSV per
image under each condition's `ground_truth/`, enumerated by that condition's `meta.json`. The
evaluator ingests folders of CSVs and **invokes no tool**.

**Every benchmark image belongs to exactly one condition** (one SNR×density cell, or one curvature
set). The condition IS the stratum. This is what makes per-cell precision well-defined (below).

## Matching (predicted ↔ ground truth)

* **Hungarian (optimal bipartite) assignment**, minimising total matched distance, then dropping
  any assigned pair beyond the gate. Implemented via `scipy.optimize.linear_sum_assignment` in
  `spotpipe.benchmark.matching` (`method="hungarian"`). **The benchmark evaluator is ALWAYS
  Hungarian** — never greedy. (`matching.py` also carries a `greedy` mode used elsewhere for
  training-time validation; the evaluator does not use it.)
* Spots are paired **only within the same `image_id`**, by their `x`/`y` columns (Euclidean px).
* **Distance gate = `match_radius_sigma × sigma_ref`**, `match_radius_sigma = 1.0` (≈1 PSF sigma).
  `sigma_ref = max(sigma1, sigma2)` — the **coarser** channel sets the localization scale, so the
  gate is the lenient "within one PSF" cutoff and never penalises a method for the harder-to-
  localize channel. `sigma1`/`sigma2` are read from `BENCH_MANIFEST.json`
  (`benchmark_constants.sigma1_px` / `sigma2_px`) — **constant across benchmark v2**
  (`sigma1 = 1.4`, `sigma2 = 1.68` → gate = 1.68 px) — and are **never hardcoded**. Both
  `match_radius_sigma` and the choice of `sigma_ref` are configurable, but the default is frozen
  here so labelled results are comparable across runs. (At the densest cell, 0.015 spots/px, the
  expected nearest-neighbour spacing ≈ 4 px ≫ the 1.68 px gate, so the gate never causes cross-
  matches.)

## Three disjoint outcome classes (this is where the OLD repo had a BUG)

Every predicted and every ground-truth spot lands in exactly one class:

* **matched** (a pred↔GT pair within the gate) → true positive. Feeds recall, precision, the
  per-channel intensity metrics, the log-ratio metrics, and the alpha fit.
* **unmatched GROUND TRUTH** → false negative → feeds recall.
* **unmatched PREDICTION** → false positive → feeds precision.

⚠️ **THE BUG WE FIX.** In the old repo an unmatched prediction had no GT spot to inherit an
SNR/density bin from, so per-bin precision came out undefined (`--`). **Fix: bin unmatched
predictions by the CONDITION (cell / set) they occur in**, never by a matched GT's properties.
Because each condition is homogeneous and its `predictions.csv` contains only that condition's
spots, **every** unmatched prediction already belongs to a defined stratum. Per-cell precision is
therefore `n_matched / (n_matched + n_fp)` and is **always defined for any cell that has
predictions** — the evaluator **asserts** this (`precision` is a number, never NaN/`--`, whenever
`n_pred > 0`). A cell with zero predictions has precision `NaN` by definition (0/0); that is
reported honestly, and the assertion does not apply to it.

## Pooling

Metrics **pool spots per condition** (all matched spots across all images of the cell/set), **not
per-image-then-averaged** — images have different spot counts and must not get equal weight.

## Metrics reported

**Per condition** (each SNR×density cell, and each curvature set):

* **Detection:** `recall = TP/(TP+FN)`, `precision = TP/(TP+FP)` (FPs binned per-cell, per above),
  `f1 = 2PR/(P+R)`.
* **Per-channel intensity SIGNED bias:** `mean(logI_pred − logI_true)` for each channel
  (`logI1`, `logI2`), over matched spots. **Signed on purpose** — unbiasedness is the whole claim;
  RMSE alone would hide a systematic offset. RMSE is also reported alongside.
* **Log-ratio:** `bias`, `std`, `RMSE` of `log(A2/A1)_pred − log(A2/A1)_true` over matched spots,
  where `log(A2/A1) = logI2 − logI1`. This is the headline per-spot number.

**Per curvature set:** recovered `alpha_hat`, its standard error `alpha_se`, `true_alpha`,
`alpha_bias = alpha_hat − true_alpha`. A per-method **`alpha_mae = mean(|alpha_hat − true_alpha|)`**
across sets is reported in the summary.

## Alpha fit (follow `docs/alpha_convention.md` exactly)

* `alpha ≡` slope of `log(A2/A1)` vs **`log(sqrt(A1))`**. Plain **UNWEIGHTED** ordinary least
  squares.
* Concretely, over the pooled matched spots of a curvature set, regress
  `y = logI2 − logI1` on `x = 0.5 · logI1` (because `log(sqrt(A1)) = 0.5 · log(A1)`, any log base;
  numerator and denominator share the base, so `alpha` is base-independent). `alpha_hat` is the
  fitted slope. **A2/A1 are the method's own PREDICTED intensities** of the matched spots — the
  recovered alpha is the method's *own* slope, exactly what a real analysis has access to.
* ⚠️ **The factor of 2 (`log(sqrt(A1))`, not `log(A1)`) is load-bearing.** Regressing on `log(A1)`
  instead would make every recovered alpha off by exactly 2× — a plausible-looking, catastrophic
  error. The x-axis uses `0.5 · logI1`; a unit test asserts that switching to `logI1` halves the
  slope (Gate C).
* ⚠️ **UNWEIGHTED ON PURPOSE.** Weighting by anything size-correlated (`A1`, radius, predicted
  uncertainty) could **bias** the recovered slope — the same "manufactured curvature" risk that is
  why there is no alpha term in the training loss (CLAUDE.md rule 3). Equal weight per matched
  spot. Do **not** weight the fit.
* **Error bar:** the analytic OLS regression standard error of the slope,
  `SE(b) = sqrt( (Σ resid² / (n−2)) / Σ(x − x̄)² )`. One number. (Bootstrap was considered and
  deliberately rejected as unnecessary machinery.)
* Report **one alpha per curvature set**, against that set's known injected `true_alpha` from
  `BENCH_MANIFEST.json`.

## Robustness

* Reads `sigma` and detector constants from `BENCH_MANIFEST.json`, never hardcoded.
* **Handles missing / failed cells gracefully:** a condition whose `predictions.csv` is absent or
  unreadable is reported with `status="missing"` (its detection counts are recorded against the GT
  so recall is still defined) rather than crashing the run.
* Tool-agnostic and vectorised where it matters (the matcher runs over ~741k spots).

## Output

Tidy CSVs, easy to build comparison tables from:

* `results/<method>/metrics_by_condition.csv` — one row per condition.
* `results/<method>/alpha_recovery.csv` — one row per curvature set.
* `results/combined_metrics_by_condition.csv`, `results/combined_alpha_recovery.csv` — all methods.
* `results/summary_by_method.csv` — per-method aggregates (mean F1, alpha MAE, null-control alpha,
  mean log-ratio RMSE, …).

## Validation gates (an instrument must be calibrated before use)

The evaluator is validated on GROUND TRUTH before it is trusted on any real method output:

* **Gate A — identity / oracle.** Feed GT as if it were predictions (GT vs GT):
  `recall = precision = f1 = 1.0`, intensity bias = 0, ratio bias = 0 (all to float tolerance), and
  recovered `alpha_hat` equals each set's `true_alpha` to within its OLS standard error. (GT carries
  real biological log-ratio scatter (~0.08), so GT-vs-GT recovers the injected slope in expectation
  and to a few ×10⁻³ in practice — not to bit-float — which the gate checks against `alpha_se`, and
  the alpha=0 null control recovers ≈ 0.)
* **Gate B — known-alpha recovery.** Across all 13 curvature sets (injected alpha in [−1.2, 1.2]
  plus the alpha=0 null control at 3× images), confirm recovered ≈ true across the range.
  ⚠️ On the **alpha=0 null control** it must recover `alpha ≈ 0`: a method reporting `alpha ≠ 0`
  there is manufacturing curvature from size-dependent intensity bias. This is the single most
  important benchmark set, and the test applies to the evaluator itself and to any method.
* **Gate C — factor of 2.** The alpha fit uses `log(sqrt(A1))` on the x-axis, asserted by a test
  that switching to `log(A1)` halves the recovered slope.
* **Gate D — precision defined.** For any cell containing predictions, per-cell precision is a
  number, never `--`/NaN (the old repo's bug).

## Rules

* ONE blind evaluator for all methods; no per-tool code; no `if method == ...`.
* Matching is **Hungarian**, gate `1.0 × max(sigma1, sigma2)` read from the manifest.
* The alpha fit is **unweighted** OLS on `log(sqrt(A1))`; error bar is the analytic OLS slope SE.
* False positives are binned by **condition**, never by a matched GT's properties.
* Do not substitute a different matcher, gate, or fit without updating this doc — a silent change
  invalidates cross-method comparisons.
