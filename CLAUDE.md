# summer_research_26 — the `spotpipe` package (THE NEW REPO)

> ⚠️ **REPO IDENTITY — read this first.**
> **THIS repo is `C:\Users\shivl\Music\summer_research_26`, branch `main`.**
> The OLD repo is `C:\Users\shivl\Videos\spotpipe`, branch `master` — it is a **READ-ONLY
> reference**. **NEVER modify anything under `Videos\spotpipe`.**
> The Python *package* inside this repo is also named `spotpipe` (it was vendored from the old
> repo), which makes the two easy to confuse. **Fastest discriminator: branch `main` = new repo,
> branch `master` = old repo.**
> If you find yourself editing files under `Videos\spotpipe`, **STOP immediately and say so.**

Two-channel microscopy spot-detection pipeline. Rebuilt from the old repo by vendoring ONLY the
precious core; everything else is rebuilt fresh here.

---

## The precious-core principle

Three tiers, hardened by how expensive they are to reproduce:

* **PRECIOUS** (cannot cheaply reproduce): the simulator / training-data generator
  (`src/spotpipe/simulator/`), the model (`src/spotpipe/models/`), the losses
  (`src/spotpipe/losses/`). Trained checkpoints (`src/spotpipe/models/checkpoints/`) are
  precious. Vendored UNCHANGED from the old repo and pinned by git SHA (see `VENDORED_NOTES.md`).
* **FROZEN INTERFACE**: the schema (`src/spotpipe/schema/schema.py`) — the column format of
  prediction and ground-truth files. This is what lets every other piece be swapped freely.
* **DISPOSABLE** (cheap to rewrite): benchmark generator, evaluator, metrics, plotting, baseline
  adapters. The old harness was deliberately NOT ported.

---

## Durable rules

1. **Simulator / models / losses / schema are VENDORED + FROZEN.** Do not modify their logic
   without an explicit instruction naming the file. Anything that looks buggy goes in
   `VENDORED_NOTES.md`; the code stays untouched.
2. **The schema is frozen.** Never add, rename, or remove fields in `spotpipe.schema`.
   Suffix convention: `_hat` = estimate.
3. **NO slope/alpha/beta loss exists or may be added.** The curvature slope is computed ONLY in
   the downstream analysis/benchmark layer, never as a training signal. `losses/ratio.py` is a
   deliberate stub documenting this; it must never become a loss.
   *Why:* an in-batch slope is a biased estimator, and size-correlated weighting could
   **manufacture** curvature sensing that isn't real. The model must never see alpha.
4. **All paths come from `spotpipe.paths`.** NO hardcoded absolute paths anywhere — dev is
   Windows local, training is Linux remote; a `C:\` string will break the remote. Paths root at
   env var `SPOTPIPE_ROOT` (default: repo root).
5. **Every compute-adding prompt ends with a timing/verification report and a golden/sanity
   test.** No long runs before the smoke config passes.
6. **Cached datasets and benchmarks are portable directory artifacts** moved by
   `scripts/sync_to_remote.sh` — never referenced by machine-specific path.
7. **Model checkpoints record the simulator SHA + config + seeds of their training data.**
   See `src/spotpipe/models/checkpoints/*/PROVENANCE.md`.
8. **Ambiguous scope, or a change touching a frozen module → STOP and ask.** Do not guess.

---

## Frozen conventions (violating these silently produces a plausible but WRONG result)

* **Alpha / slope** — `docs/alpha_convention.md`.
  `alpha` ≡ the slope of `log(A2/A1)` vs `log(sqrt(A1))`. ALWAYS. Never vs `log(A1)`.
  The simulator's internal slope (`sim_log_slope`) is vs `log(A1)`, so **alpha = 2 ·
  sim_log_slope**; the factor of 2 lives in exactly ONE tested function.
  The simulator ALSO has a field literally named `alpha` which is an **intercept**, NOT ours —
  so in code, **never use bare `alpha`/`beta` for simulator-space values** (use `sim_intercept`,
  `sim_log_slope`).
* **SNR** — `docs/snr_convention.md`. Follow it for all benchmark difficulty-axis labeling and any
  SNR computation. Never substitute a different SNR definition without updating that doc.
* **Evaluator** — `docs/evaluator_convention.md` *(FROZEN; implemented in
  `src/spotpipe/benchmark/evaluate.py`, CLI `spotpipe evaluate`)*.
  Hungarian matching within ~1 PSF sigma (`1.0 × max(sigma1, sigma2)`, read from
  `BENCH_MANIFEST.json`); **unweighted** OLS fit of `log(A2/A1)` vs `log(sqrt(A1))` for alpha
  (unweighted ON PURPOSE — size-correlated weighting could bias the slope); analytic regression
  standard error. Unmatched **predictions** are binned per-condition (the condition IS the stratum)
  so per-cell precision is always defined (this was a BUG in the old repo).
  **ONE shared, blind evaluator for EVERY tool** — no per-tool evaluation code, no `if method ==`.
  This is the fairness guarantee. Validated on ground truth (Gates A–D) before any real method.

**Channel mapping (a silent swap here inverts A2/A1 and destroys alpha):**
pipeline **channel 1 = LIPID (561 nm)**, **channel 2 = PROTEIN (488 nm)** — note this is the
OPPOSITE of the acquisition order.

**Measured detector parameters** (2026-07-13; gain from photon-transfer curve, offset + read noise
from dark frames):

| pipeline channel | dye / laser | gain (ADU/photon) | offset (ADU) | read variance (ADU²) | benchmark PSF sigma |
|---|---|---|---|---|---|
| ch1 = LIPID   | 561 nm | 6.63  | 154 | 3.1 | 1.4 px |
| ch2 = PROTEIN | 488 nm | 124.3 | 154 | 4.4 | 1.68 px |

Gain is the **variance-matching** gain (already includes PMT excess noise) — do NOT substitute the
single-photoelectron comb spacing (~37 ADU/PE). Gains differ ~19x (PMT HV 500 V vs 750 V) so
**per-channel gain is required**. Noise model:
`ADU = Poisson((spots + background) * gain) + Normal(offset, read_var)` — **background is in
PHOTONS, offset is in ADU**; do not conflate them.

**Known real-instrument feature not yet modeled:** the 488/protein PMT (750 V) produces dark-count
spikes — ~0.57% of pixels/frame carry a spurious single-photoelectron event (~one gain step above
offset). These look like dim single-pixel "spots" in the protein channel. 561 shows none.

---

## Layout

```
configs/            smoke.yaml, default.yaml, benchmark.yaml, benchmark_smoke.yaml
docs/               alpha_convention.md, snr_convention.md, (evaluator_convention.md TBD),
                    PROJECT_STATE.md
src/spotpipe/
  paths.py          ALL path resolution (SPOTPIPE_ROOT)
  config.py         typed config loader (dataclasses over yaml)
  schema/           FROZEN interface (schema.py is the source of truth)
  simulator/        VENDORED forward model + dataset/benchmark-set generators
  models/           VENDORED HRNet backbone + heads + inference (+ checkpoints/)
  losses/           VENDORED detection/localization/intensity losses (ratio.py = deliberate stub)
  benchmark/        BUILT: alpha.py (the frozen 2x conversion), generate.py (two-family
                    benchmark generator), intensity_extraction.py (shared measurement
                    instrument), infer.py (model-inference adapter / template runner)
                    TODO: the shared evaluator
  data/             (empty) dataloader — later build stage
  cli.py            `spotpipe smoke`, `spotpipe bench-gen`, `spotpipe infer`
tests/              test_import, test_schema_roundtrip, test_smoke,
                    test_benchmark_generation, test_infer, test_intensity_extraction
scripts/sync_to_remote.sh   rsync repo + named dataset dir to a remote host
```

**Current state and roadmap: see `docs/PROJECT_STATE.md`.**

---

## Environment

* Windows dev: `.venv` (Python 3.12, torch CPU). Tests: `.venv\Scripts\python.exe -m pytest tests/`.
* Old repo (**READ-ONLY, do not modify**): `C:\Users\shivl\Videos\spotpipe` @ `7b9a0b8` (branch `master`).
* Old A100 artifacts: `C:\Users\shivl\Videos\spotpipe_a100_artifacts`.
* GPU instance (Linux) for the real inference/training runs — verify `torch.cuda.is_available()`
  is True and fail loud if it silently falls back to CPU.