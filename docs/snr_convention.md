# SNR convention — CANONICAL DEFINITION (freeze this)

> Definitional spec, referenced from CLAUDE.md in summer_research_26. It fixes the meaning of
> "SNR" for the entire benchmark layer: which SNR quantity is used, and how the SNR difficulty
> axis labels a cell. Like the alpha convention, this must survive across chats and prompts,
> because a future prompt silently substituting a different SNR formula would make difficulty
> axes incomparable between runs. Governs the DISPOSABLE benchmark layer only; touches nothing
> frozen (the vendored simulator keeps its own internal definitions).
>
> This document describes what the code ACTUALLY does. In the v2 benchmark, family-1 cells are
> TRUE constant-SNR: the generator INVERTS the vendored `_features` peak-SNR (below) to solve for
> the single spot intensity that hits each cell's target SNR, and every spot in the cell gets that
> same intensity (no jitter), so the realised per-spot SNR has ~zero spread and equals the target.
> (An earlier v1 spec described a *nominal* SNR bin with a wide per-spot draw; that is no longer
> how family 1 works — see below. The curvature family still keeps a wide A1 spread on purpose.)
> A spec that misdescribes the code is worse than none.

## The SNR quantity we commit to: vendored `_features` PEAK SNR, limiting channel
This is the definition in `src/spotpipe/simulator/_features.py` (`peak_snr`), stated verbatim.
Per spot, per channel `k`, from the spot's total integrated intensity `A_k = exp(logI_k)`
(photon-proportional units):

    peak_k  = A_k / (2*pi*sigma_k^2)              # own-contribution peak (photons)
    read_k  = noise_floor_sigma_k / gain_k        # read noise in photon-equivalent units
    noise_k = sqrt( ((peak_k + B_k) + read_k^2) / n_frames )
    snr_k   = peak_k / noise_k

    A_k        = total integrated signal photons of the spot in channel k (= exp(logI_k))
    sigma_k    = true per-channel PSF width (px), from meta['ground_truth_sigma']
    B_k        = FLAT per-channel background level (photons), meta['scene']['background{k}']['level']
    gain_k     = per-channel detector gain, noise_floor_sigma_k = per-channel read-noise floor
    n_frames   = frame-averaging count (divides the added-noise variance)

Reported per-spot scalar SNR = `min(snr_1, snr_2)`   # limiting (worse) channel

`snr_1` and `snr_2` are also kept for inspection. All quantities come from per-image simulator
metadata only, so the definition applies identically to ground truth and to any method's
predictions (see `_features.attach_features`).

## Conventions retained (these were right, and the code does them)
- **`min()` over channels — limiting channel.** The `A2/A1` ratio needs BOTH channels; the
  worse-measured one governs usability, so a spot is binned by its lower-SNR channel.
- **Background included in the noise.** Measurability genuinely depends on background, so `B_k`
  sits inside the noise term. For this scalar `B_k` is the FLAT `level` only; gradient/structure
  add a little more and are deliberately ignored — this is a binning axis, not a calibration.

## How the SNR axis labels a cell (family 1: SNR x density) — TRUE constant SNR
The SNR x density family solves for a per-spot intensity that hits each cell's target SNR
**exactly** (v2). It is a controlled measurement, not domain randomisation:

- Each SNR cell is a **TRUE constant-SNR target** (`snr_targets`, default `[2, 5, 10, 20, 50]`;
  the old `0` and `inf` bin edges are dropped — they are not solvable). The cell is realised by
  **inverting** the peak-SNR formula above: given the FIXED PSF (`sigma1 = 1.4`, `sigma2 = 1.68`),
  the CONSTANT 2-photon background and the MEASURED per-channel detector, solve for the single
  spot intensity `A` such that `min(snr1(A), snr2(A)) == target`. `min()` over two channels rarely
  has a closed form, so the solve is numerical (bisection in `log A`; the composite SNR is strictly
  increasing in `A`). The factor lives in one function; nothing vendored is touched.
- **Every spot in the cell gets that same intensity** `A`. The ratio law is pinned neutral and
  **zero-scatter** (`sim_intercept = 0`, `sim_log_slope = 0`, `scatter_std = 0`), so `A2 == A1 == A`
  exactly — **no jitter**. Identical intensity → identical peak → identical SNR, so the **realised
  per-spot SNR has ~zero spread and equals the target**. Each cell's `meta.json` records
  `realised_snr` (all quantiles equal the target), `realised_snr_spread` (≈0), and the solved
  intensity; a generation-time assertion fails if the spread is not ~0 or the median ≠ target.
- **Accepted consequence:** family 1 no longer measures intensity-DEPENDENCE within a cell (every
  spot is identical). That job moves entirely to the curvature family (family 2), which keeps the
  wide A1 spread.
- **Out-of-distribution / saturation flags.** A solved `A` outside the legacy checkpoints' `[20,
  7943]` photon range is flagged (a coverage artifact, not a method difference; a measured-detector
  retrain is in progress). The **protein (ch2) gain sets a hard SNR ceiling**: the ch2 peak pixel
  reaches the 12-bit ADC ceiling at a finite SNR, so `snr_targets` are **capped below it** and
  generation **fails loud** if any cell would clip either channel (`ch2_saturates`/
  `protein_channel_saturates`). At the **chosen** protein gain **40 ADU/photon** the ch2 ADC clips
  near **~100 photons peak (SNR ≈ 16.8)**, so the grid `[2, 3, 5, 8, 10, 15]` stays unclipped with
  headroom. ⚠️ **gain 40 is a CHOSEN value for a PLANNED lower-voltage acquisition, NOT a
  measurement** — the measured gain at the current 750 V setting is 124.3 (which clips at ~32
  photons, SNR ≈ 9.5, and forced the benchmark into a single-photon regime). The benchmark simulates
  the settings we will actually use going forward; re-measure the gain at the voltage finally used.
  (ch1/lipid stays at the measured 6.63 — nowhere near its ceiling and never the limiting channel —
  a flagged simplification, since a lower voltage would also lower the lipid gain somewhat.) See
  `BENCH_MANIFEST.json` → `benchmark_constants`.

The **density axis is orthogonal to SNR** and is handled separately: it is a constant AREA
density (spots/px) set at generation, not tied to intensity or SNR (see the benchmark generator
and `docs`/CLAUDE.md). Intensity is solved from SNR alone, never coupled to density. Every density
level is generated at every SNR level (full grid).

## Curvature family carve-out (no single SNR label)
The curvature (alpha-recovery) family does NOT carry a single SNR label: fitting a slope needs a
wide `A1` spread, which by construction spans a wide SNR range. It runs at one easy operating
point (high SNR, low density) and reports **A1-spread statistics** (min/max/quartiles, decades)
per set instead of an SNR label. Its `A1` window is sized per set to keep the brightest ch2 spot
below the detector saturation knee while preserving that spread. See `docs/alpha_convention.md`
and `src/spotpipe/benchmark/alpha.py`.

## Rules
- The benchmark layer uses THIS quantity wherever it says "SNR" (the vendored `_features`
  `peak_snr`). State it in a methods-ready docstring next to the code; do not substitute a
  different SNR formula without updating this doc.
- Family-1 SNR cells are TRUE constant-SNR: the intensity is solved by inverting this formula, no
  jitter, so `realised_snr` has ~zero spread and equals the target. Cells whose solved intensity is
  out-of-distribution, or whose protein channel saturates, are flagged (not silently shipped).
- The curvature family reports A1-spread stats, not an SNR label.

## Why frozen
"Which SNR definition" is non-unique across the literature. Committing to one in writing — and
keeping it faithful to the code — makes every labelled difficulty axis reproducible and
comparable across runs; a silent substitution (or a spec that drifts from the code) would
invalidate cross-run comparisons.
