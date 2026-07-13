# SNR convention — CANONICAL DEFINITION (freeze this)

> Definitional spec, referenced from CLAUDE.md in summer_research_26. It fixes the meaning of
> "SNR" for the entire benchmark layer: which SNR quantity is used, and how the SNR difficulty
> axis labels a cell. Like the alpha convention, this must survive across chats and prompts,
> because a future prompt silently substituting a different SNR formula would make difficulty
> axes incomparable between runs. Governs the DISPOSABLE benchmark layer only; touches nothing
> frozen (the vendored simulator keeps its own internal definitions).
>
> This document describes what the code ACTUALLY does. An earlier version of this spec described
> an integrated-SNR closed-form intensity inversion that the benchmark generator never
> implemented; that description has been removed. The generator uses the vendored `_features`
> peak-SNR definition below and labels family-1 cells by a NOMINAL target SNR bin, keeping a wide
> per-spot intensity draw. A spec that misdescribes the code is worse than none.

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

## How the SNR axis labels a cell (family 1: SNR x density)
The SNR x density family does NOT solve for a per-spot intensity that hits a target SNR. Instead:

- Each SNR cell is a **nominal target bin** (frozen checkpoint edges `[0,2,5,10,20,50,inf]`,
  labelled by the bin's lower edge, half-open `[lower, next)`). The cell is realised by a
  scene-override ladder (intensity window, background, PSF) chosen to populate that bin — NOT by
  inverting the SNR formula.
- The per-spot intensity `A_1` is drawn from a deliberately **WIDE** log-intensity window (the
  natural wide draw), because intensity-dependent bias in the downstream `A2/A1` recovery can
  only be seen across a spread of intensities. Consequently **per-spot SNR is a DISTRIBUTION
  around the cell label**, not a single value.
- That distribution is reported honestly: each cell's `meta.json` records the **realised SNR
  quartiles** (`realised_snr`: min / q1 / median / q3 / max / mean over all the cell's spots),
  computed with the `_features` peak-SNR definition above. The label is the target; `realised_snr`
  is what the spots actually are.

The **density axis is orthogonal to SNR** and is handled separately: it is a constant AREA
density (spots/px) set at generation, not tied to intensity or SNR (see the benchmark generator
and `docs`/CLAUDE.md). Every density level is generated at every SNR level (full grid).

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
- Family-1 SNR cells are labelled by nominal target bins; each cell records `realised_snr`
  quartiles so the labelling stays honest.
- The curvature family reports A1-spread stats, not an SNR label.

## Why frozen
"Which SNR definition" is non-unique across the literature. Committing to one in writing — and
keeping it faithful to the code — makes every labelled difficulty axis reproducible and
comparable across runs; a silent substitution (or a spec that drifts from the code) would
invalidate cross-run comparisons.
