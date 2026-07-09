# Prompt — Benchmark generation (two families) [fresh build, main repo]

> Paste into summer_research_26 AFTER the alpha-convention rules are added to CLAUDE.md.
> This GENERATES benchmark image sets + ground truth. It does NOT run any method, does NOT
> fit slopes, does NOT build the harness. Generation and evaluation stay separate so the
> fitter can later be verified against known-alpha sets. Uses only the vendored simulator;
> modifies nothing vendored.

## Preconditions (read first)
- Follow docs/alpha_convention.md. alpha = slope of log(A2/A1) vs log(sqrt(A1)) = 2*sim_log_slope.
  The factor of 2 lives only in sim_slope_to_alpha(). Never name a simulator-space var `alpha`/`beta`.
- Homogeneous-condition design: each benchmark SET is one condition; the image is the unit.
- Every image's metadata MUST carry per-image, per-channel ground_truth_sigma (sigma1 AND sigma2).

## Output layout (portable directory artifact; no hardcoded paths, via spotpipe.paths)
```
<bench_root>/
  snr_density/
    snr={S}_density={D}/
      images/            # TIFFs (individual or stack) the baselines will read
      ground_truth/      # schema-format GT: positions + true A1,A2 (+ true logI1,logI2)
      meta.json          # condition label (S,D), per-image sigma1/sigma2, n_images, seed
  curvature/
    alpha={A}/
      images/
      ground_truth/
      meta.json          # true_alpha=A, sim_log_slope=A/2 (via sim_slope_to_alpha inverse),
                         #   per-image sigma1/sigma2, A1 spread stats, seed
  BENCH_MANIFEST.json    # everything generated, seeds, simulator SHA, config hash
```

## Family 1 — SNR x density (detection + intensity-bias-vs-difficulty)
- Grid = checkpoint edges: SNR in [0,2,5,10,20,50,inf], density in [0,1,3,6,inf]
  (use the simulator's SNR/density definitions from _features.py; document them in meta).
- Each cell: 20-50 homogeneous images (make it configurable; default 30).
- Hold ratio law at a fixed neutral setting (pin sim_log_slope=0, sim_intercept=0) so this
  family isolates difficulty, not slope. Keep the natural wide A1 draw.
- GT per spot in frozen schema: position + true A1, A2 (and true logI1/logI2).

## Family 2 — curvature (alpha recovery), at an EASY operating point
- Run at HIGH SNR / LOW density (pick the easiest grid cell) so slope recovery is tested in
  isolation from detection failure. Document the chosen operating point in meta.
- Sweep TRUE alpha via pinned sim_log_slope = alpha/2:
    full range: alpha in [-1.2,-0.9,-0.6,-0.3,0,0.3,0.6,0.9,1.2]
    PLUS dense near zero: alpha in [-0.15,-0.075,0,0.075,0.15]
    (union; alpha=0 appears once)
- **alpha=0 NULL CONTROL is critical**: generate MORE images for alpha=0 (e.g. 3x the others)
  for tight error bars. This set detects methods that manufacture curvature from size-dependent
  intensity bias. Flag it explicitly in meta as the null control.
- Each set MUST retain a wide A1 spread (needed to fit a slope). Record A1 min/max/quartiles
  in meta and assert the spread exceeds a minimum decades threshold, else warn.
- Record true_alpha in meta using the frozen conversion. Do NOT fit any slope here.

## Hard rules
- Modify NOTHING vendored. Use meta['ground_truth_sigma'] plumbing for per-image sigma1/sigma2.
- Generation only: NO method runs, NO slope fitting, NO metrics.
- All paths via spotpipe.paths. Deterministic: every set records its seed; regeneration reproduces.
- GT files validate against the frozen schema (reuse schema roundtrip).

## Sanity tests (must pass)
- A generated snr_density cell reloads; GT is schema-valid; n_images correct.
- A curvature set's recorded true_alpha == 2*sim_log_slope (asserts the convention wiring).
- A curvature set's A1 spread exceeds the minimum threshold.
- alpha=0 set exists, is flagged as null control, and has the larger image count.
- Determinism: regenerating one small set with the same seed yields identical GT.

## TIMING / SIZE report
Print total images, total spots, disk size, and generation time per family. Keep a `smoke`
bench (tiny counts) that generates in < a few seconds for pipeline tests.

## STOP
Show the tree, meta.json for one snr_density cell and one curvature set (incl. the alpha=0
null control), sanity-test results, and the size/timing report.