# Vendored code — provenance and notes

Source repo: `C:\Users\shivl\Videos\spotpipe`
Pinned git SHA: **`7b9a0b85ee527afeb73d9e68f9bdb30960775083`** (2026-06-24, "Add legacy v0 HRNet benchmark adapter")

All files below were copied UNCHANGED from that repo except for the single
permitted class of edit (import-path rewrites), listed explicitly.

## File map (old → new)

| Old path | New path | Version | Edits |
|---|---|---|---|
| `src/spotpipe/schema.py` | `src/spotpipe/schema/schema.py` | working tree (= HEAD, file was clean) | none; new `schema/__init__.py` re-exports it so `from spotpipe.schema import ...` keeps working |
| `src/spotpipe/simulator/forward_model.py` | same | **HEAD `7b9a0b8` via `git show`** (working tree had a 9-line uncommitted docstring-only addition documenting the `beta_pi = 2*beta` PI x-axis convention; user chose the HEAD version) | none |
| `src/spotpipe/simulator/{backgrounds,noise,psf,generate_dataset,benchmark_set,__init__}.py` | same | working tree (clean at HEAD) | `benchmark_set.py`: one import rewritten, `spotpipe.benchmark.features` → `spotpipe.simulator._features` |
| `src/spotpipe/benchmark/features.py` | `src/spotpipe/simulator/_features.py` | working tree (clean at HEAD) | none (moved because the frozen-test-set generator `benchmark_set.py` depends on it; the rest of the old benchmark package was NOT ported) |
| `src/spotpipe/models/{backbone,heads,spot_model,__init__}.py` | same | working tree (clean at HEAD) | none |
| `src/spotpipe/losses/{detection,intensity,localization,ratio,__init__}.py` | same | working tree (clean at HEAD) | none |

Deliberately NOT ported (disposable tier, fresh rebuild later): the old
benchmark harness (`benchmark/harness.py`, `metrics.py`, `matching.py`,
`adapters.py`, `baselines.py`, external adapters), the old training loop and
dataloader (`training/`), all `scripts/`, plotting, and `utils/` (nothing in
the precious closure imports it).

## The uncommitted forward_model.py docstring (not vendored, kept for reference)

The old working tree added this note to the `forward_model.py` module
docstring; it documents reporting conventions only (no logic change):

> `beta` here is the INTERNAL generation coefficient (slope vs `log A_1`). The
> PI / biological plots use the x-axis `log(sqrt(A_1)) = 0.5 log A_1`, against
> which the slope is `beta_pi = 2 * beta`. The downstream metric reports and
> compares everything in that PI convention. This is purely a change of x-axis
> variable: it rescales the reported slope by 2 and does NOT change the
> generated intensities, `log A_1`/`log A_2`, or this `beta` parameter — so the
> frozen benchmark and stored schema are untouched.

## Observations (code untouched, per vendoring rules)

* `losses/ratio.py` is an intentional stub (`fit_slope` raises
  `NotImplementedError`) whose docstring forbids a slope loss. It is vendored
  as-is because it *is* the documentation of that rule.
* `simulator/generate_dataset.py::_git_commit()` resolves the repo root as
  `Path(__file__).resolve().parents[3]`, which assumes the
  `src/spotpipe/simulator/` layout. This repo preserves that layout, so the
  manifest `git_commit` field now records THIS repo's SHA (correct behaviour).
* `simulator/benchmark_set.py` imports `tifffile` (raw-channel TIFF export);
  it is in the dependency list.
