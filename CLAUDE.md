# spotpipe (rebuilt repo)

Two-channel microscopy spot-detection pipeline. This repo was rebuilt from
`C:\Users\shivl\Videos\spotpipe` (old repo) by vendoring ONLY the precious core;
everything else is rebuilt fresh here.

## The precious-core principle

Three tiers, hardened by how expensive they are to reproduce:

* **PRECIOUS** (cannot cheaply reproduce): the simulator / training-data
  generator (`src/spotpipe/simulator/`), the model (`src/spotpipe/models/`),
  the losses (`src/spotpipe/losses/`). Trained checkpoints
  (`src/spotpipe/models/checkpoints/`) are precious. These are vendored
  UNCHANGED from the old repo and pinned by git SHA (see `VENDORED_NOTES.md`).
* **FROZEN INTERFACE**: the schema (`src/spotpipe/schema/schema.py`) — the
  column format of prediction files and ground-truth files. This is what lets
  every other piece be swapped freely.
* **DISPOSABLE** (cheap to rewrite): benchmark harness, metrics, plotting,
  baseline adapters. The old harness was deliberately NOT ported; it will be
  rebuilt fresh in `src/spotpipe/benchmark/` (currently empty).

## Durable rules

1. **Simulator / models / losses / schema are VENDORED + FROZEN.** Do not
   modify their logic without an explicit instruction naming the file.
   Anything that looks buggy goes in `VENDORED_NOTES.md`; the code stays
   untouched.
2. **The schema is frozen.** Never add, rename, or remove fields in
   `spotpipe.schema`. Suffix convention: `_hat` = estimate.
3. **NO slope/alpha/beta loss exists or may be added.** The curvature/ratio-law
   slope is computed ONLY in the downstream analysis/benchmark layer, never as
   a training signal. `losses/ratio.py` is a deliberate stub documenting this;
   it must never become a loss.
4. **All paths come from `spotpipe.paths`.** NO hardcoded absolute paths
   anywhere — dev is Windows local, training is Linux remote; a `C:\` string
   will break the remote. Paths root at env var `SPOTPIPE_ROOT` (default:
   repo root).
5. **Every compute-adding prompt ends with a timing/verification report and a
   golden/sanity test.** No long runs before the smoke config
   (`configs/smoke.yaml`, `spotpipe smoke`) passes.
6. **Cached datasets are portable directory artifacts** moved by
   `scripts/sync_to_remote.sh` — never referenced by machine-specific path.
7. **Model checkpoints record the simulator SHA + config + seeds of their
   training data.** See `src/spotpipe/models/checkpoints/*/PROVENANCE.md`.
8. **Ambiguous scope, or a change touching a frozen module → STOP and ask.**

## Layout

```
configs/            smoke.yaml (tiny, <60s CPU), default.yaml (real sizes)
src/spotpipe/
  paths.py          ALL path resolution (SPOTPIPE_ROOT)
  config.py         typed config loader (dataclasses over yaml)
  schema/           FROZEN interface (schema.py is the source of truth)
  simulator/        VENDORED forward model + dataset/benchmark-set generators
  models/           VENDORED HRNet backbone + heads + inference (+ checkpoints/)
  losses/           VENDORED detection/localization/intensity losses
  data/             (empty) dataloader — later build stage
  benchmark/        (empty) fresh harness — later build stage
  cli.py            `spotpipe smoke` now; train/bench later
tests/              test_import, test_schema_roundtrip, test_smoke
scripts/sync_to_remote.sh   rsync repo + named dataset dir to a remote host
```

## Environment

* Windows dev: `.venv` (Python 3.12, torch CPU). Run tests with
  `.venv\Scripts\python.exe -m pytest tests/`.
* Old repo (do not modify): `C:\Users\shivl\Videos\spotpipe` @ `7b9a0b8`.
* Old A100 artifacts: `C:\Users\shivl\Videos\spotpipe_a100_artifacts`.

Alpha/slope convention is frozen in docs/alpha_convention.md; follow it.

SNR labeling convention is frozen in docs/snr_convention.md; follow it for all benchmark difficulty-axis labeling and any SNR computation, and never substitute a different SNR definition without updating that doc.