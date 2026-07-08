# spotpipe (rebuilt)

Two-channel microscopy spot-detection pipeline: FV3000 forward-model simulator
+ HRNet spot detector emitting a frozen canonical schema.

This repo was rebuilt from the old repo by vendoring ONLY the precious core —
simulator, model, losses, schema — pinned at git SHA
`7b9a0b85ee527afeb73d9e68f9bdb30960775083` (see `VENDORED_NOTES.md`). The
benchmark harness, dataloader, training loop, and plotting were deliberately
NOT ported; they are rebuilt fresh in later stages. Read `CLAUDE.md` for the
tier rules before touching anything.

## Quick start (Windows dev)

```
.venv\Scripts\python.exe -m pip install -e . --no-deps   # once
.venv\Scripts\python.exe -m spotpipe.cli smoke           # or: spotpipe smoke
.venv\Scripts\python.exe -m pytest tests/
```

`spotpipe smoke` generates ~50 tiny synthetic images (configs/smoke.yaml),
writes ground truth in the canonical schema, runs the vendored model (random
weights) and writes schema-valid predictions under `outputs/smoke/`. It must
finish in under 60 s on CPU.

## Paths

All paths resolve through `spotpipe.paths` rooted at the `SPOTPIPE_ROOT` env
var (default: repo root). Never hardcode an absolute path.

## Checkpoints

Two trained checkpoints are carried from the old A100 runs under
`src/spotpipe/models/checkpoints/{hrnet_large,hrnet_small}/`, each with its
training `config.yaml`, `manifest.json`, and `PROVENANCE.md` (simulator SHA +
config + seeds). Full run outputs (all step checkpoints, benchmark results,
the 328 MB final archive) remain in
`C:\Users\shivl\Videos\spotpipe_a100_artifacts`.
