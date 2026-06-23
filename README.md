# spotpipe

A two-channel microscopy **spot-detection pipeline**. A neural network detects
diffraction-limited spots in two-channel confocal images and estimates per-spot
intensities in each channel, so a downstream log-ratio / slope analysis can
recover a biological relationship. Training is on synthetic data from a forward
model of an Olympus **FV3000** (analog-integration PMT detector); real-data
calibration comes later.

> This repo is built in stages. Right now it is the **skeleton only** — most
> scientific logic is stubbed with `raise NotImplementedError`. See
> [`CLAUDE.md`](CLAUDE.md) for the durable design rules and the build order.

## The shared-code / isolated-experiment rule

This is the central organizing principle of the repo:

- **Shared code lives only in `src/spotpipe/`.** There is exactly one copy. It is
  installed editable and imported as `from spotpipe.simulator import ...` from
  anywhere — never via `sys.path` hacks.
- **Experiments hold config + a results README + outputs — never code.** An
  experiment is defined as *"shared code at a pinned git commit + this config."*
  Because the code is pinned by commit and imported from the installed package,
  copying code into an experiment folder is never necessary.

To start a new experiment:

```bash
python scripts/new_experiment.py my-slug
```

This stamps `experiments/YYYY-MM-DD_my-slug/` with a `config.yaml` (which records
the current git commit for reproducibility), a `README.md` skeleton
(Motivation / Config diff from baseline / Results / Decision), and an empty
`outputs/` directory.

## Layout

```
src/spotpipe/
  simulator/   forward_model, noise, psf, backgrounds, generate_dataset   (stubs)
  models/      backbone, heads, spot_model                                (stubs)
  losses/      detection, localization, intensity, ratio                  (stubs)
  training/    train                                                      (stub)
  benchmark/   harness, adapters, metrics                                 (stubs)
  utils/       io, matching                                               (stubs)
  schema.py    canonical spot-output schema                               (implemented)
experiments/   per-experiment config + README + outputs (no code)
scripts/
  new_experiment.py   experiment scaffolder                              (implemented)
CLAUDE.md      durable design rules
pyproject.toml uv / src-layout / editable `spotpipe`
```

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) with a src-layout, editable package.

```bash
uv sync
```

`uv sync` installs `spotpipe` editable, so `from spotpipe.schema import SpotRecord`
resolves cleanly from anywhere, including a subfolder of `experiments/`.
