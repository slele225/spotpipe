# Experiment: 2026-06-23_hrnet-large

- **Git commit (pinned):** `ce697f485b7191529a1ea4fae2d2f36aebe5c09a-dirty` (update to the 5b commit before launch)
- **Config:** [`config.yaml`](config.yaml)
- **Outputs:** [`outputs/`](outputs/)
- **Baseline:** [`2026-06-23_hrnet-small`](../2026-06-23_hrnet-small)

## Motivation

The capacity arm of the 5b comparison: **does a modest 3-4x bump in backbone capacity
help in the hard regime?** Per CLAUDE.md we are *not* chasing capacity for its own sake;
this is a controlled test of whether more parameters reduce per-spot ratio bias/variance
in the dim x high-overlap corner, where the project's central low-bias / low-variance
claim lives.

Everything except model width is held **identical** to `hrnet-small`: same architecture
family, heads, losses, target construction, simulator, benchmark metrics, schedule, seed
(so the same training stream and the same fixed instrument), and the **same shared FIXED
validation set** for selection. Only `base_channels` changes (16 -> 32).

## Config diff from baseline (`hrnet-small`)

- **Model only:** `base_channels: 16 -> 32`. Confirmed param count **1,533,223**
  (~3.4x the small run's 449,351), squarely in the 1-2M target band. `head_mid_channels`
  is kept at 32 so the heads are unchanged and the comparison isolates backbone capacity.
- All training / curriculum / benchmark settings are identical to `hrnet-small`.

This is a **config-only widening** of the existing HRNet -- no new architecture family,
no new heads, no new loss terms, no simulator or benchmark-metric changes.

## Results

_Filled in after the run. Same artifact layout as `hrnet-small` under `outputs/`._
Headline comparison to fill in: small vs large **hard-corner `val_logratio_mae`**
(best checkpoint) and the benchmark hard-corner ratio bias/variance cell.

## Decision

_Did the extra capacity help in the hard corner? Keep / discard / follow-up._
