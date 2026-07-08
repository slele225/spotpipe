# Checkpoint provenance — hrnet-large

* **Checkpoint**: `best_checkpoint.pt` (5.9 MB), selected step 37000 of 40000
  by hard-corner `val_logratio_mae` = 0.4431 (see `manifest.json` for the full
  selection rule and final eval metrics).
* **Experiment**: `2026-06-23_hrnet-large` (A100 run, 2026-06-23/24).
* **Training-data simulator SHA** (old repo, from run `manifest.json`):
  `93fc0aa8c6b245c3c2294a995a3c172d55064f57-dirty`. The vendored simulator in
  this repo is pinned at the later commit `7b9a0b8` of the same repo; the
  training-time working tree was dirty — bit-exact regeneration of the
  training data would require reconstructing that tree from the old repo.
* **Seed**: 0 (master seed; detector constants derived from it once, recorded
  in `manifest.json` under `detector`).
* **Config**: `config.yaml` here (model: base_channels 32, num_branches 3,
  blocks_per_branch 2, head_mid_channels 32; images 256×256; 40k steps).
* **Full run outputs** (all step checkpoints, train_state.pt, loss curves,
  benchmark results):
  `C:\Users\shivl\Videos\spotpipe_a100_artifacts\experiments\2026-06-23_hrnet-large\`
  (path recorded for reference only — never used by code; see CLAUDE.md rule 4).
