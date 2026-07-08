# Checkpoint provenance — hrnet-small

* **Checkpoint**: `best_checkpoint.pt` (1.77 MB), selected step 4000 of 40000
  by the selection rule in `manifest.json` (selected value 1.3676; see
  `manifest.json` for final eval metrics).
* **Experiment**: `2026-06-23_hrnet-small` (A100 run, 2026-06-23/24; ~450k-param
  baseline-capacity run).
* **Training-data simulator SHA** (old repo, from run `manifest.json`):
  `93fc0aa8c6b245c3c2294a995a3c172d55064f57-dirty`. The vendored simulator in
  this repo is pinned at the later commit `7b9a0b8` of the same repo; the
  training-time working tree was dirty — bit-exact regeneration of the
  training data would require reconstructing that tree from the old repo.
* **Seed**: 0 (master seed; detector constants derived from it once, recorded
  in `manifest.json` under `detector`).
* **Config**: `config.yaml` here (model: base_channels 16, num_branches 3,
  blocks_per_branch 2, head_mid_channels 32; images 256×256; 40k steps).
* **Full run outputs**:
  `C:\Users\shivl\Videos\spotpipe_a100_artifacts\experiments\2026-06-23_hrnet-small\`
  (path recorded for reference only — never used by code; see CLAUDE.md rule 4).
