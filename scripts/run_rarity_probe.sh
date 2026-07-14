#!/usr/bin/env bash
# Rarity-hypothesis probe — run this on the A100 (ssh ubuntu@216.81.245.244).
#
# WHAT THIS DECIDES
#   Whether the intensity head's bright+dense collapse is caused by the dim-biased
#   intensity sampler (full_dim_bias = 1.6, so only ~5% of trained spots exceed ~1,000
#   photons). It trains two SHORT arms that differ in exactly one knob and reads the
#   bright-end bias off both.
#
#   The coverage explanation is already DEAD (docs/coverage_probe_findings.md:
#   brightness and density are drawn independently, corr = +0.02). Rarity is the last
#   suspect before the cause has to be in the loss/head.
#
# COST: 2 x 8,000 steps. Roughly 1/5 of the 40k run, twice -> budget ~1-2 GPU-hours.
#   Do NOT launch the full 40k retrain until STEP 4 below has read out.
#
# STOP CONDITIONS (each step gates the next -- that is the point):
#   * STEP 2 must reproduce the KNOWN defect. If it does not, the probe is wrong. Stop.
#   * STEP 4 decides the sampler for the real retrain. Do not guess it.

set -euo pipefail

REPO="${REPO:-$HOME/summer_research_26}"
DEVICE="${DEVICE:-cuda}"
cd "$REPO"

echo "############ STEP 0 — preflight (cheap; catches a bad sync before it costs GPU time)"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU — refusing to run on CPU'; \
print('gpu:', torch.cuda.get_device_name(0))"
python -m pytest tests/ -q
python scripts/coverage_probe.py --n-images 3000     # must exit 0: benchmark ⊆ training support
python scripts/make_probe_configs.py                 # regenerate + assert the arms differ once

echo
echo "############ STEP 1 — dataload gate (the 50h starvation mode must stay dead)"
spotpipe train --mode profile --config configs/train_probe_A_status_quo.yaml

echo
echo "############ STEP 2 — reproduce the defect on the CURRENT checkpoint"
echo "#   Expect log-ratio bias ~ -0.92 at snr=10/density=0.012 and ~ -1.10 at snr=15."
echo "#   If this does NOT reproduce, the probe is measuring the wrong thing -> STOP."
python scripts/bright_dense_probe.py \
    --checkpoint hrnet_large_measured --device "$DEVICE" --n-images 20 \
    --out results/bright_dense_probe_CURRENT.csv

echo
echo "############ STEP 3 — the two arms (identical but for full_dim_bias)"
spotpipe train --config configs/train_probe_A_status_quo.yaml --require-gpu \
    --out outputs/train/rarity-probe-A
spotpipe train --config configs/train_probe_B_flat.yaml --require-gpu \
    --out outputs/train/rarity-probe-B

echo
echo "############ STEP 4 — read the arms out"
python scripts/bright_dense_probe.py --checkpoint outputs/train/rarity-probe-A \
    --device "$DEVICE" --n-images 20 --out results/bright_dense_probe_ARM_A.csv
python scripts/bright_dense_probe.py --checkpoint outputs/train/rarity-probe-B \
    --device "$DEVICE" --n-images 20 --out results/bright_dense_probe_ARM_B.csv

cat <<'VERDICT'

############ HOW TO READ IT
Compare the "gap (this IS the defect)" line across the three runs:

  CURRENT  gap ~ -0.9 to -1.1     <- the defect, reproduced. If not, stop; fix the probe.
  ARM A    gap ~ same as CURRENT  <- the control. Confirms 8k steps is enough to show it.
                                     If ARM A shows NO gap, 8k steps is too short to
                                     reproduce the defect and the probe is inconclusive --
                                     lengthen the arms, do not read ARM B.
  ARM B    gap -> ~0              <- RARITY CONFIRMED. The dim-biased sampler is the cause.
                                     Set full_dim_bias for the real retrain accordingly
                                     (and re-check the DIM end did not regress -- the dim
                                     tail is where the project's low-bias claim lives).
  ARM B    gap ~ unchanged        <- RARITY REFUTED. Cause is in the loss/head, not the
                                     data. Do NOT spend 40k steps on a sampler change.
                                     Next suspects: the Gaussian-NLL intensity term and the
                                     logvar head's [-10, 6] clamp.

Whatever ARM B says, ALSO check the dim+dense column: flattening the sampler trades bright
accuracy against dim accuracy, and dim is the regime the whole thesis lives in. A "fix"
that wins the bright end and loses the dim end is not a fix.
VERDICT
