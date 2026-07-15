#!/usr/bin/env bash
# FULL 40k head-fix retrain — both candidate heads, then read out. Run on the A100.
#
# WHY BOTH: a 40k retrain is required regardless of head (hrnet_large_measured was trained
# at density 0.012 and is OOD in the v3 benchmark's crowded cells up to 0.025). So the
# previous model cannot be the v3 headline. We train both heads on the v3-covering
# distribution and pick the winner on the thesis metric.
#
# COST: 2 x 40k steps. At ~6 steps/s that is ~1h50m each -> budget ~4 GPU-hours + readouts.
#   Run it in tmux. Detach with Ctrl-b d; DO NOT Ctrl-C.
#
# DECISION (after STEP 3): the winner is the better ALPHA estimator --
#   (1) alpha=0 NULL closest to 0 (the flagship metric; PROJECT_STATE calls it THE set), THEN
#   (2) lower alpha-MAE over the sweep, THEN
#   (3) no bright+dense ratio-bias regression.
#   At 40k both nulls should be clean (the 8k INDEP null of +0.22 was undertraining; the 40k
#   independent model already reached +0.021). So the tiebreak likely falls to alpha-MAE and
#   the bright+dense corner -- where DELTA has the mechanism advantage. If they tie, ship
#   INDEP (simpler, proven). Only the WINNER goes on to the full benchmark vs baselines.

set -euo pipefail
REPO="${REPO:-$HOME/spotpipe_new}"
DEVICE="${DEVICE:-cuda}"
cd "$REPO"

echo "############ STEP 0 — preflight"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('gpu:', torch.cuda.get_device_name(0))"
python -m pytest tests/test_delta_head.py tests/test_density_coverage.py -q
python scripts/make_40k_headfix_configs.py     # regenerate + assert single-knob difference

echo
echo "############ STEP 1 — dataload gate"
spotpipe train --mode profile --config configs/train_40k_DELTA.yaml

echo
echo "############ STEP 2 — train both arms to 40k (resumable; each writes best_checkpoint.pt)"
spotpipe train --config configs/train_40k_INDEP.yaml --require-gpu \
    --out outputs/train/headfix40k-INDEP 2>&1 | tee ~/train40k_INDEP.log
spotpipe train --config configs/train_40k_DELTA.yaml --require-gpu \
    --out outputs/train/headfix40k-DELTA 2>&1 | tee ~/train40k_DELTA.log

echo
echo "############ STEP 3 — read out both (alpha = the thesis metric)"
for ARM in INDEP DELTA; do
  echo "===== $ARM : alpha recovery ====="
  python scripts/alpha_probe.py --checkpoint outputs/train/headfix40k-$ARM \
      --device "$DEVICE" --out results/alpha40k_$ARM.csv 2>&1 | tee ~/alpha40k_$ARM.log
  echo "===== $ARM : bright x dense ratio bias ====="
  python scripts/bright_dense_probe.py --checkpoint outputs/train/headfix40k-$ARM \
      --device "$DEVICE" --n-images 20 --out results/bd40k_$ARM.csv 2>&1 | tee ~/bd40k_$ARM.log
done

cat <<'VERDICT'

############ HOW TO PICK
Compare the two alpha readouts:
  alpha=0 NULL  -> both should be ~0 at 40k. If one is materially off zero, it is
                   MANUFACTURING curvature -> it loses, full stop.
  alpha-MAE     -> lower wins. This is where the 8k comparison was too noisy to trust;
                   40k should be decisive. DELTA has the mechanism edge (no s1-s2 residual).
  bright+dense  -> DELTA should keep its intensity-shrinkage fix; INDEP should still show it.
                   Not decisive for alpha, but it matters for real (bright) data.

Ship the winner: install it as src/spotpipe/models/checkpoints/<name>/ (+ PROVENANCE.md),
retune peak_threshold OFF-benchmark, then run the FULL v3 benchmark:
  spotpipe bench-gen ; spotpipe infer --checkpoint <name> ; spotpipe evaluate
with Gate A (known-alpha) + the alpha=0 null validated BEFORE any baseline is touched.
VERDICT
