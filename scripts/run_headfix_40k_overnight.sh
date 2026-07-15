#!/usr/bin/env bash
# OVERNIGHT: full 40k head-fix bake-off + high-N alpha readout + best-effort v3 benchmark
# with the GT-center ORACLE baseline. Wake up to a decision AND the killer comparison.
#
# Structure (later steps depend on earlier ones; the TESTED work is saved before the
# UNTESTED eval chain runs, so a 3am failure in the tail cannot lose the trained models):
#   STEP 0-2  preflight + train BOTH 40k arms                     (tested this session)
#   STEP 3    high-N alpha (--full-sweep) + bright-dense readout  (tested; the DECISION data)
#   STEP 4    BEST-EFFORT: install both -> bench-gen -> infer -> evaluate --oracle
#             (NOT run end-to-end on this box this session; guarded so failure is harmless)
#
# COST: ~4 GPU-h train + ~30 min readouts + ~30-45 min eval -> ~5-5.5 h wall clock.
# The GPU keeps billing after it finishes -- SHUT THE INSTANCE DOWN when you pull results.
#
# tmux:  tmux new -s train40k ; ... ; Ctrl-b d to detach.  DO NOT Ctrl-C.

set -euo pipefail
REPO="${REPO:-$HOME/spotpipe_new}"
DEVICE="${DEVICE:-cuda}"
cd "$REPO"

echo "############ STEP 0 — preflight"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('gpu:', torch.cuda.get_device_name(0))"
python -m pytest tests/test_delta_head.py tests/test_density_coverage.py -q
python scripts/make_40k_headfix_configs.py

echo
echo "############ STEP 1 — dataload gate"
spotpipe train --mode profile --config configs/train_40k_DELTA.yaml

echo
echo "############ STEP 2 — train both arms to 40k"
spotpipe train --config configs/train_40k_INDEP.yaml --require-gpu \
    --out outputs/train/headfix40k-INDEP 2>&1 | tee ~/train40k_INDEP.log
spotpipe train --config configs/train_40k_DELTA.yaml --require-gpu \
    --out outputs/train/headfix40k-DELTA 2>&1 | tee ~/train40k_DELTA.log

echo
echo "############ STEP 3 — DECISION readouts (high-N alpha is the thesis metric)"
for ARM in INDEP DELTA; do
  echo "===== $ARM : alpha recovery (FULL 13-point sweep, 40 imgs) ====="
  python scripts/alpha_probe.py --checkpoint outputs/train/headfix40k-$ARM \
      --device "$DEVICE" --full-sweep --n-images 40 \
      --out results/alpha40k_$ARM.csv 2>&1 | tee ~/alpha40k_$ARM.log
  echo "===== $ARM : bright x dense ratio bias ====="
  python scripts/bright_dense_probe.py --checkpoint outputs/train/headfix40k-$ARM \
      --device "$DEVICE" --n-images 20 --out results/bd40k_$ARM.csv 2>&1 | tee ~/bd40k_$ARM.log
done
echo ">>> DECISION DATA SAVED (results/alpha40k_*.csv, results/bd40k_*.csv). Everything below is a bonus."

echo
echo "############ STEP 4 — BEST-EFFORT full v3 benchmark + ORACLE (guarded; failure is harmless)"
(
  set +e
  # install both run-dirs as checkpoints (infer needs a NAME under models/checkpoints/)
  for ARM in INDEP DELTA; do
    DST="src/spotpipe/models/checkpoints/headfix40k-$ARM"
    mkdir -p "$DST"
    cp outputs/train/headfix40k-$ARM/best_checkpoint.pt "$DST"/ 2>/dev/null
    cp outputs/train/headfix40k-$ARM/config.yaml        "$DST"/ 2>/dev/null
    cp outputs/train/headfix40k-$ARM/manifest.json      "$DST"/ 2>/dev/null
    echo "installed checkpoint headfix40k-$ARM"
  done

  echo "[eval] generating v3 benchmark on the box (~8 min, CPU)..."
  spotpipe bench-gen 2>&1 | tail -3

  for ARM in INDEP DELTA; do
    echo "[eval] inference: headfix40k-$ARM"
    spotpipe infer --checkpoint headfix40k-$ARM --benchmark data/benchmark \
        --device "$DEVICE" --out outputs/predictions 2>&1 | tail -3
  done

  echo "[eval] evaluate (all method folders + GT-center ORACLE)..."
  spotpipe evaluate --results outputs/predictions --benchmark data/benchmark \
      --oracle --out results/eval40k 2>&1 | tail -20

  echo ">>> STEP 4 done: results/eval40k/ has stratified metrics + alpha + null for both heads AND the oracle."
) || echo ">>> STEP 4 (best-effort eval) FAILED or partial -- the STEP 3 decision data is unaffected. Debug in the morning."

cat <<'VERDICT'

############ MORNING CHECKLIST
1. DECIDE the head from STEP 3:
   grep -aA3 'HEADLINE' ~/alpha40k_INDEP.log ~/alpha40k_DELTA.log
   -> alpha=0 NULL closest to 0 wins first; then lower alpha-MAE; then bright-dense.
2. If STEP 4 completed, the ORACLE comparison is the headline:
   look in results/eval40k/ -- beating the GT-center oracle on alpha proves the win is in
   INTENSITY recovery, not detection (PROJECT_STATE Sec.8). cmeAnalysis/SpotMAX/Spotiflow
   still need their own env setup (a separate task, not overnight-able).
3. SHUT DOWN THE GPU once results are pulled -- it bills idle.
VERDICT
