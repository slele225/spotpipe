#!/usr/bin/env bash
# Intensity-head bake-off — run on the A100 (ssh ubuntu@216.81.245.244).
#
# WHAT THIS DECIDES
#   Whether the (logI1, delta) reparameterisation fixes the protein-channel-under-
#   crowding shrinkage (docs/shrinkage_probe_findings.md: dense ch2 slope 0.70 while
#   everything else is ~1.0, and the s1-s2 gap lands entirely on the ratio).
#
#   Two 8k-step arms, identical but for model.head_parameterisation:
#     INDEP  = independent (logI1, logI2)  -- the control, matched steps + distribution
#     DELTA  = (logI1, delta)              -- the fix
#
# COST: 2 x 8,000 steps ~= 45 min compute + eval. Budget ~1 GPU-hour.
#   Do NOT launch a 40k run until STEP 4 reads out.
#
# GATE (STEP 4): DELTA must
#   (a) reduce |log_ratio_bias| in the bright+dense corner vs INDEP, AND
#   (b) NOT regress dim+dense (the low-bias claim lives in the dim tail), AND
#   (c) raise the dense-ch2 shrinkage slope toward 1.0 in the shrinkage probe.
#   If DELTA wins (a)+(b) but the shrinkage probe still shows a big s1-s2, be suspicious:
#   the ratio may be right for the wrong reason. All three should move together.

set -euo pipefail

REPO="${REPO:-$HOME/summer_research_26}"
DEVICE="${DEVICE:-cuda}"
cd "$REPO"

echo "############ STEP 0 — preflight"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('gpu:', torch.cuda.get_device_name(0))"
python -m pytest tests/test_delta_head.py tests/test_density_coverage.py -q
python scripts/make_delta_configs.py       # regenerate + assert single-knob difference

echo
echo "############ STEP 1 — dataload gate"
spotpipe train --mode profile --config configs/train_headfix_DELTA.yaml

echo
echo "############ STEP 2 — train both arms (identical but for head_parameterisation)"
spotpipe train --config configs/train_headfix_INDEP.yaml --require-gpu \
    --out outputs/train/headfix-INDEP 2>&1 | tee ~/headfix_INDEP.log
spotpipe train --config configs/train_headfix_DELTA.yaml --require-gpu \
    --out outputs/train/headfix-DELTA 2>&1 | tee ~/headfix_DELTA.log

echo
echo "############ STEP 3 — bright x dense readout (the ratio bias)"
python scripts/bright_dense_probe.py --checkpoint outputs/train/headfix-INDEP \
    --device "$DEVICE" --n-images 20 --out results/headfix_bright_dense_INDEP.csv 2>&1 | tee ~/read_INDEP.log
python scripts/bright_dense_probe.py --checkpoint outputs/train/headfix-DELTA \
    --device "$DEVICE" --n-images 20 --out results/headfix_bright_dense_DELTA.csv 2>&1 | tee ~/read_DELTA.log

echo
echo "############ STEP 4 — shrinkage slopes (did dense-ch2 recover?)"
python scripts/shrinkage_probe.py --checkpoint outputs/train/headfix-INDEP \
    --out results/headfix_shrinkage_INDEP.csv 2>&1 | tee ~/shrink_INDEP.log
python scripts/shrinkage_probe.py --checkpoint outputs/train/headfix-DELTA \
    --out results/headfix_shrinkage_DELTA.csv 2>&1 | tee ~/shrink_DELTA.log

cat <<'VERDICT'

############ HOW TO READ IT
STEP 3 — the ratio, in bright+dense (SNR>=10, density 0.012):
  DELTA |log_ratio_bias| << INDEP |log_ratio_bias|   -> the fix works on the metric that matters.
  Check the dim+dense rows too: they must NOT get worse. A fix that wins bright and loses
  dim is a downgrade (that is what the flat-sampler arm did; do not repeat it).

STEP 4 — the mechanism:
  INDEP dense-ch2 slope ~ 0.70 (reproduces the defect).
  DELTA dense-ch2 slope -> toward 1.0, AND s1 - s2 shrinks toward 0.
  If the ratio improved in STEP 3 but the slopes did NOT move, the improvement is fragile
  -- understand why before spending 40k steps.

If DELTA passes all three: retrain DELTA to 40k, then threshold retune (off-benchmark),
bench-gen v3, infer, evaluate -- with the alpha=0 null control and known-alpha sets FIRST.
If DELTA does not pass: do NOT 40k it. The fallback is the channel-error-correlation penalty
(intensity_head_fix_proposal.md Sec.3), which is a separate, smaller change -- come back and
scope it rather than iterating blind.
VERDICT
