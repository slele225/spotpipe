"""Golden tests for the (logI1, delta) intensity-head reparameterisation.

This change touches FROZEN modules (models/heads.py, losses/intensity.py), so it is
tested like one. The invariants that MUST hold:

* independent mode is byte-for-byte the old behaviour (existing checkpoints unaffected);
* the two modes have IDENTICAL state_dict structure (a checkpoint loads either way);
* delta mode derives logI2 = logI1 + delta and logvar2 = logaddexp(logvar1, logvar_delta)
  EXACTLY, so predict_spots and the schema are unchanged;
* the loss routes to (logI1, delta) in delta mode and to (logI1, logI2) otherwise;
* build_targets emits delta = logI2 - logI1 at centres and 0 elsewhere;
* end to end, predict_spots on a delta head emits a canonical-schema frame whose
  log_ratio equals the head's delta.

Requires torch (runs on the dev box, not the no-torch sandbox).
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from spotpipe.losses.intensity import intensity_nll  # noqa: E402
from spotpipe.models.heads import SpotHeads, build_heads  # noqa: E402
from spotpipe.models.spot_model import build_spot_model, predict_spots  # noqa: E402
from spotpipe.schema import SCHEMA_COLUMNS  # noqa: E402
from spotpipe.training.targets import build_targets  # noqa: E402


def _feat(b=2, c=8, h=16, w=16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, c, h, w)


# --------------------------------------------------------------------------- #
# Head structure + backward compatibility                                     #
# --------------------------------------------------------------------------- #
def test_independent_mode_is_the_original_output_set():
    heads = SpotHeads(in_channels=8, parameterisation="independent")
    out = heads(_feat())
    assert set(out) == {"heatmap", "offset", "logI1", "logI2", "logvar1", "logvar2"}
    assert "delta" not in out  # independent mode must not leak the new keys


def test_default_parameterisation_is_independent():
    # Every existing checkpoint has no head_parameterisation key -> must stay independent.
    assert build_heads(8, {}).parameterisation == "independent"
    assert build_heads(8, None).parameterisation == "independent"


def test_state_dict_is_structurally_identical_across_modes():
    # The killer backward-compat guarantee: same param names AND shapes, so a checkpoint
    # trained in either mode loads into a model built in either mode. Only the config's
    # head_parameterisation decides interpretation.
    a = SpotHeads(in_channels=8, parameterisation="independent").state_dict()
    b = SpotHeads(in_channels=8, parameterisation="delta").state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, f"{k}: {a[k].shape} vs {b[k].shape}"


def test_delta_checkpoint_loads_into_independent_model_and_back():
    delta = SpotHeads(in_channels=8, parameterisation="delta")
    indep = SpotHeads(in_channels=8, parameterisation="independent")
    # load_state_dict must succeed with no missing/unexpected keys, both directions.
    missing, unexpected = indep.load_state_dict(delta.state_dict(), strict=True)
    assert not missing and not unexpected
    missing, unexpected = delta.load_state_dict(indep.state_dict(), strict=True)
    assert not missing and not unexpected


def test_invalid_parameterisation_raises():
    with pytest.raises(ValueError):
        SpotHeads(in_channels=8, parameterisation="nonsense")


# --------------------------------------------------------------------------- #
# Delta derivations are EXACT                                                  #
# --------------------------------------------------------------------------- #
def test_delta_derives_logI2_and_logvar2_exactly():
    heads = SpotHeads(in_channels=8, parameterisation="delta")
    out = heads(_feat(seed=1))
    # logI2 == logI1 + delta, to numerical exactness
    assert torch.allclose(out["logI2"], out["logI1"] + out["delta"], atol=1e-6)
    # logvar2 == logaddexp(logvar1, logvar_delta)  (var2 = var1 + var_delta)
    assert torch.allclose(
        out["logvar2"], torch.logaddexp(out["logvar1"], out["logvar_delta"]), atol=1e-6)
    # the native keys the loss needs are present
    assert "delta" in out and "logvar_delta" in out


# --------------------------------------------------------------------------- #
# Loss routing                                                                 #
# --------------------------------------------------------------------------- #
def _one_center_targets(logI1, logI2, h=4, w=4):
    t = {k: torch.zeros(1, 1, h, w) for k in ("center_mask", "logI1", "logI2", "delta")}
    t["center_mask"][0, 0, 1, 1] = 1.0
    t["logI1"][0, 0, 1, 1] = logI1
    t["logI2"][0, 0, 1, 1] = logI2
    t["delta"][0, 0, 1, 1] = logI2 - logI1
    return t


def test_loss_routes_to_delta_when_head_emits_delta():
    # Construct preds where the DERIVED logI2 is deliberately wrong but delta is right.
    # Independent routing would penalise the wrong logI2; delta routing must not.
    h = w = 4
    z = torch.zeros(1, 1, h, w)
    tgt = _one_center_targets(logI1=2.0, logI2=5.0)  # true delta = 3.0

    delta_preds = {
        "logI1": torch.full((1, 1, h, w), 2.0),      # correct
        "delta": torch.full((1, 1, h, w), 3.0),      # correct
        "logI2": torch.full((1, 1, h, w), -99.0),    # deliberately garbage (derived-ish)
        "logvar1": z, "logvar_delta": z, "logvar2": z,
    }
    out = intensity_nll(delta_preds, tgt, use_logvar=False)
    # both terms see correct targets -> ~0 loss, and the garbage logI2 is IGNORED
    assert out["intensity1"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["intensity2"].item() == pytest.approx(0.0, abs=1e-6)


def test_loss_routes_to_logI2_without_delta():
    h = w = 4
    z = torch.zeros(1, 1, h, w)
    tgt = _one_center_targets(logI1=2.0, logI2=5.0)
    indep_preds = {
        "logI1": torch.full((1, 1, h, w), 2.0),
        "logI2": torch.full((1, 1, h, w), 5.0),
        "logvar1": z, "logvar2": z,
    }
    out = intensity_nll(indep_preds, tgt, use_logvar=False)
    assert out["intensity1"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["intensity2"].item() == pytest.approx(0.0, abs=1e-6)


def test_delta_head_missing_delta_target_raises():
    h = w = 4
    z = torch.zeros(1, 1, h, w)
    preds = {"logI1": z, "delta": z, "logvar1": z, "logvar_delta": z}
    targets = {"center_mask": z, "logI1": z}  # no "delta"
    with pytest.raises(KeyError):
        intensity_nll(preds, targets, use_logvar=False)


# --------------------------------------------------------------------------- #
# Target construction                                                          #
# --------------------------------------------------------------------------- #
def test_build_targets_emits_delta_equal_to_logI2_minus_logI1():
    spots = pd.DataFrame({
        "x": [3.4, 10.2], "y": [5.1, 12.9],
        "logI1": [2.0, 4.0], "logI2": [3.5, 4.2],
    })
    t = build_targets(spots, shape=(16, 16), heatmap_sigma=1.5)
    assert "delta" in t
    m = t["center_mask"][0] > 0.5
    assert torch.allclose(t["delta"][0][m], (t["logI2"][0] - t["logI1"][0])[m], atol=1e-6)
    # zero away from centres
    assert torch.count_nonzero(t["delta"][0][~m]) == 0


# --------------------------------------------------------------------------- #
# End to end: schema is unchanged; log_ratio == delta                         #
# --------------------------------------------------------------------------- #
def test_predict_spots_on_delta_model_emits_canonical_schema():
    model = build_spot_model({
        "base_channels": 8, "blocks_per_branch": 1, "head_mid_channels": 16,
        "head_parameterisation": "delta",
        "heatmap_bias": 0.0,  # 0 bias so some pixels clear threshold on random weights
    })
    img = torch.rand(2, 32, 32) * 100.0
    df = predict_spots(model, img, image_id="t", peak_threshold=0.05, device="cpu")
    assert list(df.columns) == list(SCHEMA_COLUMNS)
    if len(df):
        # log_ratio == logI2 - logI1 == delta, reconstructed through the schema
        assert np.allclose(df["log_ratio"], df["logI2"] - df["logI1"], atol=1e-5)
