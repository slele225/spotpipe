"""Regression test: the benchmark must have ZERO registration shift.

BACKGROUND (2026-07-13)
-----------------------
`forward_model.sample_scene_params` draws an independent per-image, per-channel
registration shift ~ U(-max_px, +max_px) and renders channel k at `spot + shift_k`.
`max_px` DEFAULTS TO 1.0, so it was silently active in the benchmark even though
no benchmark config asked for it.

The ground-truth CSV stores the SCENE position. With a nonzero shift, GT therefore
marks a point the photons are NOT centred on -- in either channel -- and the shift
is not recorded anywhere. This was caught only because cmeAnalysis's localization
residual came out at sd 0.577 px = 1/sqrt(3) = the sd of U(-1,1), FLAT across a 25x
intensity range (i.e. not photon-limited, so not localization error).

Why it matters enough to test:
  * a SINGLE-channel detector sits mean 0.765 px from GT no matter how good it is;
  * a TWO-channel method can average its two views and land 1.47x closer FOR FREE,
    flattering our own model against every single-channel baseline;
  * sqrt(2) = 1.414 px of the evaluator's 1.68 px match radius is consumed before
    a single photon of noise.

Random shift is legitimate TRAINING augmentation and stays in the training configs.
It must never be in the BENCHMARK's ground truth. A benchmark measures; it does not
randomise.
"""

from __future__ import annotations

import numpy as np
import pytest

from spotpipe.benchmark.generate import _ZERO_REGISTRATION, _deep_merge
from spotpipe.simulator import forward_model
from spotpipe.simulator.psf import GaussianPSF


def test_zero_registration_override_is_explicitly_zero():
    """Must be an explicit 0.0. Omitting the key gives you 1.0 from the default."""
    assert _ZERO_REGISTRATION["registration_shift"]["max_px"] == 0.0


def test_forward_model_default_is_nonzero_so_omission_is_not_safe():
    """Guards the REASON this test file exists.

    If someone 'simplifies' the benchmark by deleting the override, they must not
    be able to believe omission is equivalent. It is not: the default is 1.0.
    """
    scene = forward_model.sample_scene_params({}, np.random.default_rng(0), (64, 64))
    assert scene.shift1 != (0.0, 0.0) or scene.shift2 != (0.0, 0.0), (
        "forward_model's registration_shift default appears to have become 0. "
        "If that is intentional, this test can be relaxed -- but the benchmark "
        "override must STILL be explicit."
    )


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_benchmark_scene_has_no_registration_shift(seed):
    """With the benchmark override applied, BOTH channels must be unshifted."""
    cfg = _deep_merge({}, _ZERO_REGISTRATION)
    scene = forward_model.sample_scene_params(cfg, np.random.default_rng(seed), (64, 64))
    assert scene.shift1 == (0.0, 0.0), f"ch1 registration shift {scene.shift1} != 0"
    assert scene.shift2 == (0.0, 0.0), f"ch2 registration shift {scene.shift2} != 0"


def test_ground_truth_position_is_where_the_light_actually_is():
    """END-TO-END: the GT coordinate must sit on the rendered spot's centroid.

    This is the property that actually broke. Render a spot with the simulator's
    own PSF at a known sub-pixel centre and recover the centre with an independent
    pixel-integrated Gaussian fit. They must agree to well under a tenth of a pixel.
    """
    from scipy.optimize import least_squares
    from scipy.special import erf

    SIG, TRUE_X, TRUE_Y, A = 1.4, 32.371, 28.842, 1000.0
    canvas = np.zeros((64, 64))
    GaussianPSF(sigma=SIG).render_into(canvas, TRUE_X, TRUE_Y, A)

    def pixel_integrated(cx, cy, sig, h, w):
        x, y = np.arange(w), np.arange(h)
        gx = 0.5 * (erf((x + 0.5 - cx) / (np.sqrt(2) * sig)) - erf((x - 0.5 - cx) / (np.sqrt(2) * sig)))
        gy = 0.5 * (erf((y + 0.5 - cy) / (np.sqrt(2) * sig)) - erf((y - 0.5 - cy) / (np.sqrt(2) * sig)))
        return np.outer(gy, gx)

    fit = least_squares(
        lambda p: (p[0] * pixel_integrated(p[1], p[2], SIG, 64, 64) + p[3] - canvas).ravel(),
        [900.0, 32.0, 29.0, 0.0],
    ).x
    assert abs(fit[1] - TRUE_X) < 0.02, f"x off by {fit[1]-TRUE_X:+.4f} px"
    assert abs(fit[2] - TRUE_Y) < 0.02, f"y off by {fit[2]-TRUE_Y:+.4f} px"
