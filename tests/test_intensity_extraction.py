"""Golden test for the intensity-extraction measurement instrument.

This module is the one instrument every baseline (and our model's "shared"
numbers) flow through, so a bias here biases every tool at once. These tests
assert both methods recover a KNOWN integrated intensity, that background
subtraction works, that the robust (median) background survives a neighbour,
and that edge spots are refused rather than silently truncated.

The synthetic spots are rendered with the VENDORED forward-model PSF
(``spotpipe.simulator.psf.render_psf``), which is area-normalised: summing the
rendered spot over pixels recovers its total integrated intensity. The extractor
is handed the SAME sigma used to render, matching the real benchmark contract
(true sigma on simulated data, bead-calibrated sigma on real data).

Tolerances: the gaussian fit uses the same erf-integrated kernel as the
renderer, so with fixed sigma it recovers the volume to well under 2%. The
aperture method truncates at radius 3*sigma (captures ~98.9% of a 2-D Gaussian:
1 - exp(-3^2/2)) and pays pixelation/annulus-median noise, so 5% is the honest
bar. Both bars are asserted below.
"""

import numpy as np
import pytest

from spotpipe.benchmark.intensity_extraction import (
    ExtractionConfig,
    extract_channel,
)
from spotpipe.simulator.psf import render_psf

SIGMA = 1.4          # px; representative of the simulator's 1.0-1.8 range
I_TRUE = 5000.0      # known integrated intensity (photon-proportional units)
GAUSSIAN_TOL = 0.02  # <2% for the gaussian fit (design bar)
APERTURE_TOL = 0.05  # <5% for the aperture sum (design bar)


def _render_spot(shape, x, y, intensity, sigma, background=0.0):
    """A clean, area-normalised spot on a flat background (float image)."""
    plane = np.full(shape, float(background), dtype=np.float64)
    render_psf(plane, x, y, intensity, sigma)
    return plane


def test_both_methods_recover_known_intensity():
    # (a) single clean spot, flat zero background.
    plane = _render_spot((64, 64), 32.3, 28.7, I_TRUE, SIGMA)
    xs, ys = np.array([32.3]), np.array([28.7])

    ap = extract_channel(plane, xs, ys, SIGMA, "aperture").intensity[0]
    ga = extract_channel(plane, xs, ys, SIGMA, "gaussian").intensity[0]

    print(f"\n[known-intensity] I_true={I_TRUE:.1f}  aperture={ap:.1f} "
          f"({100*(ap-I_TRUE)/I_TRUE:+.2f}%)  gaussian={ga:.1f} "
          f"({100*(ga-I_TRUE)/I_TRUE:+.2f}%)")

    assert abs(ga - I_TRUE) / I_TRUE < GAUSSIAN_TOL
    assert abs(ap - I_TRUE) / I_TRUE < APERTURE_TOL


def test_background_subtraction():
    # (b) same spot on a known flat offset -> both must still recover I_true.
    offset = 120.0
    plane = _render_spot((64, 64), 30.0, 33.4, I_TRUE, SIGMA, background=offset)
    xs, ys = np.array([30.0]), np.array([33.4])

    ap = extract_channel(plane, xs, ys, SIGMA, "aperture").intensity[0]
    ga = extract_channel(plane, xs, ys, SIGMA, "gaussian").intensity[0]

    print(f"\n[background+{offset:.0f}] I_true={I_TRUE:.1f}  aperture={ap:.1f} "
          f"({100*(ap-I_TRUE)/I_TRUE:+.2f}%)  gaussian={ga:.1f} "
          f"({100*(ga-I_TRUE)/I_TRUE:+.2f}%)")

    assert abs(ga - I_TRUE) / I_TRUE < GAUSSIAN_TOL
    assert abs(ap - I_TRUE) / I_TRUE < APERTURE_TOL


def test_median_background_robust_to_neighbour():
    # (c) a bright neighbour whose CORE lands in the target's background annulus.
    # The median background shrugs off the handful of contaminated annulus pixels;
    # a MEAN background is inflated by them, so it over-subtracts and drags the
    # aperture estimate far below truth. Per the prompt this is a robustness
    # sanity check, not an exact-recovery bar (the neighbour's wings unavoidably
    # leak a little flux into the aperture itself, which no background method can
    # remove), so we assert median clearly beats mean plus a loose sanity bound.
    cx, cy = 32.0, 32.0
    plane = _render_spot((64, 64), cx, cy, I_TRUE, SIGMA)
    # Neighbour centred in the annulus (~5*sigma away, between inner 4 and outer
    # 6), much brighter, so its core pixels sit in the background ring.
    render_psf(plane, cx + 5.0 * SIGMA, cy, 6.0 * I_TRUE, SIGMA)

    xs, ys = np.array([cx]), np.array([cy])
    ap = extract_channel(plane, xs, ys, SIGMA, "aperture").intensity[0]

    # Counterfactual: recompute with a MEAN annulus background by hand.
    cfg = ExtractionConfig()
    r_ap = cfg.aperture_radius_sigma * SIGMA
    r_in = cfg.annulus_inner_sigma * SIGMA
    r_out = cfg.annulus_outer_sigma * SIGMA
    yy, xx = np.mgrid[0:64, 0:64]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ap_mask = d2 <= r_ap ** 2
    an_mask = (d2 > r_in ** 2) & (d2 <= r_out ** 2)
    ap_sum = plane[ap_mask].sum()
    mean_bg_est = ap_sum - plane[an_mask].mean() * ap_mask.sum()

    print(f"\n[neighbour] I_true={I_TRUE:.1f}  aperture(median)={ap:.1f} "
          f"({100*(ap-I_TRUE)/I_TRUE:+.2f}%)  aperture(mean)={mean_bg_est:.1f} "
          f"({100*(mean_bg_est-I_TRUE)/I_TRUE:+.2f}%)")

    # Median stays far closer to truth than the neighbour-inflated mean...
    assert abs(ap - I_TRUE) < abs(mean_bg_est - I_TRUE)
    # ...and by a wide margin (mean is off by multiples of I_true here).
    assert abs(ap - I_TRUE) < 0.5 * abs(mean_bg_est - I_TRUE)
    # Loose sanity bound on the median estimate (NOT the tight recovery bar).
    assert abs(ap - I_TRUE) / I_TRUE < 0.25


def test_edge_spot_returns_nan_and_flag():
    # (d) a spot whose annulus/crop runs off the border must be refused (NaN +
    # 'edge' flag), never crash and never silently truncate.
    plane = _render_spot((64, 64), 2.0, 2.0, I_TRUE, SIGMA)
    xs, ys = np.array([2.0]), np.array([2.0])

    for method in ("aperture", "gaussian"):
        res = extract_channel(plane, xs, ys, SIGMA, method)
        print(f"\n[edge/{method}] intensity={res.intensity[0]}  flags={res.flags[0]!r}")
        assert np.isnan(res.intensity[0])
        assert "edge" in res.flags[0]


def test_sigma_is_required_and_validated():
    plane = np.zeros((16, 16))
    xs, ys = np.array([8.0]), np.array([8.0])
    with pytest.raises(ValueError):
        extract_channel(plane, xs, ys, 0.0, "aperture")
    with pytest.raises(ValueError):
        extract_channel(plane, xs, ys, -1.0, "gaussian")
