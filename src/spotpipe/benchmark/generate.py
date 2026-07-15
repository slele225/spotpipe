"""Benchmark image-set generator -- two families, generation ONLY.

Builds the two benchmark families the project needs, as a portable directory
artifact (paths via :mod:`spotpipe.paths`; no absolute paths). This module
GENERATES images + ground truth and NOTHING else: it runs no detection method,
fits no slope, computes no metric. Generation and evaluation stay separate so the
downstream slope fitter can later be validated against these known-alpha sets.

It uses ONLY the vendored simulator (``forward_model``, ``noise``, ``_features``)
and the frozen schema; it modifies nothing vendored. The design mirrors the
vendored ``simulator/benchmark_set.py`` -- targeted scene-config overrides
deep-merged onto a base scene, ratio-law parameters PINNED per set via
``min == max`` -- but writes the fresh two-family layout this build stage wants.

Layout (under ``bench_root``)::

    snr_density/
      snr={S}_density={D}/          one homogeneous CONDITION per cell; S = TRUE
                                    constant target SNR, D = constant area density (spots/px)
        images/ image_<id>.tif      raw observed counts, uint16 [2,H,W] stack
        ground_truth/ gt_<id>.csv   frozen-schema GT: x,y + true logI1/logI2/I1/I2
        meta.json                   label (S,D), fixed sigma1/sigma2, solved intensity,
                                    realised SNR stats (~zero spread), n_images, seed
    curvature/
      alpha={A}/
        images/ ...
        ground_truth/ ...
        meta.json                   true_alpha, sim_log_slope=alpha/2, per-image
                                    sigma1/sigma2, A1-spread stats, seed, null flag
    BENCH_MANIFEST.json             everything generated: seeds, git SHA, config hash

Two conventions worth stating up front (both recorded in every ``meta.json``):

* Family 1 cells are TRUE constant-SNR conditions (v2): given the FIXED PSF, the
  CONSTANT background and the MEASURED detector, the single spot intensity is
  SOLVED by inverting the frozen ``_features`` SNR so ``min(snr1, snr2) ==
  target``, and every spot gets that same intensity (no jitter). The realised
  per-spot SNR therefore has ~zero spread and equals the target. Targets are
  chosen so NO cell clips either channel's ADC (asserted at generation); a solved
  intensity outside the LEGACY checkpoints' training range is flagged (a
  measured-detector retrain is in progress -- see the manifest retrain note).
  (Family 2, the curvature family, deliberately KEEPS a wide A1 spread -- below.)
* ``ground_truth_sigma`` (true per-image, per-channel PSF width) is plumbed from
  ``meta['ground_truth_sigma']`` into every image's record -- the schema's
  ``sigma*_hat`` columns mean "model estimate" and stay NaN for GT.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from spotpipe.benchmark.alpha import alpha_to_sim_slope, sim_slope_to_alpha
from spotpipe.schema import SCHEMA_COLUMNS, write_spots
from spotpipe.simulator import forward_model, noise, psf
from spotpipe.simulator._features import (
    DetectorAxisParams,
    axis_params_from_meta,
    local_neighbor_count,
    peak_snr,
)
from spotpipe.simulator.generate_dataset import _git_commit

__all__ = [
    "BenchmarkConfig",
    "generate_benchmark",
    "generate_snr_density_family",
    "generate_curvature_family",
    "load_benchmark_config",
]


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge (override wins); returns a new dict, inputs untouched."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _edge_label(value: float) -> str:
    """Format a grid edge for a directory name (``inf`` stays ``inf``)."""
    if value is None or (isinstance(value, float) and math.isinf(value)):
        return "inf"
    return f"{value:g}"


def _set_seed(master_seed: int, family: str, label: str) -> int:
    """Deterministic per-set seed from (master seed, family, label).

    Derived by hashing so that regenerating ONE set from its recorded seed
    reproduces it byte-for-byte, independent of generation order (CLAUDE.md
    determinism rule). 32-bit so it is a clean NumPy seed.
    """
    h = hashlib.sha256(f"{int(master_seed)}::{family}::{label}".encode()).hexdigest()
    return int(h[:8], 16)


def _summary(values: np.ndarray) -> dict:
    """min / q1 / median / q3 / max / mean of a 1-D array (NaN-safe, empty-safe)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None,
                "max": None, "mean": None}
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "q1": float(np.quantile(v, 0.25)),
        "median": float(np.median(v)),
        "q3": float(np.quantile(v, 0.75)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
    }


def _config_hash(config: dict) -> str:
    """Stable short hash of a config dict (order-independent)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Benchmark-wide FIXED measurement constants (v2)                              #
# --------------------------------------------------------------------------- #
# A benchmark is a CONTROLLED measurement, not domain randomisation: the PSF, the
# background and (family 1) the per-spot SNR are PINNED, not drawn. These are the
# frozen benchmark constants; the measured per-channel detector gain / offset /
# read variance come from the config (configs/benchmark*.yaml simulator_overrides).
_BENCH_SIGMA1: float = 1.4              # lipid   (ch1) PSF sigma, px -- FIXED everywhere
_BENCH_SIGMA2: float = 1.68            # protein (ch2) PSF sigma, px -- FIXED everywhere
_BENCH_C2_MISMATCH: float = _BENCH_SIGMA2 / _BENCH_SIGMA1   # 1.2 exactly (sigma2 = 1.2 * sigma1)
_BENCH_BACKGROUND_PHOTONS: float = 2.0  # flat photon background, CONSTANT, both channels

# LEGACY-checkpoint training coverage (docs/training_distribution.md). A solved
# family-1 intensity or a curvature A1 outside this range is out-of-range for the
# LEGACY checkpoints (a coverage artifact, not a method difference) and is
# FLAGGED, never shipped silently. Density above the legacy max is a STRESS cell.
# NB: a measured-detector RETRAIN is in progress -- its per-image intensity range
# is solved to keep both channels unclipped, so this legacy range will be replaced
# by the retrained model's actual training range (see manifest retrain note).
_LEGACY_A1_MIN: float = 20.0            # ~ 10**1.3 photons (legacy checkpoints)
_LEGACY_A1_MAX: float = 7943.0         # ~ 10**3.9 photons (legacy checkpoints)
_LEGACY_DENSITY_MAX: float = 0.012     # spots/px (legacy log-uniform training max)

# CURRENT training density coverage -- ``configs/train.yaml`` scene.density.max.
# 2026-07-14: raised 0.012 -> 0.024 so the benchmark's new top density level (0.02)
# sits INSIDE the trained range with headroom instead of on the boundary. The
# density-stress flag is raised against THIS number (the headline model's coverage),
# not the legacy one; both are recorded in the manifest. KEEP IN SYNC WITH
# configs/train.yaml -- a silent divergence mislabels which cells are out-of-distribution.
_TRAINED_DENSITY_MAX: float = 0.030    # spots/px (current log-uniform training max)
_RETRAIN_NOTE: str = (
    "Ranges compared against are the LEGACY checkpoints' training distribution. A "
    "measured-detector RETRAIN is in progress (per-image intensity solved to keep "
    "both channels unclipped); update these comparisons against the retrained "
    "model's actual training range once it lands.")

# Fixed-PSF and constant-background scene overrides, applied to EVERY set in BOTH
# families (Change 2 + Change 3). No per-image sigma or background randomisation
# anywhere in the benchmark -- classical baselines take a FIXED detection sigma,
# and a benchmark measures rather than randomises. gradient == structure == 0.
_FIXED_PSF: dict = {
    "psf": {"sigma1": {"min": _BENCH_SIGMA1, "max": _BENCH_SIGMA1},
            "c2_sigma_mismatch": {"min": _BENCH_C2_MISMATCH, "max": _BENCH_C2_MISMATCH}},
}
_CONSTANT_BACKGROUND: dict = {
    "background": {"level": {"min": _BENCH_BACKGROUND_PHOTONS, "max": _BENCH_BACKGROUND_PHOTONS},
                   "gradient_frac": {"min": 0.0, "max": 0.0},
                   "structure_frac": {"min": 0.0, "max": 0.0}},
}

# ZERO REGISTRATION SHIFT -- applied to EVERY set in BOTH families.
#
# WHY (found 2026-07-13 while adapting cmeAnalysis; this was a real benchmark bug):
# `forward_model.sample_scene_params` draws an INDEPENDENT per-image, per-channel
# registration shift ~ U(-max_px, +max_px) and renders channel k at `spot + shift_k`.
# `max_px` DEFAULTS TO 1.0, so it was active in the benchmark even though no
# benchmark config ever asked for it. The ground-truth CSV stores the SCENE
# position -- so GT marked a point the photons were NOT centred on, in EITHER
# channel, and the per-image shift was never recorded anywhere.
#
# Consequences, all measured:
#   * A single-channel detector (e.g. cmeAnalysis, which detects only in its
#     master channel) sits a mean 0.765 px from GT no matter how well it fits.
#     Observed: cmeAnalysis residual sd 0.569/0.581 px per axis, FLAT across a 25x
#     intensity range -- not photon-limited, so not localization error at all.
#     U(-1,1) has sd 1/sqrt(3) = 0.577. That was the entire residual.
#   * A TWO-channel method can average its two observations and land a mean
#     0.521 px from GT -- 1.47x closer, FOR FREE, with no better detection. That
#     silently flatters our own model against every single-channel baseline.
#   * Max shift displacement sqrt(2) = 1.414 px vs the evaluator's 1.68 px match
#     radius: 84% of the tolerance budget consumed before any noise.
#   * ch1/ch2 shift INDEPENDENTLY -> median 1.02 px (max 2.79 px) channel-to-channel
#     misregistration, biasing any ch2 intensity read at ch1 positions.
#
# A real microscope's channel registration is a FIXED, calibrated affine; it does
# not re-randomise every field of view. As TRAINING augmentation the random shift is
# legitimate and stays in the training configs. In a BENCHMARK's ground truth it is a
# measurement artifact. A benchmark measures; it does not randomise -- the same
# principle as _FIXED_PSF / _CONSTANT_BACKGROUND above.
#
# Set EXPLICITLY to 0.0 -- do NOT rely on omission; the forward model's default is 1.0.
_ZERO_REGISTRATION: dict = {
    "registration_shift": {"max_px": 0.0},
}

# Density axis = CONSTANT AREA DENSITY (spots per pixel), set at generation and
# used as BOTH the knob and the cell label -- no neighbour-count labels, no
# clustering. Each level pins the vendored log-uniform density draw to a single
# value (``min == max`` -> the draw returns exactly that value), so every image
# in the cell realises the IDENTICAL area density and ``n_spots =
# round(density*H*W)`` is constant across the cell. ``cluster_prob = 0`` selects
# the vendored deterministic exact-count uniform-random placement path. Levels
# sweep across and slightly beyond the training density range.
def _constant_density_override(density_spots_per_px: float) -> dict:
    """Scene override pinning one constant area density with uniform placement.

    ``density.min == density.max`` makes ``sample_scene_params`` draw exactly this
    value for every image (``log-uniform(x, x) == x``), so the whole cell shares
    one area density and one ``n_spots``. ``oversample_dense_fraction = 0`` and
    ``clustering.cluster_prob = 0`` force the vendored uniform-random,
    exact-count spot placement -- NO clustering anywhere in the benchmark. This
    passes config to the vendored forward model; it edits nothing vendored.
    """
    return {
        "density": {"min": float(density_spots_per_px),
                    "max": float(density_spots_per_px)},
        "oversample_dense_fraction": 0.0,
        "clustering": {"cluster_prob": 0.0},
    }

# Neutral, ZERO-SCATTER ratio law for the WHOLE snr_density family: pin
# sim_intercept = 0, sim_log_slope = 0 AND scatter_std = 0 (the simulator's
# alpha / beta / scatter config fields). With all three zero the ratio law is
# log A2 = log A1 exactly (A2 == A1, no per-spot jitter), so every spot in a cell
# shares one intensity -> one peak -> one SNR (Change 4). The family isolates
# difficulty and carries no curvature (true_alpha == 0 everywhere).
_NEUTRAL_RATIO_LAW = {
    "ratio_law": {
        "alpha": {"min": 0.0, "max": 0.0},          # sim_intercept = 0
        "beta": {"min": 0.0, "max": 0.0},           # sim_log_slope = 0 -> true_alpha = 0
        "scatter_std": {"min": 0.0, "max": 0.0},    # NO jitter -> A2 == A1 exactly
    }
}


# --------------------------------------------------------------------------- #
# Family 1 -- TRUE constant-SNR: invert the frozen SNR to solve for intensity  #
# --------------------------------------------------------------------------- #
def _bench_axis_params(base_config: dict) -> DetectorAxisParams:
    """The frozen ``_features`` SNR-axis params at the FIXED benchmark point.

    Uses the benchmark's fixed PSF (``sigma1``/``sigma2``), the constant flat
    background, and the MEASURED per-channel gain / read-noise floor / ``n_frames``
    from the (already override-merged) base config detector block. This is exactly
    the vendored ``peak_snr`` configuration, so solving against it inverts the
    SAME SNR the downstream evaluator reports (see ``docs/snr_convention.md``).
    """
    det = base_config.get("detector", {})
    ch1, ch2 = det.get("ch1", {}), det.get("ch2", {})
    return DetectorAxisParams(
        sigma1=_BENCH_SIGMA1, sigma2=_BENCH_SIGMA2,
        bg1=_BENCH_BACKGROUND_PHOTONS, bg2=_BENCH_BACKGROUND_PHOTONS,
        gain1=float(ch1.get("gain", 1.0)), gain2=float(ch2.get("gain", 1.0)),
        floor1=float(ch1.get("noise_floor_sigma", 0.0)),
        floor2=float(ch2.get("noise_floor_sigma", 0.0)),
        n_frames=int(det.get("n_frames", 3)),
    )


def _solve_intensity_for_snr(target_snr: float, axis: DetectorAxisParams) -> dict:
    """Solve for the single spot intensity ``A`` whose limiting-channel SNR == target.

    Family 1 pins a neutral, zero-scatter ratio law, so ``A2 == A1 == A`` exactly
    for every spot and the per-spot scalar SNR is ``min(snr1(A), snr2(A))`` under
    the frozen ``_features`` definition. That composite is strictly increasing in
    ``A``, so bisection in ``log A`` recovers the unique intensity (Change 4). No
    closed form is assumed -- ``min()`` over two channels rarely has one -- so we
    solve numerically. Returns the solved ``A`` (photons), the per-channel SNRs at
    ``A``, and which channel limits.
    """
    if target_snr <= 0:
        raise ValueError(f"target SNR must be positive, got {target_snr}")

    def _snr(A: float) -> float:
        la = math.log(A)
        r = peak_snr(np.array([la]), np.array([la]), axis)
        return float(r["snr"][0])

    lo, hi = 1e-3, 1e3
    while _snr(hi) < target_snr:
        hi *= 10.0
        if hi > 1e18:
            raise RuntimeError(f"SNR target {target_snr} unreachable below A=1e18 photons")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if _snr(mid) < target_snr:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-13:
            break
    A = math.sqrt(lo * hi)
    la = math.log(A)
    r = peak_snr(np.array([la]), np.array([la]), axis)
    snr1, snr2 = float(r["snr1"][0]), float(r["snr2"][0])
    return {
        "intensity": float(A),
        "snr1": snr1,
        "snr2": snr2,
        "limiting_channel": (1 if snr1 <= snr2 else 2),
        "realised_snr": float(min(snr1, snr2)),
    }


def _channel_saturates(A: float, gain: float, knee: float, peak_fraction: float) -> tuple[bool, float]:
    """Does a spot of intensity ``A`` reach a channel's saturation knee?

    Mirrors the vendored ``simulate_image`` saturation flag: gained peak
    ``M = gain * (A * peak_fraction + background)`` and the spot saturates when
    ``M >= knee``. Returns ``(saturates, gained_peak_ADU)``.
    """
    gained_peak = gain * (A * peak_fraction + _BENCH_BACKGROUND_PHOTONS)
    return bool(gained_peak >= knee), float(gained_peak)


def _build_solved_intensity_table(base_config: dict, snr_targets) -> list[dict]:
    """One row per SNR target: target -> solved A (per channel) + OOD/saturation (Change 6).

    Because family 1 pins ``A2 == A1 == A``, the solved intensity is identical in
    both channels; both are reported for clarity. A row whose solved ``A`` falls
    outside the trained ``[20, 7943]`` photon range is flagged OUT-OF-DISTRIBUTION.

    We ALSO flag whether the solved spot saturates each channel's 12-bit ADC. This
    matters because of the protein (ch2) gain: the ch2 peak pixel reaches the ADC
    ceiling at a finite SNR (~16.8 at the chosen gain 40), so any SNR target above
    that yields a CLIPPED protein image even when the intensity is in-distribution.
    """
    axis = _bench_axis_params(base_config)
    det = base_config.get("detector", {})
    ch1, ch2 = det.get("ch1", {}), det.get("ch2", {})
    gain1, gain2 = float(ch1.get("gain", 1.0)), float(ch2.get("gain", 1.0))
    knee1 = float(ch1.get("saturation_knee", math.inf))
    knee2 = float(ch2.get("saturation_knee", math.inf))
    pf1, pf2 = psf.gaussian_peak_fraction(_BENCH_SIGMA1), psf.gaussian_peak_fraction(_BENCH_SIGMA2)

    table: list[dict] = []
    for target in snr_targets:
        solved = _solve_intensity_for_snr(float(target), axis)
        A = solved["intensity"]
        in_dist = _LEGACY_A1_MIN <= A <= _LEGACY_A1_MAX
        sat1, gp1 = _channel_saturates(A, gain1, knee1, pf1)
        sat2, gp2 = _channel_saturates(A, gain2, knee2, pf2)
        flags = []
        if not in_dist:
            flags.append(f"solved A={A:.1f} photons is out-of-range for the LEGACY checkpoints "
                         f"(legacy-trained [{_LEGACY_A1_MIN:g}, {_LEGACY_A1_MAX:g}]; retrain pending)")
        if sat2:
            flags.append(f"protein (ch2) SATURATES: gained peak {gp2:.0f} >= knee {knee2:g} ADU "
                         f"-> clipped protein image")
        if sat1:
            flags.append(f"lipid (ch1) SATURATES: gained peak {gp1:.0f} >= knee {knee1:g} ADU")
        table.append({
            "target_snr": float(target),
            "solved_intensity_photons": A,
            "solved_A1_photons": A,          # ch1 == ch2 (neutral zero-scatter ratio law)
            "solved_A2_photons": A,
            "snr1_at_A": solved["snr1"],
            "snr2_at_A": solved["snr2"],
            "limiting_channel": solved["limiting_channel"],
            "in_legacy_training_distribution": bool(in_dist),
            "ch1_gained_peak_adu": gp1,
            "ch2_gained_peak_adu": gp2,
            # Headroom to the ADC ceiling: how far the peak pixel sits below the knee
            # (positive == unclipped). ch2 (protein) is the binding channel.
            "ch1_headroom_adu": float(knee1 - gp1),
            "ch2_headroom_adu": float(knee2 - gp2),
            "ch1_saturates": sat1,
            "ch2_saturates": sat2,
            "flag": ("; ".join(flags) if flags else None),
        })
    return table


# --------------------------------------------------------------------------- #
# Family 2 -- curvature operating point                                       #
# --------------------------------------------------------------------------- #
# The EASIEST grid cell: high SNR (bright, low background, moderately narrow PSF)
# and low density (sparse, uniform) so slope recovery is tested in isolation from
# detection failure. The A1 window is NOT fixed across the sweep: it is sized
# PER SET (see ``_curvature_intensity_window``) so the brightest ch2 spot stays
# below the detector saturation knee even at steep pinned |slope|, while still
# spanning >= the minimum A1 spread a slope fit needs. ``dim_bias`` and the
# window WIDTH stay constant across the sweep; only its ceiling shifts with the
# slope. Ratio-law scatter is pinned for the whole family.
_CURVATURE_SCATTER_STD = 0.08          # per-spot log-ratio scatter (natural log)

# A1-window sizing knobs. The brightest realised A1 approaches ``10**log10_max``;
# with the pinned slope its ch2 partner is A2 = A1^(1+slope) (natural-log
# intercept 0) inflated by the upper tail of the log-ratio scatter. We choose
# ``log10_max`` so this worst-case spot's GAINED peak sits at ``_SAT_TARGET_FRAC``
# of the knee -- comfortably below the ``>= knee`` saturation flag, with headroom
# for detector jitter, neighbour overlap and rounding.
_CURV_SAT_TARGET_FRAC = 0.75           # brightest gained peak <= this * saturation_knee
_CURV_SCATTER_SIGMAS = 3.5             # cover the upper tail of the log-ratio scatter
_CURV_WINDOW_MARGIN = 0.4              # window width = min_alpha_decades + this (>= spread thresh)
_CURV_SAT_WARN_FRAC = 0.01             # warn if a set's saturated fraction exceeds this

_CURVATURE_OPERATING_POINT = {
    # NB: no "intensity" here -- the A1 window is computed per set for saturation
    # safety (see _curvature_intensity_window) and merged in by the family driver.
    # Adopts the benchmark-wide constants (Change 5): FIXED PSF (sigma1=1.4,
    # sigma2=1.68) and CONSTANT 2-photon background (gradient == structure == 0),
    # merged in via _FIXED_PSF / _CONSTANT_BACKGROUND by the family driver.
    "density": {"min": 0.0004, "max": 0.0015}, "oversample_dense_fraction": 0.0,
    "clustering": {"cluster_prob": 0.0},
}


def _curvature_intensity_window(
    base_config: dict, sim_log_slope: float, min_alpha_decades: float,
) -> dict:
    """A1 intensity window (log10 photons) sized to avoid ch1/ch2 saturation.

    Solves for the brightest A1 whose worst-case gained peak stays at
    ``_CURV_SAT_TARGET_FRAC`` of each channel's saturation knee, then places a
    fixed-width window below it. ch1 sees A1 directly; ch2 sees
    ``A2 = A1**(1 + sim_log_slope)`` (intercept 0) times the upper scatter tail,
    so for steep +slope ch2 binds and the ceiling drops. Uses the benchmark's
    FIXED PSF (``_BENCH_SIGMA1``/``_BENCH_SIGMA2``) and CONSTANT background
    (``_BENCH_BACKGROUND_PHOTONS``) -- there is no per-image sigma/background
    randomisation in the benchmark, so there is no "worst case" range to bound.
    Reads the vendored detector CONFIG (``jitter_frac == 0`` in our configs, so
    the sampled knees equal these); the ``_SAT_TARGET_FRAC`` headroom absorbs any
    narrow jitter. Returns an ``intensity`` override dict; edits nothing vendored.
    """
    det = base_config.get("detector", {})
    ch1, ch2 = det.get("ch1", {}), det.get("ch2", {})
    gain1 = float(ch1.get("gain", 1.0))
    gain2 = float(ch2.get("gain", 1.0))
    knee1 = float(ch1.get("saturation_knee", math.inf))
    knee2 = float(ch2.get("saturation_knee", math.inf))

    pf1 = psf.gaussian_peak_fraction(_BENCH_SIGMA1)
    pf2 = psf.gaussian_peak_fraction(_BENCH_SIGMA2)

    bg_max = _BENCH_BACKGROUND_PHOTONS
    scatter_factor = math.exp(_CURV_SCATTER_SIGMAS * _CURVATURE_SCATTER_STD)

    # ch1 ceiling: A1 itself (no ratio-law scatter on ch1).
    a1_cap_ch1 = max((_CURV_SAT_TARGET_FRAC * knee1) / gain1 - bg_max, 1.0) / pf1
    # ch2 ceiling: A2 = A1**(1+slope), plus upward scatter tail.
    a2_cap = max((_CURV_SAT_TARGET_FRAC * knee2) / gain2 - bg_max, 1.0) / pf2
    a2_cap_no_scatter = a2_cap / scatter_factor
    exponent = 1.0 + float(sim_log_slope)
    a1_cap_ch2 = a2_cap_no_scatter ** (1.0 / exponent) if exponent > 0 else math.inf

    a1_cap = min(a1_cap_ch1, a1_cap_ch2)
    log10_max = math.log10(max(a1_cap, 10.0))
    window_decades = max(float(min_alpha_decades) + _CURV_WINDOW_MARGIN, 1.4)
    log10_min = log10_max - window_decades
    return {"log10_min": float(log10_min), "log10_max": float(log10_max),
            "decades": float(window_decades), "dim_bias": 1.0}


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters for a two-family benchmark generation run.

    ``snr_targets`` are TRUE constant-SNR levels (Change 4): for each, the single
    spot intensity is solved by inverting the frozen ``_features`` SNR, and a cell
    is labelled by its target SNR and its area density in spots/px.
    ``density_levels`` are constant area densities (spots per pixel): each is a
    cell knob AND its label -- no neighbour-count binning, no clustering. Every
    density level is generated at every SNR level (full orthogonal grid), and SNR
    and density are orthogonal (intensity is solved from SNR alone).
    """

    seed: int = 0
    height: int = 256
    width: int = 256
    density_radius_px: float = 4.0       # informational realised-neighbour stat only

    # Family 1
    # TRUE constant-SNR target levels (min(snr1, snr2) per the frozen definition).
    # The 0-edge (unsolvable) and inf-edge of the old bin scheme are dropped; a
    # cell is a point target, not a half-open bin. CAPPED below the protein-ADC
    # clip: at the chosen protein gain 40 the ch2 ADC clips near SNR ~= 16.8, so
    # the grid stays under that; generation ASSERTS no cell clips either channel
    # (a clipping target fails loud).
    snr_targets: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 10.0, 15.0)
    # Constant AREA densities (spots/px): sweep the training range [0.0006, 0.024].
    # Each is used as both the knob and label. The top level 0.02 (~1310 spots per
    # 256^2 image) is the crowding stress point; it is INSIDE the current trained
    # range but ABOVE the legacy checkpoints' max (0.012).
    density_levels: tuple[float, ...] = (0.0006, 0.002, 0.006, 0.012, 0.015, 0.02)
    images_per_cell: int = 30

    # Family 2
    alpha_values: tuple[float, ...] = (
        -1.2, -0.9, -0.6, -0.3, -0.15, -0.075, 0.0, 0.075, 0.15, 0.3, 0.6, 0.9, 1.2,
    )
    images_per_alpha: int = 20
    null_control_alpha: float = 0.0
    null_control_multiplier: int = 3
    min_alpha_decades: float = 1.0

    def __post_init__(self) -> None:
        if len(self.snr_targets) < 1:
            raise ValueError("snr_targets must have at least one target SNR")
        if any(s <= 0.0 or math.isinf(s) for s in self.snr_targets):
            raise ValueError("snr_targets must be positive and finite (they are solved for)")
        if len(self.density_levels) < 1:
            raise ValueError("density_levels must have at least one area density")
        if any(d <= 0.0 for d in self.density_levels):
            raise ValueError("density_levels must be positive (spots per pixel)")
        if self.images_per_cell < 1 or self.images_per_alpha < 1:
            raise ValueError("images_per_cell and images_per_alpha must be >= 1")


# --------------------------------------------------------------------------- #
# One set (== one condition == one directory)                                 #
# --------------------------------------------------------------------------- #
def _generate_set(
    *,
    base_config: dict,
    scene_override: dict,
    set_dir: Path,
    set_seed: int,
    n_images: int,
    id_prefix: str,
    density_radius_px: float,
) -> dict:
    """Generate one homogeneous set: images/, ground_truth/, and a stats block.

    Detector is the fixed instrument, sampled once from ``set_seed``. Every image
    reuses it; each image's scene is drawn from ``scene_override`` deep-merged onto
    the base scene. Returns a dict of per-image records + realised SNR/density
    arrays for the caller to fold into ``meta.json``.
    """
    (set_dir / "images").mkdir(parents=True, exist_ok=True)
    (set_dir / "ground_truth").mkdir(parents=True, exist_ok=True)

    img_cfg = base_config.get("image", {})
    shape = (int(img_cfg.get("height", 256)), int(img_cfg.get("width", 256)))
    scene_cfg = _deep_merge(base_config.get("scene", {}), scene_override)

    root = np.random.SeedSequence(int(set_seed))
    det_seq, img_seq = root.spawn(2)
    detector = noise.sample_detector_params(
        base_config.get("detector", {}), np.random.default_rng(det_seq))
    image_seeds = img_seq.spawn(int(n_images))

    per_image: list[dict] = []
    all_snr: list[np.ndarray] = []
    all_neighbors: list[np.ndarray] = []
    all_log10_a1: list[np.ndarray] = []
    all_log10_a2: list[np.ndarray] = []
    n_spots_total = 0
    n_saturated_total = 0

    for i, child in enumerate(image_seeds):
        image_id = f"{id_prefix}_{i:05d}"
        rng = np.random.default_rng(child)
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sim = forward_model.simulate_image(
            image_id=image_id, shape=shape, scene=scene, detector=detector, rng=rng)

        # images/  -- raw observed counts, uint16 [2,H,W] stacked TIFF (2 pages).
        tifffile.imwrite(set_dir / "images" / f"image_{image_id}.tif", sim.image)
        # ground_truth/  -- frozen-schema GT rows (positions + true logI/I).
        write_spots(sim.spots, set_dir / "ground_truth" / f"gt_{image_id}.csv")

        gt = sim.spots
        log_a1 = gt["logI1"].to_numpy(float)
        log_a2 = gt["logI2"].to_numpy(float)
        # Realised difficulty via the FROZEN _features definitions (nominal label
        # is a target; this is what the spots actually are).
        snr = peak_snr(log_a1, log_a2, axis_params_from_meta(sim.meta))
        nbr = local_neighbor_count(gt["x"].to_numpy(float), gt["y"].to_numpy(float),
                                   density_radius_px)

        gts = sim.meta["ground_truth_sigma"]
        per_image.append({
            "image_id": image_id,
            "image_file": f"images/image_{image_id}.tif",
            "ground_truth_file": f"ground_truth/gt_{image_id}.csv",
            "n_spots": int(sim.meta["n_spots"]),
            "n_saturated": int(sim.meta["n_saturated"]),
            # realised area density (spots/px) this image was drawn at. For family-1
            # cells this is pinned constant across the cell (min==max density draw).
            "area_density": float(scene.density),
            # per-image, per-channel true PSF width -- the required sigma plumbing.
            "ground_truth_sigma": {"sigma1": float(gts["sigma1"]),
                                   "sigma2": float(gts["sigma2"])},
            "background_level": {
                "ch1": float(scene.background1.get("level", math.nan)),
                "ch2": float(scene.background2.get("level", math.nan)),
            },
            "snr_median": (float(np.median(snr["snr"])) if snr["snr"].size else None),
        })
        if log_a1.size:
            all_snr.append(snr["snr"])
            all_neighbors.append(nbr)
            all_log10_a1.append(log_a1 / math.log(10.0))
            all_log10_a2.append(log_a2 / math.log(10.0))
        n_spots_total += int(sim.meta["n_spots"])
        n_saturated_total += int(sim.meta["n_saturated"])

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.empty(0)

    return {
        "shape": [shape[0], shape[1]],
        "detector": forward_model._detector_to_meta(detector),
        "scene_config": scene_cfg,
        "per_image": per_image,
        "n_images": int(n_images),
        "n_spots_total": n_spots_total,
        "n_saturated_total": n_saturated_total,
        "snr": _cat(all_snr),
        "n_neighbors": _cat(all_neighbors),
        "log10_a1": _cat(all_log10_a1),
        "log10_a2": _cat(all_log10_a2),
    }


# --------------------------------------------------------------------------- #
# Family 1 -- SNR x density                                                    #
# --------------------------------------------------------------------------- #
def generate_snr_density_family(
    base_config: dict, cfg: BenchmarkConfig, family_root: Path, *, log_fn=print,
) -> dict:
    """Generate the SNR x density family (TRUE constant-SNR cells). Summary dict.

    Each cell is a TRUE constant-SNR condition (Change 4): given the FIXED PSF,
    the CONSTANT 2-photon background and the MEASURED per-channel detector, the
    single spot intensity ``A`` is solved by inverting the frozen ``_features``
    SNR so ``min(snr1(A), snr2(A)) == target``. Every spot in the cell gets that
    SAME intensity (neutral zero-scatter ratio law -> ``A2 == A1 == A``, no
    jitter), so the realised per-spot SNR has ~zero spread and equals the target.
    SNR and density stay ORTHOGONAL: intensity is solved from SNR alone, never
    coupled to the cell's density.
    """
    family_root.mkdir(parents=True, exist_ok=True)
    axis = _bench_axis_params(base_config)
    cells = []
    warnings: list[str] = []
    t0 = time.perf_counter()

    det = base_config.get("detector", {})
    _g1 = float(det.get("ch1", {}).get("gain", 1.0))
    _k1 = float(det.get("ch1", {}).get("saturation_knee", math.inf))
    _pf1 = psf.gaussian_peak_fraction(_BENCH_SIGMA1)
    _g2 = float(det.get("ch2", {}).get("gain", 1.0))
    _k2 = float(det.get("ch2", {}).get("saturation_knee", math.inf))
    _pf2 = psf.gaussian_peak_fraction(_BENCH_SIGMA2)

    for si, target_snr in enumerate(cfg.snr_targets):
        # Solve ONCE per SNR target (density-independent -> orthogonal grid).
        solved = _solve_intensity_for_snr(float(target_snr), axis)
        A = solved["intensity"]
        log10_A = math.log10(A)
        snr_in_dist = _LEGACY_A1_MIN <= A <= _LEGACY_A1_MAX

        # NO-CLIP GUARANTEE (Change/decision 1): the SNR grid is capped so every
        # cell stays below BOTH channels' ADC knee. Assert it and FAIL LOUD rather
        # than ship a clipped cell -- clipped intensity is unrecoverable, so such a
        # cell would measure nothing about method quality.
        ch1_saturates, ch1_gained_peak = _channel_saturates(A, _g1, _k1, _pf1)
        ch2_saturates, ch2_gained_peak = _channel_saturates(A, _g2, _k2, _pf2)
        if ch1_saturates or ch2_saturates:
            raise ValueError(
                f"snr_density SNR={target_snr:g}: solved A={A:.1f} photons CLIPS a channel "
                f"(ch1 gained peak {ch1_gained_peak:.0f} vs knee {_k1:g}; ch2 gained peak "
                f"{ch2_gained_peak:.0f} vs knee {_k2:g} ADU). Clipped intensity is "
                f"unrecoverable -- cap snr_targets so every cell stays below both knees. "
                f"At the chosen protein gain 40 the ch2 ADC clips near SNR ~= 16.8.")

        if not snr_in_dist:
            msg = (f"[warn] snr_density SNR={target_snr:g}: solved intensity A={A:.1f} "
                   f"photons is out-of-range for the LEGACY checkpoints (legacy-trained "
                   f"[{_LEGACY_A1_MIN:g}, {_LEGACY_A1_MAX:g}]) -- coverage artifact, not a "
                   f"method difference; {_RETRAIN_NOTE}")
            warnings.append(msg)
            log_fn(msg)

        for di, density in enumerate(cfg.density_levels):
            s_lbl = _edge_label(float(target_snr))
            d_lbl = _edge_label(density)                 # spots/px, e.g. "0.006"
            label = f"snr={s_lbl}_density={d_lbl}"
            set_dir = family_root / label
            set_seed = _set_seed(cfg.seed, "snr_density", label)
            # Stress = above the CURRENT training density max (0.024). Also track
            # "above the LEGACY max (0.012)" separately, since the legacy checkpoints
            # never saw >0.012 and degrade there as a coverage artifact.
            density_is_stress = density > _TRAINED_DENSITY_MAX
            density_above_legacy = density > _LEGACY_DENSITY_MAX

            # FIXED PSF + CONSTANT background + solved point intensity (window
            # pinned to a single value: log10_min == log10_max -> every spot gets
            # exactly A) + constant area density (uniform, no clustering) + the
            # NEUTRAL zero-scatter ratio law (A2 == A1, no jitter).
            override = _deep_merge(_FIXED_PSF, _CONSTANT_BACKGROUND)
            override = _deep_merge(override, _ZERO_REGISTRATION)
            override = _deep_merge(override, {
                "intensity": {"log10_min": log10_A, "log10_max": log10_A, "dim_bias": 1.0}})
            override = _deep_merge(override, _constant_density_override(density))
            override = _deep_merge(override, _NEUTRAL_RATIO_LAW)

            res = _generate_set(
                base_config=base_config, scene_override=override, set_dir=set_dir,
                set_seed=set_seed, n_images=cfg.images_per_cell, id_prefix=label,
                density_radius_px=cfg.density_radius_px)

            # Constant-per-cell invariant: min==max density draw -> identical area
            # density (and n_spots) for every image. Assert it, and record it.
            realised_dens = [rec["area_density"] for rec in res["per_image"]]
            constant_density = (len(set(realised_dens)) <= 1)
            assert constant_density and (not realised_dens
                                         or math.isclose(realised_dens[0], density, rel_tol=1e-9)), (
                f"{label}: area density not constant/at target: {sorted(set(realised_dens))} "
                f"vs {density}")

            # TRUE constant-SNR invariant: realised per-spot SNR has ~zero spread
            # and equals the target (Change 4 sanity test).
            snr_arr = res["snr"]
            snr_spread = (float(np.max(snr_arr)) - float(np.min(snr_arr))) if snr_arr.size else 0.0
            realised_snr_median = float(np.median(snr_arr)) if snr_arr.size else None
            assert snr_spread <= 1e-6, (
                f"{label}: realised per-spot SNR spread {snr_spread:.3e} not ~0 "
                f"(constant-SNR cell must have identical per-spot SNR)")
            assert (realised_snr_median is None
                    or math.isclose(realised_snr_median, float(target_snr),
                                    rel_tol=1e-6, abs_tol=1e-6)), (
                f"{label}: realised SNR median {realised_snr_median} != target {target_snr}")

            meta = {
                "family": "snr_density",
                "label": label,
                "condition": {
                    "target_snr": float(target_snr),
                    "snr_index": si,
                    "area_density_spots_per_px": float(density),
                    "density_index": di,
                    "placement": "uniform_random (no clustering)",
                },
                "target_snr": float(target_snr),
                "solved_intensity_photons": A,
                "solved_A1_photons": A,          # ch1 == ch2 (neutral zero-scatter ratio law)
                "solved_A2_photons": A,
                "solved_snr1_at_A": solved["snr1"],
                "solved_snr2_at_A": solved["snr2"],
                "limiting_channel": solved["limiting_channel"],
                "snr_in_legacy_training_distribution": bool(snr_in_dist),
                "protein_channel_saturates": bool(ch2_saturates),
                "protein_channel_gained_peak_adu": float(ch2_gained_peak),
                "realised_snr_spread": float(snr_spread),
                "area_density_spots_per_px": float(density),
                "area_density_constant_per_cell": bool(constant_density),
                "density_is_stress": bool(density_is_stress),
                "density_above_legacy_training_max": bool(density_above_legacy),
                "note": ("TRUE constant-SNR cell: the single spot intensity A was solved by "
                         "inverting the frozen _features SNR given the FIXED PSF (sigma1=1.4, "
                         "sigma2=1.68), the CONSTANT 2-photon background and the MEASURED "
                         "per-channel detector, so every spot has identical A (A2==A1, no "
                         "jitter) -> identical peak -> identical SNR (~zero spread, equals "
                         "target). Area density is an exact per-cell constant; spots are "
                         "placed uniformly at random with no clustering. SNR and density "
                         "are orthogonal (intensity solved from SNR only)."),
                "ratio_law": "neutral zero-scatter (sim_intercept=0, sim_log_slope=0, "
                             "scatter=0 -> A2==A1, true_alpha=0)",
                "true_alpha": 0.0,
                "sigma1": _BENCH_SIGMA1,
                "sigma2": _BENCH_SIGMA2,
                "background_photons": _BENCH_BACKGROUND_PHOTONS,
                "n_images": res["n_images"],
                "n_spots_total": res["n_spots_total"],
                "n_saturated_total": res["n_saturated_total"],
                "seed": set_seed,
                "master_seed": cfg.seed,
                "shape": res["shape"],
                "density_radius_px": cfg.density_radius_px,
                "realised_snr": _summary(res["snr"]),
                # informational only: local crowding at this uniform area density.
                "realised_n_neighbors": _summary(res["n_neighbors"]),
                "realised_log10_A1": _summary(res["log10_a1"]),
                "detector": res["detector"],
                "scene_config": res["scene_config"],
                "images": res["per_image"],
            }
            _write_json(set_dir / "meta.json", meta)
            cells.append({"label": label, "snr_index": si, "density_index": di,
                          "target_snr": float(target_snr),
                          "solved_intensity_photons": A,
                          "snr_in_legacy_training_distribution": bool(snr_in_dist),
                          "protein_channel_saturates": bool(ch2_saturates),
                          "density_is_stress": bool(density_is_stress),
                          "density_above_legacy_training_max": bool(density_above_legacy),
                          "area_density_spots_per_px": float(density),
                          "n_images": res["n_images"], "n_spots": res["n_spots_total"],
                          "seed": set_seed,
                          "realised_snr_median": realised_snr_median,
                          "realised_snr_spread": float(snr_spread),
                          "realised_neighbors_mean": (float(np.mean(res["n_neighbors"]))
                                                      if res["n_neighbors"].size else None)})
            ood = "  [OOD]" if not snr_in_dist else ""
            stress = "  [density-stress]" if density_is_stress else ""
            log_fn(f"  [snr_density] {label:>24}: {res['n_images']} imgs, "
                   f"{res['n_spots_total']:>6} spots, SNR={target_snr:g} (A={A:.0f} ph), "
                   f"density={density:g} spots/px{ood}{stress}")

    dt = time.perf_counter() - t0
    n_images = sum(c["n_images"] for c in cells)
    n_spots = sum(c["n_spots"] for c in cells)
    solved_table = _build_solved_intensity_table(base_config, cfg.snr_targets)
    return {"n_cells": len(cells), "n_images": n_images, "n_spots": n_spots,
            "seconds": dt, "cells": cells, "solved_intensity_table": solved_table,
            "warnings": warnings}


# --------------------------------------------------------------------------- #
# Family 2 -- curvature                                                        #
# --------------------------------------------------------------------------- #
def generate_curvature_family(
    base_config: dict, cfg: BenchmarkConfig, family_root: Path, *, log_fn=print,
) -> dict:
    """Generate the curvature (alpha-recovery) family. Returns a summary dict."""
    family_root.mkdir(parents=True, exist_ok=True)
    sets = []
    warnings: list[str] = []
    t0 = time.perf_counter()

    for true_alpha in cfg.alpha_values:
        is_null = math.isclose(true_alpha, cfg.null_control_alpha, abs_tol=1e-12)
        n_images = (cfg.images_per_alpha * cfg.null_control_multiplier
                    if is_null else cfg.images_per_alpha)

        # PIN the simulator slope so the set realises this physical true_alpha.
        # The factor of 2 comes from alpha_to_sim_slope (the ONE place it lives).
        sim_log_slope = alpha_to_sim_slope(true_alpha)
        label = f"alpha={_edge_label(true_alpha)}"
        set_dir = family_root / label
        set_seed = _set_seed(cfg.seed, "curvature", label)

        # Size the A1 window PER SET so the brightest ch2 spot clears the knee at
        # this pinned slope (Change 2), while keeping >= the minimum A1 spread.
        intensity_window = _curvature_intensity_window(
            base_config, sim_log_slope, cfg.min_alpha_decades)
        # Operating point + benchmark-wide FIXED PSF and CONSTANT background
        # (Change 5 adopts Change 2/3), then the per-set intensity window and the
        # pinned ratio law. WIDE A1 spread is preserved (window >= min decades).
        override = _deep_merge(_CURVATURE_OPERATING_POINT, _FIXED_PSF)
        override = _deep_merge(override, _CONSTANT_BACKGROUND)
        override = _deep_merge(override, _ZERO_REGISTRATION)
        override = _deep_merge(override, {
            "intensity": {"log10_min": intensity_window["log10_min"],
                          "log10_max": intensity_window["log10_max"],
                          "dim_bias": intensity_window["dim_bias"]},
            "ratio_law": {
                "alpha": {"min": 0.0, "max": 0.0},                       # sim_intercept = 0
                "beta": {"min": sim_log_slope, "max": sim_log_slope},    # sim_log_slope pinned
                "scatter_std": {"min": _CURVATURE_SCATTER_STD, "max": _CURVATURE_SCATTER_STD},
            }})

        res = _generate_set(
            base_config=base_config, scene_override=override, set_dir=set_dir,
            set_seed=set_seed, n_images=n_images, id_prefix=label,
            density_radius_px=cfg.density_radius_px)

        # A1 spread: assert it clears the minimum decades threshold, else WARN.
        a1_summary = _summary(res["log10_a1"])
        spread = (a1_summary["max"] - a1_summary["min"]) if a1_summary["n"] else 0.0
        spread_ok = spread >= cfg.min_alpha_decades
        if not spread_ok:
            msg = (f"[warn] curvature set {label}: A1 spread {spread:.2f} decades "
                   f"< min {cfg.min_alpha_decades} -- slope fit will be ill-conditioned.")
            warnings.append(msg)
            log_fn(msg)

        # Saturation should be ~0 everywhere now the A1 window is sized per set
        # (Change 2). Fraction of spots reaching EITHER channel's knee; ch2 is the
        # channel that saturates at steep +alpha, hence the field name. WARN loudly
        # if any survives so an affected set is never shipped silently.
        ch2_sat_frac = (res["n_saturated_total"] / res["n_spots_total"]
                        if res["n_spots_total"] else 0.0)
        if ch2_sat_frac > _CURV_SAT_WARN_FRAC:
            msg = (f"[warn] curvature set {label}: ch2_saturated_fraction {ch2_sat_frac:.3f} "
                   f"> {_CURV_SAT_WARN_FRAC} despite per-set A1-window sizing -- clipped ch2 "
                   f"intensities would bias recovered alpha at the tails.")
            warnings.append(msg)
            log_fn(msg)

        # A1 coverage vs the trained range (docs/training_distribution.md). The
        # protein gain (chosen 40 ADU/photon) caps ch2 near ~4095/40 ~= 100
        # photons/pixel, so the saturation-safe per-set A1 window sits at low
        # absolute photon counts and can dip below the trained A1 min. Record and
        # WARN so out-of-range degradation is read as a coverage artifact, not a
        # method difference (Change 6, extended to the curvature family).
        a1_min_photons = (10.0 ** a1_summary["min"]) if a1_summary["n"] else None
        a1_max_photons = (10.0 ** a1_summary["max"]) if a1_summary["n"] else None
        a1_in_dist = bool(a1_min_photons is not None
                          and a1_min_photons >= _LEGACY_A1_MIN
                          and a1_max_photons <= _LEGACY_A1_MAX)
        if a1_min_photons is not None and not a1_in_dist:
            msg = (f"[warn] curvature set {label}: realised A1 range "
                   f"[{a1_min_photons:.0f}, {a1_max_photons:.0f}] photons extends outside "
                   f"the legacy-trained [{_LEGACY_A1_MIN:g}, {_LEGACY_A1_MAX:g}] -- degradation "
                   f"at the out-of-range end is a coverage artifact, not a method difference.")
            warnings.append(msg)
            log_fn(msg)

        # Sanity wiring, recorded in meta: true_alpha == 2 * sim_log_slope exactly.
        assert math.isclose(true_alpha, sim_slope_to_alpha(sim_log_slope), abs_tol=1e-12)

        meta = {
            "family": "curvature",
            "label": label,
            "true_alpha": float(true_alpha),
            "sim_log_slope": float(sim_log_slope),
            "sim_intercept": 0.0,
            "alpha_convention": "true_alpha = 2 * sim_log_slope  (see benchmark.alpha)",
            "null_control": bool(is_null),
            "null_control_note": (
                "alpha=0 null control: extra images for tight error bars. A method that "
                "manufactures curvature from size-dependent intensity bias fails HERE."
                if is_null else None),
            "operating_point": ("easiest cell: high SNR (bright, low background, narrow "
                                "PSF) x low density (sparse, uniform) -- slope recovery "
                                "isolated from detection failure"),
            "n_images": res["n_images"],
            "n_spots_total": res["n_spots_total"],
            "seed": set_seed,
            "master_seed": cfg.seed,
            "shape": res["shape"],
            "density_radius_px": cfg.density_radius_px,
            "a1_spread_decades": float(spread),
            "a1_spread_ok": bool(spread_ok),
            "min_alpha_decades": cfg.min_alpha_decades,
            "intensity_window": intensity_window,
            "a1_range_photons": ([a1_min_photons, a1_max_photons]
                                 if a1_min_photons is not None else None),
            "a1_in_legacy_training_distribution": a1_in_dist,
            "a1_coverage_note": (
                "The protein (ch2) gain (chosen 40) is ~6x the lipid gain, so ch2 hits the "
                "12-bit ADC ceiling at a lower photon count than ch1; the saturation-safe A1 window "
                "therefore sits at low absolute intensities and may extend below the LEGACY "
                "A1 min (20 ph). This is expected and correct -- the wide A1 spread is required "
                "to fit the slope, and low intensity is a genuine consequence of the measured "
                f"detector. {_RETRAIN_NOTE} See a1_range_photons / a1_in_legacy_training_distribution."),
            "realised_log10_A1": a1_summary,
            "realised_log10_A2": _summary(res["log10_a2"]),
            "realised_snr": _summary(res["snr"]),
            "ch2_saturated_fraction": float(ch2_sat_frac),
            "ch2_saturation_note": ("A1 window sized PER SET (see intensity_window) so the "
                                    "brightest ch2 = A1^(1+sim_log_slope) spot's gained peak "
                                    f"stays at <= {_CURV_SAT_TARGET_FRAC} of the ch2 knee even "
                                    "at steep +alpha; saturated fraction is ~0 by construction, "
                                    "while the A1 spread is preserved for the slope fit."),
            "detector": res["detector"],
            "scene_config": res["scene_config"],
            "images": res["per_image"],
        }
        _write_json(set_dir / "meta.json", meta)
        sets.append({"label": label, "true_alpha": float(true_alpha),
                     "sim_log_slope": float(sim_log_slope), "null_control": bool(is_null),
                     "n_images": res["n_images"], "n_spots": res["n_spots_total"],
                     "seed": set_seed, "a1_spread_decades": float(spread),
                     "a1_spread_ok": bool(spread_ok),
                     "a1_in_legacy_training_distribution": a1_in_dist,
                     "ch2_saturated_fraction": float(ch2_sat_frac)})
        flag = "  [NULL CONTROL]" if is_null else ""
        log_fn(f"  [curvature] {label:>14}: alpha={true_alpha:+.3f} "
               f"sim_log_slope={sim_log_slope:+.3f}, {res['n_images']} imgs, "
               f"{res['n_spots_total']:>6} spots, spread={spread:.2f}dec{flag}")

    dt = time.perf_counter() - t0
    n_images = sum(s["n_images"] for s in sets)
    n_spots = sum(s["n_spots"] for s in sets)
    return {"n_sets": len(sets), "n_images": n_images, "n_spots": n_spots,
            "seconds": dt, "sets": sets, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Top-level driver                                                             #
# --------------------------------------------------------------------------- #
def generate_benchmark(
    base_config: dict, cfg: BenchmarkConfig, bench_root: str | Path, *, log_fn=print,
) -> dict:
    """Generate both families under ``bench_root`` and write ``BENCH_MANIFEST.json``.

    ``base_config`` is a vendored simulator config (``image`` / ``detector`` /
    ``scene`` blocks), e.g. the ``simulator:`` block of ``configs/default.yaml``.
    Nothing vendored is modified; only its scene block is deep-merged with the
    per-cell overrides in this module.
    """
    bench_root = Path(bench_root)
    bench_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # BenchmarkConfig size is authoritative over the base config's image block, so
    # every set uses one agreed image shape (smoke stays tiny regardless of base).
    base_config = _deep_merge(base_config, {"image": {"height": cfg.height, "width": cfg.width}})

    log_fn(f"[bench] generating into {bench_root} (seed={cfg.seed})")

    # Wipe the two family trees FIRST so a grid change cannot leave orphan cells behind.
    # bench-gen is deterministic and non-resumable -- it OVERWRITES a cell whose label
    # collides but does NOT remove a cell whose label is no longer in the grid. So after
    # shrinking or re-centring an axis (e.g. v2 SNR [2..15] -> v3 [0.75..3.0], which drops
    # the 5/8/10/15 cells), the stale cells would persist and be silently ingested by
    # infer / evaluate / the baselines -- a mixed-grid result that looks fine and is wrong.
    # Regenerating from an empty tree is the only guarantee the tree matches THIS cfg.
    for _fam in ("snr_density", "curvature"):
        _fam_dir = bench_root / _fam
        if _fam_dir.exists():
            _n_stale = sum(1 for _p in _fam_dir.iterdir() if _p.is_dir())
            shutil.rmtree(_fam_dir)
            log_fn(f"[bench] cleared {_n_stale} existing cell(s) under {_fam}/ (fresh regen)")

    log_fn("[bench] family 1: SNR x density")
    fam1 = generate_snr_density_family(base_config, cfg, bench_root / "snr_density", log_fn=log_fn)
    log_fn("[bench] family 2: curvature (alpha recovery)")
    fam2 = generate_curvature_family(base_config, cfg, bench_root / "curvature", log_fn=log_fn)

    total_dt = time.perf_counter() - t0
    detector_top = _detector_top_level(base_config)
    manifest = {
        "kind": "spotpipe_benchmark",
        "generation_only": True,
        "note": ("Generation ONLY: no method run, no slope fit, no metric. Cells/sets "
                 "are conditions; the downstream harness re-bins per spot via the frozen "
                 "spotpipe.simulator._features SNR/density definitions."),
        "git_commit": _git_commit(),
        "vendored_simulator_sha": "7b9a0b85ee527afeb73d9e68f9bdb30960775083",
        "seed": cfg.seed,
        "config_hash": _config_hash({
            "seed": cfg.seed, "height": cfg.height, "width": cfg.width,
            "snr_targets": [float(s) for s in cfg.snr_targets],
            "density_levels": [float(d) for d in cfg.density_levels],
            "images_per_cell": cfg.images_per_cell,
            "alpha_values": list(cfg.alpha_values),
            "images_per_alpha": cfg.images_per_alpha,
            "null_control_multiplier": cfg.null_control_multiplier,
            "detector": detector_top,
            "sigma1": _BENCH_SIGMA1, "sigma2": _BENCH_SIGMA2,
            "background_photons": _BENCH_BACKGROUND_PHOTONS,
        }),
        "schema_columns": list(SCHEMA_COLUMNS),
        # Measured detector + fixed benchmark constants, recorded top-level (they
        # are also in every set's meta.json). See CLAUDE.md / docs/snr_convention.md.
        "detector": detector_top,
        "benchmark_constants": {
            "sigma1_px": _BENCH_SIGMA1,
            "sigma2_px": _BENCH_SIGMA2,
            "psf_note": "FIXED PSF for the ENTIRE benchmark (both families); no per-image "
                        "sigma randomisation. sigma2 = 1.2 * sigma1.",
            "background_photons": _BENCH_BACKGROUND_PHOTONS,
            "background_gradient_frac": 0.0,
            "background_structure_frac": 0.0,
            "background_note": ("CONSTANT 2-photon flat background, both channels, every image; "
                                "gradient == structure == 0. This is an ASSUMPTION, not a "
                                "measurement: dark frames have the laser off, so they cannot "
                                "measure photon background (a laser-on/no-sample frame would be "
                                "needed)."),
            "detector_measured_2026_07_13": ("ch1 (lipid) gain from a photon-transfer curve; "
                                             "offset + read variance from dark frames; gain is the "
                                             "variance-matching gain (includes PMT excess noise), "
                                             "so excess_noise_factor = 1.0."),
            "ch2_gain_is_chosen_not_measured": (
                "ch2 (protein) gain = 40.0 is a CHOSEN value representing a PLANNED lower-voltage "
                "acquisition, NOT a measurement. The MEASURED gain at the current 750 V setting is "
                "124.3, which saturates the protein channel at ~32 photons peak and forced the "
                "benchmark into a single-photon regime (curvature A1 collapsed toward [1, 35] ph). "
                "We intend to lower the PMT voltage to avoid this, so the benchmark simulates the "
                "settings we will ACTUALLY USE. gain 40 -> protein clips at ~4095/40 ~= 100 photons "
                "peak (max unclipped SNR ~= 16.8), keeps ~20x read-noise sigma per photoelectron, "
                "and lies inside the retrain's randomized gain range [20, 150]. RE-MEASURE the gain "
                "at whatever voltage is finally used; do not mistake 40 for a measurement."),
            "ch1_gain_simplification": (
                "Lowering the PMT voltage would in reality also lower the LIPID (ch1) gain somewhat. "
                "We keep ch1 at the MEASURED 6.63 because lipid is nowhere near its ADC ceiling and "
                "is never the limiting channel -- flagged as a simplification, not a measurement."),
            "saturation_knee_note": ("saturation_knee = adc_max - offset (the physical 12-bit "
                                     "ADC ceiling above the pedestal); the soft tanh knee's "
                                     "asymptote sits at the hard clip. This is a benchmark-layer "
                                     "choice, not an independent measurement."),
            "channel_mapping": "ch1 = LIPID (561 nm), ch2 = PROTEIN (488 nm) -- pipeline order, "
                               "OPPOSITE the acquisition order.",
            "snr_grid_no_clip_guarantee": ("snr_targets are capped so EVERY family-1 cell stays "
                                           "below both channels' ADC knee (asserted at generation "
                                           "-- generation FAILS if a cell clips). At the chosen "
                                           "protein gain 40 the ch2 ADC clips near SNR ~= 16.8, so "
                                           "the grid stays below that."),
            "legacy_trained_a1_range_photons": [_LEGACY_A1_MIN, _LEGACY_A1_MAX],
            "legacy_trained_density_max_spots_per_px": _LEGACY_DENSITY_MAX,
            "trained_density_max_spots_per_px": _TRAINED_DENSITY_MAX,
            "density_ramp_note": (
                "2026-07-14: the family-1 density ramp was extended with a 6th level at 0.02 "
                "spots/px (~1310 spots / 256^2) because it previously tapered off at 0.015. The "
                "TRAINING density max was raised 0.012 -> 0.024 in the same change, so all six "
                "levels are inside the trained range with headroom. 'density_is_stress' is "
                "raised against the CURRENT trained max (0.024); 'density_above_legacy_training_max' "
                "marks cells the LEGACY checkpoints never saw. Any model trained before "
                "2026-07-14 is out-of-distribution at the top two levels."),
            "retrain_note": _RETRAIN_NOTE,
        },
        "known_unmodeled_features": {
            "protein_pmt_dark_counts": (
                "The 488/protein PMT at 750 V emits ~0.57% of pixels/frame as spurious "
                "single-photoelectron spikes (~one gain step, ~offset+124 ADU, tail to "
                "~1900 ADU); the 561/lipid PMT (500 V) shows none. These look like dim "
                "single-pixel 'spots' in ch2. NOT modelled in this benchmark; see the "
                "separate dark-count robustness check (scripts/darkcount_robustness.py)."),
        },
        "grid": {
            "snr_targets": [float(s) for s in cfg.snr_targets],
            "density_levels": [float(d) for d in cfg.density_levels],
            "density_radius_px": cfg.density_radius_px,
            "cell_label": ("snr = TRUE constant target SNR (intensity solved by inverting the "
                           "frozen SNR); density = constant area density in spots/px"),
        },
        "solved_intensity_table": fam1["solved_intensity_table"],
        "snr_definition": ("peak SNR per channel = (A/(2*pi*sigma^2)) / sqrt(((peak+B)"
                           "+read^2)/n_frames); per-spot scalar = min(snr1, snr2); "
                           "background B = flat level only. Family-1 cells INVERT this to "
                           "solve for a single spot intensity per target (true constant SNR, "
                           "zero spread). See _features.py and docs/snr_convention.md."),
        "density_definition": ("area density in spots per pixel, CONSTANT per cell, set at "
                               "generation and used as the cell label; n_spots = "
                               "round(density*H*W), uniform-random placement, no clustering. "
                               f"Realised local neighbour count within density_radius_px="
                               f"{cfg.density_radius_px} is recorded per cell as an "
                               "informational diagnostic only. See _features.py."),
        "families": {
            "snr_density": {k: v for k, v in fam1.items()
                            if k not in ("cells", "solved_intensity_table")},
            "curvature": {k: v for k, v in fam2.items() if k not in ("sets",)},
        },
        "snr_density_cells": fam1["cells"],
        "curvature_sets": fam2["sets"],
        "warnings": fam1.get("warnings", []) + fam2["warnings"],
        "totals": {
            "n_images": fam1["n_images"] + fam2["n_images"],
            "n_spots": fam1["n_spots"] + fam2["n_spots"],
            "seconds": total_dt,
        },
    }
    _write_json(bench_root / "BENCH_MANIFEST.json", manifest)
    log_fn(f"[bench] done: {manifest['totals']['n_images']} images, "
           f"{manifest['totals']['n_spots']} spots, {total_dt:.1f}s -> {bench_root}")
    return manifest


# --------------------------------------------------------------------------- #
# Config loading                                                              #
# --------------------------------------------------------------------------- #
def _detector_top_level(base_config: dict) -> dict:
    """The (already override-merged) detector config, for the manifest top-level.

    ``jitter_frac == 0`` in the benchmark configs, so these configured constants
    are exactly what every set samples; recording them here as well as in each
    set's meta.json satisfies the "record all of it top-level" requirement.
    """
    det = copy.deepcopy(base_config.get("detector", {}))
    return det


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def load_benchmark_config(path: str | Path) -> tuple[dict, BenchmarkConfig]:
    """Load a benchmark yaml into ``(base_simulator_config, BenchmarkConfig)``.

    The yaml references a base simulator config by *relative name* (resolved under
    ``paths.configs``) so the big vendored ``simulator:`` block is reused rather
    than duplicated. The ``benchmark:`` block holds the grid / counts / seed.
    """
    import yaml

    from spotpipe.paths import get_paths

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    base_name = raw.get("base_simulator_config", "default.yaml")
    base_path = get_paths().configs / base_name
    with open(base_path, "r", encoding="utf-8") as fh:
        base_raw = yaml.safe_load(fh) or {}
    base_config = base_raw.get("simulator", {})

    # Benchmark-layer simulator overrides (e.g. the MEASURED detector) deep-merged
    # onto the base config, so the DISPOSABLE benchmark can pin the real-instrument
    # detector WITHOUT touching the training configs (default.yaml / smoke.yaml,
    # which the checkpoints depend on). CLAUDE.md: modify only the benchmark layer.
    sim_overrides = raw.get("simulator_overrides", {}) or {}
    if sim_overrides:
        base_config = _deep_merge(base_config, sim_overrides)

    b = dict(raw.get("benchmark", {}) or {})

    kwargs = {}
    if "seed" in b:
        kwargs["seed"] = int(b["seed"])
    if "image" in b:
        kwargs["height"] = int(b["image"].get("height", 256))
        kwargs["width"] = int(b["image"].get("width", 256))
    if "snr_targets" in b:
        kwargs["snr_targets"] = tuple(float(s) for s in b["snr_targets"])
    if "density_levels" in b:
        kwargs["density_levels"] = tuple(float(d) for d in b["density_levels"])
    if "images_per_cell" in b:
        kwargs["images_per_cell"] = int(b["images_per_cell"])
    if "alpha_values" in b:
        kwargs["alpha_values"] = tuple(float(a) for a in b["alpha_values"])
    if "images_per_alpha" in b:
        kwargs["images_per_alpha"] = int(b["images_per_alpha"])
    if "null_control_multiplier" in b:
        kwargs["null_control_multiplier"] = int(b["null_control_multiplier"])
    if "min_alpha_decades" in b:
        kwargs["min_alpha_decades"] = float(b["min_alpha_decades"])
    if "density_radius_px" in b:
        kwargs["density_radius_px"] = float(b["density_radius_px"])

    # If the base config carries an image size and the benchmark yaml didn't
    # override it, adopt the base size so smoke stays tiny.
    if "image" not in b and "image" in base_config:
        kwargs.setdefault("height", int(base_config["image"].get("height", 256)))
        kwargs.setdefault("width", int(base_config["image"].get("width", 256)))

    return base_config, BenchmarkConfig(**kwargs)
