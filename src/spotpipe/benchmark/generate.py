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
      snr={S}_density={D}/          one homogeneous CONDITION per cell; S = SNR bin
                                    lower edge, D = constant area density (spots/px)
        images/ image_<id>.tif      raw observed counts, uint16 [2,H,W] stack
        ground_truth/ gt_<id>.csv   frozen-schema GT: x,y + true logI1/logI2/I1/I2
        meta.json                   label (S,D), per-image sigma1/sigma2, realised
                                    SNR/density stats, n_images, seed
    curvature/
      alpha={A}/
        images/ ...
        ground_truth/ ...
        meta.json                   true_alpha, sim_log_slope=alpha/2, per-image
                                    sigma1/sigma2, A1-spread stats, seed, null flag
    BENCH_MANIFEST.json             everything generated: seeds, git SHA, config hash

Two conventions worth stating up front (both recorded in every ``meta.json``):

* A cell is a *nominal target condition*, not a per-spot guarantee. Family 1 keeps
  a deliberately WIDE per-spot intensity draw (needed to see intensity-dependent
  bias), so a cell labelled ``snr=5`` contains a spread of per-spot SNR around
  that level. The frozen ``_features`` SNR/density definitions are applied to the
  realised spots and their distribution is recorded, so the labelling is honest.
* ``ground_truth_sigma`` (true per-image, per-channel PSF width) is plumbed from
  ``meta['ground_truth_sigma']`` into every image's record -- the schema's
  ``sigma*_hat`` columns mean "model estimate" and stay NaN for GT.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from spotpipe.benchmark.alpha import alpha_to_sim_slope, sim_slope_to_alpha
from spotpipe.schema import SCHEMA_COLUMNS, write_spots
from spotpipe.simulator import forward_model, noise, psf
from spotpipe.simulator._features import (
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
# Family 1 -- SNR x density scene-override ladders                             #
# --------------------------------------------------------------------------- #
# Each SNR bin is a *condition* aimed at populating that half-open bin. Because
# per-spot peak SNR is peak = A/(2*pi*sigma^2) over a noise floor, the ONLY way to
# move the SNR level while keeping a wide intensity draw is to shift a wide
# intensity window's centre (brighter -> higher SNR) and lean on background / PSF
# at the extremes. Windows stay ~1.6 decades wide so each cell still spans a real
# intensity range (that is the "keep the natural wide A1 draw" requirement -- a
# range, not a fixed value). Ratio law is pinned neutral for the whole family.
_SNR_LADDER: list[dict] = [
    # bin 0  [0, 2):  very dim + high background + wide PSF  -> lowest SNR
    {"intensity": {"log10_min": 0.8, "log10_max": 2.2, "dim_bias": 1.6},
     "background": {"level": {"min": 18.0, "max": 30.0}},
     "psf": {"sigma1": {"min": 1.4, "max": 1.8}}},
    # bin 1  [2, 5):  dim + mid background
    {"intensity": {"log10_min": 1.3, "log10_max": 2.8, "dim_bias": 1.6},
     "background": {"level": {"min": 8.0, "max": 16.0}}},
    # bin 2  [5, 10): mid intensity + mid-low background
    {"intensity": {"log10_min": 1.8, "log10_max": 3.2, "dim_bias": 1.5},
     "background": {"level": {"min": 4.0, "max": 9.0}}},
    # bin 3  [10, 20): mid-bright + low background
    {"intensity": {"log10_min": 2.3, "log10_max": 3.6, "dim_bias": 1.3},
     "background": {"level": {"min": 2.0, "max": 5.0}}},
    # bin 4  [20, 50): bright + low background + slightly narrow PSF
    {"intensity": {"log10_min": 2.9, "log10_max": 4.0, "dim_bias": 1.1},
     "background": {"level": {"min": 1.5, "max": 3.0}},
     "psf": {"sigma1": {"min": 1.0, "max": 1.3}}},
    # bin 5  [50, inf): very bright + lowest background + narrow PSF -> highest SNR
    {"intensity": {"log10_min": 3.4, "log10_max": 4.3, "dim_bias": 0.9},
     "background": {"level": {"min": 1.0, "max": 2.5}},
     "psf": {"sigma1": {"min": 1.0, "max": 1.15}}},
]

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

# Neutral ratio law for the WHOLE snr_density family: pin sim_intercept = 0 and
# sim_log_slope = 0 (the simulator's alpha / beta config fields), so this family
# isolates difficulty and carries no curvature (true_alpha == 0 everywhere).
_NEUTRAL_RATIO_LAW = {
    "ratio_law": {
        "alpha": {"min": 0.0, "max": 0.0},   # sim_intercept = 0
        "beta": {"min": 0.0, "max": 0.0},     # sim_log_slope  = 0  -> true_alpha = 0
        "scatter_std": {"min": 0.08, "max": 0.08},
    }
}


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
    "background": {"level": {"min": 1.5, "max": 3.0},
                   "gradient_frac": {"min": 0.0, "max": 0.15},
                   "structure_frac": {"min": 0.0, "max": 0.15}},
    "psf": {"sigma1": {"min": 1.1, "max": 1.4}, "c2_sigma_mismatch": {"min": 1.05, "max": 1.2}},
    "density": {"min": 0.0004, "max": 0.0015}, "oversample_dense_fraction": 0.0,
    "clustering": {"cluster_prob": 0.0},
}


def _curvature_intensity_window(
    base_config: dict, operating_point: dict, sim_log_slope: float, min_alpha_decades: float,
) -> dict:
    """A1 intensity window (log10 photons) sized to avoid ch1/ch2 saturation.

    Solves for the brightest A1 whose worst-case gained peak stays at
    ``_CURV_SAT_TARGET_FRAC`` of each channel's saturation knee, then places a
    fixed-width window below it. ch1 sees A1 directly; ch2 sees
    ``A2 = A1**(1 + sim_log_slope)`` (intercept 0) times the upper scatter tail,
    so for steep +slope ch2 binds and the ceiling drops. Worst case uses the
    SMALLEST PSF sigma (highest peak fraction) and the LARGEST flat background.
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

    pcfg = operating_point.get("psf", {})
    sigma1_min = float(pcfg.get("sigma1", {}).get("min", 1.0))
    mismatch_min = float(pcfg.get("c2_sigma_mismatch", {}).get("min", 1.05))
    sigma2_min = sigma1_min * mismatch_min           # smallest sigma -> highest peak fraction
    pf1 = psf.gaussian_peak_fraction(sigma1_min)
    pf2 = psf.gaussian_peak_fraction(sigma2_min)

    bg_max = float(operating_point.get("background", {}).get("level", {}).get("max", 0.0))
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

    ``snr_edges`` are the frozen checkpoint SNR bin edges; a cell is labelled by
    its SNR bin's LOWER edge (half-open ``[lower, next)``) and by its area density
    in spots/px. ``density_levels`` are constant area densities (spots per pixel):
    each is a cell knob AND its label -- no neighbour-count binning, no clustering.
    Every density level is generated at every SNR level (full orthogonal grid).
    The number of ``_SNR_LADDER`` entries must match ``len(snr_edges) - 1``.
    """

    seed: int = 0
    height: int = 256
    width: int = 256
    density_radius_px: float = 4.0       # informational realised-neighbour stat only

    # Family 1
    snr_edges: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, math.inf)
    # Constant AREA densities (spots/px): sweep across and slightly beyond the
    # training range [~0.0006, ~0.012]. Each is used as both the knob and label.
    density_levels: tuple[float, ...] = (0.0006, 0.002, 0.006, 0.012, 0.015)
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
        if len(_SNR_LADDER) != len(self.snr_edges) - 1:
            raise ValueError(
                f"_SNR_LADDER has {len(_SNR_LADDER)} entries but snr_edges implies "
                f"{len(self.snr_edges) - 1} bins")
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
    """Generate the SNR x density family. Returns a summary dict for the manifest."""
    family_root.mkdir(parents=True, exist_ok=True)
    n_snr = len(cfg.snr_edges) - 1
    cells = []
    t0 = time.perf_counter()

    for si in range(n_snr):
        for di, density in enumerate(cfg.density_levels):
            s_lbl = _edge_label(cfg.snr_edges[si])
            d_lbl = _edge_label(density)                 # spots/px, e.g. "0.006"
            label = f"snr={s_lbl}_density={d_lbl}"
            set_dir = family_root / label
            set_seed = _set_seed(cfg.seed, "snr_density", label)

            # SNR ladder x constant area density (uniform placement, no clustering).
            override = _deep_merge(_SNR_LADDER[si], _constant_density_override(density))
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

            meta = {
                "family": "snr_density",
                "label": label,
                "condition": {
                    "snr_bin": _bin_label(cfg.snr_edges, si),
                    "snr_index": si,
                    "snr_nominal_lower_edge": _json_edge(cfg.snr_edges[si]),
                    "area_density_spots_per_px": float(density),
                    "density_index": di,
                    "placement": "uniform_random (no clustering)",
                },
                "area_density_spots_per_px": float(density),
                "area_density_constant_per_cell": bool(constant_density),
                "note": ("Nominal SNR target condition (not a per-spot guarantee): a wide "
                         "intensity draw is kept on purpose, so per-spot SNR is a "
                         "distribution around the label (see realised_snr). Area density "
                         "IS an exact per-cell constant, set at generation; spots are "
                         "placed uniformly at random with no clustering."),
                "ratio_law": "neutral (sim_intercept=0, sim_log_slope=0 -> true_alpha=0)",
                "true_alpha": 0.0,
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
                          "area_density_spots_per_px": float(density),
                          "n_images": res["n_images"], "n_spots": res["n_spots_total"],
                          "seed": set_seed,
                          "realised_snr_median": (float(np.median(res["snr"]))
                                                  if res["snr"].size else None),
                          "realised_neighbors_mean": (float(np.mean(res["n_neighbors"]))
                                                      if res["n_neighbors"].size else None)})
            log_fn(f"  [snr_density] {label:>24}: {res['n_images']} imgs, "
                   f"{res['n_spots_total']:>6} spots, density={density:g} spots/px")

    dt = time.perf_counter() - t0
    n_images = sum(c["n_images"] for c in cells)
    n_spots = sum(c["n_spots"] for c in cells)
    return {"n_cells": len(cells), "n_images": n_images, "n_spots": n_spots,
            "seconds": dt, "cells": cells}


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
            base_config, _CURVATURE_OPERATING_POINT, sim_log_slope, cfg.min_alpha_decades)
        override = _deep_merge(_CURVATURE_OPERATING_POINT, {
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
    log_fn("[bench] family 1: SNR x density")
    fam1 = generate_snr_density_family(base_config, cfg, bench_root / "snr_density", log_fn=log_fn)
    log_fn("[bench] family 2: curvature (alpha recovery)")
    fam2 = generate_curvature_family(base_config, cfg, bench_root / "curvature", log_fn=log_fn)

    total_dt = time.perf_counter() - t0
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
            "snr_edges": [_json_edge(e) for e in cfg.snr_edges],
            "density_levels": [float(d) for d in cfg.density_levels],
            "images_per_cell": cfg.images_per_cell,
            "alpha_values": list(cfg.alpha_values),
            "images_per_alpha": cfg.images_per_alpha,
            "null_control_multiplier": cfg.null_control_multiplier,
        }),
        "schema_columns": list(SCHEMA_COLUMNS),
        "grid": {
            "snr_edges": [_json_edge(e) for e in cfg.snr_edges],
            "density_levels": [float(d) for d in cfg.density_levels],
            "density_radius_px": cfg.density_radius_px,
            "cell_label": ("snr = SNR bin lower edge (half-open [lower, next)); "
                           "density = constant area density in spots/px"),
        },
        "snr_definition": ("peak SNR per channel = (A/(2*pi*sigma^2)) / sqrt(((peak+B)"
                           "+read^2)/n_frames); per-spot scalar = min(snr1, snr2); "
                           "background B = flat level only. See _features.py and "
                           "docs/snr_convention.md."),
        "density_definition": ("area density in spots per pixel, CONSTANT per cell, set at "
                               "generation and used as the cell label; n_spots = "
                               "round(density*H*W), uniform-random placement, no clustering. "
                               f"Realised local neighbour count within density_radius_px="
                               f"{cfg.density_radius_px} is recorded per cell as an "
                               "informational diagnostic only. See _features.py."),
        "families": {
            "snr_density": {k: v for k, v in fam1.items() if k != "cells"},
            "curvature": {k: v for k, v in fam2.items() if k not in ("sets",)},
        },
        "snr_density_cells": fam1["cells"],
        "curvature_sets": fam2["sets"],
        "warnings": fam2["warnings"],
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
def _bin_label(edges: tuple[float, ...], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    if math.isinf(hi):
        return f">={lo:g}"
    return f"[{lo:g},{hi:g})"


def _json_edge(e: float):
    return "inf" if (isinstance(e, float) and math.isinf(e)) else float(e)


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

    b = dict(raw.get("benchmark", {}) or {})
    # inf-aware edge parsing (yaml `.inf` -> math.inf; string "inf" tolerated too).
    def _edges(key, default):
        vals = b.get(key, default)
        return tuple(math.inf if (v is None or (isinstance(v, str) and v.lower() == "inf")
                                  or (isinstance(v, float) and math.isinf(v)))
                     else float(v) for v in vals)

    kwargs = {}
    if "seed" in b:
        kwargs["seed"] = int(b["seed"])
    if "image" in b:
        kwargs["height"] = int(b["image"].get("height", 256))
        kwargs["width"] = int(b["image"].get("width", 256))
    if "snr_edges" in b:
        kwargs["snr_edges"] = _edges("snr_edges", None)
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
