"""Generate & freeze the stratified benchmark / test dataset (build stage 4.5).

This builds the SEPARATE, FROZEN, externally-ingestible benchmark/test set used
ONLY for final reporting and external-method comparison -- NEVER for checkpoint
selection. It is the gating artifact that makes parallel external-method
integration valid: every method (our HRNet checkpoints, the aperture/PSF-fit
baselines, and every external adapter -- DECODE, Spotiflow, ...) runs on this
EXACT same frozen set.

Why this set is special (see the prompt & CLAUDE.md):

* It is a DIFFERENT seed/manifest from the training-validation set used for
  checkpoint selection -- keeping them distinct prevents test-set leakage.
* It is FROZEN: generated once, written to disk, manifest-pinned with the git
  commit and per-artifact checksums. Once external methods start running on it,
  it must not change; :func:`verify_benchmark_set` re-checks the checksums.
* It is STRATIFIED, not random-drawn: the benchmark metrics bin by SNR and local
  density and report beta recovery, so we populate the exact SNR x density x beta
  cells the metrics use, using targeted / rejection generation until every cell
  has a solid count of GROUND-TRUTH spots (matched-pair counts are unknowable at
  generation time -- no method has run -- so we stratify on GT spots).

On-disk layout (under ``out_dir``)::

  images/<id>.npy            canonical two-channel array, uint16 [2, H, W]
  images_ch1_raw/<id>.tif    observed detector counts, ch1, uint16 [0, adc_max]
  images_ch2_raw/<id>.tif    observed detector counts, ch2, uint16 [0, adc_max]
  images_ch1_photon/<id>.tif offset-subtracted, gain-corrected photon-prop., ch1, float32
  images_ch2_photon/<id>.tif offset-subtracted, gain-corrected photon-prop., ch2, float32
  meta/meta_<id>.json        full per-image simulator metadata (feature fidelity)
  audit/background_<id>.npy   TRUE simulator background [2,H,W] float32 (NON-FAIR; debug only)
  ground_truth.csv           canonical spotpipe.schema GT table, all images
  beta_per_image.csv         true alpha / beta / beta-group per image
  metadata.csv               per-spot binning/filtering metadata (SNR, density, bins, flags)
  checksums.sha256           per-file sha256 of every artifact (sha256sum format)
  manifest.json              frozen edges, full config, git commit, counts, checksums

Image conventions (documented in ``manifest.json`` too):

* External single-channel tools (Spotiflow / DECODE) DETECT / LOCALIZE on the
  raw observed per-channel TIFFs -- that is what real microscopy hands them.
* Adapters EXTRACT intensity I1/I2 from the photon-proportional per-channel
  images using their OWN declared local-background / aperture / PSF-fit estimator
  (the photon images keep optical/background structure so they must). The exact
  per-channel correction is ``photon_k = (raw_counts - offset_k) / gain_k`` (a
  linear pedestal+gain correction only -- it deliberately does NOT invert the
  saturation knee, which a fair method cannot know). Adapters must never divide
  raw counts, and must never read ``audit/`` (true) background unless explicitly
  labeled oracle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from spotpipe.benchmark.features import axis_params_from_meta, local_neighbor_count, peak_snr
from spotpipe.schema import SCHEMA_COLUMNS, records_to_dataframe, write_spots
from spotpipe.simulator import forward_model, noise
from spotpipe.simulator.generate_dataset import _git_commit

__all__ = [
    "generate_benchmark_set",
    "verify_benchmark_set",
    "stratification_report",
    "IMAGE_DIRS",
]

# Directory names that hold bulk image arrays (reproduced from config + manifest,
# kept out of git; the small text artifacts are the freeze record).
IMAGE_DIRS = (
    "images",
    "images_ch1_raw",
    "images_ch2_raw",
    "images_ch1_photon",
    "images_ch2_photon",
)

_LN10 = math.log(10.0)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
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


def _edges(values) -> list[float]:
    """Parse YAML bin edges (``.inf`` -> ``math.inf``) into floats."""
    return [math.inf if (v is None or (isinstance(v, float) and math.isinf(v))) else float(v)
            for v in values]


def _bin_index(value: float, edges: list[float]) -> int:
    """Half-open bin ``[edges[i], edges[i+1])`` index; -1 if NaN / out of range.

    Identical convention to :func:`spotpipe.benchmark.metrics._assign_bins` so the
    frozen edges populate exactly the cells the benchmark later bins into.
    """
    if not math.isfinite(value):
        return -1
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return -1


def _bin_label(edges: list[float], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    if math.isinf(hi):
        return f">={lo:g}"
    return f"[{lo:g},{hi:g})"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Stratification design: scene-config overrides per regime                     #
# --------------------------------------------------------------------------- #
# Every override is a partial ``scene:`` config deep-merged onto the base scene,
# then the per-image beta is pinned to its group (min == max). Reusing
# ``sample_scene_params`` for the actual draw keeps ONE definition of the scene.

# --- intensity (drives SNR via per-spot peak = A / (2*pi*sigma^2)) ---------- #
_DIM = {"intensity": {"log10_min": 0.9, "log10_max": 2.0, "dim_bias": 1.8}}
_MID_I = {"intensity": {"log10_min": 1.6, "log10_max": 3.2, "dim_bias": 1.4}}
_BRIGHT = {"intensity": {"log10_min": 3.0, "log10_max": 4.2, "dim_bias": 1.0}}
_VERY_BRIGHT = {"intensity": {"log10_min": 3.6, "log10_max": 4.4, "dim_bias": 0.8}}

# --- background level (high bg lowers SNR; also the background-structure axis) #
_HIGH_BG = {"background": {"level": {"min": 16.0, "max": 30.0},
                          "gradient_frac": {"min": 0.0, "max": 0.2},
                          "structure_frac": {"min": 0.0, "max": 0.2}}}
_MID_BG = {"background": {"level": {"min": 6.0, "max": 14.0}}}
_LOW_BG = {"background": {"level": {"min": 1.5, "max": 4.0},
                         "gradient_frac": {"min": 0.0, "max": 0.15},
                         "structure_frac": {"min": 0.0, "max": 0.15}}}

# --- PSF sigma (bigger sigma spreads flux -> lower peak -> lower SNR) -------- #
_WIDE_PSF = {"psf": {"sigma1": {"min": 1.5, "max": 1.9}}}
_NARROW_PSF = {"psf": {"sigma1": {"min": 1.0, "max": 1.15}}}

# --- local density (only tight CLUSTERS reach the high-neighbour-count bins;   #
#     uniform spots at the max configured density average < 1 neighbour in 4px) #
_SPARSE = {"density": {"min": 0.0004, "max": 0.0015}, "oversample_dense_fraction": 0.0,
           "clustering": {"cluster_prob": 0.0}}
_MID_DENS = {"density": {"min": 0.002, "max": 0.005}, "oversample_dense_fraction": 0.0,
             "clustering": {"cluster_prob": 0.4, "n_clusters": {"min": 3, "max": 8},
                            "cluster_sigma_px": {"min": 6.0, "max": 12.0}}}
_DENSE = {"density": {"min": 0.006, "max": 0.012}, "oversample_dense_fraction": 0.0,
          "clustering": {"cluster_prob": 1.0, "n_clusters": {"min": 3, "max": 6},
                         "cluster_sigma_px": {"min": 5.0, "max": 9.0}}}
_ULTRA_DENSE = {"density": {"min": 0.008, "max": 0.014}, "oversample_dense_fraction": 0.0,
                "clustering": {"cluster_prob": 1.0, "n_clusters": {"min": 2, "max": 4},
                               "cluster_sigma_px": {"min": 3.0, "max": 5.0}}}


def _snr_override(snr_idx: int, n_snr_bins: int) -> dict:
    """Scene override aimed at SNR bin ``snr_idx`` (0 = dimmest, last = brightest)."""
    if snr_idx <= 0:
        ov = _deep_merge(_deep_merge(_DIM, _HIGH_BG), _WIDE_PSF)
    elif snr_idx == 1:
        ov = _deep_merge(_DIM, _MID_BG)
    elif snr_idx <= n_snr_bins - 3:
        ov = _deep_merge(_MID_I, _MID_BG)
    elif snr_idx == n_snr_bins - 2:
        ov = _deep_merge(_BRIGHT, _LOW_BG)
    else:  # brightest bin: very bright + low bg + narrow PSF
        ov = _deep_merge(_deep_merge(_VERY_BRIGHT, _LOW_BG), _NARROW_PSF)
    return ov


def _density_override(dens_idx: int, n_dens_bins: int) -> dict:
    """Scene override aimed at local-density bin ``dens_idx`` (0 = sparsest)."""
    if dens_idx <= 0:
        return _SPARSE
    if dens_idx == 1:
        return _MID_DENS
    if dens_idx == n_dens_bins - 1:
        return _ULTRA_DENSE
    return _DENSE


def _targeted_override(snr_idx: int, dens_idx: int, n_snr_bins: int, n_dens_bins: int) -> dict:
    """Combined scene override aimed at one SNR x density cell."""
    return _deep_merge(_snr_override(snr_idx, n_snr_bins), _density_override(dens_idx, n_dens_bins))


# --- mandatory stress-regime profiles (guarantee non-empty flagged subsets) -- #
# Each yields ``stress_images_each`` images; their spots ALSO feed cell coverage.
def _stress_profiles() -> "OrderedDict[str, dict]":
    bright_clip = {"intensity": {"log10_min": 4.0, "log10_max": 4.5, "dim_bias": 0.7}}
    bright_near = {"intensity": {"log10_min": 3.3, "log10_max": 3.7, "dim_bias": 1.0}}
    base_dens = _MID_DENS
    profiles: "OrderedDict[str, dict]" = OrderedDict()

    # channel imbalance -- driven by the ratio law intercept alpha (and intensity):
    #   log A2 = (1+beta) log A1 + alpha  -> alpha << 0 dims ch2, alpha >> 0 brightens it.
    # Bright ch1 + strongly negative alpha pushes median ch2 into the dim band;
    # dim ch1 + strongly positive alpha pushes median ch2 into the bright band.
    # (beta is fixed per image; the beta>=0 / beta<=0 images respectively land
    #  these classes most cleanly, so cycling betas keeps each subset non-empty.)
    profiles["ch1_bright_ch2_dim"] = _deep_merge(_deep_merge(_BRIGHT, base_dens),
        {"ratio_law": {"alpha": {"min": -3.4, "max": -2.6}}})
    profiles["ch1_dim_ch2_bright"] = _deep_merge(_deep_merge(
        {"intensity": {"log10_min": 1.2, "log10_max": 2.1, "dim_bias": 1.2}}, base_dens),
        {"ratio_law": {"alpha": {"min": 3.0, "max": 3.8}}})
    profiles["both_dim"] = _deep_merge(_deep_merge(_DIM, _MID_BG),
        {"ratio_law": {"alpha": {"min": -0.2, "max": 0.2}}, **base_dens})
    profiles["both_bright"] = _deep_merge(_deep_merge(_BRIGHT, _LOW_BG),
        {"ratio_law": {"alpha": {"min": -0.2, "max": 0.2}}, **base_dens})

    # saturation regime
    profiles["near_saturation"] = _deep_merge(_deep_merge(bright_near, _LOW_BG), base_dens)
    profiles["clipped"] = _deep_merge(_deep_merge(bright_clip, _LOW_BG), base_dens)

    # PSF mismatch
    profiles["psf_matched"] = _deep_merge(_deep_merge(_MID_I, base_dens),
        {"psf": {"sigma1": {"min": 1.2, "max": 1.5}, "c2_sigma_mismatch": {"min": 1.0, "max": 1.04}}})
    profiles["psf_large_mismatch"] = _deep_merge(_deep_merge(_MID_I, base_dens),
        {"psf": {"c2_sigma_mismatch": {"min": 1.3, "max": 1.5}}})
    profiles["large_shift"] = _deep_merge(_deep_merge(_MID_I, base_dens),
        {"registration_shift": {"max_px": 1.0}})

    # background structure
    profiles["bg_flat"] = _deep_merge(_deep_merge(_MID_I, base_dens),
        {"background": {"gradient_frac": {"min": 0.0, "max": 0.05},
                        "structure_frac": {"min": 0.0, "max": 0.05}}})
    profiles["bg_gradient"] = _deep_merge(_deep_merge(_MID_I, base_dens),
        {"background": {"gradient_frac": {"min": 0.45, "max": 0.7},
                        "structure_frac": {"min": 0.0, "max": 0.1}}})
    profiles["bg_lowfreq"] = _deep_merge(_deep_merge(_MID_I, base_dens),
        {"background": {"gradient_frac": {"min": 0.0, "max": 0.1},
                        "structure_frac": {"min": 0.45, "max": 0.7},
                        "structure_scale_px": {"min": 24.0, "max": 64.0}}})
    return profiles


# The "broad" workhorse: wide intensity so each image spans many SNR bins, mixed
# density incl. clusters so spots also spread across density bins.
_BROAD = {
    "intensity": {"log10_min": 0.9, "log10_max": 4.2, "dim_bias": 1.6},
    "density": {"min": 0.0010, "max": 0.012}, "oversample_dense_fraction": 0.3,
    "clustering": {"cluster_prob": 0.6, "n_clusters": {"min": 2, "max": 8},
                   "cluster_sigma_px": {"min": 4.0, "max": 14.0}},
    "background": {"level": {"min": 2.0, "max": 24.0}},
}


# --------------------------------------------------------------------------- #
# Per-image classification (stress regimes), from data + meta                  #
# --------------------------------------------------------------------------- #
def _channel_imbalance_class(log_a1: np.ndarray, log_a2: np.ndarray,
                             bright_log10: float, dim_log10: float) -> str:
    if log_a1.size == 0:
        return "empty"
    m1 = float(np.median(log_a1)) / _LN10  # median log10 A1
    m2 = float(np.median(log_a2)) / _LN10
    b1, d1 = m1 >= bright_log10, m1 <= dim_log10
    b2, d2 = m2 >= bright_log10, m2 <= dim_log10
    if d1 and d2:
        return "both_dim"
    if b1 and b2:
        return "both_bright"
    if b1 and d2:
        return "ch1_bright_ch2_dim"
    if d1 and b2:
        return "ch1_dim_ch2_bright"
    return "mixed"


def _saturation_regime(image: np.ndarray, n_saturated: int, adc_max: int) -> str:
    if int(image.max()) >= adc_max:
        return "clipped"
    if n_saturated > 0 or int(image.max()) >= int(round(0.9 * adc_max)):
        return "near_saturation"
    return "unsaturated"


def _psf_regime(sigma1: float, sigma2: float) -> str:
    return "matched" if (sigma2 / max(sigma1, 1e-9)) < 1.05 else "mismatch"


def _background_type(bg: dict) -> str:
    g = float(bg.get("gradient_frac", 0.0))
    s = float(bg.get("structure_frac", 0.0))
    if g < 0.1 and s < 0.1:
        return "flat"
    return "low_freq" if s >= g else "gradient"


# --------------------------------------------------------------------------- #
# Generation                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class _Cell:
    snr_idx: int
    dens_idx: int


def generate_benchmark_set(
    base_config: dict,
    bench_cfg: dict,
    out_dir: str | Path,
    *,
    log_fn=print,
) -> dict:
    """Generate and freeze the stratified benchmark/test set; return its manifest.

    Parameters
    ----------
    base_config : the forward-model config (``image`` / ``detector`` / ``scene``),
        e.g. parsed ``configs/simulator.yaml``. The detector block is the FIXED
        instrument; it is sampled ONCE from the seed and reused for every image.
    bench_cfg : the ``benchmark_set:`` stratification block (seed, frozen bin
        edges, beta groups, coverage targets, max attempts, ...).
    out_dir : destination directory (created; image subdirs are git-ignored).
    """
    out_dir = Path(out_dir)
    for d in IMAGE_DIRS + ("meta", "audit"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    seed = int(bench_cfg.get("seed", 0))
    img_cfg = _deep_merge(base_config.get("image", {}), bench_cfg.get("image", {}))
    shape = (int(img_cfg.get("height", 256)), int(img_cfg.get("width", 256)))
    base_scene = base_config.get("scene", {})

    snr_edges = _edges(bench_cfg["snr_bins"])
    density_radius = float(bench_cfg.get("density_radius_px", 4.0))
    beta_groups = [float(b) for b in bench_cfg["beta_groups"]]
    n_snr = len(snr_edges) - 1

    min_gt = int(bench_cfg.get("min_gt_per_cell", 200))
    min_imgs_beta = int(bench_cfg.get("min_images_per_beta_group", 8))
    stress_each = int(bench_cfg.get("min_images_per_stress_flag", 6))
    n_base = int(bench_cfg.get("base_images", 48))
    max_attempts = int(bench_cfg.get("max_attempts", 3000))
    bright_log10 = float(bench_cfg.get("channel_imbalance_bright_log10", 3.0))
    dim_log10 = float(bench_cfg.get("channel_imbalance_dim_log10", 2.0))
    alpha_spec = bench_cfg.get("alpha", {"min": -0.7, "max": 0.7})
    scatter_spec = bench_cfg.get("scatter_std", {"min": 0.03, "max": 0.25})

    # Detector = fixed instrument, from the seed alone (independent of N).
    root = np.random.SeedSequence(seed)
    det_seq, img_seq = root.spawn(2)
    detector = noise.sample_detector_params(base_config.get("detector", {}),
                                            np.random.default_rng(det_seq))
    adc_max = detector.adc_max
    det_meta = forward_model._detector_to_meta(detector)
    offsets = (detector.ch1.offset, detector.ch2.offset)
    gains = (detector.ch1.gain, detector.ch2.gain)

    # --- density edges: the agreed edges, or data-driven (frozen once) -------
    data_driven = bool(bench_cfg.get("data_driven_density_edges", False))
    if not data_driven:
        density_edges = _edges(bench_cfg["density_bin_edges"])
    else:
        density_edges = None  # computed from a pilot pass below
    n_dens = (len(density_edges) - 1) if density_edges is not None else \
        int(bench_cfg.get("data_driven_n_density_bins", 4))

    # --- mutable generation state -------------------------------------------
    spawn_counter = [0]

    def _next_rng() -> np.random.Generator:
        child = img_seq.spawn(1)[0]
        spawn_counter[0] += 1
        return np.random.default_rng(child)

    records: list[dict] = []   # accumulates per-image bookkeeping for emission

    def _simulate(image_id: str, scene_override: dict, beta: float, alpha_spec_i, profile: str) -> dict:
        scene_cfg = _deep_merge(base_scene, scene_override)
        scene_cfg = _deep_merge(scene_cfg, {
            "ratio_law": {
                "beta": {"min": beta, "max": beta},   # beta FIXED per image, recorded
                "alpha": alpha_spec_i,
                "scatter_std": scatter_spec,
            }})
        rng = _next_rng()
        scene = forward_model.sample_scene_params(scene_cfg, rng, shape)
        sim = forward_model.simulate_image(image_id=image_id, shape=shape, scene=scene,
                                           detector=detector, rng=rng, with_diagnostics=True)
        gt = sim.spots
        log_a1 = gt["logI1"].to_numpy(float)
        log_a2 = gt["logI2"].to_numpy(float)
        params = axis_params_from_meta(sim.meta)
        snr = peak_snr(log_a1, log_a2, params)
        nbr = local_neighbor_count(gt["x"].to_numpy(float), gt["y"].to_numpy(float), density_radius)
        return {
            "image_id": image_id, "sim": sim, "gt": gt, "scene": scene, "profile": profile,
            "beta": beta, "alpha": scene.alpha,
            "log_a1": log_a1, "log_a2": log_a2,
            "snr": snr["snr"], "snr1": snr["snr1"], "snr2": snr["snr2"], "n_neighbors": nbr,
        }

    # --- (optional) pilot pass to FREEZE data-driven density edges -----------
    pilot = []
    if data_driven:
        n_pilot = int(bench_cfg.get("data_driven_pilot_images", 24))
        log_fn(f"[pilot] data-driven density edges: generating {n_pilot} pilot images...")
        for i in range(n_pilot):
            beta = beta_groups[i % len(beta_groups)]
            pilot.append(_simulate(f"pilot_{i:05d}", _BROAD, beta, alpha_spec, "pilot"))
        all_nbr = np.concatenate([p["n_neighbors"] for p in pilot if p["n_neighbors"].size])
        qs = np.linspace(0, 1, n_dens + 1)[1:-1]
        inner = [float(np.quantile(all_nbr, q)) for q in qs]
        # de-dup / monotone, integer-ish edges
        edges = [0.0] + sorted(set(round(e) for e in inner)) + [math.inf]
        density_edges = _edges(edges)
        n_dens = len(density_edges) - 1
        log_fn(f"[pilot] frozen data-driven density edges = {density_edges}")

    cells = [_Cell(s, d) for s in range(n_snr) for d in range(n_dens)]

    def _tally(rec: dict, counts: np.ndarray) -> None:
        snr_b = np.array([_bin_index(v, snr_edges) for v in rec["snr"]])
        den_b = np.array([_bin_index(v, density_edges) for v in rec["n_neighbors"]])
        rec["snr_bin_idx"] = snr_b
        rec["dens_bin_idx"] = den_b
        for s, d in zip(snr_b, den_b):
            if s >= 0 and d >= 0:
                counts[s, d] += 1

    counts = np.zeros((n_snr, n_dens), dtype=np.int64)
    beta_img_counts = Counter()
    stress_counts = Counter()

    def _accept(rec: dict) -> None:
        _tally(rec, counts)
        beta_img_counts[rec["beta"]] += 1
        records.append(rec)

    # Re-tally pilot images (their spots count toward coverage too).
    for p in pilot:
        p["image_id"] = f"img_{len(records):05d}"
        _accept(p)

    # --- Phase 0: mandatory stress-regime images (guarantee flagged subsets) -
    log_fn("[phase0] generating mandatory stress-regime images...")
    stress = _stress_profiles()
    bi = 0
    for name, override in stress.items():
        for _ in range(stress_each):
            beta = beta_groups[bi % len(beta_groups)]; bi += 1
            # channel-imbalance profiles carry their own alpha; others use base alpha.
            a_spec = override.get("ratio_law", {}).get("alpha", alpha_spec)
            rec = _simulate(f"img_{len(records):05d}", override, beta, a_spec, f"stress:{name}")
            _accept(rec)

    # --- Phase 1: broad coverage rotation -----------------------------------
    log_fn(f"[phase1] generating {n_base} broad-coverage images...")
    for _ in range(n_base):
        beta = beta_groups[bi % len(beta_groups)]; bi += 1
        rec = _simulate(f"img_{len(records):05d}", _BROAD, beta, alpha_spec, "broad")
        _accept(rec)

    # --- Phase 2: targeted top-up of the most-deficient cell ----------------
    log_fn(f"[phase2] targeted top-up until every cell has >= {min_gt} GT spots "
           f"(max {max_attempts} total images)...")

    def _deficient_cells() -> list[tuple[int, _Cell]]:
        out = []
        for c in cells:
            deficit = min_gt - int(counts[c.snr_idx, c.dens_idx])
            if deficit > 0:
                out.append((deficit, c))
        out.sort(key=lambda t: t[0], reverse=True)  # neediest first
        return out

    while True:
        deficient = _deficient_cells()
        if not deficient:
            break
        if len(records) >= max_attempts:
            break
        _, c = deficient[0]
        override = _targeted_override(c.snr_idx, c.dens_idx, n_snr, n_dens)
        # Brightest SNR bins: keep both channels comparable (beta=0, tight alpha)
        # so min(snr1, snr2) actually lands high; else cycle beta for coverage.
        if c.snr_idx >= n_snr - 1:
            beta, a_spec = 0.0, {"min": -0.1, "max": 0.1}
        else:
            beta = beta_groups[bi % len(beta_groups)]; bi += 1
            a_spec = alpha_spec
        rec = _simulate(f"img_{len(records):05d}", override, beta, a_spec, "targeted")
        _accept(rec)

    # --- Phase 3: top-up any under-covered beta group -----------------------
    for b in beta_groups:
        while beta_img_counts[b] < min_imgs_beta and len(records) < max_attempts:
            rec = _simulate(f"img_{len(records):05d}", _BROAD, b, alpha_spec, "beta_topup")
            _accept(rec)

    # --------------------------------------------------------------------- #
    # Emit all artifacts                                                     #
    # --------------------------------------------------------------------- #
    log_fn(f"[emit] writing {len(records)} images and tables to {out_dir} ...")
    gt_frames: list[pd.DataFrame] = []
    meta_rows: list[dict] = []        # per-spot metadata.csv rows
    beta_rows: list[dict] = []        # beta_per_image.csv rows
    image_index: list[dict] = []

    for rec in records:
        image_id = rec["image_id"]
        sim = rec["sim"]
        image = sim.image  # uint16 [2,H,W]

        # canonical npy + per-channel raw/photon TIFFs
        np.save(out_dir / "images" / f"{image_id}.npy", image)
        tifffile.imwrite(out_dir / "images_ch1_raw" / f"{image_id}.tif", image[0])
        tifffile.imwrite(out_dir / "images_ch2_raw" / f"{image_id}.tif", image[1])
        photon1 = ((image[0].astype(np.float32) - np.float32(offsets[0])) / np.float32(gains[0]))
        photon2 = ((image[1].astype(np.float32) - np.float32(offsets[1])) / np.float32(gains[1]))
        tifffile.imwrite(out_dir / "images_ch1_photon" / f"{image_id}.tif", photon1)
        tifffile.imwrite(out_dir / "images_ch2_photon" / f"{image_id}.tif", photon2)

        # full per-image meta (feature fidelity for later attach_features)
        with open(out_dir / "meta" / f"meta_{image_id}.json", "w", encoding="utf-8") as fh:
            json.dump(sim.meta, fh, indent=2)

        # TRUE background -> audit/ (NON-FAIR; debug only, git-ignored)
        bg = sim.diagnostics["background"].astype(np.float32)
        np.save(out_dir / "audit" / f"background_{image_id}.npy", bg)

        # GT schema rows (re-id image_id consistently; spot_id already per-image)
        gt = rec["gt"].copy()
        gt["image_id"] = image_id
        gt_frames.append(gt)

        # per-image stress classification
        scene = rec["scene"]
        ci_class = _channel_imbalance_class(rec["log_a1"], rec["log_a2"], bright_log10, dim_log10)
        sat_regime = _saturation_regime(image, sim.meta["n_saturated"], adc_max)
        psf_reg = _psf_regime(scene.sigma1, scene.sigma2)
        bg_type1 = _background_type(scene.background1)
        bg_type2 = _background_type(scene.background2)
        shift_mag = math.hypot(*scene.shift1) + math.hypot(*scene.shift2)
        image_bg_type = bg_type1 if bg_type1 == bg_type2 else "mixed"

        stress_counts[f"channel_imbalance:{ci_class}"] += 1
        stress_counts[f"saturation:{sat_regime}"] += 1
        stress_counts[f"psf:{psf_reg}"] += 1
        stress_counts[f"background:{image_bg_type}"] += 1
        if shift_mag > 0.5:
            stress_counts["shift:subpixel_large"] += 1

        beta_rows.append({
            "image_id": image_id, "alpha": float(scene.alpha), "beta": float(rec["beta"]),
            "beta_group_index": beta_groups.index(rec["beta"]), "n_spots": int(len(gt)),
            "profile": rec["profile"],
        })

        # per-spot metadata rows
        flags = gt["flags"].to_numpy()
        snr_b, den_b = rec["snr_bin_idx"], rec["dens_bin_idx"]
        for k in range(len(gt)):
            s_i, d_i = int(snr_b[k]), int(den_b[k])
            meta_rows.append({
                "image_id": image_id, "spot_id": int(gt["spot_id"].iloc[k]),
                "x": float(gt["x"].iloc[k]), "y": float(gt["y"].iloc[k]),
                "snr": float(rec["snr"][k]), "snr1": float(rec["snr1"][k]),
                "snr2": float(rec["snr2"][k]), "n_neighbors": float(rec["n_neighbors"][k]),
                "snr_index": s_i, "snr_bin": _bin_label(snr_edges, s_i) if s_i >= 0 else "none",
                "density_index": d_i,
                "density_bin": _bin_label(density_edges, d_i) if d_i >= 0 else "none",
                "beta": float(rec["beta"]), "beta_group_index": beta_groups.index(rec["beta"]),
                "alpha": float(scene.alpha),
                "saturated": bool("saturated" in str(flags[k])),
                "channel_imbalance_class": ci_class, "saturation_regime": sat_regime,
                "psf_regime": psf_reg, "sigma1": float(scene.sigma1), "sigma2": float(scene.sigma2),
                "sigma_mismatch": float(scene.sigma2 / max(scene.sigma1, 1e-9)),
                "shift1x": float(scene.shift1[0]), "shift1y": float(scene.shift1[1]),
                "shift2x": float(scene.shift2[0]), "shift2y": float(scene.shift2[1]),
                "background_type": image_bg_type, "background_type1": bg_type1,
                "background_type2": bg_type2,
                "bg_level1": float(scene.background1.get("level", math.nan)),
                "bg_level2": float(scene.background2.get("level", math.nan)),
                "profile": rec["profile"],
            })

        image_index.append({
            "image_id": image_id,
            "image_file": f"images/{image_id}.npy",
            "ch1_raw": f"images_ch1_raw/{image_id}.tif",
            "ch2_raw": f"images_ch2_raw/{image_id}.tif",
            "ch1_photon": f"images_ch1_photon/{image_id}.tif",
            "ch2_photon": f"images_ch2_photon/{image_id}.tif",
            "meta_file": f"meta/meta_{image_id}.json",
            "n_spots": int(len(gt)),
            "beta": float(rec["beta"]),
            "profile": rec["profile"],
        })

    # tables
    gt_all = pd.concat(gt_frames, ignore_index=True) if gt_frames else records_to_dataframe([])
    write_spots(gt_all, out_dir / "ground_truth.csv")
    pd.DataFrame(beta_rows).to_csv(out_dir / "beta_per_image.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "metadata.csv", index=False)

    # --------------------------------------------------------------------- #
    # Coverage report + freeze check                                        #
    # --------------------------------------------------------------------- #
    n_gt = int(len(gt_all))
    underpopulated = [
        {"snr_bin": _bin_label(snr_edges, c.snr_idx), "density_bin": _bin_label(density_edges, c.dens_idx),
         "snr_index": c.snr_idx, "density_index": c.dens_idx,
         "count": int(counts[c.snr_idx, c.dens_idx]), "target": min_gt}
        for c in cells if int(counts[c.snr_idx, c.dens_idx]) < min_gt
    ]

    # --------------------------------------------------------------------- #
    # Checksums (freeze means freeze)                                        #
    # --------------------------------------------------------------------- #
    log_fn("[checksums] hashing artifacts...")
    checksum_lines: list[str] = []
    dir_digests: dict[str, dict] = {}

    def _hash_dir(name: str) -> dict:
        d = out_dir / name
        files = sorted(p for p in d.glob("*") if p.is_file())
        agg = hashlib.sha256()
        total = 0
        for p in files:
            h = _sha256_file(p)
            rel = f"{name}/{p.name}"
            checksum_lines.append(f"{h}  {rel}")
            agg.update(f"{rel}:{h}\n".encode())
            total += p.stat().st_size
        return {"n_files": len(files), "bytes": total, "digest": agg.hexdigest()}

    for name in IMAGE_DIRS + ("meta",):
        dir_digests[name] = _hash_dir(name)

    csv_checksums = {}
    for fn in ("ground_truth.csv", "metadata.csv", "beta_per_image.csv"):
        h = _sha256_file(out_dir / fn)
        csv_checksums[fn] = h
        checksum_lines.append(f"{h}  {fn}")
    (out_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # --------------------------------------------------------------------- #
    # Manifest                                                              #
    # --------------------------------------------------------------------- #
    manifest = {
        "name": str(bench_cfg.get("name", out_dir.name)),
        "frozen": True,
        "purpose": ("FROZEN benchmark/test set for FINAL REPORTING and EXTERNAL-METHOD "
                    "comparison ONLY. NEVER use for checkpoint selection (that is the "
                    "training-validation set, a different seed/manifest). Once external "
                    "methods start running on it, it must not change -- verify against the "
                    "checksums below (see verify_benchmark_set)."),
        "git_commit": _git_commit(),
        "seed": seed,
        "shape": [shape[0], shape[1]],
        "schema_columns": list(SCHEMA_COLUMNS),
        "n_images": len(records),
        "n_gt_spots": n_gt,
        # frozen binning (the single agreed binning; used identically downstream)
        "snr_bin_edges": [(_fmt_edge(e)) for e in snr_edges],
        "density_bin_edges": [(_fmt_edge(e)) for e in density_edges],
        "density_bin_edges_data_driven": data_driven,
        "density_radius_px": density_radius,
        "beta_groups": beta_groups,
        # coverage
        "min_gt_per_cell": min_gt,
        "counts_snr_x_density": _counts_2d(counts, snr_edges, density_edges),
        "counts_per_snr_bin": _counts_axis(counts.sum(axis=1), snr_edges),
        "counts_per_density_bin": _counts_axis(counts.sum(axis=0), density_edges),
        "counts_per_beta_group": {str(b): int(beta_img_counts[b]) for b in beta_groups},
        "counts_per_stress_flag": _grouped_stress(stress_counts),
        "underpopulated_cells": underpopulated,
        # conventions
        "file_format": {
            "images/<id>.npy": "canonical two-channel array, uint16 [2,H,W]",
            "images_ch{1,2}_raw/<id>.tif": f"observed detector counts, uint16 [0,{adc_max}]",
            "images_ch{1,2}_photon/<id>.tif": "offset-subtracted, gain-corrected photon-proportional, float32",
            "meta/meta_<id>.json": "full per-image simulator metadata (for attach_features parity)",
            "audit/background_<id>.npy": "TRUE simulator background [2,H,W] float32 -- NON-FAIR, debug only",
            "ground_truth.csv": "canonical spotpipe.schema GT table for all images",
            "beta_per_image.csv": "true alpha/beta/beta-group per image",
            "metadata.csv": "per-spot binning/filtering metadata (SNR, density, bins, stress flags)",
        },
        "raw_vs_photon_convention": {
            "raw": ("Observed detector-count images (uint16). External single-channel tools "
                    "(Spotiflow/DECODE) DETECT/LOCALIZE on these -- that is what real "
                    "microscopy provides."),
            "photon": ("Adapters EXTRACT intensity I1/I2 from the photon-proportional images "
                       "using their OWN local-background / aperture / PSF-fit estimator. "
                       "Adapters must never divide raw counts, and never use audit/ background "
                       "unless explicitly labeled oracle."),
            "photon_correction": "photon_k = (raw_counts - offset_k) / gain_k  (per channel)",
            "photon_correction_note": ("Linear pedestal+gain correction only; it does NOT invert "
                                       "the saturation knee (a fair method cannot know it), so "
                                       "saturated pixels stay compressed. Background structure is "
                                       "preserved on purpose -- adapters estimate it themselves."),
            "offsets": {"ch1": offsets[0], "ch2": offsets[1]},
            "gains": {"ch1": gains[0], "ch2": gains[1]},
        },
        # full config + detector
        "simulator_config": {"image": img_cfg, "detector": base_config.get("detector", {}),
                             "scene": base_scene},
        "benchmark_set_config": bench_cfg,
        "detector": det_meta,
        # checksums
        "checksums": {
            "csv": csv_checksums,
            "directories": dir_digests,
            "checksums_file": "checksums.sha256",
            "note": "Re-verify with spotpipe.simulator.benchmark_set.verify_benchmark_set.",
        },
        "images": image_index,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    log_fn(f"[done] {len(records)} images, {n_gt} GT spots written to {out_dir}.")
    return manifest


def _fmt_edge(e: float):
    return "inf" if math.isinf(e) else float(e)


def _counts_2d(counts: np.ndarray, snr_edges, density_edges) -> list[dict]:
    rows = []
    for s in range(counts.shape[0]):
        for d in range(counts.shape[1]):
            rows.append({
                "snr_bin": _bin_label(snr_edges, s), "density_bin": _bin_label(density_edges, d),
                "snr_index": s, "density_index": d, "count": int(counts[s, d]),
            })
    return rows


def _counts_axis(vec: np.ndarray, edges) -> list[dict]:
    return [{"bin": _bin_label(edges, i), "index": i, "count": int(vec[i])}
            for i in range(len(vec))]


def _grouped_stress(stress_counts: Counter) -> dict:
    grouped: dict[str, dict] = {}
    for key, n in sorted(stress_counts.items()):
        group, _, value = key.partition(":")
        grouped.setdefault(group, {})[value] = int(n)
    return grouped


# --------------------------------------------------------------------------- #
# Reporting & verification                                                      #
# --------------------------------------------------------------------------- #
def stratification_report(manifest: dict) -> str:
    """Human-readable stratification report (counts per cell / beta / stress)."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"STRATIFICATION REPORT -- {manifest['name']} "
                 f"({manifest['n_images']} images, {manifest['n_gt_spots']} GT spots)")
    lines.append("=" * 72)

    # SNR x density grid
    snr_labels = [r["bin"] for r in manifest["counts_per_snr_bin"]]
    den_labels = [r["bin"] for r in manifest["counts_per_density_bin"]]
    grid = {(r["snr_index"], r["density_index"]): r["count"] for r in manifest["counts_snr_x_density"]}
    lines.append(f"\nGT spots per SNR x density cell (target >= {manifest['min_gt_per_cell']}):")
    header = "  SNR \\ dens |" + "".join(f"{lab:>14}" for lab in den_labels)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for si, slab in enumerate(snr_labels):
        cells = []
        for di in range(len(den_labels)):
            c = grid.get((si, di), 0)
            mark = "" if c >= manifest["min_gt_per_cell"] else "*"
            cells.append(f"{c:>13}{mark}")
        lines.append(f"  {slab:>10} |" + "".join(cells))
    lines.append("  (* = below target)")

    lines.append("\nGT spots per SNR bin:")
    for r in manifest["counts_per_snr_bin"]:
        lines.append(f"    {r['bin']:>12}: {r['count']:>8}")
    lines.append("\nGT spots per density bin:")
    for r in manifest["counts_per_density_bin"]:
        lines.append(f"    {r['bin']:>12}: {r['count']:>8}")

    lines.append("\nImages per beta group:")
    for b, n in manifest["counts_per_beta_group"].items():
        lines.append(f"    beta={b:>5}: {n:>4} images")

    lines.append("\nImages per stress-regime flag:")
    for group, vals in manifest["counts_per_stress_flag"].items():
        lines.append(f"    {group}:")
        for v, n in sorted(vals.items()):
            lines.append(f"        {v:>22}: {n:>4}")

    up = manifest["underpopulated_cells"]
    if up:
        lines.append(f"\nFAILED: {len(up)} cell(s) below target {manifest['min_gt_per_cell']}:")
        for c in up:
            lines.append(f"    SNR {c['snr_bin']} x density {c['density_bin']}: "
                         f"{c['count']} < {c['target']}")
    else:
        lines.append(f"\nOK: every SNR x density cell has >= {manifest['min_gt_per_cell']} GT spots.")
    lines.append("=" * 72)
    return "\n".join(lines)


def verify_benchmark_set(out_dir: str | Path) -> tuple[bool, list[str]]:
    """Re-check a frozen set against its manifest checksums. Returns (ok, problems).

    Freeze means freeze: this recomputes the per-file and per-directory digests and
    compares them to ``manifest.json`` so a later run can confirm it is operating on
    the unmodified dataset before reporting numbers on it.
    """
    out_dir = Path(out_dir)
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    problems: list[str] = []

    # CSV checksums
    for fn, want in manifest["checksums"]["csv"].items():
        got = _sha256_file(out_dir / fn)
        if got != want:
            problems.append(f"{fn}: checksum mismatch")

    # directory digests
    for name, info in manifest["checksums"]["directories"].items():
        files = sorted(p for p in (out_dir / name).glob("*") if p.is_file())
        agg = hashlib.sha256()
        for p in files:
            agg.update(f"{name}/{p.name}:{_sha256_file(p)}\n".encode())
        if len(files) != info["n_files"]:
            problems.append(f"{name}/: file count {len(files)} != {info['n_files']}")
        elif agg.hexdigest() != info["digest"]:
            problems.append(f"{name}/: directory digest mismatch")

    return (len(problems) == 0), problems
