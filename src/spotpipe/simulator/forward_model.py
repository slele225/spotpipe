"""FV3000 two-channel forward model.

Generates synthetic two-channel confocal images from a *scene* description and a
*detector* description, in photon-proportional bookkeeping units, then derives
observed 12-bit counts via the detector chain in :mod:`spotpipe.simulator.noise`.

The full chain (per channel ``k``):

  1. clean photon-rate signal  ``S_k = sum_i A_{k,i} PSF_k(x-x_i-shift_k, ...) + B_k``
  2-7. detector chain (shot noise -> gain/excess -> frame averaging -> soft knee
       -> offset/floor -> quantise/clip), implemented in ``noise.py``.

Conventions (see CLAUDE.md):

* Detector-physics parameters are FIXED instrument constants (sampled once per
  dataset; passed in here as ``detector``). Scene parameters are randomised
  WIDELY per image (``sample_scene_params``).
* ``beta`` (the ratio-law slope) is varied per image, including 0 and negative.
  If it were fixed the network would learn it as a prior and bias the very
  quantity we measure.
* The ground-truth per-spot TOTAL INTEGRATED intensities ``A_{1,i}``/``A_{2,i}``
  are kept EXACT, in photon-proportional units, untouched by detector params.
  They are the supervision target for the intensity heads. The ground-truth
  ratio is ``A_{2,i}/A_{1,i}`` -- computed from intensities, never from counts.

Generative ratio law (per image: draw ``alpha, beta``; per spot draw ``A_1``)::

    log A_2 = log A_1 + alpha + beta * log A_1 + Normal(0, scatter_std)
            = (1 + beta) * log A_1 + alpha + Normal(0, scatter_std)

so regressing ``log A_2`` on ``log A_1`` has slope ``1 + beta`` (and regressing
``log_ratio = log A_2 - log A_1`` on ``log A_1`` has slope ``beta``). This is the
relationship the downstream slope analysis must recover from per-spot estimates
-- it is never trained on (no slope loss in phase 1).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from spotpipe.schema import SpotRecord, records_to_dataframe
from spotpipe.simulator import backgrounds, noise, psf

__all__ = ["SceneParams", "SimulatedImage", "sample_scene_params", "simulate_image"]


# --------------------------------------------------------------------------- #
# Scene parameters (realised per image)                                        #
# --------------------------------------------------------------------------- #
@dataclass
class SceneParams:
    """Realised (sampled) scene parameters for a single image.

    These are the WIDELY randomised domain-randomisation variables. Per-spot
    arrays (positions, intensities) are drawn inside :func:`simulate_image`; this
    holds the per-image scalars plus the intensity-distribution descriptors.
    """

    n_spots: int
    density: float                       # spots per pixel
    alpha: float                         # ratio-law intercept
    beta: float                          # ratio-law slope (incl. 0 / negative)
    scatter_std: float                   # per-spot log-ratio scatter (nat. log)
    sigma1: float                        # PSF sigma, channel 1 (px)
    sigma2: float                        # PSF sigma, channel 2 (px) -- mismatched
    shift1: tuple[float, float]          # registration offset ch1 (dx, dy) px
    shift2: tuple[float, float]          # registration offset ch2 (dx, dy) px
    clustering: str                      # 'uniform' | 'clustered'
    n_clusters: int
    cluster_sigma_px: float
    intensity_log10_min: float           # A_1 sampling range (log10 photons)
    intensity_log10_max: float
    intensity_dim_bias: float            # >1 over-samples the dim tail
    background1: dict = field(default_factory=dict)
    background2: dict = field(default_factory=dict)


@dataclass
class SimulatedImage:
    """One simulated field: image, ground-truth spot table, metadata."""

    image: np.ndarray                    # [2, H, W] uint16 (channel-first)
    spots: pd.DataFrame                  # canonical schema, ground-truth rows
    meta: dict
    diagnostics: dict | None = None      # populated only when requested (eyeball)


# --------------------------------------------------------------------------- #
# Config helpers                                                               #
# --------------------------------------------------------------------------- #
def _uniform(rng: np.random.Generator, spec: dict, default_min=0.0, default_max=0.0) -> float:
    lo = float(spec.get("min", default_min))
    hi = float(spec.get("max", default_max))
    if hi < lo:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


def _int_uniform(rng: np.random.Generator, spec: dict, default_min=1, default_max=1) -> int:
    lo = int(spec.get("min", default_min))
    hi = int(spec.get("max", default_max))
    if hi < lo:
        lo, hi = hi, lo
    return int(rng.integers(lo, hi + 1))


# --------------------------------------------------------------------------- #
# Scene sampling                                                               #
# --------------------------------------------------------------------------- #
def sample_scene_params(cfg: dict, rng: np.random.Generator, shape: tuple[int, int]) -> SceneParams:
    """Draw one image's scene parameters from the ``scene:`` config block.

    Domain randomisation: density (incl. high/overlapping), intensity (with a
    deliberately over-sampled dim tail), the ratio law (alpha, beta incl. 0 and
    negative, scatter), per-channel PSF sigma with a C1-vs-C2 mismatch,
    registration shift, background, and clustered-vs-uniform geometry.
    """
    height, width = shape

    # --- density (spots per pixel), log-uniform; the dense regime is over-sampled.
    dcfg = cfg.get("density", {})
    dmin, dmax = float(dcfg.get("min", 0.001)), float(dcfg.get("max", 0.01))
    log_density = rng.uniform(math.log(dmin), math.log(dmax))
    density = math.exp(log_density)
    if rng.random() < float(cfg.get("oversample_dense_fraction", 0.0)):
        # Force the upper density octave to over-sample the high-overlap regime
        # relative to plain log-uniform (that is the regime the project targets).
        density = math.exp(rng.uniform(0.5 * (math.log(dmin) + math.log(dmax)), math.log(dmax)))
    n_spots = max(int(round(density * height * width)), 0)

    # --- ratio law: alpha, beta (per image; beta INCLUDES 0 and negative), scatter.
    rl = cfg.get("ratio_law", {})
    alpha = _uniform(rng, rl.get("alpha", {}), -0.5, 0.5)
    beta = _uniform(rng, rl.get("beta", {}), -0.3, 0.3)
    scatter_std = _uniform(rng, rl.get("scatter_std", {}), 0.05, 0.2)

    # --- PSF: per-channel sigma with a deliberate C1-vs-C2 mismatch.
    pcfg = cfg.get("psf", {})
    sigma1 = _uniform(rng, pcfg.get("sigma1", {}), 1.0, 1.6)
    mismatch = _uniform(rng, pcfg.get("c2_sigma_mismatch", {}), 1.05, 1.3)
    sigma2 = sigma1 * mismatch

    # --- registration shift (per channel, sub-pixel).
    shift_max = float(cfg.get("registration_shift", {}).get("max_px", 1.0))
    shift1 = (float(rng.uniform(-shift_max, shift_max)), float(rng.uniform(-shift_max, shift_max)))
    shift2 = (float(rng.uniform(-shift_max, shift_max)), float(rng.uniform(-shift_max, shift_max)))

    # --- clustering geometry.
    ccfg = cfg.get("clustering", {})
    if rng.random() < float(ccfg.get("cluster_prob", 0.0)):
        clustering = "clustered"
        n_clusters = _int_uniform(rng, ccfg.get("n_clusters", {}), 2, 8)
        cluster_sigma_px = _uniform(rng, ccfg.get("cluster_sigma_px", {}), 6.0, 24.0)
    else:
        clustering = "uniform"
        n_clusters = 0
        cluster_sigma_px = 0.0

    # --- intensity distribution descriptors (A_1, log10 photons).
    icfg = cfg.get("intensity", {})
    log10_min = float(icfg.get("log10_min", 1.3))
    log10_max = float(icfg.get("log10_max", 4.0))
    dim_bias = float(icfg.get("dim_bias", 1.0))

    # --- background, per channel (drawn independently).
    bcfg = cfg.get("background", {})

    def _bg() -> dict:
        return {
            "level": _uniform(rng, bcfg.get("level", {}), 2.0, 20.0),
            "gradient_frac": _uniform(rng, bcfg.get("gradient_frac", {}), 0.0, 0.5),
            "structure_frac": _uniform(rng, bcfg.get("structure_frac", {}), 0.0, 0.5),
            "structure_scale_px": _uniform(rng, bcfg.get("structure_scale_px", {}), 16.0, 64.0),
        }

    return SceneParams(
        n_spots=n_spots,
        density=density,
        alpha=alpha,
        beta=beta,
        scatter_std=scatter_std,
        sigma1=sigma1,
        sigma2=sigma2,
        shift1=shift1,
        shift2=shift2,
        clustering=clustering,
        n_clusters=n_clusters,
        cluster_sigma_px=cluster_sigma_px,
        intensity_log10_min=log10_min,
        intensity_log10_max=log10_max,
        intensity_dim_bias=dim_bias,
        background1=_bg(),
        background2=_bg(),
    )


def _sample_positions(scene: SceneParams, shape: tuple[int, int], rng: np.random.Generator):
    """Sub-pixel spot positions (xs, ys), uniform or clustered."""
    height, width = shape
    n = scene.n_spots
    if n == 0:
        return np.empty(0), np.empty(0)
    if scene.clustering == "clustered" and scene.n_clusters > 0:
        cx = rng.uniform(0.0, width - 1, scene.n_clusters)
        cy = rng.uniform(0.0, height - 1, scene.n_clusters)
        which = rng.integers(0, scene.n_clusters, n)
        xs = rng.normal(cx[which], scene.cluster_sigma_px)
        ys = rng.normal(cy[which], scene.cluster_sigma_px)
    else:
        xs = rng.uniform(0.0, width - 1, n)
        ys = rng.uniform(0.0, height - 1, n)
    np.clip(xs, 0.0, width - 1, out=xs)
    np.clip(ys, 0.0, height - 1, out=ys)
    return xs, ys


def _sample_intensities(scene: SceneParams, rng: np.random.Generator) -> np.ndarray:
    """Per-spot channel-1 total integrated intensity ``A_1`` (photons).

    ``log10 A_1 = log10_min + (log10_max - log10_min) * u**dim_bias`` with
    ``u ~ Uniform(0,1)``. ``dim_bias == 1`` is log-uniform; ``> 1`` pushes mass
    toward the dim end, over-sampling the dim tail relative to uniform.
    """
    n = scene.n_spots
    if n == 0:
        return np.empty(0)
    u = rng.random(n)
    log10_a1 = scene.intensity_log10_min + (
        scene.intensity_log10_max - scene.intensity_log10_min
    ) * (u ** scene.intensity_dim_bias)
    return np.power(10.0, log10_a1)


# --------------------------------------------------------------------------- #
# Simulate one image                                                           #
# --------------------------------------------------------------------------- #
def simulate_image(
    *,
    image_id: str,
    shape: tuple[int, int],
    scene: SceneParams,
    detector: noise.DetectorParams,
    rng: np.random.Generator,
    with_diagnostics: bool = False,
) -> SimulatedImage:
    """Render one two-channel image plus its ground-truth spot table.

    ``detector`` is the FIXED instrument (sampled once per dataset); ``scene`` is
    this image's randomised scene. ``rng`` drives all per-spot draws and the
    detector noise so the result is reproducible from the generator's seed.
    """
    height, width = shape

    # --- per-spot scene draws -------------------------------------------------
    xs, ys = _sample_positions(scene, shape, rng)
    a1 = _sample_intensities(scene, rng)
    n = scene.n_spots

    log_a1 = np.log(a1) if n else np.empty(0)
    # ratio law: log A_2 = log A_1 + alpha + beta*log A_1 + N(0, scatter_std)
    log_a2 = (
        log_a1 + scene.alpha + scene.beta * log_a1 + rng.normal(0.0, scene.scatter_std, n)
        if n
        else np.empty(0)
    )
    a2 = np.exp(log_a2) if n else np.empty(0)

    # --- clean photon-rate signals S_k = spots + background -------------------
    b1 = backgrounds.make_parametric_background(shape, rng, **scene.background1)
    b2 = backgrounds.make_parametric_background(shape, rng, **scene.background2)
    s1 = b1 + psf.render_channel(shape, xs, ys, a1, scene.sigma1, shift=scene.shift1)
    s2 = b2 + psf.render_channel(shape, xs, ys, a2, scene.sigma2, shift=scene.shift2)

    # --- detector chain (steps 2-7) per channel ------------------------------
    obs1, diag1 = noise.apply_detector_noise(
        s1, detector.ch1, rng,
        n_frames=detector.n_frames, threshold=detector.poisson_gaussian_threshold,
        adc_max=detector.adc_max, return_diagnostics=True,
    )
    obs2, diag2 = noise.apply_detector_noise(
        s2, detector.ch2, rng,
        n_frames=detector.n_frames, threshold=detector.poisson_gaussian_threshold,
        adc_max=detector.adc_max, return_diagnostics=True,
    )
    image = np.stack([obs1, obs2], axis=0).astype(np.uint16)

    # --- per-spot clean peak (for saturation flagging) ------------------------
    # Own-contribution peak photon + local background, then gained. Ignores
    # neighbour overlap, so it is a lower bound on the true peak -- good enough
    # to flag spots driven into each channel's compressive knee.
    if n:
        ix = np.clip(np.round(xs).astype(int), 0, width - 1)
        iy = np.clip(np.round(ys).astype(int), 0, height - 1)
        peak1 = a1 * psf.gaussian_peak_fraction(scene.sigma1) + b1[iy, ix]
        peak2 = a2 * psf.gaussian_peak_fraction(scene.sigma2) + b2[iy, ix]
        mpeak1 = detector.ch1.gain * peak1
        mpeak2 = detector.ch2.gain * peak2
        saturated = (mpeak1 >= detector.ch1.saturation_knee) | (mpeak2 >= detector.ch2.saturation_knee)
    else:
        mpeak1 = mpeak2 = np.empty(0)
        saturated = np.empty(0, dtype=bool)

    # --- ground-truth spot table in the canonical schema ----------------------
    spots = _build_ground_truth_table(
        image_id=image_id, xs=xs, ys=ys, log_a1=log_a1, log_a2=log_a2,
        saturated=saturated,
    )

    meta = {
        "image_id": image_id,
        "shape": [height, width],
        "n_spots": int(n),
        "n_saturated": int(np.count_nonzero(saturated)) if n else 0,
        # True per-channel PSF sigma lives here (constant per image), NOT in the
        # schema's sigma*_hat columns -- those mean "model estimate" only.
        "ground_truth_sigma": {"sigma1": scene.sigma1, "sigma2": scene.sigma2},
        "scene": _scene_to_meta(scene),
        "detector": _detector_to_meta(detector),
    }

    diagnostics = None
    if with_diagnostics:
        diagnostics = {
            "clean_signal": np.stack([s1, s2], axis=0),
            "background": np.stack([b1, b2], axis=0),
            "xs": xs, "ys": ys,
            "a1": a1, "a2": a2, "log_a1": log_a1, "log_a2": log_a2,
            "mpeak1": mpeak1, "mpeak2": mpeak2, "saturated": saturated,
            "ch1": diag1, "ch2": diag2,
        }

    return SimulatedImage(image=image, spots=spots, meta=meta, diagnostics=diagnostics)


def _build_ground_truth_table(
    *,
    image_id: str,
    xs: np.ndarray,
    ys: np.ndarray,
    log_a1: np.ndarray,
    log_a2: np.ndarray,
    saturated: np.ndarray,
) -> pd.DataFrame:
    """Assemble GT spots in the exact canonical schema.

    True columns are filled from the simulator's exact bookkeeping (x, y, logI1,
    logI2, and the derived I1/I2/log_ratio/ratio). Every ``_hat`` / prediction
    column means "model estimate" only, so it is left NaN for ground-truth rows:
    ``p_detect``, ``sigma1_hat``/``sigma2_hat``, and ``uncertainty1``/
    ``uncertainty2``. The true per-channel PSF sigma is recorded in the per-image
    metadata (``meta['ground_truth_sigma']``) instead, where its meaning is
    unambiguous. Rows are flagged ``ground_truth`` (plus ``saturated`` where the
    spot reaches a channel's knee).
    """
    records = []
    for i in range(len(xs)):
        flags = "ground_truth"
        if bool(saturated[i]):
            flags += ",saturated"
        records.append(
            SpotRecord.from_logs(
                image_id=image_id,
                spot_id=i,
                x=float(xs[i]),
                y=float(ys[i]),
                p_detect=math.nan,            # prediction-only: NA for ground truth
                logI1=float(log_a1[i]),
                logI2=float(log_a2[i]),
                sigma1_hat=math.nan,          # model estimate only: NA for ground truth
                sigma2_hat=math.nan,
                uncertainty1=math.nan,        # prediction-only: NA for ground truth
                uncertainty2=math.nan,
                flags=flags,
            )
        )
    return records_to_dataframe(records)


def _scene_to_meta(scene: SceneParams) -> dict:
    d = asdict(scene)
    d["shift1"] = list(scene.shift1)
    d["shift2"] = list(scene.shift2)
    return d


def _detector_to_meta(detector: noise.DetectorParams) -> dict:
    return {
        "n_frames": detector.n_frames,
        "poisson_gaussian_threshold": detector.poisson_gaussian_threshold,
        "adc_max": detector.adc_max,
        "ch1": asdict(detector.ch1),
        "ch2": asdict(detector.ch2),
    }
