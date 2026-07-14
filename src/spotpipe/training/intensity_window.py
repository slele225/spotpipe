"""Per-image detector-gain sampling + saturation-safe intensity-range solving.

This is the single tested home for the two prompt-mandated per-image departures
from the old training distribution (measured-detector retrain, CHANGE 1 + 2):

* **CHANGE 1 -- domain-randomised gains.** The old simulator sampled ONE detector
  per dataset and forbade broad gain jitter *by design* ("the network must remain
  gain-aware, not gain-invariant", ``simulator/noise.py``). We deliberately
  REVERSE that here: gains are drawn per image, both channels independently, over
  a wide range, so the model is robust to a future PMT-voltage change (gain is
  exponential in voltage, so the future gain is not predictable). We do NOT edit
  vendored code -- we construct the frozen ``noise.DetectorParams`` /
  ``noise.ChannelDetector`` dataclasses directly with our sampled gains. Recorded
  in PROVENANCE.md.

* **CHANGE 2 -- intensity range solved per image from the sampled gains.** The old
  A1 range ([20, 7943] photons) is physically impossible at the measured protein
  gain (~124 ADU/photon clips ch2 at ~32 photons peak). Because the gains are now
  randomised, the saturation-safe A1 ceiling depends on the sampled gains and must
  be solved per image so NO spot clips EITHER channel. The lipid channel also
  clips (at ``adc_max/gain1`` photons peak), so both channels are checked. The
  protein ceiling constrains A1 from above via the ratio law
  ``log A2 = sim_intercept + (1 + sim_log_slope)*log A1``.

The clipping model mirrors the vendored ``forward_model.simulate_image`` saturation
flag exactly: a spot's clean gained peak is ``gain_k * (A_k * peak_fraction_k +
background)`` and it saturates when that reaches the channel's ``saturation_knee``
(the measured detector pins ``knee = adc_max - offset``; see ``configs/train.yaml``).
``peak_fraction_k = 1/(2*pi*sigma_k^2)`` so the PSF width enters the condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spotpipe.simulator import noise, psf

__all__ = [
    "DetectorConstants",
    "sample_image_detector",
    "solve_a1_ceiling",
    "clean_gained_peak",
]


@dataclass(frozen=True)
class DetectorConstants:
    """The HELD-fixed measured-detector constants + the per-image gain ranges.

    Everything except the gains is held at the measured value (CHANGE 1): offset,
    read-noise floor (``sqrt(read_var)``), saturation knee (``adc_max - offset``),
    excess-noise factor (1.0 -- the variance-matching gain already folds in PMT
    excess), ``n_frames``, and ``adc_max``. Only ``gain`` is drawn per image, per
    channel, independently, so the gain RATIO also varies (the model must not lean
    on a fixed gain relationship to recover ``log(A2/A1)``).
    """

    gain1_range: tuple[float, float]        # ch1 = LIPID (561), ADU/photon
    gain2_range: tuple[float, float]        # ch2 = PROTEIN (488), ADU/photon
    offset1: float
    offset2: float
    read_var1: float                        # ADU^2 (noise_floor_sigma = sqrt)
    read_var2: float
    saturation_knee1: float
    saturation_knee2: float
    excess_noise_factor1: float = 1.0
    excess_noise_factor2: float = 1.0
    n_frames: int = 3
    poisson_gaussian_threshold: float = 20.0
    adc_max: int = 4095

    @property
    def floor1(self) -> float:
        return math.sqrt(float(self.read_var1))

    @property
    def floor2(self) -> float:
        return math.sqrt(float(self.read_var2))

    @classmethod
    def from_config(cls, det_cfg: dict) -> "DetectorConstants":
        """Build from the ``detector:`` block of the training config.

        Reads per-channel ``gain_range: [lo, hi]`` (CHANGE 1) plus the held-fixed
        ``offset`` / ``read_var`` / ``saturation_knee`` / ``excess_noise_factor``
        and the global ``n_frames`` / ``adc_max`` / ``poisson_gaussian_threshold``.
        ``read_var`` is variance in ADU^2; the simulator wants ``noise_floor_sigma``
        (its sqrt), computed on the fly by ``floor1`` / ``floor2``.
        """
        ch1, ch2 = det_cfg["ch1"], det_cfg["ch2"]
        g1 = ch1["gain_range"]
        g2 = ch2["gain_range"]
        return cls(
            gain1_range=(float(g1[0]), float(g1[1])),
            gain2_range=(float(g2[0]), float(g2[1])),
            offset1=float(ch1["offset"]),
            offset2=float(ch2["offset"]),
            read_var1=float(ch1["read_var"]),
            read_var2=float(ch2["read_var"]),
            saturation_knee1=float(ch1["saturation_knee"]),
            saturation_knee2=float(ch2["saturation_knee"]),
            excess_noise_factor1=float(ch1.get("excess_noise_factor", 1.0)),
            excess_noise_factor2=float(ch2.get("excess_noise_factor", 1.0)),
            n_frames=int(det_cfg.get("n_frames", 3)),
            poisson_gaussian_threshold=float(det_cfg.get("poisson_gaussian_threshold", 20.0)),
            adc_max=int(det_cfg.get("adc_max", 4095)),
        )


def sample_image_detector(
    rng: np.random.Generator, consts: DetectorConstants
) -> tuple[noise.DetectorParams, float, float]:
    """Draw one image's detector: per-channel gain uniform in its range, rest fixed.

    Returns ``(DetectorParams, gain1, gain2)``. Constructs the FROZEN vendored
    dataclasses directly (no ``sample_detector_params``, which is once-per-dataset
    and forbids broad jitter) -- this is composition of the public dataclass API,
    not a vendored edit.
    """
    gain1 = float(rng.uniform(*consts.gain1_range))
    gain2 = float(rng.uniform(*consts.gain2_range))
    ch1 = noise.ChannelDetector(
        gain=gain1, offset=consts.offset1,
        excess_noise_factor=consts.excess_noise_factor1,
        saturation_knee=consts.saturation_knee1, noise_floor_sigma=consts.floor1,
    )
    ch2 = noise.ChannelDetector(
        gain=gain2, offset=consts.offset2,
        excess_noise_factor=consts.excess_noise_factor2,
        saturation_knee=consts.saturation_knee2, noise_floor_sigma=consts.floor2,
    )
    det = noise.DetectorParams(
        ch1=ch1, ch2=ch2, n_frames=consts.n_frames,
        poisson_gaussian_threshold=consts.poisson_gaussian_threshold,
        adc_max=consts.adc_max,
    )
    return det, gain1, gain2


def clean_gained_peak(a: float, gain: float, sigma: float, background: float) -> float:
    """Clean gained peak-pixel ADU of a spot: ``gain*(A*peak_fraction + bg)``.

    Mirrors the vendored ``simulate_image`` saturation flag (own-contribution
    peak + local flat background, then gained). Ignores neighbour overlap and
    detector noise, so it is the clean lower bound the ceiling solve targets.
    """
    return float(gain) * (float(a) * psf.gaussian_peak_fraction(float(sigma)) + float(background))


def solve_a1_ceiling(
    *,
    gain1: float,
    gain2: float,
    sigma1: float,
    sigma2: float,
    sim_intercept: float,
    sim_log_slope: float,
    scatter_std: float,
    background: float,
    knee1: float,
    knee2: float,
    target_frac: float = 0.85,
    scatter_sigmas: float = 3.5,
    floor_a1_photons: float = 10.0,
) -> dict:
    """Solve the widest A1 (log10 photons) that keeps BOTH channels unclipped.

    The brightest spot of intensity ``A1`` must keep its clean gained peak below
    ``target_frac * knee`` in EACH channel (headroom ``1 - target_frac`` absorbs
    Poisson + read noise, neighbour overlap and rounding, which push realised peaks
    above the clean value). ch1 sees ``A1`` directly; ch2 sees its ratio-law partner
    ``A2 = exp(sim_intercept) * A1**(1 + sim_log_slope)`` inflated by the upper tail
    of the per-spot log-ratio scatter (``exp(scatter_sigmas * scatter_std)``), so for
    a steep positive slope ch2 binds and the ceiling drops.

    Returns ``{log10_max, a1_cap_ch1_photons, a1_cap_ch2_photons, a1_cap_photons,
    binding_channel, ch1_gained_peak_at_cap, ch2_gained_peak_at_cap}``. If both caps
    fall below ``floor_a1_photons`` the window is degenerate (pinned at the floor);
    the caller reports these tight images.
    """
    pf1 = psf.gaussian_peak_fraction(sigma1)
    pf2 = psf.gaussian_peak_fraction(sigma2)

    # ch1 ceiling: A1 itself clears target_frac * knee1.
    a1_cap_ch1 = ((target_frac * knee1) / gain1 - background) / pf1

    # ch2 ceiling: solve A1 from the max unclipped INTEGRATED A2.
    a2_cap = ((target_frac * knee2) / gain2 - background) / pf2
    scatter_factor = math.exp(scatter_sigmas * float(scatter_std))
    exponent = 1.0 + float(sim_log_slope)
    if a2_cap <= 0.0:
        # ch2 clips even at A2 -> 0 (pure background over knee): impossible for the
        # measured detector (bg=2, knee=3941), but guard anyway.
        a1_cap_ch2 = 0.0
    elif exponent > 0.0:
        a2_cap_no_scatter = a2_cap / (scatter_factor * math.exp(float(sim_intercept)))
        a1_cap_ch2 = a2_cap_no_scatter ** (1.0 / exponent) if a2_cap_no_scatter > 0 else 0.0
    else:
        # slope <= -1: A2 is flat/decreasing in A1, so ch2 never binds from above.
        a1_cap_ch2 = math.inf

    a1_cap = min(a1_cap_ch1, a1_cap_ch2)
    binding = 1 if a1_cap_ch1 <= a1_cap_ch2 else 2
    a1_cap_eff = max(a1_cap, float(floor_a1_photons))
    log10_max = math.log10(a1_cap_eff)

    return {
        "log10_max": float(log10_max),
        "a1_cap_ch1_photons": float(a1_cap_ch1),
        "a1_cap_ch2_photons": (float(a1_cap_ch2) if math.isfinite(a1_cap_ch2) else math.inf),
        "a1_cap_photons": float(a1_cap),
        "binding_channel": int(binding),
        "degenerate": bool(a1_cap < floor_a1_photons),
        "ch1_gained_peak_at_cap": clean_gained_peak(a1_cap_eff, gain1, sigma1, background),
        "ch2_gained_peak_at_cap": clean_gained_peak(
            math.exp(sim_intercept) * a1_cap_eff ** exponent * scatter_factor,
            gain2, sigma2, background,
        ),
    }
