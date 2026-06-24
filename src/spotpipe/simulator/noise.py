"""Detector noise model for the FV3000 analog-integration PMT chain.

This module implements steps 2-7 of the forward chain (``forward_model`` builds
step 1, the clean photon-rate signal ``S_k``). Per channel ``k``:

  2. Shot noise:        ``P_k = Poisson(S_k)``                      (photons)
  3. Gain + excess:     ``M_k | P_k ~ Normal(g_k*P_k, (F_k-1)*g_k^2*P_k)`` above a
                        small threshold; ``M_k = g_k*P_k`` (Poisson-exact) below.
  4. Frame averaging:   mean of ``n_frames`` independent scans (steps 2-3),
                        which reduces ADDED-NOISE variance by ``n_frames`` while
                        preserving the mean -- NOT by scaling the signal.
  5. Soft saturation:   ``C_k = knee_k * tanh(M_k / knee_k)``  (per-channel,
                        linear well below knee, compressive above).
  6. Offset + floor:    ``D_k = C_k + offset_k + Normal(0, floor_k/sqrt(n_frames))``.
  7. Quantise + clip:   ``obs_k = clip(round(D_k), 0, adc_max)``, uint16.

Detector-physics parameters (per-channel ``gain``, ``offset``,
``excess_noise_factor``, ``saturation_knee``, ``noise_floor_sigma`` and global
``n_frames``) are FIXED instrument constants, sampled ONCE per dataset (an
instrument, not a scene), with at most narrow jitter. Broad randomisation here
is forbidden -- we do NOT want the network to be gain-invariant (see CLAUDE.md).
The two channels are imaged at different PMT voltages, so ``g_1 != g_2`` and
``knee_1 != knee_2``.

Note on step 3 (excess noise). ``F`` is defined as the ratio of total variance
to shot-noise variance (the standard photon-transfer-curve definition), so the
config's ``excess_noise_factor`` means what a measured F would mean. The
Poisson draw already carries the shot-noise variance ``g^2*S``; the gain stage
adds only the EXCESS ``(F-1)*g^2*P`` about ``g*P``, giving total variance
``F*g^2*S``. Frame averaging (step 4) is realised as the literal mean of
``n_frames`` independent scans, so the noise variance falls by ``n_frames``
exactly and low counts stay Poisson-faithful (averaging 3 Poisson draws == mean
of 3 scans).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ChannelDetector",
    "DetectorParams",
    "sample_detector_params",
    "apply_detector_noise",
    "transfer_curve",
]


@dataclass(frozen=True)
class ChannelDetector:
    """Fixed detector constants for one channel (one PMT voltage)."""

    gain: float                 # analog gain g_k (counts per photon-proportional unit)
    offset: float               # analog baseline / pedestal (counts)
    excess_noise_factor: float  # F_k > 1: PMT stochastic-gain noise inflation
    saturation_knee: float      # knee_k in gained-signal units (soft tanh knee)
    noise_floor_sigma: float    # read-noise floor sigma per scan (counts)


@dataclass(frozen=True)
class DetectorParams:
    """Fixed detector constants for the whole instrument (both channels)."""

    ch1: ChannelDetector
    ch2: ChannelDetector
    n_frames: int = 3                     # integration count (mean of 3 scans)
    poisson_gaussian_threshold: float = 20.0  # P below which gain stays Poisson-exact
    adc_max: int = 4095                   # 12-bit clip ceiling

    def channel(self, k: int) -> ChannelDetector:
        if k == 1:
            return self.ch1
        if k == 2:
            return self.ch2
        raise ValueError(f"channel must be 1 or 2, got {k}")


def _jitter(value: float, frac: float, rng: np.random.Generator) -> float:
    """Apply at most +-``frac`` uniform jitter to a fixed constant.

    ``frac`` MUST stay narrow (instrument-to-instrument variation), never broad
    randomisation -- the network must remain gain-aware, not gain-invariant.
    """
    if frac <= 0.0:
        return float(value)
    return float(value) * float(rng.uniform(1.0 - frac, 1.0 + frac))


def sample_detector_params(cfg: dict, rng: np.random.Generator) -> DetectorParams:
    """Build :class:`DetectorParams` from the ``detector:`` config block.

    Values are the fixed instrument constants; ``jitter_frac`` applies only a
    narrow per-instrument perturbation (default 0). Sample this ONCE per dataset
    and reuse it for every image -- detector constants are not scene variables.
    """
    frac = float(cfg.get("jitter_frac", 0.0))
    if frac > 0.25:
        raise ValueError(
            f"detector jitter_frac={frac} is too broad; detector constants must be "
            "fixed or only narrowly jittered (<=0.25). Broad gain randomisation is "
            "forbidden (no gain-invariance)."
        )

    def _channel(block: dict) -> ChannelDetector:
        return ChannelDetector(
            gain=_jitter(block["gain"], frac, rng),
            offset=_jitter(block["offset"], frac, rng),
            excess_noise_factor=_jitter(block["excess_noise_factor"], frac, rng),
            saturation_knee=_jitter(block["saturation_knee"], frac, rng),
            noise_floor_sigma=_jitter(block["noise_floor_sigma"], frac, rng),
        )

    return DetectorParams(
        ch1=_channel(cfg["ch1"]),
        ch2=_channel(cfg["ch2"]),
        n_frames=int(cfg.get("n_frames", 3)),
        poisson_gaussian_threshold=float(cfg.get("poisson_gaussian_threshold", 20.0)),
        adc_max=int(cfg.get("adc_max", 4095)),
    )


def _one_scan(
    signal: np.ndarray,
    ch: ChannelDetector,
    rng: np.random.Generator,
    threshold: float,
) -> np.ndarray:
    """One PMT scan: shot noise (step 2) + analog gain & excess noise (step 3)."""
    photons = rng.poisson(signal)                      # step 2: P ~ Poisson(S)
    gained = ch.gain * photons.astype(np.float64)      # mean g*P
    # step 3: add only the EXCESS gain variance (F-1)*g^2*P on top of the
    # shot-noise variance already carried by the Poisson draw. F is defined as
    # the ratio of total variance to shot-noise variance (a measured
    # photon-transfer-curve F), so the total per-pixel variance comes out to
    #   Var(g*P) + (F-1)*g^2*P = g^2*S + (F-1)*g^2*S = F*g^2*S.
    # Applied only where P is large enough for the Gaussian approximation; below
    # threshold keep g*P exactly (Poisson-exact), which also avoids spurious
    # negative draws at low counts. (F-1) is clamped at 0 -- F<1 is unphysical.
    high = photons >= threshold
    if np.any(high):
        excess = max(ch.excess_noise_factor - 1.0, 0.0)
        std = ch.gain * np.sqrt(excess * photons[high].astype(np.float64))
        gained[high] += rng.standard_normal(int(high.sum())) * std
    return gained


def apply_detector_noise(
    signal: np.ndarray,
    ch: ChannelDetector,
    rng: np.random.Generator,
    *,
    n_frames: int = 3,
    threshold: float = 20.0,
    adc_max: int = 4095,
    return_diagnostics: bool = False,
):
    """Run detector steps 2-7 on one channel's clean photon-rate signal.

    Parameters
    ----------
    signal : clean photon-proportional signal ``S_k`` (>= 0), 2-D.
    ch : fixed detector constants for this channel.
    rng : random generator.
    n_frames : integration count (mean of this many scans).
    threshold : Poisson count below which the gain stage stays Poisson-exact.
    adc_max : 12-bit clip ceiling.
    return_diagnostics : if True, also return clean transfer-curve fields used
        by the eyeball script (no extra randomness consumed).

    Returns
    -------
    obs : uint16 observed counts, or ``(obs, diagnostics)`` if requested.
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = np.clip(signal, 0.0, None)  # photon rate cannot be negative

    # steps 2-4: mean of n_frames independent scans. Averaging preserves the
    # mean (g*S) and divides the added-noise variance by n_frames -- the
    # "integration count = 3" frame averaging, done by reducing noise, never by
    # scaling the signal.
    gained = _one_scan(signal, ch, rng, threshold)
    for _ in range(1, max(n_frames, 1)):
        gained += _one_scan(signal, ch, rng, threshold)
    gained /= max(n_frames, 1)

    # step 5: per-channel soft saturation knee. tanh is ~linear for M << knee
    # and compresses for M >~ knee; this intensity-dependent compression is a
    # key (per-channel) ratio-bias source and must be present.
    knee = ch.saturation_knee
    post_knee = knee * np.tanh(gained / knee)

    # step 6: analog offset + read-noise floor. The floor sigma is divided by
    # sqrt(n_frames) to match the frame averaging of steps 2-4.
    floor = rng.standard_normal(signal.shape) * (ch.noise_floor_sigma / np.sqrt(max(n_frames, 1)))
    detected = post_knee + ch.offset + floor

    # step 7: quantise to integer counts and clip to 12-bit.
    obs = np.clip(np.rint(detected), 0, adc_max).astype(np.uint16)

    if not return_diagnostics:
        return obs

    diagnostics = {
        "gained_signal": gained,                          # averaged M (noisy)
        "post_knee": post_knee,                           # C after soft knee
        "clean_gained": ch.gain * signal,                 # noise-free g*S
        "knee": float(knee),
        "offset": float(ch.offset),
        "gain": float(ch.gain),
        "adc_max": int(adc_max),
    }
    return obs, diagnostics


def transfer_curve(ch: ChannelDetector, gained: np.ndarray, adc_max: int = 4095) -> np.ndarray:
    """Deterministic noise-free count for a given gained signal ``M`` (steps 5-7).

    ``clip(round(knee*tanh(M/knee) + offset), 0, adc_max)``. Used to draw the
    per-channel saturation transfer curve in the eyeball script.
    """
    gained = np.asarray(gained, dtype=np.float64)
    post_knee = ch.saturation_knee * np.tanh(gained / ch.saturation_knee)
    return np.clip(np.rint(post_knee + ch.offset), 0, adc_max)
