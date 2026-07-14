"""Dark-count robustness check -- SEPARATE from the main benchmark (Change 8).

The 488 / protein PMT (750 V) emits spurious single-photoelectron dark counts:
about **0.57% of pixels per frame** carry a spike of roughly one gain step above
the offset (~``offset + gain2`` ADU, with a Poisson multiplicity tail), while the
561 / lipid PMT (500 V) shows **none**. In real 488 images these look like very
dim, single-pixel "spots" in the exact channel we care most about for ``A2``.

A real spot is a PSF-shaped blob spanning several pixels; a dark count is a
single pixel. A PSF-matched detector *should* reject them -- but this check
**verifies** that rather than assuming it. It:

1. builds a handful of EMPTY two-channel fields (flat 2-photon background through
   the MEASURED detector), with a sparse single-pixel spike population injected
   into **ch2 only** (rate ~0.57% of pixels, amplitude ~one gain step above
   offset, Poisson multiplicity);
2. runs the vendored model detector over them;
3. reports how many injected spikes are detected as spots.

This is NOT part of the benchmark generator and adds nothing to it. If the
detector *does* pick spikes up, that is a real sim-to-real gap in the protein
channel and we would then need to model dark counts in training -- but that is a
decision for the PI, not something this check changes. It only measures and
reports. Modifies nothing vendored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spotpipe.simulator import noise

__all__ = ["DarkCountConfig", "make_darkcount_image", "run_darkcount_check"]

# Measured protein-PMT dark-count rate (fraction of pixels per frame carrying a
# spurious single-PE spike). Lipid PMT shows none, so ch1 gets no spikes.
_DARKCOUNT_RATE: float = 0.0057


@dataclass(frozen=True)
class DarkCountConfig:
    """Knobs for the dark-count robustness probe."""

    height: int = 256
    width: int = 256
    n_images: int = 6
    rate: float = _DARKCOUNT_RATE      # fraction of ch2 pixels carrying a spike
    background_photons: float = 2.0    # flat photon background (both channels), as in the benchmark
    match_radius_px: float = 1.5       # a detection within this of a spike pixel "detected the spike"


def _empty_field(signal_photons: np.ndarray, ch: noise.ChannelDetector,
                 detector: noise.DetectorParams, rng: np.random.Generator) -> np.ndarray:
    """One channel's observed ADU for a spot-free flat-background field (float)."""
    obs = noise.apply_detector_noise(
        signal_photons, ch, rng, n_frames=detector.n_frames,
        threshold=detector.poisson_gaussian_threshold, adc_max=detector.adc_max)
    return obs.astype(np.float64)


def make_darkcount_image(
    shape: tuple[int, int],
    detector: noise.DetectorParams,
    cfg: DarkCountConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one empty two-channel field with ch2 dark-count spikes injected.

    Returns ``(image[2,H,W] uint16, spike_xy[K,2])`` where ``spike_xy`` are the
    ``(x, y)`` integer pixel centres of the injected ch2 spikes. ch1 (lipid) gets
    NO spikes (the 500 V PMT shows none). Spikes are added as ``n_pe * gain2`` ADU
    (one gain step per photoelectron) on top of the empty ch2 field, then clipped
    to the ADC ceiling -- exactly how a single-pixel thermionic event would land.
    """
    height, width = shape
    bg = np.full(shape, float(cfg.background_photons), dtype=np.float64)

    ch1 = _empty_field(bg, detector.ch1, detector, rng)               # lipid: no spikes
    ch2 = _empty_field(bg, detector.ch2, detector, rng)              # protein: spikes below

    # Sparse single-pixel spikes: ~rate of pixels, >=1 photoelectron each (Poisson
    # multiplicity gives the observed higher-ADU tail). Amplitude = n_pe * gain2.
    spike_mask = rng.random(shape) < cfg.rate
    ys, xs = np.where(spike_mask)
    n_pe = np.maximum(rng.poisson(1.0, size=xs.size), 1)
    ch2[ys, xs] += n_pe.astype(np.float64) * detector.ch2.gain
    ch2 = np.clip(ch2, 0.0, detector.adc_max)

    image = np.stack([np.rint(ch1), np.rint(ch2)], axis=0).astype(np.uint16)
    spike_xy = np.stack([xs.astype(float), ys.astype(float)], axis=1) if xs.size else np.empty((0, 2))
    return image, spike_xy


def _count_spike_detections(det_xy: np.ndarray, spike_xy: np.ndarray, radius: float) -> tuple[int, int]:
    """(#detections matched to a spike, #spikes with >=1 detection) within radius."""
    if det_xy.size == 0 or spike_xy.size == 0:
        return 0, 0
    d2 = ((det_xy[:, None, 0] - spike_xy[None, :, 0]) ** 2
          + (det_xy[:, None, 1] - spike_xy[None, :, 1]) ** 2)
    within = d2 <= (radius * radius)
    n_det_matched = int(np.count_nonzero(within.any(axis=1)))
    n_spikes_hit = int(np.count_nonzero(within.any(axis=0)))
    return n_det_matched, n_spikes_hit


def run_darkcount_check(
    model,
    params,
    detector: noise.DetectorParams,
    cfg: DarkCountConfig,
    *,
    seed: int = 0,
    predict_fn=None,
    log_fn=print,
) -> dict:
    """Generate spike fields, run the detector, and report the spike-detection rate.

    ``model`` / ``params`` are a loaded checkpoint and its :class:`InferenceParams`
    (see ``spotpipe.benchmark.infer``). ``predict_fn`` defaults to the vendored
    ``predict_spots``. Returns a stats dict; prints a one-line-per-image summary
    plus a total. This VERIFIES whether a PSF-matched detector rejects single-pixel
    dark counts -- it does not change the benchmark either way.
    """
    if predict_fn is None:
        from spotpipe.models import predict_spots as predict_fn  # noqa: N806

    shape = (cfg.height, cfg.width)
    root = np.random.SeedSequence(int(seed))
    per_image = []
    tot_spikes = tot_dets = tot_matched = tot_hits = 0

    for i, child in enumerate(root.spawn(cfg.n_images)):
        rng = np.random.default_rng(child)
        image, spike_xy = make_darkcount_image(shape, detector, cfg, rng)
        df = predict_fn(
            model, image, image_id=f"darkcount_{i:04d}",
            adc_max=params.adc_max, peak_threshold=params.peak_threshold,
            nms_kernel=params.nms_kernel, max_spots=params.max_spots)
        det_xy = df[["x", "y"]].to_numpy(float) if len(df) else np.empty((0, 2))
        n_matched, n_hits = _count_spike_detections(det_xy, spike_xy, cfg.match_radius_px)

        n_spikes, n_dets = len(spike_xy), len(df)
        tot_spikes += n_spikes
        tot_dets += n_dets
        tot_matched += n_matched
        tot_hits += n_hits
        per_image.append({
            "image_id": f"darkcount_{i:04d}", "n_spikes": n_spikes, "n_detections": n_dets,
            "n_detections_matched_to_spike": n_matched, "n_spikes_detected": n_hits,
        })
        log_fn(f"  [darkcount] img {i}: {n_spikes:>5} spikes, {n_dets:>4} detections, "
               f"{n_hits:>4} spikes detected as spots")

    spike_detect_rate = (tot_hits / tot_spikes) if tot_spikes else 0.0
    stats = {
        "n_images": cfg.n_images,
        "shape": [cfg.height, cfg.width],
        "rate": cfg.rate,
        "match_radius_px": cfg.match_radius_px,
        "n_spikes_total": tot_spikes,
        "n_detections_total": tot_dets,
        "n_detections_matched_to_spike": tot_matched,
        "n_spikes_detected_as_spots": tot_hits,
        "spike_detection_rate": spike_detect_rate,
        "per_image": per_image,
        "interpretation": (
            "A PSF-matched detector should reject single-pixel dark counts (a real "
            "spot is a multi-pixel PSF blob). spike_detection_rate near 0 confirms "
            "the protein-channel dark counts are NOT a sim-to-real gap for detection; "
            "a non-trivial rate would mean training should model them."),
    }
    log_fn(f"[darkcount] TOTAL: {tot_hits}/{tot_spikes} spikes detected as spots "
           f"(rate {spike_detect_rate:.4%}); {tot_dets} detections over {cfg.n_images} images.")
    return stats
