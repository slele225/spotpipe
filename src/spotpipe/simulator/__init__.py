"""Forward model of the Olympus FV3000 analog-integration PMT imaging chain.

Build stage 2 (implemented): the synthetic-data forward model in isolation --
area-normalised PSF (:mod:`psf`), parametric backgrounds (:mod:`backgrounds`),
the per-channel detector noise / saturation chain (:mod:`noise`), the per-image
scene sampler + renderer (:mod:`forward_model`), and the dataset driver
(:mod:`generate_dataset`). Models, losses, training, and benchmark remain
stubbed until their own build stages.
"""

from spotpipe.simulator import backgrounds, forward_model, noise, psf  # noqa: F401
from spotpipe.simulator.forward_model import (  # noqa: F401
    SceneParams,
    SimulatedImage,
    sample_scene_params,
    simulate_image,
)
from spotpipe.simulator.noise import (  # noqa: F401
    ChannelDetector,
    DetectorParams,
    apply_detector_noise,
    sample_detector_params,
)

__all__ = [
    "backgrounds",
    "forward_model",
    "noise",
    "psf",
    "SceneParams",
    "SimulatedImage",
    "sample_scene_params",
    "simulate_image",
    "ChannelDetector",
    "DetectorParams",
    "apply_detector_noise",
    "sample_detector_params",
]
