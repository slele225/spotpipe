"""Detector noise model for the FV3000 analog-integration PMT chain (STUB).

Implemented in build stage 2. Models per-channel excess-noise factor, noise
floor, and the saturation knee, in photon-proportional units. Frame averaging
is ``integration count = 3`` (mean of 3 scans): simulate by reducing the added
noise variance accordingly, NOT by scaling the signal.
"""

from __future__ import annotations


def apply_detector_noise(*args, **kwargs):
    """Apply PMT noise / saturation to a photon-proportional signal. STUB."""
    raise NotImplementedError("noise.apply_detector_noise is implemented in build stage 2.")
